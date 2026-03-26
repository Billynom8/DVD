import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.io import read_video, save_video_ffmpeg, save_depth_png_sequence
from core.utils import (
    compute_scale_and_shift,
    resize_for_training_scale,
    resize_depth_back,
    pad_time_mod4,
    get_window_index,
)
from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule


# =============================
# Core Inference
# =============================
def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, scale_only=False):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")

    depth_res_list = []

    # 1. Inference per window
    for start, end in tqdm(depth_windows, desc="Inferencing Slices"):
        _input_rgb_slice = input_rgb[:, start:end]

        # Ensure 4n+1 padding
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            input_image=_input_rgb_slice[:, 0],
            extra_images=_input_rgb_slice,
            extra_image_frame_index=torch.ones([B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        # Drop the padded frames and ensure it's a tensor
        depth_slice = outputs["depth"][:, :origin_T]
        if isinstance(depth_slice, np.ndarray):
            depth_slice = torch.from_numpy(depth_slice)
        depth_res_list.append(depth_slice)

    # 2. Overlap Alignment
    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = torch.sum(curr_frames * ref_frames) / (torch.sum(curr_frames * curr_frames) + 1e-6)
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames.numpy(), ref_frames.numpy())

            scale = max(min(scale, 1.5), 0.7)
            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            # Smooth blending
            alpha = torch.linspace(0, 1, real_overlap).view(1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_t[:, :real_overlap]

            depth_list_aligned = torch.cat(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap, aligned_t[:, real_overlap:]], dim=1
            )
        else:
            depth_list_aligned = torch.cat([depth_list_aligned, t], dim=1)
        prev_end = end

    return depth_list_aligned[:, :T]


def load_model(ckpt_dir, yaml_args):
    """Initializes and loads the model checkpoint."""
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
    )

    ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
    state_dict = load_file(ckpt_path, device="cpu")
    dit_state_dict = {k.replace("pipe.dit.", ""): v for k, v in state_dict.items() if "pipe.dit." in k}
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to("cuda")
    return model


def load_video_data(args):
    """Loads and resizes the input video."""
    input_tensor, origin_fps = read_video(args.input_video)
    input_tensor, orig_size = resize_for_training_scale(input_tensor, args.height, args.width)
    return input_tensor, orig_size, origin_fps


def predict_depth(model, input_tensor, orig_size, args):
    """Runs depth prediction and post-processes the output to original size."""
    depth = generate_depth_sliced(model, input_tensor, args.window_size, args.overlap)[0]
    depth_np = depth.cpu().numpy()
    if not getattr(args, "skip_upscale", False):
        depth_np = resize_depth_back(depth_np, orig_size)
    return depth_np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--model_config", default="ckpt/model_config.yaml")
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument("--grayscale", action="store_true", help="Output grayscale depth video")
    parser.add_argument("--color", action="store_true", help="Output colorized depth video")
    parser.add_argument("--use_10bit", action="store_true", help="Elevates video output to 10-bit HEVC")
    parser.add_argument("--save_16bit_png", action="store_true", help="Save as 16-bit PNG sequence")
    parser.add_argument("--skip_upscale", action="store_true", help="Skip upscaling the output")
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_args = OmegaConf.load(args.model_config)

    model = load_model(args.ckpt, yaml_args)
    input_tensor, orig_size, origin_fps = load_video_data(args)
    depth = predict_depth(model, input_tensor, orig_size, args)

    bit_depth = 10 if args.use_10bit else 8

    if args.grayscale:
        save_video_ffmpeg(depth, origin_fps, args.input_video, args.output_dir, grayscale=True, bit_depth=bit_depth)

    if args.color:
        save_video_ffmpeg(depth, origin_fps, args.input_video, args.output_dir, grayscale=False, bit_depth=bit_depth)

    if args.save_16bit_png:
        save_depth_png_sequence(depth, args.input_video, args.output_dir)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
