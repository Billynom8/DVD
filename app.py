import streamlit as st
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json

CONFIG_FILE = Path("app_config.json")


def load_settings():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_settings(settings):
    with open(CONFIG_FILE, "w") as f:
        json.dump(settings, f)


default_settings = load_settings()

st.set_page_config(page_title="Depth Estimation", layout="wide")


def read_video_early_downsample(video_path, target_w):
    """Read video and downsample during loading to save RAM. Width fixed, height scales."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ratio = target_w / orig_w
    new_w = target_w
    new_h = int(np.ceil(orig_h * ratio))
    new_h = (new_h + 15) // 16 * 16
    new_w = (new_w + 15) // 16 * 16

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if new_h != orig_h or new_w != orig_w:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

    return video_tensor.unsqueeze(0), fps, (orig_h, orig_w)


def get_video_info(video_path):
    """Get video metadata using cv2."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {"fps": fps, "width": width, "height": height, "total_frames": total_frames}


def calculate_segments(T, window_size, overlap):
    """Calculate number of segments and final chunk size."""
    if T <= window_size:
        return 1, T

    segments = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = min(start + window_size, T)
        segments.append((start, end))
        start += window_size - overlap

    final_chunk = T - segments[-1][0]
    return len(segments), final_chunk


if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
elif st.session_state.model is not None:
    st.session_state.model_loaded = True
if "current_video" not in st.session_state:
    st.session_state.current_video = None
if "depth_result" not in st.session_state:
    st.session_state.depth_result = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "is_inferring" not in st.session_state:
    st.session_state.is_inferring = False


st.title("Depth Estimation Pipeline")

if st.session_state.model_loaded and st.session_state.model is not None:
    st.success(f"Model loaded and ready (VRAM in use)")

is_inferring = st.session_state.is_inferring

st.sidebar.header("Configuration")

ckpt_dir = st.sidebar.text_input("Checkpoint Directory", value="./ckpt", disabled=is_inferring)
model_config = st.sidebar.text_input("Model Config", value="ckpt/model_config.yaml", disabled=is_inferring)

st.sidebar.subheader("Processing Options")
early_downsample = st.sidebar.checkbox(
    "Early Downsample (saves RAM)",
    value=default_settings.get("early_downsample", True),
    disabled=is_inferring,
    help="Downsample video during loading instead of after. Reduces RAM usage.",
)
skip_upscale = st.sidebar.checkbox(
    "Skip Upscale (keep low-res output)",
    value=default_settings.get("skip_upscale", True),
    disabled=is_inferring,
    help="Skip upscaling at the end. Output will be at inference resolution.",
)
window_size = st.sidebar.number_input(
    "Window Size", value=default_settings.get("window_size", 81), min_value=9, step=9, disabled=is_inferring
)
overlap = st.sidebar.number_input(
    "Overlap", value=default_settings.get("overlap", 9), min_value=0, disabled=is_inferring
)
width = st.sidebar.number_input("Width", value=default_settings.get("width", 640), step=16, disabled=is_inferring)
grayscale = st.sidebar.checkbox(
    "Grayscale Output", value=default_settings.get("grayscale", False), disabled=is_inferring
)
color_output = st.sidebar.checkbox(
    "Color Output", value=default_settings.get("color_output", True), disabled=is_inferring
)

if not grayscale and not color_output:
    st.sidebar.warning("Warning: Both outputs are disabled! Enable at least one.")

output_dir = st.sidebar.text_input(
    "Output Directory", value=default_settings.get("output_dir", "./inference_results"), disabled=is_inferring
)

if st.sidebar.button("Save Settings", disabled=is_inferring):
    settings = {
        "early_downsample": early_downsample,
        "skip_upscale": skip_upscale,
        "window_size": window_size,
        "overlap": overlap,
        "width": width,
        "grayscale": grayscale,
        "color_output": color_output,
        "output_dir": output_dir,
    }
    save_settings(settings)
    st.sidebar.success("Settings saved!")


class Args:
    pass


args = Args()
args.ckpt = ckpt_dir
args.model_config = model_config
args.window_size = window_size
args.overlap = overlap
args.height = 480
args.width = width
args.grayscale = grayscale
args.output_dir = output_dir
args.skip_upscale = skip_upscale


st.header("Step 1: Load Model")
if st.session_state.model_loaded and st.session_state.model is not None:
    st.success("Model is already loaded!")
    if st.button("Reload Model"):
        if not os.path.exists(ckpt_dir):
            st.error(f"Checkpoint directory not found: {ckpt_dir}")
        elif not os.path.exists(model_config):
            st.error(f"Model config not found: {model_config}")
        else:
            with st.spinner("Loading model... this may take a few minutes"):
                try:
                    from test_script.test_batch_video import load_model
                    from omegaconf import OmegaConf

                    yaml_args = OmegaConf.load(model_config)
                    model = load_model(ckpt_dir, yaml_args)

                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
else:
    if st.button("Load Model", type="primary", disabled=is_inferring):
        if not os.path.exists(ckpt_dir):
            st.error(f"Checkpoint directory not found: {ckpt_dir}")
        elif not os.path.exists(model_config):
            st.error(f"Model config not found: {model_config}")
        else:
            with st.spinner("Loading model... this may take a few minutes"):
                try:
                    from test_script.test_batch_video import load_model
                    from omegaconf import OmegaConf

                    yaml_args = OmegaConf.load(model_config)
                    model = load_model(ckpt_dir, yaml_args)

                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

if st.session_state.model_loaded and st.session_state.model is not None:
    st.success("Model is already loaded! VRAM is in use.")

if st.session_state.model_loaded:
    st.info("Model is loaded and ready")
else:
    st.warning("Please load the model first")


st.header("Utility")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear CUDA Cache", disabled=st.session_state.is_inferring):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            st.success("CUDA cache cleared!")
        else:
            st.warning("No CUDA device available")

with col2:
    if st.button("Unload Model", disabled=st.session_state.is_inferring):
        if st.session_state.model is not None:
            del st.session_state.model
            st.session_state.model = None
            st.session_state.model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.success("Model unloaded!")
        else:
            st.warning("No model loaded")

with col3:
    if st.button("Stop Inference", type="primary", disabled=not st.session_state.is_inferring):
        st.session_state.stop_requested = True
        st.warning("Stop requested - current inference will complete")


st.header("Step 2: Select Video")
if is_inferring:
    st.warning("Inference in progress... Please wait.")
    input_videos = None
else:
    input_videos = st.file_uploader(
        "Upload video(s)", type=["mp4", "avi", "mov", "mkv", "webm"], accept_multiple_files=True
    )

batch_mode = len(input_videos) > 1 if input_videos else False

if input_videos:
    if batch_mode:
        st.session_state.batch_results = None
        st.success(f"{len(input_videos)} videos loaded for batch processing")

        total_frames_estimate = 0
        for vid in input_videos[:3]:
            temp_path = Path("temp_uploads") / vid.name
            if not temp_path.exists():
                with open(temp_path, "wb") as f:
                    f.write(vid.getbuffer())
            info = get_video_info(str(temp_path))
            if info:
                total_frames_estimate += info["total_frames"]

        st.info(f"Total videos: {len(input_videos)}")
    else:
        input_video = input_videos[0]
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        input_path = temp_dir / input_video.name

        if st.session_state.current_video != input_video.name:
            with open(input_path, "wb") as f:
                f.write(input_video.getbuffer())
            st.session_state.current_video = input_video.name
            st.session_state.depth_result = None
            st.session_state.batch_results = None
        input_video = input_videos[0]
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        input_path = temp_dir / input_video.name

        if st.session_state.current_video != input_video.name:
            with open(input_path, "wb") as f:
                f.write(input_video.getbuffer())
            st.session_state.current_video = input_video.name
            st.session_state.depth_result = None

        st.success(f"Video loaded: {input_video.name}")
        video_bytes = input_video.getvalue()
        st.video(video_bytes, format="video/mp4")

        video_info = get_video_info(str(input_path))
        if video_info:
            num_segments, final_chunk = calculate_segments(video_info["total_frames"], window_size, overlap)

            st.info(
                f"**Video Info:** {video_info['width']}x{video_info['height']} | "
                f"{video_info['total_frames']} frames | {video_info['fps']:.2f} fps | "
                f"**Segments:** {num_segments} (final chunk: {final_chunk} frames)"
            )

        args.input_video = str(input_path)


st.header("Step 3: Run Depth Estimation")

run_depth = st.checkbox("Run Depth Estimation", value=False)

if run_depth:
    if not st.session_state.model_loaded:
        st.error("Please load the model first")
    elif not input_videos:
        st.error("Please upload video(s) first")
    else:
        if batch_mode:
            if st.button(f"Generate Depth ({len(input_videos)} videos)", type="primary", disabled=is_inferring):
                st.session_state.is_inferring = True
                st.session_state.stop_requested = False

                results = []
                progress_bar = st.progress(0)

                for i, vid in enumerate(input_videos):
                    if st.session_state.stop_requested:
                        st.warning("Batch processing stopped by user")
                        break

                    st.info(f"Processing {i + 1}/{len(input_videos)}: {vid.name}")

                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    input_path = temp_dir / vid.name

                    with open(input_path, "wb") as f:
                        f.write(vid.getbuffer())

                    args.input_video = str(input_path)

                    try:
                        if early_downsample:
                            input_tensor, origin_fps, orig_size = read_video_early_downsample(
                                args.input_video, args.width
                            )
                        else:
                            from test_script.test_batch_video import load_video_data

                            input_tensor, orig_size, origin_fps = load_video_data(args)

                        if skip_upscale:
                            from test_script.test_batch_video import generate_depth_sliced

                            depth = generate_depth_sliced(
                                st.session_state.model, input_tensor, args.window_size, args.overlap
                            )[0]
                        else:
                            from test_script.test_batch_video import predict_depth

                            depth = predict_depth(st.session_state.model, input_tensor, orig_size, args)

                        args.output_dir = default_settings.get("output_dir", "./inference_results")
                        from test_script.test_batch_video import save_results

                        saved_paths = []

                        if color_output:
                            args.grayscale = False
                            output_path = save_results(depth, origin_fps, args)
                            saved_paths.append(output_path)
                            st.success(f"Saved color: {vid.name}")

                        if grayscale:
                            args.grayscale = True
                            output_path = save_results(depth, origin_fps, args)
                            saved_paths.append(output_path)
                            st.success(f"Saved grayscale: {vid.name}")

                        results.append({"name": vid.name, "output": saved_paths, "success": True})

                    except Exception as e:
                        results.append({"name": vid.name, "error": str(e), "success": False})
                        st.error(f"Failed: {vid.name} - {str(e)}")

                    progress_bar.progress((i + 1) / len(input_videos))

                st.session_state.batch_results = results
                st.session_state.is_inferring = False

                success_count = sum(1 for r in results if r.get("success"))
                st.success(f"Batch complete: {success_count}/{len(results)} videos processed")

        else:
            if st.button("Generate Depth", type="primary", disabled=is_inferring):
                st.session_state.is_inferring = True
                st.session_state.stop_requested = False
                with st.spinner("Processing video... this may take a while"):
                    try:
                        if early_downsample:
                            input_tensor, origin_fps, orig_size = read_video_early_downsample(
                                args.input_video, args.width
                            )
                            print(f"Early downsampled: {input_tensor.shape}, orig_size: {orig_size}")
                        else:
                            from test_script.test_batch_video import load_video_data

                            input_tensor, orig_size, origin_fps = load_video_data(args)

                        if skip_upscale:
                            from test_script.test_batch_video import generate_depth_sliced

                            depth = generate_depth_sliced(
                                st.session_state.model, input_tensor, args.window_size, args.overlap
                            )[0]
                            print(f"Depth (no upscale): {depth.shape}, range {depth.min()} - {depth.max()}")
                        else:
                            from test_script.test_batch_video import predict_depth

                            depth = predict_depth(st.session_state.model, input_tensor, orig_size, args)

                        if st.session_state.stop_requested:
                            st.warning("Inference stopped by user")
                            st.session_state.stop_requested = False
                        else:
                            st.session_state.depth_result = depth
                            st.session_state.origin_fps = origin_fps
                            st.success("Depth estimation completed!")
                    except Exception as e:
                        st.error(f"Error during depth estimation: {str(e)}")
                    finally:
                        st.session_state.is_inferring = False

if st.session_state.depth_result is not None:
    st.info("Depth estimation result ready")


st.header("Step 4: Save Results")

has_single = st.session_state.depth_result is not None
has_batch = st.session_state.batch_results is not None and len(st.session_state.batch_results) > 0

if has_single or has_batch:
    if has_batch:
        st.info(f"Batch results ready: {len(st.session_state.batch_results)} videos")

        success_files = [r["output"] for r in st.session_state.batch_results if r.get("success")]
        if success_files:
            st.success(
                f"All {len(success_files)} files saved to: {default_settings.get('output_dir', './inference_results')}"
            )

    if has_single:
        st.info("Single video result ready")

        if st.button("Save Result", disabled=is_inferring):
            try:
                from test_script.test_batch_video import save_results

                args.output_dir = default_settings.get("output_dir", "./inference_results")
                saved_paths = []

                if color_output:
                    args.grayscale = False
                    output_path = save_results(st.session_state.depth_result, st.session_state.origin_fps, args)
                    saved_paths.append(output_path)
                    st.success(f"Color saved: {output_path}")
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Color", data=f, file_name=os.path.basename(output_path), mime="video/mp4"
                        )

                if grayscale:
                    args.grayscale = True
                    output_path = save_results(st.session_state.depth_result, st.session_state.origin_fps, args)
                    saved_paths.append(output_path)
                    st.success(f"Grayscale saved: {output_path}")
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Grayscale",
                            data=f,
                            file_name=os.path.basename(output_path),
                            mime="video/mp4",
                        )

            except Exception as e:
                st.error(f"Error saving results: {str(e)}")
else:
    st.info("No depth result to save yet")
