import streamlit as st
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

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


if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "current_video" not in st.session_state:
    st.session_state.current_video = None
if "depth_result" not in st.session_state:
    st.session_state.depth_result = None
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "is_inferring" not in st.session_state:
    st.session_state.is_inferring = False


st.title("Depth Estimation Pipeline")

st.sidebar.header("Configuration")

ckpt_dir = st.sidebar.text_input("Checkpoint Directory", value="./ckpt")
model_config = st.sidebar.text_input("Model Config", value="ckpt/model_config.yaml")

st.sidebar.subheader("Processing Options")
early_downsample = st.sidebar.checkbox(
    "Early Downsample (saves RAM)",
    value=True,
    help="Downsample video during loading instead of after. Reduces RAM usage.",
)
skip_upscale = st.sidebar.checkbox(
    "Skip Upscale (keep low-res output)",
    value=True,
    help="Skip upscaling at the end. Output will be at inference resolution.",
)
window_size = st.sidebar.number_input("Window Size", value=81, min_value=9, step=9)
overlap = st.sidebar.number_input("Overlap", value=9, min_value=0)
width = st.sidebar.number_input("Width", value=640, step=16)
grayscale = st.sidebar.checkbox("Grayscale Output", value=False)
output_dir = st.sidebar.text_input("Output Directory", value="./inference_results")


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
if st.button("Load Model", type="primary"):
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
input_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"])

if input_video:
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

    args.input_video = str(input_path)


st.header("Step 3: Run Depth Estimation")

run_depth = st.checkbox("Run Depth Estimation", value=False)

if run_depth:
    if not st.session_state.model_loaded:
        st.error("Please load the model first")
    elif not input_video:
        st.error("Please upload a video first")
    else:
        if st.button("Generate Depth", type="primary"):
            st.session_state.is_inferring = True
            st.session_state.stop_requested = False
            with st.spinner("Processing video... this may take a while"):
                try:
                    if early_downsample:
                        input_tensor, origin_fps, orig_size = read_video_early_downsample(args.input_video, args.width)
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

if st.session_state.depth_result is not None:
    if st.button("Save Results"):
        try:
            from test_script.test_batch_video import save_results

            output_path = save_results(st.session_state.depth_result, st.session_state.origin_fps, args)

            st.success(f"Results saved to: {output_path}")

            # Show download button
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Result", data=f, file_name=os.path.basename(output_path), mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Error saving results: {str(e)}")
else:
    st.info("No depth result to save yet")
