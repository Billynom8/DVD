import streamlit as st
import os
import sys
import torch
from pathlib import Path

# Add core to path if needed
sys.path.append(str(Path(__file__).parent))

from core.config import load_settings, save_settings
from core.io import get_video_info, read_video_early_downsample, save_video_ffmpeg, save_depth_png_sequence
from core.utils import calculate_segments

default_settings = load_settings()

st.set_page_config(page_title="Depth Estimation", layout="wide")

# Session State Initialization
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
if "input_paths" not in st.session_state:
    st.session_state.input_paths = []
if "single_results" not in st.session_state:
    st.session_state.single_results = []


st.title("Depth Estimation Pipeline")

if st.session_state.model_loaded and st.session_state.model is not None:
    st.success("Model loaded and ready (VRAM in use)")

is_inferring = st.session_state.is_inferring


def start_inference():
    st.session_state.is_inferring = True
    st.session_state.stop_requested = False


st.sidebar.header("Configuration")

ckpt_dir = st.sidebar.text_input("Checkpoint Directory", value="./ckpt", disabled=st.session_state.is_inferring)
model_config_path = st.sidebar.text_input(
    "Model Config", value="ckpt/model_config.yaml", disabled=st.session_state.is_inferring
)

st.sidebar.subheader("Processing Options")
early_downsample = st.sidebar.checkbox(
    "Early Downsample (saves RAM)",
    value=default_settings.get("early_downsample", True),
    disabled=st.session_state.is_inferring,
    help="Downsample video during loading instead of after. Reduces RAM usage.",
)
skip_upscale = st.sidebar.checkbox(
    "Skip Upscale (keep low-res output)",
    value=default_settings.get("skip_upscale", True),
    disabled=st.session_state.is_inferring,
    help="Skip upscaling at the end. Output will be at inference resolution.",
)
window_size = st.sidebar.number_input(
    "Window Size",
    value=default_settings.get("window_size", 81),
    min_value=9,
    step=9,
    disabled=st.session_state.is_inferring,
)
overlap = st.sidebar.number_input(
    "Overlap", value=default_settings.get("overlap", 9), min_value=0, disabled=st.session_state.is_inferring
)
width = st.sidebar.number_input(
    "Width", value=default_settings.get("width", 640), step=16, disabled=st.session_state.is_inferring
)

st.sidebar.subheader("Output Options")
grayscale = st.sidebar.checkbox(
    "Grayscale Output", value=default_settings.get("grayscale", False), disabled=st.session_state.is_inferring
)
color_output = st.sidebar.checkbox(
    "Color Output", value=default_settings.get("color_output", True), disabled=st.session_state.is_inferring
)
use_10bit = st.sidebar.checkbox(
    "Use 10-bit HEVC",
    value=default_settings.get("use_10bit", False),
    disabled=st.session_state.is_inferring,
    help="Elevates Grayscale and Color video bit rate to 10-bit HEVC.",
)
save_16bit_png = st.sidebar.checkbox(
    "Save 16-bit PNG seq", value=default_settings.get("save_16bit_png", False), disabled=st.session_state.is_inferring
)

if not grayscale and not color_output and not save_16bit_png:
    st.sidebar.warning("Warning: No outputs selected.")

st.sidebar.subheader("File & Folder Settings")

# 1. Source Folder
source_folder = st.sidebar.text_input(
    "Local Source Folder",
    value=default_settings.get("source_folder", ""),
    help="Path to a folder containing videos to process.",
    disabled=st.session_state.is_inferring,
)

# 2. Output Directory
output_dir = st.sidebar.text_input(
    "Output Directory",
    value=default_settings.get("output_dir", "./inference_results"),
    disabled=st.session_state.is_inferring,
)

resume_mode = st.sidebar.checkbox(
    "Resume",
    value=default_settings.get("resume_mode", False),
    help="Moves finished videos to a 'finished' subfolder.",
    disabled=st.session_state.is_inferring,
)

if st.sidebar.button("Scan Folder", disabled=st.session_state.is_inferring, use_container_width=True):
    if os.path.exists(source_folder):
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        folder_paths = [
            str(Path(source_folder) / f)
            for f in os.listdir(source_folder)
            if any(f.lower().endswith(ext) for ext in extensions)
        ]
        if folder_paths:
            st.session_state.input_paths = folder_paths
            st.session_state.depth_result = None
            st.session_state.batch_results = None
            st.session_state.single_results = []
            st.sidebar.success(f"Found {len(folder_paths)} videos.")
        else:
            st.sidebar.warning("No videos found.")
    else:
        st.sidebar.error("Source folder does not exist.")

if st.sidebar.button("Save Settings", disabled=st.session_state.is_inferring):
    settings = {
        "early_downsample": early_downsample,
        "skip_upscale": skip_upscale,
        "window_size": window_size,
        "overlap": overlap,
        "width": width,
        "grayscale": grayscale,
        "color_output": color_output,
        "use_10bit": use_10bit,
        "save_16bit_png": save_16bit_png,
        "output_dir": output_dir,
        "source_folder": source_folder,
        "resume_mode": resume_mode,
    }
    save_settings(settings)
    st.sidebar.success("Settings saved!")


class Args:
    pass


args = Args()
args.ckpt = ckpt_dir
args.model_config = model_config_path
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
    if st.button("Reload Model", disabled=st.session_state.is_inferring):
        with st.spinner("Loading model... this may take a few minutes"):
            try:
                from test_script.test_batch_video import load_model
                from omegaconf import OmegaConf

                yaml_args = OmegaConf.load(model_config_path)
                model = load_model(ckpt_dir, yaml_args)

                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
else:
    if st.button("Load Model", type="primary", disabled=st.session_state.is_inferring):
        if not os.path.exists(ckpt_dir):
            st.error(f"Checkpoint directory not found: {ckpt_dir}")
        elif not os.path.exists(model_config_path):
            st.error(f"Model config not found: {model_config_path}")
        else:
            with st.spinner("Loading model... this may take a few minutes"):
                try:
                    from test_script.test_batch_video import load_model
                    from omegaconf import OmegaConf

                    yaml_args = OmegaConf.load(model_config_path)
                    model = load_model(ckpt_dir, yaml_args)

                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")


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


st.header("Step 2: Processing Queue")
if st.session_state.is_inferring:
    st.info("Inference in progress... Queue locked.")
    if st.session_state.input_paths:
        st.write(f"Currently processing {len(st.session_state.input_paths)} video(s):")
        for p in st.session_state.input_paths:
            st.text(f"- {os.path.basename(p)}")
else:
    # 1. Option A: Manual Upload
    input_videos = st.file_uploader(
        "Upload video(s) manually", type=["mp4", "avi", "mov", "mkv", "webm"], accept_multiple_files=True
    )
    if input_videos:
        current_names = [os.path.basename(p) for p in st.session_state.input_paths]
        new_names = [vid.name for vid in input_videos]

        if set(current_names) != set(new_names):
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            new_paths = []
            for vid in input_videos:
                input_path = temp_dir / vid.name
                with open(input_path, "wb") as f:
                    f.write(vid.getbuffer())
                new_paths.append(str(input_path))

            st.session_state.input_paths = new_paths
            st.session_state.depth_result = None
            st.session_state.batch_results = None
            st.session_state.single_results = []
            st.success(f"Queue updated with {len(new_paths)} uploaded video(s).")

    # 2. Display Queue Status
    if st.session_state.input_paths:
        num_vids = len(st.session_state.input_paths)
        if num_vids > 1:
            st.info(f"📋 **Batch Queue:** {num_vids} videos ready for processing.")
            with st.expander("Show file list"):
                for p in st.session_state.input_paths:
                    st.text(f"- {os.path.basename(p)}")
        else:
            st.success(f"🎯 **Single Video:** {os.path.basename(st.session_state.input_paths[0])}")
            st.video(st.session_state.input_paths[0])

            # Show video info only for single mode
            video_info = get_video_info(st.session_state.input_paths[0])
            if video_info:
                num_segments, final_chunk = calculate_segments(video_info["total_frames"], window_size, overlap)
                st.info(
                    f"**Video Info:** {video_info['width']}x{video_info['height']} | "
                    f"{video_info['total_frames']} frames | {video_info['fps']:.2f} fps | "
                    f"**Segments:** {num_segments}"
                )
    else:
        st.warning("Queue is empty. Upload videos or use 'Scan Folder' in the sidebar.")

input_paths = st.session_state.input_paths
batch_mode = len(input_paths) > 1


st.header("Step 3: Run Depth Estimation")

run_depth = st.checkbox("Run Depth Estimation", value=False, disabled=st.session_state.is_inferring)

if run_depth:
    if not st.session_state.model_loaded:
        st.error("Please load the model first")
    elif not input_paths:
        st.error("Please upload video(s) first")
    else:
        if batch_mode:
            if st.button(
                f"Generate Depth ({len(input_paths)} videos)",
                type="primary",
                disabled=st.session_state.is_inferring,
                on_click=start_inference,
            ):
                try:
                    results = []
                    progress_bar = st.progress(0)

                    for i, input_path in enumerate(input_paths):
                        if st.session_state.stop_requested:
                            st.warning("Batch processing stopped by user")
                            break

                        vid_name = os.path.basename(input_path)
                        st.info(f"Processing {i + 1}/{len(input_paths)}: {vid_name}")

                        try:
                            if early_downsample:
                                input_tensor, origin_fps, orig_size = read_video_early_downsample(input_path, width)
                            else:
                                from test_script.test_batch_video import read_video, resize_for_training_scale

                                input_tensor, origin_fps = read_video(input_path)
                                input_tensor, orig_size = resize_for_training_scale(input_tensor, 480, width)

                            from test_script.test_batch_video import predict_depth

                            depth = predict_depth(st.session_state.model, input_tensor, orig_size, args)

                            current_output_dir = output_dir
                            saved_paths = []
                            bit_depth = 10 if use_10bit else 8

                            if color_output:
                                out = save_video_ffmpeg(
                                    depth,
                                    origin_fps,
                                    input_path,
                                    current_output_dir,
                                    grayscale=False,
                                    bit_depth=bit_depth,
                                )
                                saved_paths.append(out)
                            if grayscale:
                                out = save_video_ffmpeg(
                                    depth,
                                    origin_fps,
                                    input_path,
                                    current_output_dir,
                                    grayscale=True,
                                    bit_depth=bit_depth,
                                )
                                saved_paths.append(out)
                            if save_16bit_png:
                                out = save_depth_png_sequence(depth, input_path, current_output_dir)
                                saved_paths.append(out)

                            results.append({"name": vid_name, "output": saved_paths, "success": True})

                            # Resume mode: Move original file to 'finished' folder
                            if resume_mode:
                                finished_dir = Path(os.path.dirname(input_path)) / "finished"
                                finished_dir.mkdir(exist_ok=True)
                                import shutil

                                shutil.move(input_path, finished_dir / vid_name)
                                st.success(f"Moved {vid_name} to finished folder.")

                        except Exception as e:
                            results.append({"name": vid_name, "error": str(e), "success": False})
                            st.error(f"Failed: {vid_name} - {str(e)}")

                        progress_bar.progress((i + 1) / len(input_paths))

                    st.session_state.batch_results = results
                    success_count = sum(1 for r in results if r.get("success"))
                    st.success(f"Batch complete: {success_count}/{len(results)} videos processed")
                finally:
                    st.session_state.is_inferring = False
                    st.rerun()

        else:
            if st.button(
                "Generate Depth", type="primary", disabled=st.session_state.is_inferring, on_click=start_inference
            ):
                try:
                    input_path = input_paths[0]
                    with st.spinner("Processing video... this may take a while"):
                        if early_downsample:
                            input_tensor, origin_fps, orig_size = read_video_early_downsample(input_path, width)
                        else:
                            from test_script.test_batch_video import read_video, resize_for_training_scale

                            input_tensor, origin_fps = read_video(input_path)
                            input_tensor, orig_size = resize_for_training_scale(input_tensor, 480, width)

                        from test_script.test_batch_video import predict_depth

                        depth = predict_depth(st.session_state.model, input_tensor, orig_size, args)

                        if not st.session_state.stop_requested:
                            st.session_state.depth_result = depth
                            st.session_state.origin_fps = origin_fps
                            st.success("Depth estimation completed!")
                except Exception as e:
                    st.error(f"Error during depth estimation: {str(e)}")
                finally:
                    st.session_state.is_inferring = False
                    st.rerun()


st.header("Step 4: Save Results")

has_single = st.session_state.depth_result is not None
has_batch = st.session_state.batch_results is not None and len(st.session_state.batch_results) > 0

if has_single or has_batch:
    if has_batch:
        st.info(f"Batch results ready: {len(st.session_state.batch_results)} videos")
        success_files = [r["output"] for r in st.session_state.batch_results if r.get("success")]
        if success_files:
            st.success(f"Files saved to: {output_dir}")

    if has_single:
        st.info("Single video result ready")
        if st.button("Save Result", disabled=st.session_state.is_inferring):
            results_to_show = []
            try:
                input_path = input_paths[0]
                bit_depth = 10 if use_10bit else 8

                if color_output:
                    st.info("Generating color video...")
                    out = save_video_ffmpeg(
                        st.session_state.depth_result,
                        st.session_state.origin_fps,
                        input_path,
                        output_dir,
                        grayscale=False,
                        bit_depth=bit_depth,
                    )
                    results_to_show.append({"label": "Download Color", "path": out, "type": "video/mp4"})

                if grayscale:
                    st.info("Generating grayscale video...")
                    out = save_video_ffmpeg(
                        st.session_state.depth_result,
                        st.session_state.origin_fps,
                        input_path,
                        output_dir,
                        grayscale=True,
                        bit_depth=bit_depth,
                    )
                    results_to_show.append({"label": "Download Grayscale", "path": out, "type": "video/mp4"})

                if save_16bit_png:
                    st.info("Generating 16-bit PNG sequence...")
                    out = save_depth_png_sequence(st.session_state.depth_result, input_path, output_dir)
                    st.success(f"16-bit PNG sequence saved in: {out}")

                st.session_state.single_results = results_to_show
                st.success("All requested files generated!")
            except Exception as e:
                st.error(f"Error saving results: {str(e)}")

        if st.session_state.single_results:
            for res in st.session_state.single_results:
                if os.path.exists(res["path"]):
                    with open(res["path"], "rb") as f:
                        st.download_button(
                            label=res["label"],
                            data=f,
                            file_name=os.path.basename(res["path"]),
                            mime=res["type"],
                            key=f"dl_{res['label']}",
                        )
else:
    st.info("No depth result to save yet")
