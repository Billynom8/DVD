import os
import cv2
import numpy as np
import torch
import imageio
import imageio_ffmpeg
import subprocess
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

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

def read_video(video_path):
    """Standard video read."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), fps

def read_video_early_downsample(video_path, target_w):
    """Read video and downsample during loading to save RAM."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ratio = target_w / orig_w
    new_w = target_w
    new_h = int(np.ceil(orig_h * ratio))
    
    # Align to 16
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

def _ensure_3d_depth(depth):
    """Internal helper to ensure depth is [T, H, W]."""
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    
    # Remove all dimensions of size 1
    depth = np.squeeze(depth)
    
    if depth.ndim == 2:
        # [H, W] -> [1, H, W]
        depth = depth[np.newaxis, ...]
    elif depth.ndim == 4:
        # Still 4D? Likely [T, H, W, C] where C > 1. Take first channel.
        depth = depth[..., 0]
        
    return depth

def save_video_ffmpeg(depth, origin_fps, input_path, output_dir, grayscale=True, bit_depth=8):
    """
    Unified FFmpeg-based video saver.
    Handles 8-bit (libx264) and 10-bit (libx265 main10).
    """
    depth = _ensure_3d_depth(depth)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).split('.')[0]
    suffix = 'gray' if grayscale else 'color'
    bit_suffix = f'{bit_depth}bit'
    output_path = os.path.join(output_dir, f"{base_name}_{suffix}_{bit_suffix}.mp4")

    # Normalize depth to 0-1
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    if bit_depth == 10:
        codec = "libx265"
        pix_fmt = "yuv420p10le"
        profile = ["-profile:v", "main10"]
        crf = "18"
        scale_factor = 65535.0
        dtype = np.uint16
        input_pix_fmt = "rgb48le"
    else:
        codec = "libx264"
        pix_fmt = "yuv420p"
        profile = []
        crf = "20"
        scale_factor = 255.0
        dtype = np.uint8
        input_pix_fmt = "rgb24"

    if grayscale:
        frames = (depth_norm * scale_factor).astype(dtype)
        frames_out = np.stack([frames]*3, axis=-1)
    else:
        cmap = plt.get_cmap('Spectral_r')
        frames_out = (cmap(depth_norm)[:, :, :, :3] * scale_factor).astype(dtype)

    T, H, W, C = frames_out.shape

    cmd = [
        ffmpeg_exe, "-y",
        "-loglevel", "error", # Prevent pipe deadlock by minimizing output
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", input_pix_fmt,
        "-r", str(origin_fps),
        "-i", "-",
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        *profile,
        "-crf", crf,
        "-preset", "slow",
        output_path
    ]

    print(f"Encoding {bit_depth}-bit {suffix}: {' '.join(cmd)}")
    # Use DEVNULL for stdout to prevent hanging
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    try:
        for frame in tqdm(frames_out, desc=f"Saving {bit_depth}-bit {suffix}", leave=False):
            process.stdin.write(frame.tobytes())
        process.stdin.close()
    except Exception as e:
        process.kill()
        _, stderr = process.communicate()
        raise RuntimeError(f"Error writing to FFmpeg: {str(e)}\nFFmpeg stderr: {stderr.decode()}")
        
    process.wait()

    if process.returncode != 0:
        _, stderr = process.communicate()
        raise RuntimeError(f"FFmpeg failed with return code {process.returncode}\nStderr: {stderr.decode()}")

    # Clear memory
    del frames_out
    
    return output_path

def save_depth_png_sequence(depth, input_path, output_dir):
    """
    Saves depth map as 16-bit PNG sequence (Master).
    """
    depth = _ensure_3d_depth(depth)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).split('.')[0]
    seq_dir = os.path.join(output_dir, f"{base_name}_16bit_png")
    os.makedirs(seq_dir, exist_ok=True)
    
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_16 = (depth_norm * 65535.0).astype(np.uint16)
    
    for i in tqdm(range(len(depth_16)), desc="Saving 16-bit PNGs", leave=False):
        img = Image.fromarray(depth_16[i], mode='I;16')
        img.save(os.path.join(seq_dir, f"{i:04d}.png"))
    
    with open(os.path.join(seq_dir, "range.txt"), "w") as f:
        f.write(f"min: {d_min}\nmax: {d_max}")

    return seq_dir
