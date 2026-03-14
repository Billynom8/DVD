import numpy as np
import torch
import torch.nn.functional as F

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

def get_window_index(T, window_size, overlap):
    """Get window indices for sliced inference."""
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            # Last window ensures full window_size length if possible
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res

def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    """Computes scale and shift for overlap alignment."""
    if mask is None:
        mask = np.ones_like(ref_frames)

    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0

    return scale, shift

def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = int(np.ceil(H * ratio))
    new_W = int(np.ceil(W * ratio))

    # Align to 16
    new_H = (new_H + 15) // 16 * 16
    new_W = (new_W + 15) // 16 * 16

    if new_H == H and new_W == W:
        return video_tensor, (H, W)

    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(
        new_H, new_W), mode="bilinear", align_corners=False)
    resized = resized.view(B, T, C, new_H, new_W)
    return resized, (H, W)

def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    # Ensure 4D for interpolate: [Batch/Time, Channel, H, W]
    if depth_np.ndim == 3:
        # [T, H, W] -> [T, 1, H, W]
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(1).float()
    else:
        # [T, H, W, C] -> [T, C, H, W]
        depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
        
    resized = F.interpolate(depth_tensor, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
    
    if depth_np.ndim == 3:
        # [T, 1, H, W] -> [T, H, W]
        return resized.squeeze(1).cpu().numpy()
    else:
        # [T, C, H, W] -> [T, H, W, C]
        return resized.permute(0, 2, 3, 1).cpu().numpy()

def pad_time_mod4(video_tensor):
    """Pads the temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T
