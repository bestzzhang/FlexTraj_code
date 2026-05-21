# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import matplotlib.pyplot as plt
# from tqdm import tqdm
import imageio
from pycocotools import mask as coco_mask
from pathlib import Path
import pickle
from scipy.ndimage import binary_dilation
from decord import VideoReader


def load_rles_pickle(path="masks_rle.pkl"):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)

def rles_to_masks(rles):
    if not isinstance(rles, (list, tuple)):
        raise TypeError("rles 应为 list[dict]")

    n = len(rles)
    H, W = rles[0]["size"]
    masks = np.empty((n, H, W), dtype=np.bool_)

    for i, rle in enumerate(rles):
        mask_i = coco_mask.decode(rle)      # uint8 (H, W)
        masks[i] = mask_i.astype(bool)

    return masks

def load_seg_from_rle(rle_path, frame_start=0, load_m_frames=1):
    # Return: seg_masks: (n, H, W)
    rles = load_rles_pickle(rle_path)
    if load_m_frames == 1:
        rles = rles[frame_start]
        seg_masks = rles_to_masks(rles)
    else:
        m_rles = [rles[i] for i in range(frame_start, frame_start+load_m_frames)]
        seg_masks = np.stack([rles_to_masks(rles) for rles in m_rles])
    return seg_masks

def load_multiple_seg(rle_path, frame_starts=[0]):
    # Return: seg_masks: (n, H, W)
    rles = load_rles_pickle(rle_path)
    m_rles = [rles[i] for i in frame_starts]
    seg_masks = [rles_to_masks(rles) for rles in m_rles]
    return seg_masks

def min_pairwise_distances(points_a, points_b):
    """
    Returns:
        min_dists: (m,) array, minimum Euclidean distance from each point in A to any point in B
    """
    # Compute squared norms
    a_sq = np.sum(points_a ** 2, axis=1, keepdims=True)  # (m, 1)
    b_sq = np.sum(points_b ** 2, axis=1, keepdims=True).T  # (1, n)
    ab = points_a @ points_b.T  # (m, n)

    # Compute pairwise distances
    dists = np.sqrt(np.maximum(a_sq - 2 * ab + b_sq, 0))  # (m, n)

    # Minimum distance from each point in A to B
    min_dists = np.min(dists, axis=1)  # (m,)

    return min_dists


def is_new_points(points_a, points_b, threshold = 11):
    min_dists = min_pairwise_distances(points_a, points_b)
    return min_dists > threshold

def load_tracks_np(track_path, frame_start=0, frame_end=49, also_first=False):
    # Return: tracks: (n, 4900, 3), visibility: (n, 4900)
    tracks, visibility = np.load(track_path, allow_pickle=True).item().values()
    track_at_start = np.squeeze(tracks, axis=0)[0:1] if also_first else None
    tracks = np.squeeze(tracks, axis=0)[frame_start:frame_end]
    visibility = np.squeeze(visibility, axis=0)[frame_start:frame_end]

    if also_first:
        return tracks, visibility, track_at_start
    return tracks, visibility

def load_multiple_tracks(track_path, frame_start, frame_end):
    t1, t2 = track_path.split(",")
    tracks1, visibility1 = load_tracks_np(t1, frame_start, frame_end)
    tracks2, visibility2 = load_tracks_np(t2, frame_start, frame_end)

    mask = is_new_points(tracks1[0, :, :2], tracks1[-1, :, :2])
    tracks2 = tracks2[:, mask]
    visibility2 = visibility2[:, mask]

    n_tracks_arr = [len(tracks1[0]), len(tracks1[0]) + len(tracks2[0])]
    tracks = np.concatenate([tracks1, tracks2], axis=1)
    visibility = np.concatenate([visibility1, visibility2], axis=1)
    return tracks, visibility, n_tracks_arr

def load_video(video_path, frame_start=0, frame_ends=49):
    video_reader = VideoReader(video_path)
    frame_indices = list(range(frame_start, frame_ends))
    frames = video_reader.get_batch(frame_indices).numpy()
    return frames

def load_dwpose(dw_path, frame_start=0, frame_end=49):
    npy_array = np.load(dw_path)[frame_start: frame_end]
    return npy_array

def get_seg_ids(tracks: np.ndarray, segm_mask: np.ndarray, is_reverse=False):
    S, H, W = segm_mask.shape

    # Step 1: Compute integer pixel coordinates (from first frame)
    if not is_reverse:
        coords = np.rint(tracks[0, :, :2]).astype(int)
    else:
        coords = np.rint(tracks[-1, :, :2]).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)

    # Step 2: Extract segment index for each (x, y)
    # Shape: (S, N), where each [s, n] is 0 or 1 (segmentation presence)
    seg_at_points = segm_mask[:, coords[:, 1], coords[:, 0]]

    # Take the first mask that includes the point, or -1 if none
    has_mask = seg_at_points.any(axis=0)
    seg_ids = np.where(has_mask, seg_at_points.argmax(axis=0), -1)
    return seg_ids


def make_id_colors(tracks: np.ndarray, segm_mask: np.ndarray, seg_ids: np.ndarray, large_color_dist: bool=False):
    T, N, _ = tracks.shape
    S, H, W = segm_mask.shape

    # Generate colors
    r_values = np.linspace(0, 255, S+2, dtype=np.uint8)[1:]
    r_default = r_values[0]
    r_values = r_values[1:]

    R = np.where(seg_ids >= 0, r_values[seg_ids], r_default)
    # # For simplicity, just assign G/B based on index (you can change this)
    # X = np.linspace(0, 65536, N, dtype=np.int32) if large_color_dist else np.arange(N)
    # G, B = divmod(X, 256)

    # Initialize G and B
    G = np.zeros(N, dtype=np.uint8)
    B = np.zeros(N, dtype=np.uint8)

    # Assign G and B locally per segment
    # start_idxs = [int(i) for i in np.linspace(0, 25000, S+1)]
    # print(start_idxs)

    lst = np.arange(-1, S)
    np.random.shuffle(lst)
    for seg_id in lst:
        # start_idx = start_idxs[seg_id]
        mask = seg_ids == seg_id
        idxs = np.flatnonzero(mask)  # indexes of this segment
        local_idx = np.arange(len(idxs))
        local_val = np.linspace(0, 4900, len(idxs), dtype=np.int32) if large_color_dist else local_idx
        G[mask], B[mask] = divmod(local_val, 256)

    # Shape: (N, 3)
    colors = np.stack([R, G, B], axis=1)
    return colors

def make_id_colors_double(tracks, segm_mask_list, n_tracks_arr, large_color_dist: bool=False):
    S = len(segm_mask_list[0])

    # modified logic
    seg_ids = []

    st, ed = 0, n_tracks_arr[0]
    seg_ids_part = get_seg_ids(tracks[:, st:ed], segm_mask_list[0])
    seg_ids.append(seg_ids_part)

    st, ed = n_tracks_arr[0], n_tracks_arr[1]
    seg_ids_part = get_seg_ids(tracks[:, st:ed], segm_mask_list[1], is_reverse=True)
    seg_ids.append(seg_ids_part)

    seg_ids = np.concatenate(seg_ids)
    N = ed
    
    # Generate colors
    r_values = np.linspace(1, 255, S+1, dtype=np.uint8)[1:]
    R = np.where(seg_ids >= 0, r_values[seg_ids], 0)

    # # For simplicity, just assign G/B based on index (you can change this)
    # X = np.linspace(0, 25536, N, dtype=np.int32) if large_color_dist else np.arange(N)
    # G, B = divmod(X, 256)

    # Initialize G and B
    G = np.zeros(N, dtype=np.uint8)
    B = np.zeros(N, dtype=np.uint8)

    # Assign G and B locally per segment
    for seg_id in range(S):
        mask = seg_ids == seg_id
        idxs = np.flatnonzero(mask)  # indexes of this segment
        local_idx = np.arange(len(idxs))
        local_val = np.linspace(0, 25536, len(idxs), dtype=np.int32) if large_color_dist else local_idx
        G[mask], B[mask] = divmod(local_val, 256)

    # Shape: (N, 3)
    colors = np.stack([R, G, B], axis=1)
    return colors, seg_ids


def draw_track_fast(
    canvas: np.ndarray,
    tracks: np.ndarray,
    vector_colors: np.ndarray,
    visibility: np.ndarray = None,
    rect_size: int = 1,
):
    """
    Draws tracks on a video canvas with maximum efficiency using direct
    NumPy array manipulation, avoiding iterative OpenCV calls.
    """
    # 0. Get Canvas Dimensions
    T, H, W, C = canvas.shape

    # 1. Pre-computation and Data Conversion (same as previous optimization)
    _, N, _ = tracks.shape

    if visibility is None:
        visibility = np.ones((T, N), dtype=bool)

    rect_w = rect_size
    rect_h = rect_size / 1.5

    # 2. Main Loop (Iterating through frames)
    for t in range(T):
        # 3. Vectorized Data Preparation (same as previous optimization)
        visibility_t = visibility[t].astype(bool)
        if not np.any(visibility_t):
            continue

        visible_tracks = tracks[t][visibility_t]
        visible_colors = vector_colors[visibility_t]

        
        # 4. Vectorized Sorting (Crucial for correct occlusion)
        # Sort from FARTHEST to NEAREST. When we "stamp" rectangles, the
        # nearest ones (drawn last) will correctly overwrite the farther ones.
        if visible_tracks.shape[-1] == 3:
            visible_depth = visible_tracks[:, 2]
            sorted_indices = np.argsort(visible_depth)
            sorted_tracks = visible_tracks[sorted_indices]
            sorted_colors = visible_colors[sorted_indices]
        else:
            sorted_tracks = visible_tracks
            sorted_colors = visible_colors

        # 5. Vectorized Coordinate Calculation
        coords = sorted_tracks[:, :2]
        top_lefts = (coords - np.array([rect_w, rect_h])).astype(np.int32)
        bottom_rights = (coords + np.array([rect_w, rect_h])).astype(np.int32)
        
        # 6. High-Performance Direct Array Manipulation (The Core Improvement)
        # This loop replaces all cv2.rectangle calls.
        # It directly writes pixel values into the canvas array.
        frame_canvas = canvas[t]
        for i in range(len(sorted_tracks)):
            # Get the rectangle's coordinates
            tl = top_lefts[i]
            br = bottom_rights[i]

            # Clip coordinates to be within the canvas boundaries
            # This prevents indexing errors for tracks near or off the edge.
            x1, y1 = max(0, tl[0]), max(0, tl[1])
            x2, y2 = min(W, br[0]), min(H, br[1])

            # If the rectangle is not outside the screen, draw it
            if x1 < x2 and y1 < y2:
                # This is the "stamp": assign the color to the rectangular slice.
                frame_canvas[y1:y2, x1:x2] = sorted_colors[i]

    return canvas

def group_based_selection(seg_big, seg_small, ratio):
    ret_indices = np.arange(len(seg_big))

    # Lexicographical sort: sort by big then small segment
    sort_idx = np.lexsort((seg_small, seg_big))
    
    # Sort points and segments accordingly
    indices_sorted = ret_indices[sort_idx]
    seg_big_sorted = seg_big[sort_idx]
    seg_small_sorted = seg_small[sort_idx]

    # Create composite keys (big_id, small_id) as tuples
    seg_pairs = np.stack([seg_big_sorted, seg_small_sorted], axis=1)

    # Find unique segment pairs and their starting indices
    unique_pairs, start_indices = np.unique(seg_pairs, axis=0, return_index=True)
    small_groups = np.split(indices_sorted, start_indices[1:])

    unique_seg_big, seg_big_start_indices, seg_big_counts = np.unique(unique_pairs[:,0], return_index=True, return_counts=True)
    
    selection = []
    for i in range(len(unique_seg_big)):
        start_idx = seg_big_start_indices[i]
        end_idx = start_idx + seg_big_counts[i]
        x = small_groups[start_idx: end_idx]
        np.random.shuffle(x)
        x = np.concatenate(x)
        sample_num = int(len(x) * ratio)
        selection.append(x[:sample_num])

    selection = np.concatenate(selection)
    return selection

def foreground_prior_selection(seg_ids, masks, ratio):
    n, H, W = masks.shape
    if n == 0:
        return []

    # sort based on foreground scores
    sort_arr = []
    for i in range(n):
        mask = masks[i]
        S_inst, S_a = compute_foreground_scores(mask, H, W)
        sort_arr.append((S_inst, S_a, i))
    sorted_arr = sorted(sort_arr, key=lambda x: x[0], reverse=True)

    # collect
    accu_size = 0
    select_indices = []
    for _, S_a, i in sorted_arr:
        select_indices.append(np.flatnonzero(seg_ids == i))
        accu_size += S_a
        if accu_size >= ratio:
            break
    if len(select_indices) > 0:
        select_indices = np.concatenate(select_indices)
    return select_indices

def get_sparsified_visibility(
    pred_visibility,
    spatial_scale=1.0,
    temporal_scale=1.0,
    region_sparse_setting={}
):
    """
    Returns a boolean mask.

    Args:
        pred_tracks (np.ndarray): Array of shape [T, N, 3]
        pred_visibility (np.ndarray): Array of shape [T, N]
        spatial_scale (float): Fraction of spatial points to keep (0 < scale ≤ 1)
        temporal_scale (float): Fraction of frames to keep (0 ≤ scale ≤ 1)
        bg_indices (np.ndarray or None): Optional mask of shape [N]. 

    Returns:
        visibility_mask (np.ndarray): Boolean mask of shape [T, N]
    """

    if spatial_scale == 1.0 and temporal_scale == 1.0:
        return pred_visibility
    
    T, N = pred_visibility.shape

    # Step 1: Spatial sparsification
    if region_sparse_setting.get("spatial_mode", "random") == "random":
        all_indices = np.arange(N)
        keep_N = max(1, int(N * spatial_scale))
        selected_point_indices = np.random.permutation(all_indices)[:keep_N]
    else:
        # regional sparse
        if "small_seg_ids" in region_sparse_setting:
            selected_point_indices = group_based_selection(region_sparse_setting["seg_ids"], region_sparse_setting["small_seg_ids"], ratio=spatial_scale)
        else:
            selected_point_indices = foreground_prior_selection(region_sparse_setting["seg_ids"], region_sparse_setting["seg_masks"], ratio=spatial_scale)

    # Step 2: Temporal sparsification
    keep_T = max(2, int(T * temporal_scale))
    selected_frame_indices = (
        np.linspace(0, T - 1, num=keep_T, dtype=int) if keep_T > 0 else []
    )

    time_mask = np.zeros(T, dtype=bool)
    time_mask[selected_frame_indices] = True

    point_mask = np.zeros(N, dtype=bool)
    point_mask[selected_point_indices] = True

    visibility_mask = np.logical_and(time_mask[:, None], point_mask[None, :])
    visibility_mask = np.logical_and(pred_visibility, visibility_mask)

    return visibility_mask

def is_traj_inside_mask(mask, tracks, kernel_size=7):
    # select a ref points
    mask = binary_dilation(mask, structure=np.ones((kernel_size, kernel_size)))
    x = tracks[0, :, 0].astype(np.int64)
    y = tracks[0, :, 1].astype(np.int64)

    H, W = mask.shape

    mask_values = np.zeros_like(x, dtype=mask.dtype)
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    mask_values[valid] = mask[y[valid], x[valid]]

    return mask_values

def shifted_tracks(tracks, fg_mask, scale=1.0):
    if isinstance(scale, float):
        mode = np.random.choice(["x", "y", "xy"])
        scale = scale if mode != "xy" else (scale, scale)
    elif isinstance(scale, tuple):
        mode = "xy"

    true_indices = np.flatnonzero(fg_mask)
    selected_index = np.random.choice(true_indices)
    
    # scaled from ref_point
    ref_point = tracks[:, selected_index:selected_index+1, :2]
    points = tracks[:, true_indices, :2]

    shifted = points - ref_point
    # print(mode)
    if mode == "x":
        shifted[..., 0] *= scale
    elif mode == "y":
        shifted[..., 1] *= scale
    else:
        shifted[..., 0] *= scale[0]
        shifted[..., 1] *= scale[1]
    new_points = shifted + ref_point
    
    # update tracks
    tracks[:,true_indices, :2] = new_points

    return tracks

def compute_xy_shifted(fg1_area, fg2_area):
    # Get foreground pixel coordinates
    y1, x1 = np.where(fg1_area)
    y2, x2 = np.where(fg2_area)

    if len(x1) == 0 or len(x2) == 0:
        return 0, 0, 0  # No foreground found

    # Bounding box for fg1
    x1_min, x1_max = x1.min(), x1.max()
    y1_min, y1_max = y1.min(), y1.max()
    cx1, cy1 = (x1_min + x1_max) // 2, (y1_min + y1_max) // 2

    # Bounding box for fg2
    x2_min, x2_max = x2.min(), x2.max()
    y2_min, y2_max = y2.min(), y2.max()
    cx2, cy2 = (x2_min + x2_max) // 2, (y2_min + y2_max) // 2

    # Compute shift (fg1 center - fg2 center)
    delta_x = cx1 - cx2
    delta_y = cy1 - cy2

    return delta_x, delta_y, 1

def shift_mask(fg_area: np.ndarray, delta_x: int, delta_y: int):
    H, W = fg_area.shape
    shifted = np.zeros_like(fg_area, dtype=fg_area.dtype)

    # 1. Find bounding-box of the foreground in the *source* mask
    ys, xs = np.where(fg_area)
    if xs.size == 0:                     # No foreground
        return shifted, (0, -1, 0, -1), (0, -1, 0, -1)

    x1, x2 = xs.min(), xs.max()          # inclusive bounds in source
    y1, y2 = ys.min(), ys.max()

    # 2. Desired destination box after shifting
    dst_x1 = x1 + delta_x
    dst_y1 = y1 + delta_y
    dst_x2 = x2 + delta_x               # inclusive
    dst_y2 = y2 + delta_y

    # 3. Clip destination box to the canvas bounds
    clip_x1 = max(0, dst_x1)
    clip_y1 = max(0, dst_y1)
    clip_x2 = min(W - 1, dst_x2)
    clip_y2 = min(H - 1, dst_y2)

    # If the whole box is outside, nothing to paste
    if clip_x1 > clip_x2 or clip_y1 > clip_y2:
        return shifted, (0, -1, 0, -1), (0, -1, 0, -1)

    # 4. Compute matching source slice (same width/height as clipped destination)
    src_x1 = clip_x1 - dst_x1 + x1
    src_y1 = clip_y1 - dst_y1 + y1
    src_x2 = src_x1 + (clip_x2 - clip_x1)
    src_y2 = src_y1 + (clip_y2 - clip_y1)

    # 5. Paste
    shifted[clip_y1:clip_y2 + 1, clip_x1:clip_x2 + 1] = \
        fg_area[src_y1:src_y2 + 1, src_x1:src_x2 + 1]

    return shifted, (clip_x1, clip_x2, clip_y1, clip_y2), (src_x1, src_x2, src_y1, src_y2)

def shift_video(seg_masks: np.ndarray, canvas: np.ndarray, x_shifted: int, y_shifted: int):
    # seg_masks: (49, 1, 480, 720); canvas: (49, 480, 720, 3)
    for T in range(len(seg_masks)):
        # shift seg_masks
        seg_masks[T, 0], (clip_x1, clip_x2, clip_y1, clip_y2), (src_x1, src_x2, src_y1, src_y2) = shift_mask(seg_masks[T, 0], x_shifted, y_shifted)

        # shift canvs
        frame, new_frame = canvas[T], np.zeros_like(canvas[T])
        new_frame[clip_y1:clip_y2 + 1, clip_x1:clip_x2 + 1] = frame[src_y1:src_y2 + 1, src_x1:src_x2 + 1]
        canvas[T] = new_frame
    return seg_masks, canvas

def compute_foreground_scores(mask, H, W, border=20):
    # border 用来判断是否贴边，可以根据需要调整
    p_inst = np.sum(mask)  # 掩码总像素数
    if p_inst == 0:
        return -1, 0

    p_image = H * W
    S_a = p_inst / p_image  # 面积占比

    # 判断掩码中多少像素在边缘
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    p_border = np.sum(border_mask)
    p_instborder = np.sum(mask & border_mask)
    S_b = 1 - (p_instborder / p_border)  # 不贴边得分

    S_inst = 0.6 * S_b + 0.4 * S_a  # 综合得分
    return S_inst, S_a

def select_foreground_mask(masks: np) -> np.ndarray:
    """
    从多个候选掩码中选出一个最可能是前景的掩码。
    输入:
        masks: shape (n, H, W)，每个元素是bool类型的掩码
    返回:
        best_mask: shape (H, W)，选出的单个掩码
    """
    n, H, W = masks.shape
    if n == 1:
        return masks.squeeze(0),  np.sum(masks[0]) / (H*W)
    
    best_score = -1
    area_size = -1
    best_mask = None

    for i in range(n):
        mask = masks[i]

        S_inst, S_a = compute_foreground_scores(mask, H, W)
        if S_inst > best_score:
            best_score = S_inst
            best_mask = mask
            area_size = S_a

    if best_mask is None:
        # fallback：如果没有合法掩码，返回全0
        return np.zeros((H, W), dtype=bool), 0

    return best_mask, area_size


class Track_Loader():
    @staticmethod
    def load_track_real(track_path, rle_path, frame_start, num_frames, spatial_config, temporal_config, unalign_config):
        """
        shift_scale is 1: spatial ALIGN condition -> allow spatial_sparsify, temporal_sparsify
        shift_scale is not 1: spatial UNALIGN condition -> sparsify background to avoid leaking object true shape
        """
        spatial_scale = spatial_config.get("spatial_scale", 1.0)
        spatial_mode = spatial_config.get("spatial_mode", "random")
        temporal_scale = temporal_config.get("temporal_scale", 1.0)
        shift_scale = unalign_config.get("shift_scale", 1.0)
        unalign_mode = shift_scale != 1.0
        region_sparse_setting = {"spatial_mode": spatial_mode}

        if len(track_path.split(",")) > 1:
            seg_masks_list = load_multiple_seg(rle_path, [frame_start, frame_start+num_frames-1])
            tracks, visibility, n_tracks_arr = load_multiple_tracks(track_path, frame_start, frame_start+num_frames)
            vector_colors, seg_ids = make_id_colors_double(tracks, seg_masks_list, n_tracks_arr)
        elif track_path.split(".")[-1] == "npy":
            # load seg results and tracking results
            tracks, visibility = load_tracks_np(track_path, frame_start, frame_start+num_frames)
            paths = rle_path.split(",")
            rle_path = paths[0]

            # make seg ids
            if "reverse" in track_path:
                seg_masks = load_seg_from_rle(rle_path, frame_start+num_frames-1)
                seg_ids = get_seg_ids(tracks, seg_masks, is_reverse=True)
            else:
                seg_masks = load_seg_from_rle(rle_path, frame_start)
                seg_ids = get_seg_ids(tracks, seg_masks)

            if spatial_mode == "region":
                if len(paths) > 1:
                    small_rle_path = paths[1]
                    small_seg_masks = load_seg_from_rle(small_rle_path, frame_start)
                    small_seg_ids = get_seg_ids(tracks, small_seg_masks)
                    region_sparse_setting.update({"seg_ids": seg_ids, "small_seg_ids": small_seg_ids})
                else:
                    region_sparse_setting.update({"seg_ids": seg_ids, "seg_masks": seg_masks})

            # set color scheme
            vector_colors = make_id_colors(tracks, seg_masks, seg_ids=seg_ids)
        else:
            raise ValueError("track_path should ends in npy or pt")
        
        if not unalign_mode:
            visibility = get_sparsified_visibility(visibility, spatial_scale=spatial_scale, temporal_scale=temporal_scale, region_sparse_setting=region_sparse_setting)
        else:
            fg_mask, area_size = select_foreground_mask(seg_masks)
            if area_size > 0.2:
                is_inside_flag = is_traj_inside_mask(fg_mask, tracks)
                tracks = shifted_tracks(tracks, is_inside_flag, shift_scale)

                # manually spatial sparse for the background
                sparse_indices = np.flatnonzero(~is_inside_flag) # Points to sparsify (bg_indices)
                keep_sparse = max(1, int(sparse_indices.size * 0.05))
                selected_sparse_indices = np.random.permutation(sparse_indices)[:keep_sparse]
                is_inside_flag[selected_sparse_indices] = True
                visibility = np.logical_and(visibility, is_inside_flag[None, ...])

                visibility = get_sparsified_visibility(visibility, spatial_scale=1.0, temporal_scale=1.0)
            else: 
                random_sign = 1 if np.random.random() < 0.5 else -1
                tracks[..., 0] += np.random.randint(30, 50) * random_sign

        return tracks, visibility, vector_colors

    @staticmethod
    def load_track_syn(track_path, rle_path, frame_start, num_frames, temporal_config):
        '''
        Remove rendered region tracks and return sparsifid tracks, visibility and vector_colors
        '''
        npy_track = track_path.split(',')[0].strip()
        tracks, visibility = load_tracks_np(npy_track, frame_start, frame_start+num_frames)
        
        # Configs
        temporal_scale = temporal_config.get("temporal_scale", 1.0)
        
        ### 1. Get area to remove ###
        remove_area_masks = ~load_seg_from_rle(rle_path, frame_start) 
            
        ### 2. Remove tracks in the area ####
        fg_mask = is_traj_inside_mask(np.squeeze(remove_area_masks, axis=0), tracks, kernel_size=10)
        tracks = tracks[:, ~fg_mask]
        visibility = visibility[:, ~fg_mask]

        ### 3. Compute vector_colors and sparsify tracks ###
        seg_ids = get_seg_ids(tracks, remove_area_masks)
        vector_colors = make_id_colors(tracks, remove_area_masks, seg_ids)
        
        spatial_scale = 0.01 if len(visibility[0]) > 500 else 1.0
        visibility = get_sparsified_visibility(visibility, spatial_scale=spatial_scale, temporal_scale=temporal_scale)

        return tracks, visibility, vector_colors

    @staticmethod
    def load_track_syn_unaligned(track_path, rle_path, frame_start, num_frames, unalign_config):
        '''
        1. Remove rendered region tracks and return sparsifid tracks, visibility and vector_colors
        2. Return x_shifted, y_shifted (for unaligned case)
        '''
        npy_track = track_path.split(',')[0].strip()
        tracks, visibility = load_tracks_np(npy_track, frame_start, frame_start+num_frames)
        
        # Configs
        rle_path_unalign = unalign_config.get("rle_path_unalign", None)
        
        ### 1. Get area to remove ###
        fg1_area = ~load_seg_from_rle(rle_path, frame_start).squeeze(0)
        fg2_area = ~load_seg_from_rle(rle_path_unalign, frame_start).squeeze(0)
        x_shifted, y_shifted, is_valid = compute_xy_shifted(fg1_area, fg2_area)
        if is_valid > 0:
            fg2_area, _, _ = shift_mask(fg2_area, x_shifted, y_shifted)
            remove_area_masks = (fg1_area | fg2_area)[None, :, :]
        else:
            remove_area_masks = fg1_area[None, :, :]

        ### 2. Remove tracks in the area ####
        fg_mask = is_traj_inside_mask(np.squeeze(remove_area_masks, axis=0), tracks, kernel_size=10)
        tracks = tracks[:, ~fg_mask]
        visibility = visibility[:, ~fg_mask]

        ### 3. Compute vector_colors and sparsify tracks ###
        seg_ids = get_seg_ids(tracks, remove_area_masks)
        vector_colors = make_id_colors(tracks, remove_area_masks, seg_ids)
        visibility = get_sparsified_visibility(visibility, spatial_scale=0.01, temporal_scale=1.0)

        return tracks, visibility, vector_colors, x_shifted, y_shifted

    @staticmethod
    def load_track_humanvid(track_path, rle_path, frame_start, num_frames):
        # if render_track_unalign and rle_path_unalign are provided, need to produce unaligned condition maps
        npy_track, dw_track = [s.strip() for s in track_path.split(',')]

        # load seg results and tracking results
        tracks, visibility, track_at_start = load_tracks_np(npy_track, frame_start, frame_start+num_frames, also_first=True)
        fg_tracks = load_dwpose(dw_track, frame_start, frame_start+num_frames)
        
        # remove the tracked fg points
        remove_area_masks = load_seg_from_rle(rle_path, frame_start=0) 
        fg_mask = is_traj_inside_mask(np.squeeze(remove_area_masks, axis=0), track_at_start)

        # get bg_tracks
        bg_tracks = tracks[:, ~fg_mask]
        bg_visibility = visibility[:, ~fg_mask]
        bg_visibility = get_sparsified_visibility(bg_visibility, spatial_scale=0.01)

        segm_mask = np.zeros((1, 480, 720), dtype=bool)
        seg_ids = get_seg_ids(bg_tracks, segm_mask)
        bg_vector_colors = make_id_colors(bg_tracks, segm_mask = segm_mask, seg_ids=seg_ids)

        # get fg_tracks
        fg_depth = bg_tracks[0,:,2].min() * np.ones((*fg_tracks.shape[:2], 1), dtype=bool)
        fg_tracks = np.concatenate([fg_tracks, fg_depth], axis=-1)
        fg_visibility = np.ones(fg_tracks.shape[:2], dtype=bool)

        segm_mask = np.ones((1, 480, 720), dtype=bool)
        seg_ids = get_seg_ids(fg_tracks, segm_mask)
        fg_vector_colors = make_id_colors(fg_tracks, segm_mask = segm_mask, seg_ids=seg_ids, large_color_dist=True)
        
        # merge fg_tracks and bg_tracks
        tracks = np.concatenate([fg_tracks, bg_tracks], axis=1)
        visibility = np.concatenate([fg_visibility, bg_visibility], axis=1)
        vector_colors = np.concatenate([fg_vector_colors, bg_vector_colors])
        return tracks, visibility, vector_colors


class Canvas_Loader():
    @staticmethod
    def load_empty_canvas(num_frames, H, W, C):
        return np.zeros((num_frames, H, W, C), dtype=np.uint8)

    @staticmethod
    def load_canvas_syn(render_track, frame_start, num_frames):
        canvas = load_video(render_track, frame_start, frame_start+num_frames)
        return canvas
    
    @staticmethod
    def load_canvas_syn_sparse(render_track, small_rle, frame_start, num_frames):
        canvas = load_video(render_track, frame_start, frame_start+num_frames)
        seg_masks = load_seg_from_rle(small_rle, frame_start, load_m_frames=num_frames) 
        mask = np.random.rand(seg_masks.shape[1]) > 0.3
        mask = np.any(seg_masks[:, mask], axis=1)
        canvas = canvas * mask[...,None]
        return canvas

    @staticmethod
    def load_canvas_syn_unalign(frame_start, num_frames, unalign_config, x_shifted, y_shifted):
        rle_path_unalign = unalign_config.get("rle_path_unalign", None)
        render_track_unalign = unalign_config.get("render_track_unalign", None)
        seg_masks = ~load_seg_from_rle(rle_path_unalign, frame_start, load_m_frames=num_frames) 
        canvas = load_video(render_track_unalign, frame_start, frame_start+num_frames)
        seg_masks, canvas = shift_video(seg_masks, canvas, x_shifted, y_shifted)
        seg_masks = seg_masks.repeat(3, axis=1).transpose(0, 2, 3, 1) 
        return canvas, seg_masks


class Visualizer():
    def __init__(self, h=480, w=720, c=3, num_frames=49):
        self.H, self.W, self.C = h, w, c
        self.num_frames = num_frames
        self.cond_option = "id+color"

    def get_cond_maps(self, track_path, rle_path, frames, use_color=True, frame_start=0, is_real_video=True,
                      spatial_config={}, temporal_config={}, unalign_config={}):
        # 1. get the id_maps
        if is_real_video:
            process_func = self.get_id_maps_for_real
        else:
            process_func = self.get_id_maps_for_synthesis
                
        id_maps = process_func(track_path, rle_path, frame_start, spatial_config, temporal_config, unalign_config)
        id_maps = torch.from_numpy(id_maps) if id_maps is not None else None

        # 2. get the color_maps
        if "color" in self.cond_option.split("+"):
            color_maps = torch.zeros_like(id_maps)
            # print("use_color: ", use_color)
            if use_color:
                if is_real_video:
                    mask = (id_maps[...,0] > 0).unsqueeze(-1)
                else:
                    mask = (id_maps[...,0] > 20).unsqueeze(-1)
                color_maps = frames*mask
        else:
            color_maps = None

        # 3. add maps to cond_maps_dict
        cond_maps_dict = {
            "id_maps": id_maps,
            "color_maps": color_maps
        }
        for k in list(cond_maps_dict.keys()):
            if cond_maps_dict[k] is None:
                cond_maps_dict.pop(k)
        
        return cond_maps_dict
    
    def get_id_maps_for_real(self, track_path, rle_path, frame_start,
                      spatial_config, temporal_config, unalign_config):
        # 1. Get tracks and canvas
        tracks, visibility, vector_colors = Track_Loader.load_track_real(track_path, rle_path, frame_start, self.num_frames, spatial_config, temporal_config, unalign_config)
        canvas = Canvas_Loader.load_empty_canvas(self.num_frames, self.H, self.W, self.C)

        # 2. Set up rect_size
        spatial_mode = spatial_config.get("spatial_mode", "random")
        spatial_scale = spatial_config.get("spatial_scale", 1.0)
        if spatial_mode=="random" and spatial_scale < 0.1:
            rect_size = 6
        elif len(tracks[0]) < 300:
            rect_size = 6
        else:
            rect_size = 3

        # 3. Create id_maps
        id_maps = draw_track_fast(
            canvas,
            tracks=tracks,
            vector_colors=vector_colors,
            visibility=visibility,
            rect_size=rect_size,
        )

        return id_maps

    def get_id_maps_for_synthesis(self, track_path, rle_path, frame_start,
                      spatial_config, temporal_config, unalign_config):

        temporal_scale = temporal_config.get("temporal_scale", 1.0)
        unalign_mode = "rle_path_unalign" in unalign_config

        # 1. Get case_name
        t_paths = track_path.split(',')
        r_paths = rle_path.split(',')
        rle_path = r_paths[0]
        small_rle_path = r_paths[1] if len(r_paths) > 1 else None
        if len(t_paths) == 1:
            case_name = "direct_load"
        else:
            ext_name = t_paths[-1].strip().split('.')[-1]
            if ext_name == "mp4":
                case_name = "dense|temporal|unalign"
            elif ext_name == "npy":
                case_name = "random_spatial"
        
        # 2. Get tracks and canvas
        if case_name == "dense|temporal|unalign":
            npy_path, mp4_path = t_paths
            if not unalign_mode:
                tracks, visibility, vector_colors = Track_Loader.load_track_syn(npy_path, rle_path, frame_start, self.num_frames, temporal_config)
                if small_rle_path:
                    canvas = Canvas_Loader.load_canvas_syn_sparse(mp4_path, small_rle_path, frame_start, self.num_frames)
                else:
                    canvas = Canvas_Loader.load_canvas_syn(mp4_path, frame_start, self.num_frames)
                    
            else:
                tracks, visibility, vector_colors, x_shifted, y_shifted = Track_Loader.load_track_syn_unaligned(npy_path, rle_path, frame_start, self.num_frames, unalign_config)
                canvas, seg_masks = Canvas_Loader.load_canvas_syn_unalign(frame_start, self.num_frames, unalign_config, x_shifted, y_shifted)
        elif case_name == "random_spatial":
            tracks, visibility, vector_colors  = Track_Loader.load_track_humanvid(track_path, rle_path, frame_start, self.num_frames)
            canvas = Canvas_Loader.load_empty_canvas(self.num_frames, self.H, self.W, self.C)

        # 3. Create id_maps
        if case_name ==  "direct_load":
            id_maps = Canvas_Loader.load_canvas_syn(track_path, frame_start, self.num_frames)
        else:
            id_maps = draw_track_fast(
                canvas.copy(),
                tracks=tracks,
                vector_colors=vector_colors,
                visibility=visibility,
                rect_size=6,
            )
            if unalign_mode: # post-process
                id_maps = np.where(seg_masks, canvas, id_maps)

        # 4. Sparify the final
        if temporal_scale != 1.0:
            keep_T = max(2, int(self.num_frames * temporal_scale))
            selected_frame_indices = (
                np.linspace(0, self.num_frames - 1, num=keep_T, dtype=int) if keep_T > 0 else []
            )
            time_mask = np.ones(self.num_frames, dtype=bool)
            time_mask[selected_frame_indices] = False
            id_maps[time_mask] = 0

        return id_maps
    
 