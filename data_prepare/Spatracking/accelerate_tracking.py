
#-------- import the base packages -------------
import sys
import os

import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.cuda

#-------- import spatialtracker -------------
from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer, read_video_from_path

#-------- import Depth Estimator -------------
from PIL import Image
from image_gen_aux import DepthPreprocessor

# set the arguments
parser = argparse.ArgumentParser()
# add the video and segmentation
parser.add_argument('--root', type=str, default='./assets', help='path to the video folder')
# set the gpu
# set the downsample factor
parser.add_argument('--downsample', type=float, default=0.8, help='downsample factor')
parser.add_argument('--grid_size', type=int, default=70, help='grid size')
# set the results outdir
parser.add_argument('--outdir', type=str, default='./vis_results', help='output directory')
# set the fps
parser.add_argument('--fps', type=float, default=1, help='fps')
# draw the track length
parser.add_argument('--len_track', type=int, default=0, help='len_track')
parser.add_argument('--output_fps', type=int, default=24, help='Output video fps and total frames')
# crop the video
parser.add_argument('--crop', action='store_true', help='whether to crop the video')
parser.add_argument('--crop_factor', type=float, default=1, help='whether to crop the video')
# backward tracking
parser.add_argument('--backward', action='store_true', help='whether to backward the tracking')
# if visualize the support points
parser.add_argument('--vis_support', action='store_true', help='whether to visualize the support points')
# query frame
parser.add_argument('--query_frame', type=int, default=0, help='query frame')
# set the visualized point size
parser.add_argument('--point_size', type=int, default=10, help='point size')
# take the RGBD as input
parser.add_argument('--rgbd', action='store_true', help='whether to take the RGBD as input')
parser.add_argument("--part", type=str)
parser.add_argument('--do_inv', action='store_true')

args = parser.parse_args()

# set input
root_dir = args.root
outdir = args.outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
# set the paras
grid_size = args.grid_size
downsample = args.downsample
# set the gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

def get_available_gpus():
    return list(range(torch.cuda.device_count()))

from accelerate import Accelerator

def write_depth_videos(depth_maps, save_name):
    depth_maps_np = (1-depth_maps).squeeze(1).cpu().numpy()
    depth_maps_uint8 = (depth_maps_np * 255).astype(np.uint8)
    depth_maps_rgb = [np.stack([frame]*3, axis=-1) for frame in depth_maps_uint8]
    clip = ImageSequenceClip(depth_maps_rgb, fps=24)
    clip.write_videofile(save_name, codec="libx264", logger=None)

def process_video(args, accelerator, model, MonoDEst_M, vid_name, root_dir, outdir, do_vis=True, do_inv=False):
    device = next(model.parameters()).device
    vid_dir = os.path.join(root_dir, vid_name)
    vid_name_without_ext = os.path.splitext(vid_name)[0]

    # Read original video fps
    cap = cv2.VideoCapture(vid_dir)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Set fps_vis to original fps
    fps_vis = original_fps

    # Read video
    try:
        video = read_video_from_path(vid_dir)
        video = np.flip(video[:49], axis=0).copy() if do_inv else video[:49]

        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    except:
        print(f"Error reading video {vid_name}")
        return

    _, T, _, H, W = video.shape
    segm_mask = np.ones((H, W), dtype=np.uint8)
    # print("video: ", video.shape)

    video = video.to(device)

    # ✅ Modified depth inference
    if not args.rgbd:
        video_depths = []
        with torch.no_grad():  # ✅ Wrap depth inference loop
            for i in range(video.shape[1]):
                frame = (video[0, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                depth = MonoDEst_M(Image.fromarray(frame))[0]
                depth_tensor = transforms.ToTensor()(depth)  # [1, H, W]
                video_depths.append(depth_tensor)
        depths = torch.stack(video_depths, dim=0).to(device)  # ✅ Use .to(device) instead of .cuda()
        # print("Depth maps shape:", depths.shape)
    else:
        depths = None

    # write_depth_videos(depths, f"{outdir}_depth/{vid_name}")

    # Use accelerator.unwrap_model() to get original model
    unwrapped_model = accelerator.unwrap_model(model)

    pred_tracks, pred_visibility, T_Firsts = (
        unwrapped_model(video, video_depth=depths,
                        grid_size=args.grid_size, backward_tracking=args.backward,
                        depth_predictor=None, grid_query_frame=args.query_frame,
                        segm_mask=torch.from_numpy(segm_mask)[None, None].to(device),
                        wind_length=12, progressive_tracking=False)
    )

    vis = Visualizer(save_dir=outdir, grayscale=False,
                     fps=args.output_fps, pad_value=0, linewidth=args.point_size,
                     tracks_leave_trace=args.len_track)

    msk_query = (T_Firsts == args.query_frame)
    if not msk_query.any():
        print(f"[Warning] No tracks for query frame {args.query_frame} in {vid_name}")
        return

    pred_tracks = pred_tracks[:, :, msk_query.squeeze()]
    pred_visibility = pred_visibility[:, :, msk_query.squeeze()]

    if do_inv:
        pred_tracks = torch.flip(pred_tracks, dims=[1])
        pred_visibility = torch.flip(pred_visibility, dims=[1])
    
    if do_vis:
        video_vis = vis.visualize(video=video, tracks=pred_tracks,
                                visibility=pred_visibility,
                                filename=f"{vid_name_without_ext}",
                                save_video=False)
        vis_folder = os.path.join(outdir, "vis")
        os.makedirs(vis_folder, exist_ok=True)
        vis.save_video(video_vis,
                       filename=f"{vid_name_without_ext}_tracking",
                       savedir=vis_folder)

    tracks_vis = pred_tracks.detach().cpu().numpy()
    visbility_vis = pred_visibility.detach().cpu().numpy()
    combined_data = {"tracks": tracks_vis, "visibility": visbility_vis}

    npy_folder = os.path.join(outdir, "npy")
    os.makedirs(npy_folder, exist_ok=True)
    np.save(os.path.join(npy_folder, f'{vid_name_without_ext}_tracks.npy'), combined_data)

    print(f"Processed {vid_name}. Results saved in {outdir}")

def get_splits(video_paths, parts):
    a, b = map(int, parts.split("/"))

    length = len(video_paths)
    split_size = length // b
    arrs = [video_paths[i * split_size:(i + 1) * split_size] for i in range(b)]
    current_inputs = arrs[a-1]
    print(f"Processing {a}/{b} of all times; Total Size {len(current_inputs)}/{length}")
    return current_inputs

def main():
    accelerator = Accelerator()
    device = accelerator.device
    args = parser.parse_args()

    accelerator.print(f"[Rank {accelerator.process_index}] Starting on device {device}")

    # os.makedirs(f"{args.outdir}_depth", exist_ok=True)

    # Initialize model (same across all ranks)
    model = SpaTrackerPredictor(
        checkpoint=os.path.join('../../checkpoints/spaT_final.pth'),
        interp_shape=(384, 576),
        seq_length=12
    )

    # Load depth model consistently across all processes
    depth_preprocessor = None
    if not args.rgbd:
        if accelerator.is_main_process:
            print("Loading depth model...")
        depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        depth_preprocessor.to(device)

    # Prepare components with accelerator
    if depth_preprocessor is not None:
        model, depth_preprocessor = accelerator.prepare(model, depth_preprocessor)
    else:
        model = accelerator.prepare(model)

    # Gather and split video list
    all_videos0 = sorted([f for f in os.listdir(args.root) if f.endswith('.mp4')])
    all_videos = []
    for vid_name in all_videos0:
        out_path = os.path.join(args.outdir, "npy", f"{vid_name[:-4]}_tracks.npy")
        if os.path.exists(out_path):
            accelerator.print(f"[{accelerator.process_index}] Skipping {vid_name}")
            continue
        else:
            all_videos.append(vid_name)
    accelerator.print(f"Total_number: {len(all_videos)}")

    # chunk_size = (len(all_videos) + accelerator.num_processes - 1) // accelerator.num_processes
    # start_idx = accelerator.process_index * chunk_size
    # end_idx = min(start_idx + chunk_size, len(all_videos))
    my_videos = get_splits(all_videos, args.part)

    do_vis = True
    for vid_count, vid_name in enumerate(my_videos):
        if vid_count > 10:
            do_vis = False
        out_path = os.path.join(args.outdir, "npy", f"{vid_name[:-4]}_tracks.npy")
        accelerator.print(f"[{accelerator.process_index}] Processing {vid_name}")
        process_video(args, accelerator, model, depth_preprocessor, vid_name, args.root, args.outdir, do_vis, do_inv=args.do_inv)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("✅ All videos processed")

if __name__ == "__main__":
    main()