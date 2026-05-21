# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import time

from tqdm import tqdm
from models.spatracker.models.core.spatracker.spatracker import get_points_on_a_grid
from models.spatracker.models.core.model_utils import smart_cat
from models.spatracker.models.build_spatracker import (
    build_spatracker,
)
from models.spatracker.models.core.model_utils import (
    meshgrid2d, bilinear_sample2d, smart_cat
)

def vis_points(new_grid_pts, last_track_xy, save_name):
    import matplotlib.pyplot as plt
    H, W = 384, 576
    plt.figure(figsize=(6, 4))
    plt.imshow(torch.ones(H, W, 3))  # white background

    if new_grid_pts != None:
        plt.scatter(new_grid_pts[:, 0], new_grid_pts[:, 1], c='red', s=10, label='new_grid_pts')
    if last_track_xy != None:
        plt.scatter(last_track_xy[:, 0], last_track_xy[:, 1], c='green', s=10, label='last_track_xy')

    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)

class SpaTrackerPredictor(torch.nn.Module):
    def __init__(
        self, checkpoint="cotracker/checkpoints/cotracker_stride_4_wind_8.pth",
        interp_shape=(384, 512),
        seq_length=16
    ):
        super().__init__()
        self.interp_shape = interp_shape
        self.support_grid_size = 6
        model = build_spatracker(checkpoint, seq_length=seq_length)

        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video,  # (1, T, 3, H, W)
        video_depth = None, # (T, 1, H, W)
        # input prompt types:
        # - None. Dense tracks are computed in this case. You can adjust *query_frame* to compute tracks starting from a specific frame.
        # *backward_tracking=True* will compute tracks in both directions.
        # - queries. Queried points of shape (1, N, 3) in format (t, x, y) for frame index and pixel coordinates.
        # - grid_size. Grid of N*N points from the first frame. if segm_mask is provided, then computed only for the mask.
        # You can adjust *query_frame* and *backward_tracking* for the regular grid in the same way as for dense tracks.
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,  # Segmentation mask of shape (B, 1, H, W)
        grid_size: int = 0,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
        depth_predictor=None,
        wind_length: int = 8,
        progressive_tracking: bool = False,
        num_rolling: int = 4,
    ):
        tracks, visibilities = self._compute_sparse_tracks(
            video,
            queries,
            segm_mask,
            grid_size,
            add_support_grid=False, #(grid_size == 0 or segm_mask is not None),
            grid_query_frame=grid_query_frame,
            backward_tracking=backward_tracking,
            video_depth=video_depth,
            depth_predictor=depth_predictor,
            wind_length=wind_length,
            num_rolling=num_rolling
        )
        
        return tracks, visibilities

    def get_query_depth(self, depth, queries_wo_depth):
        depth_interp=[]
        for i in range(queries_wo_depth.shape[1]):
            depth_interp_i = bilinear_sample2d(depth[queries_wo_depth[:, i:i+1, 0].long()], 
                                queries_wo_depth[:, i:i+1, 1], queries_wo_depth[:, i:i+1, 2])
            depth_interp.append(depth_interp_i)

        depth_interp = torch.cat(depth_interp, dim=1)
        return depth_interp

    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
        depth_predictor=None,
        num_rolling=4,
        video_depth=None,
        wind_length=8,
    ):
        B, T, C, H, W = video.shape

        assert B == 1
        self.grid_size = grid_size
        ## ----------- initialize the RGBD video ----------- ##
        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear")
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1]) # shape: [1, 49, 3, 384, 576]
        # print("video: ", video.shape) 

        video_depth = F.interpolate(video_depth,
                                     tuple(self.interp_shape), mode="nearest")
        

        depths = video_depth
        rgbds = torch.cat([video, depths[None,...]], dim=2) # shape: [1, 49, 4, 384, 576]

        ## ----------- estimate the video depth ----------- ##
        grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)

        # get min_distance
        distances = torch.norm(grid_pts[0,0] - grid_pts[0,1:], dim=1)
        min_distance = torch.min(distances)

        queries = torch.cat(
            [torch.zeros_like(grid_pts[:, :, :1]), grid_pts],
            dim=2,
        ) # shape: [1, 4900, 3]

        ## ----------- get the 3D queries ----------- ##
        depth_interp = self.get_query_depth(depth = video_depth, queries_wo_depth = queries)

        # print("queries (w/o D):", queries.shape)
        queries = smart_cat(queries, depth_interp,dim=-1) # shape: [1, 4900, 4]
        # print("queries (w/ D):", queries.shape)

        # # Free the memory of depth_predictor
        # del depth_predictor
        # torch.cuda.empty_cache()

        ## ----------- Run inference  ----------- ##
        t0 = time.time()
        # tracks, __, visibilities = self.model(rgbds=rgbds, queries=queries, iters=6, wind_S=wind_length)
        # print("Time taken for inference: ", time.time()-t0)
        # print("tracks: ", tracks.shape) shape: [1, 49, 4900, 3]
        # print("visibilities: ", visibilities.shape) shape: [1, 49, 4900]

        tracks, visibilities = self.rolling_tracking2(rgbds=rgbds,
                                                          depths=depths, 
                                                          queries=queries, 
                                                          min_distance=min_distance, 
                                                          wind_length=wind_length, 
                                                          num_rolling=num_rolling,
                                                          WH=(W, H))

        # thr = 0.9
        # visibilities = visibilities > thr
        
        # ## ----------- Write Query Points to Tracking Output ----------- ##
        # for i in range(len(queries)):
        #     queries_t = queries[i, :tracks.size(2), 0].to(torch.int64)
        #     arange = torch.arange(0, len(queries_t))

        #     tracks[i, queries_t, arange] = queries[i, :tracks.size(2), 1:]
        #     visibilities[i, queries_t, arange] = True
        
        # ## ----------- Prepare final outputs ----------- ##
        # T_First = queries[..., :tracks.size(2), 0].to(torch.uint8)
        # tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        # tracks[:, :, :, 1] *= H / float(self.interp_shape[0])
        # return tracks, visibilities, T_First

        return tracks, visibilities

    '''
    def rolling_tracking1(self, rgbds, depths, queries, min_distance, wind_length, num_rolling):
        print("min_distance: ", min_distance)
        keyframes = torch.linspace(0, rgbds.shape[1] - 1, steps = num_rolling)

        caches = [queries.clone()]
        for i in range(len(keyframes)-1):
            k1, k2 = int(keyframes[i]), int(keyframes[i+1])
            tracks, __, visibilities = self.model(rgbds=rgbds[:,k1:k2+1], queries=queries, iters=6, wind_S=wind_length)
            last_track_xy = tracks[0, -1, :, :2] # 4900, 2

            # update new_query
            cached, queries = self.update_queries(last_track_xy = last_track_xy, 
                                depths = depths, 
                                threshold = 1.5 * min_distance, 
                                time_idx = i, 
                                device=tracks.device)
            caches.append(cached)
            
        queries = torch.cat(caches, dim=1)
        print(f"total number points: {len(queries[0])}")
        tracks, __, visibilities = self.model(rgbds=rgbds, queries=queries, iters=6, wind_S=wind_length)
        
        torch.save(tracks, "tracks.pt")
        return tracks, visibilities
    '''

    def rolling_tracking2(self, rgbds, depths, queries, min_distance, wind_length, num_rolling, WH):
        T = rgbds.shape[1]
        thr = 0.9
        W, H = WH

        keyframes = torch.linspace(0, T - 1, steps = num_rolling)
        keyframes = [int(k) for k in keyframes]

        tracks_list = []
        visibilities_list = []
        num_new_list = []
        num_new = queries.shape[1]

        # total: all points added so far
        all_points_visility = None
        all_points_vis_list = [] 

        # print("queries: ", queries.shape)
        # print("keyframes: ", keyframes)
        for i in range(len(keyframes)-1):
            if all_points_visility is None:
                all_points_visility = torch.ones(num_new, dtype=bool, device="cuda")
            else:
                all_points_visility = torch.cat([all_points_visility, torch.ones(num_new, dtype=bool, device="cuda")])

            k1, k2 = int(keyframes[i]), int(keyframes[i+1])
            tracks, __, visibilities = self.model(rgbds=rgbds[:,k1:k2+1], queries=queries, iters=6, wind_S=wind_length)
            visibilities = visibilities > thr

            # print(i, tracks.shape, visibilities.shape)
            
            # Obtain the propagate track in last frame
            last_track_xy = tracks[0, -1, :, :2] # (n, 3)
            last_track_vis = visibilities[0, -1, :]  # n
            last_track_xy = last_track_xy[last_track_vis]

            # update visibility for all points added so far
            all_points_visility[all_points_visility==1] = last_track_vis
            
            # Write Query Points to Tracking Output
            tracks[0, 0, :] = queries[0, :, 1:] 
            visibilities[0, 0, :] = True

            # Append to the Track List
            if i != len(keyframes)-2:
                tracks = tracks[:,:-1]
                visibilities = visibilities[:,:-1]

            tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
            tracks[:, :, :, 1] *= H / float(self.interp_shape[0])

            tracks_list.append(tracks.clone()) # (1, timestep, n, 3)
            visibilities_list.append(visibilities.clone())
            all_points_vis_list.append(all_points_visility.clone())
            num_new_list.append(num_new)

            # Update queris
            if i != len(keyframes)-2:
                # update new_query
                num_new, queries = self.update_queries(last_track_xy = last_track_xy, 
                                    depths = depths, 
                                    threshold = 1.5 * min_distance, 
                                    time_idx = i, 
                                    device=tracks.device)
        
        # print("num_new_list: ", num_new_list)
        # print("tracks_list: ", [x.shape for x in tracks_list])
        # print("all_points_vis_list: ", [x.shape for x in all_points_vis_list])
        
        # torch.save(tracks_list, "tracks_list.pt")
        # torch.save(visibilities_list, "visibilities_list.pt")
        # torch.save(all_points_vis_list, "all_points_vis_list.pt")

        # calculate the valid points left for each points_group after each update (also index range)
        num_updates = num_points_group = len(num_new_list)
        valids_points_left = {}
        valid_index_range = {}
        all_index_range = {}
        for update_j in range(num_updates):
            all_points_vis = all_points_vis_list[update_j]
            num_groups_exists = update_j+1
            added_points_num_before = 0
            start_index = 0
            for group_k in range(num_groups_exists):
                points_num_curr_group = num_new_list[group_k]
                valid_count = all_points_vis[added_points_num_before: added_points_num_before+points_num_curr_group].sum().item()

                valids_points_left[(group_k, update_j)] = valid_count
                valid_index_range[(group_k, update_j)] = (start_index, start_index+valid_count)
                all_index_range[(group_k, update_j)] = (added_points_num_before, added_points_num_before+points_num_curr_group)

                added_points_num_before += points_num_curr_group
                start_index += valid_count
        
        # print("valids_points_left")
        # print(valids_points_left)
        # print("valid_index_range")
        # print(valid_index_range)
        # print("all_index_range")
        # print(all_index_range)

        final_tracks = []
        final_visibilities = []
        keyframes[-1] = keyframes[-1] + 1

        for group_k in range(num_points_group):
            num = num_new_list[group_k]
            update_start = group_k
            time_start = keyframes[group_k]

            tracks = torch.zeros([T,num,3], device="cuda")
            visibilities = torch.zeros([T,num], device="cuda", dtype=torch.bool)
            for update_j in range(update_start, num_updates):
                t_st = keyframes[update_j]
                t_ed = keyframes[update_j+1]

                if update_j > group_k:
                    ai_st, ai_ed = all_index_range[(group_k, update_j-1)]
                    vi_st, vi_ed = valid_index_range[(group_k, update_j-1)]
                    mask = all_points_vis_list[update_j-1][ai_st: ai_ed]
                    tracks[t_st:t_ed, mask] = tracks_list[update_j][0,:,vi_st:vi_ed]
                    visibilities[t_st:t_ed, mask] = visibilities_list[update_j][0,:,vi_st:vi_ed]
                else:
                    tracks[t_st:t_ed] = tracks_list[update_j][0,:,-num:]
                    visibilities[t_st:t_ed] = visibilities_list[update_j][0,:,-num:]

            tracks = tracks[time_start:, :]
            visibilities = visibilities[time_start:, :]
            final_tracks.append(tracks)
            final_visibilities.append(visibilities)


        return final_tracks, final_visibilities
        

    def get_new_pts(self, last_track_xy, threshold, device):
        new_grid_pts = get_points_on_a_grid(self.grid_size, self.interp_shape, device=device)[0]
        
        # filter
        dists = torch.cdist(new_grid_pts, last_track_xy, p=2)
        min_dists, _ = torch.min(dists, dim=1)
        new_grid_pts = new_grid_pts[min_dists > threshold]
        return new_grid_pts

    def update_queries(self, last_track_xy, threshold, time_idx, depths, device):
        new_grid_pts = self.get_new_pts(last_track_xy, threshold=threshold, device=device)
        num_new = len(new_grid_pts)
        print(f"{num_new} new points added!")
        # vis_points(new_grid_pts, last_track_xy, f"{time_idx}.png")
        grid_pts = torch.cat([last_track_xy, new_grid_pts]).unsqueeze(0)

        update_queries = torch.cat(
            [time_idx*torch.ones_like(grid_pts[:, :, :1]), grid_pts],
            dim=2,
        ) 
        depth_interp = self.get_query_depth(depth = depths, queries_wo_depth = update_queries)
        update_queries = smart_cat(update_queries, depth_interp,dim=-1) 
        return num_new, update_queries
        
    



        
