from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .cond_vis import Visualizer
import json
from PIL import Image
import imageio
import os

def load_video(path, start_position, max_num_frames):
    reader = imageio.get_reader(path)
    video_length = int(reader.count_frames())
    if start_position == "random":
        frame_start = np.random.randint(0, video_length - max_num_frames)
    else:
        frame_start = 0
    
    frames = []
    for fid in range(frame_start, frame_start + max_num_frames):
        frame = reader.get_data(fid)
        frame = Image.fromarray(frame)
        frames.append(frame)
    reader.close()

    return frames, frame_start

def crop_and_resize(frames, target_height, target_width):
    if frames is None:
        return
    width, height = frames[0].size
    if height == target_height and width == target_width:
        return frames
    for i in range(len(frames)):
        image = frames[i]
        scale = max(target_width / width, target_height / height)
        image = transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        image = transforms.functional.center_crop(image, (target_height, target_width))
        frames[i] = image
    return frames 


class VideoDatasetUniform(Dataset):
    def __init__(self, 
                dataset_file: Optional[str] = None,
                max_num_frames: int = 49,
                height: int = 480,
                width: int = 720,
                **kwargs) -> None:
        super().__init__()

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.is_random_spatial = kwargs.pop("is_random_spatial", False)
        self.is_random_temporal = kwargs.pop("is_random_temporal", False)
        self.is_random_shift = kwargs.pop("is_random_shift", False)
        dataset_option =  kwargs.pop("dataset_option", "all")
        ns_from_each = kwargs.pop("ns_from_each", -1) # this is used for debug only

        cond_option = "id+color"
        
        self.curr_task = self.set_curr_task()
        print("Current Task: ", self.curr_task)

        (
            self.prompts,
            self.video_paths,
            self.seg_paths,
            self.tracking_paths,
            self.belongs_to,
            self.datasets_meta,
            self.action_groups
        ) = self._load_dataset_from_json(dataset_file, self.curr_task, dataset_option, ns_from_each)
    
        with open(dataset_file) as f:
            json_dict_from_file = json.load(f)
            self.use_color_prob = json_dict_from_file.get("use_color_prob", 0.5)
            self.use_random_prob = json_dict_from_file.get("use_random_prob", 0.5)

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        
        if len(self.video_paths) != len(self.seg_paths):
            raise ValueError(
                f"Expected length of seg_paths and videos to be the same but found {len(self.seg_paths)=} and {len(self.video_paths)=}. Please ensure that the number of seg_paths and videos match in your dataset."
            )
        
        if len(self.tracking_paths) != len(self.seg_paths):
            raise ValueError(
                f"Expected length of tracking_paths and videos to be the same but found {len(self.tracking_paths)=} and {len(self.video_paths)=}. Please ensure that the number of tracking_paths and videos match in your dataset."
            )

        self.vis = Visualizer(cond_option)

    def __len__(self) -> int:
        return len(self.video_paths)
    
    @staticmethod
    def init_action_groups(tracking_paths, seg_paths):
        action_groups = {"seg": {}, "track": {}}
        for track_path, seg_path in zip(tracking_paths, seg_paths):
            track_video_path = track_path.split(",")[-1].strip()
            _, action_id, _ = VideoDatasetUniform.get_syn_info(seg_path)

            if action_id not in action_groups["seg"]:
                action_groups["seg"][action_id] = []
                action_groups["track"][action_id] = []
            action_groups["seg"][action_id].append(seg_path)
            action_groups["track"][action_id].append(track_video_path)
        
        d = action_groups["seg"]
        print("logging action counts ...")
        for k in sorted(d):
            print(f"action {k}: {len(d[k])}")
        return action_groups

    @staticmethod
    def get_syn_info(seg_path): # "actor_xx_ani_yy_traj_z.pkl"
        xx_idx = len("actor_")
        yy_idx = len("actor_xx_ani_") 
        zz_idx = len("actor_xx_ani_yy_traj_")
        s = str(seg_path).split("/")[-1]
        actor_id = s[xx_idx:xx_idx+2]
        action_id = s[yy_idx:yy_idx+2]
        traj_id = s[zz_idx:zz_idx+1]
        return actor_id, action_id, traj_id


    @staticmethod
    def _load_dataset_from_local_path(data_root, kwargs) -> Tuple[List[str], List[str], List[str]]:
        if not data_root.exists():
            raise ValueError("Root folder for videos does not exist")
        
        seg_column = kwargs.pop("seg_column", None)
        tracking_column = kwargs.pop("tracking_column", None)
        video_column = kwargs.pop("video_column", None)
        caption_column = kwargs.pop("caption_column", None)
        
        prompt_path = data_root.joinpath(caption_column)
        video_path = data_root.joinpath(video_column)
        seg_path = data_root.joinpath(seg_column)
        tracking_path = data_root.joinpath(tracking_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )
        if not seg_path.exists() or not seg_path.is_file():
            raise ValueError(
                "Expected `--seg_column` to be path to a file in `--data_root` containing line-separated seg information."
            )
        if not tracking_path.exists() or not tracking_path.is_file():
            raise ValueError(
                "Expected `--tracking_column` to be path to a file in `--data_root` containing line-separated seg information."
            )
            

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        with open(seg_path, "r", encoding="utf-8") as file:
            seg_paths = []
            for line in file.readlines():
                if len(line.strip()) > 0:
                    s_path = ",".join([str(data_root.joinpath(s.strip())) for s in line.split(",")])
                    seg_paths.append(s_path)

        with open(tracking_path, "r", encoding="utf-8") as file:
            tracking_paths = []
            for line in file.readlines():
                if len(line.strip()) > 0:
                    track_path = ",".join([str(data_root.joinpath(s.strip())) for s in line.split(",")])
                    tracking_paths.append(track_path)
        
        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{video_column=}` to be a path to a file in `{data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths, seg_paths, tracking_paths
    
    @staticmethod
    def repeat_rows(rows_tuple, repeats = 1):
        if repeats == 1:
            return rows_tuple
        
        rows_list = list(rows_tuple)
        total_nums = len(rows_list[0])
        
        frac, integer = np.modf(repeats)
        sample_n = int(total_nums*frac)
        sample_idxs = np.random.choice(total_nums, size=int(sample_n), replace=False)

        repeated_rows = []
        for arr in rows_list:
            sampled = [arr[idx] for idx in sample_idxs]
            repeated = arr * int(integer) + sampled
            repeated_rows.append(repeated)

        return repeated_rows
    
    def set_curr_task(self):
        task_mode = sum([self.is_random_spatial, self.is_random_temporal, self.is_random_shift])
        if task_mode == 3:
            return "multi_tasks"
        elif task_mode == 0:
            return "dense"
        elif task_mode == 1:
            if self.is_random_spatial:
                return "sparse_spatial"
            elif self.is_random_temporal:
                return "sparse_temporal"
            elif self.is_random_shift:
                return "unaligned"
        else:
            if self.is_random_spatial and self.is_random_temporal:
                return "sparse"
            else:
                raise NotImplementedError("Not implement this task.") 

    @staticmethod
    def is_select_dataset(dataset_option, is_real_video):
        if dataset_option == "all":
            return True
        elif dataset_option == "real":
            return is_real_video
        elif dataset_option == "syn":
            return not is_real_video

    @staticmethod
    def has_curr_task(curr_task, support_tasks):
        if curr_task == "multi_tasks":
            return True
        elif curr_task in support_tasks:
            return True
        elif curr_task == "sparse":
            support_sparse = ("sparse_temporal" in support_tasks) or ("sparse_spatial" in support_tasks)
            return support_sparse
        return False

    @staticmethod
    def _load_dataset_from_json(dataset_file, curr_task, dataset_option, ns_from_each):
        with open(dataset_file, 'r') as f:
            datasets_dict = json.load(f)
        
        txt_file = "/".join(dataset_file.split('/')[:-1]) + "/train_prefix.txt"
        with open(txt_file) as f:
            prefix = f.read().splitlines()[0]
        print("prefix: ", prefix)

        prompts, video_paths, seg_paths, tracking_paths, belongs_to = [], [], [], [], []
        
        action_groups = {}
        datasets_meta = {}
        for i, d in enumerate(datasets_dict["datasets"]):
            if not VideoDatasetUniform.has_curr_task(curr_task, d["support_tasks"]):
                continue
            if not VideoDatasetUniform.is_select_dataset(dataset_option, d['is_real_video']):
                continue

            data_root = Path(prefix + d["data_root"])
            prompts_i, video_paths_i, seg_paths_i, tracking_paths_i = VideoDatasetUniform._load_dataset_from_local_path(data_root, d)

            if not d['is_real_video'] and "unaligned" in d["support_tasks"]:
                action_groups = VideoDatasetUniform.init_action_groups(tracking_paths_i, seg_paths_i)

            prompts_i, video_paths_i, seg_paths_i, tracking_paths_i = VideoDatasetUniform.repeat_rows((prompts_i, video_paths_i, seg_paths_i, tracking_paths_i), repeats = d["repeats"])
            print(f"{i}: {len(video_paths_i)}")

            if ns_from_each > 0:
                prompts_i, video_paths_i, seg_paths_i, tracking_paths_i = prompts_i[:ns_from_each], video_paths_i[:ns_from_each], seg_paths_i[:ns_from_each], tracking_paths_i[:ns_from_each]

            prompts += prompts_i
            video_paths += video_paths_i
            seg_paths += seg_paths_i
            tracking_paths += tracking_paths_i
            belongs_to += [i for _ in range(len(video_paths_i))]
            datasets_meta[i] = {
                "support_tasks": d['support_tasks'], 
                "is_real_video": d['is_real_video'],
                "auto_trim": d.get('auto_trim', False)
            }

            if curr_task in ["multi_tasks", "sparse"]:
                if curr_task == "multi_tasks":
                    datasets_meta[i]['tasks_weights'] = d['multi_tasks_weights']
                else:
                    datasets_meta[i]['tasks_weights'] = d['sparse_tasks_weights']

        return prompts, video_paths, seg_paths, tracking_paths, belongs_to, datasets_meta, action_groups

    def _preprocess_video(self, path: Path, rle_path: Path, track_path: Path, task_configs: dict) -> torch.Tensor:
        if task_configs["is_real_video"] or task_configs["auto_trim"]:
            start_position = "frame_0"
        else:
            start_position = "random"
        frames, frame_start = load_video(path.as_posix(), start_position, max_num_frames=self.max_num_frames)

        cond_videos = self.vis.get_cond_maps(track_path, 
                                                rle_path, 
                                                frames=frames,
                                                use_color=task_configs["use_color"],
                                                frame_start=frame_start,
                                                is_real_video=task_configs["is_real_video"],
                                                spatial_config=task_configs["spatial_config"],
                                                temporal_config=task_configs["temporal_config"],
                                                unalign_config=task_configs["unalign_config"])
        
        frames = crop_and_resize(frames, target_height=self.height, target_width=self.width)
        cond_videos = [crop_and_resize(video, target_height=self.height, target_width=self.width) for video in cond_videos]
        return frames, cond_videos
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        task_configs = self.prepare_task_configs(index)
        video, control_videos = self._preprocess_video(self.video_paths[index], 
                                                                self.seg_paths[index], 
                                                                self.tracking_paths[index],
                                                                task_configs)
        belong = self.belongs_to[index]

        flag_insert = "unaligned, " if len(task_configs["unalign_config"]) > 0 else ""
        return {
            "prompt": flag_insert + self.prompts[index],
            "video": video,
            "control_videos": control_videos,
            "index": index,
            "belong": belong
        }

    def prepare_task_configs(self, index):
        dataset_i = self.belongs_to[index]
        dataset_meta = self.datasets_meta[dataset_i]
        is_real_video = dataset_meta["is_real_video"]
        task_configs = {
            "use_color": True,
            "spatial_config": {},
            "temporal_config": {},
            "unalign_config": {},
            "is_real_video": is_real_video,
            "auto_trim": dataset_meta["auto_trim"]
        }

        if self.curr_task in ["multi_tasks", "sparse"]:
            sampled_task = np.random.choice(
                dataset_meta["support_tasks"],
                p=dataset_meta["tasks_weights"]
            )
        else:
            sampled_task = self.curr_task

        if np.random.rand() > self.use_color_prob or sampled_task == "unaligned":
            task_configs["use_color"] = False

        if sampled_task == "dense":
            return task_configs
        elif sampled_task == "sparse_spatial":
            if is_real_video:
                if np.random.rand() > self.use_random_prob:
                    spatial_mode = "region"
                    spatial_scale = np.random.choice([0.3, 0.2, 0.1])
                else:
                    spatial_mode = "random"
                    spatial_scale = 10 ** np.random.uniform(-3, -1)
                task_configs["spatial_config"] = {"spatial_scale": spatial_scale, "spatial_mode": spatial_mode}
            else:
                spatial_mode = "random" if dataset_i == 4 else "region"
                task_configs["spatial_config"] = {"spatial_mode": spatial_mode}
            # print("spatial_mode: ", spatial_mode)
        elif sampled_task == "sparse_temporal":
            task_configs["temporal_config"] = {"temporal_scale": np.random.choice([0.2, 0.1, 0.1, 0, 0])}
            # print("temp_mode")
        elif sampled_task == "unaligned":
            if is_real_video:
                task_configs["unalign_config"] = {"shift_scale": np.random.uniform(0.3, 1.8)}
            else:
                _, action_id, _ = self.get_syn_info(self.seg_paths[index])
                unalign_idx1, unalign_idx2 = np.random.choice(len(self.action_groups["seg"][action_id]), size=2, replace=False)

                rle_path_unalign = self.action_groups["seg"][action_id][unalign_idx1]
                render_track_unalign = self.action_groups["track"][action_id][unalign_idx1]
                if rle_path_unalign == self.seg_paths[index]:
                    rle_path_unalign = self.action_groups["seg"][action_id][unalign_idx2]
                    render_track_unalign = self.action_groups["track"][action_id][unalign_idx2]

                task_configs["unalign_config"] = {
                    "rle_path_unalign": rle_path_unalign,
                    "render_track_unalign": render_track_unalign
                }
        else:
            raise NotImplementedError("Encounter unimplemented task!")

        return task_configs


def get_validation_inputs(validation_file, 
                          test_root,
                          spatial_scale=1.0, 
                          spatial_mode="random", 
                          temporal_scale=1.0, 
                          use_color=False, 
                          height=480,
                          width=736,
                          num_frames=49,
                          one_validation_is_enough=False):
    with open(validation_file) as f:
        val_datasets = json.load(f)
    
    vis = Visualizer(num_frames=num_frames)

    for ds_i, val_dataset in enumerate(val_datasets):
        prefix = test_root + val_dataset["prefix"] 
        validations = val_dataset["validations"]
        for idx, val in enumerate(validations):
            is_real_video = val["is_real_video"]
            if len(val["tracking_map_path"].split(',')) > 1:
                t1, t2 = val['tracking_map_path'].split(',')
                t1 = os.path.join(prefix, t1)
                t2 = os.path.join(prefix, t2)
                tracking_map_path = ",".join([t1, t2])
            else:
                tracking_map_path = os.path.join(prefix, val["tracking_map_path"])
                
            if len(val["seg_map_path"].split(',')) > 1:
                s1, s2 = val["seg_map_path"].split(',')
                s1 = os.path.join(prefix, s1)
                s2 = os.path.join(prefix, s2)
                seg_map_path =",".join([s1, s2])
            else:
                seg_map_path = os.path.join(prefix, val["seg_map_path"])

            if "use_color" in val:
                assert "validation_videos" in val, "must provide color information"
                use_color = val["use_color"]
            else:
                use_color = use_color
            
            if "spatial_config" in val:
                spatial_config = val["spatial_config"]
            else:
                spatial_config = {"spatial_scale": spatial_scale, "spatial_mode": spatial_mode}

            if "temporal_config" in val:
                temporal_config = val["temporal_config"]
            else:
                temporal_config = {"temporal_scale": temporal_scale}

            if "unalign_config" in val:
                unalign_config = val["unalign_config"]
            else:
                unalign_config = {}

            if "validation_video" in val:
                video,_ = load_video(prefix+val["validation_video"], 0, 49)
                input_image = video[0]
            else:
                video = None

            if "input_image" in val:
                input_image = Image.open(prefix+val["input_image"]).convert("RGB")
            
            vid_name = val.get("name", f"{ds_i}_{idx}")

            cond_videos = vis.get_cond_maps(
                track_path = tracking_map_path, 
                rle_path = seg_map_path, 
                frames = video,
                use_color=use_color,
                frame_start=0, 
                is_real_video=is_real_video,
                spatial_config=spatial_config, 
                temporal_config=temporal_config, 
                unalign_config=unalign_config
            )

            video = crop_and_resize(video, target_height=height, target_width=width)
            input_image = crop_and_resize([input_image], target_height=height, target_width=width)[0]
            cond_videos = [crop_and_resize(video, target_height=height, target_width=width) for video in cond_videos]

            valid_frame_idxs = val.get("valid_frame_idxs", None)
            if valid_frame_idxs is not None:
                null_frame = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
                for cond_video in cond_videos:
                    for i in range(len(cond_video)):
                        if i not in valid_frame_idxs:
                            cond_video[i] = null_frame

            yield {
                "vid_name": vid_name,
                "video": video,
                "input_image": input_image,
                "control_videos": cond_videos,
                "prompt": val["validation_prompt"]
            }

            if one_validation_is_enough:
                break


def v_concat(videos):
    concat_frames = []
    for i in range(len(videos[0])):
        stacked = np.concatenate([v[i] for v in videos], axis=0)
        concat_frames.append(stacked)
    return concat_frames