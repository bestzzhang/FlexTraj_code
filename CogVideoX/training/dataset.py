import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from cond_vis import Visualizer
import json

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
            frames = video_reader.get_batch(indices)
            frames = frames[: self.max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds

class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

class VideoDatasetWithFlexControl(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.seg_column = kwargs.pop("seg_column", None)
        self.tracking_column = kwargs.pop("tracking_column", None)
        self.cond_option = kwargs.pop("cond_option", None)

        self.is_random_spatial = kwargs.pop("is_random_spatial", False)
        self.is_random_temporal = kwargs.pop("is_random_temporal", False)
        self.is_random_shift = kwargs.pop("is_random_shift", False)

        self.vis = Visualizer(self.cond_option)
        self.action_groups = {}
        super().__init__(*args, **kwargs)

    def init_action_groups(self):
        self.action_groups = {"seg": {}, "track": {}}
        for track_path, seg_path in zip(self.tracking_paths, self.seg_paths):
            track_video_path = track_path.split(",")[-1].strip()
            _, action_id, _ = self.get_syn_info(seg_path)

            if action_id not in self.action_groups["seg"]:
                self.action_groups["seg"][action_id] = []
                self.action_groups["track"][action_id] = []
            self.action_groups["seg"][action_id].append(seg_path)
            self.action_groups["track"][action_id].append(track_video_path)
        
        d = self.action_groups["seg"]
        print("logging action counts ...")
        for k in sorted(d):
            print(f"action {k}: {len(d[k])}")

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

    def _preprocess_video(self, path: Path, rle_path: Path, track_path: Path) -> torch.Tensor:
        is_real_video = len(track_path.split(',')) == 1
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_length = len(video_reader)

        frame_start = 0 if is_real_video else np.random.randint(0, video_length - self.max_num_frames)
        frame_end = frame_start + self.max_num_frames
        frame_indices = list(range(frame_start, frame_end))

        frames = video_reader.get_batch(frame_indices).float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

        image = frames[:1].clone() if self.image_to_video else None
        
        spatial_scale = 10 ** np.random.uniform(-3, -1) if self.is_random_spatial else 1.0
        temporal_scale = np.random.choice([1.0, 0.2, 0.1, 0]) if self.is_random_temporal else 1.0
        unalign_config = {}
        
        if self.is_random_shift:
            if is_real_video:
                unalign_config["shift_scale"] = np.random.uniform(0.5, 1.5)
            else:
                if len(self.action_groups) == 0:
                    self.init_action_groups()
                _, action_id, _ = self.get_syn_info(rle_path)
                
                unalign_idx = np.random.randint(len(self.action_groups["seg"][action_id])) 
                unalign_config["rle_path_unalign"] = self.action_groups["seg"][action_id][unalign_idx]
                unalign_config["render_track_unalign"] = self.action_groups["track"][action_id][unalign_idx]

        cond_maps_dict = self.vis.get_cond_maps(track_path, 
                                                rle_path, 
                                                frame_start=frame_start,
                                                is_real_video=is_real_video,
                                                spatial_scale=spatial_scale,
                                                temporal_scale=temporal_scale,
                                                unalign_config=unalign_config)
        for k in cond_maps_dict:
            cond_maps = cond_maps_dict[k].permute(0, 3, 1, 2).contiguous() 
            cond_maps_dict[k] = cond_maps / 127.5 - 1
        
        return image, frames, cond_maps_dict

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        seg_path = self.data_root.joinpath(self.seg_column)
        tracking_path = self.data_root.joinpath(self.tracking_column)

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
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(seg_path, "r", encoding="utf-8") as file:
            seg_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(tracking_path, "r", encoding="utf-8") as file:
            tracking_paths = []
            for line in file.readlines():
                if len(line.strip()) > 0:
                    track_path = ",".join([str(self.data_root.joinpath(s.strip())) for s in line.split(",")])
                    tracking_paths.append(track_path)
        
        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        self.seg_paths = seg_paths
        self.tracking_paths = tracking_paths 
        return prompts, video_paths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        image, video, cond_maps_dict = self._preprocess_video(self.video_paths[index], 
                                                                self.seg_paths[index], 
                                                                self.tracking_paths[index])
        cond_keys = list(cond_maps_dict.keys())

        return {
            "prompt": self.id_token + self.prompts[index],
            "image": image,
            "video": video,
            "cond_keys": cond_keys,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
            "index": index,
            **cond_maps_dict
        }

class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []

class VideoDatasetUniform(Dataset):
    def __init__(self, 
                data_root: str,
                dataset_file: Optional[str] = None,
                max_num_frames: int = 49,
                id_token: Optional[str] = None,
                height_buckets: List[int] = None,
                width_buckets: List[int] = None,
                frame_buckets: List[int] = None,
                load_tensors: bool = False,
                random_flip: Optional[float] = None,
                image_to_video: bool = False,
                run_abl: str = "",
                **kwargs) -> None:
        super().__init__()

        self.max_num_frames = max_num_frames
        self.random_flip = random_flip
        self.image_to_video = image_to_video
        self.is_random_spatial = kwargs.pop("is_random_spatial", False)
        self.is_random_temporal = kwargs.pop("is_random_temporal", False)
        self.is_random_shift = kwargs.pop("is_random_shift", False)
        dataset_option = kwargs.pop("dataset_option", "all")
        ns_from_each = kwargs.pop("ns_from_each", -1) # this is used for debug only
        self.curr_task = self.set_curr_task()
        print("Current Task: ", self.curr_task)

        if dataset_file is None:
            raise NotImplementedError
        else:
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

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )


        cond_option = kwargs.pop("cond_option", None)
        self.vis = Visualizer(cond_option, run_abl=run_abl)

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

    def _preprocess_video(self, path: Path, rle_path: Path, track_path: Path, task_configs: dict) -> torch.Tensor:
        is_real_video = task_configs["is_real_video"]
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_length = len(video_reader)

        frame_start = 0 if is_real_video or task_configs["auto_trim"] else np.random.randint(0, video_length - self.max_num_frames)
        frame_end = frame_start + self.max_num_frames
        frame_indices = list(range(frame_start, frame_end))

        frames = video_reader.get_batch(frame_indices)
        cond_maps_dict = self.vis.get_cond_maps(track_path, 
                                                rle_path, 
                                                frames=frames,
                                                use_color=task_configs["use_color"],
                                                frame_start=frame_start,
                                                is_real_video=is_real_video,
                                                spatial_config=task_configs["spatial_config"],
                                                temporal_config=task_configs["temporal_config"],
                                                unalign_config=task_configs["unalign_config"])
        
        frames = frames.float().permute(0, 3, 1, 2).contiguous()
        frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)
        image = frames[:1].clone() if self.image_to_video else None

        for k in cond_maps_dict:
            cond_maps = cond_maps_dict[k].permute(0, 3, 1, 2).contiguous() 
            cond_maps_dict[k] = cond_maps / 127.5 - 1
        
        return image, frames, cond_maps_dict

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
        
        # prefix = datasets_dict.pop("prefix")
        txt_file = "/".join(dataset_file.split('/')[:-1]) + "/train_prefix.txt"
        with open(txt_file) as f:
            prefix = f.read().splitlines()[0]
        print("prefix: ", prefix)

        prompts, video_paths, seg_paths, tracking_paths, belongs_to = [], [], [], [], []
        
        action_groups = {}
        datasets_meta = {}
        print("dataset information")
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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        task_configs = self.prepare_task_configs(index)
        image, video, cond_maps_dict = self._preprocess_video(self.video_paths[index], 
                                                                self.seg_paths[index], 
                                                                self.tracking_paths[index],
                                                                task_configs)
        cond_keys = list(cond_maps_dict.keys())
        belong = self.belongs_to[index]

        flag_insert = "unaligned, " if len(task_configs["unalign_config"]) > 0 else ""
        return {
            "prompt": flag_insert + self.prompts[index],
            "image": image,
            "video": video,
            "cond_keys": cond_keys,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
            "index": index,
            "belong": belong,
            **cond_maps_dict
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
        
    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0