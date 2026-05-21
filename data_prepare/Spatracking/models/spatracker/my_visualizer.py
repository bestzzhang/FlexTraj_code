from pathlib import Path
from typing import List, Sequence, Union

import torch
from torchvision.io import read_video, write_video
from torchvision.utils import draw_keypoints


class Visualizer:
    """
    Overlay 2-D point tracks on a video and write the result to disk.

    Parameters
    ----------
    save_dir : str | Path
        Directory in which the MP4 files will be written.
    fps : int | None
        FPS to write.  If ``None`` we keep the source video's fps (when the
        input is a path); otherwise this value overrides it.
    palette : Sequence[tuple[int, int, int]], optional
        RGB colour palette, one entry per track.  If you pass fewer colours
        than tracks, they will cycle automatically.
    """

    _default_palette: Sequence[tuple[int, int, int]] = [
        # 13 bright-ish, colour-blind-friendly colours
        (255,   0,   0),   # red
        (  0, 255,   0),   # green
        (  0,   0, 255),   # blue
        (255, 255,   0),   # yellow
        (255,   0, 255),   # magenta
        (  0, 255, 255),   # cyan
        (255, 165,   0),   # orange
        (128,   0, 128),   # purple
        (  0, 128, 128),   # teal
        (128, 128,   0),   # olive
        (  0,   0, 128),   # navy
        (128,   0,   0),   # maroon
        (  0, 128,   0),   # dark green
    ]

    def __init__(
        self,
        save_dir: Union[str, Path],
        fps: Union[int, None] = None,
        palette: Sequence[tuple[int, int, int]] | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.palette = palette or self._default_palette

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def visualize(
        self,
        *,
        video: Union[str, Path, torch.Tensor],
        tracks: List[torch.Tensor],
        visibility: List[torch.Tensor],
        filename: str,
        radius: int = 2,
        width: int = 2,
    ) -> Path:
        """
        Create <filename>.mp4 inside ``self.save_dir`` and return its path.

        Parameters
        ----------
        video
            Either a path to an existing video **or** a tensor of
            shape (T, C, H, W) in uint8 RGB 0-255.
        tracks / visibility
            Each entry is length-Tᵢ.  Tracks that started later than the
            first frame are automatically aligned at the tail end.
        """
        frames, fps = self._load_video(video)
        num_frames, _, H, W = frames.shape

        if len(tracks) != len(visibility):
            raise ValueError("tracks and visibility must be the same length")

        # overlay every track
        for i, (track, vis) in enumerate(zip(tracks, visibility)):
            col = self.palette[i % len(self.palette)]
            start_f = num_frames - track.shape[0]

            for local_f, (xy, vm) in enumerate(zip(track, vis)):
                keypts = xy[vm].unsqueeze(0)  # (1, Nv, 2)
                global_f = start_f + local_f
                frames[global_f] = draw_keypoints(
                    frames[global_f],
                    keypts,
                    colors=col,
                    radius=radius,
                    width=width,
                )

        # write the result
        out_path = self.save_dir / f"{filename}.mp4"
        write_video(
            filename=str(out_path),
            video_array=frames.permute(0, 2, 3, 1).contiguous(),  # (T, H, W, C)
            fps=self.fps or fps or 10,
            video_codec="libx264",
        )
        return out_path

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_video(video):
        """
        Accepts
        • a file path (str/Path)  → reads with read_video
        • a 5-D tensor            → shape (1, T, 3, H, W), uint8 0-255

        Returns
        -------
        frames : (T, 3, H, W)  uint8 RGB
        fps    : int | None    (None for tensor input)
        """
        # ---- case 1: path --------------------------------------------------
        if isinstance(video, (str, Path)):
            frames, _, meta = read_video(str(video), pts_unit="sec")
            fps = meta["video_fps"]
            frames = frames.permute(0, 3, 1, 2).contiguous()           # (T, C, H, W)
            return frames.to(torch.uint8), fps

        # ---- case 2: tensor (1, T, 3, H, W) -------------------------------
        if isinstance(video, torch.Tensor):
            # squeeze batch dim and make sure channel dim is second
            frames = video.contiguous()
            return frames.to(torch.uint8), None

        # ---- anything else -------------------------------------------------
        raise TypeError("video must be a file path or a torch.Tensor")
