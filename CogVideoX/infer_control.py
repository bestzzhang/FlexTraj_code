import os
import argparse
import torch
from torchvision.io import read_video, write_video
import torchvision.transforms.functional as F
from training.test_utils import load_pipeline_kv
import json
from PIL import Image
from training.cond_vis import Visualizer
import decord
from training.utils import get_inputs_from_json, infer
decord.bridge.set_bridge("torch")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--test_root", type=str)
    parser.add_argument("--out_folder", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    vis = Visualizer()
    pipe = load_pipeline_kv(args.model_path, args.checkpoint_path)
    os.makedirs(args.out_folder, exist_ok=True)
    
    for i, inputs in enumerate(get_inputs_from_json(args.validation_file, args.test_root)):
        concatenated, vid_name = infer(
            inputs, vis, pipe, kv_cache=True, num_inference_steps=40
        )

        out_path = os.path.join(args.out_folder, f"{vid_name}.mp4")
        write_video(out_path, concatenated * 255, fps=24)
