import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_control import WanVideoPipeline, ModelConfig
import argparse
import os
import numpy as np
from lora_util import load_lora
from diffsynth.trainers.flex_dataset import get_validation_inputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_ckpt", default=None)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--test_root", default="")
    parser.add_argument("--output_folder", default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--kv_cache", action="store_true")
    return parser.parse_args()

def v_concat(videos):
    concat_frames = []
    for i in range(len(videos[0])):
        stacked = np.concatenate([v[i] for v in videos], axis=0)
        concat_frames.append(stacked)
    return concat_frames

def load_pipe(args):
    model_configs=[
        ModelConfig(path=[
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
        ]),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    ]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs
    )
    load_lora(pipe, lora_checkpoint=args.trained_ckpt)
    pipe.enable_vram_management()
    return pipe


args = parse_args()
os.makedirs(args.output_folder, exist_ok=True)

if __name__ == "__main__":
    pipe = load_pipe(args)
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    for inputs in get_validation_inputs(args.validation_file, args.test_root, height=args.height, width=args.width, num_frames=args.num_frames):
        save_name = args.output_folder + inputs['vid_name'] + f".mp4"
        if os.path.exists(save_name):
            continue
        video = pipe(
            prompt=inputs['prompt'],
            negative_prompt=neg_prompt,
            control_videos=inputs['control_videos'],
            input_image=inputs['input_image'],
            num_frames=args.num_frames,
            seed=args.seed, tiled=True,
            width=args.width, height=args.height,
            num_inference_steps=50,
            kv_cache=args.kv_cache
        )
        
        cat_video = v_concat([inputs['video'], inputs['control_videos'][0], video])
        save_video(cat_video, save_name, fps=24, quality=5)