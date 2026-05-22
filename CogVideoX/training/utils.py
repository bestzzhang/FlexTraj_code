import gc
import inspect
from typing import Optional, Tuple, Union

import torch
import os
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils import load_image, load_video
from torchvision import transforms
import json
from torchvision.io import read_video
import torchvision.transforms.functional as F
import shutil

logger = get_logger(__name__)


def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()

    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )

    if use_8bit and use_4bit:
        raise ValueError("Cannot set both `use_8bit` and `use_4bit` to True.")

    if (use_torchao and (use_8bit or use_4bit)) or use_cpu_offload_optimizer:
        try:
            import torchao

            torchao.__version__
        except ImportError:
            raise ImportError(
                "To use optimizers from torchao, please install the torchao library: `USE_CPP=0 pip install torchao`."
            )

    if not use_torchao and use_4bit:
        raise ValueError("4-bit Optimizers are only supported with torchao.")

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy", "came"]
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if (use_8bit or use_4bit) and optimizer_name not in ["adam", "adamw"]:
        raise ValueError("`use_8bit` and `use_4bit` can only be used with the Adam and AdamW optimizers.")

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": epsilon,
            "weight_decay": weight_decay,
            "decouple": prodigy_decouple,
            "use_bias_correction": prodigy_use_bias_correction,
            "safeguard_warmup": prodigy_safeguard_warmup,
        }

    elif optimizer_name == "came":
        try:
            import came_pytorch
        except ImportError:
            raise ImportError("To use CAME, please install the came-pytorch library: `pip install came-pytorch`")

        optimizer_class = came_pytorch.CAME

        init_kwargs = {
            "lr": learning_rate,
            "eps": (1e-30, 1e-16),
            "betas": (beta1, beta2, beta3),
            "weight_decay": weight_decay,
        }

    if use_cpu_offload_optimizer:
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize, optimizer_class=optimizer_class, offload_gradients=offload_gradients, **init_kwargs
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
    position_delta: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)

    # print("before: ", grid_crops_coords)
    (y1, x1), (y2, x2) = grid_crops_coords # ((0, 0), (30, 45)
    y1 += position_delta[1]
    y2 += position_delta[1]
    x1 += position_delta[0]
    x2 += position_delta[0]
    grid_crops_coords = (y1, x1), (y2, x2)
    # print("after: ", grid_crops_coords)

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")



def load_media(media_path, max_frames=49, transform=None):
    """Load video or image frames and convert to tensor
    
    Args:
        media_path (str): Path to video or image file
        max_frames (int): Maximum number of frames to load
        transform (callable): Transform to apply to frames
        
    Returns:
        Tuple[torch.Tensor, float]: Video tensor [T,C,H,W] and FPS
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])
    
    # Determine if input is video or image based on extension
    ext = os.path.splitext(media_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']
    
    if is_video:
        frames = load_video(media_path)
    else:
        # Handle image as single frame
        image = load_image(media_path)
        frames = [image]
    
    # Ensure we have exactly max_frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        last_frame = frames[-1]
        while len(frames) < max_frames:
            frames.append(last_frame.copy())
            
    # Convert frames to tensor
    video_tensor = torch.stack([transform(frame) for frame in frames])
    
    return video_tensor

def get_inputs_from_json(validation_file, test_root='./', spatial_scale=1.0, spatial_mode="random", temporal_scale=1.0, use_color=False, one_validation_is_enough=False):
    with open(validation_file) as f:
        val_datasets = json.load(f)

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

            if "validation_videos" in val:
                validation_video = os.path.join(prefix, val["validation_videos"])
                video_tensor, _, _ = read_video(validation_video)
                video_tensor = video_tensor[:49]
            else:
                video_tensor = None

            if "input_image" in val:
                input_image = load_image(prefix+val["input_image"]).convert("RGB").resize((720, 480))
            elif "validation_videos" in val:
                image = (video_tensor[0] / 255).permute(2, 0, 1)
                input_image = F.to_pil_image(image)
            else:
                raise ValueError("Please provide input image")
            
            validation_prompt = val["validation_prompt"]
            vid_name = val.get("name", f"{ds_i}_{idx}")

            yield {
                "vid_name": vid_name,
                "is_real_video": is_real_video,
                "use_color": use_color,
                "input_image": input_image,
                "video_tensor": video_tensor,
                "tracking_map_path": tracking_map_path,
                "seg_map_path": seg_map_path,
                "validation_prompt": validation_prompt,
                "spatial_config": spatial_config,
                "temporal_config": temporal_config,
                "unalign_config": unalign_config,
            }

            if one_validation_is_enough:
                break

@torch.no_grad()
def infer(inputs, vis, pipe, visualize_only=False, kv_cache=False, num_inference_steps=10):
    frames = inputs["video_tensor"]
    vid_name = inputs["vid_name"]
    cond_maps_dict = vis.get_cond_maps(
                track_path = inputs["tracking_map_path"], 
                rle_path = inputs["seg_map_path"], 
                frames = frames,
                use_color=inputs["use_color"],
                frame_start=0, 
                is_real_video=inputs["is_real_video"],
                spatial_config=inputs["spatial_config"], 
                temporal_config=inputs["temporal_config"], 
                unalign_config=inputs["unalign_config"]
            ) 

    vis_cond_maps = cond_maps_dict[list(cond_maps_dict.keys())[0]].clone()

    for k in cond_maps_dict:
        cond_maps = cond_maps_dict[k].permute(0, 3, 1, 2).contiguous() 
        cond_maps_dict[k] = cond_maps / 127.5 - 1
    
    if visualize_only:
        video_generate = torch.empty((49, 0, 720, 3))
    else:
        video_generate = pipe(
                prompt=inputs["validation_prompt"],
                negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                image=inputs["input_image"],
                num_inference_steps=num_inference_steps,
                num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=6.0,
                generator=torch.Generator(device="cuda").manual_seed(0),
                cond_maps_dict=cond_maps_dict,
                height=480,
                width=720,
                output_type='pt',
                kv_cache=kv_cache
            ).frames[0].to('cpu').to(torch.float32)
        video_generate = video_generate.permute(0, 2, 3, 1)
        print("video_generate: ", video_generate.shape)
        
    frames = torch.empty((49, 0, 720, 3)) if frames is None else frames
    concatenated = torch.cat([frames / 255, vis_cond_maps / 255, video_generate], dim=1)
    return concatenated, vid_name
    


def rmdir_except_lora(removing_checkpoint: str, keep_file: str = "pytorch_lora_weights.safetensors"):
    """
    Remove everything inside `removing_checkpoint` except the LoRA weights file.

    Args:
        removing_checkpoint (str): Path to the checkpoint folder.
        keep_file (str): The filename to keep. Defaults to 'pytorch_lora_weights.safetensors'.
    """
    for item in os.listdir(removing_checkpoint):
        path = os.path.join(removing_checkpoint, item)

        # Skip the LoRA file
        if item == keep_file:
            continue

        # Remove file or directory
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    print(f"Cleaned {removing_checkpoint}, only kept {keep_file}.")
