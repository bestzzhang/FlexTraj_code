
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from safetensors.torch import load_file
from accelerate.utils import set_module_tensor_to_device
import torch


def _pass_kwarg_through_offload_hook(module, key):
    """Tell accelerate's AlignDevicesHook(s) on `module` to leave kwarg `key` as-is.

    `enable_sequential_cpu_offload` attaches an `AlignDevicesHook` whose
    `pre_forward` calls `send_to_device(kwargs, ...)`. For dict / list kwargs
    that recursively rebuilds a fresh container, which breaks anything passed
    by reference (e.g. our `cache_setting={"cache_storage": kv_cond}` used by
    the kv-cache flow: writes from step 0 would land in a throwaway copy and
    step 1's reads would see `None`). Setting `skip_keys` on the hook makes
    `send_to_device` pass that top-level kwarg through untouched.
    """
    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return

    def _apply(h):
        if hasattr(h, "hooks"):  # SequentialHook
            for sub in h.hooks:
                _apply(sub)
            return
        if not hasattr(h, "skip_keys"):
            return
        existing = h.skip_keys
        if existing is None:
            existing = []
        elif isinstance(existing, str):
            existing = [existing]
        else:
            existing = list(existing)
        if key not in existing:
            existing.append(key)
        h.skip_keys = existing

    _apply(hook)



def load_pipeline_kv(model_path="THUDM/CogVideoX-5b-I2V", checkpoint_path=None, cond_mode="full", device="cuda", dtype=torch.bfloat16): 
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
    tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype)
    num_inputs = len(cond_mode.split("+"))

    from models.cogvideox_flextraj import CogVideoXFlexControlPipeline, CogVideoXTransformerFlex
    transformer = CogVideoXTransformerFlex.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype, num_inputs=num_inputs)
    pipe = CogVideoXFlexControlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    if checkpoint_path:
        state_dict = load_file(checkpoint_path)
        set_module_tensor_to_device(transformer.fuse_linear, "weight", "cpu", value=state_dict["transformer.fuse_linear.weight"])
        set_module_tensor_to_device(transformer.fuse_linear, "bias", "cpu", value=state_dict["transformer.fuse_linear.bias"])
        for key in ['transformer.fuse_linear.bias', 'transformer.fuse_linear.weight']:
            state_dict.pop(key)
        pipe.load_lora_weights(state_dict)
        print("successfully load lora")

    # pipe.to(device, dtype=dtype)
    pipe.enable_sequential_cpu_offload()
    # Preserve our `cache_setting` kwarg by-reference across the transformer's
    # accelerate hook, otherwise kv_cache writes are lost (see helper docstring).
    _pass_kwarg_through_offload_hook(pipe.transformer, "cache_setting")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe

