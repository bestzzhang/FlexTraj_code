from peft import LoraConfig, inject_adapter_in_model
from diffsynth import load_state_dict

def add_lora_to_model(model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


def mapping_lora_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A.weight" in key or "lora_B.weight" in key:
            new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
            new_state_dict[new_key] = value
        elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
            new_state_dict[key] = value
        elif "fuse_linear" in key:
            new_state_dict[key] = value
    return new_state_dict



def load_lora(pipe, lora_base_model="dit", lora_target_modules="q,k,v,o", lora_rank=128, lora_checkpoint=None):
    # Add LoRA to the base models
    if lora_base_model is not None:
        model = add_lora_to_model(
            getattr(pipe, lora_base_model),
            target_modules=lora_target_modules.split(","),
            lora_rank=lora_rank,
            upcast_dtype=pipe.torch_dtype,
        )
        if lora_checkpoint is not None:
            state_dict = load_state_dict(lora_checkpoint)
            state_dict = mapping_lora_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)
            print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
        setattr(pipe, lora_base_model, model)
