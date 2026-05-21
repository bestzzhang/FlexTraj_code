from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import torch, os, math
from torch import nn
from PIL import Image
from tqdm import tqdm

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.pipelines import DiffusionPipeline   
from diffusers.models.modeling_utils import ModelMixin
from .block import block_forward
from diffusers.models.attention_processor import Attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def patch_forward(patch_embed, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
    r"""
    Args:
        text_embeds (`torch.Tensor`):
            Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
        image_embeds (`torch.Tensor`):
            Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
    """
    text_embeds = patch_embed.text_proj(text_embeds)

    batch_size, num_frames, channels, height, width = image_embeds.shape

    if patch_embed.patch_size_t is None:
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = patch_embed.proj(image_embeds)
        image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
    else:
        p = patch_embed.patch_size
        p_t = patch_embed.patch_size_t

        image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
        image_embeds = image_embeds.reshape(
            batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
        )
        image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
        image_embeds = patch_embed.proj(image_embeds)

    embeds = torch.cat(
        [text_embeds, image_embeds], dim=1
    ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

    if patch_embed.use_positional_embeddings or patch_embed.use_learned_positional_embeddings:
        if patch_embed.use_learned_positional_embeddings and (patch_embed.sample_width != width or patch_embed.sample_height != height):
            raise ValueError(
                "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
            )

        pre_time_compression_frames = (num_frames - 1) * patch_embed.temporal_compression_ratio + 1

        if (
            patch_embed.sample_height != height
            or patch_embed.sample_width != width
            or patch_embed.sample_frames != pre_time_compression_frames
        ):
            pos_embedding = patch_embed._get_positional_embeddings(
                height, width, pre_time_compression_frames, device=embeds.device
            )
        else:
            pos_embedding = patch_embed.pos_embedding

        pos_embedding = pos_embedding.to(dtype=embeds.dtype)
        ########## modified start #####################
        # embeds = embeds + pos_embedding
        return embeds, pos_embedding
        ########## modified end #####################

    return embeds


class CogVideoXTransformerFlex(CogVideoXTransformer3DModel, ModelMixin):
    def __init__(
        self,
        num_inputs: int = 3,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            **kwargs
        )

        # Initialize weights of combine_linears to zero
        inner_dim = num_attention_heads * attention_head_dim
        self.fuse_linear = nn.Linear(inner_dim, inner_dim)
        self.fuse_linear = self.fuse_linear.to_empty(device="cuda")

        self._init_linear(inner_dim)

    def _init_linear(self, inner_dim: int):
        with torch.no_grad():
            self.fuse_linear.weight.zero_()
            if self.fuse_linear.bias is not None:
                self.fuse_linear.bias.zero_()

    def patchify(self, cond_maps, prompt_embed, text_seq_length):
        # Process cond maps
        cond_maps_hidden_states, pos_embedding = patch_forward(self.patch_embed, prompt_embed, cond_maps)
        cond_maps_hidden_states += pos_embedding
        cond_maps_hidden_states = self.embedding_dropout(cond_maps_hidden_states)
        cond_maps = cond_maps_hidden_states[:, text_seq_length:]
        return cond_maps

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        multi_cond_states: List[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        cache_setting: dict = {}
    ):
        use_condition = multi_cond_states is not None
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        
        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if use_condition:
            cond_temb = self.time_proj(torch.zeros_like(timestep)).to(dtype=hidden_states.dtype)
            cond_temb = self.time_embedding(cond_temb, timestep_cond)
        
        # 2. Patch embedding
        prompt_embed = encoder_hidden_states.clone()
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        
        if use_condition:
            # prompt_embed has no usage but is an input
            fused_cond_states = [self.patchify(cond_states, prompt_embed, text_seq_length) for cond_states in multi_cond_states]
            fused_cond_tokens = fused_cond_states[0] + self.fuse_linear(fused_cond_states[1])
            del prompt_embed
                    
        # Process transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states, fused_cond_tokens = torch.utils.checkpoint.checkpoint(
                    block_forward,
                    self=block,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    condition_latents=fused_cond_tokens if use_condition else None,
                    temb=emb,
                    cond_temb=cond_temb if use_condition else None,
                    cond_rotary_emb=image_rotary_emb if use_condition else None,
                    image_rotary_emb=image_rotary_emb,
                    cache_setting=cache_setting,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states, fused_cond_tokens = block_forward(
                    block,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    condition_latents=fused_cond_tokens if use_condition else None,
                    temb=emb,
                    cond_temb=cond_temb if use_condition else None,
                    cond_rotary_emb=image_rotary_emb if use_condition else None,
                    image_rotary_emb=image_rotary_emb,
                    cache_setting=cache_setting
                )
        

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

class CogVideoXFlexControlPipeline(CogVideoXImageToVideoPipeline, DiffusionPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformerFlex,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        if not isinstance(self.transformer, CogVideoXTransformerFlex):
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelFull")
            
        # self.transformer = torch.compile(self.transformer)

    @staticmethod
    def cond_to_latents(cond_maps, vae, latent_shape, flag_zero_image=False, device=None):
        VAE_SCALING_FACTOR = vae.config.scaling_factor
        weight_dtype = vae.dtype

        # With `enable_sequential_cpu_offload`, vae.device is `meta` (params are on
        # meta and are copied to the execution device on demand by accelerate hooks).
        # Sending inputs to `vae.device` would produce a meta tensor with no data,
        # which then crashes the VAE's pre_forward hook with
        # "Cannot copy out of meta tensor; no data!".
        if device is None:
            device = vae.device if vae.device.type != "meta" else torch.device("cuda")
        cond_maps = cond_maps.to(device, non_blocking=True)
        cond_image = cond_maps[:,:1].clone()

        cond_maps = cond_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        cond_latent_dist = vae.encode(cond_maps).latent_dist
        
        cond_maps = cond_latent_dist.sample() * VAE_SCALING_FACTOR
        cond_maps = cond_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        cond_maps = cond_maps.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

        if not flag_zero_image:
            cond_image = cond_image.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            cond_image_latent_dist = vae.encode(cond_image).latent_dist
            cond_image_latent_dist = cond_image_latent_dist.sample() * VAE_SCALING_FACTOR
            cond_image_latent_dist = cond_image_latent_dist.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            cond_image_latent_dist = cond_image_latent_dist.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            padding_shape = (latent_shape[0], latent_shape[1] - 1, *latent_shape[2:])
            cond_latent_padding = cond_image_latent_dist.new_zeros(padding_shape)

            cond_image_latents = torch.cat([cond_image_latent_dist, cond_latent_padding], dim=1)
        else:
            cond_image_latents =  torch.zeros_like(cond_maps)

        cond_latents = torch.cat([cond_maps, cond_image_latents], dim=2)
        return cond_latents

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        cond_maps_dict: Optional[Dict[str, torch.Tensor]] = None,
        kv_cache=False,
        enough_memory=True
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        # Most of the implementation remains the same as the parent class

        # 1. Check inputs and set default values
        self.check_inputs(
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance and enough_memory:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents       
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        ) 
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            self.transformer.config.in_channels // 2,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        del image
        
        # 5.5. Prepare multi_cond_states
        multi_cond_states = []
        for cond in cond_maps_dict.keys():
            cond_states = self.cond_to_latents(cond_maps_dict[cond].unsqueeze(0).to(dtype=self.vae.dtype), 
                                               self.vae, 
                                               latents.shape,
                                               device=device)
            if do_classifier_free_guidance and enough_memory:
                cond_states = torch.cat([cond_states] * 2)
            multi_cond_states.append(cond_states)
            
        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if kv_cache:
            attn_counter = 0
            for module in self.transformer.modules():
                if isinstance(module, Attention):
                    setattr(module, "cache_idx", attn_counter)
                    attn_counter += 1
            kv_cond = [[None, None] for _ in range(attn_counter)]
            kv_uncond = [[None, None] for _ in range(attn_counter)]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                latent_model_input = torch.cat([latents] * 2) if (do_classifier_free_guidance and enough_memory) else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if (do_classifier_free_guidance and enough_memory) else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                del latent_image_input

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if kv_cache:
                    mode = "write" if i == 0 else "read"
                else:
                    mode = ""

                use_cond = not (kv_cache) or mode == "write"

                # Predict noise
                self.transformer.to(dtype=latent_model_input.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    multi_cond_states=multi_cond_states if use_cond else None,
                    cache_setting={"cache_mode": mode, "cache_storage": kv_cond} if kv_cache else {},
                    return_dict=False
                )[0]
                noise_pred = noise_pred.float()

                if do_classifier_free_guidance:
                    if enough_memory:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    else:
                        noise_pred_text = noise_pred
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=negative_prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs,
                            multi_cond_states=multi_cond_states if use_cond else None,
                            cache_setting={"cache_mode": mode, "cache_storage": kv_uncond} if kv_cache else {},
                            return_dict=False
                        )[0]

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    del noise_pred_uncond, noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                del noise_pred
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Post-processing
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
