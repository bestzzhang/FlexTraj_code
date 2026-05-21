import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora

def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    cache_setting: dict = {},
) -> torch.FloatTensor:
    text_seq_length = encoder_hidden_states.size(1)

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
    
    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), False
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # Apply RoPE if needed
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
        if not attn.is_cross_attention:
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if len(cache_setting) > 0:
        cache_mode = cache_setting["cache_mode"]
        cache_storage = cache_setting["cache_storage"]
        if cache_mode == "write":
            cache_storage[attn.cache_idx][0] = cond_key
            cache_storage[attn.cache_idx][1] = cond_value
            # print(f"cond_key {attn.cache_idx}: {cond_key.shape}")
        elif cache_mode == "read":
            cond_key = cache_storage[attn.cache_idx][0]
            cond_value = cache_storage[attn.cache_idx][1]

    if condition_latents is not None or cache_setting.get("cache_mode", "") == "read":
        # query = torch.cat([query, cond_query], dim=2)
        key = torch.cat([key, cond_key], dim=2)
        value = torch.cat([value, cond_value], dim=2)
    
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

    if condition_latents is not None:
        condition_latents = F.scaled_dot_product_attention(
            cond_query, cond_key, cond_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        condition_latents = condition_latents.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # linear proj
        condition_latents = attn.to_out[0](condition_latents)
        # dropout
        condition_latents = attn.to_out[1](condition_latents)
    
    with enable_lora((attn.to_out[0],), False):
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
    return hidden_states, encoder_hidden_states, condition_latents


'''
def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
) -> torch.FloatTensor:
    text_seq_length = encoder_hidden_states.size(1)

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
    
    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # Apply RoPE if needed
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
        if not attn.is_cross_attention:
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if condition_latents is not None:
        query = torch.cat([query, cond_query], dim=2)
        key = torch.cat([key, cond_key], dim=2)
        value = torch.cat([value, cond_value], dim=2)
    
    if True:
        attention_mask = torch.ones(
            query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
        )
        condition_n = cond_query.shape[2]
        attention_mask[-condition_n:, :-condition_n] = False

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

    if condition_latents is not None:
        cond_seq_length = condition_latents.shape[1]
        hidden_states, condition_latents = hidden_states.split(
            [hidden_states.size(1) - cond_seq_length, cond_seq_length], dim=1
        )
        # linear proj
        condition_latents = attn.to_out[0](condition_latents)
        # dropout
        condition_latents = attn.to_out[1](condition_latents)
    
    with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
    return hidden_states, encoder_hidden_states, condition_latents
'''


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    cache_setting={},
):
    # TODO: rewrite norm1 and norm2 to improve efficiency
    use_cond = condition_latents is not None
    text_seq_length = encoder_hidden_states.size(1)

    # norm & modulate
    norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
        hidden_states, encoder_hidden_states, temb
    )
    
    if use_cond:
        norm_condition_latents, _, cond_gate_msa, _ = self.norm1(
            condition_latents, encoder_hidden_states, cond_temb
        )

    # attention
    # attn_hidden_states, attn_encoder_hidden_states = self.attn1(
    #     hidden_states=norm_hidden_states,
    #     encoder_hidden_states=norm_encoder_hidden_states,
    #     image_rotary_emb=image_rotary_emb,
    # )
    result = attn_forward(
        self.attn1,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
        cache_setting=cache_setting,
    )
    attn_hidden_states, attn_encoder_hidden_states = result[:2]
    cond_attn_output = result[2] if use_cond else None

    hidden_states = hidden_states + gate_msa * attn_hidden_states
    encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
    if use_cond:
        condition_latents = condition_latents + cond_gate_msa * cond_attn_output

    # norm & modulate
    norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
        hidden_states, encoder_hidden_states, temb
    )
    
    if use_cond:
        norm_condition_latents, _, cond_gate_ff, _ = self.norm2(
            condition_latents, encoder_hidden_states, cond_temb
        )

    # feed-forward
    norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
    concate_length = norm_hidden_states.shape[1]
    if use_cond:
        norm_hidden_states = torch.cat([norm_hidden_states, norm_condition_latents], dim=1)
    ff_output = self.ff(norm_hidden_states)

    hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:concate_length]
    encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
    if use_cond:
        condition_latents = condition_latents + cond_gate_ff * ff_output[:, concate_length:]

    return hidden_states, encoder_hidden_states, condition_latents
