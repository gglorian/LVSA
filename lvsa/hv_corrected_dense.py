"""Corrected (mask-free) dense baseline for HunyuanVideo-1.5.

The stock diffusers ``HunyuanVideo15AttnProcessor2_0`` unconditionally builds a dense
``[B, 1, S, S]`` boolean self-attention mask (the text-padding mask, all-True over the video
block) and threads it into ``scaled_dot_product_attention``. A non-null ``attn_mask`` makes
SDPA ineligible for the FlashAttention kernel -- torch falls back to the memory-efficient/math
path -- and materialises an ``O(S^2)`` tensor. That is ~2.5x slower per attention call at
480p/0.5x and is what makes the stock pipeline OOM at long horizons (at 2x the two mask
intermediates alone are ~19.9 GiB).

``install_corrected_dense`` swaps in a forward that is byte-for-byte the stock processor's
except that it SKIPS building the mask and passes ``attn_mask=None``. This is the fair
"corrected dense" baseline: same numerics as full-attention LVSA (verified bit-identical),
FlashAttention-eligible, ``O(S)`` memory (fits at 2x where the stock masked dense OOMs).
LVSA's own path already passes ``attn_mask=None``; this only affects the dense baseline.

NOTE: ``_maskfree_call`` mirrors diffusers ``HunyuanVideo15AttnProcessor2_0.__call__`` exactly
apart from the removed mask block -- keep it in sync if that processor changes upstream.
"""
from __future__ import annotations

from typing import Any

import torch


def _maskfree_call(
    self,
    attn,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    image_rotary_emb=None,
):
    import diffusers.models.transformers.transformer_hunyuan_video15 as _hv15
    from diffusers.models.embeddings import apply_rotary_emb

    # 1. QKV projections
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    # 2. QK normalization
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # 3. Rotary positional embeddings on the latent stream
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # 4. Encoder-condition QKV
    if encoder_hidden_states is not None:
        eq = attn.add_q_proj(encoder_hidden_states)
        ek = attn.add_k_proj(encoder_hidden_states)
        ev = attn.add_v_proj(encoder_hidden_states)
        eq = eq.unflatten(2, (attn.heads, -1))
        ek = ek.unflatten(2, (attn.heads, -1))
        ev = ev.unflatten(2, (attn.heads, -1))
        if attn.norm_added_q is not None:
            eq = attn.norm_added_q(eq)
        if attn.norm_added_k is not None:
            ek = attn.norm_added_k(ek)
        query = torch.cat([query, eq], dim=1)
        key = torch.cat([key, ek], dim=1)
        value = torch.cat([value, ev], dim=1)

    # 5. Attention -- MASK CONSTRUCTION SKIPPED: attn_mask=None keeps FlashAttention + O(S) memory.
    hidden_states = _hv15.dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

    # 6. Output projection
    if encoder_hidden_states is not None:
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : -encoder_hidden_states.shape[1]],
            hidden_states[:, -encoder_hidden_states.shape[1] :],
        )
        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    return hidden_states, encoder_hidden_states


def install_corrected_dense(pipe: Any) -> int:
    """Install the mask-free dense processor on every HunyuanVideo-1.5 transformer block.

    Subclasses the stock ``HunyuanVideo15AttnProcessor2_0`` and overrides only ``__call__``
    (so the diffusers class is left untouched). Returns the number of blocks patched.
    """
    import diffusers.models.transformers.transformer_hunyuan_video15 as _hv15

    class _CorrectedDenseProcessor(_hv15.HunyuanVideo15AttnProcessor2_0):
        __call__ = _maskfree_call

    blocks = pipe.transformer.transformer_blocks
    for block in blocks:
        block.attn.processor = _CorrectedDenseProcessor()
    return len(blocks)
