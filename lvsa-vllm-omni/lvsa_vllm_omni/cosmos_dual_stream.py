"""Shared pure-tensor dual-stream helpers for the Cosmos 3.0 gen pathway.

Used by BOTH the cosmos hook (legacy monkey-patch path) and the LVSA attention
backend. Pure tensor ops (no model/projection state) → CPU-unit-testable.
"""
import torch
import torch.nn.functional as F

from .global_kv import build_global_kv


def _dense_sdpa_gqa(q, k, v):
    """Dense full attention over ``[B, S_q, H, D]`` q against ``[B, S_kv, Hkv, D]``
    k/v, with native grouped-query broadcasting (``enable_gqa``). Returns
    ``[B, S_q, H, D]``.

    This mirrors the standalone ``lvsa/cosmos3.py`` tail term, which calls
    diffusers' ``dispatch_attention_fn(..., is_causal=False, enable_gqa=True)``.
    The plugin does not have that diffusers symbol available at hook-install
    time (Cosmos3 lives in vllm-omni, not diffusers), so we use torch SDPA with
    ``enable_gqa`` directly — numerically identical full attention, and it keeps
    K/V at ``Hkv`` (no repeat-KV), matching the rest of the cosmos hook's native
    GQA. The action/sound tail is tiny, so the per-call SDPA cost is negligible.
    """
    enable_gqa = k.shape[2] != q.shape[2]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=None, dropout_p=0.0,
        is_causal=False, enable_gqa=enable_gqa,
    )
    return out.transpose(1, 2)


def _cosmos_split_attention(
    q, k, v, k_und, v_und, video_len, metadata, P,
    *, sparse_fn, dense_fn,
):
    """Action-tail split for the cosmos gen pathway (mirror of the standalone
    ``Cosmos3LVSAAttnProcessor.__call__`` gen path).

    Pure tensor helper (no model/projection state) so it is CPU-unit-testable.

    Parameters
    ----------
    q, k, v : ``[B, S_gen, H/Hkv, D]`` — already QK-normed + RoPE'd gen Q/K/V.
    k_und, v_und : ``[B, S_und, Hkv, D]`` — always-global understanding K/V.
    video_len : leading ``T_lat * P`` video tokens (the frame grid).
    metadata : ``LVSAMetadata`` built for ``(T_lat, P)``.
    P : true patches-per-frame.
    sparse_fn(q_v, k_v, v_v, k_global, v_global, metadata) -> out_v
        LVSA over the video queries (e.g. ``lvsa_sdpa`` or the FlashInfer runner).
    dense_fn(q_ex, k_all, v_all) -> out_ex
        Dense full attention for the trailing action/sound queries.

    Returns ``[B, S_gen, H, D]``. ``s_extra == 0`` -> identical to the no-tail
    path (only ``sparse_fn`` runs; the cat is a no-op).
    """
    # Video queries (leading frame grid): LVSA windowed.
    q_v = q[:, :video_len]
    k_v = k[:, :video_len]
    v_v = v[:, :video_len]
    k_global, v_global = build_global_kv(k_v, v_v, metadata.global_indices, P)
    # und is always-global -> appended after the gen anchors.
    k_global = torch.cat([k_global, k_und], dim=1)
    v_global = torch.cat([v_global, v_und], dim=1)
    s_extra = q.shape[1] - video_len
    if s_extra:
        # Action/sound tail: also always-global for the video queries.
        k_global = torch.cat([k_global, k[:, video_len:]], dim=1)
        v_global = torch.cat([v_global, v[:, video_len:]], dim=1)
    out_v = sparse_fn(q_v, k_v, v_v, k_global, v_global, metadata)
    if not s_extra:
        return out_v
    # Tail queries (few): dense full attention over the whole gen stream + und,
    # mirroring the dense model for those rows.
    q_ex = q[:, video_len:]
    k_all = torch.cat([k, k_und], dim=1)
    v_all = torch.cat([v, v_und], dim=1)
    out_ex = dense_fn(q_ex, k_all, v_all)
    return torch.cat([out_v, out_ex], dim=1)
