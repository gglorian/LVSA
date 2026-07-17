"""Tests for the NVIDIA Cosmos 3.0 LVSA geometry and action-tail split helper.

On CPU we cover:
  - The cosmos GEOMETRY fed to the (reused) ``HunyuanLVSAState.get_metadata`` —
    Cosmos3-Nano's native horizon (48 latent frames, P=920 @720p): the dense
    regime at 1x and genuine sparsity above it. This is the path the backend
    seam-swap dispatch relies on, validated at cosmos numbers (the hunyuan
    tests use 33/1560).
  - ``_cosmos_split_attention``, the pure-tensor core of the action-V2V port
    (mirror of ``lvsa/cosmos3.py``'s gen path), proven against a dense
    full-attention reference.
"""
from __future__ import annotations

import os

import pytest
import torch

from lvsa_vllm_omni.config import LVSAConfig

# Cosmos3-Nano @720p: 189 frames -> 48 latent frames; P = ceil(720/32)*ceil(1280/32)
COSMOS3_REF_LAT = 48
COSMOS3_P_720P = 920


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    yield


# ── Cosmos geometry through the reused state (the dispatch's metadata path) ───


class TestCosmos3Geometry:
    def test_dense_regime_at_native_horizon(self):
        """T_lat == reference (48) → kfi=1, every frame global (dense LVSA path)."""
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        s = HunyuanLVSAState(LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT))
        m = s.get_metadata(COSMOS3_REF_LAT, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert m.num_patches == COSMOS3_P_720P
        assert m.total_latent_frames == COSMOS3_REF_LAT
        assert m.key_frame_interval == 1
        assert len(m.global_indices) == COSMOS3_REF_LAT   # all frames global → dense

    def test_sparse_engages_above_horizon(self):
        """T_lat = 2x reference (96) → fewer than all frames are global anchors."""
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        s = HunyuanLVSAState(LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT))
        T_lat = 2 * COSMOS3_REF_LAT
        m = s.get_metadata(T_lat, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert m.key_frame_interval > 1
        assert 0 < len(m.global_indices) < T_lat          # genuinely sparse

    def test_metadata_cached_when_unchanged(self):
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        s = HunyuanLVSAState(LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT))
        m1 = s.get_metadata(96, COSMOS3_P_720P, 0, torch.device("cpu"))
        m2 = s.get_metadata(96, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert m1 is m2

    def test_condition_frames_forced_global_in_sparse_regime(self):
        """V2V cond frames land in the global set even when sparsity would skip
        them. T_lat=100 over ref=48 -> kfi>1; frame 1 is not a periodic anchor,
        so without condition_latent_frames it is windowed, reproducing the
        cond-frame-dropout drift bug."""
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        base = HunyuanLVSAState(LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT))
        mb = base.get_metadata(100, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert 1 not in mb.global_set  # baseline: frame 1 dropped

        s = HunyuanLVSAState(LVSAConfig(
            reference_latent_frames=COSMOS3_REF_LAT,
            condition_latent_frames=[0, 1],
        ))
        m = s.get_metadata(100, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert 0 in m.global_set
        assert 1 in m.global_set   # forced global by condition_latent_frames

    def test_condition_frames_forced_global_on_BACKEND_path(self):
        """Same guarantee, but through ``LVSAAttentionImpl._build_lvsa_metadata`` —
        the metadata path the Cosmos backend seam-swap actually dispatches through
        (``cosmos3_backend`` -> ``_lvsa_cosmos_dual_stream``), which is now the sole
        Cosmos plugin path. The hook path already wired condition_latent_frames; the
        backend path did not, so V2V cond frames silently dropped there. Guards that fix."""
        from lvsa_vllm_omni.attention_impl import LVSAAttentionImpl

        base = LVSAAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.125)
        base.config = LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT)
        mb = base._build_lvsa_metadata(100, COSMOS3_P_720P, 0)
        assert 1 not in mb.global_set  # baseline: frame 1 dropped in the sparse regime

        impl = LVSAAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.125)
        impl.config = LVSAConfig(
            reference_latent_frames=COSMOS3_REF_LAT,
            condition_latent_frames=[0, 1],
        )
        m = impl._build_lvsa_metadata(100, COSMOS3_P_720P, 0)
        assert 0 in m.global_set
        assert 1 in m.global_set   # forced global on the backend path (the fix)


# ── Action-tail split helper (CPU numeric proof) ──────────────────────────────


def _dense_full_attention(q, k, v):
    """Reference dense full attention ``[B, Sq, H, D]`` vs ``[B, Skv, Hkv, D]``
    with native GQA broadcasting. Used as the ground truth for the split."""
    import torch.nn.functional as F

    enable_gqa = k.shape[2] != q.shape[2]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=enable_gqa,
    )
    return out.transpose(1, 2)


class TestActionTailSplit:
    """``_cosmos_split_attention`` is the pure-tensor core of the action-V2V
    port (mirror of ``lvsa/cosmos3.py``'s gen path). At a 1x horizon the LVSA
    window covers every frame (kfi=1, all frames global), so the windowed video
    path is full attention. The whole gen stream (video + action tail) plus und
    must then match a single dense full-attention reference, row-for-row —
    exactly like ``tests/test_cosmos3_processor.py::
    test_lvsa_processor_matches_dense_with_action_tokens``.
    """

    def _build_window_all_metadata(self, T_lat, P):
        from lvsa.sparse_attention import LVSAMetadata
        # reference_frames=T_lat -> 1x horizon -> kfi=1 -> every frame global.
        return LVSAMetadata.build(
            total_latent_frames=T_lat, num_patches=P, window_size=4,
            n_first_frames=4, key_frame_interval=1, rank=0, world=1,
            expand_window=True, reference_frames=T_lat, sparsity_scale=1.0,
        )

    def _run_case(self, T_lat, P, s_extra, H, Hkv, D):
        import torch

        from lvsa.sparse_attention import lvsa_sdpa
        from lvsa_vllm_omni.cosmos_dual_stream import (
            _cosmos_split_attention,
            _dense_sdpa_gqa,
        )

        torch.manual_seed(0)
        video_len = T_lat * P
        S_gen = video_len + s_extra
        S_und = 5
        q = torch.randn(1, S_gen, H, D)
        k = torch.randn(1, S_gen, Hkv, D)
        v = torch.randn(1, S_gen, Hkv, D)
        k_und = torch.randn(1, S_und, Hkv, D)
        v_und = torch.randn(1, S_und, Hkv, D)
        md = self._build_window_all_metadata(T_lat, P)

        out = _cosmos_split_attention(
            q, k, v, k_und, v_und, video_len, md, P,
            sparse_fn=lambda qv, kv, vv, kg, vg, m: lvsa_sdpa(qv, kv, vv, kg, vg, m),
            dense_fn=_dense_sdpa_gqa,
        )
        # Reference: dense full attention over cat([gen(all), und]) for ALL gen
        # queries (video + tail) — what the native cosmos dense path computes.
        k_all = torch.cat([k, k_und], dim=1)
        v_all = torch.cat([v, v_und], dim=1)
        ref = _dense_full_attention(q, k_all, v_all)
        assert out.shape == (1, S_gen, H, D)
        max_diff = (out - ref).abs().max().item()
        assert torch.allclose(out, ref, atol=1e-5), (
            f"split must == dense full at window-all (max diff {max_diff:.2e})"
        )
        return out, ref

    def test_split_matches_dense_with_action_tail_gqa(self):
        # GQA (Hkv < H) + an action tail.
        self._run_case(T_lat=6, P=2, s_extra=3, H=4, Hkv=2, D=16)

    def test_split_matches_dense_with_action_tail_mha(self):
        # No GQA (Hkv == H), tail present.
        self._run_case(T_lat=8, P=3, s_extra=5, H=4, Hkv=4, D=16)

    def test_split_no_tail_equals_plain_lvsa(self):
        """s_extra == 0: the split must collapse to a single LVSA call (no tail
        cat). Guards the no-regression requirement for clean T2V/I2V/clean-V2V:
        the helper output must equal a direct ``lvsa_sdpa`` over the video grid
        with und appended as global — byte-identical to the pre-port path."""
        import torch

        from lvsa.sparse_attention import build_global_kv, lvsa_sdpa
        from lvsa_vllm_omni.cosmos_dual_stream import (
            _cosmos_split_attention,
            _dense_sdpa_gqa,
        )

        torch.manual_seed(1)
        T_lat, P, H, Hkv, D = 6, 2, 4, 2, 16
        video_len = T_lat * P
        q = torch.randn(1, video_len, H, D)
        k = torch.randn(1, video_len, Hkv, D)
        v = torch.randn(1, video_len, Hkv, D)
        k_und = torch.randn(1, 5, Hkv, D)
        v_und = torch.randn(1, 5, Hkv, D)
        md = self._build_window_all_metadata(T_lat, P)

        out = _cosmos_split_attention(
            q, k, v, k_und, v_und, video_len, md, P,
            sparse_fn=lambda qv, kv, vv, kg, vg, m: lvsa_sdpa(qv, kv, vv, kg, vg, m),
            dense_fn=_dense_sdpa_gqa,
        )
        # Pre-port path: build_global_kv + und-as-global + one lvsa_sdpa call.
        kg, vg = build_global_kv(k, v, md.global_indices, P)
        kg = torch.cat([kg, k_und], dim=1)
        vg = torch.cat([vg, v_und], dim=1)
        ref = lvsa_sdpa(q, k, v, kg, vg, md)
        assert torch.equal(out, ref), "no-tail split must be byte-identical to plain LVSA"
