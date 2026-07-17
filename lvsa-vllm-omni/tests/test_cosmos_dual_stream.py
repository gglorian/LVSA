import types

import torch
import torch.nn.functional as F
from lvsa_vllm_omni.attention_impl import LVSAAttentionImpl
from lvsa_vllm_omni.cosmos_dual_stream import _cosmos_split_attention, _dense_sdpa_gqa


def _dense_full(q, k, v):
    """Reference: plain dense full attention, GQA-aware. [B,Sq,H,D]."""
    enable_gqa = k.shape[2] != q.shape[2]
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=False, enable_gqa=enable_gqa,
    )
    return o.transpose(1, 2)


def test_dense_sdpa_gqa_matches_reference():
    torch.manual_seed(0)
    q = torch.randn(1, 6, 4, 8)
    k = torch.randn(1, 9, 2, 8)   # GQA: Hkv=2 < H=4
    v = torch.randn(1, 9, 2, 8)
    assert torch.allclose(_dense_sdpa_gqa(q, k, v), _dense_full(q, k, v), atol=1e-5)


def test_split_no_tail_equals_plain_sparse():
    """s_extra == 0 -> only sparse_fn runs, the cat is a no-op."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2, 8); k = torch.randn(1, 4, 2, 8); v = torch.randn(1, 4, 2, 8)
    k_und = torch.randn(1, 3, 2, 8); v_und = torch.randn(1, 3, 2, 8)
    seen = {}

    def sparse_fn(q_v, k_v, v_v, k_g, v_g, md):
        seen["called"] = True
        return q_v * 0.0  # sentinel; value irrelevant, we assert routing

    def dense_fn(q_ex, k_all, v_all):
        raise AssertionError("dense_fn must NOT run when s_extra == 0")

    out = _cosmos_split_attention(q, k, v, k_und, v_und, video_len=4,
                                  metadata=_FakeMeta(), P=2,
                                  sparse_fn=sparse_fn, dense_fn=dense_fn)
    assert seen.get("called") and out.shape == (1, 4, 2, 8)


class _FakeMeta:
    global_indices = [0]


def test_backend_joint_stream_equals_dense_at_1x(monkeypatch):
    """At 1x horizon (window covers every frame) LVSA over the gen grid == dense,
    and the und (joint) stream is always-global. So backend(gen, joint=und) must
    equal dense full attention of gen queries over cat([und, gen])."""
    import os
    for k in list(os.environ):
        if k.startswith("LVSA_"):
            monkeypatch.delenv(k, raising=False)
    T_lat, P, H, D = 4, 2, 2, 8          # S_gen = 8; ref == T_lat -> 1x -> kfi=1 -> dense regime
    monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_REFERENCE_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
    monkeypatch.setenv("LVSA_SCHEDULE_START", "0"); monkeypatch.setenv("LVSA_SCHEDULE_END", "0")

    torch.manual_seed(0)
    impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D ** 0.5))
    q = torch.randn(1, T_lat * P, H, D)
    k = torch.randn(1, T_lat * P, H, D)
    v = torch.randn(1, T_lat * P, H, D)
    k_und = torch.randn(1, 5, H, D)
    v_und = torch.randn(1, 5, H, D)
    meta = types.SimpleNamespace(joint_key=k_und, joint_value=v_und, joint_query=None,
                                 joint_strategy="front")

    out = impl.forward_cuda(q, k, v, meta)

    ref = _dense_full(q, torch.cat([k_und, k], dim=1), torch.cat([v_und, v], dim=1))
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-4), f"max diff {(out - ref).abs().max():.2e}"


def test_backend_concat_stream_equals_dense_at_1x(monkeypatch):
    """Single-GPU _forward_local prepends und into key: k_all = cat([k_und, k_gen]).
    A cosmos_gen-marked impl must split that and match dense full-attn over cat([und,gen])
    at 1x — same result as the joint_* path, just a different und source.

    NOTE: this is a *sanity* check only. At 1x, sparse == dense, so this test would
    still pass even if the concat split were wrong (or never fired at all, falling
    through to the pre-existing dense cross-attn guard) — see
    ``test_backend_concat_form_equals_joint_form_at_sparse_horizon`` below for the
    non-degenerate version that actually proves the split path fires and is correct.
    """
    import os
    for k in list(os.environ):
        if k.startswith("LVSA_"):
            monkeypatch.delenv(k, raising=False)
    T_lat, P, H, D = 4, 2, 2, 8
    monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_REFERENCE_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
    torch.manual_seed(0)
    impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D ** 0.5),
                             cosmos_gen=True)
    q = torch.randn(1, T_lat * P, H, D)
    k_gen = torch.randn(1, T_lat * P, H, D); v_gen = torch.randn(1, T_lat * P, H, D)
    k_und = torch.randn(1, 5, H, D); v_und = torch.randn(1, 5, H, D)
    k_all = torch.cat([k_und, k_gen], dim=1); v_all = torch.cat([v_und, v_gen], dim=1)
    out = impl.forward_cuda(q, k_all, v_all)                 # no attn_metadata → concat path
    ref = _dense_full(q, torch.cat([k_und, k_gen], dim=1), torch.cat([v_und, v_gen], dim=1))
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-4), f"max diff {(out - ref).abs().max():.2e}"


def test_backend_concat_form_equals_joint_form_at_sparse_horizon(monkeypatch):
    """Non-degenerate concat-split correctness, at a genuinely SPARSE horizon
    (T_lat=24, ref=6 -> 4x -> kfi>1, so sparse != dense — see the negative
    assertion at the end).

    The joint-form (attn_metadata.joint_key/joint_value — the Ulysses/multi-GPU
    shape) and the concat-form (und prepended into `key` — the single-GPU
    _forward_local shape) both carry the SAME q/k_gen/v_gen/k_und/v_und, so a
    correct concat split must recover byte-identical inputs to
    ``_lvsa_cosmos_dual_stream`` and therefore produce IDENTICAL output. If the
    split mis-slices (wrong `s_und`, wrong order) or falls through to the dense
    cross-attn guard instead of the sparse split, the two forms would disagree
    here — unlike at 1x, where sparse == dense and everything trivially matches.
    """
    import os
    for k in list(os.environ):
        if k.startswith("LVSA_"):
            monkeypatch.delenv(k, raising=False)
    T_lat, P, H, D = 24, 2, 2, 8   # ref=6 -> 4x horizon -> kfi>1 -> genuinely sparse
    monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_REFERENCE_LATENT_FRAMES", "6")
    monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))

    torch.manual_seed(0)
    impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D ** 0.5),
                             cosmos_gen=True)
    q = torch.randn(1, T_lat * P, H, D)
    k_gen = torch.randn(1, T_lat * P, H, D); v_gen = torch.randn(1, T_lat * P, H, D)
    k_und = torch.randn(1, 5, H, D); v_und = torch.randn(1, 5, H, D)

    joint_meta = types.SimpleNamespace(joint_key=k_und, joint_value=v_und,
                                       joint_query=None, joint_strategy="front")
    out_joint = impl.forward_cuda(q, k_gen, v_gen, joint_meta)

    k_all = torch.cat([k_und, k_gen], dim=1); v_all = torch.cat([v_und, v_gen], dim=1)
    out_concat = impl.forward_cuda(q, k_all, v_all, None)

    assert out_joint.shape == q.shape == out_concat.shape
    assert torch.allclose(out_joint, out_concat, atol=1e-5), (
        f"joint-form vs concat-form max diff "
        f"{(out_joint - out_concat).abs().max():.2e}"
    )

    # Confirm this horizon really is sparse (not accidentally degenerate to
    # dense) — otherwise the equivalence above wouldn't discriminate a broken
    # split at all (a wrong split could still coincidentally match a dense
    # fallback, same failure mode this test replaces).
    ref_dense = _dense_full(q, k_all, v_all)
    assert not torch.allclose(out_joint, ref_dense, atol=1e-3), (
        "expected sparse output to differ from dense at a 4x horizon"
    )


def test_unmarked_impl_longer_key_stays_dense_crossattn(monkeypatch):
    """Gate correctness, non-degenerate. At the SAME sparse horizon (4x) as
    above, an UNMARKED impl (cosmos_gen=False, i.e. Wan/HunyuanVideo) given a
    longer key is a genuine cross-attention -> must take the dense fallback,
    NOT the cosmos concat split. If the ``self.cosmos_gen`` gate were removed,
    an unmarked impl would split -> sparse here too, and sparse != dense at
    4x (verified in the test above), so this test would then fail.
    """
    import os
    for k in list(os.environ):
        if k.startswith("LVSA_"):
            monkeypatch.delenv(k, raising=False)
    T_lat, P, H, D = 24, 2, 2, 8   # same sparse (4x) horizon as the split test above
    monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_REFERENCE_LATENT_FRAMES", "6")
    monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))

    torch.manual_seed(0)
    impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D ** 0.5))
    # cosmos_gen defaults False
    q = torch.randn(1, T_lat * P, H, D)
    k_gen = torch.randn(1, T_lat * P, H, D); v_gen = torch.randn(1, T_lat * P, H, D)
    k_und = torch.randn(1, 5, H, D); v_und = torch.randn(1, 5, H, D)
    k_all = torch.cat([k_und, k_gen], dim=1); v_all = torch.cat([v_und, v_gen], dim=1)

    out = impl.forward_cuda(q, k_all, v_all)   # no attn_metadata -> concat-shaped key
    ref = _dense_full(q, k_all, v_all)         # plain dense cross-attn
    assert torch.allclose(out, ref, atol=1e-4), f"max diff {(out - ref).abs().max():.2e}"


def test_cosmos_flashinfer_degrades_to_sdpa(monkeypatch):
    """With backend=flashinfer but FlashInfer unavailable, the Cosmos path must
    degrade to SDPA (no AssertionError), producing the SDPA result."""
    import os
    for k in list(os.environ):
        if k.startswith("LVSA_"):
            monkeypatch.delenv(k, raising=False)
    T_lat, P, H, D = 4, 2, 2, 8
    monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_REFERENCE_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
    monkeypatch.setenv("LVSA_BACKEND", "flashinfer")
    import lvsa_vllm_omni.flashinfer_runner as fr
    monkeypatch.setattr(fr, "FLASHINFER_AVAILABLE", False)   # force the degrade branch
    torch.manual_seed(0)
    impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D ** 0.5),
                             cosmos_gen=True)
    q = torch.randn(1, T_lat * P, H, D); k = torch.randn(1, T_lat * P, H, D)
    v = torch.randn(1, T_lat * P, H, D)
    k_und = torch.randn(1, 5, H, D); v_und = torch.randn(1, 5, H, D)
    meta = types.SimpleNamespace(joint_key=k_und, joint_value=v_und, joint_query=None,
                                 joint_strategy="front")
    out = impl.forward_cuda(q, k, v, meta)                   # must NOT raise
    ref = _dense_full(q, torch.cat([k_und, k], dim=1), torch.cat([v_und, v], dim=1))
    assert torch.allclose(out, ref, atol=1e-4)              # SDPA degrade == dense at 1x


def test_ulysses_prepended_und_not_double_counted(monkeypatch):
    """REGRESSION (the SP motion bug): under Ulysses the framework FRONT-prepends the
    und into ``key`` (key = [und ++ gen], longer than query) AND leaves joint_key in
    metadata. The backend must split the prepended und out of ``key`` ONCE — not treat
    [und ++ gen] as the gen grid while ALSO re-adding joint_key, which misaligns the
    frame grid by S_und and double-counts the und (→ the uniform motion damping).
    At 1x horizon, correct output == dense full-attn over cat([und, gen])."""
    import os
    for k in list(os.environ):
        if k.startswith("LVSA_"):
            monkeypatch.delenv(k, raising=False)
    T_lat, P, H, D, S_und = 4, 2, 2, 8, 5
    monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
    monkeypatch.setenv("LVSA_REFERENCE_LATENT_FRAMES", str(T_lat))   # ref == T_lat -> 1x
    monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
    torch.manual_seed(0)
    impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D ** 0.5),
                             cosmos_gen=True)
    q = torch.randn(1, T_lat * P, H, D)                             # gen queries
    gen_k = torch.randn(1, T_lat * P, H, D); gen_v = torch.randn(1, T_lat * P, H, D)
    und_k = torch.randn(1, S_und, H, D); und_v = torch.randn(1, S_und, H, D)
    key = torch.cat([und_k, gen_k], dim=1)                          # und PREPENDED into key
    val = torch.cat([und_v, gen_v], dim=1)
    meta = types.SimpleNamespace(joint_key=und_k, joint_value=und_v,  # und ALSO in metadata
                                 joint_query=None, joint_strategy="front")
    out = impl.forward_cuda(q, key, val, meta)          # key longer than q AND joint present
    ref = _dense_full(q, torch.cat([und_k, gen_k], dim=1), torch.cat([und_v, gen_v], dim=1))
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-4), f"max diff {(out - ref).abs().max():.2e}"
