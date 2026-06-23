import pytest


def test_cosmos3_geometry_720p_189f():
    from lvsa.cosmos3 import (cosmos3_latent_frames, cosmos3_patches_per_frame,
                              COSMOS3_REFERENCE_LATENT_FRAMES)
    # VAE temporal factor 4: (189-1)//4 + 1 = 48
    assert cosmos3_latent_frames(189) == 48
    assert cosmos3_latent_frames(1) == 1
    # VAE spatial 16, latent patch 2 -> P = ceil(H/32)*ceil(W/32); 720x1280 -> 23*40
    assert cosmos3_patches_per_frame(720, 1280) == 23 * 40
    assert cosmos3_patches_per_frame(720, 1280) == 920
    assert COSMOS3_REFERENCE_LATENT_FRAMES == 48


def test_cosmos3_geometry_non_round_resolution():
    """Lock the ceil() behavior the gen-path geometry assertion depends on:
    non-multiple-of-32 dims must round UP (ceil), not floor. 540x960:
    ceil(540/32)=17, ceil(960/32)=30 -> 510 (floor would give 16*30=480)."""
    from lvsa.cosmos3 import cosmos3_patches_per_frame
    assert cosmos3_patches_per_frame(540, 960) == 17 * 30 == 510
    assert cosmos3_patches_per_frame(481, 481) == 16 * 16  # ceil(481/32)=16


import torch
import torch.nn as nn


class _FakeCosmosAttn(nn.Module):
    """Minimal stand-in exposing the attributes both processors read."""
    def __init__(self, hidden=64, heads=4, kv_heads=2, head_dim=16):
        super().__init__()
        self.num_attention_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        qd, kvd = heads * head_dim, kv_heads * head_dim
        self.to_q = nn.Linear(hidden, qd, bias=False)
        self.to_k = nn.Linear(hidden, kvd, bias=False)
        self.to_v = nn.Linear(hidden, kvd, bias=False)
        self.add_q_proj = nn.Linear(hidden, qd, bias=False)
        self.add_k_proj = nn.Linear(hidden, kvd, bias=False)
        self.add_v_proj = nn.Linear(hidden, kvd, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.Linear(qd, hidden, bias=False)
        self.to_add_out = nn.Linear(qd, hidden, bias=False)


def _rope_tuple(s_und, s_gen, head_dim):
    # identity-ish cos/sin (cos=1, sin=0) so RoPE is a no-op -> exact compare
    cos_u = torch.ones(s_und, head_dim); sin_u = torch.zeros(s_und, head_dim)
    cos_g = torch.ones(s_gen, head_dim); sin_g = torch.zeros(s_gen, head_dim)
    return (cos_u, sin_u, cos_g, sin_g)


# NOTE (CI coverage): the two tests below — the 1x==dense equivalence and the
# sparse-engagement check — are the only ones that exercise the actual processor
# *numerics*, and both importorskip on diffusers main. CI runs release diffusers,
# so CI does NOT cover the gen/und attention math; it covers only geometry, the
# batched guard, the geometry-mismatch guard, and the installer. A green CI does
# not certify the processor numerics — run these locally on diffusers main (or a
# scheduled main-pinned job) after any change to the attention path.
def test_lvsa_processor_matches_dense_at_1x():
    # Cosmos3 lives in diffusers main only — skip on release diffusers (e.g. CI).
    pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3AttnProcessor
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    torch.manual_seed(0)
    T_lat, P, head_dim = 6, 2, 16          # tiny grid; T_lat == ref below
    attn = _FakeCosmosAttn(head_dim=head_dim).eval()
    und = torch.randn(5, 64)               # 5 und (text) tokens
    gen = torch.randn(T_lat * P, 64)       # gen = clean frame grid
    rot = _rope_tuple(5, T_lat * P, head_dim)
    ref_und, ref_gen = Cosmos3AttnProcessor()(attn, und, gen, rot)
    # ref=T_lat -> 1x horizon -> kfi=1 -> every gen frame global -> dense
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=T_lat, num_patches=P,
                                    reference_latent_frames=T_lat)
    my_und, my_gen = proc(attn, und, gen, rot)
    assert torch.allclose(my_und, ref_und, atol=1e-5), "und path must be untouched"
    assert torch.allclose(my_gen, ref_gen, atol=1e-5), "gen LVSA must == dense at 1x"


def test_lvsa_processor_engages_sparse_above_ref():
    # The processor's __call__ lazily imports transformer_cosmos3 (main-only).
    pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    torch.manual_seed(0)
    T_lat, P, ref, head_dim = 24, 2, 6, 16      # 4x horizon -> sparse
    attn = _FakeCosmosAttn(head_dim=head_dim).eval()
    und = torch.randn(4, 64)
    gen = torch.randn(T_lat * P, 64)
    rot = _rope_tuple(4, T_lat * P, head_dim)
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=T_lat, num_patches=P,
                                    reference_latent_frames=ref)
    my_und, my_gen = proc(attn, und, gen, rot)
    assert my_gen.shape == (T_lat * P, 64)
    assert torch.isfinite(my_gen).all()
    # sparse: fewer than all frames are global anchors
    assert 0 < len(proc.metadata.global_indices) < T_lat


def test_install_swaps_all_layers():
    import torch.nn as nn
    from lvsa.cosmos3 import install_cosmos3_lvsa, Cosmos3LVSAAttnProcessor

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _FakeCosmosAttn()
            self.self_attn.processor = object()   # stand-in original processor
            self.self_attn.set_processor = lambda p: setattr(self.self_attn, "processor", p)

    class _Transformer(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(n)])

    tf = _Transformer(5)
    proc = install_cosmos3_lvsa(tf, num_frames=189, height=720, width=1280)
    assert isinstance(proc, Cosmos3LVSAAttnProcessor)
    assert hasattr(proc, "metadata")
    for layer in tf.layers:
        assert isinstance(layer.self_attn.processor, Cosmos3LVSAAttnProcessor)
        assert layer.self_attn.processor is proc  # same shared instance


def test_batched_input_raises_not_corrupts():
    """B>1 must raise, never silently corrupt. The processor mirrors the stock
    ``Cosmos3AttnProcessor`` byte-for-byte, and stock flattens batch into
    sequence (``view(-1, H, D)`` + ``unsqueeze(0)``): at B>1 the und causal
    mask would leak across batch elements and the gen LVSA metadata (built for
    S_gen) would mis-cover a B*S_gen query. ``Cosmos3OmniPipeline`` never
    batches (sequential CFG, one sample per call — verified on diffusers main),
    so the guard only trips if a future caller batches. No diffusers import
    needed: the guard fires before ``__call__``'s lazy import, so this test
    runs on release diffusers too.
    """
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor, _require_unbatched

    # unit: the guard itself
    ok_2d = torch.randn(6, 64)            # [S, C] (implicit single sample)
    ok_3d = torch.randn(1, 6, 64)         # [1, S, C] (explicit B=1)
    bad = torch.randn(2, 6, 64)           # [B=2, S, C]
    _require_unbatched(ok_2d, ok_2d)      # no raise
    _require_unbatched(ok_3d, ok_3d)      # no raise
    with pytest.raises(NotImplementedError, match="batch"):
        _require_unbatched(bad, ok_3d)
    with pytest.raises(NotImplementedError, match="batch"):
        _require_unbatched(ok_3d, bad)

    # integration: __call__ rejects batched inputs before touching attn
    proc = Cosmos3LVSAAttnProcessor(
        total_latent_frames=4, num_patches=2, reference_latent_frames=2,
    )
    with pytest.raises(NotImplementedError, match="batch"):
        proc(attn=None, und_seq=bad, gen_seq=bad, rotary_emb=None)


def test_gen_geometry_mismatch_raises():
    """A gen stream whose length != T_lat*P must raise loudly, not silently
    corrupt: build_global_kv indexes frame*P and the output is sliced per frame,
    so a one-token layout drift garbles attention with no error otherwise. The
    check fires before __call__'s lazy diffusers import (runs on release
    diffusers); attn=None proves it raises before any model use. T_lat=4, P=2
    -> expected gen length 8."""
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    proc = Cosmos3LVSAAttnProcessor(
        total_latent_frames=4, num_patches=2, reference_latent_frames=2,
    )
    good = torch.randn(8, 16)            # 4 * 2 = 8 tokens, correct
    wrong = torch.randn(7, 16)           # off-by-one -> must raise
    und = torch.randn(3, 16)
    # correct length passes the geometry gate (then fails later on attn=None,
    # NOT with a ValueError about geometry)
    with pytest.raises(Exception) as ei:
        proc(attn=None, und_seq=und, gen_seq=good, rotary_emb=None)
    assert "T_lat*P" not in str(ei.value), "correct geometry must pass the gate"
    # wrong length is rejected at the geometry gate
    with pytest.raises(ValueError, match=r"T_lat\*P"):
        proc(attn=None, und_seq=und, gen_seq=wrong, rotary_emb=None)
    # 3-D [1, S, C] form is handled too (shape[-2])
    with pytest.raises(ValueError, match=r"T_lat\*P"):
        proc(attn=None, und_seq=und.unsqueeze(0), gen_seq=wrong.unsqueeze(0),
             rotary_emb=None)


def test_cosmos_generate_has_no_hardcoded_cuda():
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "examples" / "cosmos_generate.py"
    text = src.read_text()
    assert 'torch.cuda.max_memory_allocated' not in text
    assert 'torch.cuda.reset_peak_memory_stats' not in text
    assert 'Generator("cuda")' not in text and "Generator('cuda')" not in text
    assert '.to("cuda")' not in text and ".to('cuda')" not in text
    assert "from lvsa.device import" in text and "get_device" in text


def test_cosmos_processor_print_mask(capsys):
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=32, num_patches=4,
                                    reference_latent_frames=21, window_size=4,
                                    n_first_frames=4, sparsity_scale=1.0)
    proc.print_attention_mask_compact()
    assert capsys.readouterr().out.strip() != ""


def test_cosmos_processor_set_step_rotates():
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    from lvsa.sparse_attention import LVSAMetadata
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=32, num_patches=4,
                                    reference_latent_frames=21, window_size=4,
                                    n_first_frames=4, sparsity_scale=1.0)
    base = list(proc.metadata.global_indices)
    proc.set_step(0)
    assert list(proc.metadata.global_indices) == base            # offset 0 == base
    proc.set_step(1)
    if proc.key_frame_interval:                                  # deterministic vs a direct build
        off = 1 % proc.key_frame_interval
        ref = LVSAMetadata.build(
            total_latent_frames=32, num_patches=4, window_size=proc.window_lat,
            n_first_frames=proc.n_first_lat, key_frame_interval=proc.key_frame_interval,
            rank=0, world=1, expand_window=True, keyframe_offset=off,
            reference_frames=21, sparsity_scale=1.0).global_indices
        assert list(proc.metadata.global_indices) == list(ref)
    proc.set_step(0)
    assert list(proc.metadata.global_indices) == base            # round-trips back


def test_cosmos_dualstream_runner_importable():
    from lvsa.cosmos_flashinfer import Cosmos3DualStreamRunner, get_shared_runner
    r = Cosmos3DualStreamRunner()            # constructs without CUDA
    assert hasattr(r, "run") and callable(r.run)
    s1 = get_shared_runner(); s2 = get_shared_runner()
    assert s1 is s2                          # process-wide singleton


def test_cosmos_processor_use_flashinfer_flag():
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    from lvsa.cosmos_flashinfer import FLASHINFER_AVAILABLE
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=32, num_patches=4,
                                    reference_latent_frames=21, window_size=4,
                                    n_first_frames=4, sparsity_scale=1.0,
                                    use_flashinfer=True)
    # honored only when flashinfer is importable; otherwise gracefully False
    assert proc.use_flashinfer == bool(FLASHINFER_AVAILABLE)
    proc2 = Cosmos3LVSAAttnProcessor(total_latent_frames=32, num_patches=4,
                                     reference_latent_frames=21, window_size=4,
                                     n_first_frames=4, sparsity_scale=1.0)
    assert proc2.use_flashinfer is False    # default = SDPA


import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer needs CUDA")
def test_cosmos_fused_matches_sdpa():
    from lvsa.sparse_attention import LVSAMetadata, lvsa_sdpa, build_global_kv, compute_auto_kfi
    from lvsa.cosmos_flashinfer import get_shared_runner, FLASHINFER_AVAILABLE
    if not FLASHINFER_AVAILABLE:
        pytest.skip("flashinfer not installed")
    torch.manual_seed(0)
    T, P, H, Hkv, D = 16, 64, 8, 8, 64
    dev, dt = "cuda", torch.bfloat16
    kfi = compute_auto_kfi(T, 2, 1, reference_frames=21, sparsity_scale=1.0)
    md = LVSAMetadata.build(total_latent_frames=T, num_patches=P, window_size=2,
                            n_first_frames=1, key_frame_interval=kfi, rank=0, world=1,
                            expand_window=True, reference_frames=21, sparsity_scale=1.0)
    S_gen = T * P
    S_und = 37  # ragged und length (NOT a multiple of P) — the case that motivates the runner
    qg = torch.randn(1, S_gen, H, D, device=dev, dtype=dt)
    kg = torch.randn(1, S_gen, Hkv, D, device=dev, dtype=dt)
    vg = torch.randn(1, S_gen, Hkv, D, device=dev, dtype=dt)
    k_und = torch.randn(1, S_und, Hkv, D, device=dev, dtype=dt)
    v_und = torch.randn(1, S_und, Hkv, D, device=dev, dtype=dt)
    kglob, vglob = build_global_kv(kg, vg, md.global_indices, P)
    kglob = torch.cat([kglob, k_und], dim=1); vglob = torch.cat([vglob, v_und], dim=1)
    out_sdpa = lvsa_sdpa(qg, kg, vg, kglob, vglob, md).float()
    out_fi = get_shared_runner().run(qg, kg, vg, kglob, vglob, md).float()
    max_diff = (out_sdpa - out_fi).abs().max().item()
    assert max_diff < 2e-2, f"fused vs SDPA max abs diff {max_diff}"


def test_cosmos_processor_explicit_kfi():
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=32, num_patches=4,
                                    reference_latent_frames=21, window_size=4,
                                    n_first_frames=4, sparsity_scale=1.0,
                                    key_frame_interval=5)
    assert proc.key_frame_interval == 5      # explicit override honored


def test_cosmos_new_flags_present():
    pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
    from examples import cosmos_generate
    flags = {a.option_strings[0] for a in cosmos_generate.build_parser()._actions if a.option_strings}
    for f in ["--flashinfer", "--auto-keyframes", "--key-frame-interval",
              "--reference-latent-frames", "--negative-prompt", "--profile",
              "--show-mask", "--show-mask-compact", "--rotate-keyframes"]:
        assert f in flags, f"missing {f}"


def test_cosmos_processor_expand_window_param():
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    p_exp = Cosmos3LVSAAttnProcessor(total_latent_frames=32, num_patches=4,
                                     reference_latent_frames=21, window_size=4,
                                     n_first_frames=4, sparsity_scale=1.0)            # default
    assert p_exp._expand_window is True
    p_ada = Cosmos3LVSAAttnProcessor(total_latent_frames=32, num_patches=4,
                                     reference_latent_frames=21, window_size=4,
                                     n_first_frames=4, sparsity_scale=1.0,
                                     expand_window=False)
    assert p_ada._expand_window is False


def test_cosmos_no_expand_window_flag():
    pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
    from examples import cosmos_generate
    flags = {a.option_strings[0] for a in cosmos_generate.build_parser()._actions if a.option_strings}
    assert "--no-expand-window" in flags
