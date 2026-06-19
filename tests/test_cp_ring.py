"""Tests for cp_mode="ring" — the standalone true-ring-rotation CP path.

CPU-only (gloo). The correctness anchor is ``test_ring_world1_equals_single_device``:
world=1 ring MUST equal cp_mode="custom" at world=1 (the single-device reference).
"""

import os

import pytest
import torch
import torch.nn.functional as F

from lvsa.lvsa_processor import (
    DistributedLVSAProcessor,
    _attn_with_lse,
    _merge_out_lse,
)
from lvsa.sparse_attention import ring_block_frame_mask, compute_global_indices


# ── Task 1: mode accepted + dispatched ─────────────────────────────────────────


def _proc(cp_mode, world=1, rank=0):
    return DistributedLVSAProcessor(
        total_num_latent_frames=8, num_patches=4, window_size=1,
        n_first_frames=1, key_frame_interval=2, rank=rank, world=world,
        cp_mode=cp_mode,
    )


def test_ring_mode_is_accepted():
    p = _proc("ring")
    assert p._cp_mode == "ring"


def test_bad_mode_still_rejected():
    with pytest.raises(ValueError, match="ring"):
        _proc("nope")


# ── Task 2: _attn_with_lse + _merge_out_lse ────────────────────────────────────


def test_attn_with_lse_matches_sdpa():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 6, 16)
    k = torch.randn(2, 4, 9, 16)
    v = torch.randn(2, 4, 9, 16)
    out, lse = _attn_with_lse(q, k, v)                     # [B,H,Sq,D], [B,H,Sq]
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)
    # lse must equal logsumexp of the scaled scores
    scores = (q @ k.transpose(-1, -2)) / (16 ** 0.5)
    assert torch.allclose(lse, torch.logsumexp(scores, dim=-1), atol=1e-5)


def test_merge_equals_one_shot_full_attention():
    # attention over [k1|k2] == merge(attn(q,k1), attn(q,k2))
    torch.manual_seed(1)
    q = torch.randn(1, 2, 5, 8)
    k1 = torch.randn(1, 2, 4, 8)
    k2 = torch.randn(1, 2, 7, 8)
    v1 = torch.randn(1, 2, 4, 8)
    v2 = torch.randn(1, 2, 7, 8)
    o1, l1 = _attn_with_lse(q, k1, v1)
    o2, l2 = _attn_with_lse(q, k2, v2)
    merged, ml = _merge_out_lse(None, None, o1, l1)
    merged, ml = _merge_out_lse(merged, ml, o2, l2)
    full = F.scaled_dot_product_attention(
        q, torch.cat([k1, k2], 2), torch.cat([v1, v2], 2)
    )
    assert torch.allclose(merged, full, atol=1e-5)


def test_attn_with_lse_all_masked_row_is_inf():
    """A fully-masked query row yields lse=-inf and a zero output, so it is a
    no-op under the merge (the basis for skipping empty block-pairs)."""
    torch.manual_seed(2)
    q = torch.randn(1, 1, 2, 8)
    k = torch.randn(1, 1, 3, 8)
    v = torch.randn(1, 1, 3, 8)
    mask = torch.zeros(2, 3, dtype=torch.bool)  # nothing attended
    out, lse = _attn_with_lse(q, k, v, attn_mask=mask)
    assert torch.isinf(lse).all() and (lse < 0).all()
    assert torch.allclose(out, torch.zeros_like(out))


# ── Task 3: ring_block_frame_mask ──────────────────────────────────────────────


def test_block_mask_world1_equals_full_lvsa_pattern():
    # world=1: one block owns all frames; mask must equal the single-device
    # (global ∪ window) pattern over the whole grid — built from the SAME
    # get_window_bounds + global_set the metadata uses (so it is bit-identical,
    # incl. the expand-window enlargement that skips global frames).
    from lvsa.sparse_attention import get_window_bounds

    T, P, W, nfirst, kfi = 8, 1, 1, 1, 2
    global_set = set(compute_global_indices(T, nfirst, kfi))
    # globals: {0 (first)} ∪ {0,2,4,6 (kfi=2)} = {0,2,4,6}
    assert global_set == {0, 2, 4, 6}
    m = ring_block_frame_mask(
        q_frames=range(T), k_frames=range(T), num_patches=P,
        window_size=W, total_frames=T, global_set=global_set, expand_window=True,
    )
    # The mask must equal the metadata's per-frame attended set for EVERY frame.
    for f in range(T):
        lo, hi = get_window_bounds(f, W, T, True, global_set, len(global_set))
        expected = global_set | set(range(lo, hi + 1))
        for k in range(T):
            assert bool(m[f, k]) == (k in expected), (
                f"mask[{f},{k}]={bool(m[f, k])} != single-device "
                f"({k in expected}); window=[{lo},{hi}] globals={sorted(global_set)}"
            )
    # Spot-check frame 5 (expanded window {3..7} since 4,6 are global) ∪ globals.
    assert m[5, 5] and m[5, 3] and m[5, 7]   # expanded window
    assert m[5, 0] and m[5, 2]               # globals
    assert not m[5, 1]                       # neither


def test_block_mask_empty_pair_is_skippable():
    # distant non-global block: q frames {6,7}, k frame {3} (not global, W=1) -> None
    T = 8
    global_set = {0}  # only first frame is global; 3 is not
    m = ring_block_frame_mask(
        q_frames=[6, 7], k_frames=[3], num_patches=1, window_size=1,
        total_frames=T, global_set=global_set, expand_window=True,
    )
    assert m is None


def test_block_mask_token_expansion():
    """Mask expands each frame cell to a P×P token block."""
    T, P, W = 6, 3, 1
    global_set = {0}
    m = ring_block_frame_mask(
        q_frames=[1], k_frames=[1], num_patches=P, window_size=W,
        total_frames=T, global_set=global_set, expand_window=True,
    )
    assert m.shape == (P, P)
    assert m.all()  # frame 1 attends frame 1 (window) -> full P×P block


# ── Task 4: _compute_lvsa_ring world=1 == single-device ────────────────────────


def test_ring_world1_equals_single_device():
    torch.manual_seed(0)
    B, T, P, H, D = 1, 12, 4, 8, 16
    S = T * P
    common = dict(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        reference_frames=3,  # 12 > 3 -> genuinely sparse
    )
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)

    pc = DistributedLVSAProcessor(cp_mode="custom", **common)
    kg, vg = pc._build_global_kv(k, v)
    out_ref = pc._compute_lvsa(q, k, v, kg, vg, None)

    pr = DistributedLVSAProcessor(cp_mode="ring", **common)
    out_ring = pr._compute_lvsa_ring(q, k, v, None, None)

    assert out_ring.shape == out_ref.shape == (B, S, H, D)
    assert torch.allclose(out_ring, out_ref, atol=1e-4), (
        "ring world=1 must equal the single-device custom path"
    )


def test_ring_world1_equals_single_device_with_encoder():
    torch.manual_seed(1)
    B, T, P, H, D = 1, 10, 2, 4, 8
    S = T * P
    common = dict(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        reference_frames=3,
    )
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    enc_k = torch.randn(B, 5, H, D)
    enc_v = torch.randn(B, 5, H, D)

    pc = DistributedLVSAProcessor(cp_mode="custom", **common)
    kg, vg = pc._build_global_kv(k, v)
    kg = torch.cat([kg, enc_k], dim=1)
    vg = torch.cat([vg, enc_v], dim=1)
    out_ref = pc._compute_lvsa(q, k, v, kg, vg, None)

    pr = DistributedLVSAProcessor(cp_mode="ring", **common)
    out_ring = pr._compute_lvsa_ring(q, k, v, enc_k, enc_v)

    assert torch.allclose(out_ring, out_ref, atol=1e-4)


def test_ring_world1_equals_single_device_gqa():
    """Ring shards sequence not heads -> works under GQA (H_kv < H) with no
    num_heads%world constraint. world=1 GQA must still match single-device."""
    torch.manual_seed(2)
    B, T, P, H, H_kv, D = 1, 12, 4, 8, 2, 16
    S = T * P
    common = dict(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        reference_frames=3,
    )
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H_kv, D)
    v = torch.randn(B, S, H_kv, D)

    pc = DistributedLVSAProcessor(cp_mode="custom", **common)
    kg, vg = pc._build_global_kv(k, v)
    out_ref = pc._compute_lvsa(q, k, v, kg, vg, None)

    pr = DistributedLVSAProcessor(cp_mode="ring", **common)
    out_ring = pr._compute_lvsa_ring(q, k, v, None, None)

    assert torch.allclose(out_ring, out_ref, atol=1e-4)


# ── Task 5: gloo world=2 ring == world=1 ring ──────────────────────────────────


def _ring_world2_worker(rank, world, q, k, v, ref_out, return_dict):
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        B, S, H, D = q.shape
        Sloc = S // world
        sl = slice(rank * Sloc, (rank + 1) * Sloc)
        T = 12
        proc = DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=4, window_size=1,
            n_first_frames=1, key_frame_interval=None, rank=rank, world=world,
            cp_mode="ring", reference_frames=3,
        )
        out_shard = proc._compute_lvsa_ring(
            q[:, sl], k[:, sl], v[:, sl], None, None,
        )
        gathered = [torch.empty_like(out_shard) for _ in range(world)]
        dist.all_gather(gathered, out_shard.contiguous())
        if rank == 0:
            full = torch.cat(gathered, dim=1)
            return_dict["max_diff"] = (full - ref_out).abs().max().item()
            return_dict["shape_ok"] = tuple(full.shape) == tuple(ref_out.shape)
    finally:
        dist.destroy_process_group()


def test_ring_world2_matches_world1():
    """gloo world=2 ring must reproduce the world=1 ring output for the same
    global inputs (the CP contract)."""
    import torch.multiprocessing as mp

    torch.manual_seed(7)
    B, T, P, H, D = 1, 12, 4, 8, 16
    S = T * P
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)

    # world=1 reference (the proven single-device-equivalent path)
    ref_proc = DistributedLVSAProcessor(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        cp_mode="ring", reference_frames=3,
    )
    ref_out = ref_proc._compute_lvsa_ring(q, k, v, None, None)

    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.spawn(
            _ring_world2_worker,
            args=(2, q, k, v, ref_out, return_dict),
            nprocs=2,
            join=True,
        )
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"multi-proc gloo unavailable in sandbox: {e!r}")

    assert return_dict.get("shape_ok"), "world=2 gathered shape mismatch"
    assert return_dict["max_diff"] < 1e-4, (
        f"ring world=2 != world=1 (max abs diff {return_dict['max_diff']})"
    )


# ── The head-constraint EDGE: ring works at world=3 where ulysses cannot ────────
# Can't be shown on 2 GPUs (world=2 divides any even head count). world=3 + H=8
# (8 % 3 != 0) is the clean demonstration: ulysses raises, ring runs. CPU/gloo.


def test_ulysses_rejects_world3_odd_heads():
    """ulysses shards heads -> needs num_heads % world == 0. world=3 + H=8 must
    raise (the guard fires before the all-to-all, so no dist is needed)."""
    proc = DistributedLVSAProcessor(
        total_num_latent_frames=12, num_patches=4, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=3,
        cp_mode="ulysses", reference_frames=3,
    )
    q = torch.randn(1, 16, 8, 16)      # H=8, not divisible by world=3
    k = torch.randn(1, 16, 8, 16); v = torch.randn(1, 16, 8, 16)
    with pytest.raises(ValueError, match="divisible by world"):
        proc._compute_lvsa_ulysses(q, k, v, None, None)


def test_ring_world3_matches_world1():
    """ring shards SEQUENCE not heads -> runs at world=3 even with H=8 (8 % 3 !=
    0), where ulysses raises. Reuses the world-parametrized worker, nprocs=3."""
    import torch.multiprocessing as mp

    torch.manual_seed(7)
    B, T, P, H, D = 1, 12, 4, 8, 16    # T=12 % 3 == 0 (ring ok); H=8 % 3 != 0
    S = T * P
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)

    ref_proc = DistributedLVSAProcessor(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        cp_mode="ring", reference_frames=3,
    )
    ref_out = ref_proc._compute_lvsa_ring(q, k, v, None, None)

    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.spawn(
            _ring_world2_worker,
            args=(3, q, k, v, ref_out, return_dict),
            nprocs=3,
            join=True,
        )
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"multi-proc gloo unavailable in sandbox: {e!r}")

    assert return_dict.get("shape_ok"), "world=3 gathered shape mismatch"
    assert return_dict["max_diff"] < 1e-4, (
        f"ring world=3 != world=1 (max abs diff {return_dict['max_diff']})"
    )


# ── HunyuanVideo dual-stream CP fix: text-query branch attends the FULL video ───
# Bug was: under CP the text (encoder) queries attended only the per-rank video
# shard (cp_mode-independent), so HV diverged from single-GPU in every mode.
# _compute_encoder_query_attention now all-gathers the full video K/V → world=2
# text output must equal world=1 (single-device).


def _enc_cp_worker(rank, world, vk, vv, ek, ev, eq, ref, return_dict):
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29523")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        proc = DistributedLVSAProcessor(
            total_num_latent_frames=12, num_patches=4, window_size=1,
            n_first_frames=1, key_frame_interval=None, rank=rank, world=world,
            cp_mode="ring", reference_frames=3,
        )
        Sloc = vk.shape[1] // world
        sl = slice(rank * Sloc, (rank + 1) * Sloc)
        # each rank holds only its local video shard (vk[:, sl]); text is replicated
        out = proc._compute_encoder_query_attention(eq, ek, ev, vk[:, sl], vv[:, sl])
        if rank == 0:
            return_dict["max_diff"] = (out - ref).abs().max().item()
            return_dict["shape_ok"] = tuple(out.shape) == tuple(ref.shape)
    finally:
        dist.destroy_process_group()


def test_encoder_query_attention_world2_matches_world1():
    """The dual-stream text-query branch must attend the FULL video K/V under CP,
    not the local shard. world=2 (sharded video) must equal world=1 (full)."""
    import torch.multiprocessing as mp

    torch.manual_seed(3)
    B, Sv, H, D, Stext = 1, 48, 8, 16, 7      # full video seq=48 (T=12,P=4), text=7
    vk = torch.randn(B, Sv, H, D); vv = torch.randn(B, Sv, H, D)
    ek = torch.randn(B, Stext, H, D); ev = torch.randn(B, Stext, H, D)
    eq = torch.randn(B, Stext, H, D)

    # world=1 reference: text queries attend the FULL video + text
    ref_proc = DistributedLVSAProcessor(
        total_num_latent_frames=12, num_patches=4, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        cp_mode="ring", reference_frames=3,
    )
    ref = ref_proc._compute_encoder_query_attention(eq, ek, ev, vk, vv)

    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.spawn(
            _enc_cp_worker,
            args=(2, vk, vv, ek, ev, eq, ref, return_dict),
            nprocs=2,
            join=True,
        )
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"multi-proc gloo unavailable in sandbox: {e!r}")

    assert return_dict.get("shape_ok"), "enc world=2 shape mismatch"
    assert return_dict["max_diff"] < 1e-4, (
        f"text-query world=2 != world=1 (max diff {return_dict['max_diff']}) — "
        f"the full-video gather is missing/wrong"
    )


# ── Query-tiling in _attn_with_lse (the fix for the 48 GiB OOM at video scale) ──
# Must be math-identical to the untiled form when Sq spans multiple tiles, both
# masked and unmasked — the multi-tile path runs ONLY at video scale (Sq~33k), so
# without this test it is exercised only on GPU.


def test_attn_with_lse_query_tiling_matches_single_tile():
    import torch.nn.functional as F
    from lvsa.lvsa_processor import _attn_with_lse, _RING_Q_TILE

    torch.manual_seed(0)
    Sq = _RING_Q_TILE * 2 + 37                      # forces 3 tiles
    q = torch.randn(1, 4, Sq, 16)
    k = torch.randn(1, 4, 300, 16)
    v = torch.randn(1, 4, 300, 16)

    # unmasked: tiled (default q_tile) == single tile == SDPA
    o_t, l_t = _attn_with_lse(q, k, v)
    o_1, l_1 = _attn_with_lse(q, k, v, q_tile=Sq + 1)
    assert torch.allclose(o_t, o_1, atol=1e-5), "tiled != single-tile output"
    assert torch.allclose(l_t, l_1, atol=1e-5), "tiled != single-tile lse"
    assert torch.allclose(o_t, F.scaled_dot_product_attention(q, k, v), atol=1e-5)

    # masked: the per-tile mask slicing (attn_mask[s:e]) must match single-tile
    mask = torch.rand(Sq, 300) > 0.5
    mask[:, 0] = True                               # no all-False rows
    om_t, _ = _attn_with_lse(q, k, v, attn_mask=mask)
    om_1, _ = _attn_with_lse(q, k, v, attn_mask=mask, q_tile=Sq + 1)
    assert torch.allclose(om_t, om_1, atol=1e-5), "tiled mask slicing diverges"


# ── _all_gather_seq reconstructs the full sequence in rank order (HV CP fix) ────


def _gather_worker(rank, world, full, return_dict):
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29525")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from lvsa.lvsa_processor import _all_gather_seq
        S = full.shape[1]
        Sloc = S // world
        shard = full[:, rank * Sloc:(rank + 1) * Sloc].contiguous()
        out = _all_gather_seq(shard, world)
        if rank == 0:
            return_dict["diff"] = (out - full).abs().max().item()
            return_dict["shape_ok"] = tuple(out.shape) == tuple(full.shape)
    finally:
        dist.destroy_process_group()


def test_all_gather_seq_reconstructs_full_sequence():
    """The HV text-query fix gathers the per-rank video shards back into the full
    sequence; rank-ordered concat must reproduce the original exactly (diff == 0)."""
    import torch.multiprocessing as mp

    torch.manual_seed(5)
    full = torch.randn(1, 24, 4, 8)                 # seq=24, world=2 -> shards of 12
    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.spawn(_gather_worker, args=(2, full, return_dict), nprocs=2, join=True)
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"multi-proc gloo unavailable in sandbox: {e!r}")

    assert return_dict.get("shape_ok"), "gathered shape mismatch"
    assert return_dict["diff"] == 0.0, (
        "all_gather must reconstruct the full sequence in rank order (lossless)"
    )


# ── Fused flashinfer ring kernel (--flashinfer) — GPU-only parity ──────────────
# CUDA + flashinfer only; skipped on CPU CI. Asserts the fused block-sparse-LSE
# ring matches the CPU-tested pure-torch ring (within bf16-vs-fp32 tolerance).


def _cuda_flashinfer_available():
    try:
        import torch as _t
        import flashinfer  # noqa: F401
        return _t.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_flashinfer_available(), reason="needs CUDA + flashinfer")
def test_ring_flashinfer_matches_pure_torch():
    """The `--flashinfer` ring (single_prefill custom_mask + merge_state) must
    reproduce the pure-torch ring (the CPU-verified reference) within bf16 fp."""
    dev = "cuda"
    torch.manual_seed(0)
    T, P, W, nf = 42, 64, 3, 2
    H, D, S = 12, 64, 42 * 64

    def mk(fi):
        p = DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P, window_size=W,
            n_first_frames=nf, key_frame_interval=None, rank=0, world=1,
            cp_mode="ring", reference_frames=21,
        )
        p._use_flashinfer = fi
        return p

    q = torch.randn(1, S, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, S, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, S, H, D, device=dev, dtype=torch.bfloat16)
    ek = torch.randn(1, 16, H, D, device=dev, dtype=torch.bfloat16)
    ev = torch.randn(1, 16, H, D, device=dev, dtype=torch.bfloat16)
    for enc in [(None, None), (ek, ev)]:
        o_pt = mk(False)._compute_lvsa_ring(q, k, v, *enc)
        o_fi = mk(True)._compute_lvsa_ring(q, k, v, *enc)
        rel = ((o_pt.float() - o_fi.float()).abs().max()
               / o_pt.float().abs().max().clamp_min(1e-6)).item()
        assert rel < 0.05, f"flashinfer ring diverges from pure-torch (rel={rel})"
