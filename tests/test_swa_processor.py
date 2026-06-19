"""Tests for lvsa_processor.py — static helpers, window bounds, CSR, and processor init."""

import math
from unittest.mock import patch

import torch
import pytest

from lvsa.lvsa_processor import DistributedLVSAProcessor, WanDistributedLVSAProcessor


# ── _adaptive_window_bounds ───────────────────────────────────────────────────


class TestAdaptiveWindowBounds:
    """Tests for the static _adaptive_window_bounds method."""

    def test_center_frame(self):
        """Middle frame with plenty of room on both sides."""
        lo, hi = WanDistributedLVSAProcessor._adaptive_window_bounds(10, 3, 21)
        assert lo == 7
        assert hi == 13

    def test_first_frame_shifts_right(self):
        """Frame 0 should shift the window to start at 0."""
        lo, hi = WanDistributedLVSAProcessor._adaptive_window_bounds(0, 3, 21)
        assert lo == 0
        assert hi >= 2 * 3  # at least 2W

    def test_last_frame_shifts_left(self):
        """Last frame should shift the window to end at T-1."""
        T = 21
        lo, hi = WanDistributedLVSAProcessor._adaptive_window_bounds(T - 1, 3, T)
        assert hi == T - 1
        assert lo <= T - 1 - 2 * 3

    def test_constant_width(self):
        """Window width should always be 2W+1 (or T if T < 2W+1)."""
        W, T = 3, 21
        for f in range(T):
            lo, hi = WanDistributedLVSAProcessor._adaptive_window_bounds(f, W, T)
            assert hi - lo + 1 == min(2 * W + 1, T)

    def test_small_T(self):
        """When T <= 2W+1, window should cover the entire sequence."""
        W, T = 5, 8  # 2*5+1 = 11 > 8
        for f in range(T):
            lo, hi = WanDistributedLVSAProcessor._adaptive_window_bounds(f, W, T)
            assert lo == 0
            assert hi == T - 1

    def test_window_always_contains_query_frame(self):
        """The query frame should always be within its own window."""
        W, T = 4, 50
        for f in range(T):
            lo, hi = WanDistributedLVSAProcessor._adaptive_window_bounds(f, W, T)
            assert lo <= f <= hi


# ── _compute_boundary_guard_frames ────────────────────────────────────────────


class TestComputeBoundaryGuardFrames:
    """Tests for boundary guard frame computation."""

    def test_single_gpu_no_guards(self):
        """Single GPU should have no boundary guards."""
        guards = WanDistributedLVSAProcessor._compute_boundary_guard_frames(
            total_frames=21, local_seq=21 * 30, num_patches=30, world=1, window_size=3,
        )
        assert guards == []

    def test_two_gpus_guards_at_boundary(self):
        """Two GPUs: boundary at mid-sequence should produce guards."""
        T, P, W = 20, 10, 2
        local_seq = T * P // 2  # 100 tokens per rank
        guards = WanDistributedLVSAProcessor._compute_boundary_guard_frames(
            total_frames=T, local_seq=local_seq, num_patches=P, world=2, window_size=W,
        )
        # Boundary frame = (1 * 100) // 10 = 10
        # Guard range [10-2, 10+2] = [8, 9, 10, 11, 12]
        boundary_frame = local_seq // P
        expected = list(range(max(0, boundary_frame - W), min(T, boundary_frame + W + 1)))
        assert guards == expected

    def test_guards_are_sorted(self):
        guards = WanDistributedLVSAProcessor._compute_boundary_guard_frames(
            total_frames=41, local_seq=41 * 30 // 4, num_patches=30, world=4, window_size=3,
        )
        assert guards == sorted(guards)

    def test_guards_within_frame_range(self):
        T = 21
        guards = WanDistributedLVSAProcessor._compute_boundary_guard_frames(
            total_frames=T, local_seq=T * 30 // 2, num_patches=30, world=2, window_size=5,
        )
        for g in guards:
            assert 0 <= g < T

    def test_guards_no_duplicates(self):
        guards = WanDistributedLVSAProcessor._compute_boundary_guard_frames(
            total_frames=41, local_seq=41 * 30 // 4, num_patches=30, world=4, window_size=5,
        )
        assert len(guards) == len(set(guards))


# ── _compute_global_indices ───────────────────────────────────────────────────


class TestComputeGlobalIndices:
    """Tests for global index computation with rotation offsets."""

    def test_first_frames_always_included(self):
        indices = WanDistributedLVSAProcessor._compute_global_indices(
            total_frames=21, n_first_frames=3, key_frame_interval=5, offset=0,
        )
        assert all(f in indices for f in [0, 1, 2])

    def test_periodic_keyframes(self):
        indices = WanDistributedLVSAProcessor._compute_global_indices(
            total_frames=21, n_first_frames=1, key_frame_interval=5, offset=0,
        )
        # offset=0 → {0, 5, 10, 15, 20}
        for f in [0, 5, 10, 15, 20]:
            assert f in indices

    def test_offset_shifts_keyframes(self):
        indices = WanDistributedLVSAProcessor._compute_global_indices(
            total_frames=21, n_first_frames=1, key_frame_interval=5, offset=2,
        )
        # offset=2 → {0, 2, 7, 12, 17} (wrapping: (2+4*5)%21 = 1 → wait, let me check)
        # n_keyframes = ceil(21/5) = 5
        # frames: (2+0*5)%21=2, (2+1*5)%21=7, (2+2*5)%21=12, (2+3*5)%21=17, (2+4*5)%21=1
        # first_frames: {0}
        # total: {0, 1, 2, 7, 12, 17}
        for f in [0, 2, 7, 12, 17]:
            assert f in indices

    def test_sorted_output(self):
        indices = WanDistributedLVSAProcessor._compute_global_indices(
            total_frames=41, n_first_frames=3, key_frame_interval=7, offset=3,
        )
        assert indices == sorted(indices)

    def test_no_keyframe_interval(self):
        """key_frame_interval=None should return only first frames."""
        indices = WanDistributedLVSAProcessor._compute_global_indices(
            total_frames=21, n_first_frames=3, key_frame_interval=None, offset=0,
        )
        assert indices == [0, 1, 2]

    def test_constant_periodic_count_across_offsets(self):
        """Each offset produces exactly ceil(T/kfi) periodic keyframes.

        The *total* global count may vary because the union with
        n_first_frames can overlap differently per offset.  But the
        periodic set itself is always the same size.
        """
        T, kfi = 21, 4
        n_keyframes = math.ceil(T / kfi)  # 6
        for offset in range(kfi):
            periodic = set()
            for i in range(n_keyframes):
                periodic.add((offset + i * kfi) % T)
            assert len(periodic) == n_keyframes

    def test_all_indices_in_range(self):
        T = 30
        indices = WanDistributedLVSAProcessor._compute_global_indices(
            T, 3, 6, offset=1,
        )
        for f in indices:
            assert 0 <= f < T

    def test_full_rotation_covers_all_frames(self):
        """Over a full rotation cycle, every frame should appear as a global at least once."""
        T, n, kfi = 21, 1, 4
        all_frames = set()
        for offset in range(kfi):
            indices = WanDistributedLVSAProcessor._compute_global_indices(
                T, n, kfi, offset,
            )
            all_frames.update(indices)
        assert all_frames == set(range(T))


# ── _compute_auto_kfi ─────────────────────────────────────────────────────────


class TestComputeAutoKfi:
    """Tests for automatic kfi computation targeting ~21 attended frames."""

    def test_basic_21_frames(self):
        """For T=21, kfi should produce close to 21 attended frames."""
        kfi = WanDistributedLVSAProcessor._compute_auto_kfi(
            total_frames=21, window_size=3, n_first_frames=3,
        )
        assert isinstance(kfi, int)
        assert kfi >= 1

    def test_large_T_reasonable_kfi(self):
        """For large T, kfi should be large enough to keep attended count manageable."""
        kfi = WanDistributedLVSAProcessor._compute_auto_kfi(
            total_frames=200, window_size=3, n_first_frames=3,
        )
        # With T=200, we need fewer globals to stay near 21 attended
        assert kfi > 1

    def test_kfi_always_positive(self):
        for T in [5, 21, 41, 81, 161]:
            kfi = WanDistributedLVSAProcessor._compute_auto_kfi(T, 3, 3)
            assert kfi >= 1


# ── WanDistributedLVSAProcessor.__init__ ──────────────────────────────────────


def _make_processor(T=21, P=30, W=3, n_first=3, kfi=4, rank=0, world=1):
    """Helper to create a processor with typical parameters."""
    return WanDistributedLVSAProcessor(
        total_num_latent_frames=T,
        num_patches=P,
        window_size=W,
        n_first_frames=n_first,
        key_frame_interval=kfi,
        rank=rank,
        world=world,
    )


class TestGeometryGuard:
    """The __call__ geometry guard: LVSA must only engage when the runtime
    sequence is the expected frame grid (local_seq × world == T × P), else it
    falls back to dense attention rather than silently misaligning."""

    def test_matches_exact_single_gpu(self):
        proc = _make_processor(T=21, P=30, world=1)
        assert proc._geometry_matches(21 * 30) is True

    def test_mismatch_extra_tokens(self):
        # Prepended history / non-frame-structured layout → extra tokens.
        proc = _make_processor(T=21, P=30, world=1)
        assert proc._geometry_matches(21 * 30 + 100) is False

    def test_matches_multi_gpu_shard(self):
        # CP: per-rank local_seq = T*P // world.
        proc = _make_processor(T=20, P=30, world=2)
        assert proc._geometry_matches(20 * 30 // 2) is True
        assert proc._geometry_matches(20 * 30) is False  # full seq on a sharded rank

    def test_helios_like_mismatch(self):
        # Helios empirically: configured 33*960, runtime 11040 (history+chunk).
        proc = _make_processor(T=33, P=960, world=1)
        assert proc._geometry_matches(33 * 960) is True
        assert proc._geometry_matches(11040) is False


class TestProcessorInit:
    """Tests for processor initialization and derived data structures."""

    def _make_processor(self, T=21, P=30, W=3, n_first=3, kfi=4, rank=0, world=1):
        return _make_processor(T=T, P=P, W=W, n_first=n_first, kfi=kfi, rank=rank, world=world)

    def test_local_seq_computation(self):
        T, P, world = 21, 30, 1
        proc = self._make_processor(T=T, P=P, world=world)
        assert proc.local_seq == T * P

    def test_local_seq_multi_gpu(self):
        T, P, world = 20, 30, 2  # 20*30=600, 600/2=300
        proc = self._make_processor(T=T, P=P, world=world, rank=0)
        assert proc.local_seq == 300

    def test_global_token_start(self):
        T, P = 20, 30
        proc0 = self._make_processor(T=T, P=P, world=2, rank=0)
        proc1 = self._make_processor(T=T, P=P, world=2, rank=1)
        assert proc0.global_token_start == 0
        assert proc1.global_token_start == 300

    def test_global_indices_include_first_frames(self):
        proc = self._make_processor(n_first=3, kfi=4)
        assert all(f in proc._global_set for f in [0, 1, 2])

    def test_global_indices_include_periodic(self):
        proc = self._make_processor(T=21, n_first=1, kfi=5)
        # Should include 0, 5, 10, 15, 20
        for f in [0, 5, 10, 15, 20]:
            assert f in proc._global_set

    def test_local_frames_cover_local_seq(self):
        """Sum of local frame token ranges should equal local_seq."""
        proc = self._make_processor(T=21, P=30, world=1)
        total_tokens = sum(q_e - q_s for _, q_s, q_e in proc._local_frames)
        assert total_tokens == proc.local_seq

    def test_local_frames_multi_gpu_partial(self):
        """With multi-GPU, frames may be partially owned."""
        T, P = 21, 30  # 630 tokens
        world = 2  # 315 per rank
        proc = self._make_processor(T=T, P=P, world=world, rank=0)
        # Frame 10 starts at token 300, frame 10 ends at 330.
        # Rank 0 has tokens [0, 315), so frame 10 is partially owned [300, 315)
        f10 = [(f, s, e) for f, s, e in proc._local_frames if f == 10]
        assert len(f10) == 1
        _, s, e = f10[0]
        assert e - s == 15  # only 15 of 30 tokens

    def test_window_ctx_keys_match_local_frames(self):
        proc = self._make_processor()
        local_frame_ids = {f for f, _, _ in proc._local_frames}
        assert set(proc._window_ctx.keys()) == local_frame_ids

    def test_window_ctx_excludes_globals(self):
        """Window context slices should not include global frames."""
        proc = self._make_processor(T=21, P=30, W=3, n_first=3, kfi=4)
        for f_global, parts in proc._window_ctx.items():
            for l_start, l_end in parts:
                # Convert back to global frame
                for tok in range(l_start, l_end):
                    global_tok = proc.global_token_start + tok
                    frame = global_tok // proc.num_patches
                    assert frame not in proc._global_set

    def test_global_frame_mask_shape(self):
        T = 21
        proc = self._make_processor(T=T)
        assert proc._global_frame_mask.shape == (T,)
        assert proc._global_frame_mask.dtype == torch.int8

    def test_global_frame_mask_values(self):
        proc = self._make_processor(T=21, n_first=3, kfi=4)
        for f in range(21):
            expected = 1 if f in proc._global_set else 0
            assert proc._global_frame_mask[f].item() == expected

    def test_window_bounds_shape(self):
        T = 21
        proc = self._make_processor(T=T)
        assert proc._window_bounds.shape == (T, 2)
        assert proc._window_bounds.dtype == torch.int32

    def test_window_bounds_valid_range(self):
        T = 21
        proc = self._make_processor(T=T, W=3)
        for f in range(T):
            lo = proc._window_bounds[f, 0].item()
            hi = proc._window_bounds[f, 1].item()
            assert 0 <= lo <= hi < T

    def test_index_tensors_consistency(self):
        """Global src/dst index tensors should have matching lengths."""
        proc = self._make_processor()
        assert len(proc._global_src_idx) == len(proc._global_dst_idx)
        assert len(proc._local_src_idx) == len(proc._local_dst_idx)

    def test_total_seq_divisibility_assertion(self):
        """Should raise AssertionError if total_seq not divisible by world."""
        with pytest.raises(AssertionError, match="must be divisible by"):
            WanDistributedLVSAProcessor(
                total_num_latent_frames=21,  # 21 * 30 = 630, not divisible by 4
                num_patches=30,
                window_size=3,
                n_first_frames=3,
                key_frame_interval=4,
                rank=0,
                world=4,
            )


# ── _expanded_window_bounds ──────────────────────────────────────────────────


class TestExpandedWindowBounds:
    """Tests for expanded window bounds that exclude global frames from count."""

    def _make_processor_for_expansion(self, T=21, W=3, n_first=3, kfi=4):
        return WanDistributedLVSAProcessor(
            total_num_latent_frames=T,
            num_patches=30,
            window_size=W,
            n_first_frames=n_first,
            key_frame_interval=kfi,
            rank=0,
            world=1,
            expand_window=True,
        )

    def test_expanded_window_contains_enough_nonglobal(self):
        """Expanded window should contain >= 2W+1 non-global frames (when possible)."""
        proc = self._make_processor_for_expansion(T=41, W=3, n_first=3, kfi=8)
        W = 3
        T = 41
        target = min(2 * W + 1, T - len(proc._global_indices))
        for f in range(T):
            lo, hi = proc._expanded_window_bounds(f, W, T)
            non_global = sum(1 for wf in range(lo, hi + 1) if wf not in proc._global_set)
            assert non_global >= target or (lo == 0 and hi == T - 1)

    def test_expanded_bounds_contain_query(self):
        proc = self._make_processor_for_expansion()
        T = 21
        for f in range(T):
            lo, hi = proc._expanded_window_bounds(f, 3, T)
            assert lo <= f <= hi

    def test_expanded_bounds_valid_range(self):
        proc = self._make_processor_for_expansion()
        T = 21
        for f in range(T):
            lo, hi = proc._expanded_window_bounds(f, 3, T)
            assert 0 <= lo <= hi < T


# ── set_window_size / set_step ────────────────────────────────────────────────


class TestDynamicReconfiguration:
    """Tests for set_window_size and set_step runtime reconfiguration."""

    def _make_processor(self, T=21, P=30, W=3, n_first=3, kfi=4):
        return WanDistributedLVSAProcessor(
            total_num_latent_frames=T,
            num_patches=P,
            window_size=W,
            n_first_frames=n_first,
            key_frame_interval=kfi,
            rank=0,
            world=1,
        )

    def test_set_window_size_zero_globals_only(self):
        """W=0 should make all window contexts empty (globals-only mode)."""
        proc = self._make_processor()
        proc.set_window_size(0)
        assert proc.window_size == 0
        for parts in proc._window_ctx.values():
            assert parts == []

    def test_set_window_size_restore(self):
        """Restoring original W should bring back non-empty window contexts."""
        # Use T=41 so restoring W=3 leaves plenty of non-global frames
        proc = self._make_processor(T=41, P=30, W=3, n_first=3, kfi=8)
        proc.set_window_size(0)
        # All window contexts should be empty with W=0
        assert all(len(parts) == 0 for parts in proc._window_ctx.values())
        proc.set_window_size(3)
        has_parts = any(len(parts) > 0 for parts in proc._window_ctx.values())
        assert has_parts

    def test_set_window_size_idempotent(self):
        """Setting the same W twice should not change state."""
        proc = self._make_processor(W=3)
        indices_before = list(proc._global_indices)
        proc.set_window_size(3)  # same as current, should be no-op
        assert proc._global_indices == indices_before

    def test_set_step_rotates_keyframes(self):
        """Different steps should produce different global index sets."""
        proc = self._make_processor(T=21, kfi=4)
        proc.set_step(0)
        indices0 = set(proc._global_indices)
        proc.set_step(1)
        indices1 = set(proc._global_indices)
        # Different offsets should produce different sets
        assert indices0 != indices1

    def test_set_step_same_offset_idempotent(self):
        """Steps with the same offset (mod kfi) should not rebuild."""
        proc = self._make_processor(kfi=4)
        proc.set_step(0)
        indices0 = list(proc._global_indices)
        proc.set_step(4)  # 4 % 4 = 0, same offset
        assert proc._global_indices == indices0

    def test_set_step_no_kfi_is_noop(self):
        proc = self._make_processor(kfi=None)
        proc.set_step(5)  # should not crash


# ── FlashInfer CSR builder ────────────────────────────────────────────────────


class TestBuildFlashinferCSR:
    """Tests for _build_flashinfer_csr."""

    def _make_processor(self, T=21, P=30, W=3, n_first=3, kfi=4, rank=0, world=1):
        return WanDistributedLVSAProcessor(
            total_num_latent_frames=T,
            num_patches=P,
            window_size=W,
            n_first_frames=n_first,
            key_frame_interval=kfi,
            rank=rank,
            world=world,
        )

    def test_csr_indptr_shape(self):
        proc = self._make_processor()
        # indptr has MB+1 entries
        assert len(proc._fi_indptr) == proc._fi_MB + 1

    def test_csr_indptr_monotonic(self):
        proc = self._make_processor()
        for i in range(len(proc._fi_indptr) - 1):
            assert proc._fi_indptr[i] <= proc._fi_indptr[i + 1]

    def test_csr_indptr_starts_at_zero(self):
        proc = self._make_processor()
        assert proc._fi_indptr[0].item() == 0

    def test_csr_indices_count(self):
        proc = self._make_processor()
        total_nnz = proc._fi_indptr[-1].item()
        assert len(proc._fi_indices) == total_nnz

    def test_csr_indices_in_compact_range(self):
        """All column indices should be in [0, compact_n)."""
        proc = self._make_processor()
        for idx in proc._fi_indices:
            assert 0 <= idx.item() < proc._fi_compact_n

    def test_copies_cover_all_compact_frames(self):
        """Global + local copies should account for all compact frames."""
        proc = self._make_processor()
        copy_count = len(proc._fi_global_copies) + len(proc._fi_local_copies)
        assert copy_count == proc._fi_compact_n

    def test_fi_M_and_N(self):
        P = 30
        proc = self._make_processor(P=P)
        assert proc._fi_M == proc._fi_MB * P
        assert proc._fi_N == proc._fi_compact_n * P

    def test_every_query_block_attends_to_globals(self):
        """Each query block row in the CSR should have at least num_global entries."""
        proc = self._make_processor()
        num_global = len(proc._global_indices)
        for qi in range(proc._fi_MB):
            row_start = proc._fi_indptr[qi].item()
            row_end = proc._fi_indptr[qi + 1].item()
            row_nnz = row_end - row_start
            assert row_nnz >= num_global

    def test_csr_rebuilt_on_window_size_change(self):
        """Changing window size should rebuild the CSR."""
        proc = self._make_processor(W=3)
        old_compact_n = proc._fi_compact_n
        proc.set_window_size(0)  # globals only
        # With W=0, compact_n should be <= old (fewer attended frames)
        assert proc._fi_compact_n <= old_compact_n


# ── Attention mask printing ──────────────────────────────────────────────────


class TestAttentionMask:
    """Tests for print_attention_mask and print_attention_mask_compact."""

    def _make_processor(self, T=10, P=4, W=2, n_first=2, kfi=3):
        return WanDistributedLVSAProcessor(
            total_num_latent_frames=T,
            num_patches=P,
            window_size=W,
            n_first_frames=n_first,
            key_frame_interval=kfi,
            rank=0,
            world=1,
        )

    def test_print_mask_runs_without_error(self, capsys):
        proc = self._make_processor()
        proc.print_attention_mask()
        captured = capsys.readouterr()
        assert "Q\\K" in captured.out
        assert "Legend" in captured.out

    def test_print_mask_compact_runs_without_error(self, capsys):
        proc = self._make_processor()
        proc.print_attention_mask_compact()
        captured = capsys.readouterr()
        assert "G=global" in captured.out


# ── _is_local_frame helper (inferred from _build_flashinfer_csr logic) ────────


class TestFrameOwnership:
    """Tests for frame-to-rank token mapping logic."""

    def test_rank0_owns_early_frames(self):
        T, P = 20, 30
        proc = WanDistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P,
            window_size=3, n_first_frames=3, key_frame_interval=4,
            rank=0, world=2,
        )
        # Rank 0 owns tokens [0, 300), which is frames 0-9
        local_frame_ids = {f for f, _, _ in proc._local_frames}
        assert 0 in local_frame_ids
        assert 9 in local_frame_ids

    def test_rank1_owns_later_frames(self):
        T, P = 20, 30
        proc = WanDistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P,
            window_size=3, n_first_frames=3, key_frame_interval=4,
            rank=1, world=2,
        )
        local_frame_ids = {f for f, _, _ in proc._local_frames}
        assert 10 in local_frame_ids
        assert 19 in local_frame_ids


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_frame(self):
        """T=1 should work (degenerate case)."""
        proc = WanDistributedLVSAProcessor(
            total_num_latent_frames=1, num_patches=30,
            window_size=0, n_first_frames=1, key_frame_interval=None,
            rank=0, world=1,
        )
        assert proc._global_indices == [0]
        assert len(proc._local_frames) == 1

    def test_window_larger_than_T(self):
        """W > T should still work without errors."""
        T = 5
        proc = WanDistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=30,
            window_size=10, n_first_frames=1, key_frame_interval=None,
            rank=0, world=1,
        )
        # Every frame should be in every window
        for f in range(T):
            lo, hi = proc._get_window_bounds(f, 10, T)
            assert lo == 0
            assert hi == T - 1

    def test_all_frames_global(self):
        """When kfi=1, every frame is global."""
        T = 10
        proc = WanDistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=30,
            window_size=0, n_first_frames=1, key_frame_interval=1,
            rank=0, world=1,
        )
        assert len(proc._global_indices) == T

    def test_kfi_equals_T(self):
        """kfi=T should produce minimal periodic keyframes."""
        T = 21
        proc = WanDistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=30,
            window_size=3, n_first_frames=3, key_frame_interval=T,
            rank=0, world=1,
        )
        # Only first frames + frame 0 (periodic with kfi=T → only frame 0)
        expected_periodic = {0}  # ceil(21/21)=1 keyframe: just frame 0
        expected = set(range(3)) | expected_periodic
        assert set(proc._global_indices) >= expected


# ── FlashInfer CSR edge cases ────────────────────────────────────────────────


class TestFlashinferCSREdgeCases:
    """Edge-case tests for _build_flashinfer_csr."""

    def _make_processor(self, T=21, P=30, W=3, n_first=3, kfi=4, rank=0, world=1):
        return WanDistributedLVSAProcessor(
            total_num_latent_frames=T,
            num_patches=P,
            window_size=W,
            n_first_frames=n_first,
            key_frame_interval=kfi,
            rank=rank,
            world=world,
        )

    def test_csr_last_frame_partial_ownership(self):
        """When the last frame is partially owned, CSR should still be valid."""
        T, P = 21, 30  # 630 tokens
        world = 2  # 315 tokens per rank — frame 10 partially owned by rank 0
        proc = self._make_processor(T=T, P=P, world=world, rank=0)
        # Basic CSR validity
        assert proc._fi_indptr[0].item() == 0
        assert proc._fi_indptr[-1].item() == len(proc._fi_indices)
        # All indices in range
        if len(proc._fi_indices) > 0:
            assert proc._fi_indices.max().item() < proc._fi_compact_n

    def test_csr_rank1_partial_first_frame(self):
        """Rank 1 partially owns its first frame — CSR should handle this."""
        T, P = 21, 30
        world = 2
        proc = self._make_processor(T=T, P=P, world=world, rank=1)
        assert proc._fi_indptr[0].item() == 0
        assert len(proc._fi_indptr) == proc._fi_MB + 1
        # Copies should cover all compact frames
        assert len(proc._fi_global_copies) + len(proc._fi_local_copies) == proc._fi_compact_n

    def test_csr_window_zero_globals_only(self):
        """W=0 (globals-only) should produce a valid CSR with only global frames."""
        T, P = 10, 4
        proc = self._make_processor(T=T, P=P, W=0, n_first=2, kfi=3)
        # All indices should reference only global frames in compact space
        assert proc._fi_compact_n == len(proc._global_indices)
        for qi in range(proc._fi_MB):
            row_start = proc._fi_indptr[qi].item()
            row_end = proc._fi_indptr[qi + 1].item()
            # Every row attends to the same global set
            assert row_end - row_start == len(proc._global_indices)

    def test_csr_single_frame(self):
        """T=1 degenerate case should produce valid CSR."""
        proc = self._make_processor(T=1, P=4, W=0, n_first=1, kfi=None)
        assert proc._fi_compact_n == 1
        assert proc._fi_indptr[-1].item() == 1  # one attended frame per Q block

    def test_csr_all_frames_global(self):
        """When all frames are global (kfi=1), compact should equal T."""
        T = 8
        proc = self._make_processor(T=T, P=4, W=0, n_first=1, kfi=1)
        assert proc._fi_compact_n == T
        assert len(proc._fi_local_copies) == 0  # all copies are global

    def test_csr_indices_sorted_per_row(self):
        """CSR column indices within each row should be sorted."""
        proc = self._make_processor(T=21, P=30, W=3, n_first=3, kfi=4)
        for qi in range(proc._fi_MB):
            row_start = proc._fi_indptr[qi].item()
            row_end = proc._fi_indptr[qi + 1].item()
            row_indices = proc._fi_indices[row_start:row_end].tolist()
            assert row_indices == sorted(row_indices)


# ── Multi-GPU mock tests ─────────────────────────────────────────────────────


class TestMultiGPUConsistency:
    """Tests that verify multi-GPU data structure consistency without actual GPUs."""

    def test_all_ranks_cover_full_sequence(self):
        """Token ranges across all ranks should cover the entire sequence with no gaps."""
        T, P, world = 20, 30, 4  # 600 tokens, 150 per rank
        all_tokens = set()
        for r in range(world):
            proc = WanDistributedLVSAProcessor(
                total_num_latent_frames=T, num_patches=P,
                window_size=3, n_first_frames=3, key_frame_interval=4,
                rank=r, world=world,
            )
            for _, q_s, q_e in proc._local_frames:
                for tok in range(q_s + proc.global_token_start, q_e + proc.global_token_start):
                    all_tokens.add(tok)
        assert all_tokens == set(range(T * P))

    def test_global_indices_same_across_ranks(self):
        """All ranks should agree on the global frame set."""
        T, P, world = 20, 30, 2
        procs = [
            WanDistributedLVSAProcessor(
                total_num_latent_frames=T, num_patches=P,
                window_size=3, n_first_frames=3, key_frame_interval=4,
                rank=r, world=world,
            )
            for r in range(world)
        ]
        for proc in procs[1:]:
            assert proc._global_set == procs[0]._global_set

    def test_boundary_guards_symmetric(self):
        """Boundary guards should be symmetric — same set seen from all ranks."""
        T, P, world = 20, 30, 2
        procs = [
            WanDistributedLVSAProcessor(
                total_num_latent_frames=T, num_patches=P,
                window_size=3, n_first_frames=3, key_frame_interval=4,
                rank=r, world=world,
            )
            for r in range(world)
        ]
        assert procs[0]._boundary_guards == procs[1]._boundary_guards

    def test_rotation_preserves_global_count_across_ranks(self):
        """After rotation, all ranks should have the same global count."""
        T, P, world = 20, 30, 2
        procs = [
            WanDistributedLVSAProcessor(
                total_num_latent_frames=T, num_patches=P,
                window_size=3, n_first_frames=3, key_frame_interval=4,
                rank=r, world=world,
            )
            for r in range(world)
        ]
        for step in range(4):
            for proc in procs:
                proc.set_step(step)
            assert len(procs[0]._global_indices) == len(procs[1]._global_indices)

    def test_four_gpu_csr_valid_per_rank(self):
        """CSR should be valid for each rank in a 4-GPU setup."""
        T, P, world = 20, 30, 4  # 600 tokens, 150 per rank
        for r in range(world):
            proc = WanDistributedLVSAProcessor(
                total_num_latent_frames=T, num_patches=P,
                window_size=3, n_first_frames=3, key_frame_interval=4,
                rank=r, world=world,
            )
            # CSR validity checks
            assert proc._fi_indptr[0].item() == 0
            assert proc._fi_indptr[-1].item() == len(proc._fi_indices)
            if len(proc._fi_indices) > 0:
                assert proc._fi_indices.max().item() < proc._fi_compact_n
            assert len(proc._fi_global_copies) + len(proc._fi_local_copies) == proc._fi_compact_n


# ── reference_frames propagation (regression for HV T<=ref bug) ────────────


class TestReferenceFramesPropagation:
    """Regression tests: the processor must receive and honor reference_frames
    so that globals-only mode covers 100% of frames at T <= ref.

    Historically, parallel.py did not pass reference_frames to the processor,
    so HunyuanVideo (ref=33) used Wan's default ref=21 and had accidental
    sparsity at T=33 even though the budget claimed full coverage.
    """

    def _make(self, T, ref, W=3, n_first=1):
        from lvsa.sparse_attention import compute_auto_kfi
        kfi = compute_auto_kfi(T, W, n_first, reference_frames=ref)
        return WanDistributedLVSAProcessor(
            total_num_latent_frames=T,
            num_patches=30,
            window_size=W,
            n_first_frames=n_first,
            key_frame_interval=kfi,
            rank=0, world=1,
            reference_frames=ref,
        )

    def test_hv_1x_globals_cover_all_frames(self):
        """T=33 = HV reference: all 33 frames should be globals."""
        proc = self._make(T=33, ref=33, W=3, n_first=1)
        assert proc.key_frame_interval == 1
        assert len(proc._metadata.global_set) == 33

    def test_hv_0_5x_globals_cover_all_frames(self):
        """T=17 < HV reference: all 17 frames should be globals."""
        proc = self._make(T=17, ref=33, W=3, n_first=1)
        assert proc.key_frame_interval == 1
        assert len(proc._metadata.global_set) == 17

    def test_set_window_size_zero_preserves_full_coverage(self):
        """Regression: globals-only mode (window=0) at T<=ref must keep 100%
        coverage. Previously set_window_size(0) recomputed kfi with default
        ref=21, collapsing globals to n_first anchors on HV.
        """
        proc = self._make(T=33, ref=33)
        proc.set_window_size(0)
        assert len(proc._metadata.global_set) == 33

    def test_set_sparsity_scale_preserves_reference(self):
        """Regression: set_sparsity_scale recomputes kfi and must also honor
        the stored reference_frames, not fall back to default 21."""
        proc = self._make(T=33, ref=33)
        proc.set_sparsity_scale(0.99)
        # scaled_ref = int(33*0.99) = 32, T=33 > scaled_ref → kfi > 1 path,
        # but globals should still be much more than the 1 n_first anchor.
        assert len(proc._metadata.global_set) > proc.n_first_frames

    def test_above_reference_unchanged(self):
        """T > ref: behaviour should be unchanged (sparsity by design)."""
        proc = self._make(T=65, ref=33, W=3, n_first=1)
        assert len(proc._metadata.global_set) < 65

    def test_wan_1x_still_works(self):
        """Wan at T=21 ref=21: all 21 frames should be globals."""
        proc = self._make(T=21, ref=21, W=3, n_first=1)
        assert proc.key_frame_interval == 1
        assert len(proc._metadata.global_set) == 21


def test_cp_mode_plumbing():
    import pytest
    from lvsa.lvsa_processor import DistributedLVSAProcessor
    common = dict(total_num_latent_frames=6, num_patches=2, window_size=1,
                  n_first_frames=1, key_frame_interval=1, rank=0, world=1)
    p_default = DistributedLVSAProcessor(**common)
    assert p_default._cp_mode == "custom"          # default = current behaviour
    p_uly = DistributedLVSAProcessor(cp_mode="ulysses", **common)
    assert p_uly._cp_mode == "ulysses"
    with pytest.raises(ValueError):
        DistributedLVSAProcessor(cp_mode="bogus", **common)


def test_ulysses_metadata_is_single_device():
    from lvsa.lvsa_processor import DistributedLVSAProcessor
    T, P = 24, 2          # 24 latent frames, 2 patches/frame
    p = DistributedLVSAProcessor(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        cp_mode="ulysses", reference_frames=6,   # 24 > 6 -> sparse
    )
    m = p._ulysses_metadata
    assert m.local_seq == T * P                 # full sequence, not a shard
    assert m.global_token_start == 0
    assert 0 < len(m.global_indices) < T        # genuinely sparse at 4x ref
    # custom mode must NOT build the ulysses metadata
    p_custom = DistributedLVSAProcessor(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=1,
        cp_mode="custom", reference_frames=6,
    )
    assert p_custom._ulysses_metadata is None


def test_ulysses_equals_custom_at_world1():
    import torch
    from lvsa.lvsa_processor import DistributedLVSAProcessor
    torch.manual_seed(0)
    T, P, H, D = 12, 4, 8, 16
    S = T * P; B = 1

    def make(mode):
        return DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P, window_size=1,
            n_first_frames=1, key_frame_interval=None, rank=0, world=1,
            cp_mode=mode, reference_frames=3,    # 12 > 3 -> sparse
        )
    q = torch.randn(B, S, H, D); k = torch.randn(B, S, H, D); v = torch.randn(B, S, H, D)

    pc = make("custom")
    kg, vg = pc._build_global_kv(k, v)
    out_custom = pc._compute_lvsa(q, k, v, kg, vg, None)
    out_uly = make("ulysses")._compute_lvsa_ulysses(q, k, v, None, None)

    assert out_uly.shape == out_custom.shape == (B, S, H, D)
    assert torch.allclose(out_uly, out_custom, atol=1e-5), \
        "ulysses world=1 path must equal the custom world=1 path"


def test_ulysses_metadata_rebuilds_on_rotation():
    from lvsa.lvsa_processor import DistributedLVSAProcessor
    T, P = 24, 2
    # Explicit kfi so the periodic keyframe grid is rotatable via set_step.
    p = DistributedLVSAProcessor(
        total_num_latent_frames=T, num_patches=P, window_size=1,
        n_first_frames=1, key_frame_interval=6, rank=0, world=1,
        cp_mode="ulysses", reference_frames=6,
    )
    assert p.key_frame_interval, "expected a keyframe interval to rotate"
    g0 = list(p._ulysses_metadata.global_indices)
    m0 = list(p._metadata.global_indices)
    # Pick a step that produces a non-zero offset so the keyframe grid shifts.
    p.set_step(1)
    g1 = list(p._ulysses_metadata.global_indices)
    m1 = list(p._metadata.global_indices)
    # The ulysses metadata must track the custom metadata in lockstep.
    assert g1 == m1, "ulysses metadata must mirror _metadata after rotation"
    assert (g1 != g0) or (m1 != m0), \
        "rotation must change both metadatas (or neither) — stale ulysses meta"
    # And they must still agree with each other after the rebuild.
    assert g0 == m0


def test_ulysses_equals_custom_with_encoder_world1():
    import torch
    from lvsa.lvsa_processor import DistributedLVSAProcessor
    torch.manual_seed(1)
    T, P, H, D = 10, 2, 4, 8
    S = T * P
    q = torch.randn(1, S, H, D); k = torch.randn(1, S, H, D); v = torch.randn(1, S, H, D)
    enc_k = torch.randn(1, 5, H, D); enc_v = torch.randn(1, 5, H, D)   # 5 text tokens

    def make(mode):
        return DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P, window_size=1,
            n_first_frames=1, key_frame_interval=None, rank=0, world=1,
            cp_mode=mode, reference_frames=3,
        )
    # custom: append enc to the (single-GPU) global K/V then compute
    pc = make("custom")
    kg, vg = pc._build_global_kv(k, v)
    kg = torch.cat([kg, enc_k], dim=1); vg = torch.cat([vg, enc_v], dim=1)
    out_custom = pc._compute_lvsa(q, k, v, kg, vg, None)
    out_uly = make("ulysses")._compute_lvsa_ulysses(q, k, v, enc_k, enc_v)
    assert torch.allclose(out_uly, out_custom, atol=1e-5)


def test_ulysses_validates_kv_head_count_under_gqa():
    """Ulysses CP all-to-alls BOTH Q and KV heads (scatter dim=2), so both must
    divide world — not just num_heads. Under GQA (num_kv_heads < num_heads) the
    KV count can be indivisible even when Q divides. The guard must reject it
    (and slice encoder K/V by the KV count). Fires before the all-to-all, so it
    is CPU-testable. PR #6 review (GQA head-count). All shipped ulysses-CP models
    are MHA today; this protects a future GQA model."""
    import torch
    from lvsa.lvsa_processor import DistributedLVSAProcessor
    p = DistributedLVSAProcessor(
        total_num_latent_frames=12, num_patches=3, window_size=1,
        n_first_frames=1, key_frame_interval=None, rank=0, world=3,
        cp_mode="ulysses", reference_frames=3,
    )
    S = 12 * 3
    q = torch.randn(1, S, 12, 8)          # 12 query heads — divisible by world=3
    k = torch.randn(1, S, 8, 8)           # 8 KV heads — NOT divisible by 3 (GQA)
    v = torch.randn(1, S, 8, 8)
    ek = torch.randn(1, 5, 8, 8); ev = torch.randn(1, 5, 8, 8)
    with pytest.raises(ValueError, match="num_kv_heads"):
        p._compute_lvsa_ulysses(q, k, v, ek, ev)


# ── FlashInfer ulysses path (GPU-only) ────────────────────────────────────────

def _flashinfer_gpu_unavailable():
    """True when the block-sparse FlashInfer kernel can't run here (no CUDA, or
    no AOT/JIT cache so plan() would invoke nvcc). Used to skip GPU-only tests."""
    if not torch.cuda.is_available():
        return True
    try:
        import flashinfer  # noqa: F401
    except Exception:
        return True
    return False


_FI_SKIP = pytest.mark.skipif(
    _flashinfer_gpu_unavailable(),
    reason="needs CUDA + a runnable FlashInfer (AOT/JIT cache)",
)


@_FI_SKIP
def test_ulysses_flashinfer_equals_custom_flashinfer_world1():
    """The flashinfer ulysses path (self._ufi / _ulysses_metadata) must match the
    flashinfer custom path (self._fi / _metadata) at world=1: both reduce to the
    exact single-device block-sparse pattern, so they share the same CSR."""
    torch.manual_seed(0)
    T, P, H, D = 12, 16, 8, 64   # D=64, P=16 -> FlashInfer-friendly block geometry
    S = T * P; B = 1
    dev = "cuda"

    def make(mode):
        p = DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P, window_size=1,
            n_first_frames=1, key_frame_interval=None, rank=0, world=1,
            cp_mode=mode, reference_frames=3,        # 12 > 3 -> sparse
        )
        p._use_flashinfer = True
        return p

    q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)

    pc = make("custom")
    kg, vg = pc._build_global_kv(k, v)
    out_custom = pc._compute_lvsa(q, k, v, kg, vg, None)
    out_uly = make("ulysses")._compute_lvsa_ulysses(q, k, v, None, None)

    assert out_uly.shape == out_custom.shape == (B, S, H, D)
    # Same CSR, same kernel, same inputs at world=1 -> bit-identical.
    assert torch.equal(out_uly, out_custom), \
        "ulysses-flashinfer world=1 must equal custom-flashinfer world=1"


@_FI_SKIP
def test_ulysses_flashinfer_equals_custom_flashinfer_with_encoder_world1():
    """Same world=1 flashinfer equivalence, dual-stream (encoder K/V appended)."""
    torch.manual_seed(1)
    T, P, H, D = 10, 16, 8, 64
    S = T * P; B = 1
    dev = "cuda"
    q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    enc_k = torch.randn(B, 5, H, D, device=dev, dtype=torch.bfloat16)   # 5 text tokens
    enc_v = torch.randn(B, 5, H, D, device=dev, dtype=torch.bfloat16)

    def make(mode):
        p = DistributedLVSAProcessor(
            total_num_latent_frames=T, num_patches=P, window_size=1,
            n_first_frames=1, key_frame_interval=None, rank=0, world=1,
            cp_mode=mode, reference_frames=3,
        )
        p._use_flashinfer = True
        return p

    pc = make("custom")
    kg, vg = pc._build_global_kv(k, v)
    kg = torch.cat([kg, enc_k], dim=1); vg = torch.cat([vg, enc_v], dim=1)
    out_custom = pc._compute_lvsa(q, k, v, kg, vg, None)
    out_uly = make("ulysses")._compute_lvsa_ulysses(q, k, v, enc_k, enc_v)

    assert torch.equal(out_uly, out_custom), \
        "ulysses-flashinfer world=1 (dual-stream) must equal custom-flashinfer"


# ── __call__ tests via mock adapter ──────────────────────────────────────────

import types
from lvsa.adapters.base import ModelAdapter


class _MinimalAdapter(ModelAdapter):
    """Minimal mock adapter for testing __call__ without a real model.

    hidden_states is [B, S, C] with C = H * D.  extract_qkv reshapes into
    [B, S, H, D] per head; output_projection flattens back to [B, S, C].
    """

    H = 2
    D = 8

    # ── Geometry (stubs — not called by __call__) ─────────────────────────────

    def patches_per_frame(self, height, width, pipe):
        return 0

    def latent_frames(self, num_frames, pipe):
        return 0

    def reference_latent_frames(self, pipe):
        return 0

    def install_processor(self, pipe, processor):
        return 0

    def setup_context_parallel(self, transformer, world):
        pass

    def patch_rotary_for_cp(self, rank, world):
        pass

    def extract_cross_attn_kv(self, attn, encoder_hidden_states):
        return None

    # ── Called by __call__ ────────────────────────────────────────────────────

    def extract_qkv(self, attn, hidden_states, encoder_hidden_states):
        """[B, S, C] -> three [B, S, H, D] tensors."""
        B, S, C = hidden_states.shape
        # C must equal H * D
        qkv = hidden_states.reshape(B, S, self.H, self.D)
        return qkv.clone(), qkv.clone(), qkv.clone()

    def apply_rotary(self, query, key, rotary_emb, local_seq, rank, world):
        return query, key

    def split_encoder_for_cross_attn(self, attn, encoder_hidden_states):
        return encoder_hidden_states, None

    def cross_attention(self, attn, query, encoder_image, attention_backend):
        return None

    def output_projection(self, attn, hidden_states, query_dtype):
        """[B, S, H, D] -> [B, S, H*D]."""
        B, S, H, D = hidden_states.shape
        return hidden_states.reshape(B, S, H * D).to(query_dtype)


def _make_call_processor(T=8, P=4, W=2, n_first=1, kfi=2, rank=0, world=1):
    """Build a processor with a mock adapter injected."""
    proc = WanDistributedLVSAProcessor(
        total_num_latent_frames=T,
        num_patches=P,
        window_size=W,
        n_first_frames=n_first,
        key_frame_interval=kfi,
        rank=rank,
        world=world,
    )
    proc.adapter = _MinimalAdapter()
    return proc


class TestCallSingleDevice:
    """Tests for __call__ branches reachable on CPU (world=1, no encoder)."""

    def test_call_single_device_lvsa(self):
        """Happy path: geometry OK, world=1 → LVSA SDPA runs, output shape is correct."""
        T, P, H, D = 8, 4, _MinimalAdapter.H, _MinimalAdapter.D
        C = H * D
        proc = _make_call_processor(T=T, P=P)
        hidden_states = torch.randn(1, T * P, C)
        attn = types.SimpleNamespace()

        out = proc(attn, hidden_states, encoder_hidden_states=None, rotary_emb=None)

        assert out.shape == (1, T * P, C), f"expected (1, {T*P}, {C}), got {out.shape}"

    def test_call_dense_fallback_on_geometry_mismatch(self):
        """Geometry mismatch (extra tokens) → dense fallback runs without error."""
        T, P, H, D = 8, 4, _MinimalAdapter.H, _MinimalAdapter.D
        C = H * D
        proc = _make_call_processor(T=T, P=P)
        # Extra tokens cause geometry mismatch (T*P + 7 ≠ T*P)
        S_bad = T * P + 7
        hidden_states = torch.randn(1, S_bad, C)
        attn = types.SimpleNamespace()

        out = proc(attn, hidden_states, encoder_hidden_states=None, rotary_emb=None)

        assert out.shape == (1, S_bad, C), f"expected (1, {S_bad}, {C}), got {out.shape}"

    def test_call_cp_geometry_mismatch_raises(self):
        """world=2 + geometry mismatch → NotImplementedError (CP dense fallback unsupported)."""
        # T=8, P=4, world=2 → expected local_seq = 8*4//2 = 16
        # Feed 8*4=32 tokens (full seq instead of shard) → mismatch
        T, P, H, D = 8, 4, _MinimalAdapter.H, _MinimalAdapter.D
        C = H * D
        proc = _make_call_processor(T=T, P=P, world=2, rank=0)
        # Feed T*P tokens; the world=2 proc expects T*P//world = 16 per rank
        hidden_states = torch.randn(1, T * P, C)  # 32 != 16 → mismatch
        attn = types.SimpleNamespace()

        with pytest.raises(NotImplementedError, match="geometry mismatch"):
            proc(attn, hidden_states, encoder_hidden_states=None, rotary_emb=None)


# ── GQA-safe FlashInfer plan (GPU-gated parity) ───────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer needs CUDA")
def test_processor_flashinfer_gqa_matches_sdpa():
    """The processor's FlashInfer block-sparse path must match SDPA under GQA
    (num_kv_heads < num_heads). Before the fix the plan was built with
    num_kv_heads=H and the compact KV buffer allocated at width H, so a GQA
    layout (here H=8 query heads, H_kv=2 KV heads) either errors in plan() or
    silently produces wrong numerics. Mirrors
    tests/test_cosmos3_processor.py::test_cosmos_fused_matches_sdpa but drives
    the processor's own FlashInfer entry (_compute_lvsa_flashinfer)."""
    from lvsa.sparse_attention import (
        LVSAMetadata, lvsa_sdpa, build_global_kv, compute_auto_kfi,
    )
    try:
        import flashinfer  # noqa: F401
    except Exception:
        pytest.skip("flashinfer not installed")

    torch.manual_seed(0)
    T, P, H, H_kv, D = 16, 64, 8, 2, 64        # GQA: 8 query heads, 2 KV heads
    dev, dt = "cuda", torch.bfloat16

    proc = WanDistributedLVSAProcessor(
        total_num_latent_frames=T,
        num_patches=P,
        window_size=2,
        n_first_frames=1,
        key_frame_interval=compute_auto_kfi(
            T, 2, 1, reference_frames=21, sparsity_scale=1.0,
        ),
        rank=0,
        world=1,
    )
    proc._use_flashinfer = True
    md = proc._metadata

    S_gen = T * P
    qg = torch.randn(1, S_gen, H, D, device=dev, dtype=dt)
    kg = torch.randn(1, S_gen, H_kv, D, device=dev, dtype=dt)
    vg = torch.randn(1, S_gen, H_kv, D, device=dev, dtype=dt)
    kglob, vglob = build_global_kv(kg, vg, md.global_indices, P)

    out_sdpa = lvsa_sdpa(qg, kg, vg, kglob, vglob, md).float()
    out_fi = proc._compute_lvsa_flashinfer(
        qg, kg, vg, kglob, vglob, md, proc._fi,
    ).float()

    max_diff = (out_sdpa - out_fi).abs().max().item()
    assert max_diff < 2e-2, f"processor FlashInfer vs SDPA max abs diff {max_diff}"
