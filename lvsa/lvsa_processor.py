"""
lvsa_processor.py — Block-sparse Sliding Window Attention processor for context-parallel inference.

Model-agnostic: all model-specific logic (QKV projections, RoPE format,
output projection, cross-attention) is delegated to a ``ModelAdapter``
instance provided at construction time.

The stateless attention math lives in ``sparse_attention.py``.  This module
provides the stateful wrapper that manages communication (all-reduce),
buffer allocation, KV caching, and adapter interaction.
"""

from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

from .sparse_attention import (
    LVSAMetadata,
    adaptive_window_bounds,
    build_global_kv,
    compute_auto_kfi,
    compute_boundary_guard_frames,
    compute_global_indices,
    expanded_window_bounds,
    get_window_bounds,
    lvsa_sdpa,
)

try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

from .adapters.base import ModelAdapter


def _expand_chunk_copies_to_indices(
    copies: list[tuple[int, int]],
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a list of (src_start, dst_start) chunk-offset tuples into
    flat per-token src/dst index tensors.

    Each tuple describes a contiguous copy of ``chunk_size`` tokens from
    ``src_start`` to ``dst_start``. The result tensors index each token
    individually, suitable for ``buf[..., dst_idx] = src_buf[..., src_idx]``
    vectorized advanced-indexing.

    Returns
    -------
    src_idx, dst_idx : long tensors of shape ``[len(copies) * chunk_size]``
        on ``device``. Empty tensors (shape ``[0]``) if ``copies`` is empty.
    """
    if not copies:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone()
    src_starts = torch.tensor([s for s, _ in copies], dtype=torch.long)
    dst_starts = torch.tensor([d for _, d in copies], dtype=torch.long)
    offsets = torch.arange(chunk_size, dtype=torch.long)
    src = (src_starts[:, None] + offsets[None, :]).reshape(-1).to(device)
    dst = (dst_starts[:, None] + offsets[None, :]).reshape(-1).to(device)
    return src, dst


def _ulysses_all_to_all(
    x: torch.Tensor, scatter_dim: int, gather_dim: int, world: int, group=None
) -> torch.Tensor:
    """Ulysses all-to-all on a [B, S, H, D] tensor (inference-only, no autograd).

    Gather  (scatter_dim=2, gather_dim=1): [B, S_local, H, D] -> [B, S_full, H/world, D]
    Scatter (scatter_dim=1, gather_dim=2): [B, S_full, H/world, D] -> [B, S_local, H, D]
    """
    import torch.distributed as dist
    parts = [t.contiguous() for t in x.chunk(world, dim=scatter_dim)]
    out = [torch.empty_like(parts[0]) for _ in range(world)]
    dist.all_to_all(out, parts, group=group)
    return torch.cat(out, dim=gather_dim).contiguous()


# Query-chunk size for ``_attn_with_lse``. The score matrix is ``[B,H,q_tile,Sk]``
# rather than the full ``[B,H,Sq,Sk]`` — at video scale Sq=Sk can be ~33k, so the
# untiled matrix is ~48 GiB (OOM). Tiling caps it to ``O(q_tile × Sk)``. Queries
# are independent, so per-tile (out, lse) just concatenate along Sq → exact.
_RING_Q_TILE = 1024


def _attn_with_lse(q, k, v, attn_mask=None, scale=None, q_tile=_RING_Q_TILE):
    """Masked attention returning ``(output, log-sum-exp)`` — the online-softmax
    primitive for the ring path.

    q: ``[B, H, Sq, D]``; k/v: ``[B, H, Sk, D]`` (GQA: expand K/V to H first).
    ``attn_mask``: optional bool ``[Sq, Sk]`` (True = attend). Rows that are
    all-False yield ``lse = -inf`` and a zero output for that query, so an empty
    block-pair contributes nothing under ``_merge_out_lse`` (callers may skip
    such pairs entirely — see ``ring_block_frame_mask`` returning ``None``).

    Computed in query tiles of ``q_tile`` to bound memory at video scale (the
    untiled ``[Sq,Sk]`` scores OOM). Math is identical to the untiled form: each
    query attends to all of ``k`` independently, so chunking Sq and concatenating
    is exact. At small (test) sizes ``Sq <= q_tile`` → a single tile.

    Returns ``out [B, H, Sq, D]``, ``lse [B, H, Sq]`` (both float32 for a stable
    merge regardless of input dtype).
    """
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    B, H, Sq, D = q.shape
    kt = k.float().transpose(-1, -2)                          # [B,H,D,Sk]
    vf = v.float()
    out = torch.empty((B, H, Sq, D), dtype=torch.float32, device=q.device)
    lse = torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
    for s in range(0, Sq, q_tile):
        e = min(s + q_tile, Sq)
        scores = (q[:, :, s:e, :].float() @ kt) * scale      # [B,H,tile,Sk]
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask[s:e, :], float("-inf"))
        l = torch.logsumexp(scores, dim=-1)                  # [B,H,tile]; -inf if all-masked
        a = torch.nan_to_num(torch.exp(scores - l.unsqueeze(-1)), nan=0.0)
        out[:, :, s:e, :] = a @ vf
        lse[:, :, s:e] = l
    return out, lse


def _merge_out_lse(out_a, lse_a, out_b, lse_b):
    """Online-softmax combine of two attention results over DISJOINT key sets.

    First call: pass ``out_a=None`` (and ``lse_a`` is ignored) to initialize.
    Handles ``-inf`` lse (empty block) via ``nan_to_num`` on the reweighted terms.
    Returns ``(merged_out, merged_lse)``.
    """
    if out_a is None:
        return out_b, lse_b
    new_lse = torch.logaddexp(lse_a, lse_b)                   # [B,H,Sq]
    a = torch.exp(lse_a - new_lse).unsqueeze(-1)
    b = torch.exp(lse_b - new_lse).unsqueeze(-1)
    merged = torch.nan_to_num(out_a * a, nan=0.0) + torch.nan_to_num(out_b * b, nan=0.0)
    return merged, new_lse


def _all_gather_seq(x, world):
    """All-gather a per-rank sequence shard ``[B, local_seq, H, D]`` into the full
    ``[B, full_seq, H, D]`` on every rank. CP shards the sequence contiguously
    (rank r owns frames ``[r·T/world, (r+1)·T/world)``), so the rank-ordered
    concatenation reconstructs the original sequence order.
    """
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=1)


class _FIState:
    """Mutable runtime caches for one FlashInfer block-sparse plan.

    The processor keeps two independent instances: ``self._fi`` for the
    ``custom`` per-rank metadata and ``self._ufi`` for the ``ulysses``
    full-grid metadata. Each owns its own planned wrapper + compact/Q buffers
    because the two metadatas have different CSRs and compact shapes. Both are
    reset (``planned=False``, buffers dropped) whenever the pattern rebuilds
    (e.g. rotating keyframes); the 128 MB ``workspace`` scratch is kept and
    reused across rebuilds.
    """

    __slots__ = (
        "wrapper", "workspace", "planned",
        "compact_k", "compact_v", "q_pad", "enc_tokens", "N_actual",
    )

    def __init__(self) -> None:
        self.workspace = None  # 128 MB scratch, allocated once on first plan()
        self.reset()

    def reset(self) -> None:
        """Drop the plan + per-pattern buffers (keep the workspace scratch)."""
        self.wrapper = None
        self.planned = False
        self.compact_k = None
        self.compact_v = None
        self.q_pad = None
        self.enc_tokens = 0
        self.N_actual = 0


class DistributedLVSAProcessor:
    """
    Block-sparse attention processor combining sliding window attention (LVSA),
    automatic keyframes, and multi-GPU context parallelism.

    **Model-agnostic**: all model-specific logic is delegated to a
    :class:`~lvsa.adapters.base.ModelAdapter` instance.

    Attention pattern (global frame space)
    ----------------------------------------
    Each query token at global position i belongs to frame (i // P).
    That token attends to:
      1. Global frames   : {0 … n_first_frames-1}  ∪  {0, kfi, 2*kfi, …}
         K/V gathered via all_reduce(SUM) in global_broadcast mode.
      2. Local window    : frames in [f-W, f+W] that have tokens on THIS rank
         and are not already in the global set.

    Why token-based instead of frame-based
    ----------------------------------------
    CP splits the sequence as seq_len // world tokens per rank.
    Typical video models produce latent frame counts that are not divisible
    by common world_size values (e.g. Wan's 4k+1 grid gives 21, 41, 61…).
    Reshaping to [B, local_frames, P, H, D] would therefore be invalid.

    We instead work purely in token space: for each local token i its global
    frame is  (global_token_start + i) // P,  where
    global_token_start = rank * local_seq.  This is correct for any T_lat
    and any world_size with zero constraints on their relationship.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        total_num_latent_frames: int,
        num_patches: int,
        window_size: int,
        n_first_frames: int,
        key_frame_interval: Optional[int],
        rank: int,
        world: int,
        cp_mode: str = "custom",
        expand_window: bool = True,
        adapter: Optional[ModelAdapter] = None,
        sparsity_scale: float = 1.0,
        reference_frames: int = 21,
    ) -> None:
        self.total_num_frames = total_num_latent_frames
        self.num_patches = num_patches
        self.window_size = window_size
        self.n_first_frames = n_first_frames
        self.rank = rank
        self.world = world
        if cp_mode not in ("custom", "ulysses", "ring"):
            raise ValueError(
                f"cp_mode must be 'custom', 'ulysses', or 'ring', got {cp_mode!r}"
            )
        self._cp_mode = cp_mode
        self._num_patches = num_patches
        self._expand_window = expand_window
        self._sparsity_scale = sparsity_scale
        self._reference_frames = reference_frames

        # Model adapter — if None, a WanAdapter is created lazily for
        # backwards compatibility.
        if adapter is None:
            from .adapters.wan import WanAdapter
            adapter = WanAdapter()
        self.adapter: ModelAdapter = adapter

        # ── Boundary guards (fixed across rotations) ──
        self._boundary_guards = compute_boundary_guard_frames(
            total_num_latent_frames,
            total_num_latent_frames * num_patches // world,
            num_patches, world, window_size,
        )
        self._current_offset = 0

        # ── Build all pattern metadata via LVSAMetadata ──
        self._metadata = LVSAMetadata.build(
            total_latent_frames=total_num_latent_frames,
            num_patches=num_patches,
            window_size=window_size,
            n_first_frames=n_first_frames,
            key_frame_interval=key_frame_interval,
            rank=rank,
            world=world,
            expand_window=expand_window,
            keyframe_offset=0,
            boundary_guards=self._boundary_guards,
            reference_frames=reference_frames,
            sparsity_scale=sparsity_scale,
        )
        self._copy_metadata_to_self()

        # Plain-Ulysses metadata: the all-to-all reconstructs the FULL frame grid
        # on each rank (head-sharded), so the sparse pattern is the single-device
        # one — world=1, no boundary guards. Built only when that mode is active.
        self._ulysses_metadata = None
        if self._cp_mode == "ulysses":
            self._ulysses_metadata = LVSAMetadata.build(
                total_latent_frames=total_num_latent_frames,
                num_patches=num_patches,
                window_size=window_size,
                n_first_frames=n_first_frames,
                key_frame_interval=key_frame_interval,
                rank=0,
                world=1,
                expand_window=expand_window,
                keyframe_offset=0,
                boundary_guards=[],
                reference_frames=reference_frames,
                sparsity_scale=sparsity_scale,
            )

        # Ring-CP metadata: like ulysses, the ring rotation reconstructs the
        # single-device pattern (each query attends to the full global ∪ window
        # set, just over rotating K/V blocks instead of a gathered grid). So the
        # ring path reads the SINGLE-DEVICE global set (rank=0, world=1, no
        # boundary guards) — boundary guards are a custom-mode seam fix that ring
        # does not need (a global block is attended whenever it rotates past).
        self._ring_metadata = None
        if self._cp_mode == "ring":
            self._ring_metadata = LVSAMetadata.build(
                total_latent_frames=total_num_latent_frames,
                num_patches=num_patches,
                window_size=window_size,
                n_first_frames=n_first_frames,
                key_frame_interval=key_frame_interval,
                rank=0,
                world=1,
                expand_window=expand_window,
                keyframe_offset=0,
                boundary_guards=[],
                reference_frames=reference_frames,
                sparsity_scale=sparsity_scale,
            )

        # ── Ring per-block-pair caches ──────────────────────────────────────
        # The per-block-pair mask/CSR is identical across all ~N transformer
        # layers within a denoising step, so build it ONCE and reuse — not ~N×.
        # (Rebuilding the ~1 GB dense mask every layer was the real bottleneck,
        # dwarfing even the fast flashinfer attention.) Cleared on keyframe
        # rotation in ``_rebuild_for_current_params``. Benefits BOTH backends.
        self._ring_mask_cache: dict = {}   # src -> GPU bool mask (or False if empty)
        self._ring_fi_cache: dict = {}     # src -> flashinfer BlockSparse wrapper (or False)
        self._ring_workspaces: dict = {}   # src -> persistent 128 MB flashinfer scratch

        # ── Cached buffers (reused across calls, avoids repeated allocation) ──
        self._kv_buf: Optional[torch.Tensor] = None
        self._kv_reduce_work: Optional[dist.Work] = None
        self._unified_buf: Optional[torch.Tensor] = None
        self._bsa_compact_buf: Optional[torch.Tensor] = None
        self._bsa_dispatch_cache: Optional[dict] = None
        self._use_flashinfer = False     # enabled only via --flashinfer flag

        # ── FlashInfer block-sparse runtime caches ──
        # Two independent states: custom per-rank metadata vs ulysses full-grid
        # metadata (different CSRs + compact shapes -> separate plans/buffers).
        self._fi = _FIState()    # custom path (self._metadata)
        self._ufi = _FIState()   # ulysses path (self._ulysses_metadata)

        # Device-cached tensor refs (lazily populated)
        self._global_frame_mask_device: Optional[torch.Tensor] = None
        self._window_bounds_device: Optional[torch.Tensor] = None
        self._attended_indices_device: Optional[torch.Tensor] = None

        P = num_patches
        gts = self.global_token_start
        first_frame = gts // P
        last_frame = (gts + self.local_seq - 1) // P

        if rank == 0:
            kfi_str = (
                str(self.key_frame_interval) if self.key_frame_interval else "none"
            )
            # Compute attended count for a sample frame (middle of rank 0)
            mid_f = (first_frame + last_frame) // 2
            mid_lo, mid_hi = self._get_window_bounds(
                mid_f, window_size, total_num_latent_frames
            )
            mid_win_set = set(range(mid_lo, mid_hi + 1))
            mid_attended = len(self._global_set | mid_win_set)
            print(
                f"[LVSA] total_lat_frames={total_num_latent_frames}  "
                f"local_seq={self.local_seq}  "
                f"rank0_frames~[{first_frame},{last_frame}]  "
                f"window={window_size}  n_first={n_first_frames}  "
                f"kfi={kfi_str}  "
                f"global_count={len(self._global_indices)}  "
                f"attended_per_frame={mid_attended}/{total_num_latent_frames}"
            )
            if self._cp_mode == "ulysses" and self._ulysses_metadata is not None:
                # The line above reflects the per-rank "custom" metadata, which the
                # ulysses path does NOT use. The ulysses path all-to-all gathers the
                # FULL grid and runs the single-device pattern: fewer global anchors
                # (no boundary guards) than custom's seam-closing inflation.
                print(
                    f"[LVSA] cp_mode=ulysses: runs the single-device pattern on the "
                    f"full grid via all-to-all -> "
                    f"global_count={len(self._ulysses_metadata.global_indices)} "
                    f"(vs custom {len(self._global_indices)} above; no boundary guards)"
                )

    def _copy_metadata_to_self(self) -> None:
        """Copy LVSAMetadata fields to self for backward compatibility.

        Existing code and tests access attributes like ``self._global_indices``,
        ``self._fi_indptr``, etc. directly.  This method bridges the new
        ``LVSAMetadata`` dataclass with the old attribute-based interface.
        """
        m = self._metadata
        self.local_seq = m.local_seq
        self.global_token_start = m.global_token_start
        self.key_frame_interval = m.key_frame_interval
        self._global_indices = m.global_indices
        self._global_set = m.global_set
        self._local_frames = m.local_frames
        self._window_ctx = m.window_ctx
        self._global_frame_mask = m.global_frame_mask
        self._window_bounds = m.window_bounds
        self._attended_indices = m.attended_indices
        self._attended_C = m.attended_C
        self._global_src_idx = m.global_src_idx
        self._global_dst_idx = m.global_dst_idx
        self._local_src_idx = m.local_src_idx
        self._local_dst_idx = m.local_dst_idx
        self._fi_indptr = m.fi_indptr
        self._fi_indices = m.fi_indices
        self._fi_M = m.fi_M
        self._fi_N = m.fi_N
        self._fi_MB = m.fi_MB
        self._fi_compact_n = m.fi_compact_n
        self._fi_global_copies = m.fi_global_copies
        self._fi_local_copies = m.fi_local_copies

    def print_attention_mask(self) -> None:
        """Print a visual T×T attention matrix showing which frames each query
        frame attends to.  Legend: G=global, W=window, .=not attended."""
        T = self.total_num_frames
        W = self.window_size

        # Header: frame indices
        # Use 2-char columns for readability
        hdr = "Q\\K " + "".join(f"{k:>3}" for k in range(T))
        print(hdr)
        print("    " + "---" * T)

        counts = []
        for f in range(T):
            win_lo, win_hi = self._get_window_bounds(f, W, T)
            win_set = set(range(win_lo, win_hi + 1))
            attended = self._global_set | win_set

            row_chars = []
            for k in range(T):
                if k in self._global_set and k in win_set:
                    row_chars.append("  X")  # both global and window
                elif k in self._global_set:
                    row_chars.append("  G")
                elif k in win_set:
                    row_chars.append("  W")
                else:
                    row_chars.append("  .")
            counts.append(len(attended))
            print(f"{f:>3} |" + "".join(row_chars) + f"  | {len(attended)}/{T}")

        print("    " + "---" * T)
        min_c, max_c = min(counts), max(counts)
        avg_c = sum(counts) / len(counts)
        print(
            f"Legend: G=global, W=window, X=both  |  "
            f"attended min={min_c} max={max_c} avg={avg_c:.1f}"
        )

    def print_attention_mask_compact(self) -> None:
        """Compact 1-char-per-column attention mask for narrow terminals.
        Delegates to the free function in ``lvsa.sparse_attention``.
        """
        from .sparse_attention import print_attention_mask_compact as _print
        _print(
            total_frames=self.total_num_frames,
            window_size=self.window_size,
            global_set=self._global_set,
            expand_window=self._expand_window,
        )

    # ── Step-wise keyframe rotation ────────────────────────────────────────────

    def set_window_size(self, new_window_size: int) -> None:
        """Dynamically switch window size (e.g. between windowed and globals-only).

        When *new_window_size* is 0 every frame sees the same set of global
        anchors — equivalent to full attention at *reference_frames* budget.
        When restored to the original value, normal LVSA resumes.

        Recomputes kfi for the new budget split and rebuilds all dependent
        data structures.
        """
        if new_window_size == self.window_size:
            return

        self.window_size = new_window_size

        # Recompute kfi for the new window/global budget split
        self.key_frame_interval = compute_auto_kfi(
            self.total_num_frames,
            self.window_size,
            self.n_first_frames,
            reference_frames=self._reference_frames,
            sparsity_scale=self._sparsity_scale,
        )

        # Force full rebuild (reset offset so _rebuild recalculates)
        old_offset = self._current_offset
        self._current_offset = -1  # sentinel to force rebuild
        self._rebuild_for_current_params(old_offset)

    def set_step(self, step_idx: int) -> None:
        """Rotate periodic keyframes for this denoising step.

        The periodic keyframes shift by ``step_idx % key_frame_interval``
        positions each step, so over multiple steps every frame gets a turn
        as a global anchor.  n_first_frames and boundary guards stay fixed.
        """
        if not self.key_frame_interval:
            return  # no periodic keyframes to rotate

        offset = step_idx % self.key_frame_interval
        if offset == self._current_offset:
            return  # same pattern — skip recomputation

        self._rebuild_for_current_params(offset)

    def set_sparsity_scale(self, new_sparsity_scale: float) -> None:
        """Dynamically adjust the sparsity scale.

        *sparsity_scale* < 1.0 → more sparse (fewer attended frames).
        *sparsity_scale* > 1.0 → less sparse (more attended frames).
        Default 1.0 preserves the original behaviour.

        Recomputes kfi for the new budget and rebuilds all dependent
        data structures.
        """
        if new_sparsity_scale == self._sparsity_scale:
            return

        self._sparsity_scale = new_sparsity_scale

        # Recompute kfi for the new sparsity budget
        self.key_frame_interval = compute_auto_kfi(
            self.total_num_frames,
            self.window_size,
            self.n_first_frames,
            reference_frames=self._reference_frames,
            sparsity_scale=self._sparsity_scale,
        )

        # Force full rebuild (reset offset so _rebuild recalculates)
        old_offset = self._current_offset
        self._current_offset = -1  # sentinel to force rebuild
        self._rebuild_for_current_params(old_offset)

    def _rebuild_for_current_params(self, offset: int) -> None:
        """Recompute all derived data structures for the current window_size,
        key_frame_interval, and the given keyframe offset.

        Called by both ``set_step()`` (rotation) and ``set_window_size()``.
        Delegates to ``LVSAMetadata.build()``.
        """
        self._current_offset = offset

        # Wait for any in-flight async all-reduce before clearing buffers
        if self._kv_reduce_work is not None:
            self._kv_reduce_work.wait()
            self._kv_reduce_work = None

        # Rebuild all pattern metadata
        self._metadata = LVSAMetadata.build(
            total_latent_frames=self.total_num_frames,
            num_patches=self.num_patches,
            window_size=self.window_size,
            n_first_frames=self.n_first_frames,
            key_frame_interval=self.key_frame_interval,
            rank=self.rank,
            world=self.world,
            expand_window=self._expand_window,
            keyframe_offset=offset,
            boundary_guards=self._boundary_guards,
            reference_frames=self._reference_frames,
            sparsity_scale=self._sparsity_scale,
        )
        self._copy_metadata_to_self()

        # Keep the plain-Ulysses metadata in lockstep with the rotated pattern.
        # It mirrors _metadata but is always single-device (rank=0, world=1, no
        # boundary guards), since the all-to-all reconstructs the full grid on
        # every rank. Without this it goes stale under --rotate-keyframes.
        if self._cp_mode == "ulysses":
            self._ulysses_metadata = LVSAMetadata.build(
                total_latent_frames=self.total_num_frames,
                num_patches=self.num_patches,
                window_size=self.window_size,
                n_first_frames=self.n_first_frames,
                key_frame_interval=self.key_frame_interval,
                rank=0,
                world=1,
                expand_window=self._expand_window,
                keyframe_offset=offset,
                boundary_guards=[],
                reference_frames=self._reference_frames,
                sparsity_scale=self._sparsity_scale,
            )

        # Keep the ring metadata's single-device global set in lockstep with the
        # rotated pattern (same reasoning as ulysses above).
        if self._cp_mode == "ring":
            self._ring_metadata = LVSAMetadata.build(
                total_latent_frames=self.total_num_frames,
                num_patches=self.num_patches,
                window_size=self.window_size,
                n_first_frames=self.n_first_frames,
                key_frame_interval=self.key_frame_interval,
                rank=0,
                world=1,
                expand_window=self._expand_window,
                keyframe_offset=offset,
                boundary_guards=[],
                reference_frames=self._reference_frames,
                sparsity_scale=self._sparsity_scale,
            )

        # Force re-upload of device tensors
        self._global_frame_mask_device = None
        self._window_bounds_device = None
        self._attended_indices_device = None

        # Reset cached KV buffers (global count may change)
        self._kv_buf = None

        # Reset FlashInfer state (both paths; keeps the workspace scratch)
        self._fi.reset()
        self._ufi.reset()

        # Ring masks/CSRs depend on the (now-rebuilt) global_set → invalidate.
        # Keep _ring_workspaces (persistent scratch; wrappers re-plan onto them).
        self._ring_mask_cache = {}
        self._ring_fi_cache = {}

    # ── Static helpers (delegators to sparse_attention module functions) ────────

    def _get_window_bounds(self, f: int, W: int, T: int) -> Tuple[int, int]:
        """Dispatch to expanded or adaptive window bounds based on config."""
        return get_window_bounds(
            f, W, T, self._expand_window, self._global_set, len(self._global_indices),
        )

    @staticmethod
    def _adaptive_window_bounds(f: int, W: int, T: int) -> Tuple[int, int]:
        return adaptive_window_bounds(f, W, T)

    def _expanded_window_bounds(self, f: int, W: int, T: int) -> Tuple[int, int]:
        return expanded_window_bounds(
            f, W, T, self._global_set, len(self._global_indices),
        )

    @staticmethod
    def _compute_boundary_guard_frames(
        total_frames: int, local_seq: int, num_patches: int,
        world: int, window_size: int,
    ) -> List[int]:
        return compute_boundary_guard_frames(
            total_frames, local_seq, num_patches, world, window_size,
        )

    @staticmethod
    def _compute_auto_kfi(
        total_frames: int, window_size: int, n_first_frames: int,
        reference_frames: int = 21, sparsity_scale: float = 1.0,
    ) -> int:
        return compute_auto_kfi(
            total_frames, window_size, n_first_frames, reference_frames,
            sparsity_scale,
        )

    @staticmethod
    def _compute_global_indices(
        total_frames: int, n_first_frames: int,
        key_frame_interval: Optional[int], offset: int = 0,
    ) -> List[int]:
        return compute_global_indices(
            total_frames, n_first_frames, key_frame_interval, offset,
        )

    # ── Main forward ──────────────────────────────────────────────────────────

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Any] = None,
        image_rotary_emb: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Distributed block-sparse self-attention.

        hidden_states        : [B, local_seq, C]   — already sharded by CP hook
        encoder_hidden_states: [B, text_len, C]    — full text, replicated
        rotary_emb           : Wan-style RoPE (cos, sin) tuple
        image_rotary_emb     : HunyuanVideo-style RoPE tensor

        All model-specific operations (QKV extraction, RoPE, output projection,
        cross-attention) are delegated to ``self.adapter``.

        Returns a single tensor for single-stream models (Wan) or a tuple
        ``(hidden_states, encoder_hidden_states)`` for dual-stream (HunyuanVideo).
        """
        adapter = self.adapter

        # ── Unify rotary_emb from either parameter ────────────────────────────
        # Wan passes rotary_emb directly; HunyuanVideo passes image_rotary_emb.
        if rotary_emb is None and image_rotary_emb is not None:
            rotary_emb = image_rotary_emb

        # ── Split encoder states for cross-attention (if applicable) ──────────
        encoder_hidden_states_img = None
        if encoder_hidden_states is not None:
            encoder_hidden_states, encoder_hidden_states_img = (
                adapter.split_encoder_for_cross_attn(attn, encoder_hidden_states)
            )

        # ── Q / K / V projections for video tokens (model-specific) ──────────
        query, key, value = adapter.extract_qkv(
            attn, hidden_states, encoder_hidden_states,
        )
        query_dtype = query.dtype

        # ── Dual-stream encoder projections (if applicable) ──────────────────
        # For models like HunyuanVideo with separate add_q/k/v_proj for
        # encoder tokens, project them separately.  Encoder K/V becomes
        # extra global context for LVSA; encoder Q gets full attention.
        enc_q = enc_k = enc_v = None
        encoder_seq_len = 0
        if encoder_hidden_states is not None:
            enc_qkv = adapter.extract_encoder_qkv(attn, encoder_hidden_states)
            if enc_qkv is not None:
                enc_q, enc_k, enc_v = enc_qkv
                encoder_seq_len = enc_q.shape[1]

        # ── Rotary embeddings (model-specific) ────────────────────────────────
        # RoPE applies only to video tokens — adapter.apply_rotary handles this.
        local_seq = hidden_states.shape[1]
        query, key = adapter.apply_rotary(
            query, key, rotary_emb, local_seq, self.rank, self.world,
        )

        # ── Geometry guard ───────────────────────────────────────────────────
        # LVSA's window/keyframe math assumes the self-attention sequence is a
        # frame grid: ``local_seq == (total_num_frames × num_patches) // world``,
        # with tokens grouped by frame (a token's frame is ``token // P``). If a
        # model feeds a sequence that does not match — e.g. prepended history
        # tokens (Helios chunked-AR), a non-frame-structured layout, or an
        # unexpected resolution — the per-frame windowing silently MISaligns and
        # corrupts the output (no error). Detect that and fall back to dense full
        # attention so a geometry mismatch degrades safely instead of silently.
        geometry_ok = self._geometry_matches(local_seq)

        if geometry_ok:
            if self._cp_mode == "ulysses":
                # ── Plain-Ulysses CP ─────────────────────────────────────────
                # All-to-all reconstructs the full frame grid (head-sharded) on
                # each rank, then the single-device LVSA pattern runs and the
                # result is scattered back. world==1 is pure single-device.
                hidden_states = self._compute_lvsa_ulysses(
                    query, key, value, enc_k, enc_v,
                )
            elif self._cp_mode == "ring":
                # ── Ring CP (sequence-sharded, true ring rotation) ───────────
                # Rotate K/V around the ring, attend the local Q to each incoming
                # block under the LVSA frame-mask, accumulate via online-softmax.
                # No num_heads%world constraint (shards sequence, not heads).
                hidden_states = self._compute_lvsa_ring(
                    query, key, value, enc_k, enc_v,
                )
            else:
                # ── Custom CP (sequence-sharded, global-broadcast) ───────────
                # Start global KV build — the all-reduce runs asynchronously so
                # we can overlap communication with the encoder KV concat below.
                k_global, v_global = self._build_global_kv(key, value)

                # ── While the all-reduce is in flight, prepare encoder KV ────
                # This work is independent of the global KV buffer contents, so
                # it can execute concurrently with the NCCL collective.
                _enc_k_ready = enc_k
                _enc_v_ready = enc_v

                # ── Wait for the async all-reduce to complete ────────────────
                if self._kv_reduce_work is not None:
                    self._kv_reduce_work.wait()
                    self._kv_reduce_work = None

                # Dual-stream: append encoder K/V to global context so every
                # video frame attends to all text tokens.
                if _enc_k_ready is not None:
                    k_global = torch.cat([k_global, _enc_k_ready], dim=1)
                    v_global = torch.cat([v_global, _enc_v_ready], dim=1)

                hidden_states = self._compute_lvsa(
                    query, key, value, k_global, v_global, encoder_hidden_states,
                )
        else:
            # ── Dense fallback (geometry mismatch) ───────────────────────────
            # Video queries attend densely to all video (+ encoder) K/V. The
            # shared encoder-query / cross-attn / output / format tail below is
            # identical for both paths, so a fallback here is correct dense
            # attention — just without LVSA's sparsity.
            #
            # This path uses the LOCAL key/value only. Under context parallelism
            # (world > 1) the local shard is not the full sequence, so per-rank
            # dense attention would be silently incorrect. LVSA + CP requires the
            # clean T_lat × P frame grid; a geometry mismatch under CP is an
            # unsupported layout, not a recoverable single-rank fallback — so
            # fail loudly rather than emit per-shard output. (The plugin hooks'
            # equivalent path delegates to vllm-omni's SP-aware forward; the
            # standalone processor has no such delegate.)
            if self.world > 1:
                raise NotImplementedError(
                    f"LVSA dense fallback under context parallelism is unsupported "
                    f"(geometry mismatch: local_seq={local_seq}, world={self.world}). "
                    f"The local shard is not the full sequence — run single-rank, or "
                    f"use a layout whose sequence is a clean T_lat x P frame grid."
                )
            self._warn_geometry_mismatch_once(local_seq)
            if enc_k is not None:
                full_k = torch.cat([key, enc_k], dim=1)
                full_v = torch.cat([value, enc_v], dim=1)
                hidden_states = self._compute_full_attention(query, full_k, full_v)
            else:
                hidden_states = self._compute_full_attention(query, key, value)

        # ── Dual-stream: encoder query full attention ────────────────────────
        # Encoder query tokens need full attention against all K/V (video + encoder).
        # Under CP the video K/V must be the FULL grid, not the local shard
        # (see _compute_encoder_query_attention).
        enc_output = None
        if enc_q is not None:
            enc_output = self._compute_encoder_query_attention(
                enc_q, enc_k, enc_v, key, value,
            )

        # ── Cross-attention (model-specific, e.g. I2V image context) ─────────
        cross_out = adapter.cross_attention(
            attn, query, encoder_hidden_states_img, self._attention_backend,
        )
        if cross_out is not None:
            hidden_states = hidden_states + cross_out

        # ── Output projection (model-specific) ────────────────────────────────
        hidden_states = adapter.output_projection(attn, hidden_states, query_dtype)

        # ── Format output for the model's block forward() ────────────────────
        if enc_output is not None:
            # Dual-stream: project encoder output and return tuple
            return adapter.format_output(
                attn, hidden_states, enc_output, encoder_seq_len, query_dtype,
            )
        return hidden_states

    def _geometry_matches(self, local_seq: int) -> bool:
        """True when ``local_seq`` is the expected per-rank video token count —
        a clean frame grid: ``local_seq × world == total_num_frames × num_patches``.
        When False, the sequence is not frame-structured (e.g. prepended history
        tokens) and LVSA's windowing must not engage (see the guard in __call__)."""
        return (
            self.num_patches > 0
            and local_seq * self.world == self.total_num_frames * self.num_patches
        )

    def _warn_geometry_mismatch_once(self, local_seq: int) -> None:
        """Warn once per processor that the runtime sequence is not the expected
        frame grid, so LVSA fell back to dense attention (see the geometry guard
        in ``__call__``)."""
        if getattr(self, "_geometry_warned", False):
            return
        self._geometry_warned = True
        if self.rank == 0:
            expected = self.total_num_frames * self.num_patches
            print(
                f"[LVSA] WARNING geometry mismatch: runtime self-attention "
                f"seq={local_seq} (world={self.world}) != configured "
                f"T_lat*P={expected} (total_num_frames={self.total_num_frames}, "
                f"num_patches={self.num_patches}). The token sequence is not the "
                f"expected per-frame grid (e.g. prepended history tokens or a "
                f"non-frame-structured layout) -> falling back to DENSE attention "
                f"to avoid silent corruption. LVSA is not engaging for this model.",
                flush=True,
            )

    # ── Global K/V construction ───────────────────────────────────────────────

    def _build_global_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build K/V tensors for all global (anchor) frames.

        Token-based overlap logic
        -------------------------
        Global frame gf occupies global tokens [gf*P, (gf+1)*P).
        Our local range is [global_token_start, global_token_start+local_seq).
        The intersection tells us which local tokens to write into the buffer.
        A frame may be split across two ranks (partial ownership at boundaries);
        both ranks contribute their portion and all_reduce(SUM) assembles them.

        K and V are stacked into a single [2, B, num_global*P, H, D] buffer so
        that one all_reduce(SUM) collective covers both, halving round trips vs.
        two separate all_reduce calls.  The buffer is cached on the processor
        instance and reused across calls to avoid repeated GPU allocations.
        """
        B, local_seq, H, D = key.shape
        P = self.num_patches
        gts = self.global_token_start
        num_global = len(self._global_indices)

        # Reuse cached buffer when possible; reallocate only on shape change.
        buf_shape = (2, B, num_global * P, H, D)
        if self._kv_buf is None or self._kv_buf.shape != buf_shape:
            self._kv_buf = key.new_zeros(*buf_shape)
        else:
            self._kv_buf.zero_()

        for gi, gf in enumerate(self._global_indices):
            gf_start = gf * P
            gf_end = (gf + 1) * P
            ovl_start = max(gf_start, gts)
            ovl_end = min(gf_end, gts + local_seq)
            if ovl_start >= ovl_end:
                continue  # this global frame is entirely on another rank
            l_start = ovl_start - gts
            l_end = ovl_end - gts
            g_start = ovl_start - gf_start
            g_end = ovl_end - gf_start
            buf_sl = slice(gi * P + g_start, gi * P + g_end)
            self._kv_buf[0, :, buf_sl] = key[:, l_start:l_end]
            self._kv_buf[1, :, buf_sl] = value[:, l_start:l_end]

        # Single collective for both K and V — SUM is correct because every
        # buffer position is written by exactly the rank(s) that own those tokens.
        # Launch as async_op so the caller can overlap other work (encoder KV
        # preparation, output buffer allocation) while the all-reduce is in flight.
        if self.world > 1:
            self._kv_reduce_work = dist.all_reduce(
                self._kv_buf, op=dist.ReduceOp.SUM, async_op=True,
            )
        else:
            # Single-GPU: no communication needed.  We still copy global frame
            # tokens into _kv_buf rather than indexing key/value directly because
            # downstream consumers (Triton unified kernel, FlashInfer) require a
            # contiguous, compactly-packed global KV buffer.
            self._kv_reduce_work = None

        return self._kv_buf[0], self._kv_buf[1]

    # ── Full attention (for encoder queries in dual-stream) ────────────────────

    def _compute_encoder_query_attention(self, enc_q, enc_k, enc_v, key, value):
        """Dual-stream text-query attention: encoder (text) queries attend the
        FULL video K/V plus the encoder K/V (one dense softmax — the text stream
        of the joint attention, matching single-GPU).

        Under context-parallelism the per-rank ``key``/``value`` are only the
        LOCAL video shard, so the full video K/V is gathered across ranks first.
        Without this, each rank's text-query output sees only ``1/world`` of the
        video and diverges from single-GPU — the bug that broke HunyuanVideo CP
        for EVERY cp_mode (the text stream then poisons every later block through
        the joint attention). ``world == 1`` is a no-op (local shard == full).
        """
        if self.world > 1:
            key = _all_gather_seq(key, self.world)
            value = _all_gather_seq(value, self.world)
        full_k = torch.cat([key, enc_k], dim=1)
        full_v = torch.cat([value, enc_v], dim=1)
        return self._compute_full_attention(enc_q, full_k, full_v)

    def _compute_full_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Full (dense) attention for encoder query tokens in dual-stream models.

        Parameters
        ----------
        query : [B, enc_seq, H, D] — encoder query tokens
        key   : [B, full_seq, H, D] — all K (video + encoder)
        value : [B, full_seq, H, D] — all V (video + encoder)

        Returns
        -------
        [B, enc_seq, H, D] — attention output for encoder queries
        """
        import torch.nn.functional as F

        # Transpose to [B, H, seq, D] for scaled_dot_product_attention
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2)  # back to [B, seq, H, D]

    # ── LVSA loop ──────────────────────────────────────────────────────────────

    def _compute_lvsa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_global: torch.Tensor,
        v_global: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._use_flashinfer:
            return self._compute_lvsa_flashinfer(
                query, key, value, k_global, v_global,
                self._metadata, self._fi,
            )
        return self._compute_lvsa_sdpa(query, key, value, k_global, v_global)

    def _compute_lvsa_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_global: torch.Tensor,
        v_global: torch.Tensor,
    ) -> torch.Tensor:
        """Per-frame block-sparse LVSA — delegates to ``lvsa_sdpa()``."""
        return lvsa_sdpa(
            query, key, value, k_global, v_global,
            self._metadata, self._attention_backend,
        )

    def _compute_lvsa_ulysses(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        enc_k: Optional[torch.Tensor],
        enc_v: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Plain-Ulysses LVSA: all-to-all gather the full frame grid (head-
        sharded), run the single-device pattern, all-to-all scatter back.
        Inputs are the per-rank shard [B, local_seq, H, D]; returns the same.
        world==1 skips the all-to-alls (identity) -> pure single-device.
        """
        H = query.shape[2]
        H_kv = key.shape[2]   # may be < H under GQA (num_kv_heads < num_heads)
        if self.world > 1:
            # Both Q and KV head counts are sharded by the all-to-all (scatter
            # dim=2), so both must divide world — and the per-rank GQA ratio
            # H/H_kv is preserved. For MHA (H == H_kv) this is identical to the
            # single-count path. All currently-supported ulysses-CP models are
            # MHA; this keeps it correct if a GQA model gains CP support.
            if H % self.world != 0 or H_kv % self.world != 0:
                raise ValueError(
                    f"cp_mode='ulysses' needs both num_heads ({H}) and "
                    f"num_kv_heads ({H_kv}) divisible by world ({self.world}); "
                    f"use cp_mode='custom' for this layout."
                )
            query = _ulysses_all_to_all(query, 2, 1, self.world)
            key = _ulysses_all_to_all(key, 2, 1, self.world)
            value = _ulysses_all_to_all(value, 2, 1, self.world)
            if enc_k is not None:
                # Encoder K/V carry KV heads — slice with the KV head count, not
                # the (possibly larger) query head count.
                hl_kv = H_kv // self.world
                h0_kv = self.rank * hl_kv
                enc_k = enc_k[:, :, h0_kv:h0_kv + hl_kv, :].contiguous()
                enc_v = enc_v[:, :, h0_kv:h0_kv + hl_kv, :].contiguous()

        k_global, v_global = build_global_kv(
            key, value, self._ulysses_metadata.global_indices, self._num_patches,
        )
        if enc_k is not None:
            k_global = torch.cat([k_global, enc_k], dim=1)
            v_global = torch.cat([v_global, enc_v], dim=1)

        # Same single-device pattern, either backend. FlashInfer uses the
        # ulysses metadata's CSR + its own runtime cache (self._ufi).
        if self._use_flashinfer:
            out = self._compute_lvsa_flashinfer(
                query, key, value, k_global, v_global,
                self._ulysses_metadata, self._ufi,
            )
        else:
            out = lvsa_sdpa(
                query, key, value, k_global, v_global,
                self._ulysses_metadata, self._attention_backend,
            )

        if self.world > 1:
            out = _ulysses_all_to_all(out, 1, 2, self.world)
        return out

    # ── Ring block-sparse path ───────────────────────────────────────────────

    def _compute_lvsa_ring(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        enc_k: Optional[torch.Tensor],
        enc_v: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Variant-A ring LVSA: rotate K/V around the ring, attend the LOCAL Q to
        each incoming block under the LVSA frame-mask, accumulate via online-
        softmax (LSE merge).

        Inputs are the per-rank shard ``[B, local_seq, H, D]``; returns the same
        ``[B, local_seq, H, D]`` (output projection happens in ``__call__``,
        identical to ``_compute_lvsa_ulysses``).

        Globals are NOT broadcast separately — a global frame's K/V block is
        attended whenever it rotates past this rank (the block-mask includes it),
        so the rotation handles globals + window uniformly. ``world==1`` is a
        single step (no rotation), proving the masked-block + LSE math equals the
        single-device pattern before any comm is involved.

        Ring shards SEQUENCE, not heads -> no ``num_heads % world`` constraint
        (GQA ok: K/V are expanded to the H query heads for the masked matmul).

        ``--flashinfer`` switches to the fused block-sparse-LSE kernel
        (``_compute_lvsa_ring_flashinfer``); this pure-torch path is the default
        and the CPU-tested correctness reference.
        """
        device = query.device
        if getattr(self, "_last_device", None) != device:
            self._last_device = device
            self._ring_mask_cache = {}
            self._ring_fi_cache = {}
            self._ring_workspaces = {}

        if self._use_flashinfer:
            return self._compute_lvsa_ring_flashinfer(
                query, key, value, enc_k, enc_v,
            )

        from .sparse_attention import ring_block_frame_mask

        B, Sloc, H, D = query.shape
        H_kv = key.shape[2]
        P = self.num_patches
        T = self.total_num_frames
        assert T % self.world == 0, (
            f"cp_mode='ring' needs total_frames ({T}) divisible by world "
            f"({self.world}) for whole-frame blocks; use 'custom' otherwise."
        )
        Tloc = T // self.world

        meta = self._ring_metadata
        global_set = meta.global_set

        # [B, H, S, D] layout for matmul; expand GQA K/V to H heads.
        q = query.transpose(1, 2).contiguous()                # [B,H,Sloc,D]
        k_cur = key.transpose(1, 2).contiguous()              # [B,H_kv,Sloc,D]
        v_cur = value.transpose(1, 2).contiguous()
        if H_kv != H:
            rep = H // H_kv
            k_cur = k_cur.repeat_interleave(rep, dim=1).contiguous()
            v_cur = v_cur.repeat_interleave(rep, dim=1).contiguous()

        q_frames = range(self.rank * Tloc, (self.rank + 1) * Tloc)
        src = self.rank                                       # block currently held
        out = lse = None
        for step in range(self.world):
            # Cached per ``src`` (built once per step, reused across layers).
            mask = self._ring_mask_cache.get(src)
            if mask is None:
                m = ring_block_frame_mask(
                    q_frames, range(src * Tloc, (src + 1) * Tloc),
                    P, self.window_size, T, global_set, self._expand_window,
                )
                mask = m.to(q.device) if m is not None else False  # False = empty pair
                self._ring_mask_cache[src] = mask
            if mask is not False:                             # skip empty block-pairs
                b_out, b_lse = _attn_with_lse(q, k_cur, v_cur, attn_mask=mask)
                out, lse = _merge_out_lse(out, lse, b_out, b_lse)
            if step < self.world - 1:
                k_cur, v_cur, src = self._ring_rotate(k_cur, v_cur, src)

        # Encoder/text K/V: every video Q attends to all text (a fixed global
        # term), exactly as the custom/ulysses paths append enc to global K/V.
        if enc_k is not None:
            ek = enc_k.transpose(1, 2).contiguous()
            ev = enc_v.transpose(1, 2).contiguous()
            if ek.shape[1] != H:
                rep = H // ek.shape[1]
                ek = ek.repeat_interleave(rep, 1).contiguous()
                ev = ev.repeat_interleave(rep, 1).contiguous()
            e_out, e_lse = _attn_with_lse(q, ek, ev)          # full attention, no mask
            out, lse = _merge_out_lse(out, lse, e_out, e_lse)

        out = out.transpose(1, 2).contiguous().to(query.dtype)  # [B,Sloc,H,D]
        return out

    def _ring_rotate(self, k_cur, v_cur, src):
        """Send the held K/V block to rank+1, receive the next block from rank-1.

        Standard ring P2P (batched isend/irecv to avoid deadlock). Returns
        ``(k_next, v_next, new_src)``; pure ring topology, O(world) total.
        Only reached when ``world > 1``.
        """
        nxt = (self.rank + 1) % self.world
        prv = (self.rank - 1) % self.world
        k_send = k_cur.contiguous()
        v_send = v_cur.contiguous()
        k_next = torch.empty_like(k_send)
        v_next = torch.empty_like(v_send)
        ops = [
            dist.P2POp(dist.isend, k_send, nxt),
            dist.P2POp(dist.isend, v_send, nxt),
            dist.P2POp(dist.irecv, k_next, prv),
            dist.P2POp(dist.irecv, v_next, prv),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        return k_next, v_next, (src - 1) % self.world

    def _compute_lvsa_ring_flashinfer(self, query, key, value, enc_k, enc_v):
        """Fused flashinfer block-sparse-LSE ring (``--flashinfer``). Same
        variant-A algorithm as the pure-torch ``_compute_lvsa_ring`` — rotate
        K/V, attend the local Q to each block under the LVSA pattern, accumulate
        via online-softmax — but each block-pair runs through a
        ``flashinfer.BlockSparseAttentionWrapper`` over a per-block-pair frame
        CSR (it SKIPS the non-attended frame-blocks: no dense ``[Sq,Sk]`` mask,
        no compute on masked pairs), with blocks combined via flashinfer's native
        ``merge_state``. The CSR + planned wrapper are CACHED per ``src`` (built
        once per step, reused across the ~N transformer layers; cleared on
        keyframe rotation). FlashInfer's layout ``[seq, H, D]`` == the input
        layout (no transpose); GQA-native. CUDA-only; GPU-validated against the
        CPU-tested pure-torch ring.
        """
        import flashinfer
        from .sparse_attention import ring_block_frame_csr

        B, Sloc, H, D = query.shape
        H_kv = key.shape[2]
        P = self.num_patches
        T = self.total_num_frames
        assert T % self.world == 0, (
            f"cp_mode='ring' needs total_frames ({T}) divisible by world "
            f"({self.world}); use 'custom' otherwise."
        )
        Tloc = T // self.world
        global_set = self._ring_metadata.global_set
        dt = {torch.float16: "float16", torch.bfloat16: "bfloat16",
              torch.float32: "float32"}.get(query.dtype, "bfloat16")

        k_cur = key.contiguous()
        v_cur = value.contiguous()
        src = self.rank
        q_frames = range(self.rank * Tloc, (self.rank + 1) * Tloc)

        acc_o = [None] * B            # per-batch flashinfer-layout accumulators
        acc_l = [None] * B
        for step in range(self.world):
            wrapper = self._ring_fi_cache.get(src)
            if wrapper is None:       # build CSR + plan once per (step, src)
                csr = ring_block_frame_csr(
                    q_frames, range(src * Tloc, (src + 1) * Tloc),
                    self.window_size, T, global_set, self._expand_window,
                )
                if csr is None:
                    wrapper = False                       # empty block-pair
                else:
                    ws = self._ring_workspaces.get(src)
                    if ws is None:
                        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                         device=query.device)
                        self._ring_workspaces[src] = ws   # persistent scratch
                    wrapper = flashinfer.BlockSparseAttentionWrapper(ws)
                    wrapper.plan(
                        indptr=csr[0].to(query.device),
                        indices=csr[1].to(query.device),
                        M=Sloc, N=Tloc * P, R=P, C=P,
                        num_qo_heads=H, num_kv_heads=H_kv, head_dim=D,
                        q_data_type=dt, kv_data_type=dt, o_data_type=dt,
                    )
                self._ring_fi_cache[src] = wrapper
            if wrapper is not False:
                for b in range(B):
                    o, l = wrapper.run(
                        query[b].contiguous(), k_cur[b].contiguous(),
                        v_cur[b].contiguous(), return_lse=True,
                    )
                    if acc_o[b] is None:
                        acc_o[b], acc_l[b] = o, l
                    else:
                        acc_o[b], acc_l[b] = flashinfer.merge_state(
                            acc_o[b], acc_l[b], o, l)
            if step < self.world - 1:
                k_cur, v_cur, src = self._ring_rotate(k_cur, v_cur, src)

        # Encoder/text term: gen Q attends ALL text (full, always-global).
        if enc_k is not None:
            for b in range(B):
                o, l = flashinfer.single_prefill_with_kv_cache(
                    query[b].contiguous(), enc_k[b].contiguous(),
                    enc_v[b].contiguous(), causal=False, return_lse=True)
                if acc_o[b] is None:
                    acc_o[b], acc_l[b] = o, l
                else:
                    acc_o[b], acc_l[b] = flashinfer.merge_state(
                        acc_o[b], acc_l[b], o, l)

        out = query.new_empty((B, Sloc, H, D))
        for b in range(B):
            out[b] = acc_o[b].to(query.dtype)
        return out

    # ── FlashInfer block-sparse path ─────────────────────────────────────────

    def _ensure_flashinfer_planned(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        enc_tokens: int,
        meta: "LVSAMetadata",
        st: _FIState,
    ) -> None:
        """Initialize a FlashInfer wrapper and call plan() for ``meta``'s CSR.

        Parametrized by ``(meta, st)`` so the same machinery serves both the
        ``custom`` per-rank metadata (``self._metadata`` / ``self._fi``) and the
        ``ulysses`` full-grid metadata (``self._ulysses_metadata`` /
        ``self._ufi``). The plan is reused across diffusion steps until the
        pattern rebuilds (which calls ``st.reset()``, e.g. rotating keyframes).

        Parameters
        ----------
        enc_tokens : number of encoder tokens appended to k_global (0 for
                     single-stream models like Wan).
        meta       : the LVSAMetadata whose CSR (fi_indptr/fi_indices/fi_M/
                     fi_N) is planned.
        st         : the runtime cache holder to populate.
        """
        # Gen-only CSR: the encoder/text K/V is NOT appended as block columns
        # here — it is handled as a separate dense term and combined via
        # log-sum-exp in _compute_lvsa_flashinfer (avoiding the zero-padded
        # encoder-block phantom-key bug). Replanned after a metadata rebuild
        # (which calls st.reset(), e.g. rotating keyframes).
        if st.planned:
            return

        device = query.device
        H = query.shape[2]
        H_kv = key.shape[2]   # may be < H under GQA (num_kv_heads < num_heads)
        D = query.shape[3]
        P = self.num_patches

        indptr = meta.fi_indptr.to(device)
        indices = meta.fi_indices.to(device)

        # Allocate workspace (128 MB), once per state, reused across rebuilds.
        if st.workspace is None:
            st.workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device,
            )

        st.wrapper = flashinfer.BlockSparseAttentionWrapper(st.workspace)

        dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        q_dtype_str = dtype_map.get(query.dtype, "bfloat16")

        st.wrapper.plan(
            indptr=indptr,
            indices=indices,
            M=meta.fi_M,
            N=meta.fi_N,
            R=P,
            C=P,
            num_qo_heads=H,
            num_kv_heads=H_kv,
            head_dim=D,
            q_data_type=q_dtype_str,
            kv_data_type=q_dtype_str,
            o_data_type=q_dtype_str,
        )
        st.planned = True
        st.enc_tokens = enc_tokens
        st.N_actual = meta.fi_N

    def _compute_lvsa_flashinfer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_global: torch.Tensor,
        v_global: torch.Tensor,
        meta: "LVSAMetadata",
        st: _FIState,
    ) -> torch.Tensor:
        """FlashInfer block-sparse LVSA via LSE-merge.

        Gen video tokens go through the fused block-sparse kernel; the
        encoder/text tokens are computed as a SEPARATE dense term and combined
        exactly via log-sum-exp. This replaces the old zero-padded encoder block
        (phantom zero keys diluting every query when the text length was not a
        multiple of P). CP-aware: under world>1 each rank merges its own Q shard
        against the replicated encoder K/V.

        Parametrized by ``(meta, st)`` — the caller passes either the custom
        per-rank pair (``self._metadata``, ``self._fi``) or the ulysses
        full-grid pair (``self._ulysses_metadata``, ``self._ufi``). All CSR /
        copy-list geometry is read from ``meta``; all runtime buffers live on
        ``st``.
        """
        B, local_seq, H, D = query.shape
        H_kv = key.shape[2]   # may be < H under GQA (num_kv_heads < num_heads)
        P = self.num_patches
        M = meta.fi_M

        # ── Detect encoder tokens appended to k_global ──
        num_global_video_tokens = len(meta.global_indices) * P
        total_global_tokens = k_global.shape[1]
        enc_tokens = total_global_tokens - num_global_video_tokens

        self._ensure_flashinfer_planned(query, key, enc_tokens, meta, st)
        # ── Gen-only compact KV buffer (encoder handled separately) ──
        # Width is the K/V head count H_kv (== H for MHA), NOT the query head
        # count: under GQA (H_kv < H) the compact buffer mirrors the GQA K/V
        # layout and the plan's num_kv_heads — no repeat-KV here (FlashInfer's
        # block-sparse kernel broadcasts KV heads internally), exactly as in
        # cosmos_flashinfer.py.
        compact_N = meta.fi_compact_n * P
        compact_shape = (B, compact_N, H_kv, D)
        if st.compact_k is None or st.compact_k.shape != compact_shape:
            st.compact_k = query.new_zeros(*compact_shape)
            st.compact_v = query.new_zeros(*compact_shape)
        ck, cv = st.compact_k, st.compact_v

        # Video globals → compact positions (from pre-computed copy list)
        for src_s, dst_s in meta.fi_global_copies:
            ck[:, dst_s:dst_s + P] = k_global[:, src_s:src_s + P]
            cv[:, dst_s:dst_s + P] = v_global[:, src_s:src_s + P]
        # Local (non-global) video frames → compact positions
        for src_s, dst_s in meta.fi_local_copies:
            ck[:, dst_s:dst_s + P] = key[:, src_s:src_s + P]
            cv[:, dst_s:dst_s + P] = value[:, src_s:src_s + P]

        # Encoder/text K/V split back out for a SEPARATE dense term (no padding).
        k_text = k_global[:, num_global_video_tokens:]
        v_text = v_global[:, num_global_video_tokens:]

        # ── Pad Q to M = MB * P if needed (cached, tail stays zero) ──
        if local_seq < M:
            if st.q_pad is None or st.q_pad.shape != (B, M, H, D):
                st.q_pad = query.new_zeros(B, M, H, D)
            st.q_pad[:, :local_seq] = query
            q_padded = st.q_pad
        else:
            q_padded = query

        # ── gen block-sparse + dense encoder term, combined via exact
        #    log-sum-exp. FlashInfer returns LSE in log2 (= log2(sum exp)),
        #    so the merge weights use exp2. This replaces the old zero-padded
        #    encoder block (which attended to phantom zero keys when the text
        #    length was not a multiple of P).
        out = query.new_empty(B, local_seq, H, D)
        for b in range(B):
            o_gen, lse_gen = st.wrapper.run(
                q_padded[b], ck[b], cv[b], return_lse=True,
            )
            o_gen = o_gen[:local_seq].float()
            lse_gen = lse_gen[:local_seq]
            if enc_tokens > 0:
                o_txt, lse_txt = flashinfer.single_prefill_with_kv_cache(
                    query[b].contiguous(), k_text[b].contiguous(),
                    v_text[b].contiguous(), causal=False, return_lse=True,
                )
                o_txt = o_txt.float()
                mm = torch.maximum(lse_gen, lse_txt)
                w_gen = torch.exp2(lse_gen - mm).unsqueeze(-1)
                w_txt = torch.exp2(lse_txt - mm).unsqueeze(-1)
                out[b] = ((o_gen * w_gen + o_txt * w_txt) / (w_gen + w_txt)).to(query.dtype)
            else:
                out[b] = o_gen.to(query.dtype)
        return out


# Backwards-compatible alias for existing code that references the old name.
WanDistributedLVSAProcessor = DistributedLVSAProcessor
