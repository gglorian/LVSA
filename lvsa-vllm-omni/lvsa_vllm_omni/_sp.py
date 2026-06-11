"""Sequence-parallel detection for the diffusion hooks.

LVSA's frame-grid geometry only holds when the hook sees the FULL sequence:
  - **Tensor-parallel** keeps the full sequence per rank (it shards *heads*), so
    LVSA's block-sparse pattern is correct under TP (the mask is head-independent;
    ``lvsa_sdpa``/FlashInfer handle the per-rank head count, and ``to_out``'s
    RowParallel reduces across ranks). Verified on GPU: Wan2.1-14B at TP=2 engages
    sparse LVSA (21/41 attended) and generates correctly.
  - **Sequence-parallel** (Ulysses / Ring) shards the *sequence* before the hook
    runs, so the per-rank tensor is a fragment of ``T_lat × P`` → geometry
    detection would silently match the wrong P → must fall back to dense.

So the hooks gate on SP specifically, NOT on total ``world_size`` (which counts
TP×SP×PP×CFG×DP and would wrongly disable LVSA under pure TP).
"""


def is_sp_active() -> bool:
    """True iff sequence-parallel is active in the current diffusion forward.

    Reads the framework's ``forward_context.sp_active`` (the "Bagel pattern":
    True when ``sequence_parallel_size > 1`` even without ``_sp_plan`` hooks) —
    the same signal Cosmos3's ``_is_sp_active`` uses. Returns False when vllm-omni
    is absent or no forward context is set (CPU tests, single-GPU, warmup).
    """
    try:
        from vllm_omni.diffusion.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        return False
    if not is_forward_context_available():
        return False
    try:
        return bool(get_forward_context().sp_active)
    except Exception:
        return False


def cfg_parallel_world_size() -> int:
    """CFG-parallel world size (1 when inactive or vllm-omni is absent).

    Under ``cfg_parallel_size=N`` the framework distributes the CFG passes
    (cond/uncond) across N ranks, so each rank sees ``cfg_passes / N`` forward
    passes per denoising step. The step counters divide by this so step
    detection (and with it ``--rotate-keyframes`` rotation and ``[LVSA-TIME]``)
    stays correct under CFG-parallel. Defensive like ``is_sp_active``: returns
    1 when the framework or the distributed group is unavailable (CPU tests,
    single-GPU, pre-init warmup).
    """
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_classifier_free_guidance_world_size,
        )
    except Exception:
        return 1
    try:
        return max(1, int(get_classifier_free_guidance_world_size()))
    except Exception:
        return 1
