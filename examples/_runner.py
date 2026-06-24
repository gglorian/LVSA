"""
_runner.py — Shared argparse helpers for LVSA example scripts.

Two functions are provided:

  add_common_args(p)  — flags shared by every model (video dims, sampling,
                        output).  Defaults use Wan values as the source of
                        truth; per-model entry points call p.set_defaults(...)
                        to override where they differ.

  add_lvsa_args(p)    — LVSA / block-sparse-attention flags (all models share
                        these unchanged).

All other functions (runner logic, dataclasses, etc.) belong to later tasks.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


# ── Common flags (shared across all model example scripts) ────────────────────


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add flags that are common to every LVSA example script.

    Defaults follow ``wan_generate.py`` as the source of truth.
    Per-model entry points should call ``p.set_defaults(...)`` afterwards to
    override any value that differs for their model.  Flags whose defaults
    vary substantially across models are registered *without* a hardcoded
    default (they default to ``None``) so that ``set_defaults`` is the sole
    authority — those flags are: ``--steps``, ``--num-frames``, ``--height``,
    ``--width``, ``--fps``, ``--negative-prompt``.
    """

    # ── Model / prompt ────────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="PATH",
        help="Path or HuggingFace Hub ID of the model pipeline.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the video to generate.",
    )
    p.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt. Default varies per model; set via set_defaults.",
    )

    # ── Video dimensions ──────────────────────────────────────────────────────
    # num-frames, height, width, fps all vary across models — no shared default.
    p.add_argument(
        "--num-frames",
        type=int,
        help="Frames to generate. Default varies per model; set via set_defaults.",
    )
    p.add_argument(
        "--height",
        type=int,
        help="Frame height (px). Default varies per model; set via set_defaults.",
    )
    p.add_argument(
        "--width",
        type=int,
        help="Frame width (px). Default varies per model; set via set_defaults.",
    )

    # ── Sampling ──────────────────────────────────────────────────────────────
    # steps varies (wan=40, hunyuan=50, cogvideox=50, cosmos=35) — no default.
    p.add_argument(
        "--steps",
        type=int,
        help="Denoising steps. Default varies per model; set via set_defaults.",
    )
    # guidance and seed pin to wan's values (5.0 and 16); tests assert this.
    p.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="CFG scale.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=16,
        help="Random seed.",
    )
    p.add_argument(
        "--fps",
        type=int,
        help="Output FPS. Default varies per model; set via set_defaults.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for the generated video. Created if missing.",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename inside --output-dir. If omitted, a descriptive "
        "name encoding model, geometry, backend, and run parameters is "
        "auto-generated. Extension (.mp4) is appended automatically if missing.",
    )

    # ── Profiling ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--profile",
        action="store_true",
        help="Log per-step wall-clock timing for profiling attention phases.",
    )


# ── LVSA / block-sparse-attention flags ───────────────────────────────────────


def add_lvsa_args(p: argparse.ArgumentParser) -> None:
    """Add LVSA block-sparse attention flags to *p*.

    All defaults are identical across models, copied verbatim from
    ``wan_generate.py``.
    """

    g = p.add_argument_group(
        "Sliding Window Attention (LVSA)",
        "Block-sparse attention to reduce memory for long videos. "
        "Add --lvsa to enable; all other LVSA flags are ignored otherwise.",
    )

    g.add_argument(
        "--lvsa",
        action="store_true",
        help="Enable block-sparse LVSA processor (off = dense full attention).",
    )
    g.add_argument(
        "--flashinfer",
        action="store_true",
        help="Use FlashInfer BlockSparseAttentionWrapper for LVSA instead of "
        "per-frame SDPA. Requires flashinfer to be installed.",
    )
    g.add_argument(
        "--window-size",
        type=int,
        default=12,
        help="Half-width of the LVSA sliding window in *video* frames. "
        "Converted to latent frames internally.",
    )
    g.add_argument(
        "--n-first-frames",
        type=int,
        default=4,
        help="Number of leading video frames always included as global context. "
        "Converted to latent frames internally.",
    )
    g.add_argument(
        "--sparsity-scale",
        type=float,
        default=1.0,
        help="Scale factor for the attention sparsity budget. "
        "<1.0 makes attention more sparse (fewer attended frames), "
        ">1.0 makes it less sparse (more attended frames). "
        "Default 1.0 preserves the original behaviour.",
    )
    g.add_argument(
        "--auto-keyframes",
        action="store_true",
        help="Automatically compute key-frame-interval so that total attended "
        "frames per query approximates 21 (the reference budget). "
        "Overrides --key-frame-interval. Enabled by default; use "
        "--no-auto-keyframes to disable.",
    )
    g.add_argument(
        "--no-auto-keyframes",
        dest="auto_keyframes",
        action="store_false",
        help="Disable auto keyframe interval; use the explicit "
        "--key-frame-interval instead.",
    )
    p.set_defaults(auto_keyframes=True)
    g.add_argument(
        "--no-expand-window",
        dest="expand_window",
        action="store_false",
        help="Use adaptive (non-expanded) window bounds.",
    )
    p.set_defaults(expand_window=True)
    g.add_argument(
        "--rotate-keyframes",
        action="store_true",
        help="Shift periodic keyframes by 1 position each denoising step, "
        "cycling through all positions over key_frame_interval steps. "
        "This ensures every frame acts as a global anchor at some point.",
    )
    g.add_argument(
        "--show-mask",
        action="store_true",
        help="Print the T×T attention mask matrix showing which latent frames "
        "each query frame attends to (G=global, W=window, X=both). "
        "Useful for debugging LVSA patterns. Requires --lvsa.",
    )
    g.add_argument(
        "--show-mask-compact",
        nargs="?",
        const="once",
        default=None,
        choices=["once", "step"],
        help="Compact 1-char-per-column attention mask. "
        "'once' (default if flag given alone) prints at init. "
        "'step' prints at every denoising step (useful with --rotate-keyframes).",
    )


# ── Distributed context ───────────────────────────────────────────────────────


@dataclass
class RunContext:
    rank: int
    world: int
    device: Any
    distributed: bool


def resolve_distributed(init: bool = True) -> "RunContext":
    """Detect distributed environment and optionally initialise the process group.

    When *init* is ``False`` the function reads ``RANK``/``WORLD_SIZE`` from the
    environment but does NOT call ``dist.init_process_group`` or trigger any
    GPU/device side-effects.  This keeps the function safe for CPU-only unit
    tests.

    When *init* is ``True`` (the default, used by real entry points) the
    function also:

    - calls ``dist.init_process_group`` if distributed and not yet initialised,
    - sets ``HF_ENABLE_PARALLEL_LOADING=YES`` (must run before pipeline load),
    - calls ``enable_fast_matmul()``,
    - prints an ``[init]`` line on rank 0.
    """
    import torch.distributed as dist
    from lvsa.device import get_device, get_distributed_backend, enable_fast_matmul

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    if distributed and init and not dist.is_initialized():
        dist.init_process_group(get_distributed_backend())
        # After init, prefer the authoritative values from the process group.
        rank = dist.get_rank()
        world = dist.get_world_size()

    device = get_device(rank)

    if init:
        # Preserve wan's pre-load side effects (must run before the pipeline loads).
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
        enable_fast_matmul()
        if rank == 0:
            mode = f"distributed (world_size={world})" if distributed else "single-GPU"
            print(f"[init] {mode}  device={device}")

    return RunContext(rank=rank, world=world, device=device, distributed=distributed)


# ── Step callback (drift fix: canonical implementation shared by all models) ──


def _per_step_actions(args, lvsa_processor, rank, step_index, state):
    """Shared per-step body used by both the diffusers callback and the
    scheduler-step hook. `state` is a dict persisting across steps
    (holds '_step_times' and '_last_t')."""
    if lvsa_processor is not None:
        if getattr(args, "rotate_keyframes", False):
            lvsa_processor.set_step(step_index)
        if rank == 0 and getattr(args, "show_mask_compact", None) == "step":
            print(f"\n[LVSA-WINDOW] step {step_index}:")
            lvsa_processor.print_attention_mask_compact()
    if getattr(args, "profile", False) and rank == 0:
        now = time.time()
        st = state.setdefault("_step_times", [])
        st.append(now)
        if len(st) > 1:
            print(f"[profile] step {step_index}: {st[-1] - st[-2]:.3f}s")
    if rank == 0 and os.environ.get("LVSA_STEP_TIME_LOG", "0") == "1":
        _now = time.perf_counter()
        _last = state.get("_last_t")
        if _last is not None:
            print(f"[LVSA-TIME] step={step_index - 1} dt={_now - _last:.3f}s", flush=True)
        state["_last_t"] = _now
    if rank == 0 and os.environ.get("LVSA_MEM_LOG", "0") == "1":
        from lvsa.device import memory_stats
        _stats = memory_stats()
        if _stats is not None:
            _kind, _dev, _alloc, _reserved, _peak = _stats
            print(f"[LVSA-MEM] step={step_index} {_kind}={_dev} "
                  f"alloc={_alloc:.2f}GB reserved={_reserved:.2f}GB peak={_peak:.2f}GB", flush=True)


def make_step_callback(args, lvsa_processor, rank: int):
    """Build and return the denoising step callback, or ``None`` when inactive.

    Handles:
    - keyframe rotation (``--rotate-keyframes``)
    - compact mask printing (``--show-mask-compact step``)
    - per-step wall-clock profiling (``--profile``)
    - ``LVSA_STEP_TIME_LOG=1`` env-var timing
    - ``LVSA_MEM_LOG=1`` env-var memory logging

    Returns ``None`` when none of these are requested, so callers can skip
    passing a callback to the pipeline entirely.
    """
    need_callback = (
        (lvsa_processor and getattr(args, "rotate_keyframes", False))
        or (lvsa_processor and getattr(args, "show_mask_compact", None) == "step")
        or getattr(args, "profile", False)
        or os.environ.get("LVSA_STEP_TIME_LOG", "0") == "1"
        or os.environ.get("LVSA_MEM_LOG", "0") == "1"
    )
    if not need_callback:
        return None

    if rank == 0 and lvsa_processor:
        print("[LVSA] windowed attention for all steps")

    state = {}

    def step_callback(pipe_obj, step_index, timestep, callback_kwargs):
        _per_step_actions(args, lvsa_processor, rank, step_index, state)
        return callback_kwargs

    return step_callback


def install_scheduler_step_hook(pipe, args, lvsa_processor, rank) -> bool:
    """For pipelines that don't support `callback_on_step_end` (HunyuanVideo):
    wrap `pipe.scheduler.step` with an internal counter and run the shared
    per-step actions before each original step. Returns True if installed,
    False if nothing requested (scheduler left untouched)."""
    need = (
        lvsa_processor is not None
        or getattr(args, "profile", False)
        or os.environ.get("LVSA_STEP_TIME_LOG", "0") == "1"
        or os.environ.get("LVSA_MEM_LOG", "0") == "1"
    )
    if not need:
        return False
    if rank == 0 and lvsa_processor:
        print("[LVSA] windowed attention for all steps")
    counter = [0]
    state = {}
    orig_step = pipe.scheduler.step

    def hooked_step(*s_args, **s_kwargs):
        step_index = counter[0]
        _per_step_actions(args, lvsa_processor, rank, step_index, state)
        result = orig_step(*s_args, **s_kwargs)
        counter[0] += 1
        return result

    pipe.scheduler.step = hooked_step
    return True


# ── Context-parallel setup ────────────────────────────────────────────────────

def setup_cp(adapter, pipe, world: int, _setup=None) -> None:
    """Install context-parallel processors on the pipeline transformer(s).

    Wan2.2-A14B is a dual-expert model: it exposes a second transformer as
    ``pipe.transformer_2`` which needs its own CP plan. This wires both.
    """
    if _setup is None:
        from lvsa.parallel import setup_context_parallel as _setup
    _setup(adapter, pipe.transformer, world)
    if getattr(pipe, "transformer_2", None) is not None:
        _setup(adapter, pipe.transformer_2, world)


# ── Output path / descriptive auto-naming ─────────────────────────────────────

def build_output_path(args, world: int, gen_duration: float, mem_mb: float,
                      *, stem: str, ext: str = "mp4") -> str:
    """Return the full output path, reproducing the scripts' descriptive
    auto-name when ``--output-name`` is omitted (byte-identical to the prior
    per-script logic). When ``--output-name`` is given, append ``.{ext}`` if
    missing.
    """
    if args.output_name:
        filename = args.output_name
        if not filename.endswith(f".{ext}"):
            filename += f".{ext}"
    else:
        if args.lvsa:
            backend = "flashinfer" if args.flashinfer else "sdpa"
            kfi_tag = "auto" if args.auto_keyframes else str(getattr(args, "key_frame_interval", None))
            rot_tag = "_rot" if args.rotate_keyframes else ""
            ring_tag = ""
            lvsa_tag = (
                f"_lvsa_w{args.window_size}_f{args.n_first_frames}"
                f"_kfi{kfi_tag}{rot_tag}{ring_tag}_{backend}"
            )
        else:
            lvsa_tag = "_fullatt"
        gpu_tag = "balanced" if getattr(args, "balanced", False) else f"gpu{world}"
        model_tag = os.path.basename(args.model.rstrip("/\\"))
        prompt_tag = args.prompt.replace(" ", "_")[:30]
        filename = (
            f"{stem}"
            f"_{model_tag}"
            f"_{gpu_tag}"
            f"_{args.height}x{args.width}@{args.fps}"
            f"_frames{args.num_frames}"
            f"{lvsa_tag}"
            f"_steps{args.steps}_cfg{args.guidance}"
            f"_seed{args.seed}"
            f"_dur{gen_duration:.0f}s_mem{mem_mb:.0f}MB"
            f"_{prompt_tag}"
            f".{ext}"
        )
    return os.path.join(args.output_dir, filename)
