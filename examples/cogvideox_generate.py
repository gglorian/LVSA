"""
cogvideox_generate.py — Video generation with CogVideoX, supporting single and multi-GPU inference.
========================================================================================================

Usage
-----
# Single GPU — baseline (no LVSA)
python cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720

# Single GPU with LVSA
python cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720 --lvsa

# Single GPU with LVSA + FlashInfer
python cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720 --lvsa --flashinfer

# Multi-GPU context-parallel (requires torchrun)
torchrun --nproc_per_node=2 cogvideox_parallel_lvsa.py \\
    --model THUDM/CogVideoX-2b \\
    --prompt "A dog running in the forest." \\
    --num-frames 49 --height 480 --width 720 --lvsa
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from examples._runner import (
    add_common_args, add_lvsa_args, resolve_distributed,
    setup_cp, build_output_path, install_scheduler_step_hook,
)

import os
import time
import argparse

import torch
import torch.distributed as dist

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from lvsa.adapters.cogvideox import CogVideoXAdapter
from lvsa.device import (
    max_memory_allocated,
    mem_get_info,
)
from lvsa.parallel import (
    patch_rotary_emb_for_context_parallel,
    install_lvsa_processors,
    compute_and_validate_seq_len,
)


# ── Argument parsing ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CogVideoX generation with optional LVSA and multi-GPU context parallelism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_args(parser)
    add_lvsa_args(parser)

    # Restore CogVideoX's per-model defaults for flags that add_common_args left unset:
    parser.set_defaults(
        steps=50,
        num_frames=49,
        height=480,
        width=720,
        fps=8,
        guidance=6.0,
        seed=16,
        negative_prompt="",
    )

    # ── CogVideoX-specific args NOT covered by the shared helpers ─────────────
    parser.add_argument(
        "--cp-mode", choices=["custom", "ulysses", "ring"], default="custom",
        help="Context-parallel attention mode (multi-GPU only). 'custom' (default) = "
             "all_reduce of global K/V + boundary guards (no head-count constraint). "
             "'ulysses' = all-to-all gather the full sequence, run the single-device "
             "LVSA pattern (needs num_heads %% world == 0; HunyuanVideo 1.5 has 16 "
             "heads -> world must divide 16). 'ring' = true ring rotation of K/V with "
             "the LVSA block-mask per step (no num_heads %% world constraint, unlike "
             "ulysses; compute-only savings, O(world) comm).",
    )

    parser.add_argument(
        "--key-frame-interval",
        type=int,
        default=16,
        help="Periodic keyframe interval (video frames).",
    )

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Distributed vs single-GPU detection ────────────────────────────────────
    ctx = resolve_distributed()
    rank, world, device, distributed = ctx.rank, ctx.world, ctx.device, ctx.distributed

    # ── Create model adapter ─────────────────────────────────────────────────
    adapter = CogVideoXAdapter()

    # ── Patch standard rotary BEFORE loading (multi-GPU only) ─────────────────
    if world > 1:
        patch_rotary_emb_for_context_parallel(adapter, rank, world)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    t0 = time.time()
    if rank == 0:
        print(f"[model] loading from {args.model} ...")

    pipe = CogVideoXPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # ── Enable VAE tiling to avoid OOM during decode ─────────────────────────
    pipe.vae.enable_tiling()
    if rank == 0:
        print("[vae] tiling enabled")

    # ── Sequence-length validation ────────────────────────────────────────────
    compute_and_validate_seq_len(
        args.num_frames,
        args.height,
        args.width,
        pipe.transformer.config,
        pipe.vae_scale_factor_temporal,
        getattr(pipe, "vae_scale_factor_spatial", 8),
        world,
        rank,
    )

    # ── LVSA processor installation ────────────────────────────────────────────
    lvsa_processor = None
    if args.lvsa:
        lvsa_processor = install_lvsa_processors(
            pipe, args, rank, world, adapter,
            sparsity_scale=args.sparsity_scale,
        )
    elif rank == 0:
        print("[attn] using standard full attention (no LVSA)")

    # ── Context-parallel plan (multi-GPU only) ────────────────────────────────
    if world > 1:
        setup_cp(adapter, pipe, world)

    if rank == 0:
        print(f"[model] loaded in {time.time() - t0:.1f}s")

    # ── Generate ──────────────────────────────────────────────────────────────
    if rank == 0:
        print(
            f"[generate] {args.num_frames} frames  "
            f"{args.height}x{args.width}  "
            f"{args.steps} steps  guidance={args.guidance}  seed={args.seed}"
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_gen = time.time()

    # ── Rotating keyframes / profiling ─────────────────────────────────────
    # CogVideoX does not support callback_on_step_end, so we hook into the
    # scheduler's step() method to inject per-step logic (gains [LVSA-TIME]/[LVSA-MEM]).
    install_scheduler_step_hook(pipe, args, lvsa_processor, rank)

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).frames[0]

    gen_duration = time.time() - t_gen
    if rank == 0:
        print(f"[generate] done in {gen_duration:.1f}s")

    mem_mb = max_memory_allocated() / 1024**2
    print(f"[rank{rank}][mem] peak allocated: {mem_mb:.0f} MB")

    # ── Save (rank 0 only) ────────────────────────────────────────────────────
    if rank == 0:
        mem_mb = max_memory_allocated() / 1024**2
        out_path = build_output_path(args, world, gen_duration, mem_mb,
                                     stem="cogvideox_generate", ext="mp4")
        os.makedirs(args.output_dir, exist_ok=True)

        t_exp = time.time()
        export_to_video(output, out_path, fps=args.fps)
        print(f"[export] {out_path}  ({time.time() - t_exp:.1f}s)")

        free, total = mem_get_info(device)
        print(
            f"[mem] peak allocated: {mem_mb:.0f} MB  |  current used: {(total - free) / 1024**2:.0f} MB"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
