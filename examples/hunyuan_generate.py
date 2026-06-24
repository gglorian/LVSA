"""
hunyuan_generate.py — Video generation with HunyuanVideo-1.5, supporting single and multi-GPU inference.
================================================================================================================

**Status: experimental.**  HunyuanVideo-1.5 is **all dual-stream (MMDiT) blocks** — there are
no single-stream blocks, and LVSA installs on *every* block (video stream = sparse LVSA,
text stream = full attention over the joint sequence).  Under multi-GPU CP the dual-stream
**text-query** branch now gathers the full video K/V across ranks (see
``DistributedLVSAProcessor._compute_encoder_query_attention``); before the 2026-06-18 fix it
attended only the per-rank video shard, so HV diverged from single-GPU in every cp_mode.

Usage
-----
# Single GPU (no torchrun needed)
python hunyuan_parallel_lvsa.py \\
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
    --prompt "A dog running in the forest." \\
    --num-frames 61 --height 480 --width 832

# Single GPU with LVSA
python hunyuan_parallel_lvsa.py \\
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
    --prompt "A dog running in the forest." \\
    --num-frames 61 --height 480 --width 832 --lvsa

# Multi-GPU context-parallel (requires torchrun)
torchrun --nproc_per_node=2 hunyuan_parallel_lvsa.py \\
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
    --prompt "A dog running in the forest." \\
    --num-frames 61 --height 480 --width 832 --lvsa
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

from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

from lvsa.adapters.hunyuan_video import HunyuanVideoAdapter
from lvsa.parallel import (
    patch_rotary_emb_for_context_parallel,
    install_lvsa_processors,
    compute_and_validate_seq_len,
)


# ── Argument parsing ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HunyuanVideo generation with optional LVSA and multi-GPU context parallelism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_args(parser)
    add_lvsa_args(parser)

    # Restore hunyuan's per-model defaults for the common flags that add_common_args left unset:
    parser.set_defaults(
        steps=50,
        num_frames=61,
        height=480,
        width=832,
        fps=24,
        guidance=6.0,
        seed=16,
        negative_prompt="",
    )

    # ── Hunyuan-specific args NOT covered by the shared helpers ───────────────────

    parser.add_argument(
        "--output-latent",
        action="store_true",
        help="Save the denoised latent tensor (.pt) instead of decoding to mp4. "
        "Use for long ratios where VAE decode OOMs; the latent can be decoded "
        "offline on higher-memory hardware.",
    )

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
    from lvsa.device import max_memory_allocated, mem_get_info

    ctx = resolve_distributed()
    rank, world, device, distributed = ctx.rank, ctx.world, ctx.device, ctx.distributed

    # ── Create model adapter ─────────────────────────────────────────────────
    adapter = HunyuanVideoAdapter()

    # ── Patch standard rotary BEFORE loading (multi-GPU only) ─────────────────
    if world > 1:
        patch_rotary_emb_for_context_parallel(adapter, rank, world)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    t0 = time.time()
    if rank == 0:
        print(f"[model] loading from {args.model} ...")

    pipe = HunyuanVideo15Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # ── Enable VAE tiling to avoid OOM during decode ─────────────────────────
    # HunyuanVideo's 3D VAE is very memory-hungry; without tiling, decoding
    # 121+ frames at 480×832 requires ~12 GB for a single conv3d layer.
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
        pipe.vae_scale_factor_spatial,
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
    # HunyuanVideo15Pipeline does not support callback_on_step_end, so we
    # hook into the scheduler's step() method to inject per-step logic.
    install_scheduler_step_hook(pipe, args, lvsa_processor, rank)

    # HunyuanVideo-1.5 uses a guider (ClassifierFreeGuidance) instead of
    # guidance_scale at runtime.  Set the guidance scale on the guider.
    if hasattr(pipe, "guider") and pipe.guider is not None:
        pipe.guider = pipe.guider.new(guidance_scale=args.guidance)

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=generator,
    )
    if args.output_latent:
        pipe_kwargs["output_type"] = "latent"
        pipe_out = pipe(**pipe_kwargs)
        output = pipe_out.frames if hasattr(pipe_out, "frames") else pipe_out
        if hasattr(output, "__getitem__") and not torch.is_tensor(output):
            output = output[0]
    else:
        output = pipe(**pipe_kwargs).frames[0]

    gen_duration = time.time() - t_gen
    if rank == 0:
        print(f"[generate] done in {gen_duration:.1f}s")

    mem_mb = max_memory_allocated() / 1024**2
    print(f"[rank{rank}][mem] peak allocated: {mem_mb:.0f} MB")

    # ── Save (rank 0 only) ────────────────────────────────────────────────────
    if rank == 0:
        mem_mb = max_memory_allocated() / 1024**2
        ext = "pt" if args.output_latent else "mp4"
        out_path = build_output_path(args, world, gen_duration, mem_mb,
                                     stem="hunyuan_generate", ext=ext)
        os.makedirs(args.output_dir, exist_ok=True)

        t_exp = time.time()
        if args.output_latent:
            torch.save({"latent": output.detach().cpu(),
                        "num_frames": args.num_frames,
                        "height": args.height,
                        "width": args.width,
                        "prompt": args.prompt,
                        "seed": args.seed,
                        "steps": args.steps}, out_path)
            print(f"[save-latent] {out_path}  shape={tuple(output.shape)}  ({time.time() - t_exp:.1f}s)")
        else:
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
