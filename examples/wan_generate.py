"""
wan_generate.py — Video generation with Wan 2.x, supporting single and multi-GPU inference.
=================================================================================================

Usage
-----
# Single GPU (no torchrun needed)
python wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A dog running in the forest." \\
    --num-frames 81 --height 480 --width 832

# Single GPU with LVSA
python wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A dog running in the forest." \\
    --num-frames 81 --height 480 --width 832 --lvsa

# Multi-GPU context-parallel (requires torchrun)
torchrun --nproc_per_node=2 wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A dog running in the forest." \\
    --num-frames 81 --height 480 --width 832 --lvsa

# Multi-GPU, 4 GPUs, 481 frames at 720p with LVSA
torchrun --nproc_per_node=4 wan_parallel_lvsa.py \\
    --model /models/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "A timelapse of a blooming flower." \\
    --num-frames 481 --height 720 --width 1280 \\
    --lvsa --window-size 32 --n-first-frames 16 --key-frame-interval 16
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from examples._runner import (
    add_common_args, add_lvsa_args, resolve_distributed,
    make_step_callback, setup_cp, build_output_path,
)

import os
import time
import argparse

import torch
import torch.distributed as dist

from diffusers import WanPipeline
from diffusers.utils import export_to_video

from lvsa.adapters.wan import WanAdapter
from lvsa.parallel import (
    patch_rotary_emb_for_context_parallel,
    install_lvsa_processors,
    compute_and_validate_seq_len,
)


# ── Argument parsing ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wan video generation with optional LVSA and multi-GPU context parallelism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_args(parser)
    add_lvsa_args(parser)

    # Restore wan's per-model defaults for the common flags that add_common_args left unset:
    parser.set_defaults(
        steps=40,
        num_frames=81,
        height=480,
        width=832,
        fps=16,
        negative_prompt=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
    )

    # ── Wan-specific args NOT covered by the shared helpers ───────────────────

    # ── Loading ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Load pipeline with device_map='balanced' to spread model layers "
        "across all visible GPUs. Useful to fit large models without CP.",
    )

    # ── LVSA wan-specific args ─────────────────────────────────────────────────
    lvsa = parser._action_groups[-1]  # the LVSA group added by add_lvsa_args

    # We need to add these to the parser directly (not to the group) to match
    # the flat namespace, but add to a fresh group or the top-level parser.
    # Use the top-level parser for the wan-specific LVSA args:
    parser.add_argument(
        "--cp-mode", choices=["custom", "ulysses", "ring"], default="custom",
        help="Context-parallel attention mode (multi-GPU only). "
             "'custom' (default) = all_reduce of global K/V + boundary guards "
             "(no head-count constraint). 'ulysses' = all-to-all gather the full "
             "sequence, run the single-device LVSA pattern (needs num_heads %% world "
             "== 0; budget == single-GPU, no boundary-guard inflation). "
             "'ring' = true ring rotation of K/V with the LVSA block-mask per step "
             "(also no num_heads %% world constraint, unlike ulysses; compute-only "
             "savings, O(world) comm).",
    )
    parser.add_argument(
        "--reference-latent-frames",
        type=int,
        default=None,
        help="Override the model's training horizon in LATENT frames (the "
        "adapter default is 21 for Wan2.1). Set 31 for Wan2.2-TI2V-5B (121 "
        "frames / new 16x VAE) so its native 1x isn't mistaken for an extension.",
    )
    parser.add_argument(
        "--key-frame-interval",
        type=int,
        default=16,
        help="Interval (in video frames) between periodic keyframes used as "
        "global context. 0 disables periodic keyframes beyond n-first-frames. "
        "Converted to latent frames internally; auto-adjusted if too small. "
        "Ignored when --auto-keyframes is set.",
    )

    # ── RIFLEx (training-free length extrapolation via RoPE) ──────────────────
    riflex = parser.add_argument_group(
        "RIFLEx (arXiv 2502.15894)",
        "Training-free temporal RoPE rescaling. Orthogonal to LVSA; can be "
        "combined with --lvsa to stack with LVSA. Use --riflex-s > 1.0 when "
        "generating beyond the training horizon.",
    )
    riflex.add_argument(
        "--riflex",
        action="store_true",
        help="Apply RIFLEx RoPE rescaling before generation.",
    )
    riflex.add_argument(
        "--riflex-s",
        type=float,
        default=1.0,
        help="Extrapolation ratio s. s=1 is a no-op; s=2 targets 2x training "
        "length, etc.",
    )
    riflex.add_argument(
        "--riflex-k",
        type=int,
        default=None,
        help="Override auto-detected temporal frequency index k. Auto-detected "
        "if omitted (argmin_j |period_j - L|).",
    )
    riflex.add_argument(
        "--riflex-train-len",
        type=int,
        default=None,
        help="Training latent-frame count L. Defaults to the adapter's "
        "reference (21 for Wan 1.3B).",
    )

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Distributed vs single-GPU detection ────────────────────────────────────
    from lvsa.device import (
        max_memory_allocated,
        device_count,
        mem_get_info,
    )

    ctx = resolve_distributed()
    rank, world, device, distributed = ctx.rank, ctx.world, ctx.device, ctx.distributed

    # ── Create model adapter ─────────────────────────────────────────────────
    adapter = WanAdapter()

    # ── Patch standard rotary BEFORE loading (multi-GPU only) ─────────────────
    if world > 1:
        patch_rotary_emb_for_context_parallel(adapter, rank, world)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    t0 = time.time()
    if rank == 0:
        print(f"[model] loading from {args.model} ...")

    if args.balanced:
        pipe = WanPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        if rank == 0:
            print(f"[model] device_map: {pipe.hf_device_map}")
    else:
        pipe = WanPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to(device)

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

    # ── Warn about LVSA-dependent flags used without --lvsa ────────────────────
    if not args.lvsa:
        lvsa_deps = []
        if args.flashinfer:
            lvsa_deps.append("--flashinfer")
        if args.window_size != 12:
            lvsa_deps.append("--window-size")
        if args.n_first_frames != 4:
            lvsa_deps.append("--n-first-frames")
        if args.key_frame_interval != 16:
            lvsa_deps.append("--key-frame-interval")
        if args.auto_keyframes:
            lvsa_deps.append("--auto-keyframes")
        if args.show_mask:
            lvsa_deps.append("--show-mask")
        if args.show_mask_compact:
            lvsa_deps.append("--show-mask-compact")
        if args.rotate_keyframes:
            lvsa_deps.append("--rotate-keyframes")
        if lvsa_deps and rank == 0:
            print(f"[WARNING] {', '.join(lvsa_deps)} ignored without --lvsa")

    # ── RIFLEx RoPE rescaling (independent of LVSA; apply before LVSA install) ──
    if args.riflex:
        from lvsa.riflex import apply_riflex_to_wan_pipe
        info = apply_riflex_to_wan_pipe(
            pipe,
            s=args.riflex_s,
            k=args.riflex_k,
            training_length=args.riflex_train_len,
        )
        if rank == 0:
            if info["applied"]:
                print(
                    f"[riflex] applied: s={info['s']}, k={info['k']}, "
                    f"L={info['training_length']}, t_dim={info['t_dim']}"
                )
            else:
                print(f"[riflex] no-op (s={info['s']}, k={info['k']})")
    elif rank == 0 and (args.riflex_s != 1.0 or args.riflex_k is not None
                        or args.riflex_train_len is not None):
        print("[WARNING] --riflex-s/--riflex-k/--riflex-train-len ignored "
              "without --riflex")

    # ── LVSA processor installation (works on both single and multi-GPU) ───────
    lvsa_processor = None
    if args.lvsa:
        lvsa_processor = install_lvsa_processors(
            pipe, args, rank, world, adapter,
            sparsity_scale=args.sparsity_scale,
            reference_latent_frames=args.reference_latent_frames,
        )
    elif rank == 0:
        print("[attn] using standard full attention (no LVSA)")

    # ── Context-parallel plan (multi-GPU only) ────────────────────────────────
    if world > 1:
        setup_cp(adapter, pipe, world)
        if rank == 0 and getattr(pipe, "transformer_2", None) is not None:
            print("[LVSA] context-parallel enabled on transformer_2 (dual-expert)")

    if rank == 0:
        print(f"[model] loaded in {time.time() - t0:.1f}s")

    # ── Generate ──────────────────────────────────────────────────────────────
    if rank == 0:
        print(
            f"[generate] {args.num_frames} frames  "
            f"{args.height}×{args.width}  "
            f"{args.steps} steps  guidance={args.guidance}  seed={args.seed}"
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_gen = time.time()

    # ── Step callback for rotating keyframes / mask display / profiling
    step_callback = make_step_callback(args, lvsa_processor, rank)

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        callback_on_step_end=step_callback,
    ).frames[0]

    gen_duration = time.time() - t_gen
    if rank == 0:
        print(f"[generate] done in {gen_duration:.1f}s")

    mem_mb = max_memory_allocated() / 1024**2
    print(f"[rank{rank}][mem] peak allocated: {mem_mb:.0f} MB")

    # ── Save (rank 0 only) ────────────────────────────────────────────────────
    if rank == 0:
        if args.balanced:
            mem_mb = sum(
                max_memory_allocated(di) / 1024**2
                for di in range(device_count())
            )
        else:
            mem_mb = max_memory_allocated() / 1024**2
        out_path = build_output_path(args, world, gen_duration, mem_mb,
                                     stem="wan_generate", ext="mp4")
        os.makedirs(args.output_dir, exist_ok=True)

        t_exp = time.time()
        export_to_video(output, out_path, fps=args.fps)
        print(f"[export] {out_path}  ({time.time() - t_exp:.1f}s)")

        if args.balanced:
            total_peak = 0
            total_used = 0
            num_devices = device_count()
            for di in range(num_devices):
                peak_i = max_memory_allocated(di) / 1024**2
                free_i, total_i = mem_get_info(di)
                used_i = (total_i - free_i) / 1024**2
                total_peak += peak_i
                total_used += used_i
                print(f"[mem] gpu{di}: peak={peak_i:.0f} MB  used={used_i:.0f} MB")
            print(f"[mem] total: peak={total_peak:.0f} MB  used={total_used:.0f} MB")
        else:
            free, total = mem_get_info(device)
            print(
                f"[mem] peak allocated: {mem_mb:.0f} MB  |  current used: {(total - free) / 1024**2:.0f} MB"
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
