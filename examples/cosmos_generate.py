"""Standalone LVSA generation for NVIDIA Cosmos 3.0 (diffusers).

Example:
  python examples/cosmos_generate.py --model /data/Cosmos3-Nano \
      --prompt "A dog running in the forest." --num-frames 189 \
      --height 720 --width 1280 --steps 35 --lvsa --output-name cosmos_1x

Requires diffusers main (>=0.39.0.dev0) for Cosmos3OmniPipeline.

Note on output attribute: Cosmos3OmniPipelineOutput uses `.video` (not `.frames`).
The pipeline's video_processor.postprocess_video already indexes batch dim 0, so
out.video is a list[PIL.Image] (for default output_type="pil") — the frames directly.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video

from lvsa.cosmos3 import install_cosmos3_lvsa, COSMOS3_REFERENCE_LATENT_FRAMES
from lvsa.device import get_device, max_memory_allocated
from examples._runner import make_step_callback


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Cosmos 3.0 video generation with optional LVSA sparse attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True, metavar="PATH",
                    help="Path or HuggingFace Hub ID of the Cosmos3OmniPipeline model.")
    ap.add_argument("--prompt", default="A dog running in the forest.",
                    help="Text prompt describing the video to generate.")
    ap.add_argument("--negative-prompt", type=str, default=None,
                    help="Negative prompt (optional).")
    ap.add_argument("--num-frames", type=int, default=189,
                    help="Number of frames to generate.")
    ap.add_argument("--height", type=int, default=720, help="Frame height (px).")
    ap.add_argument("--width", type=int, default=1280, help="Frame width (px).")
    ap.add_argument("--steps", type=int, default=35, help="Denoising steps.")
    ap.add_argument("--guidance", type=float, default=6.0, help="CFG scale.")
    ap.add_argument("--seed", type=int, default=16, help="Random seed.")
    ap.add_argument("--fps", type=int, default=16, help="Output FPS.")
    ap.add_argument("--lvsa", action="store_true",
                    help="Install LVSA block-sparse attention (else dense baseline).")
    ap.add_argument("--flashinfer", action="store_true",
                    help="Use FlashInfer fused kernel instead of SDPA. "
                         "Requires flashinfer to be installed.")
    ap.add_argument("--window-size", type=int, default=12,
                    help="Half-width of the LVSA sliding window in video frames.")
    ap.add_argument("--n-first-frames", type=int, default=4,
                    help="Number of leading video frames always included as global context.")
    ap.add_argument("--sparsity-scale", type=float, default=1.0,
                    help="Scale factor for the attention sparsity budget. "
                         "<1.0 = more sparse, >1.0 = less sparse.")
    ap.add_argument("--auto-keyframes", action="store_true",
                    help="Automatically compute key-frame-interval to match the "
                         "reference attended-frame budget. Overrides --key-frame-interval.")
    ap.add_argument("--key-frame-interval", type=int, default=None,
                    help="Interval (in LATENT frames) between periodic keyframes. "
                         "None = auto (default). Ignored when --auto-keyframes is set.")
    ap.add_argument("--reference-latent-frames", type=int,
                    default=COSMOS3_REFERENCE_LATENT_FRAMES,
                    help="Model training horizon in latent frames (default: "
                         f"{COSMOS3_REFERENCE_LATENT_FRAMES} for Cosmos3-Nano 189f).")
    ap.add_argument("--rotate-keyframes", action="store_true",
                    help="Shift periodic keyframes by 1 each denoising step, "
                         "cycling through all positions over key_frame_interval steps.")
    ap.add_argument("--no-expand-window", dest="expand_window", action="store_false",
                    help="Use adaptive (non-expanded) window bounds.")
    ap.set_defaults(expand_window=True)
    ap.add_argument("--show-mask", action="store_true",
                    help="Print the compact attention mask once at init. Requires --lvsa.")
    ap.add_argument("--show-mask-compact", nargs="?", const="once", default=None,
                    choices=["once", "step"],
                    help="Compact 1-char-per-column attention mask. "
                         "'once' (default when flag given alone) prints at init. "
                         "'step' prints at every denoising step.")
    ap.add_argument("--profile", action="store_true",
                    help="Log per-step wall-clock timing for profiling attention phases.")
    ap.add_argument("--output-dir", type=Path, default=Path("out/adhoc"),
                    help="Output directory for the generated video. Created if missing.")
    ap.add_argument("--output-name", required=True,
                    help="Output filename inside --output-dir. Extension .mp4 appended if missing.")
    return ap


def main():
    args = build_parser().parse_args()
    device = get_device(rank=0)

    print(f"[init] loading {args.model} ...")
    # enable_safety_checker=False prevents the pipeline __init__ from constructing
    # CosmosSafetyChecker (which requires the external `cosmos_guardrail` package).
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, enable_safety_checker=False
    ).to(device)

    lvsa_processor = None
    if args.lvsa:
        # --auto-keyframes overrides --key-frame-interval (None = auto in the engine)
        kfi = None if args.auto_keyframes else args.key_frame_interval
        lvsa_processor = install_cosmos3_lvsa(
            pipe.transformer,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            window_size=args.window_size,
            n_first_frames=args.n_first_frames,
            sparsity_scale=args.sparsity_scale,
            reference_latent_frames=args.reference_latent_frames,
            key_frame_interval=kfi,
            expand_window=args.expand_window,
            use_flashinfer=args.flashinfer,
        )
        if args.show_mask:
            lvsa_processor.print_attention_mask_compact()
    else:
        print("[attn] dense (no LVSA)")

    cb = make_step_callback(args, lvsa_processor, rank=0)

    print(
        f"[generate] {args.num_frames} frames  "
        f"{args.height}x{args.width}  "
        f"{args.steps} steps  guidance={args.guidance}  seed={args.seed}"
    )
    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        enable_safety_check=False,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        callback_on_step_end=cb,
    )
    wall = time.time() - t0
    peak = max_memory_allocated() / 1024**2

    # Cosmos3OmniPipelineOutput exposes .video (not .frames).
    # video_processor.postprocess_video already strips the batch dim, so
    # out.video is list[PIL.Image] directly — no [0] needed.
    frames = out.video

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name if args.output_name.endswith(".mp4") else args.output_name + ".mp4"
    out_path = args.output_dir / name
    export_to_video(frames, str(out_path), fps=args.fps)
    print(f"[cosmos] wrote {out_path} ({len(frames)} frames)")
    print(f"[BENCH] gen_s={wall:.2f} peak_mb={peak:.0f} steps={args.steps} frames={len(frames)}")


if __name__ == "__main__":
    main()
