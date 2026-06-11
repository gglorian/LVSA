# Multi-GPU & Parallelism Support

LVSA's sparse pattern is over the **sequence** (the `T_lat × patches_per_frame` frame grid) and is **head-independent**. So whether LVSA engages under a given parallelism axis depends entirely on **what the attention op sees**: it needs the *full frame grid* per rank (the heads may be sharded). Axes that shard weights / heads / batch / layers leave the full sequence intact → LVSA engages. Axes that shard the *sequence* (Ulysses / Ring) only work if the full sequence is reconstructed (all-to-all) before the sparse compute.

> **Legend:** ✅ verified on GPU · ✗ not supported (falls back to dense) · — not applicable
> **Last verified:** 2026-06-10 — Wan 2.1 1.3B / 14B on 2×A100 (plugin + standalone); standalone `ulysses`+FlashInfer bit-identical to single-GPU at world=2. Update the tables when a new axis/model/path is validated (see [Updating this doc](#updating-this-doc)).

## vLLM-Omni plugin

The plugin has two LVSA paths: the **attention backend** (the default for Wan / HunyuanVideo, selected via `diffusion_attention_config`) and the **monkey-patch hooks** (`LVSA_*_HOOK=1`, used by some serve setups). **Prefer the backend** — it runs *after* the framework's all-to-all, so it sees the full grid even under sequence-parallel.

| Axis | Shards | Backend path | Hook path |
|---|---|---|---|
| Tensor-parallel (`tensor_parallel_size`) | weights + heads | ✅ | ✅ |
| CFG-parallel (`cfg_parallel_size`) | CFG cond/uncond branches | ✅ | — |
| Data-parallel (`data_parallel_size`) | batch | ✅ | — |
| Pipeline-parallel (`pipeline_parallel_size`) | layers | ✅ | — |
| HSDP / FSDP2 (`use_hsdp`) | weights | ✅ | — |
| Ulysses sequence-parallel (`ulysses_degree`) | sequence | ✅ (full grid via the framework all-to-all) | ✗ dense (intercepts pre-all-to-all) |
| Ring sequence-parallel (`ring_degree`) | sequence (P2P K/V) | ✗ dense | ✗ dense |
| Expert-parallel | MoE experts | — | — |

*Expert-parallel is N/A for Wan 2.2-A14B — it is a boundary-ratio dual-transformer (two timestep-switched experts), not a token-routed MoE.*

**Big models:** tensor-parallel (and/or HSDP) shard the weights, so e.g. **Wan 2.2 14B / A14B run on 4–8 GPUs via `tensor_parallel_size`** with LVSA engaged and sparse. (Verified: Wan2.1-14B at TP=2 → sparse LVSA, 21/41 attended, generated.)

Drive any axis from the offline runner:
```bash
.venv-vllm-main/bin/python lvsa-vllm-omni/examples/offline_lvsa.py \
    --family wan --model /models/Wan2.2-T2V-A14B-Diffusers \
    --tp 4 --backend flashinfer --num-frames 161 --steps 40 \
    --prompt "..." --output-name demo
# also: --ulysses N   and   --omni-kw cfg_parallel_size=2 / pipeline_parallel_size=2 / use_hsdp=true
```

## Standalone (diffusers, `torchrun`)

| Capability | Status | Notes |
|---|---|---|
| Single-GPU | ✅ | |
| `device_map="balanced"` (`--balanced`) | ✅ | weight-shard to **fit** a big model on N GPUs; **no** frame-scaling (single process, world=1) |
| **CP `custom` mode** (`--cp-mode custom`, default) | ✅ | frame-scaling: seq-shard + `all_reduce` of global K/V + boundary guards; budget slightly inflated at rank seams |
| **CP `ulysses` mode** (`--cp-mode ulysses`) | ✅ | frame-scaling: all-to-all gather → **exact** single-device pattern (no boundary-guard inflation); needs `num_heads % world == 0` |
| Both CP modes × **SDPA / FlashInfer** (`--flashinfer`) | ✅ | both backends work in both CP modes; ulysses+FlashInfer verified bit-identical to single-GPU at world=2 |
| Tensor-parallel / FSDP weight-sharding | ✗ | not available standalone — use the plugin for weight-sharded big models |

```bash
# Frame-scaling across 2 GPUs with the exact single-device sparse pattern (FlashInfer):
torchrun --nproc_per_node=2 examples/wan_generate.py \
    --model /models/Wan2.1-T2V-1.3B-Diffusers --num-frames 321 \
    --lvsa --flashinfer --auto-keyframes --cp-mode ulysses --prompt "..." --output-name demo
```

`custom` vs `ulysses` — both correct, both scale frames, **both backends (SDPA + FlashInfer)**:
- **`ulysses`** reproduces the single-GPU sparse pattern exactly (all-to-all is lossless; no seam inflation), verified to track single-GPU within head-shard fp (~37 dB; bit-identical at the attention op). Requires `num_heads % world == 0` (Wan = 40 → world ∈ {1,2,4,5,8,10,20,40}). FlashInfer runs the same block-sparse CSR as single-GPU on each rank's head-shard after the all-to-all.
- **`custom`** has no head-count constraint, but boundary guards make it a slightly denser approximation near rank seams.

## What does NOT work

- **Ring-SP + LVSA** (any path) — Ring streams K/V point-to-point and never materializes the full sequence → LVSA falls back to dense. Use **Ulysses** for sequence-parallel.
- **Plugin hook path under sequence-parallel** — falls back to dense (it intercepts before the all-to-all); use the **backend** path, which engages.
- **Standalone weight-sharding (TP / FSDP)** — not implemented. Route big models that don't fit one GPU (e.g. Wan 2.2-A14B) through the **plugin** (TP / HSDP).
- **`ulysses` mode with `num_heads % world ≠ 0`** — raises a clear `ValueError`; use `custom` mode for those layouts (e.g. Wan-40 on world = 3 or 6).

## Why this shape (one paragraph)

The block-sparse mask is computed per **frame** and applied identically across **heads**, so the attention kernel only needs the full `T_lat × P` token grid — the head dimension can be sharded freely. Tensor-parallel, HSDP, pipeline, CFG and data parallelism never touch the sequence dimension at the attention call, so LVSA "just works" (the kernel sees full seq, fewer heads/weights). Ulysses *does* shard the sequence, but the framework (plugin) or an explicit all-to-all (standalone `ulysses` mode) reconstructs the full grid head-sharded **before** the sparse compute, so LVSA engages there too. Ring never reconstructs the full grid (it streams K/V), which is fundamentally incompatible with a contiguous block-sparse pattern.

## Updating this doc

When you validate a new axis / model / path, add a row marked ✅ and bump the **Last verified** date. The decisive check at runtime is whether the attention op receives the full grid (LVSA engages) or a sequence fragment (dense fallback):

- **Engaged:** the run log shows `[LVSA-MASK] … |G|=… kfi=…` (plugin) or `attended_per_frame=N/T` (standalone), with `video_seq`/`local_seq == T_lat × P`.
- **Fell back:** no mask line, or `[LVSA-FALLBACK] … reason=sequence_parallel|geometry_detect|…`.

A 1-step run with the same `--seed` on single-GPU vs the parallel config is enough to confirm correctness: single-GPU is deterministic, and a correct parallel path diverges only by the benign head-shard kernel-fp (~36–37 dB PSNR on the decoded frames). A much larger divergence indicates a real bug, not fp drift.
