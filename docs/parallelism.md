# Multi-GPU & Parallelism Support

LVSA's sparse pattern is over the **sequence** (the `T_lat × patches_per_frame` frame grid) and is **head-independent**. So whether LVSA engages under a given parallelism axis depends entirely on **what the attention op sees**: it needs the *full frame grid* per rank (the heads may be sharded). Axes that shard weights / heads / batch / layers leave the full sequence intact → LVSA engages. Axes that shard the *sequence* (Ulysses / Ring) only work if the full sequence is reconstructed (all-to-all) before the sparse compute.

> **Legend:** ✅ verified on GPU · ⚠️ runs/engages but with a caveat (see note) · ✗ not supported (falls back to dense) · — not applicable
> **Last verified:** 2026-06-11 — full step-time smoke sweep (every model × axis, world=2, 2× horizon, Dense vs LVSA-FlashInfer) on 2×A100; standalone `ulysses`+FlashInfer bit-identical to single-GPU at world=2. **2026-06-18:** Wan LVSA confirmed on the HTTP `serve.py` path under Ulysses (backend engages; hook delegates→backend), and HunyuanVideo confirmed to use the same `self.attn` backend seam as Wan (so its hook also delegates, not dense). **Also 2026-06-18:** Wan-**14B** serve+Ulysses engages sparse + completes (scale-up of the 1.3B serve confirm); HunyuanVideo plugin Ulysses @145f engages sparse + completes with no OOM (the OOM is only at 2× horizon). Update the tables when a new axis/model/path is validated (see [Updating this doc](#updating-this-doc)).

## vLLM-Omni plugin

The plugin has two LVSA paths: the **attention backend** (the default for Wan / HunyuanVideo, selected via `diffusion_attention_config`) and the **monkey-patch hooks** (`LVSA_*_HOOK=1`, used by some serve setups). **Prefer the backend** — it runs *after* the framework's all-to-all, so it sees the full grid even under sequence-parallel.

| Axis | Shards | Backend path | Hook path |
|---|---|---|---|
| Tensor-parallel (`tensor_parallel_size`) | weights + heads | ✅ | ✅ |
| CFG-parallel (`cfg_parallel_size`) | CFG cond/uncond branches | ✅ | — |
| Data-parallel (`data_parallel_size`) | batch | ✅ | — |
| Pipeline-parallel (`pipeline_parallel_size`) | layers | ✅ | — |
| HSDP / FSDP2 (`use_hsdp`) | weights | ✅ | — |
| Ulysses sequence-parallel (`ulysses_degree`) | sequence | ✅ (full grid via the framework all-to-all; Last-verified 2026-06-18, Wan2.1-1.3B @2× ulysses, backend) | ⚠️ inline path bails but **delegates to `self.attn` → backend → re-engages** for Wan/HunyuanVideo (verified Wan serve 2026-06-18). Cosmos's hook is SP-correct (bails + delegates, verified 2026-06-18), but **Cosmos+Ulysses crashes upstream** in vllm-omni's `transformer_cosmos3.py` `unpatchify` (sequence-sharded tokens → full-grid reshape) — not an LVSA bug; the whole Cosmos SP path is non-functional |
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
| **CP `ring` mode** (`--cp-mode ring`) | 🧪 prototype | frame-scaling via true ring rotation (P2P K/V rotate + per-block LVSA mask + LSE merge). **No `num_heads % world` constraint** (shards sequence, not heads → works for any head count / odd world). Compute-only savings, O(world) comm (see `out/plans/2026-06-18-lvsa-ring-attention-study.md`). **CPU-equivalence verified** (world=1 == single-device; gloo world=2 & world=3 == world=1; world=3 ring runs where ulysses' `num_heads%world` fails). **GPU (2×A100, query-tiled `_attn_with_lse` caps memory ~5 GiB — was 48 GiB OOM): RUNS end-to-end. Wan ring == ulysses (35.92 dB = benign head-shard fp) ✅. HunyuanVideo dual-stream CP bug FIXED (2026-06-18): the text-query branch attended only the per-rank video shard under CP (all modes, `lvsa_processor.py:806`); now gathers the full video K/V (`_compute_encoder_query_attention`). After fix: single↔ulysses = bit-identical (∞ dB, was 20.5) ✅; single↔ring 29.1 dB (was 18.3 — residual is ring's LSE-merge fp at HV's 54-layer depth, not a bug).** Mask correct both (Wan 25/42 attended). **Perf: `--flashinfer` selects a fused block-sparse-LSE kernel (per-block CSR → `BlockSparseAttentionWrapper` + `merge_state`, with per-step mask/CSR caching that also speeds the pure-torch path) → **16.7× faster — 313s vs pure-torch's 1h27m (40-step/165f/world=2), now at ulysses parity** (ulysses ~300s); `--cp-mode ring` alone stays the pure-torch CPU-tested default.** Needs `total_frames % world == 0` |
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

## Per-model support (across all three paths)

Which path each model works in. **Standalone CP** = `torchrun` context-parallel; **Plugin backend** = `LVSABackend` (offline + serve, all axes); **Plugin hook** = monkey-patch serve path (engages under TP; under sequence-parallel the **Wan / HunyuanVideo** hooks *delegate to the backend*, which re-engages; **Cosmos's** hook is SP-correct but Cosmos+Ulysses crashes upstream in vllm-omni's Cosmos3 `unpatchify` — not LVSA). "all axes" = TP/CFG/PP/HSDP/Ulysses engage sparse (Ring always dense).

| Model | Standalone CP | Plugin backend | Plugin hook (serve) |
|---|---|---|---|
| Wan 2.1 1.3B | ✅ custom + ulysses (SDPA + FlashInfer) | ✅ all axes | ✅ TP; SP→delegates to backend |
| Wan 2.1 14B | ✅ custom + ulysses | ✅ all axes (big-model via TP) | ✅ TP; SP→delegates to backend |
| Wan 2.2 A14B | ✅ custom + ulysses — **both experts** wired (LVSA + CP on `transformer_2` too; verified 1.23× flat, matches 14B) | ✅ all axes | ✅ TP; SP→delegates to backend |
| Wan 2.2 TI2V-5B | ✅ custom + ulysses (per-token-timestep CP fix verified on 2×A100; needs `720p, --reference-latent-frames 31`) | ✅ all axes — needs `720p, ref_lat 31, ppf 880` | ✅ TP; SP→delegates to backend |
| HunyuanVideo 1.5 | ✅ custom + ulysses (16 heads → world ∈ {1,2,4,8,16}). **Dual-stream text-query CP fixed 2026-06-18 — ulysses now bit-identical to single-GPU (∞ dB); ring 29 dB (LSE-merge fp).** All blocks are dual-stream MMDiT (no single-stream); LVSA installs on all | ✅ all axes — Ulysses **verified: engages sparse + completes @145f** on 2×A100 (`\|G\|=19, video_seq=57720, text=16`, 2026-06-18); **OOMs only at 2× horizon** (261-frame decode) — a memory ceiling, not a logic failure (only HSDP fits the full 2× clip) | ✅ TP; SP→delegates to backend |
| Cosmos 3.0 | ⚠️ **single-GPU only** (diffusers processor swap; CP deferred) | `self.attn`=FrameworkAttention seam exists, but Ulysses path crashes upstream (see hook) | ✅ **hook: TP/CFG/PP/HSDP** (single-GPU hook engages sparse, verified: `\|G\|=25 P=390 seq=19110`) · ✗ **Ulysses — crashes upstream** (vllm-omni `transformer_cosmos3.py:1155 unpatchify`, sequence-sharded tokens; hook itself is SP-correct) · ✗ Ring |
| CogVideoX 5B | experimental (correctness only, no speedup) | — | — |

**Reading it:** Wan 2.1 (1.3B/14B) is the "works everywhere" case. Standalone CP is Wan-shaped (HV custom-only; TI2V-5B broken; A14B no win; Cosmos single-GPU only). The plugin **backend** is the most universal — every model engages under every axis except Ring, given the right `ref_lat`/`ppf`. **Cosmos** is split: *standalone* (`lvsa/cosmos3.py`) is a **diffusers processor swap → single-GPU only** (CP deferred). In the **plugin**, though, `Cosmos3CrossAttention.forward` routes through `self.attn = FrameworkAttention` with an explicit **SP-aware `_forward_sp`** (joint K/V + Ulysses all-to-all) — the same seam as Wan / HunyuanVideo. Under SP the Cosmos hook bails and delegates correctly (verified 2026-06-18), **but Cosmos+Ulysses crashes upstream** — vllm-omni's `transformer_cosmos3.py:1155 unpatchify` reshapes the sequence-sharded token tensor to the full spatial grid and throws (an SP bug in vllm-omni's Cosmos3, independent of LVSA; the LVSA hook had already delegated). It engages under the non-sequence-sharding axes (TP/CFG/PP/HSDP) and at single-GPU (hook verified sparse: `|G|=25, P=390, seq=19110`). Bottom line: **Cosmos has no working sequence-parallel path anywhere** — standalone has no CP, and the plugin's Ulysses path is broken upstream. Best speedup axes for the other models: **Ulysses + HSDP**; **TP is weakest** but is what unlocks the big models by sharding weights.

## What does NOT work

- **Plugin Ring-SP + LVSA** — the framework dispatches ring traffic to its own kernel and bypasses the LVSA backend (`attention/layer.py:245`), so LVSA never engages → dense. Use **Ulysses** for plugin sequence-parallel. (Standalone now has a working **`--cp-mode ring`** prototype — variant-A block-sparse ring; CPU-verified, GPU pending — see the ring-attention study. Note ring gives compute-only savings, not a comm win, so Ulysses remains preferred where its `num_heads%world` constraint is satisfiable.)
- **Cosmos under sequence-parallel** — *standalone*: single-GPU only (diffusers processor swap, no CP). *Plugin*: the LVSA hook bails and delegates correctly, **but Cosmos+Ulysses crashes upstream** in vllm-omni's `transformer_cosmos3.py:1155 unpatchify` (sequence-sharded tokens → full-grid reshape, `RuntimeError: shape '[1,1,16,16,2,2,48]' invalid for input of size 24576`) — a vllm-omni Cosmos3 SP bug, **not** LVSA. So Cosmos has no working frame-scaling path. (Wan / HunyuanVideo, by contrast, verifiably delegate to the backend and re-engage under SP.)
- **Standalone weight-sharding (TP / FSDP)** — not implemented. Route big models that don't fit one GPU (e.g. Wan 2.2-A14B) through the **plugin** (TP / HSDP).
- **`ulysses` mode with `num_heads % world ≠ 0`** — raises a clear `ValueError`; use `custom` mode for those layouts (e.g. Wan-40 on world = 3 or 6).
- ~~**Standalone CP for Wan 2.2-TI2V-5B**~~ — **FIXED + verified on 2×A100 (2026-06-11).** Root cause: TI2V's per-token timestep conditioning (`temb`/`timestep_proj` are `[B, seq, …]`) was not sharded by the CP entry-split → every block's `norm1(hidden) * (1 + scale_msa)` crashed shard-vs-full-seq. The Wan adapter `_cp_plan` now splits the root `timestep` input (mirrors diffusers-main's native plan; Wan 2.1's 1-D timestep unaffected).
- ~~**`ulysses` CP standalone for HunyuanVideo**~~ — **FIXED + verified on 2×A100 (2026-06-11).** `--cp-mode {custom,ulysses}` is now on `hunyuan_generate.py` too. HV 1.5 has 16 heads → `ulysses` needs `world ∈ {1, 2, 4, 8, 16}`.

## Why this shape (one paragraph)

The block-sparse mask is computed per **frame** and applied identically across **heads**, so the attention kernel only needs the full `T_lat × P` token grid — the head dimension can be sharded freely. Tensor-parallel, HSDP, pipeline, CFG and data parallelism never touch the sequence dimension at the attention call, so LVSA "just works" (the kernel sees full seq, fewer heads/weights). Ulysses *does* shard the sequence, but the framework (plugin) or an explicit all-to-all (standalone `ulysses` mode) reconstructs the full grid head-sharded **before** the sparse compute, so LVSA engages there too. Ring never reconstructs the full grid (it streams K/V), which is fundamentally incompatible with a contiguous block-sparse pattern.

## Updating this doc

When you validate a new axis / model / path, add a row marked ✅ and bump the **Last verified** date. The decisive check at runtime is whether the attention op receives the full grid (LVSA engages) or a sequence fragment (dense fallback):

- **Engaged:** the run log shows one of — `[LVSA] Geometry detected: … video_seq=N` (plugin Wan/HV backend), `[LVSA] engaged (cosmos3): … -> SPARSE` (Cosmos hook), or `attended_per_frame=N/T` (standalone) — with `video_seq`/`local_seq == T_lat × P`.
- **Fell back:** no engage line, or `[LVSA-FALLBACK] … reason=sequence_parallel|geometry_detect|…`. **Note:** a benign warmup `[LVSA-FALLBACK] … seq_len=1024` (a small dummy probe) appears even on successful plugin runs — only a fallback at the *real* grid size (`seq_len == T_lat × P`) means LVSA actually fell back.

A 1-step run with the same `--seed` on single-GPU vs the parallel config is enough to confirm correctness: single-GPU is deterministic, and a correct parallel path diverges only by the benign head-shard kernel-fp (~36–37 dB PSNR on the decoded frames). A much larger divergence indicates a real bug, not fp drift.
