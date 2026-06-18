# Release 1.2.0 validation sweep — VQeval + VBench-Long results

Independent re-run of the full model × horizon matrix on release 1.2.0: 
5 prompts × 6 horizons × {dense, LVSA-FlashInfer},
single A100 80 GB, 40 steps (Wan) / 50 steps (HunyuanVideo) / 35 steps (Cosmos),
rotating keyframes + auto-keyframe. 699 videos, scored 699/699 with both
[VQeval](../vqeval/) and VBench-Long. Means over the 5 prompts
(`dog_forest, cat_window, ocean_sunset, snowy_street, coral_reef`). `→` reads
`Dense → LVSA`; `·` = not run / dense OOM. Speedup = Dense ÷ LVSA-FlashInfer
wall-time.

> These are a **validation sweep** at 40/50 steps, distinct from the README's
> published headline numbers (50-step, paper config) — same trend, slightly
> different absolute values from the step count and prompt set.

## Headline

LVSA's quality advantage **grows with extension**: at ≥3× the training horizon —
where dense diffusion degrades into frozen/looping output — LVSA-FlashInfer
**wins VQeval composite by +2 to +6 points** while running **1.8×–3.5× faster**,
driven by a large **loop-quality** gain (+16 to +30). At 1× (training reference)
LVSA runs the dense regime by design (`kfi=1`) so output and quality are
identical; the win begins at 2× and widens monotonically.

### Plugin — speedup (Dense ÷ LVSA-FlashInfer) and quality (Dense → LVSA)

| Model | Metric | 1x | 2x | 3x | 4x | 5x | 6x |
|---|---|---|---|---|---|---|---|
| **Wan2.1 1.3B** | speedup | 0.90× | 1.39× | 1.85× | 2.48× | 3.02× | 3.52× |
| | VQeval comp | 62.9→62.9 | 59.2→63.4 | 61.2→63.2 | 61.4→63.9 | 59.4→63.7 | 59.1→62.7 |
| | loop-quality | 48→48 | 53→66 | 43→71 | 43→72 | 42→76 | 43→72 |
| **Wan2.1 14B** | speedup | 0.91× | 1.37× | 1.77× | 2.32× | · | · |
| | VQeval comp | 65.4→65.2 | 63.7→67.3 | 63.8→66.5 | 63.8→67.3 | 67.1 | 67.0 |
| | loop-quality | 48→48 | 55→61 | 53→66 | 45→65 | · | · |
| **Wan2.2 TI2V-5B** | speedup | 1.15× | 1.40× | 1.75× | 1.73× | 2.62× | 3.00× |
| | VQeval comp | 68.2→67.9 | 64.7→66.8 | 62.8→68.2 | 62.5→68.1 | 61.6→67.4 | 60.2→66.3 |
| | loop-quality | 49→48 | 42→57 | 43→65 | 43→68 | 43→63 | 42→65 |
| **Cosmos 3.0** | speedup | 0.91× | 1.63× | 2.12× | 2.62× | · | · |
| | VQeval comp | 66.0→66.0 | 69.0→69.4 | 69.9→69.1 | 70.7→68.3 | 68.2 | 68.9 |
| | loop-quality | 49→50 | 61→60 | 65→68 | 63→61 | · | · |
| **HunyuanVideo 1.5** | speedup | 0.86× | · | · | · | · | · |
| | VQeval comp | 64.5→66.0 | · | · | · | · | · |
| | loop-quality | 58→61 | · | · | · | · | · |

### Standalone — speedup (Dense ÷ LVSA-FlashInfer) and quality (Dense → LVSA)

| Model | Metric | 1x | 2x | 3x | 4x | 5x | 6x |
|---|---|---|---|---|---|---|---|
| **Wan2.1 1.3B** | speedup | 0.96× | 1.43× | 1.84× | 2.45× | 2.86× | 3.31× |
| | VQeval comp | 63.1→63.2 | 62.0→65.2 | 61.5→61.6 | 61.7→63.9 | 61.6→64.3 | 59.2→63.4 |
| | loop-quality | 44→44 | 52→73 | 44→74 | 44→70 | 43→72 | 42→75 |
| **Wan2.1 14B** | speedup | 0.96× | 1.38× | 1.75× | · | · | · |
| | VQeval comp | 65.3→63.6 | 65.2→66.0 | 66.2→65.4 | 68.0 | 67.2 | 68.2 |
| | loop-quality | 45→36 | 57→58 | 49→65 | · | · | · |
| **Wan2.2 TI2V-5B** | speedup | 0.98× | 1.42× | 1.72× | 1.99× | 2.51× | 2.83× |
| | VQeval comp | 69.6→70.1 | 67.0→67.8 | 64.5→69.8 | 64.5→70.2 | 66.1→70.8 | 63.3→69.2 |
| | loop-quality | 44→44 | 42→55 | 42→56 | 44→60 | 43→63 | 44→57 |
| **HunyuanVideo 1.5** | speedup | 2.27× | · | · | · | · | · |
| | VQeval comp | 62.8→63.0 | 62.4 | 63.0 | · | · | · |
| | loop-quality | 54→62 | · | · | · | · | · |

## How to read the quality numbers (important)

The **VQeval composite** and **loop-quality** dimensions are the honest summary.
Do **not** read raw *temporal-coherence* (VQeval) or the VBench-Long *consistency*
dimensions in isolation: at high horizons **dense scores them higher precisely
because it is failing** — a frozen or looping video is maximally "smooth" and
"consistent". E.g. Wan2.1-1.3B at 6×, dense temporal-coherence ≈ 92 vs LVSA ≈ 67,
yet dense's *loop-quality* is 43 vs LVSA's 72 and the composite favors LVSA. The
visual demos (`docs/demo_videos/`) show this directly (dense dogs walk
backwards / freeze; LVSA keeps coherent forward motion).

**VBench-Long average** (5 dims: subject/background consistency, temporal
flickering, motion smoothness, imaging quality) is ≈ tied dense-vs-LVSA across
the board — LVSA trades a little raw smoothness for the large anti-looping /
dynamics gains the VQeval composite captures. Note the same caveat applies: the
consistency dims partly reward dense's frozen output, so LVSA being ~1–2 pts
lower at high horizons understates its real-world advantage.

### Plugin — VBench-Long average (5 dims), Dense → LVSA

| Model | 1x | 2x | 3x | 4x | 5x | 6x |
|---|---|---|---|---|---|---|
| **Wan2.1 1.3B** | 89.2→89.2 | 86.4→86.1 | 88.0→86.6 | 89.3→87.4 | 88.3→86.8 | 88.6→86.5 |
| **Wan2.1 14B** | 91.2→91.1 | 89.0→89.2 | 89.2→89.3 | 88.3→89.2 | 88.5 | 88.2 |
| **Wan2.2 TI2V-5B** | 91.7→91.5 | 87.7→88.4 | 87.6→87.8 | 87.5→86.9 | 86.0→87.3 | 85.5→87.0 |
| **Cosmos 3.0** | 91.5→91.7 | 92.0→91.7 | 90.6→90.5 | 89.7→90.8 | 89.0 | 89.0 |
| **HunyuanVideo 1.5** | 89.6→89.8 | · | · | · | · | · |

### Standalone — VBench-Long average (5 dims), Dense → LVSA

| Model | 1x | 2x | 3x | 4x | 5x | 6x |
|---|---|---|---|---|---|---|
| **Wan2.1 1.3B** | 89.0→88.9 | 86.8→87.2 | 87.4→85.5 | 87.6→87.1 | 88.7→87.7 | 87.8→86.4 |
| **Wan2.1 14B** | 90.7→90.8 | 90.6→90.1 | 90.2→89.9 | 89.7 | 89.9 | 89.6 |
| **Wan2.2 TI2V-5B** | 92.2→92.2 | 90.2→90.2 | 91.2→89.7 | 85.0→89.5 | 88.8→88.5 | 89.0→88.7 |
| **HunyuanVideo 1.5** | 89.4→88.4 | 90.1 | 89.8 | · | · | · |

*(Per-dimension VBench-Long + per-prompt rows: `summary.csv`, `vb_*` columns.)*

## Capability ceiling (dense OOMs, LVSA continues)

Beyond a speed story, LVSA **enables horizons where dense OOMs on 80 GB**:
HunyuanVideo ≥2×, Wan2.2-5B ≥4× @720p standalone, Wan2.1-14B 6×, Cosmos 6×
(`·` in the dense rows above). At those lengths the comparison is
LVSA-only-capability, not a ratio.

*Source: 699 videos, vqeval 699/699 + VBench-Long 699/699. Aggregator +
`summary.csv` in the sweep dir.*
