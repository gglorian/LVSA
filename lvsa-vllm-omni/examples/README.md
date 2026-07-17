# LVSA × vllm-omni — Examples

Two ways to use the LVSA plugin with vllm-omni:

| Mode | Files | Use case |
|---|---|---|
| **Online** (HTTP server) | `serve_wan.sh`, `serve_hunyuan.sh`, `online_client.py`, `online_curl.sh` | Production: serve generation behind an OpenAI-compatible API |
| **Offline** (in-process Python) | **`offline_lvsa.py`** | Scripting / batch eval / quick smoke tests — one script for all model families |

Both modes engage LVSA through `LVSA_*` environment variables — see [`../README.md`](../README.md) for the full env-var reference.

## Prerequisites

`vllm==0.24.0` pins `torch==2.11` (CUDA 13), which is incompatible with the
torch the standalone LVSA engine uses (`2.12`). **Use a separate venv for
vllm-omni work** so it doesn't break the standalone engine in your main `.venv`.

```bash
# From the LVSA repo root:
uv venv .venv-vllm --python 3.12        # separate venv, dedicated to vllm-omni
source .venv-vllm/bin/activate

uv pip install -e .                                # core lvsa
uv pip install -e lvsa-vllm-omni/                  # this plugin

# vllm-omni 0.24.0rc1 is a release candidate — install it from the git tag to match
# vllm 0.24.0. Install vllm FIRST (vllm-omni does not declare vllm as a
# dependency, so a lone vllm-omni install pulls no vllm).
uv pip install "vllm==0.24.0"
uv pip install --no-build-isolation \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.24.0rc1"
```

> **Two-venv rule**: keep `.venv` for standalone (`examples/wan_generate.py`,
> CPU tests, benchmark scripts) and `.venv-vllm` for vllm-omni serving + the
> scripts in this folder. The `.venv-vllm` torch is 2.11 / CUDA 13, so
> standalone GPU work that depends on torch 2.12 features won't run there.
>
> **Version pairing is symmetric.** vllm-omni 0.24.0rc1 is a release candidate, so the
> pair is `vllm==0.24.0` + `vllm-omni==0.24.0rc1`. vllm-omni is built from the git
> tag `@v0.24.0rc1` (also pinned in [`../Dockerfile`](../Dockerfile)).
> To stay on the older pip-installable line, use the `release/v0.18.x` branch
> (`vllm==0.18.0` + `vllm-omni==0.18.0`).

## Online — server + HTTP client

### 1. Start the server

```bash
# Wan 2.x
examples/serve_wan.sh /path/to/Wan2.1-T2V-1.3B-Diffusers
#  → http://localhost:8098

# HunyuanVideo 1.5
examples/serve_hunyuan.sh /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v
```

Override env vars as needed:

```bash
PORT=8200 TP=2 \
LVSA_SPARSITY_SCALE=0.5 \
examples/serve_wan.sh /path/to/Wan2.1-T2V-1.3B-Diffusers
```

The wrapper sets the right `LVSA_*` env vars per model family before forwarding to `python -m lvsa_vllm_omni.serve`.

### 2. Submit a generation request

**Python client** (recommended):

```bash
python examples/online_client.py \
    --host localhost:8098 \
    --prompt "A dog running in the forest." \
    --num-frames 81 --size 832x480 --fps 16 \
    --steps 40 --guidance 4.0 --seed 42 \
    --output dog.mp4
```

**Or bash + curl + jq**:

```bash
PROMPT="A dog running in the forest." \
NUM_FRAMES=81 \
OUTPUT=dog.mp4 \
examples/online_curl.sh
```

Both submit the request to `POST /v1/videos`, poll the job status, and download the resulting mp4 once `status == "completed"`.

### Request payload reference

vllm-omni accepts form-data (not JSON) on `POST /v1/videos`. Common fields:

| Field | Notes |
|---|---|
| `prompt` (required) | Text description |
| `num_frames` or `seconds` | Generation length |
| `size` | `WIDTHxHEIGHT` (e.g. `832x480`) |
| `fps` | Frame rate (16-24 typical) |
| `num_inference_steps` | Denoising steps (40-50) |
| `guidance_scale` | CFG (Wan 2.2 high-noise stage, or HunyuanVideo single stage) |
| `guidance_scale_2` | Wan 2.2 low-noise stage (omit for Wan 2.1 / HV) |
| `flow_shift` | HunyuanVideo: 5.0 @ 480p, 9.0 @ 720p typical |
| `boundary_ratio` | Wan 2.2 only (~0.875 typical) |
| `seed` | RNG seed |
| `negative_prompt` | Optional |

## Offline — direct Python API (`offline_lvsa.py`)

For when you want to script generation without standing up an HTTP server (e.g.
batch eval, smoke tests, integration into a larger pipeline). **`offline_lvsa.py`
is the single canonical offline runner** for every model family — pick the family
with `--family {wan,hunyuan,cosmos}` and the attention path with `--backend`.

```bash
# Wan 2.x (Wan 2.1 1.3B/14B, Wan 2.2-5B)
python examples/offline_lvsa.py --family wan \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 81 --steps 40 --guidance 5.0 --flow-shift 12 --seed 42 \
    --output-name dog_offline

# HunyuanVideo 1.5
python examples/offline_lvsa.py --family hunyuan \
    --model /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "A dog running in the forest." \
    --num-frames 129 --steps 50 --guidance 6.0 --flow-shift 5 --seed 42 \
    --output-name dog_hv_offline

# Cosmos 3.0 (720p; LVSA via the attention-backend seam-swap)
python examples/offline_lvsa.py --family cosmos \
    --model /path/to/Cosmos3-Nano \
    --num-frames 189 --height 720 --width 1280 --steps 35 --guidance 6.0 \
    --output-name cosmos_offline
```

### Backends and key flags

| Flag | Values / default | Meaning |
|---|---|---|
| `--family` | `wan` \| `hunyuan` \| `cosmos` (required) | Model family — picks geometry + the LVSA integration path |
| `--backend` | `flashinfer` (default) \| `sdpa` \| `dense` | `flashinfer`/`sdpa` = LVSA; **`dense` = no LVSA (baseline)** |
| `--num-frames` | int | `(floor(N×ref_lat)−1)×4+1` for horizon N (ref_lat: wan2.1=21, wan2.2-5b=31, hunyuan=33, cosmos=48) |
| `--flow-shift` | float (omit → pipeline default) | wan 480p=12 / 720p=5, hunyuan=5; see `../README.md` |
| `--no-rotate` | flag | Disable rotating keyframes (rotation is **on** by default) |
| `--sparsity-scale` | float (default 1.0) | `LVSA_SPARSITY_SCALE` — <1 = more sparse, >1 = less (scales the auto-keyframe target). LVSA-only |
| `--eager` | flag | `enforce_eager` — disable torch.compile (lower peak mem, slower/step) |
| `--offload` | `none` (default) \| `cpu` \| `layerwise` | DiT offload to free VRAM for the VAE decode on long clips |
| `--steps` `--guidance` `--seed` `--height` `--width` `--fps` `--output-dir` `--output-name` | | standard; `--output-name` required |

> Replaces the old per-family `offline_wan.py` / `offline_hunyuan.py`. The
> `--no-lvsa` baseline is now `--backend dense`. (`--tensor-parallel-size` is not
> exposed — `offline_lvsa.py` runs single-GPU, TP=1; use the **server** scripts
> for tensor parallelism.)

What it does:
1. Sets the `LVSA_*` env vars (latent-frame counts, backend, keyframes; `LVSA_COSMOS3_BACKEND=1` for cosmos) **before** importing `vllm_omni`.
2. Calls `register_lvsa_backend()` to wire LVSA into vllm-omni's backend enum. For `wan`/`hunyuan` it selects the LVSA attention **backend** (`diffusion_attention_config={self: LVSA}`); for `cosmos` it installs the same attention-backend seam-swap on `Cosmos3CrossAttention.attn` (`cosmos3_backend.py`) — engages sparse single-GPU **and under Ulysses SP** (GPU-verified 2026-07-07).
3. Instantiates `Omni(...)`, runs `generate(...)`, and emits a parseable `[BENCH] gen_s=… peak_mb=…` line.
4. Writes the frames to `.mp4` via `diffusers.utils.export_to_video`.

## Verifying LVSA engaged

`offline_lvsa.py` prints a config header and a parseable benchmark line:

```
[offline_lvsa] family=hunyuan backend=flashinfer lvsa=True T_lat=33 frames=129 832x480 steps=50
[BENCH] gen_s=52.36 peak_mb=… steps=50 frames=129 s_per_step=…
```

Plus an engagement line from the attention **backend** — `[LVSA] Geometry
detected: …` — shared by all three families (`wan`/`hunyuan`/`cosmos`; cosmos
via the `cosmos3_backend.py` seam-swap). A persistent `[LVSA-FALLBACK]`
warning means LVSA did **not** engage (e.g. geometry mismatch) — see
[`../../docs/troubleshooting.md`](../../docs/troubleshooting.md).
(`--backend dense` runs with **no** LVSA on purpose — the baseline.)

## Multi-GPU

For tensor-parallel (Ulysses-style) generation across GPUs, set the `TP` env var on the server scripts:

```bash
TP=2 examples/serve_wan.sh /path/to/Wan2.1-T2V-14B-Diffusers
```

`offline_lvsa.py` is single-GPU (TP=1) by design — for multi-GPU use the server
path above. **Constraint**: `seq_len = T_lat × patches_per_frame` must be
divisible by `tensor_parallel_size`.

## Common gotchas

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: vllm` | `pip install vllm` separately — vllm-omni does not pull it as a hard dep |
| `[LVSA-FALLBACK] reason=geometry_detect` | Set `LVSA_PATCHES_PER_FRAME` or `LVSA_VIDEO_HEIGHT/WIDTH` for non-default resolutions |
| Server stuck in `queued` state | Worker process crashed — check stderr of the serve command |
| `[LVSA] Warning: LVSA_TOTAL_LATENT_FRAMES not set` | The LVSA install (wan/hunyuan hook or cosmos backend) needs the latent-frame count at install time. Set `LVSA_TOTAL_LATENT_FRAMES=(num_frames-1)//4 + 1` |
