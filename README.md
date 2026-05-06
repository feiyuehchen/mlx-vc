# mlx-vc

[![Tests](https://github.com/feiyuehchen/mlx-vc/actions/workflows/tests.yml/badge.svg)](https://github.com/feiyuehchen/mlx-vc/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-orange.svg)](https://developer.apple.com/metal/pytorch/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Voice conversion library for Apple Silicon. A unified subprocess-isolated runner for ten zero-shot VC and voice-cloning models, with MPS-accelerated inference where the model permits.

## What's included

Two categories. Only the first is true voice conversion (audio → audio, source prosody preserved).

### True audio→audio VC

| Model | Backend | Device | Notes |
|-------|---------|--------|-------|
| **Seed-VC** | subprocess | MPS + fp16 | Whisper-small content + DiT (CFM) + BigVGAN. Best quality. |
| **OpenVoice V2** | subprocess | MPS | VITS tone-color converter. Fast, preserves prosody. |
| **kNN-VC** | subprocess | CPU | WavLM frame retrieval + HiFi-GAN. Non-parametric. |
| **FreeVC / FreeVC-s** | subprocess | MPS | WavLM + VITS decoder, with / without speaker encoder. |
| **MeanVC** | subprocess | CPU | DiT + Mean Flow, 14M params, Chinese-trained. |
| **SpeechT5** | subprocess | MPS | Microsoft transformer seq2seq, CMU-ARCTIC-trained. |
| **RVC** | subprocess (own venv) | MLX | Acelogic MLX port. Speaker baked into per-model `.npz`. |

### TTS-clone (text path — Whisper transcribes source then synthesizes)

| Model | Backend | Device |
|-------|---------|--------|
| **Chatterbox** (`cosyvoice` slot) | in-process via mlx-audio | MPS |
| **Pocket-TTS** | subprocess (~235 MB) | MPS |

The TTS-clone path is **not true VC** — output prosody is regenerated from text, not preserved from source. Included for comparison.

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

Each backend has per-model setup (clone a reference repo, download checkpoints). See the per-backend setup notes inside each `mlx_vc/backends/<name>_infer.py` docstring.

## Quick start

### Subprocess runner — all 10 models

The canonical entry point. Each call spawns a fresh Python interpreter, loads the model, runs inference, exits — weights are released between calls so memory doesn't accumulate.

```python
from mlx_vc.backend import run_backend, BACKENDS
print(list(BACKENDS.keys()))
# ['seed-vc', 'openvoice', 'knn-vc', 'meanvc', 'rvc',
#  'freevc', 'freevc-s', 'pocket-tts', 'speecht5']

audio = run_backend(
    "seed-vc",
    source="src.wav",
    reference="ref.wav",
    output="out.wav",
)
```

### In-process Python API

For models that have a Python wrapper class (`AVAILABLE_MODELS` in `generate.py`):

```python
from mlx_vc.models.seed_vc import SeedVC
vc = SeedVC()
audio = vc.convert(source="src.wav", reference="ref.wav")

from mlx_vc.audio_io import save_audio
save_audio("out.wav", audio, sample_rate=22050)
```

In-process loading is convenient but does NOT release weights — use the subprocess runner when chaining multiple models.

### CLI

```bash
mlx_vc.generate --model seed-vc --source src.wav --ref_audio ref.wav --output out.wav
```

### FastAPI server

```bash
python -m mlx_vc.server                 # default :8000
```

Endpoints:
- `POST /v1/audio/convert` — single model
- `POST /v1/audio/convert/batch` — multiple models (sequential, semaphore-serialised)
- `GET  /v1/jobs/{job_id}` — poll batch status
- `POST /v1/audio/upload-reference` — upload a reference WAV for later use
- `WS   /ws/realtime` — OpenVoice singleton, ~300 ms latency

Set `MLX_VC_REF_DIR=/path/to/refs` to let the server resolve bare filenames in WS init messages.  Without it, clients must use absolute paths or upload via `/v1/audio/upload-reference`.

## Quality benchmark

`scripts/evaluate_quality.py` scores VC outputs on three objective metrics:

- **UTMOS** — no-reference naturalness MOS via `torchaudio.pipelines.SQUIM_SUBJECTIVE` (1–5)
- **SECS** — ECAPA speaker embedding cosine similarity vs reference (0–1)
- **WER** — Whisper-small re-transcription word error rate vs source

```bash
python scripts/evaluate_quality.py \
    --source src.wav \
    --reference ref.wav \
    --outputs out_seedvc.wav out_openvoice.wav out_knnvc.wav ... \
    --json metrics.json
```

For TTS-clone outputs, WER ≈ 0 is trivially achieved (text roundtrip).  Compare TTS-clone models among themselves — not against true VC.

## Reference audio preparation

`scripts/prepare_reference.py` runs Demucs vocal separation and extracts two clean reference clips (60 s + 3 min):

```bash
python scripts/prepare_reference.py \
    --input long_recording.wav \
    --output_dir refs/ \
    --name speaker
# → refs/speaker_ref_clean.wav   (60 s)
# → refs/speaker_ref_3min.wav    (3 min, for kNN-VC)
```

Clean reference is the single biggest factor in VC output quality.

## Real-time demo

Mic → speaker via OpenVoice, ~300 ms latency:

```bash
python -m mlx_vc.demo.realtime_vc --reference speaker.wav
```

Or via WebSocket from a browser to `ws://127.0.0.1:8000/ws/realtime`.

## Architecture

```
mlx_vc/
├── backend.py        # BACKENDS registry + subprocess runner
├── jobs.py           # In-memory JobManager for /v1/audio/convert/batch
├── server.py         # FastAPI: convert / batch / WS realtime
├── realtime.py       # OpenVoiceSession singleton (warm OpenVoice for WS)
├── generate.py       # CLI + AVAILABLE_MODELS (in-process registry)
├── audio_io.py       # WAV load/save
├── utils.py          # HF model download, config loading
├── models/           # In-process Python wrappers
└── backends/         # Subprocess inference scripts
    ├── seed_vc_infer.py
    ├── openvoice_infer.py
    ├── knn_vc_infer.py
    ├── meanvc_infer.py
    ├── rvc_infer.py             (subprocess into rvc-mlx-ref/.venv)
    ├── freevc_infer.py
    ├── speecht5_infer.py
    └── tts_clone_infer.py       (generic mlx-audio TTS-clone runner)
```

**Two registries**:

- `AVAILABLE_MODELS` (`generate.py`) — Python class wrappers for direct in-process use.
- `BACKENDS` (`backend.py`) — subprocess scripts. Canonical list, includes the newer additions (freevc/-s, speecht5, pocket-tts).

Use `BACKENDS` whenever isolating dependencies or controlling memory matters. Use `AVAILABLE_MODELS` when calling from the same Python process is fine.

## Reference repos to clone alongside this one

```
../seed-vc-ref/      git@github.com:Plachtaa/seed-vc.git
../meanvc-ref/       git@github.com:ASLP-lab/MeanVC.git
                     + manual download of speaker-verification ckpt from upstream Drive link
../freevc-ref/       git@github.com:OlaWod/FreeVC.git
                     + HF mirrors for weights (see backends/freevc_infer.py docstring)
../rvc-mlx-ref/      git@github.com:Acelogic/Retrieval-based-Voice-Conversion-MLX.git
                     + own .venv (Python 3.10, numpy<2)
```

Each backend's docstring lists the exact setup steps and HF / external download URLs.

## Environment variables

| Var | Purpose |
|-----|---------|
| `MLX_VC_REF_DIR` | Search path for bare reference filenames in the server. No default. |
| `RVC_MODEL_PATH` | Path to the `.npz` RVC speaker model used by `rvc_infer.py`. |
| `MEANVC_DEVICE` | `cpu` (default) / `mps` (will hit TorchScript graph-fuser bug). |
| `KNN_VC_DEVICE` | `cpu` (default). MPS blocked on upstream fp64 weights + cross-device child modules. |
| `SEED_VC_PATH`, `MEANVC_PATH`, `FREEVC_PATH`, `RVC_MLX_PATH` | Override sibling repo paths. |

## Testing

```bash
pytest -s mlx_vc/tests/ -v
```

## License

MIT
