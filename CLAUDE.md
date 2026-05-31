# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-vc is a voice conversion library for Apple Silicon. It provides a unified API for multiple VC model backends. Models run via subprocess isolation to avoid dependency conflicts.

## Versioning & Workflow

This project follows **two complementary protocols** (see `~/research/semver-development-workflow.md` and `~/research/ai-research-code-workflow.md`):

- **Library API** → SemVer workflow. Public API = CLI flags, server endpoints, Python `convert()` signatures.
- **Model quality evaluation** → Research workflow BENCHMARK.md. Metric definitions in Part A; per-model results in Part B.

### Version Rules (SemVer, currently 0.x)

| Change | Bump |
|--------|------|
| New model backend, new API endpoint, new CLI flag | MINOR |
| Bug fix, refactor, docs, speed improvement (API unchanged) | PATCH |
| Remove/rename public API, change endpoint schema, change BENCHMARK.md Part A metric definition | MAJOR (or 0.x MINOR with `!`) |

### Commit Convention (Conventional Commits)

```
feat:      new capability (model, endpoint, flag)  → MINOR
fix:       bug fix                                  → PATCH
perf:      speed improvement (no API change)        → PATCH
bench:     BENCHMARK.md Part B result update        → no release
bench!:    BENCHMARK.md Part A metric change        → MAJOR
docs/test/chore/ci/style:                           → no release
```

### Tags

- `v*` — release versions (`v0.1.0`, `v0.2.0`)
- `paper/*` — publication milestones
- `archived/*` — abandoned experiments

### BENCHMARK.md Role

`BENCHMARK.md` tracks objective quality (UTMOS, SECS, WER) across models and versions.
- Part A metric definitions are fixed unless a `bench!` commit changes them.
- Part B accumulates per-model results; new rows are added, never overwritten.
- All numbers come from `scripts/evaluate_quality.py` with recorded git commit + audio pair.

## Build & Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

Seed-VC additionally requires: `torch torchaudio einops descript-audio-codec munch`
and the reference repo cloned to `../seed-vc-ref/`.

## Testing

```bash
pytest -s mlx_vc/tests/ -v            # All tests (27 tests)
pytest -s mlx_vc/tests/test_server.py  # Server tests only
```

## Linting & Formatting

```bash
pre-commit run --all-files   # Black (line-length=88) + isort (black profile)
```

## Server

```bash
python -m mlx_vc.server                          # Start API server
python -m mlx_vc.server --host 0.0.0.0 --port 8000  # Custom host/port
# API docs at http://localhost:8000/docs
```

## Documentation

```bash
uv pip install mkdocs-material
mkdocs serve   # Local preview at http://localhost:8000
mkdocs build   # Build static site to site/
```

## Architecture

```
mlx_vc/
├── models/           # Model wrappers (unified API)
│   ├── cosyvoice/    # TTS + voice cloning via mlx-audio
│   └── seed_vc/      # Zero-shot VC (Whisper + DiT + BigVGAN)
├── backends/         # Inference scripts (run via subprocess)
│   └── seed_vc_infer.py
├── backend.py        # Subprocess runner for model backends
├── generate.py       # Unified CLI + AVAILABLE_MODELS registry
├── audio_io.py       # Audio load/save
└── utils.py          # HuggingFace model download, config loading
```

**Key design**: Each model has a wrapper in `models/` (Python API) and optionally
an inference script in `backends/` (subprocess isolation). The subprocess approach
lets models with conflicting deps coexist in one package.

## Adding a New Model

1. Create `mlx_vc/models/<name>/` with a class exposing `convert(source_audio, ref_audio) -> np.ndarray`
2. If the model has heavy/conflicting deps, create `mlx_vc/backends/<name>_infer.py`
3. Register in `AVAILABLE_MODELS` in `generate.py` and in `BACKENDS` in `backend.py`
4. Add tests in `mlx_vc/tests/`

## Supported Models

True audio→audio VC:

| Model       | Backend             | Device         | Notes                                       |
|-------------|---------------------|----------------|---------------------------------------------|
| Seed-VC     | Subprocess          | MPS + fp16     | Diffusion DiT + BigVGAN. Best quality.      |
| OpenVoice V2| Subprocess          | MPS            | Tone-color converter, fast.                 |
| kNN-VC      | Subprocess          | CPU            | Forced CPU: upstream uses np.float64 (MPS rejects), HiFi-GAN device sync issues. |
| FreeVC / -s | Subprocess          | MPS            | VITS + WavLM. Speaker encoder runs on MPS now (torch 2.11). |
| MeanVC      | Subprocess          | CPU            | TorchScript Vocos / ASR hit MPS graph fuser bug. Set `MEANVC_DEVICE=mps` to override. |
| SpeechT5-VC | Subprocess          | MPS            | Microsoft transformer seq2seq. CMU-ARCTIC-trained — collapses on natural speech. |
| RVC         | Subprocess (own venv) | MLX (Acelogic) | Speaker baked into .npz. rvc-mlx-ref/.venv pinned to py3.10 + numpy<2. |

TTS-clone (text path, NOT true VC):

| Model       | Backend             | Device         | Notes                                       |
|-------------|---------------------|----------------|---------------------------------------------|
| Chatterbox  | In-process (mlx-audio) | MPS         | Whisper-transcribe source → resynth.        |
| Pocket-TTS  | Subprocess          | MPS            | 235MB Kyutai voice-cloning TTS via mlx-audio. |

## Torch / MPS audit (torch 2.11)

We periodically run `mlx_vc/tests/test_mps_ops.py` to confirm what works
on MPS in the current torch.  As of torch 2.11 these ALL pass:
complex64 STFT/ISTFT, Conv1dTranspose (HiFi-GAN), fp16 autocast, linalg.svd
(via CPU fallback), cdist+topk (kNN core), torchaudio resample,
transformers WavLM forward, scaled_dot_product_attention, freshly-traced
torch.jit modules.

Still failing on MPS:
- **Pre-saved TorchScript .pt graph fuser** — Vocos in MeanVC, ASR in MeanVC.
  "NotImplementedError: Unknown device for graph fuser". Force CPU.
- **float64 inputs** — MPS rejects float64 entirely. Bites us when upstream
  code does `torch.tensor(np.float64_array, device='mps')` (kNN-VC SPEAKER_INFORMATION_WEIGHTS).
- **Cross-device modules** — bshall/knn-vc loads HiFi-GAN on CPU regardless
  of device arg, then conv1d hits "MPSFloat vs torch.Float" mismatch.

## Real-time Demo

```bash
# List audio devices
python -m mlx_vc.demo.realtime_vc --reference speaker.wav --list-devices

# Run real-time VC (mic -> speaker)
python -m mlx_vc.demo.realtime_vc --reference speaker.wav

# With specific devices and settings
python -m mlx_vc.demo.realtime_vc --reference speaker.wav \
  --input-device 1 --output-device 2 --diffusion-steps 10
```

Uses Seed-VC XLSR-tiny model (25M params, ~300ms latency) for real-time inference.
