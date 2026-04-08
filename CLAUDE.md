# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-vc is a voice conversion library for Apple Silicon. It provides a unified API for multiple VC model backends. Models run via subprocess isolation to avoid dependency conflicts.

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

| Model | Type | Backend | Notes |
|-------|------|---------|-------|
| CosyVoice3/Chatterbox | TTS + voice cloning | In-process (mlx-audio) | Text input only |
| Seed-VC | Zero-shot VC | Subprocess (PyTorch MPS) | True audio-to-audio VC |
| Seed-VC SVC | Singing VC | Subprocess (PyTorch MPS) | F0-conditioned, 44kHz |
| OpenVoice V2 | Tone color conversion | Subprocess (PyTorch) | Very fast, multilingual |
| RVC | Per-speaker VC | Subprocess | Needs fine-tuned model |

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
