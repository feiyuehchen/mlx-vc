# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-vc is a voice conversion library for Apple Silicon using MLX. It provides a unified API for multiple VC model backends.

## Build & Install

```bash
# Create venv and install with uv
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"

# Modular installs
uv pip install -e ".[cosyvoice]"   # CosyVoice3/Chatterbox only
uv pip install -e ".[realtime]"    # Real-time demo deps
```

## Testing

```bash
pytest -s mlx_vc/tests/
pytest -s mlx_vc/tests/test_imports.py   # Single test file
```

## Linting & Formatting

```bash
pre-commit run --all-files   # Black (line-length=88) + isort (black profile)
```

## CLI

```bash
mlx_vc.generate --model cosyvoice --text "Hello" --ref_audio speaker.wav
```

## Architecture

- `mlx_vc/models/` — Each VC model in its own subpackage (e.g., `cosyvoice/`)
- `mlx_vc/generate.py` — Unified CLI with `AVAILABLE_MODELS` registry
- `mlx_vc/audio_io.py` — Audio load/save utilities
- `mlx_vc/utils.py` — HuggingFace model download, config loading
- `mlx_vc/demo/` — Real-time VC demos
- `scripts/` — Weight conversion tools

## Adding a New Model

1. Create `mlx_vc/models/<model_name>/` with `__init__.py` and implementation
2. Expose a class with `convert(text, ref_audio, ...) -> np.ndarray` method
3. Register in `AVAILABLE_MODELS` dict in `mlx_vc/generate.py`
4. Add tests in `mlx_vc/tests/`

## Key Dependencies

- `mlx>=0.25.2` — Core ML framework
- `mlx-audio[tts]` — CosyVoice3 backend (optional)
- `sounddevice` — Audio playback
- `librosa` — Audio loading/resampling
