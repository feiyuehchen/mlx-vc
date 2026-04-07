# mlx-vc

Voice conversion on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

## Features

- Zero-shot voice conversion optimized for Apple Silicon (M series)
- Multiple model backends (CosyVoice3, more coming)
- Unified Python API and CLI
- Real-time voice conversion demo (coming soon)

## Installation

```bash
# Using uv
uv pip install -e ".[all]"

# Or pip
pip install -e ".[all]"
```

### Optional dependencies

```bash
pip install -e ".[cosyvoice]"  # CosyVoice3/Chatterbox backend
pip install -e ".[realtime]"   # Real-time VC demo
pip install -e ".[dev]"        # Development tools
```

## Quick Start

### Command Line

```bash
# Voice conversion with CosyVoice3 (zero-shot voice cloning)
mlx_vc.generate --model cosyvoice --text "Hello, world!" --ref_audio speaker.wav

# Play output immediately
mlx_vc.generate --model cosyvoice --text "Hello!" --ref_audio speaker.wav --play

# Save to specific file
mlx_vc.generate --model cosyvoice --text "Hello!" --ref_audio speaker.wav --output out.wav
```

### Python API

```python
from mlx_vc.models.cosyvoice import CosyVoiceVC

# Load model
vc = CosyVoiceVC()

# Convert: synthesize text with reference speaker's voice
audio = vc.convert(
    text="Hello, how are you?",
    ref_audio="speaker_reference.wav",
)

# Save output
from mlx_vc.audio_io import save_audio
save_audio("output.wav", audio, sample_rate=vc.sample_rate)
```

## Supported Models

| Model | Type | Zero-shot | Status |
|-------|------|-----------|--------|
| CosyVoice3/Chatterbox | TTS + Voice Cloning | Yes | Available |
| RVC | Retrieval-based VC | No (fine-tuned) | Planned |
| Seed-VC | Diffusion VC | Yes | Planned |
| MeanVC | Lightweight streaming VC | Yes | Planned |
| Vevo | Unified speech+singing VC | Yes | Planned |

## License

MIT
