# mlx-vc

Voice conversion on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

## Features

- **6 VC models** with a unified Python API and CLI
- Zero-shot voice conversion (no fine-tuning needed for most models)
- Singing voice conversion (SVC) with pitch preservation
- Real-time microphone-to-speaker demo
- FastAPI server for HTTP/WebSocket access
- Subprocess isolation — models with conflicting deps coexist cleanly

## Supported Models

| Model | Type | Zero-shot | Speed |
|-------|------|-----------|-------|
| [Seed-VC](models/seed-vc.md) | Diffusion VC (speech + singing) | Yes | RTF ~2 |
| [OpenVoice V2](models/openvoice.md) | Tone color conversion | Yes | Very fast |
| [kNN-VC](models/knn-vc.md) | Non-parametric (WavLM + kNN) | Yes | Fast |
| [CosyVoice3](models/cosyvoice.md) | TTS + voice cloning | Yes | Fast |
| [RVC](models/rvc.md) | Retrieval-based VC | No | Fast |
| [MeanVC](models/meanvc.md) | Lightweight streaming (14M) | Yes | Very fast |

## Quick Example

```python
from mlx_vc.models.seed_vc import SeedVC

vc = SeedVC()
audio = vc.convert(source_audio="my_voice.wav", ref_audio="target_speaker.wav")
```
