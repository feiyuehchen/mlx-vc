# Models Overview

mlx-vc supports 6 voice conversion models with different trade-offs:

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **Seed-VC** | High-quality VC + SVC | Best quality, zero-shot, singing support | Slower (RTF ~2) |
| **OpenVoice V2** | Fast tone transfer | Extremely fast (0.7s/16s), multilingual | Only transfers timbre, not accent |
| **kNN-VC** | Simple, no training | Non-parametric, MIT license | WavLM is large (315M) |
| **CosyVoice3** | Text-to-speech cloning | Good TTS quality | Text input only, not true VC |
| **RVC** | Best per-speaker quality | Community proven (35k stars) | Requires fine-tuning |
| **MeanVC** | Real-time streaming | Only 14M params, 1-step | Needs setup |

All models share the same API: `model.convert(source_audio, ref_audio) -> np.ndarray`.
