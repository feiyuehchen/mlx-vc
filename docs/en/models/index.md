# Models Overview

mlx-vc currently ships 10 backends in two categories.

## True audio→audio VC (preserves source prosody)

| Model | Best for | Pros | Cons |
|-------|----------|------|------|
| **Seed-VC** | High-quality VC + SVC | Best quality (UTMOS 3.97, SECS 0.847), zero-shot, singing support | RTF ~1.9; ~140 s for 10 s at 50 diffusion steps |
| **OpenVoice V2** | Fast tone transfer | Sub-second; multilingual; preserves source prosody | Only timbre, not accent / emotion |
| **kNN-VC** | No-training baseline | Non-parametric, MIT-licensed | WavLM-Large 1.18 GB; CPU only |
| **FreeVC / FreeVC-s** | One-shot WavLM-style VC | Two architecture variants (with / without speaker encoder) | SECS lower than Seed-VC |
| **MeanVC** | Lightweight Chinese VC | DiT only 14M params; RTF 0.14 | Trained on Chinese — high WER on English |
| **SpeechT5-VC** | Reference / contrastive | Microsoft transformer seq2seq | Trained on read-speech; collapses on natural lecture audio |
| **RVC** | Per-speaker fine-tuned | High quality with proper model | Not zero-shot; speaker baked into `.npz`; needs Python 3.10 venv |

## TTS-clone (text path — NOT true VC)

| Model | Best for | Pros | Cons |
|-------|----------|------|------|
| **Chatterbox** (`cosyvoice` slot) | Voice cloning via text | Clean output, content perfect by construction | Source prosody / emotion / timing regenerated, not preserved |
| **Pocket-TTS** | Lightweight English voice cloning | Only ~235 MB; fast | Same text-path caveat as Chatterbox |

All `BACKENDS` entries share the same call signature: `run_backend(name, source=..., reference=..., output=...)`. See [Quick Start](../getting-started/quickstart.md) and [Evaluation Metrics](../guides/evaluation.md).
