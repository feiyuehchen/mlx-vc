# Seed-VC

Zero-shot voice conversion using Whisper + Diffusion Transformer + BigVGAN.

**Paper**: [arXiv:2411.09943](https://arxiv.org/abs/2411.09943)

## Features

- Zero-shot: no fine-tuning needed, just provide reference audio
- Speech VC (22kHz) and Singing VC (44kHz with F0 conditioning)
- Chunked processing for arbitrarily long audio
- SECS 0.868 (speaker similarity benchmark)

## Usage

```python
from mlx_vc.models.seed_vc import SeedVC

# Speech VC
vc = SeedVC(diffusion_steps=25)
audio = vc.convert("source.wav", "reference.wav")

# Singing VC (preserves melody/pitch)
vc_svc = SeedVC(f0_condition=True, diffusion_steps=25)
audio = vc_svc.convert("singing.wav", "reference.wav")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diffusion_steps` | 30 | More steps = better quality, slower |
| `inference_cfg_rate` | 0.7 | Classifier-free guidance (0=no guidance, 1=strong) |
| `f0_condition` | False | Enable for singing VC (44kHz, preserves pitch) |
| `length_adjust` | 1.0 | Speed adjustment factor |

## Architecture

```
Source audio → Whisper encoder → content features
Ref audio → CAMPPlus → speaker style (192-dim)
Content + Style + Ref mel → CFM (DiT) → mel spectrogram → BigVGAN → audio
```

## Performance

| Mode | Audio | Time | RTF |
|------|-------|------|-----|
| Speech (25 steps) | 10s | ~19s | 1.9 |
| SVC (25 steps) | 16s | ~18min | 68 |
| Speech (10 steps) | 15s | ~18s | 1.2 |
