# MeanVC

Lightweight streaming voice conversion — only 14M parameters, 1-step inference.

**Paper**: Mean Flows for One-step Voice Conversion (2025) | **License**: Apache 2.0

## Features

- Only 14M parameters (smallest model in mlx-vc)
- 1-step inference via mean flows (no iterative diffusion)
- Streaming chunk-wise inference with KV-cache
- RTF 0.136 on single CPU core

## Usage

```python
from mlx_vc.models.meanvc import MeanVC

vc = MeanVC(steps=1)
audio = vc.convert("source.wav", "reference.wav")
```

## Setup

Requires cloning the MeanVC reference repo:

```bash
git clone https://github.com/ASLP-lab/MeanVC.git meanvc-ref
cd meanvc-ref && python download_ckpt.py
```
