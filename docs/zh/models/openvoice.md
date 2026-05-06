# OpenVoice V2

> 中文翻譯尚未補完，請參考 [English version](https://feiyuehchen.github.io/mlx-vc/en/models/openvoice.md).
> Translation pending — see the English version for now.


Tone color conversion — transfers speaker timbre from reference to source.

**Paper**: [arXiv:2312.01479](https://arxiv.org/abs/2312.01479) | **License**: MIT

## Features

- Extremely fast: 16s audio in 0.7s
- Multilingual: EN, ZH, JA, KO, FR, ES
- Zero-shot tone color cloning

## Usage

```python
from mlx_vc.models.openvoice import OpenVoiceVC

vc = OpenVoiceVC(tau=0.3)
audio = vc.convert("source.wav", "reference.wav")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.3 | Style control (0=more target, 1=more source) |

## Note

OpenVoice transfers **timbre only** — accent and emotion come from the source audio. For full voice conversion (including accent), use Seed-VC.
