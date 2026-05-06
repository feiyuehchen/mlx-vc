# RVC (Retrieval-based Voice Conversion)

> 中文翻譯尚未補完，請參考 [English version](https://feiyuehchen.github.io/mlx-vc/en/models/rvc.md).
> Translation pending — see the English version for now.


Community-standard VC using HuBERT + VITS + FAISS retrieval. Requires per-speaker fine-tuning.

## Usage

```python
from mlx_vc.models.rvc import RVCVC

vc = RVCVC(model_path="my_speaker_model.pth")
audio = vc.convert(source_audio="input.wav")
```

!!! warning
    RVC is **not zero-shot**. You need a fine-tuned model per speaker (~10 min clean audio). Train with [Applio](https://github.com/IAHispano/Applio) or RVC WebUI.
