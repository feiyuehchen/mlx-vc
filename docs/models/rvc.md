# RVC (Retrieval-based Voice Conversion)

Community-standard VC using HuBERT + VITS + FAISS retrieval. Requires per-speaker fine-tuning.

## Usage

```python
from mlx_vc.models.rvc import RVCVC

vc = RVCVC(model_path="my_speaker_model.pth")
audio = vc.convert(source_audio="input.wav")
```

!!! warning
    RVC is **not zero-shot**. You need a fine-tuned model per speaker (~10 min clean audio). Train with [Applio](https://github.com/IAHispano/Applio) or RVC WebUI.
