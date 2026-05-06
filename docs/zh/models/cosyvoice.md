# CosyVoice3 / Chatterbox

> 中文翻譯尚未補完，請參考 [English version](https://feiyuehchen.github.io/mlx-vc/en/models/cosyvoice.md).
> Translation pending — see the English version for now.


TTS with voice cloning — synthesizes text with a reference speaker's voice.

!!! note
    This is TTS + voice cloning, not true voice conversion. You provide **text** input, not source audio.

## Usage

```python
from mlx_vc.models.cosyvoice import CosyVoiceVC

vc = CosyVoiceVC()
audio = vc.convert(text="Hello world", ref_audio="speaker.wav")
```
