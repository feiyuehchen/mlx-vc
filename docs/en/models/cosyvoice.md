# CosyVoice3 / Chatterbox

TTS with voice cloning — synthesizes text with a reference speaker's voice.

!!! note
    This is TTS + voice cloning, not true voice conversion. You provide **text** input, not source audio.

## Usage

```python
from mlx_vc.models.cosyvoice import CosyVoiceVC

vc = CosyVoiceVC()
audio = vc.convert(text="Hello world", ref_audio="speaker.wav")
```
