# Quick Start

## Command Line

```bash
# Seed-VC: true voice conversion (audio -> audio)
mlx_vc.generate --model seed-vc --text unused --ref_audio target_speaker.wav

# OpenVoice: fast tone color conversion
python -m mlx_vc.backends.openvoice_infer --args '{"source": "my.wav", "reference": "target.wav", "output": "out.wav"}'
```

## Python API

Every model follows the same pattern: `convert(source_audio, ref_audio) -> numpy array`.

```python
# Seed-VC (zero-shot, best quality)
from mlx_vc.models.seed_vc import SeedVC
vc = SeedVC(diffusion_steps=25)
audio = vc.convert("my_voice.wav", "professor.wav")

# OpenVoice V2 (fastest)
from mlx_vc.models.openvoice import OpenVoiceVC
vc = OpenVoiceVC()
audio = vc.convert("my_voice.wav", "professor.wav")

# kNN-VC (non-parametric, no neural vocoder training)
from mlx_vc.models.knn_vc import KnnVC
vc = KnnVC(topk=4)
audio = vc.convert("my_voice.wav", "professor.wav")
```

## Save Output

```python
from mlx_vc.audio_io import save_audio
save_audio("output.wav", audio, sample_rate=vc.sample_rate)
```

## Real-time Demo

```bash
python -m mlx_vc.demo.realtime_vc --reference professor.wav
```

Speak into the microphone — converted audio plays through your headphones.
