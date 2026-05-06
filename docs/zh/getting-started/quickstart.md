# 快速入門

## Subprocess runner（10 個 model 的標準入口）

```python
from mlx_vc.backend import BACKENDS, run_backend

print(list(BACKENDS.keys()))
# ['seed-vc', 'openvoice', 'knn-vc', 'meanvc', 'rvc',
#  'freevc', 'freevc-s', 'pocket-tts', 'speecht5']

audio = run_backend(
    "seed-vc",
    source="src.wav",
    reference="ref.wav",
    output="out.wav",
)
```

每次呼叫會 spawn 新的 Python interpreter，跑完 model weights 自動釋放，不會累積記憶體。

## In-process Python API

```python
# Seed-VC
from mlx_vc.models.seed_vc import SeedVC
vc = SeedVC(diffusion_steps=50)
audio = vc.convert("my_voice.wav", "target_speaker.wav")

# OpenVoice V2
from mlx_vc.models.openvoice import OpenVoiceVC
vc = OpenVoiceVC()
audio = vc.convert("my_voice.wav", "target_speaker.wav")

# kNN-VC
from mlx_vc.models.knn_vc import KnnVC
vc = KnnVC(topk=4)
audio = vc.convert("my_voice.wav", "target_speaker.wav")
```

## 存檔

```python
from mlx_vc.audio_io import save_audio
save_audio("output.wav", audio, sample_rate=vc.sample_rate)
```

## CLI

```bash
mlx_vc.generate --model seed-vc --source src.wav --ref_audio ref.wav --output out.wav
```

## Realtime demo

```bash
python -m mlx_vc.demo.realtime_vc --reference target_speaker.wav
```

對麥克風講話，耳機會聽到轉換後的聲音。

## 評估品質

```bash
python scripts/evaluate_quality.py \
    --source src.wav \
    --reference ref.wav \
    --outputs out_seedvc.wav out_openvoice.wav ... \
    --json metrics.json
```

詳見[評估指標](../guides/evaluation.md)。
