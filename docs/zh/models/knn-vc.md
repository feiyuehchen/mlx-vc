# kNN-VC

> 中文翻譯尚未補完，請參考 [English version](https://feiyuehchen.github.io/mlx-vc/en/models/knn-vc.md).
> Translation pending — see the English version for now.


Non-parametric voice conversion using WavLM + k-nearest neighbors + HiFi-GAN.

**Paper**: [arXiv:2305.18975](https://arxiv.org/abs/2305.18975) | **License**: MIT

## Features

- No neural network training for the conversion step (kNN is non-parametric)
- Uses WavLM-Large (315M) for feature extraction
- Fast on CPU: 10s audio in 3.4s

## Usage

```python
from mlx_vc.models.knn_vc import KnnVC

vc = KnnVC(topk=4)
audio = vc.convert("source.wav", "reference.wav")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topk` | 4 | Number of nearest neighbors for matching |
| `prematched` | True | Use prematched HiFi-GAN (better quality) |
