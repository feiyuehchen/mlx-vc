# Real-time Voice Conversion

> 中文翻譯尚未補完，請參考 [English version](https://feiyuehchen.github.io/mlx-vc/en/guides/realtime.md).
> Translation pending — see the English version for now.


Convert your voice in real-time through the microphone.

## Usage

```bash
# List audio devices
python -m mlx_vc.demo.realtime_vc --reference speaker.wav --list-devices

# Run with default devices
python -m mlx_vc.demo.realtime_vc --reference speaker.wav

# Specify devices
python -m mlx_vc.demo.realtime_vc --reference speaker.wav \
  --input-device 2 --output-device 1
```

## Tips

- **Use headphones** to avoid feedback loop (speaker output → microphone)
- First block is slow (model warm-up), subsequent blocks run at RTF ~0.7
- Reduce `--diffusion-steps 5` for lower latency at the cost of quality
- Increase `--block-time 1.0` for better quality per block

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--reference` | Required | Path to target speaker audio |
| `--block-time` | 0.5 | Block size in seconds |
| `--diffusion-steps` | 10 | Fewer = faster, more = better |
| `--cfg-rate` | 0.7 | Classifier-free guidance strength |
| `--extra-time` | 2.5 | Context window for content encoder |
