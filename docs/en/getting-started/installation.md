# Installation

## Basic Install

```bash
git clone https://github.com/feiyuehchen/mlx-vc.git
cd mlx-vc
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

## Model-specific Dependencies

Seed-VC and OpenVoice require PyTorch and the Seed-VC reference repo:

```bash
# PyTorch (for Seed-VC, OpenVoice, kNN-VC)
uv pip install torch torchaudio einops descript-audio-codec munch

# Seed-VC reference repo (required for Seed-VC and OpenVoice backends)
cd .. && git clone --depth 1 https://github.com/Plachtaa/seed-vc.git seed-vc-ref
```

## Verify Installation

```bash
pytest -s mlx_vc/tests/
```

All 23 tests should pass.
