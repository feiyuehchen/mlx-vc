# 安裝

## 基本安裝

```bash
git clone https://github.com/feiyuehchen/mlx-vc.git
cd mlx-vc
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

## 個別模型的依賴

每個 backend 的 docstring 列出該模型需要的 reference repo 跟 weight 下載步驟。常見的有：

```bash
# Seed-VC / OpenVoice — 共用 seed-vc-ref repo
cd .. && git clone --depth 1 https://github.com/Plachtaa/seed-vc.git seed-vc-ref

# FreeVC
cd .. && git clone --depth 1 https://github.com/OlaWod/FreeVC.git freevc-ref

# MeanVC（含手動下載 speaker verification ckpt）
cd .. && git clone --depth 1 https://github.com/ASLP-lab/MeanVC.git meanvc-ref
gdown 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP \
      -O meanvc-ref/src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth

# RVC（自帶 venv）
cd .. && git clone https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX.git rvc-mlx-ref
cd rvc-mlx-ref && uv venv --python 3.10 .venv
```

## 驗證安裝

```bash
pytest -s mlx_vc/tests/ -v --timeout=60
```

預期：46 個測試全過，1 個 skip（`test_filename_resolved_under_ref_dir` 在沒設 `MLX_VC_REF_DIR` 時 skip）。
