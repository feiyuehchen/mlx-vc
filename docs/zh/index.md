# mlx-vc

Apple Silicon 上的 voice conversion 函式庫。10 個 zero-shot VC 與 voice-cloning 模型透過 subprocess 隔離統一 runner，能用 MPS 加速的就用 MPS。

## 為什麼這個 project

- **Apple Silicon 為主**：所有 backend 都在 MPS 或 MLX 跑過，沒能上 MPS 的會明確說明（如 MeanVC 卡 TorchScript graph fuser、kNN-VC 卡 upstream `np.float64`）
- **Subprocess 隔離**：每個 backend 在獨立 Python process 跑，避免多 model 連續載 OOM
- **客觀 benchmark**：UTMOS / SECS / WER 三個指標一鍵比所有模型，不是憑感覺
- **可擴充**：加新 backend 是一個 inference script + 一個 `BACKENDS` 條目

## 開始

- [安裝](getting-started/installation.md)
- [快速入門](getting-started/quickstart.md)
- [模型總覽](models/index.md) | [完整比較表](models.md)
- [評估指標](guides/evaluation.md)

## 目前已整合

| 類別 | Model | 何時選 |
|------|-------|--------|
| True VC | Seed-VC | 品質首選（UTMOS 3.97、SECS 0.847） |
| True VC | OpenVoice V2 | 速度首選（28ms/block，realtime demo 用） |
| True VC | kNN-VC | 不要訓練的 baseline |
| True VC | FreeVC / FreeVC-s | 經典 VITS one-shot VC |
| True VC | MeanVC | 中文 source（英文 WER 高） |
| True VC | SpeechT5-VC | 對照組 — 訓練在 read-speech，自然語音會崩 |
| True VC | RVC | 預先 fine-tuned 的 per-speaker model |
| TTS-clone | Chatterbox | 文字路徑 voice cloning |
| TTS-clone | Pocket-TTS | 輕量英文 voice cloning |

完整列表（含未整合的 GPT-SoVITS、Vevo 等）在[完整比較表](models.md)。

## License

MIT
