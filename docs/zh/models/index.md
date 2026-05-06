# 模型總覽

mlx-vc 內建 10 個 backends，分兩類：

## True audio→audio VC（保留 source prosody）

| Model | 何時選 | 優點 | 缺點 |
|-------|--------|------|------|
| **Seed-VC** | 高品質 VC + SVC | 全方位最佳（UTMOS 3.97, SECS 0.847）；zero-shot；支援唱歌 | RTF ~1.9；50 steps 跑 10s 需 ~140s |
| **OpenVoice V2** | 快速音色轉換 | 1 秒內完成；多語言；保留 source prosody | 只轉音色，不轉口音 / 情緒 |
| **kNN-VC** | 不訓練的 baseline | Non-parametric；MIT | WavLM-Large 1.18GB；只 CPU |
| **FreeVC / FreeVC-s** | One-shot WavLM 路線 | 兩個架構變體（含/不含 speaker encoder） | SECS 不及 Seed-VC |
| **MeanVC** | 中文 source 輕量 | DiT 僅 14M params；RTF 0.14 | 中文訓練 — 英文 WER 高 |
| **SpeechT5-VC** | 對照 / 參考組 | Microsoft transformer seq2seq | 訓練 read-speech；自然 lecture 會崩 |
| **RVC** | 預先 fine-tune 的 per-speaker | 配對好的 model 品質高 | 非 zero-shot；speaker 寫死 `.npz`；需 Python 3.10 venv |

## TTS-clone（文字路徑 — **不是**真 VC）

| Model | 何時選 | 優點 | 缺點 |
|-------|--------|------|------|
| **Chatterbox**（cosyvoice 槽位） | 透過文字做 voice cloning | 輸出乾淨；內容 by construction perfect | Source 的 prosody / 情緒 / 時序都重新生成不保留 |
| **Pocket-TTS** | 輕量英文 voice cloning | 只 ~235 MB；快 | 跟 Chatterbox 一樣的 text-path 限制 |

`BACKENDS` 條目都共用同一個 call signature：`run_backend(name, source=..., reference=..., output=...)`。詳見[快速入門](../getting-started/quickstart.md)和[評估指標](../guides/evaluation.md)。
