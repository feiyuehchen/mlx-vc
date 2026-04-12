# Voice Conversion 模型全覽

本文件整理了 mlx-vc 開發過程中調研和實測的所有 VC 模型，包含已整合的、測試未通過的、以及評估後放棄的模型。

---

## 已整合模型

### Seed-VC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot voice conversion（speech + singing） |
| **架構** | Whisper encoder + DiT (Diffusion Transformer) + CFM (Conditional Flow Matching) + BigVGAN |
| **參數量** | Speech: ~98M (512-dim, 8-head, 13-layer)；SVC: ~200M (768-dim, 12-head, 17-layer) |
| **取樣率** | Speech: 22050Hz；SVC: 44100Hz |
| **推論速度** | Speech 25 steps: RTF ~1.9（10s 音訊需 19s）；10 steps: RTF ~1.2 |
| **品質** | SECS 0.868（speaker similarity），優於 OpenVoice 0.755 和 CosyVoice 0.844 |
| **Zero-shot** | 是，1-30 秒 reference audio 即可 |
| **F0 conditioning** | SVC 模式支援（保留原始音高/旋律） |
| **授權** | MIT |
| **狀態** | ✅ 已整合（PyTorch MPS，subprocess backend） |
| **限制** | SVC 模式非常慢（16s 音訊需 ~18 分鐘）；需要 seed-vc-ref 外部 repo |

### OpenVoice V2

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot tone color conversion |
| **架構** | VITS-based SynthesizerTrn + Tone Color Converter |
| **參數量** | ~30M（相對輕量） |
| **取樣率** | 22050Hz |
| **推論速度** | **極快**：16s 音訊僅需 0.7s；0.3s block 僅需 28ms |
| **品質** | 音色轉換良好，但只轉換 timbre（不轉換口音、情緒、語調） |
| **Zero-shot** | 是 |
| **多語言** | EN, ZH, JA, KO, FR, ES |
| **授權** | MIT |
| **狀態** | ✅ 已整合（PyTorch MPS，subprocess backend）；**real-time demo 使用此模型** |
| **限制** | 只轉換音色，不轉換說話方式；speaker similarity 不如 Seed-VC |

### kNN-VC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot non-parametric voice conversion |
| **架構** | WavLM-Large (frozen, layer 6) + k-nearest neighbors + HiFi-GAN vocoder |
| **參數量** | WavLM ~316M + HiFi-GAN ~16.5M ≈ 333M（但 kNN 本身無參數） |
| **取樣率** | 16000Hz |
| **推論速度** | 10s 音訊約 3.4s（CPU） |
| **品質** | 中等，適合原型開發 |
| **Zero-shot** | 是（kNN 本身 non-parametric，不需訓練） |
| **授權** | MIT |
| **狀態** | ✅ 已整合（PyTorch CPU，subprocess backend） |
| **限制** | WavLM-Large 模型 1.18GB 很大；MPS 不穩定需 fallback CPU；16kHz 取樣率較低 |

### CosyVoice3 / Chatterbox

| 項目 | 內容 |
|------|------|
| **類型** | TTS + voice cloning（非真正的 voice conversion） |
| **架構** | S3Tokenizer + T3 Language Model + S3Gen Vocoder |
| **參數量** | ~500M（Chatterbox fp16） |
| **取樣率** | 24000Hz |
| **推論速度** | 快（3s 文字約 7s 生成） |
| **品質** | TTS 品質好，voice cloning 效果不錯 |
| **Zero-shot** | 是 |
| **授權** | Apache 2.0 |
| **狀態** | ✅ 已整合（透過 mlx-audio，in-process） |
| **限制** | 輸入是**文字**不是音訊——不是真正的 VC，是 TTS with voice cloning |

### RVC (Retrieval-based Voice Conversion)

| 項目 | 內容 |
|------|------|
| **類型** | Per-speaker fine-tuned voice conversion |
| **架構** | HuBERT/ContentVec + VITS generator + FAISS retrieval + RMVPE pitch |
| **參數量** | HuBERT ~95M + VITS ~55M ≈ 150M |
| **取樣率** | 48000Hz |
| **推論速度** | 快（real-time capable，MLX 版 8.71× faster than PyTorch MPS） |
| **品質** | 社群驗證最成熟，品質很高 |
| **Zero-shot** | **否**——需要每位說話者 ~10 分鐘乾淨音訊做 fine-tuning |
| **授權** | MIT |
| **狀態** | ⚠️ 已整合 wrapper，但需要使用者自己提供 fine-tuned model |
| **限制** | 不是 zero-shot；`rvc-python` PyPI 套件有 dependency 問題（需要舊版 numpy），無法直接 pip install |

### MeanVC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot lightweight streaming voice conversion |
| **架構** | ASR encoder + ECAPA-TDNN speaker encoder + DiT decoder (Mean Flows) + Vocos vocoder |
| **參數量** | **僅 14M**（最小的 VC 模型） |
| **取樣率** | 16000Hz |
| **推論速度** | RTF 0.136 on single CPU core；支援 1-step inference |
| **品質** | 中等偏上（1-step 品質可接受，2-step 更好） |
| **Zero-shot** | 是 |
| **授權** | Apache 2.0 |
| **狀態** | ⚠️ 已整合 wrapper + backend，但需要 clone meanvc-ref repo 並下載 checkpoints |
| **限制** | 需要額外的 ASR model (WeNet/FunASR) 和 speaker verification model；checkpoints 部分需手動從 Google Drive 下載 |

---

## 評估後未整合的模型

### FreeVC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot text-free one-shot voice conversion |
| **架構** | WavLM-Large (~316M) + VITS decoder + information bottleneck |
| **參數量** | WavLM 316M + VITS decoder ≈ ~350M |
| **取樣率** | 16000Hz / 24000Hz |
| **授權** | MIT |
| **未整合原因** | 官方程式碼假設 CUDA（寫死 `CUDA_VISIBLE_DEVICES`），無 CPU/MPS 支援；checkpoint 需手動從 OneDrive 下載（無 HuggingFace）；無 Python API（只有 CLI）；WavLM-Large 與 kNN-VC 重疊 |
| **GitHub** | https://github.com/OlaWod/FreeVC (~2,000 stars) |

### GPT-SoVITS

| 項目 | 內容 |
|------|------|
| **類型** | Hybrid GPT + SoVITS 語音生成（TTS 為主，支援 VC） |
| **架構** | GPT (semantic prediction) + SoVITS (VITS-based acoustic model) |
| **參數量** | V3: 330M+77M = 407M；V4: 更大 |
| **推論速度** | RTF 0.028 (4060Ti) |
| **授權** | MIT |
| **未整合原因** | 多元件系統（GPT + SoVITS + 音訊前處理），移植工作量極大（難度 ⭐⭐⭐⭐）；主要設計為 TTS 不是 VC |
| **GitHub** | https://github.com/RVC-Boss/GPT-SoVITS (~56,400 stars) |

### Vevo (Amphion)

| 項目 | 內容 |
|------|------|
| **類型** | Unified speech + singing voice conversion |
| **架構** | Autoregressive Transformer (~780M) + Flow-Matching Transformer + Vocos vocoder |
| **參數量** | AR: ~780M, FM: 數百M |
| **推論速度** | 未實測（AR 模型較大，推論速度待評估） |
| **授權** | MIT / Apache 2.0 |
| **未整合原因** | AR model 780M params 較大；需要移植 VQ-VAE tokenizers；架構複雜度高（難度 ⭐⭐⭐）；但**未來最有潛力**——統一 speech + singing，Vocos vocoder MLX 已有 |
| **GitHub** | https://github.com/open-mmlab/Amphion (~9,400 stars) |

### so-vits-svc (SoftVC VITS SVC)

| 項目 | 內容 |
|------|------|
| **類型** | Per-speaker singing voice conversion |
| **架構** | HuBERT/ContentVec + NSF-HiFiGAN + VITS generator |
| **參數量** | ~150M |
| **授權** | AGPL-3.0（已 archived） |
| **未整合原因** | 需要 per-speaker fine-tuning（非 zero-shot）；已被 archived 不再維護；NSF-HiFiGAN 需 STFT/iSTFT（MLX 尚未原生支援）；AGPL 授權限制 |
| **GitHub** | https://github.com/svc-develop-team/so-vits-svc (~28,000 stars, archived) |

### DDSP-SVC

| 項目 | 內容 |
|------|------|
| **類型** | Singing voice conversion (DDSP-based) |
| **架構** | HuBERT + DDSP (Differentiable DSP) synthesizer + optional diffusion enhancer |
| **參數量** | 較小（DDSP 本身輕量） |
| **授權** | 未明確 |
| **未整合原因** | DDSP 需要 FFT + harmonic synthesis，MLX FFT 支援有限；需要 per-speaker 訓練（非 zero-shot）；被 RVC 和 Seed-VC SVC 模式取代 |
| **GitHub** | https://github.com/yxlllc/DDSP-SVC (~1,500 stars) |

### HierSpeech++

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot voice conversion + super-resolution |
| **架構** | Hierarchical Conditional VAE + Normalizing Flows + SpeechSR (16kHz→48kHz) |
| **參數量** | 數百M（多版本） |
| **授權** | MIT |
| **未整合原因** | 架構複雜（VAE + normalizing flows + super-resolution 三階段）；checkpoint 格式為 .pth，需轉換；社群活躍度低於 Seed-VC |
| **GitHub** | https://github.com/sh-lee-prml/HierSpeechpp (~1,240 stars) |

### DiffVC

| 項目 | 內容 |
|------|------|
| **類型** | One-shot many-to-many voice conversion |
| **架構** | Transformer encoder + Diffusion decoder (U-Net) + HiFi-GAN |
| **參數量** | 未明確 |
| **授權** | Apache 2.0 |
| **未整合原因** | U-Net diffusion 較舊（已被 DiT 取代）；HiFi-GAN 無 MLX 版 |
| **GitHub** | https://github.com/huawei-noah/Speech-Backbones |

### EZ-VC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot VC using pure self-supervised features |
| **架構** | WavLM + CFM (Conditional Flow Matching) decoder |
| **參數量** | 未明確（架構精簡） |
| **授權** | 未明確 |
| **未整合原因** | 依賴 WavLM（與 kNN-VC 重疊）；論文較新（EMNLP 2025），開源程式碼成熟度待確認 |

### StableVC / AdaptVC / OneVoice / GenVC

| 模型 | 年份 | 核心特色 | 未整合原因 |
|------|------|---------|-----------|
| **StableVC** | AAAI 2025 | 獨立控制 timbre 和 style | 開源程式碼不成熟 |
| **AdaptVC** | ICASSP 2025 | HuBERT learnable adapter | 依賴 HuBERT MLX port |
| **OneVoice** | 2026 | 統一 speech + expressive + singing VC | Patch Diffusion + MoE，架構過於複雜 |
| **GenVC** | 2025 | 自監督 discrete token + AR Transformer | 開源程式碼不成熟 |

### StreamVC (Google Research)

| 項目 | 內容 |
|------|------|
| **類型** | Real-time streaming voice conversion |
| **架構** | SoundStream-based fully causal codec，~20M params |
| **推論速度** | 60ms lookahead + ~10ms inference = ~70ms total latency |
| **未整合原因** | **無官方開源**——僅有非官方 PyTorch 實作且無 checkpoint |

---

## 模型比較速查表

| 模型 | 參數量 | Zero-shot | 推論速度 | 品質 | 狀態 |
|------|-------|-----------|---------|------|------|
| **OpenVoice V2** | ~30M | ✅ | ⚡ 28ms/block | 中（音色） | ✅ Real-time 首選 |
| **MeanVC** | 14M | ✅ | ⚡ RTF 0.14 | 中上 | ⚠️ 需額外 setup |
| **Seed-VC** | 98M | ✅ | 🐢 RTF ~1.9 | 高 | ✅ 品質首選 |
| **Seed-VC SVC** | 200M | ✅ | 🐌 RTF ~68 | 高 | ✅ 唱歌首選 |
| **kNN-VC** | 333M | ✅ | 🏃 3.4s/10s | 中 | ✅ |
| **CosyVoice3** | 500M | ✅ | 🏃 快 | 好（TTS） | ✅ 文字輸入 |
| **RVC** | 150M | ❌ | ⚡ 快 | 高 | ⚠️ 需 fine-tune |
| **Vevo** | 780M+ | ✅ | 未測 | 很高 | ❌ 未來目標 |
| **FreeVC** | 350M | ✅ | 未測 | 高 | ❌ CUDA only |
| **GPT-SoVITS** | 407M | ✅ | ⚡ RTF 0.03 | 很高 | ❌ 太複雜 |

---

## MLX 生態共用元件

以下元件已在 mlx-audio 中可用，未來純 MLX 移植可直接復用：

| 元件 | 狀態 | 用途 |
|------|------|------|
| Whisper encoder | ✅ | Content feature extraction (Seed-VC) |
| Vocos vocoder | ✅ | Mel → waveform (MeanVC, Vevo) |
| BigVGAN | ✅ | Mel → waveform (Seed-VC) |
| EnCodec / SNAC / DAC | ✅ | Audio codec (各模型) |
| SpeakerEncoder | ✅ | Speaker embedding (Spark TTS) |
| ECAPA-TDNN | ✅ | Speaker embedding (MeanVC) |
| HuBERT / ContentVec | ❌ | Content encoding (RVC, FreeVC, so-vits-svc) |
| WavLM | ❌ | Content encoding (kNN-VC, FreeVC, EZ-VC) |
| HiFi-GAN | ❌ | Vocoder (RVC, kNN-VC, FreeVC) |
| RMVPE | ⚠️ 有 MLX 版 | Pitch extraction (RVC, Seed-VC SVC) |
