# Voice Conversion 模型全覽

本文件整理 mlx-vc 開發過程中調研和實測的所有 VC 模型，包含已整合的、測試未通過的、以及評估後放棄的模型。

最新客觀 benchmark（UTMOS / SECS / WER）請見 [評估指標](../guides/evaluation.md)。

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
| **品質** | 最新 benchmark SECS 0.847 — 整合的所有模型中最高 |
| **Zero-shot** | 是，1–30 秒 reference audio 即可 |
| **F0 conditioning** | SVC 模式支援（保留原始音高/旋律） |
| **授權** | MIT |
| **狀態** | 已整合（PyTorch MPS，subprocess backend） |
| **限制** | SVC 模式非常慢（16s 音訊需 ~18 分鐘）；需要 `seed-vc-ref` 外部 repo |

### OpenVoice V2

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot tone color conversion |
| **架構** | VITS-based SynthesizerTrn + ToneColorConverter |
| **參數量** | ~30M（相對輕量） |
| **取樣率** | 22050Hz |
| **推論速度** | **極快**：16s 音訊僅需 0.7s；0.3s block 僅需 28ms |
| **品質** | 音色轉換良好，但只轉換 timbre（不轉換口音、情緒、語調） |
| **Zero-shot** | 是 |
| **多語言** | EN, ZH, JA, KO, FR, ES |
| **授權** | MIT |
| **狀態** | 已整合（PyTorch MPS，subprocess backend）；**realtime demo 用此模型** |
| **限制** | 只轉換音色，不轉換說話方式；speaker similarity 不如 Seed-VC |

### kNN-VC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot non-parametric voice conversion |
| **架構** | WavLM-Large (frozen, layer 6) + k-nearest neighbors + HiFi-GAN vocoder |
| **參數量** | WavLM ~316M + HiFi-GAN ~16.5M ≈ 333M（kNN 本身無參數） |
| **取樣率** | 16000Hz |
| **推論速度** | 10s 音訊約 3.4s（CPU） |
| **品質** | 中等；reference < 3 分鐘會聽到明顯切片感 |
| **Zero-shot** | 是（kNN 是 non-parametric） |
| **授權** | MIT |
| **狀態** | 已整合（PyTorch CPU，subprocess backend） |
| **限制** | WavLM-Large 1.18GB 很大；MPS 因 upstream `np.float64` weights + 跨 device 子模組無法搬上去；16kHz 取樣率較低 |

### FreeVC / FreeVC-s

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot one-shot voice conversion |
| **架構** | WavLM-Large content encoder + VITS decoder + information bottleneck |
| **變體** | `freevc`（含 d-vector speaker encoder）/ `freevc-s`（mel-spectrogram resize，無 speaker encoder） |
| **參數量** | WavLM 316M + VITS decoder ≈ ~350M |
| **取樣率** | 16000Hz |
| **推論速度** | 10s 音訊約 3s |
| **授權** | MIT |
| **狀態** | 已整合（PyTorch MPS，subprocess backend） |
| **限制** | SECS 不及 Seed-VC；`-s` 變體實測勝過 speaker-encoder 變體 |

### MeanVC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot lightweight streaming voice conversion |
| **架構** | ASR encoder + WavLM-Large + ECAPA-TDNN speaker encoder + DiT decoder (Mean Flows) + Vocos vocoder |
| **參數量** | DiT 僅 **14M**（已整合中最小） |
| **取樣率** | 16000Hz |
| **推論速度** | RTF ~0.14（單 CPU core）；支援 1-step inference |
| **品質** | 中文好（訓練語料）；英文差（OOD） |
| **Zero-shot** | 是 |
| **授權** | Apache 2.0 |
| **狀態** | 已整合；speaker verification checkpoint 需手動從 Google Drive 下載（不在 HF） |
| **限制** | 訓練語料是中文 — 英文 WER 高；預存 TorchScript Vocos 撞 MPS graph-fuser bug，故 MeanVC 留 CPU |

### SpeechT5-VC

| 項目 | 內容 |
|------|------|
| **類型** | Audio-to-audio voice conversion via transformer seq2seq |
| **架構** | SpeechT5 encoder–decoder + x-vector speaker conditioning + HiFi-GAN vocoder |
| **取樣率** | 16000Hz |
| **推論速度** | 10s 音訊約 30s |
| **授權** | MIT |
| **狀態** | 透過 HuggingFace `transformers` 整合 |
| **限制** | 訓練語料是 read-speech (CMU-ARCTIC / VCTK)；遇到自然 lecture audio 會崩（benchmark UTMOS 1.28）。當作對照組記錄，不做品質選項 |

### RVC（Retrieval-based Voice Conversion）

| 項目 | 內容 |
|------|------|
| **類型** | Per-speaker fine-tuned voice conversion |
| **架構** | HuBERT/ContentVec content + VITS generator + FAISS retrieval + RMVPE pitch |
| **參數量** | HuBERT ~95M + VITS ~55M ≈ 150M |
| **取樣率** | 48000Hz |
| **推論速度** | 快（real-time capable，MLX 版比 PyTorch MPS 快 ~8.7×） |
| **品質** | 配對好的 fine-tuned model 時品質很高 |
| **Zero-shot** | **否** — 需要每位 speaker ~10 分鐘乾淨音訊做 fine-tuning |
| **授權** | MIT |
| **狀態** | Wrapper 透過 Acelogic 的 MLX port 整合；user 自備 `.npz` model |
| **限制** | 不是 zero-shot；speaker baked 進 `.npz` — `reference` 參數會被忽略。跑在自己的 Python 3.10 venv（pin numpy<2） |

### Chatterbox / cosyvoice（TTS-clone）

| 項目 | 內容 |
|------|------|
| **類型** | TTS with voice cloning（**不是**真正的 voice conversion） |
| **架構** | S3Tokenizer + T3 Language Model + S3Gen Vocoder |
| **參數量** | ~500M（Chatterbox fp16） |
| **取樣率** | 24000Hz |
| **推論速度** | 3s 文字約 7s 生成 |
| **授權** | Apache 2.0 |
| **狀態** | 已整合（透過 mlx-audio，in-process） |
| **限制** | 輸入是**文字** — source 先 Whisper 轉錄再合成。Source 的 prosody / 情緒 / 時序都是重新生成，不是保留 |

### Pocket-TTS（TTS-clone）

| 項目 | 內容 |
|------|------|
| **類型** | 輕量英文 voice-cloning TTS |
| **架構** | Kyutai 的 Pocket-TTS 透過 mlx-audio |
| **參數量** | ~100M；磁碟 ~235 MB |
| **取樣率** | 24000Hz |
| **推論速度** | 4s 生成音檔 ~1.4s 合成 |
| **授權** | MIT |
| **狀態** | 透過通用 `tts_clone_infer.py` subprocess runner 整合 |
| **限制** | 跟 Chatterbox 同樣的 text-path 限制：source 走 Whisper |

---

## 評估後未整合的模型

### GPT-SoVITS

| 項目 | 內容 |
|------|------|
| **類型** | Hybrid GPT + SoVITS 語音生成（TTS 為主，支援 VC） |
| **架構** | GPT (semantic prediction) + SoVITS (VITS-based acoustic model) |
| **參數量** | V3: 330M+77M = 407M；V4: 更大 |
| **推論速度** | RTF 0.028（RTX 4060 Ti） |
| **授權** | MIT |
| **未整合原因** | 多元件系統（GPT + SoVITS + 音訊前處理），移植工作量極大（難度 ⭐⭐⭐⭐）；主要設計為 TTS 不是 VC |
| **GitHub** | <https://github.com/RVC-Boss/GPT-SoVITS>（~56k stars） |

### Vevo (Amphion)

| 項目 | 內容 |
|------|------|
| **類型** | Unified speech + singing voice conversion |
| **架構** | Autoregressive Transformer (~780M) + Flow-Matching Transformer + Vocos vocoder |
| **參數量** | AR: ~780M；FM: 數百 M |
| **授權** | MIT / Apache 2.0 |
| **未整合原因** | AR model 較大；需要移植 VQ-VAE tokenizers；架構複雜（難度 ⭐⭐⭐）。**未來最有潛力** — 統一 speech + singing，Vocos vocoder MLX 已有 |
| **GitHub** | <https://github.com/open-mmlab/Amphion>（~9.4k stars） |

### so-vits-svc（SoftVC VITS SVC）

| 項目 | 內容 |
|------|------|
| **類型** | Per-speaker singing voice conversion |
| **架構** | HuBERT/ContentVec + NSF-HiFi-GAN + VITS generator |
| **參數量** | ~150M |
| **授權** | AGPL-3.0（已 archived） |
| **未整合原因** | 需要 per-speaker fine-tuning（非 zero-shot）；repo archived；NSF-HiFi-GAN 需 STFT/iSTFT（MLX 還沒原生支援）；AGPL 限制較嚴 |
| **GitHub** | <https://github.com/svc-develop-team/so-vits-svc>（~28k stars，archived） |

### DDSP-SVC

| 項目 | 內容 |
|------|------|
| **類型** | Singing voice conversion (DDSP-based) |
| **架構** | HuBERT + DDSP synthesizer + optional diffusion enhancer |
| **參數量** | 較小（DDSP 本身輕量） |
| **授權** | 未明確 |
| **未整合原因** | DDSP 需要 FFT + harmonic synthesis，MLX FFT 支援有限；需要 per-speaker 訓練（非 zero-shot）；被 RVC 和 Seed-VC SVC 取代 |
| **GitHub** | <https://github.com/yxlllc/DDSP-SVC>（~1.5k stars） |

### HierSpeech++

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot voice conversion + super-resolution |
| **架構** | Hierarchical Conditional VAE + Normalizing Flows + SpeechSR (16kHz → 48kHz) |
| **參數量** | 數百 M（多版本） |
| **授權** | MIT |
| **未整合原因** | 架構複雜（VAE + flows + super-resolution 三階段）；checkpoint `.pth` 需轉換；社群活躍度低於 Seed-VC |
| **GitHub** | <https://github.com/sh-lee-prml/HierSpeechpp>（~1.2k stars） |

### DiffVC

| 項目 | 內容 |
|------|------|
| **類型** | One-shot many-to-many voice conversion |
| **架構** | Transformer encoder + Diffusion decoder (U-Net) + HiFi-GAN |
| **授權** | Apache 2.0 |
| **未整合原因** | U-Net diffusion 較舊（已被 DiT 取代）；HiFi-GAN 無 MLX 版 |
| **GitHub** | <https://github.com/huawei-noah/Speech-Backbones> |

### EZ-VC

| 項目 | 內容 |
|------|------|
| **類型** | Zero-shot VC using pure self-supervised features |
| **架構** | WavLM + CFM (Conditional Flow Matching) decoder |
| **授權** | 未明確 |
| **未整合原因** | 依賴 WavLM（與 kNN-VC 重疊）；論文較新（EMNLP 2025），開源程式碼成熟度待確認 |

### StableVC / AdaptVC / OneVoice / GenVC

| 模型 | 年份 | 核心特色 | 未整合原因 |
|------|------|---------|-----------|
| **StableVC** | AAAI 2025 | 獨立控制 timbre 和 style | 開源程式碼不成熟 |
| **AdaptVC** | ICASSP 2025 | HuBERT learnable adapter | 需要 HuBERT MLX port |
| **OneVoice** | 2026 | 統一 speech + expressive + singing VC | Patch Diffusion + MoE，架構過於複雜 |
| **GenVC** | 2025 | 自監督 discrete token + AR Transformer | 開源程式碼不成熟 |

### StreamVC（Google Research）

| 項目 | 內容 |
|------|------|
| **類型** | Real-time streaming voice conversion |
| **架構** | SoundStream-based fully causal codec，~20M params |
| **延遲** | 60ms lookahead + ~10ms inference ≈ 70ms |
| **未整合原因** | **無官方開源** — 僅有非官方 PyTorch 實作且無 checkpoint |

---

## 模型比較速查表

| 模型 | 參數量 | Zero-shot | 推論速度 | 品質 | 狀態 |
|------|-------|-----------|---------|------|------|
| **OpenVoice V2** | ~30M | ✅ | ⚡ 28ms/block | 中（音色） | ✅ Real-time 首選 |
| **MeanVC** | 14M | ✅ | ⚡ RTF 0.14 | 中上（中文） | ⚠️ 需 setup |
| **Seed-VC** | 98M | ✅ | 🐢 RTF ~1.9 | 高 | ✅ 品質首選 |
| **Seed-VC SVC** | 200M | ✅ | 🐌 RTF ~68 | 高 | ✅ 唱歌首選 |
| **kNN-VC** | 333M | ✅ | 🏃 3.4s/10s | 中 | ✅ |
| **FreeVC / -s** | 350M | ✅ | 🏃 3s/10s | 中 | ✅ |
| **SpeechT5-VC** | ~150M | ✅ | 🐢 30s/10s | 低（lecture） | ✅ 對照組 |
| **Chatterbox** | 500M | ✅ | 🏃 快 | 好（TTS） | ✅ 文字輸入 |
| **Pocket-TTS** | 100M | ✅ | ⚡ 快 | 好（TTS） | ✅ 文字輸入 |
| **RVC** | 150M | ❌ | ⚡ 快 | 高 | ⚠️ 需 fine-tune |
| **Vevo** | 780M+ | ✅ | 未測 | 很高 | ❌ 未來目標 |
| **GPT-SoVITS** | 407M | ✅ | ⚡ RTF 0.03 | 很高 | ❌ 太複雜 |

---

## MLX 生態共用元件

下列元件已在 mlx-audio 可用，未來純 MLX 移植可直接復用：

| 元件 | 狀態 | 用途 |
|------|------|------|
| Whisper encoder | ✅ | Content feature extraction（Seed-VC） |
| Vocos vocoder | ✅ | Mel → waveform（MeanVC, Vevo） |
| BigVGAN | ✅ | Mel → waveform（Seed-VC） |
| EnCodec / SNAC / DAC | ✅ | Audio codec |
| SpeakerEncoder | ✅ | Speaker embedding（Spark TTS） |
| ECAPA-TDNN | ✅ | Speaker embedding（MeanVC） |
| HuBERT / ContentVec | ❌ | Content encoding（RVC, FreeVC, so-vits-svc） |
| WavLM | ❌ | Content encoding（kNN-VC, FreeVC, EZ-VC） |
| HiFi-GAN | ❌ | Vocoder（RVC, kNN-VC, FreeVC） |
| RMVPE | ⚠️ MLX port 已有 | Pitch extraction（RVC, Seed-VC SVC） |
