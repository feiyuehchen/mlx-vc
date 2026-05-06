# Voice Conversion Model Catalog

This document records every voice conversion model surveyed and tested during mlx-vc development — those integrated, those that failed integration, and those evaluated and skipped.

For the latest objective benchmark numbers (UTMOS / SECS / WER), see [Evaluation Metrics](../guides/evaluation.md).

---

## Integrated models

### Seed-VC

| Field | Value |
|-------|-------|
| **Type** | Zero-shot voice conversion (speech + singing) |
| **Architecture** | Whisper encoder + DiT (Diffusion Transformer) + CFM (Conditional Flow Matching) + BigVGAN |
| **Params** | Speech: ~98M (512-dim, 8-head, 13-layer); SVC: ~200M (768-dim, 12-head, 17-layer) |
| **Sample rate** | Speech: 22050 Hz; SVC: 44100 Hz |
| **Inference speed** | Speech 25 steps: RTF ~1.9 (10 s audio → 19 s); 10 steps: RTF ~1.2 |
| **Quality** | SECS 0.847 in our latest benchmark — best of the integrated set |
| **Zero-shot** | Yes, 1–30 s reference audio |
| **F0 conditioning** | SVC mode preserves pitch / melody |
| **License** | MIT |
| **Status** | Integrated (PyTorch MPS, subprocess backend) |
| **Limitations** | SVC mode is very slow (~18 min for 16 s); requires `seed-vc-ref` external repo |

### OpenVoice V2

| Field | Value |
|-------|-------|
| **Type** | Zero-shot tone color conversion |
| **Architecture** | VITS-based SynthesizerTrn + ToneColorConverter |
| **Params** | ~30M (lightweight) |
| **Sample rate** | 22050 Hz |
| **Inference speed** | Very fast: 0.7 s for 16 s audio; 28 ms for a 0.3 s block |
| **Quality** | Good timbre transfer, but only timbre — no accent / emotion / prosody transfer |
| **Zero-shot** | Yes |
| **Languages** | EN, ZH, JA, KO, FR, ES |
| **License** | MIT |
| **Status** | Integrated (PyTorch MPS, subprocess backend); **used by the realtime demo** |
| **Limitations** | Tone-color only; speaker similarity below Seed-VC |

### kNN-VC

| Field | Value |
|-------|-------|
| **Type** | Zero-shot non-parametric voice conversion |
| **Architecture** | WavLM-Large (frozen, layer 6) + k-nearest neighbors + HiFi-GAN vocoder |
| **Params** | WavLM ~316M + HiFi-GAN ~16.5M ≈ 333M (kNN itself parameter-free) |
| **Sample rate** | 16000 Hz |
| **Inference speed** | ~3.4 s for 10 s audio (CPU) |
| **Quality** | Reasonable; chops audibly without 3+ min reference |
| **Zero-shot** | Yes (kNN is non-parametric) |
| **License** | MIT |
| **Status** | Integrated (PyTorch CPU, subprocess backend) |
| **Limitations** | WavLM-Large weight is 1.18 GB; MPS blocked by upstream `np.float64` weights and cross-device child modules; 16 kHz is on the lower end |

### FreeVC / FreeVC-s

| Field | Value |
|-------|-------|
| **Type** | Zero-shot one-shot voice conversion |
| **Architecture** | WavLM-Large content encoder + VITS decoder + information bottleneck |
| **Variants** | `freevc` (with d-vector speaker encoder) / `freevc-s` (mel-spectrogram resize, no speaker encoder) |
| **Params** | WavLM 316M + VITS decoder ≈ ~350M |
| **Sample rate** | 16000 Hz |
| **Inference speed** | ~3 s for 10 s audio |
| **License** | MIT |
| **Status** | Integrated (PyTorch MPS, subprocess backend) |
| **Limitations** | Lower SECS than Seed-VC; -s variant tends to outperform the speaker-encoder variant |

### MeanVC

| Field | Value |
|-------|-------|
| **Type** | Zero-shot lightweight streaming voice conversion |
| **Architecture** | ASR encoder + WavLM-Large + ECAPA-TDNN speaker encoder + DiT decoder (Mean Flows) + Vocos vocoder |
| **Params** | **14M** for the DiT (smallest VC integrated) |
| **Sample rate** | 16000 Hz |
| **Inference speed** | RTF ~0.14 on a single CPU core; supports 1-step inference |
| **Quality** | Good on Chinese (training distribution); poor on English (out of distribution) |
| **Zero-shot** | Yes |
| **License** | Apache 2.0 |
| **Status** | Integrated; speaker-verification checkpoint must be downloaded manually from Google Drive (not on HF) |
| **Limitations** | Trained on Chinese — English WER is high; pre-saved TorchScript Vocos hits the MPS graph-fuser bug, so MeanVC stays on CPU |

### SpeechT5-VC

| Field | Value |
|-------|-------|
| **Type** | Audio-to-audio voice conversion via transformer seq2seq |
| **Architecture** | SpeechT5 encoder–decoder + x-vector speaker conditioning + HiFi-GAN vocoder |
| **Sample rate** | 16000 Hz |
| **Inference speed** | ~30 s for 10 s audio |
| **License** | MIT |
| **Status** | Integrated via HuggingFace `transformers` |
| **Limitations** | Trained on read-speech corpora (CMU-ARCTIC / VCTK); collapses on natural lecture audio (UTMOS 1.28 in our benchmark). Documented as a contrastive datapoint, not a quality option |

### RVC (Retrieval-based Voice Conversion)

| Field | Value |
|-------|-------|
| **Type** | Per-speaker fine-tuned voice conversion |
| **Architecture** | HuBERT/ContentVec content + VITS generator + FAISS retrieval + RMVPE pitch |
| **Params** | HuBERT ~95M + VITS ~55M ≈ 150M |
| **Sample rate** | 48000 Hz |
| **Inference speed** | Fast (real-time capable; the MLX port is ~8.7× faster than PyTorch MPS) |
| **Quality** | High when paired with a properly fine-tuned model |
| **Zero-shot** | **No** — requires ~10 minutes of clean per-speaker audio for fine-tuning |
| **License** | MIT |
| **Status** | Wrapper integrated via Acelogic's MLX port; user supplies the `.npz` model |
| **Limitations** | Not zero-shot; the speaker is baked into the `.npz` — the `reference` argument is ignored. Runs in its own Python 3.10 venv (numpy<2 pin) |

### Chatterbox / cosyvoice (TTS-clone)

| Field | Value |
|-------|-------|
| **Type** | TTS with voice cloning (NOT true voice conversion) |
| **Architecture** | S3Tokenizer + T3 language model + S3Gen vocoder |
| **Params** | ~500M (Chatterbox fp16) |
| **Sample rate** | 24000 Hz |
| **Inference speed** | ~7 s for 3 s of text |
| **License** | Apache 2.0 |
| **Status** | Integrated via mlx-audio (in-process) |
| **Limitations** | Input is **text**, not audio — source is Whisper-transcribed first, then synthesized in the target voice. Source prosody / emotion / timing are regenerated, not preserved |

### Pocket-TTS (TTS-clone)

| Field | Value |
|-------|-------|
| **Type** | Lightweight English voice-cloning TTS |
| **Architecture** | Kyutai's Pocket-TTS via mlx-audio |
| **Params** | ~100M; ~235 MB on disk |
| **Sample rate** | 24000 Hz |
| **Inference speed** | ~1.4 s of synthesis for 4 s of generated audio |
| **License** | MIT |
| **Status** | Integrated via the generic `tts_clone_infer.py` subprocess runner |
| **Limitations** | Same text-path caveat as Chatterbox: source content goes through Whisper |

---

## Evaluated but not integrated

### GPT-SoVITS

| Field | Value |
|-------|-------|
| **Type** | Hybrid GPT + SoVITS speech generation (TTS-primary, supports VC) |
| **Architecture** | GPT (semantic prediction) + SoVITS (VITS-based acoustic model) |
| **Params** | V3: 330M+77M = 407M; V4: larger |
| **Inference speed** | RTF 0.028 (RTX 4060 Ti) |
| **License** | MIT |
| **Why skipped** | Multi-component pipeline (GPT + SoVITS + audio preprocessing); large porting cost (difficulty ⭐⭐⭐⭐); primarily designed as TTS, not VC |
| **Repo** | <https://github.com/RVC-Boss/GPT-SoVITS> (~56k stars) |

### Vevo (Amphion)

| Field | Value |
|-------|-------|
| **Type** | Unified speech + singing voice conversion |
| **Architecture** | Autoregressive Transformer (~780M) + Flow-Matching Transformer + Vocos vocoder |
| **Params** | AR: ~780M; FM: hundreds of M |
| **License** | MIT / Apache 2.0 |
| **Why skipped** | Large AR model; needs porting of VQ-VAE tokenizers; complex (difficulty ⭐⭐⭐). Highest **future potential** — unified speech + singing, Vocos vocoder is already on MLX |
| **Repo** | <https://github.com/open-mmlab/Amphion> (~9.4k stars) |

### so-vits-svc (SoftVC VITS SVC)

| Field | Value |
|-------|-------|
| **Type** | Per-speaker singing voice conversion |
| **Architecture** | HuBERT/ContentVec + NSF-HiFi-GAN + VITS generator |
| **Params** | ~150M |
| **License** | AGPL-3.0 (archived) |
| **Why skipped** | Per-speaker fine-tuning required (not zero-shot); repo archived; NSF-HiFi-GAN needs STFT/iSTFT (not yet native MLX); AGPL is restrictive |
| **Repo** | <https://github.com/svc-develop-team/so-vits-svc> (~28k stars, archived) |

### DDSP-SVC

| Field | Value |
|-------|-------|
| **Type** | Singing voice conversion (DDSP-based) |
| **Architecture** | HuBERT + DDSP synthesizer + optional diffusion enhancer |
| **Params** | Small (DDSP itself is lightweight) |
| **License** | Unspecified |
| **Why skipped** | DDSP needs FFT + harmonic synthesis; MLX FFT support is limited; per-speaker training required (not zero-shot); largely superseded by RVC and Seed-VC SVC |
| **Repo** | <https://github.com/yxlllc/DDSP-SVC> (~1.5k stars) |

### HierSpeech++

| Field | Value |
|-------|-------|
| **Type** | Zero-shot voice conversion + super-resolution |
| **Architecture** | Hierarchical Conditional VAE + Normalizing Flows + SpeechSR (16kHz → 48kHz) |
| **Params** | Hundreds of M (multiple variants) |
| **License** | MIT |
| **Why skipped** | Complex 3-stage pipeline (VAE + flows + SR); checkpoints in `.pth` need conversion; less community traction than Seed-VC |
| **Repo** | <https://github.com/sh-lee-prml/HierSpeechpp> (~1.2k stars) |

### DiffVC

| Field | Value |
|-------|-------|
| **Type** | One-shot many-to-many voice conversion |
| **Architecture** | Transformer encoder + Diffusion decoder (U-Net) + HiFi-GAN |
| **License** | Apache 2.0 |
| **Why skipped** | U-Net diffusion is dated (superseded by DiT); HiFi-GAN has no MLX implementation |
| **Repo** | <https://github.com/huawei-noah/Speech-Backbones> |

### EZ-VC

| Field | Value |
|-------|-------|
| **Type** | Zero-shot VC using pure self-supervised features |
| **Architecture** | WavLM + CFM (Conditional Flow Matching) decoder |
| **License** | Unspecified |
| **Why skipped** | Depends on WavLM (overlap with kNN-VC); recent paper (EMNLP 2025), open-source maturity unclear |

### StableVC / AdaptVC / OneVoice / GenVC

| Model | Year | Highlight | Why skipped |
|-------|------|-----------|-------------|
| **StableVC** | AAAI 2025 | Independent timbre + style control | Open-source code immature |
| **AdaptVC** | ICASSP 2025 | HuBERT learnable adapter | Requires HuBERT MLX port |
| **OneVoice** | 2026 | Unified speech + expressive + singing VC | Patch Diffusion + MoE; over-complex |
| **GenVC** | 2025 | Self-supervised discrete tokens + AR transformer | Open-source code immature |

### StreamVC (Google Research)

| Field | Value |
|-------|-------|
| **Type** | Real-time streaming voice conversion |
| **Architecture** | SoundStream-based fully causal codec, ~20M params |
| **Latency** | 60 ms lookahead + ~10 ms inference ≈ 70 ms total |
| **Why skipped** | **No official open source** — only unofficial PyTorch implementations with no checkpoint |

---

## Quick comparison table

| Model | Params | Zero-shot | Speed | Quality | Status |
|-------|--------|-----------|-------|---------|--------|
| **OpenVoice V2** | ~30M | ✅ | ⚡ 28 ms/block | Mid (timbre) | ✅ realtime pick |
| **MeanVC** | 14M | ✅ | ⚡ RTF 0.14 | Mid–high (zh) | ⚠️ needs setup |
| **Seed-VC** | 98M | ✅ | 🐢 RTF ~1.9 | High | ✅ quality pick |
| **Seed-VC SVC** | 200M | ✅ | 🐌 RTF ~68 | High | ✅ singing pick |
| **kNN-VC** | 333M | ✅ | 🏃 3.4 s / 10 s | Mid | ✅ |
| **FreeVC / -s** | 350M | ✅ | 🏃 3 s / 10 s | Mid | ✅ |
| **SpeechT5-VC** | ~150M | ✅ | 🐢 30 s / 10 s | Low (lecture) | ✅ contrastive |
| **Chatterbox** | 500M | ✅ | 🏃 fast | Good (TTS) | ✅ text input |
| **Pocket-TTS** | 100M | ✅ | ⚡ fast | Good (TTS) | ✅ text input |
| **RVC** | 150M | ❌ | ⚡ fast | High | ⚠️ needs fine-tune |
| **Vevo** | 780M+ | ✅ | not measured | Very high | ❌ future target |
| **GPT-SoVITS** | 407M | ✅ | ⚡ RTF 0.03 | Very high | ❌ too complex |

---

## MLX ecosystem reusable components

These components are already available in mlx-audio and can be reused if a future port goes pure MLX:

| Component | Status | Used by |
|-----------|--------|---------|
| Whisper encoder | ✅ | Content feature extraction (Seed-VC) |
| Vocos vocoder | ✅ | Mel → waveform (MeanVC, Vevo) |
| BigVGAN | ✅ | Mel → waveform (Seed-VC) |
| EnCodec / SNAC / DAC | ✅ | Audio codec |
| SpeakerEncoder | ✅ | Speaker embedding (Spark TTS) |
| ECAPA-TDNN | ✅ | Speaker embedding (MeanVC) |
| HuBERT / ContentVec | ❌ | Content encoding (RVC, FreeVC, so-vits-svc) |
| WavLM | ❌ | Content encoding (kNN-VC, FreeVC, EZ-VC) |
| HiFi-GAN | ❌ | Vocoder (RVC, kNN-VC, FreeVC) |
| RMVPE | ⚠️ MLX port exists | Pitch extraction (RVC, Seed-VC SVC) |
