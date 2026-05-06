# Changelog

All notable changes to mlx-vc are documented here.  The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) starting
from v1.0.0.  Pre-1.0 releases may make breaking changes between minor
versions.

## [Unreleased]

### Added
- LICENSE, CONTRIBUTING, SECURITY, CHANGELOG, GitHub issue / PR templates,
  Dependabot configuration — full OSS metadata pass.
- README badges (CI status, license, Python version, MPS support, code style).

## [0.1.0] — 2025-04-20

Initial public release.

### Added
- Subprocess-isolated runner (`mlx_vc.backend.run_backend`) with `BACKENDS`
  registry covering 9 inference scripts.
- In-process Python wrapper classes for 6 models under `mlx_vc.models.*`.
- FastAPI server (`mlx_vc.server`) with single-model `/v1/audio/convert`,
  multi-model `/v1/audio/convert/batch`, job polling
  `/v1/jobs/{job_id}`, reference upload `/v1/audio/upload-reference`,
  and WebSocket realtime endpoint `/ws/realtime`.
- `JobManager` (`mlx_vc/jobs.py`) for in-memory async batch coordination
  with single-GPU semaphore serialisation.
- OpenVoice singleton (`mlx_vc/realtime.py`) for sub-300 ms WS latency.

### Backends shipped
- **Seed-VC** (Plachtaa/seed-vc): Whisper-small content + DiT (CFM) +
  BigVGAN, MPS + fp16 autocast.
- **OpenVoice V2** (myshell-ai): VITS tone-color converter, MPS.
- **kNN-VC** (bshall/knn-vc): WavLM frame retrieval + HiFi-GAN, CPU
  (upstream fp64 weights and cross-device child modules block MPS).
- **FreeVC / FreeVC-s** (OlaWod/FreeVC): WavLM + VITS decoder, with /
  without speaker encoder, MPS.
- **MeanVC** (ASLP-lab/MeanVC): DiT + Mean Flow, 14M params, CPU
  (TorchScript Vocos hits MPS graph-fuser bug).
- **SpeechT5-VC** (microsoft/speecht5_vc): transformer seq2seq, MPS.
- **RVC** (Acelogic MLX port): retrieval-based with baked speaker, MLX,
  runs in its own Python 3.10 venv.

### TTS-clone backends (text path, not true VC)
- **Chatterbox** (Resemble AI via mlx-audio).
- **Pocket-TTS** (Kyutai, ~235 MB) via the generic `tts_clone_infer.py`
  subprocess runner.

### Quality + tooling
- `scripts/evaluate_quality.py`: UTMOS + SECS + WER on each model's
  output (no-reference where possible).
- `scripts/prepare_reference.py`: Demucs vocal separation + 60 s / 3 min
  reference extraction.
- `benchmarks/bench_models.py`: end-to-end speed regression test.
- 46 unit tests covering audio I/O, server endpoints, job lifecycle,
  CLI registry consistency, reference resolution.

### Workarounds documented
- BigVGAN `_from_pretrained` signature break under huggingface_hub ≥ 1.0.
- s3prl `torchaudio.set_audio_backend` / `sox_effects` removal stubs.
- torchcodec broken `.dylib` stub with synthetic `__spec__`.
- TorchScript graph-fuser failures on pre-saved `.pt` files under MPS.
- TorchScript JIT alias-analysis bug with chunked encoder calls.
- `inference_mode` × `autocast` × autograd interaction in Seed-VC vocoder.

[Unreleased]: https://github.com/feiyuehchen/mlx-vc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/feiyuehchen/mlx-vc/releases/tag/v0.1.0
