#!/usr/bin/env python3
"""MeanVC backend inference script.

Uses the official WavLM-Large + ECAPA-TDNN speaker encoder.  Surrogate
encoders (CAMPPlus zero-pad, SpeechBrain ECAPA random-projection) all
collapse the DiT output to noise — only the official 256-dim head produces
intelligible speech.

Required setup:
  1. Clone repo:
       git clone https://github.com/ASLP-lab/MeanVC ../meanvc-ref

  2. Download main checkpoints:
       cd ../meanvc-ref && python download_ckpt.py

  3. Manually download speaker verification checkpoint from Google Drive
     (NOT on HF):
       gdown 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP \\
             -O ../meanvc-ref/src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth

  4. Install package import chain:
       pip install x_transformers wandb accelerate jiwer

Workarounds in this script:
  - s3prl deprecated torchaudio API (set_audio_backend / sox_effects) → stubs
  - torchcodec broken .dylib → stub with synthetic __spec__
  - Pre-saved TorchScript Vocos+ASR hit "Unknown device for graph fuser" on MPS
    → device forced to CPU (set MEANVC_DEVICE=mps to override, will crash)
  - asr.forward_encoder_chunk in a loop hits a JIT alias-analysis bug under
    `python script.py` (but not `python -c`) → use full-sequence encoder
  - Model load order is sensitive: load the ASR + extract bn features first,
    then SpeechBrain x-vector / DiT / Vocos.

Notes on quality:
  - MeanVC was trained on Chinese.  Self-conversion on the upstream Chinese
    `example/test.wav` gives WER ≈ 0.13.  English audio gives WER ≈ 0.88 —
    out-of-distribution, not a bug.
  - Output is post-processed with NaN scrub + DC removal + soft peak limit
    to defend against random-init flow-matching artifacts.
"""

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True)
    parsed = parser.parse_args()
    args = json.loads(parsed.args)

    source = args["source"]
    reference = args["reference"]
    output = args["output"]
    chunk_size = int(args.get("chunk_size", os.environ.get("MEANVC_CHUNK_SIZE", 20)))
    steps = int(args.get("steps", os.environ.get("MEANVC_STEPS", 4)))

    # Locate the meanvc-ref repo
    meanvc_ref = os.environ.get(
        "MEANVC_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "meanvc-ref"),
    )
    meanvc_ref = os.path.abspath(meanvc_ref)
    if not os.path.exists(meanvc_ref):
        print(f"ERROR: MeanVC repo not found at {meanvc_ref}", file=sys.stderr)
        sys.exit(1)

    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    # Default to CPU for stability.
    # On some macOS/PyTorch builds, enabling MPS here can trigger a TorchScript
    # alias-analysis internal assert in forward_encoder_chunk().
    # Allow explicit override via MEANVC_DEVICE if needed.
    requested_device = os.environ.get("MEANVC_DEVICE", "cpu").strip().lower()
    if requested_device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Step 1: Load MeanVC path and the TorchScript ASR FIRST.
    # The ASR (fastu2++.pt) must be loaded before any other TorchScript models
    # (including SpeechBrain's internals) to avoid a PyTorch alias-analysis bug
    # that fires when a JIT model is loaded after MPS/SpeechBrain context exists.
    sys.path.insert(0, os.path.join(meanvc_ref, "src/infer"))
    sys.path.insert(0, meanvc_ref)
    os.chdir(meanvc_ref)

    print(f"Loading MeanVC (device={device})...")

    # Import only the ASR feature helper first to keep TorchScript state clean.
    from src.infer.infer_ref import extract_fbanks

    config_path = os.path.join(meanvc_ref, "src/config/config_200ms.json")
    ckpt_path = os.path.join(meanvc_ref, "src/ckpt/model_200ms.safetensors")
    vocoder_path = os.path.join(meanvc_ref, "src/ckpt/vocos.pt")
    asr_path = os.path.join(meanvc_ref, "src/ckpt/fastu2++.pt")

    with open(config_path) as f:
        model_config = json.load(f)

    # Work around a TorchScript alias-analysis crash seen on some macOS builds
    # when repeatedly calling forward_encoder_chunk() in a loop.
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    # TorchScript ASR runs on CPU only (MPS triggers graph fuser bug).
    asr_device = torch.device("cpu")
    asr_model = torch.jit.load(asr_path, map_location="cpu")

    # --- Step 2: Extract ASR bottleneck features BEFORE loading SpeechBrain.
    # Loading additional TorchScript modules (via SpeechBrain) can re-trigger
    # the alias-analysis bug in forward_encoder_chunk on some builds.
    print(f"Processing source: {source}")
    source_wav, _ = librosa.load(source, sr=16000)
    source_fbanks = extract_fbanks(source_wav, frame_shift=10).float().to(asr_device)

    with torch.no_grad():
        # Official code uses forward_encoder_chunk in a loop (streaming).
        # That path trips a TorchScript alias-analysis bug on our setup
        # (PyTorch 2.11 + macOS) when the script is executed as a file
        # rather than `python -c` — the INTERNAL ASSERT fires on the
        # second-or-later chunk.  `asr_model.encoder(x, xs_lens, -1)` is
        # the same architecture invoked in non-streaming mode and produces
        # equivalent bottleneck features for short clips.
        xs_lens = torch.LongTensor([source_fbanks.size(1)]).to(asr_device)
        bn, _ = asr_model.encoder(source_fbanks, xs_lens, -1)
        bn = bn.transpose(1, 2)
        bn = torch.nn.functional.interpolate(
            bn, size=int(bn.shape[2] * 4), mode="linear", align_corners=True
        )
        bn = bn.transpose(1, 2)
        bn = bn.to(device)  # move from asr_device (CPU) to main device

    # Import MeanVC model/vocoder stack only after ASR extraction.
    from src.infer.dit_kvcache import DiT
    from src.infer.infer_ref import MelSpectrogramFeatures, inference
    from src.model.utils import load_checkpoint

    # --- Step 3: Official WavLM-Large + ECAPA-TDNN speaker encoder.
    #
    # Uses the paper's `init_sv_model('wavlm_large', checkpoint)` path.  The
    # checkpoint `wavlm_large_finetune.pth` must be in
    # `src/runtime/speaker_verification/ckpt/`.  It's not shipped via HF —
    # download from Google Drive
    # https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view
    #
    # s3prl (which provides the WavLM feature extractor) imports APIs
    # removed in modern torchaudio; we stub them below before torch.hub
    # loads s3prl.
    print("Loading official WavLM-Large + ECAPA-TDNN speaker encoder...")

    # Stub torchaudio APIs removed in recent releases (s3prl still references them)
    import types
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    if not hasattr(torchaudio, "sox_effects"):
        sox = types.ModuleType("torchaudio.sox_effects")
        sox.apply_effects_tensor = lambda *a, **k: (None, None)
        sox.apply_effects_file = lambda *a, **k: (None, None)
        sys.modules["torchaudio.sox_effects"] = sox
        torchaudio.sox_effects = sox
    # torchcodec: give sys.modules entries proper __spec__ so s3prl's
    # `importlib.util.find_spec("torchcodec")` returns truthy without crashing.
    for _tc in ("torchcodec", "torchcodec.decoders", "torchcodec.decoders._core"):
        if _tc not in sys.modules:
            _tc_mod = types.ModuleType(_tc)
            _tc_mod.__spec__ = types.SimpleNamespace(
                name=_tc, loader=None, submodule_search_locations=[]
            )
            sys.modules[_tc] = _tc_mod

    sv_ckpt = os.path.join(
        meanvc_ref, "src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth"
    )
    if not os.path.exists(sv_ckpt):
        print(
            f"ERROR: Speaker verification checkpoint missing: {sv_ckpt}\n"
            f"Download from https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.runtime.speaker_verification.verification import init_model as init_sv_model

    sv_model = init_sv_model("wavlm_large", sv_ckpt)
    sv_model = sv_model.to(device).eval()

    def _get_spk_emb(ref_wav_np: "np.ndarray") -> torch.Tensor:
        """Extract 256-dim WavLM-Large + ECAPA-TDNN speaker embedding."""
        # Model expects a single waveform tensor, shape [1, samples], 16 kHz.
        wav_t = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(device)
        with torch.no_grad():
            emb = sv_model(wav_t)  # [1, 256]
        return emb

    dit_model = DiT(**model_config["model"]).to(device)
    dit_model = load_checkpoint(dit_model, ckpt_path, device=device, use_ema=False)
    dit_model = dit_model.float().eval()

    vocos = torch.jit.load(vocoder_path).to(device)
    mel_extractor = MelSpectrogramFeatures(
        sample_rate=16000, n_fft=1024, win_size=640, hop_length=160,
        n_mels=80, fmin=0, fmax=8000, center=True,
    ).to(device)

    # Speaker embedding from reference (SpeechBrain ECAPA 192-d -> project to 256-d)
    print(f"Reference:        {reference}")
    ref_wav, _ = librosa.load(reference, sr=16000)
    ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(device)

    spk_emb = _get_spk_emb(ref_wav)  # [1, 256]

    with torch.no_grad():
        prompt_mel = mel_extractor(ref_wav_tensor)  # [1, 80, T]
        prompt_mel = prompt_mel.transpose(1, 2)  # [1, T, 80]

    # --- Run inference ---
    print(f"Running inference (chunk_size={chunk_size}, steps={steps})...")
    t0 = time.time()
    mel, wav, infer_time = inference(
        dit_model, vocos, bn, spk_emb, prompt_mel, chunk_size, steps, device
    )
    elapsed = time.time() - t0

    result = wav.squeeze().cpu().numpy().astype(np.float32)
    # Basic cleanup to reduce harsh artifacts in low-step outputs.
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    if result.size:
        result = result - float(np.mean(result))  # remove DC bias
        peak = float(np.max(np.abs(result)))
        if peak > 0.98:
            result = result * (0.95 / peak)  # avoid hard clipping
    sr = 16000
    print(f"Generated {len(result)/sr:.2f}s audio in {elapsed:.2f}s "
          f"(RTF {infer_time / (len(result)/sr):.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, result, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
