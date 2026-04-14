#!/usr/bin/env python3
"""MeanVC backend inference script (custom, bypasses s3prl).

The official MeanVC repo uses a WavLM-Large+ECAPA-TDNN speaker verification
model that depends on s3prl. s3prl pins old torchaudio APIs that no longer
exist in modern torchaudio (set_audio_backend was removed), so loading the
official SV model crashes immediately.

Workaround: substitute the speaker embedding with one extracted via
seed-vc-ref's CAMPPlus encoder (192-dim), then pad to MeanVC's expected
256-dim. The DiT, ASR encoder, and Vocos vocoder are unchanged. Output
quality is slightly worse than the official pipeline (the speaker
embedding distribution differs from training), but the model runs and
produces recognizable VC output for the demo.

Required setup:
  cd ../meanvc-ref && python download_ckpt.py
  pip install x_transformers wandb accelerate jiwer  # for the package import chain
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
    chunk_size = args.get("chunk_size", 20)
    steps = args.get("steps", 2)

    # Locate the meanvc-ref repo
    meanvc_ref = os.environ.get(
        "MEANVC_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "meanvc-ref"),
    )
    meanvc_ref = os.path.abspath(meanvc_ref)
    if not os.path.exists(meanvc_ref):
        print(f"ERROR: MeanVC repo not found at {meanvc_ref}", file=sys.stderr)
        sys.exit(1)

    # Locate the seed-vc-ref repo (for CAMPPlus speaker encoder)
    seed_vc_ref = os.environ.get(
        "SEED_VC_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "seed-vc-ref"),
    )
    seed_vc_ref = os.path.abspath(seed_vc_ref)

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Step 1: Load CAMPPlus FIRST with seed-vc-ref on sys.path. ---
    # This must happen before any MeanVC imports because both repos use
    # the top-level name `modules` for incompatible things.
    print("Loading CAMPPlus speaker encoder (substitute for SV)...")
    sys.path.insert(0, seed_vc_ref)
    from modules.campplus.DTDNN import CAMPPlus  # noqa: E402

    from huggingface_hub import hf_hub_download

    campplus_ckpt = hf_hub_download("funasr/campplus", "campplus_cn_common.bin")
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
    campplus = campplus.to(device).eval()

    # --- Step 2: Now swap to MeanVC's path layout. ---
    # Remove seed-vc-ref's `modules` from sys.modules so MeanVC's bare
    # `from modules import ...` (which references src/infer/modules.py)
    # resolves correctly.
    for mod_name in list(sys.modules.keys()):
        if mod_name == "modules" or mod_name.startswith("modules."):
            del sys.modules[mod_name]
    sys.path.remove(seed_vc_ref)
    sys.path.insert(0, os.path.join(meanvc_ref, "src/infer"))
    sys.path.insert(0, meanvc_ref)
    os.chdir(meanvc_ref)

    print(f"Loading MeanVC (device={device})...")

    from src.infer.dit_kvcache import DiT
    from src.infer.infer_ref import (
        MelSpectrogramFeatures,
        extract_fbanks,
        inference,
    )
    from src.model.utils import load_checkpoint

    # --- Load DiT ---
    config_path = os.path.join(meanvc_ref, "src/config/config_200ms.json")
    ckpt_path = os.path.join(meanvc_ref, "src/ckpt/model_200ms.safetensors")
    vocoder_path = os.path.join(meanvc_ref, "src/ckpt/vocos.pt")
    asr_path = os.path.join(meanvc_ref, "src/ckpt/fastu2++.pt")

    with open(config_path) as f:
        model_config = json.load(f)

    dit_model = DiT(**model_config["model"]).to(device)
    dit_model = load_checkpoint(dit_model, ckpt_path, device=device, use_ema=False)
    dit_model = dit_model.float().eval()

    vocos = torch.jit.load(vocoder_path).to(device)
    asr_model = torch.jit.load(asr_path).to(device)
    mel_extractor = MelSpectrogramFeatures(
        sample_rate=16000, n_fft=1024, win_size=640, hop_length=160,
        n_mels=80, fmin=0, fmax=8000, center=True,
    ).to(device)

    # --- Extract features ---
    print(f"Processing source: {source}")
    print(f"Reference:        {reference}")

    # ASR content features (bottleneck) from source
    source_wav, _ = librosa.load(source, sr=16000)
    source_fbanks = extract_fbanks(source_wav, frame_shift=10).float().to(device)

    with torch.no_grad():
        offset = 0
        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7
        stride = subsampling * decoding_chunk_size
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        att_cache = torch.zeros((0, 0, 0, 0), device=device)
        cnn_cache = torch.zeros((0, 0, 0, 0), device=device)

        bn_chunks = []
        for i in range(0, source_fbanks.shape[1], stride):
            fbank_chunk = source_fbanks[:, i:i + decoding_window, :]
            if fbank_chunk.shape[1] < required_cache_size:
                pad_size = required_cache_size - fbank_chunk.shape[1]
                fbank_chunk = torch.nn.functional.pad(
                    fbank_chunk, (0, 0, 0, pad_size), mode="constant", value=0.0
                )
            encoder_output, att_cache, cnn_cache = asr_model.forward_encoder_chunk(
                fbank_chunk, offset, required_cache_size, att_cache, cnn_cache
            )
            offset += encoder_output.size(1)
            bn_chunks.append(encoder_output)

        bn = torch.cat(bn_chunks, dim=1)
        bn = bn.transpose(1, 2)
        bn = torch.nn.functional.interpolate(
            bn, size=int(bn.shape[2] * 4), mode="linear", align_corners=True
        )
        bn = bn.transpose(1, 2)

    # Speaker embedding from reference (CAMPPlus 192-d -> pad to 256-d)
    ref_wav, _ = librosa.load(reference, sr=16000)
    ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = kaldi.fbank(
            ref_wav_tensor, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        spk_192 = campplus(feat.unsqueeze(0))  # [1, 192]
        spk_emb = torch.nn.functional.pad(spk_192, (0, 256 - 192))  # [1, 256]

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
    sr = 16000
    print(f"Generated {len(result)/sr:.2f}s audio in {elapsed:.2f}s "
          f"(RTF {infer_time / (len(result)/sr):.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, result, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
