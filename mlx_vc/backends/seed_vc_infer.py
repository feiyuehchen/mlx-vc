#!/usr/bin/env python3
"""Seed-VC backend inference script.

Runs in its own virtual environment with Seed-VC dependencies.
Communicates via command-line args (JSON) and audio files.

Usage:
    python seed_vc_infer.py --args '{"source": "src.wav", "reference": "ref.wav", "output": "out.wav"}'
"""

import argparse
import json
import os
import sys
import time


def _patch_bigvgan(seed_vc_ref):
    """Patch BigVGAN._from_pretrained for newer huggingface_hub.

    New huggingface_hub (>=1.0) removed 'proxies' and 'resume_download'
    from the parent _from_pretrained signature, but Seed-VC's BigVGAN
    override still requires them as keyword-only args. We patch by
    providing defaults for the missing params.
    """
    try:
        sys.path.insert(0, seed_vc_ref)
        from modules.bigvgan import bigvgan

        original_fp = bigvgan.BigVGAN._from_pretrained.__func__

        @classmethod
        def patched_fp(cls, *, proxies=None, resume_download=False, **kwargs):
            return original_fp(
                cls, proxies=proxies, resume_download=resume_download, **kwargs
            )

        bigvgan.BigVGAN._from_pretrained = patched_fp
    except Exception as e:
        print(f"Warning: BigVGAN patch failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True, help="JSON args")
    parsed = parser.parse_args()
    args = json.loads(parsed.args)

    source = args["source"]
    reference = args["reference"]
    output = args["output"]
    # Upstream default is 10 (fast), recommended 50-100 for best quality.
    # 50 is the "best-quality" sweet spot from the upstream demo examples.
    diffusion_steps = args.get("diffusion_steps", 50)
    cfg_rate = args.get("inference_cfg_rate", 0.7)
    length_adjust = args.get("length_adjust", 1.0)
    f0_condition = args.get("f0_condition", False)
    # Upstream default enables fp16 autocast on the CFM step.
    # MPS supports fp16 well for Seed-VC per upstream testing.
    fp16 = args.get("fp16", True)

    sr = 22050 if not f0_condition else 44100

    # Add seed-vc-ref to path
    seed_vc_ref = os.environ.get(
        "SEED_VC_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "seed-vc-ref"),
    )
    seed_vc_ref = os.path.abspath(seed_vc_ref)
    if not os.path.exists(seed_vc_ref):
        print(f"ERROR: Seed-VC repo not found at {seed_vc_ref}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, seed_vc_ref)
    os.environ.setdefault(
        "HF_HUB_CACHE", os.path.join(seed_vc_ref, "checkpoints", "hf_cache")
    )

    # Patch BigVGAN for newer huggingface_hub compatibility
    _patch_bigvgan(seed_vc_ref)

    # Load models
    print(f"Loading Seed-VC models...")
    from types import SimpleNamespace

    model_args = SimpleNamespace(
        checkpoint=None,
        config=None,
        f0_condition=f0_condition,
        fp16=fp16,
    )

    import librosa
    import numpy as np
    import torch
    import torchaudio
    from inference import crossfade, load_models

    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = (
        load_models(model_args)
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(reference, sr=sr)[0]
    ref_audio = ref_audio[: sr * 25]  # max 25s reference

    source_tensor = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_tensor = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    t0 = time.time()

    # Content encoding
    source_16k = torchaudio.functional.resample(source_tensor, sr, 16000)
    ref_16k = torchaudio.functional.resample(ref_tensor, sr, 16000)

    if source_16k.size(-1) <= 16000 * 30:
        S_source = semantic_fn(source_16k)
    else:
        # Chunk encoding for long audio
        overlap = 5
        chunks = []
        buf = None
        pos = 0
        while pos < source_16k.size(-1):
            if buf is None:
                chunk = source_16k[:, pos : pos + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buf, source_16k[:, pos : pos + 16000 * (30 - overlap)]], dim=-1
                )
            S = semantic_fn(chunk)
            chunks.append(S if pos == 0 else S[:, 50 * overlap :])
            buf = chunk[:, -16000 * overlap :]
            pos += 30 * 16000 if pos == 0 else chunk.size(-1) - 16000 * overlap
        S_source = torch.cat(chunks, dim=1)

    S_ref = semantic_fn(ref_16k)

    source_mel = mel_fn(source_tensor.float())
    ref_mel = mel_fn(ref_tensor.float())

    # Speaker style
    feat = torchaudio.compliance.kaldi.fbank(
        ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    style = campplus_model(feat.unsqueeze(0))

    # Length regulation
    target_lengths = torch.LongTensor([int(source_mel.size(2) * length_adjust)]).to(
        device
    )
    target2_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)

    cond, _, _, _, _ = model.length_regulator(
        S_source, ylens=target_lengths, n_quantizers=3
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ref, ylens=target2_lengths, n_quantizers=3
    )

    # CFM inference with chunking
    max_source_window = max_context_window - ref_mel.size(2)
    processed_frames = 0
    wave_chunks = []

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if fp16 else torch.float32,
            ),
        ):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                ref_mel,
                style,
                None,
                diffusion_steps,
                inference_cfg_rate=cfg_rate,
            )
            vc_target = vc_target[:, :, ref_mel.size(-1) :]
            # Vocoder must be inside inference_mode too — otherwise passing
            # the inference-mode tensor into vocoder's autograd-tracked
            # convs raises "Inference tensors cannot be saved for backward".
            # Force output to fp32 (autocast may have left it as fp16, which
            # soundfile can't write).
            vc_wave = vocoder_fn(vc_target.float()).float().squeeze()[None, :]

        if processed_frames == 0:
            if is_last:
                wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last:
            wave_chunks.append(
                crossfade(
                    previous.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
                )
            )
            break
        else:
            wave_chunks.append(
                crossfade(
                    previous.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len,
                )
            )
            previous = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    result = np.concatenate(wave_chunks)
    elapsed = time.time() - t0
    print(
        f"Generated {len(result)/sr:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed / (len(result)/sr):.3f})"
    )

    # Save output
    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, result, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
