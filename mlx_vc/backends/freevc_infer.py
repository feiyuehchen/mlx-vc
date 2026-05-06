#!/usr/bin/env python3
"""FreeVC backend inference script (OlaWod/FreeVC).

FreeVC is a one-shot WavLM-based VC using a VITS decoder with an
information-bottleneck content extractor.  Clean English-friendly model,
different architecture from Seed-VC (no diffusion, single-pass synthesis).

Setup:
    git clone https://github.com/OlaWod/FreeVC.git ../freevc-ref
    # Weights from HuggingFace mirror:
    #   huggingface_hub.hf_hub_download('Pranjal4554/FreeVC', 'freevc.pth')
    #   huggingface_hub.hf_hub_download('Pranjal4554/FreeVC', 'WavLM-Large.pt')
    #   huggingface_hub.hf_hub_download('jn-jairo/freevc', 'speaker_encoder.pt')
    # Place in freevc-ref/checkpoints/, wavlm/, speaker_encoder/ckpt/
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
    # variant: "freevc" (default, uses speaker encoder) or "freevc-s"
    # (uses target mel-spectrogram directly, no speaker encoder — different
    # architecture variant from the same paper).
    variant = args.get("variant", "freevc")

    freevc_ref = os.environ.get(
        "FREEVC_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "freevc-ref"),
    )
    freevc_ref = os.path.abspath(freevc_ref)
    if not os.path.exists(freevc_ref):
        print(f"ERROR: freevc-ref not found at {freevc_ref}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, freevc_ref)
    os.chdir(freevc_ref)

    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    # FreeVC uses CUDA by default; override to MPS/CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Monkeypatch torch's "cuda" references by aliasing — the convert.py uses
    # .cuda() which only works on CUDA. We avoid running convert.py; we build
    # the pipeline inline using the public API and .to(device).

    import utils
    from models import SynthesizerTrn
    from mel_processing import mel_spectrogram_torch
    from speaker_encoder.voice_encoder import SpeakerEncoder

    if variant == "freevc-s":
        hp_file = os.path.join(freevc_ref, "configs/freevc-s.json")
        ckpt_file = os.path.join(freevc_ref, "checkpoints/freevc-s.pth")
    else:
        hp_file = os.path.join(freevc_ref, "configs/freevc.json")
        ckpt_file = os.path.join(freevc_ref, "checkpoints/freevc.pth")

    if not os.path.exists(ckpt_file):
        print(f"ERROR: FreeVC checkpoint missing: {ckpt_file}", file=sys.stderr)
        sys.exit(1)

    print("Loading FreeVC SynthesizerTrn...")
    hps = utils.get_hparams_from_file(hp_file)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_g.eval()

    utils.load_checkpoint(ckpt_file, net_g, None, True)

    # Content model (WavLM)
    print("Loading WavLM content model...")
    # utils.get_cmodel hardcodes device=0 (cuda); replicate manually
    from wavlm import WavLM, WavLMConfig

    wavlm_ckpt = os.path.join(freevc_ref, "wavlm/WavLM-Large.pt")
    if not os.path.exists(wavlm_ckpt):
        print(f"ERROR: WavLM checkpoint missing: {wavlm_ckpt}", file=sys.stderr)
        sys.exit(1)

    wavlm_state = torch.load(wavlm_ckpt, map_location="cpu", weights_only=False)
    cmodel = WavLM(WavLMConfig(wavlm_state["cfg"]))
    cmodel.load_state_dict(wavlm_state["model"])
    cmodel = cmodel.to(device).eval()

    # Speaker encoder (only needed if use_spk=True)
    use_spk = bool(hps.model.use_spk)
    smodel = None
    if use_spk:
        print("Loading speaker encoder...")
        # torch 2.11: SpeakerEncoder runs cleanly on MPS now (was forced
        # to CPU under older torch due to torch.hub interaction issues).
        smodel = SpeakerEncoder(
            os.path.join(freevc_ref, "speaker_encoder/ckpt/pretrained_bak_5805000.pt"),
            device=str(device),
        )

    # --- Load audio ---
    sr = hps.data.sampling_rate  # 16000
    wav_tgt, _ = librosa.load(reference, sr=sr)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

    wav_src, _ = librosa.load(source, sr=sr)
    wav_src_t = torch.from_numpy(wav_src).unsqueeze(0).to(device)

    print(f"Source len: {len(wav_src)/sr:.2f}s  Reference len: {len(wav_tgt)/sr:.2f}s")

    t0 = time.time()
    with torch.no_grad():
        # Content features from source via WavLM
        # utils.get_content uses layer-6 features with normalization
        c = get_content(cmodel, wav_src_t)

        if use_spk:
            # Speaker encoder on CPU (SpeakerEncoder's torch.hub internals
            # don't MPS-cleanly). Output is numpy.
            g_tgt = smodel.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
            audio = net_g.infer(c, g=g_tgt)
        else:
            wav_tgt_t = torch.from_numpy(wav_tgt).unsqueeze(0).to(device)
            mel_tgt = mel_spectrogram_torch(
                wav_tgt_t,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            audio = net_g.infer(c, mel=mel_tgt)

    audio = audio[0][0].data.cpu().float().numpy()
    elapsed = time.time() - t0
    print(f"Generated {len(audio)/sr:.2f}s audio in {elapsed:.2f}s (RTF {elapsed/(len(audio)/sr):.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, audio, sr)
    print(f"Saved to {output}")


def get_content(cmodel, y):
    """Extract WavLM content features — mirrors FreeVC's utils.get_content
    but respects whatever device cmodel lives on.
    """
    import torch
    with torch.no_grad():
        c = cmodel.extract_features(y)[0]
    c = c.transpose(1, 2)
    return c


if __name__ == "__main__":
    main()
