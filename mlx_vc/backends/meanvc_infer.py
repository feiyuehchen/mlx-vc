#!/usr/bin/env python3
"""MeanVC backend inference script.

Uses the MeanVC reference repo for lightweight streaming VC.
Requires: clone https://github.com/ASLP-lab/MeanVC.git to ../meanvc-ref/
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
    steps = args.get("steps", 1)
    chunk_size = args.get("chunk_size", 40)

    # Setup MeanVC path
    meanvc_ref = os.environ.get(
        "MEANVC_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "meanvc-ref"),
    )
    meanvc_ref = os.path.abspath(meanvc_ref)
    if not os.path.exists(meanvc_ref):
        print(f"ERROR: MeanVC not found at {meanvc_ref}", file=sys.stderr)
        print("Clone it: git clone https://github.com/ASLP-lab/MeanVC.git meanvc-ref", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, meanvc_ref)
    os.chdir(meanvc_ref)

    import torch
    import soundfile as sf

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Download checkpoints if needed
    ckpt_dir = os.path.join(meanvc_ref, "checkpoints")
    if not os.path.exists(ckpt_dir) or not os.listdir(ckpt_dir):
        print("Downloading MeanVC checkpoints...")
        os.system(f"cd {meanvc_ref} && python download_ckpt.py")

    # Load models
    print("Loading MeanVC models...")
    from src.infer.infer_ref import (
        extract_features_from_audio,
        inference,
        MelSpectrogramFeatures,
    )
    from src.infer.dit_kvcache import DiT
    from src.model.utils import load_checkpoint
    from src.runtime.speaker_verification.verification import init_model as init_sv_model
    from vocos import Vocos

    # Load ASR model (WeNet-based)
    try:
        import wenet
        asr_model = wenet.load_model("chinese")
        asr_model = asr_model.to(device).eval()
    except Exception:
        # Fallback: try loading from checkpoint
        print("Loading ASR encoder...")
        from funasr import AutoModel
        asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4")

    # Load speaker verification model
    print("Loading speaker verification model...")
    sv_model = init_sv_model(device=device)

    # Load DiT model
    print("Loading DiT model...")
    config_path = os.path.join(meanvc_ref, "default_config.yaml")
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dit_model = DiT(
        dim=config.get("dim", 512),
        depth=config.get("depth", 6),
        heads=config.get("heads", 8),
        ff_mult=config.get("ff_mult", 4),
        mel_dim=80,
        text_num_embeds=256,
        cond_drop_prob=0.0,
    ).to(device).eval()

    ckpt_path = os.path.join(ckpt_dir, "meanvc.pt")
    if os.path.exists(ckpt_path):
        dit_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Load Vocos vocoder
    print("Loading Vocos vocoder...")
    vocos = Vocos.from_pretrained("patriotyk/vocos-mel-hifigan-compat-44100hz")
    vocos = vocos.to(device).eval()

    # Mel extractor
    mel_extractor = MelSpectrogramFeatures().to(device)

    # Extract features
    print("Extracting features...")
    bn, spk_emb, prompt_mel = extract_features_from_audio(
        source, reference, asr_model, sv_model, mel_extractor, device
    )

    # Run inference
    print(f"Running inference (steps={steps}, chunk_size={chunk_size})...")
    t0 = time.time()
    mel, audio_out, infer_time = inference(
        dit_model, vocos, bn, spk_emb, prompt_mel, chunk_size, steps, device
    )
    elapsed = time.time() - t0

    result = audio_out.squeeze().cpu().numpy()
    sr = 16000
    print(f"Generated {len(result)/sr:.2f}s audio in {elapsed:.2f}s")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, result, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
