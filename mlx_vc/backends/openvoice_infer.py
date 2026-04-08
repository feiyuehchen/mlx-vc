#!/usr/bin/env python3
"""OpenVoice V2 backend inference script.

Uses the OpenVoice module from seed-vc-ref for tone color conversion.
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
    tau = args.get("tau", 0.3)

    # Setup seed-vc-ref path
    seed_vc_ref = os.environ.get(
        "SEED_VC_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "seed-vc-ref"),
    )
    seed_vc_ref = os.path.abspath(seed_vc_ref)
    if not os.path.exists(seed_vc_ref):
        print(f"ERROR: seed-vc-ref not found at {seed_vc_ref}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, seed_vc_ref)

    import torch
    import librosa
    import numpy as np
    import soundfile as sf

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Download checkpoint if needed
    ckpt_dir = os.path.join(seed_vc_ref, "modules", "openvoice", "checkpoints_v2", "converter")
    config_path = os.path.join(ckpt_dir, "config.json")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")

    if not os.path.exists(ckpt_path):
        print("Downloading OpenVoice V2 checkpoint...")
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth")
        os.makedirs(ckpt_dir, exist_ok=True)
        import shutil
        shutil.copy2(downloaded, ckpt_path)

    if not os.path.exists(config_path):
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/config.json")
        os.makedirs(ckpt_dir, exist_ok=True)
        import shutil
        shutil.copy2(downloaded, config_path)

    # Load model
    print("Loading OpenVoice V2 converter...")
    from modules.openvoice.api import ToneColorConverter

    converter = ToneColorConverter(config_path, device=device)
    converter.load_ckpt(ckpt_path)
    sr = converter.hps.data.sampling_rate  # 22050

    # Load audio
    src_audio, _ = librosa.load(source, sr=sr)
    ref_audio, _ = librosa.load(reference, sr=sr)

    src_tensor = torch.FloatTensor(src_audio).unsqueeze(0).to(device)
    ref_tensor = torch.FloatTensor(ref_audio).unsqueeze(0).to(device)

    src_lengths = torch.LongTensor([src_tensor.size(1)]).to(device)
    ref_lengths = torch.LongTensor([ref_tensor.size(1)]).to(device)

    # Extract speaker embeddings
    print("Extracting speaker embeddings...")
    with torch.no_grad():
        src_se = converter.extract_se([src_tensor.squeeze(0)], [src_lengths.item()])
        tgt_se = converter.extract_se([ref_tensor.squeeze(0)], [ref_lengths.item()])

    # Convert
    print(f"Converting (tau={tau})...")
    t0 = time.time()
    with torch.no_grad():
        converted = converter.convert(src_tensor, src_lengths, src_se, tgt_se, tau=tau)

    result = converted.squeeze().cpu().numpy()
    elapsed = time.time() - t0
    print(f"Generated {len(result)/sr:.2f}s audio in {elapsed:.2f}s")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, result, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
