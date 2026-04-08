#!/usr/bin/env python3
"""kNN-VC backend inference script.

Uses torch.hub to load kNN-VC (auto-downloads WavLM + HiFi-GAN).
"""

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True)
    parsed = parser.parse_args()
    args = json.loads(parsed.args)

    source = args["source"]
    reference = args["reference"]
    output = args["output"]
    topk = args.get("topk", 4)
    prematched = args.get("prematched", True)

    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # kNN-VC on MPS can have issues with some ops, fallback to CPU
    if device.type == "mps":
        print("Note: kNN-VC on MPS may have issues, using CPU for stability")
        device = torch.device("cpu")

    print(f"Loading kNN-VC (device={device})...")
    knn_vc = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=prematched,
        trust_repo=True,
        pretrained=True,
        device=device,
    )

    # Monkey-patch torchaudio.load to use soundfile (avoids torchcodec/ffmpeg issues)
    import torchaudio
    _orig_load = torchaudio.load

    def _patched_load(path, **kwargs):
        import librosa
        y, sr_orig = librosa.load(str(path), sr=None, mono=True)
        y = y[None, :]  # [1, samples]
        return torch.FloatTensor(y), sr_orig

    torchaudio.load = _patched_load

    print("Extracting source features...")
    t0 = time.time()
    query_seq = knn_vc.get_features(source)

    print("Building reference matching set...")
    matching_set = knn_vc.get_matching_set([reference])

    print(f"Matching (topk={topk})...")
    out_wav = knn_vc.match(query_seq, matching_set, topk=topk)

    torchaudio.load = _orig_load  # restore

    elapsed = time.time() - t0
    result = out_wav.squeeze().cpu().numpy()
    sr = 16000
    print(f"Generated {len(result)/sr:.2f}s audio in {elapsed:.2f}s")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, result, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
