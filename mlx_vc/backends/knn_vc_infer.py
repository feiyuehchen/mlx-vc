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
    # Optional list of additional reference files — concatenated into the
    # kNN matching pool for denser target-speaker coverage.  kNN-VC quality
    # scales directly with pool size; ideally ≥ 3 min of clean reference.
    extra_refs = args.get("extra_references", [])

    import torch

    # bshall/knn-vc upstream code:
    #   - `torch.tensor(np.float64_array, device=device)` fails on MPS
    #     (MPS rejects float64).
    #   - Internal HiFi-GAN vocoder isn't moved to the device passed in,
    #     so even after fixing the float64 issue, conv1d hits a
    #     "MPSFloatType vs torch.FloatTensor" device-mismatch error.
    # CPU is faster to set up and the WavLM forward+kNN+vocoder is fast
    # enough on CPU for our 10s-source benchmark. Set KNN_VC_DEVICE=mps
    # to override and patch upstream if motivated.
    forced = os.environ.get("KNN_VC_DEVICE", "cpu").strip().lower()
    if forced == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif forced == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
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

    all_refs = [reference] + list(extra_refs)
    print(f"Building reference matching set from {len(all_refs)} file(s)...")
    matching_set = knn_vc.get_matching_set(all_refs)
    print(f"Matching pool: {matching_set.shape}")

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
