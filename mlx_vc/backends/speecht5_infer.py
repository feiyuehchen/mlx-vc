#!/usr/bin/env python3
"""SpeechT5 VC backend (Microsoft, HF transformers).

SpeechT5 is a unified seq2seq transformer trained on TTS/STT/VC.  The VC
checkpoint takes source audio + a speaker embedding (x-vector) from a
reference and generates converted mel + waveform via HiFi-GAN.

Size: ~620MB main model + ~50MB HiFi-GAN + ~20MB x-vector → ~700MB total.
Fully MPS-compatible via transformers.
English-trained (CMU-ARCTIC).
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

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import (
        SpeechT5ForSpeechToSpeech,
        SpeechT5Processor,
        SpeechT5HifiGan,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Loading SpeechT5 VC...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    # Speaker embedding: use SpeechBrain's ECAPA via x-vector from reference.
    # The SpeechT5 VC docs use a fixed x-vector from CMU-ARCTIC; to get
    # arbitrary-speaker conditioning we extract an x-vector using
    # speechbrain ECAPA-TDNN-xvector.
    print("Extracting x-vector from reference...")
    # Stub torchcodec (broken .dylib on this Mac)
    import types as _types
    for _m in ("torchcodec", "torchcodec.decoders", "torchcodec.decoders._core"):
        if _m not in sys.modules:
            sm = _types.ModuleType(_m)
            sm.__spec__ = _types.SimpleNamespace(
                name=_m, loader=None, submodule_search_locations=[]
            )
            sys.modules[_m] = sm

    from speechbrain.inference.classifiers import EncoderClassifier

    spkrec = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=os.path.expanduser("~/.cache/sb_xvect"),
    )
    spkrec.eval()

    ref_wav, _ = librosa.load(reference, sr=16000, mono=True)
    ref_t = torch.from_numpy(ref_wav).unsqueeze(0)
    with torch.no_grad():
        xvec = spkrec.encode_batch(ref_t)  # [1, 1, 512]
        xvec = torch.nn.functional.normalize(xvec, dim=2)
    speaker_embedding = xvec.squeeze(0).to(device)  # [1, 512]

    print(f"Loading source audio: {source}")
    src_wav, _ = librosa.load(source, sr=16000, mono=True)

    inputs = processor(audio=src_wav, sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    print("Converting...")
    t0 = time.time()
    with torch.no_grad():
        speech = model.generate_speech(
            input_values, speaker_embedding, vocoder=vocoder
        )
    elapsed = time.time() - t0

    audio = speech.float().cpu().numpy()
    sr = 16000
    print(
        f"Generated {len(audio)/sr:.2f}s audio in {elapsed:.2f}s "
        f"(RTF {elapsed/(len(audio)/sr):.3f})"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, audio, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
