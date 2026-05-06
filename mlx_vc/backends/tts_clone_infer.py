#!/usr/bin/env python3
"""Generic mlx-audio TTS + voice-cloning backend (subprocess).

Runs any mlx-audio TTS model that accepts (text, ref_audio) via `load_model`.
Each call spawns a fresh Python interpreter — weights are freed when the
subprocess exits, so stacking multiple TTS-clone models in one benchmark
job doesn't leak memory or OOM the host.

Args JSON:
  {
    "source": "<src wav>",
    "reference": "<ref wav>",
    "output": "<out wav>",
    "hf_model": "<e.g. mlx-community/Pocket-TTS>",
    "max_ref_seconds": 10.0,  (optional, trim reference)
    "needs_ref_text": false,  (optional, Sesame-style)
    "text": "<optional, else Whisper-transcribe source>"
  }
"""

import argparse
import json
import os
import sys
import time


def _whisper_transcribe(path: str) -> str:
    import whisper

    m = whisper.load_model("small")
    return m.transcribe(path, fp16=False, verbose=False).get("text", "").strip()


def _trim_reference(path: str, max_seconds: float, out_path: str) -> str:
    """Trim `path` to first `max_seconds` if longer. Return effective path."""
    import librosa
    import soundfile as sf

    try:
        dur = librosa.get_duration(path=path)
    except Exception:
        return path
    if dur <= max_seconds:
        return path
    wav, sr = librosa.load(path, sr=None, mono=True)
    n = int(sr * max_seconds)
    sf.write(out_path, wav[:n], sr)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True)
    parsed = parser.parse_args()
    args = json.loads(parsed.args)

    source = args["source"]
    reference = args["reference"]
    output = args["output"]
    hf_model = args["hf_model"]
    text = args.get("text") or ""
    max_ref_seconds = args.get("max_ref_seconds")
    needs_ref_text = bool(args.get("needs_ref_text", False))

    # Transcribe source if no text provided
    if not text.strip() or text == "Welcome to the demo.":
        print("Transcribing source audio with Whisper...")
        text = _whisper_transcribe(source)
        if not text:
            text = "Welcome to the demo."
    print(f"Text ({len(text.split())} words): {text[:80]}...")

    # Trim reference if requested
    effective_ref = reference
    if max_ref_seconds:
        tmp_ref = os.path.join(
            os.path.dirname(os.path.abspath(output)),
            f"_ref_{int(max_ref_seconds)}s.wav",
        )
        effective_ref = _trim_reference(reference, max_ref_seconds, tmp_ref)

    # ref_text for Sesame-style models
    ref_text = None
    if needs_ref_text:
        print("Transcribing reference audio for ref_text...")
        ref_text = _whisper_transcribe(effective_ref)

    print(f"Loading mlx-audio model: {hf_model}")
    import inspect

    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load_model

    model = load_model(hf_model)
    sr = getattr(model, "sample_rate", 24000)

    # Build kwargs only with keys the model's generate() accepts
    sig = inspect.signature(model.generate)
    accepts_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    candidate = {
        "text": text,
        "ref_audio": effective_ref,
        "verbose": False,
    }
    if ref_text is not None:
        candidate["ref_text"] = ref_text
    if accepts_varkw:
        call_kwargs = candidate
    else:
        call_kwargs = {k: v for k, v in candidate.items() if k in sig.parameters}

    print(f"Synthesizing with {hf_model}...")
    t0 = time.time()
    chunks = []
    for result in model.generate(**call_kwargs):
        a = result.audio
        if hasattr(a, "tolist"):
            a = np.array(a, dtype=np.float32)
        chunks.append(a.flatten())

    if not chunks:
        print("ERROR: no audio produced", file=sys.stderr)
        sys.exit(1)

    audio = np.concatenate(chunks)
    elapsed = time.time() - t0
    print(f"Generated {len(audio)/sr:.2f}s audio in {elapsed:.2f}s")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    sf.write(output, audio, sr)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
