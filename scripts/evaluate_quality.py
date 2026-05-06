#!/usr/bin/env python3
"""Objective quality metrics for voice conversion outputs.

Three no-reference / lightweight-reference metrics:

  - UTMOS — torchaudio.pipelines.SQUIM_SUBJECTIVE
      No-reference naturalness MOS predictor, 1..5 scale.
      Higher = more natural / cleaner.

  - SECS — speaker embedding cosine similarity
      Cosine between SpeechBrain ECAPA-TDNN embeddings of output vs reference.
      0..1 scale.  Higher = output speaker closer to reference.

  - WER — Whisper re-transcription word error rate
      Whisper-small transcribes both source and output, then word-level
      Levenshtein distance / source-length.  Lower = content better preserved.
      Can exceed 1.0 when hypothesis contains insertions.

Caveat for TTS-clone outputs (text-path models): WER → 0 is trivially
achievable because the model goes through Whisper-text-Whisper roundtrip.
SECS / UTMOS remain meaningful comparisons.

Usage:
    python evaluate_quality.py \\
        --source path/to/source.wav \\
        --reference path/to/reference.wav \\
        --outputs path/to/openvoice.wav path/to/seed-vc.wav ... \\
        --json out_metrics.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Whisper + text normalization for WER
# ---------------------------------------------------------------------------

_WHISPER_MODEL = None


def _whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper

        _WHISPER_MODEL = whisper.load_model("small")
    return _WHISPER_MODEL


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _wer(ref: str, hyp: str) -> float:
    """Word-level Levenshtein distance / len(ref)."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return float("nan")
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[n][m] / n


def transcribe(path: str) -> str:
    model = _whisper_model()
    r = model.transcribe(path, fp16=False, verbose=False)
    return _normalize_text(r.get("text", ""))


# ---------------------------------------------------------------------------
# SECS via SpeechBrain ECAPA
# ---------------------------------------------------------------------------

_ECAPA = None


def _ecapa():
    global _ECAPA
    if _ECAPA is None:
        # torchcodec stub: SpeechBrain's torchaudio integration may try to
        # `find_spec("torchcodec")` even when not invoking it.  Stub it with
        # a synthetic spec so import succeeds on hosts without a working
        # torchcodec dylib.
        import types as _types

        for _m in ("torchcodec", "torchcodec.decoders", "torchcodec.decoders._core"):
            if _m not in sys.modules:
                sm = _types.ModuleType(_m)
                sm.__spec__ = _types.SimpleNamespace(
                    name=_m, loader=None, submodule_search_locations=[]
                )
                sys.modules[_m] = sm
        from speechbrain.inference.classifiers import EncoderClassifier

        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/sb_ecapa"),
        )
        _ECAPA.eval()
    return _ECAPA


def speaker_embedding(path: str) -> np.ndarray:
    import librosa
    import torch

    model = _ecapa()
    wav, _ = librosa.load(path, sr=16000, mono=True)
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(wav_t).squeeze()  # [192]
    return emb.cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# UTMOS via torchaudio SQUIM_SUBJECTIVE
# ---------------------------------------------------------------------------

_SQUIM = None
_NMR = None  # non-matching reference tensor


def _squim(nmr_path: str):
    global _SQUIM, _NMR
    if _SQUIM is None:
        import librosa
        import torch
        import torchaudio

        _SQUIM = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
        nmr, _ = librosa.load(nmr_path, sr=16000, mono=True)
        # SQUIM subjective wants ≥ 1s context; cap at 10s of clean speech
        nmr = nmr[: 16000 * 10]
        _NMR = torch.from_numpy(nmr).unsqueeze(0)
    return _SQUIM


def utmos(path: str, nmr_path: str) -> float:
    import librosa
    import torch

    model = _squim(nmr_path)
    wav, _ = librosa.load(path, sr=16000, mono=True)
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        mos = model(wav_t, _NMR)
    return float(mos.item())


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(source: str, reference: str, outputs: list, nmr_path: str = None):
    """Return list of dicts {name, utmos, secs, wer}."""
    nmr_path = nmr_path or reference
    print(f"Source:    {source}")
    print(f"Reference: {reference}")
    print(f"NMR (for SQUIM): {nmr_path}")
    print()

    print("Transcribing source for WER baseline...")
    src_text = transcribe(source)
    print(f"  source transcript ({len(src_text.split())} words): {src_text[:80]}...")
    print()

    print("Computing reference speaker embedding...")
    ref_emb = speaker_embedding(reference)

    rows = []

    def _row(name, path):
        print(f"Evaluating {name}...")
        try:
            u = utmos(path, nmr_path)
        except Exception as e:
            u = float("nan")
            print(f"  utmos failed: {e}")
        try:
            s = cosine(speaker_embedding(path), ref_emb)
        except Exception as e:
            s = float("nan")
            print(f"  secs failed: {e}")
        try:
            hyp = transcribe(path)
            w = _wer(src_text, hyp)
        except Exception as e:
            hyp = ""
            w = float("nan")
            print(f"  wer failed: {e}")
        rows.append(
            {
                "name": name,
                "path": path,
                "utmos": u,
                "secs": s,
                "wer": w,
                "hyp_text": hyp,
            }
        )
        print(f"  UTMOS={u:.2f}  SECS={s:.3f}  WER={w:.2f}")
        print()

    _row("[ref]", reference)

    for path in outputs:
        name = Path(path).stem
        _row(name, path)

    return rows, src_text


def print_table(rows):
    print()
    print("=" * 76)
    print(f"{'model':20s} {'UTMOS↑':>8s} {'SECS↑':>8s} {'WER↓':>8s}")
    print("-" * 76)
    for r in rows:
        u = f"{r['utmos']:.2f}" if not np.isnan(r["utmos"]) else "—"
        s = f"{r['secs']:.3f}" if not np.isnan(r["secs"]) else "—"
        w = f"{r['wer']:.2f}" if not np.isnan(r["wer"]) else "—"
        print(f"{r['name']:20s} {u:>8s} {s:>8s} {w:>8s}")
    print("=" * 76)
    print("UTMOS: 1–5, higher = more natural / cleaner (no-reference MOS)")
    print("SECS:  0–1, higher = speaker closer to reference  (ECAPA cos-sim)")
    print("WER:   0–inf, lower = content better preserved  (Whisper-small)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="Source audio path")
    parser.add_argument(
        "--reference", required=True, help="Reference (target speaker) audio path"
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        help="Output WAV files to score (one per model)",
    )
    parser.add_argument("--json", default=None, help="Save results as JSON")
    args = parser.parse_args()

    rows, src_text = evaluate(args.source, args.reference, args.outputs)
    print_table(rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"source_text": src_text, "results": rows}, f, indent=2)
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()
