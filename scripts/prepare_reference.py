#!/usr/bin/env python3
"""Prepare clean reference audio from a long recording.

Pipeline:
  1. Run Demucs (htdemucs_ft, four-bag ensemble) two-stem vocal separation
     to strip background noise / music / room tone from the target speaker's
     audio.
  2. Extract two trimmed reference clips from the clean vocals track:
       - <name>_ref_clean.wav  (60 s, default reference for most VC models)
       - <name>_ref_3min.wav   (3 min, kNN-VC's matching pool benefits from
                                 a much larger reference)

Requires `demucs` and `ffmpeg` in PATH:
    pip install demucs

Usage:
    python prepare_reference.py \\
        --input path/to/long_recording.wav \\
        --output_dir path/to/dest/ \\
        --name speaker \\
        [--start_sec 30]    # skip the first N seconds (intros, silence)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result


def separate_vocals(input_path: Path, work_dir: Path) -> Path:
    """Run demucs htdemucs_ft and return the vocals track path."""
    print("\n[1/3] Separating vocals with demucs (htdemucs_ft)...")
    print("      Multi-bag ensemble — expect 2–5 min per 10 min of input audio.")

    run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs_ft",
            "--out", str(work_dir),
            str(input_path),
        ]
    )

    stem = input_path.stem
    vocals = work_dir / "htdemucs_ft" / stem / "vocals.wav"
    if not vocals.exists():
        # Fallback to single-bag htdemucs (rare)
        vocals = work_dir / "htdemucs" / stem / "vocals.wav"
    if not vocals.exists():
        print(f"ERROR: demucs output not found. Expected: {vocals}", file=sys.stderr)
        sys.exit(1)

    print(f"  Vocals extracted: {vocals}")
    return vocals


def trim_and_copy(
    vocals: Path, output_path: Path, start_sec: float, duration_sec: float
):
    """Extract [start_sec, start_sec + duration_sec] from vocals → mono 22.05 kHz WAV."""
    run(
        [
            "ffmpeg", "-y",
            "-i", str(vocals),
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-ar", "22050",          # standard VC sample rate
            "-ac", "1",              # mono
            "-c:a", "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
    )
    size_kb = output_path.stat().st_size // 1024
    print(f"  Saved {output_path.name}  ({duration_sec:.0f}s, {size_kb}KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True,
                        help="Long recording (WAV/M4A/MP3 — anything ffmpeg reads)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the trimmed reference WAVs")
    parser.add_argument("--name", default=None,
                        help="Output stem (default: input filename stem). "
                             "Outputs <name>_ref_clean.wav and <name>_ref_3min.wav.")
    parser.add_argument("--start_sec", type=float, default=30.0,
                        help="Skip first N seconds (intros, silence)")
    parser.add_argument("--clean_seconds", type=float, default=60.0,
                        help="Length of <name>_ref_clean.wav (default 60s)")
    parser.add_argument("--long_seconds", type=float, default=180.0,
                        help="Length of <name>_ref_3min.wav (default 180s)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    name = args.name or input_path.stem

    with tempfile.TemporaryDirectory(prefix="mlx_vc_demucs_") as work_dir:
        work_dir = Path(work_dir)

        vocals = separate_vocals(input_path, work_dir)

        clean_path = output_dir / f"{name}_ref_clean.wav"
        long_path = output_dir / f"{name}_ref_3min.wav"

        print(f"\n[2/3] Extracting {args.clean_seconds:.0f}s general reference...")
        trim_and_copy(vocals, clean_path, args.start_sec, args.clean_seconds)

        print(f"\n[3/3] Extracting {args.long_seconds:.0f}s reference (for kNN-VC)...")
        trim_and_copy(vocals, long_path, args.start_sec, args.long_seconds)

    print("\nDone.")
    print(f"  {clean_path.name} — default reference for most VC models")
    print(f"  {long_path.name} — pass to kNN-VC via `extra_references` for denser matching pool")


if __name__ == "__main__":
    main()
