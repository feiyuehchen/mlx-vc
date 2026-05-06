#!/usr/bin/env python3
"""RVC backend inference script (uses Acelogic's pure-MLX RVC).

This wrapper runs the rvc-mlx-ref/rvc-mlx-cli.py via subprocess in its own
isolated venv (rvc-mlx-ref/.venv) because rvc-mlx-ref pins numpy<2 and is
incompatible with the main mlx-vc venv.

Required setup:
  git clone https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX.git rvc-mlx-ref
  cd rvc-mlx-ref && uv venv --python 3.10 .venv
  source .venv/bin/activate
  pip install "numpy<2" mlx torch torchaudio librosa scipy soundfile faiss-cpu \\
              einops transformers omegaconf pyworld noisereduce ffmpeg-python \\
              soxr stftpitchshift edge-tts wget pyyaml requests bs4

  # Convert an RVC .pth model to MLX format:
  python tools/convert_rvc_model.py generator path/to/voice.pth out.npz

The "reference" parameter is IGNORED — RVC uses a per-speaker fine-tuned
model (provided via $RVC_MODEL_PATH or args["model_path"]). The reference
voice is baked INTO the model.
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True)
    parsed = parser.parse_args()
    args = json.loads(parsed.args)

    source = args["source"]
    output = args["output"]
    pitch = args.get("pitch", 0)
    f0_method = args.get("f0_method", "rmvpe")

    # Model path: explicit arg > env var.  RVC has no sensible default —
    # the model is per-speaker fine-tuned, caller must provide one.
    model_path = (
        args.get("model_path")
        or os.environ.get("RVC_MODEL_PATH")
    )

    if not model_path or not os.path.exists(model_path):
        print(
            f"ERROR: RVC model not found: {model_path!r}\n"
            f"Pass `model_path` in args or set RVC_MODEL_PATH env var.\n"
            f"Convert a .pth model with rvc-mlx-ref/tools/convert_rvc_model.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Locate rvc-mlx-ref repo and its venv
    rvc_mlx_ref = os.environ.get(
        "RVC_MLX_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "rvc-mlx-ref"),
    )
    rvc_mlx_ref = os.path.abspath(rvc_mlx_ref)

    venv_python = os.path.join(rvc_mlx_ref, ".venv", "bin", "python")
    cli = os.path.join(rvc_mlx_ref, "rvc-mlx-cli.py")

    if not os.path.exists(venv_python) or not os.path.exists(cli):
        print(
            f"ERROR: rvc-mlx-ref not set up at {rvc_mlx_ref}\n"
            f"Clone it: git clone https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX.git rvc-mlx-ref",
            file=sys.stderr,
        )
        sys.exit(1)

    # Subprocess the rvc-mlx CLI
    cmd = [
        venv_python, cli, "infer",
        "--model_path", model_path,
        "--input_path", os.path.abspath(source),
        "--output_path", os.path.abspath(output),
        "--pitch", str(pitch),
        "--f0_method", f0_method,
        "--index_rate", "0",
        "--volume_envelope", "1.0",
        "--protect", "0.5",
        "--export_format", "WAV",
    ]
    print(f"Running RVC: {' '.join(cmd[:3])}...")
    print(f"Model: {model_path}")
    print(f"Source: {source}")

    result = subprocess.run(
        cmd,
        cwd=rvc_mlx_ref,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("RVC stdout:", result.stdout[-500:], file=sys.stderr)
        print("RVC stderr:", result.stderr[-500:], file=sys.stderr)
        sys.exit(result.returncode)

    print(result.stdout[-300:])
    if not os.path.exists(output):
        print(f"ERROR: RVC didn't produce output {output}", file=sys.stderr)
        sys.exit(1)

    # Acelogic's rvc-mlx output is often very quiet (peak ~0.2).  Peak-
    # normalize so downstream metrics and listeners don't mistake volume
    # for quality.
    try:
        import librosa
        import numpy as np
        import soundfile as sf

        wav, sr_out = librosa.load(output, sr=None, mono=True)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if 0.0 < peak < 0.7:
            wav = wav * (0.95 / peak)
            sf.write(output, wav, sr_out)
            print(f"Peak-normalized (was {peak:.3f}, now 0.95)")
    except Exception as e:
        print(f"Warning: peak normalization failed: {e}", file=sys.stderr)

    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
