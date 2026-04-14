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
model (provided via ECE433_RVC_MODEL_PATH or args["model_path"]). The
reference voice is baked INTO the model.
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

    # The model path: explicit arg > env var > default
    model_path = (
        args.get("model_path")
        or os.environ.get("ECE433_RVC_MODEL_PATH")
        or "/Users/fychen/research/ADP/data/ece472course/zundamon_rvc.npz"
    )

    if not os.path.exists(model_path):
        print(
            f"ERROR: RVC model not found: {model_path}\n"
            f"Set ECE433_RVC_MODEL_PATH or pass model_path in args.",
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
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
