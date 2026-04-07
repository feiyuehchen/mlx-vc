"""Backend runner: each model runs inference via subprocess.

Each VC model has its own inference script. The main process communicates
via subprocess + temporary audio files. This keeps the main package
lightweight and avoids import-time dependency conflicts.

Usage:
    from mlx_vc.backend import run_backend

    audio = run_backend(
        "seed-vc",
        source="source.wav",
        reference="ref.wav",
    )
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from mlx_vc.audio_io import load_audio

# Map backend names to their inference scripts
_SCRIPT_DIR = Path(__file__).parent / "backends"

BACKENDS = {
    "seed-vc": {
        "script": "seed_vc_infer.py",
        "sample_rate": 22050,
        "description": "Seed-VC zero-shot voice conversion (Whisper + DiT + BigVGAN)",
    },
}


def run_backend(
    backend: str,
    source: str,
    reference: str,
    output: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """Run a VC backend via subprocess.

    Args:
        backend: Backend name (e.g., "seed-vc")
        source: Path to source audio file
        reference: Path to reference audio file
        output: Path for output (auto-generated temp file if None)
        verbose: Print backend output
        **kwargs: Additional args passed to the backend

    Returns:
        Converted audio as numpy array
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend}. Available: {list(BACKENDS.keys())}"
        )

    info = BACKENDS[backend]
    script = _SCRIPT_DIR / info["script"]
    sr = info["sample_rate"]

    # Create temp output if needed
    cleanup = False
    if output is None:
        fd, output = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cleanup = True

    try:
        args_json = json.dumps({
            "source": os.path.abspath(source),
            "reference": os.path.abspath(reference),
            "output": os.path.abspath(output),
            **kwargs,
        })

        if verbose:
            print(f"[{backend}] Running inference...")

        result = subprocess.run(
            [sys.executable, str(script), "--args", args_json],
            capture_output=not verbose,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr if not verbose else ""
            raise RuntimeError(f"Backend {backend} failed (exit {result.returncode})\n{stderr}")

        return load_audio(output, sample_rate=sr)
    finally:
        if cleanup and os.path.exists(output):
            os.unlink(output)
