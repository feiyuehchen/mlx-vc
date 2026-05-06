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
    "openvoice": {
        "script": "openvoice_infer.py",
        "sample_rate": 22050,
        "description": "OpenVoice V2 tone color conversion (VITS-based)",
    },
    "knn-vc": {
        "script": "knn_vc_infer.py",
        "sample_rate": 16000,
        "description": "kNN-VC non-parametric voice conversion (WavLM + kNN + HiFi-GAN)",
    },
    "meanvc": {
        "script": "meanvc_infer.py",
        "sample_rate": 16000,
        "description": "MeanVC lightweight streaming VC (14M params, 1-step)",
    },
    "rvc": {
        "script": "rvc_infer.py",
        "sample_rate": 48000,
        "description": "RVC via Acelogic MLX (requires pre-converted .npz model)",
    },
    "freevc": {
        "script": "freevc_infer.py",
        "sample_rate": 16000,
        "description": "FreeVC one-shot VC (WavLM + VITS decoder, MIT)",
    },
    "freevc-s": {
        "script": "freevc_infer.py",
        "sample_rate": 16000,
        "description": "FreeVC-s variant (no speaker encoder, uses mel-spec of target)",
        "extra_args": {"variant": "freevc-s"},
    },
    "pocket-tts": {
        "script": "tts_clone_infer.py",
        "sample_rate": 24000,
        "description": "Kyutai Pocket-TTS (~235MB) English voice-cloning TTS via mlx-audio (NOTE: TTS-clone path, not true audio→audio VC)",
        "extra_args": {
            "hf_model": "mlx-community/Pocket-TTS",
            "max_ref_seconds": 10.0,
        },
    },
    "speecht5": {
        "script": "speecht5_infer.py",
        "sample_rate": 16000,
        "description": "Microsoft SpeechT5 VC — transformer seq2seq audio→audio VC (English, VCTK/CMU-ARCTIC trained)",
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
        # Merge in backend-registered extra_args (e.g. variant selection)
        # before user kwargs so users can still override.
        call_args = {
            "source": os.path.abspath(source),
            "reference": os.path.abspath(reference),
            "output": os.path.abspath(output),
            **info.get("extra_args", {}),
            **kwargs,
        }
        args_json = json.dumps(call_args)

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
