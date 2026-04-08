"""MeanVC: Lightweight streaming voice conversion (14M params).

Uses ASR encoder + ECAPA speaker embedding + DiT + Vocos vocoder.
Supports streaming chunk-wise inference with KV-cache.

Paper: Mean Flows for One-step Voice Conversion (2025)
"""

import os
import tempfile
from typing import Optional, Union

import numpy as np

from mlx_vc.audio_io import load_audio, save_audio


class MeanVC:
    """MeanVC: lightweight zero-shot voice conversion.

    Only 14M params, streaming capable, RTF 0.136 on CPU.
    """

    def __init__(
        self,
        steps: int = 1,
        chunk_size: int = 40,
        verbose: bool = True,
    ):
        self.steps = steps
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.sr = 16000
        self.sample_rate = self.sr

    def convert(
        self,
        source_audio: Union[str, np.ndarray],
        ref_audio: Union[str, np.ndarray],
        steps: Optional[int] = None,
    ) -> np.ndarray:
        """Convert source voice to match reference speaker.

        Args:
            source_audio: Path or numpy array of source speech
            ref_audio: Path or numpy array of reference speaker
            steps: Override default inference steps (1=fastest, 2=better)

        Returns:
            Converted audio as numpy array at 16kHz
        """
        from mlx_vc.backend import run_backend

        src_path, src_cleanup = self._to_path(source_audio, "src")
        ref_path, ref_cleanup = self._to_path(ref_audio, "ref")

        try:
            return run_backend(
                "meanvc",
                source=src_path,
                reference=ref_path,
                verbose=self.verbose,
                steps=steps or self.steps,
                chunk_size=self.chunk_size,
            )
        finally:
            if src_cleanup:
                os.unlink(src_path)
            if ref_cleanup:
                os.unlink(ref_path)

    def _to_path(self, audio, prefix):
        if isinstance(audio, str):
            return audio, False
        fd, path = tempfile.mkstemp(suffix=".wav", prefix=f"mlx_vc_{prefix}_")
        os.close(fd)
        save_audio(path, audio, sample_rate=self.sr)
        return path, True

    @property
    def model_info(self) -> dict:
        return {
            "name": "MeanVC",
            "type": "zero-shot (streaming, 14M params)",
            "sr": self.sr,
            "backbone": "DiT + Mean Flows (1-step)",
            "vocoder": "Vocos",
            "steps": self.steps,
        }
