"""RVC (Retrieval-based Voice Conversion) wrapper.

RVC requires a per-speaker fine-tuned model (not zero-shot).
Supports both PyTorch (.pth) and Acelogic MLX models.

To use: provide a pre-trained RVC model path.
Models available at: https://voice-models.com or train with Applio.
"""

import os
import tempfile
from typing import Optional, Union

import numpy as np

from mlx_vc.audio_io import load_audio, save_audio


class RVCVC:
    """RVC voice conversion (requires per-speaker model).

    This is a placeholder wrapper. RVC models must be fine-tuned
    per speaker (~10 min of clean audio). Use Applio or RVC WebUI
    to train, then load the .pth model here.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        f0_method: str = "rmvpe",
        verbose: bool = True,
    ):
        self.model_path = model_path
        self.f0_method = f0_method
        self.verbose = verbose
        self.sr = 48000
        self.sample_rate = self.sr

        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RVC model not found: {model_path}. "
                "Train one with Applio or download from voice-models.com"
            )

    def convert(
        self,
        source_audio: Union[str, np.ndarray],
        ref_audio: Union[str, np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Convert source audio using the loaded RVC model.

        Note: ref_audio is ignored for RVC (the model IS the speaker).

        Args:
            source_audio: Path or numpy array of source speech
            ref_audio: Ignored (kept for API compatibility)

        Returns:
            Converted audio as numpy array
        """
        if self.model_path is None:
            raise RuntimeError(
                "No RVC model loaded. Provide model_path= when initializing. "
                "RVC requires per-speaker fine-tuned models."
            )

        from mlx_vc.backend import run_backend

        src_path, cleanup = self._to_path(source_audio, "src")
        try:
            return run_backend(
                "rvc",
                source=src_path,
                reference=src_path,  # not used but required by interface
                verbose=self.verbose,
                model_path=self.model_path,
                f0_method=self.f0_method,
            )
        finally:
            if cleanup:
                os.unlink(src_path)

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
            "name": "RVC",
            "type": "fine-tuned per-speaker",
            "sr": self.sr,
            "f0_method": self.f0_method,
            "model_path": self.model_path,
            "note": "Requires per-speaker model. Train with Applio or RVC WebUI.",
        }
