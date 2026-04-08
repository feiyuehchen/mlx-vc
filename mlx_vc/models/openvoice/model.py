"""OpenVoice V2: Tone color conversion wrapper.

Zero-shot voice conversion by transferring tone color (timbre) from
a reference speaker to source audio. Based on MyShell's OpenVoice V2.
"""

import os
import tempfile
from typing import Optional, Union

import numpy as np

from mlx_vc.audio_io import load_audio, save_audio


class OpenVoiceVC:
    """OpenVoice V2 tone color converter.

    Extracts speaker embeddings and converts tone color between speakers.
    Supports English, Chinese, Japanese, Korean, French, Spanish.
    """

    def __init__(self, tau: float = 0.3, verbose: bool = True):
        self.tau = tau
        self.verbose = verbose
        self.sr = 22050
        self.sample_rate = self.sr

    def convert(
        self,
        source_audio: Union[str, np.ndarray],
        ref_audio: Union[str, np.ndarray],
        tau: Optional[float] = None,
    ) -> np.ndarray:
        """Convert source audio tone color to match reference speaker.

        Args:
            source_audio: Path or numpy array of source speech
            ref_audio: Path or numpy array of target speaker reference
            tau: Style control (0=more target style, 1=more source style)

        Returns:
            Converted audio as numpy array
        """
        from mlx_vc.backend import run_backend

        src_path, src_cleanup = self._to_path(source_audio, "src")
        ref_path, ref_cleanup = self._to_path(ref_audio, "ref")

        try:
            return run_backend(
                "openvoice",
                source=src_path,
                reference=ref_path,
                verbose=self.verbose,
                tau=tau or self.tau,
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
            "name": "OpenVoice V2",
            "type": "zero-shot tone color conversion",
            "sr": self.sr,
            "backbone": "VITS-based SynthesizerTrn",
            "languages": "EN, ZH, JA, KO, FR, ES",
        }
