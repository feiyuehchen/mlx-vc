"""kNN-VC: Voice conversion with just nearest neighbors.

Non-parametric approach: WavLM features + k-nearest neighbors + HiFi-GAN.
Zero-shot, no training needed. MIT license.

Paper: "Voice Conversion With Just Nearest Neighbors" (Interspeech 2023)
"""

import os
import tempfile
from typing import Optional, Union

import numpy as np

from mlx_vc.audio_io import load_audio, save_audio


class KnnVC:
    """kNN-VC: non-parametric voice conversion.

    Uses WavLM-Large for feature extraction, k-nearest neighbors for
    voice matching, and HiFi-GAN for waveform synthesis.
    """

    def __init__(
        self,
        topk: int = 4,
        prematched: bool = True,
        verbose: bool = True,
    ):
        self.topk = topk
        self.prematched = prematched
        self.verbose = verbose
        self.sr = 16000
        self.sample_rate = self.sr

    def convert(
        self,
        source_audio: Union[str, np.ndarray],
        ref_audio: Union[str, np.ndarray],
        topk: Optional[int] = None,
    ) -> np.ndarray:
        """Convert source voice to match reference speaker.

        Args:
            source_audio: Path or numpy array of source speech
            ref_audio: Path or numpy array (or list of paths) of reference
            topk: Number of nearest neighbors (default: 4)

        Returns:
            Converted audio as numpy array at 16kHz
        """
        from mlx_vc.backend import run_backend

        src_path, src_cleanup = self._to_path(source_audio, "src")
        ref_path, ref_cleanup = self._to_path(ref_audio, "ref")

        try:
            return run_backend(
                "knn-vc",
                source=src_path,
                reference=ref_path,
                verbose=self.verbose,
                topk=topk or self.topk,
                prematched=self.prematched,
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
            "name": "kNN-VC",
            "type": "zero-shot (non-parametric)",
            "sr": self.sr,
            "backbone": "WavLM-Large + kNN + HiFi-GAN",
            "topk": self.topk,
        }
