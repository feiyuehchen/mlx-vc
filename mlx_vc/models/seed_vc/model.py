"""Seed-VC: Zero-shot voice conversion wrapper.

Uses subprocess to run the Seed-VC inference script,
keeping the main process clean from heavy PyTorch imports.
"""

import os
import tempfile
from typing import Optional, Union

import numpy as np

from mlx_vc.audio_io import load_audio, save_audio


class SeedVC:
    """Seed-VC zero-shot voice conversion.

    Wraps the Seed-VC inference backend. The actual model runs in a
    subprocess via backends/seed_vc_infer.py to isolate dependencies.
    """

    def __init__(
        self,
        diffusion_steps: int = 30,
        inference_cfg_rate: float = 0.7,
        length_adjust: float = 1.0,
        f0_condition: bool = False,
        verbose: bool = True,
    ):
        self.diffusion_steps = diffusion_steps
        self.inference_cfg_rate = inference_cfg_rate
        self.length_adjust = length_adjust
        self.f0_condition = f0_condition
        self.verbose = verbose

        self.sr = 22050 if not f0_condition else 44100
        self.sample_rate = self.sr

    def convert(
        self,
        source_audio: Union[str, np.ndarray],
        ref_audio: Union[str, np.ndarray],
        diffusion_steps: Optional[int] = None,
        inference_cfg_rate: Optional[float] = None,
        length_adjust: Optional[float] = None,
    ) -> np.ndarray:
        """Convert source audio to target speaker's voice.

        Args:
            source_audio: Path or numpy array of source speech
            ref_audio: Path or numpy array of reference speaker audio
            diffusion_steps: Override default steps
            inference_cfg_rate: Override default CFG rate
            length_adjust: Override default length adjustment

        Returns:
            Converted audio as numpy array
        """
        from mlx_vc.backend import run_backend

        # Save numpy arrays to temp files if needed
        src_path, src_cleanup = self._to_path(source_audio, "src")
        ref_path, ref_cleanup = self._to_path(ref_audio, "ref")

        try:
            result = run_backend(
                "seed-vc",
                source=src_path,
                reference=ref_path,
                verbose=self.verbose,
                diffusion_steps=diffusion_steps or self.diffusion_steps,
                inference_cfg_rate=inference_cfg_rate or self.inference_cfg_rate,
                length_adjust=length_adjust or self.length_adjust,
                f0_condition=self.f0_condition,
            )
            return result
        finally:
            if src_cleanup:
                os.unlink(src_path)
            if ref_cleanup:
                os.unlink(ref_path)

    def _to_path(self, audio, prefix: str):
        """Convert audio input to a file path."""
        if isinstance(audio, str):
            return audio, False
        # numpy array -> temp file
        fd, path = tempfile.mkstemp(suffix=".wav", prefix=f"mlx_vc_{prefix}_")
        os.close(fd)
        save_audio(path, audio, sample_rate=self.sr)
        return path, True

    @property
    def model_info(self) -> dict:
        return {
            "name": "Seed-VC",
            "type": "zero-shot",
            "sr": self.sr,
            "content_encoder": "Whisper-small",
            "style_encoder": "CAMPPlus (192-dim)",
            "backbone": "DiT + WaveNet + CFM",
            "vocoder": "BigVGAN v2 22kHz",
            "diffusion_steps": self.diffusion_steps,
        }
