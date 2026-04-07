"""CosyVoice3 voice conversion wrapper using mlx-audio.

This wraps mlx-audio's TTS models that support voice cloning via reference audio,
providing a unified VC interface. Supported models include Chatterbox, Spark,
OuteTTS, Qwen3-TTS, and other mlx-audio TTS models with ref_audio support.
"""

from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np

try:
    from mlx_audio.tts.utils import load_model as load_tts_model
except ImportError:
    load_tts_model = None


DEFAULT_MODEL = "mlx-community/chatterbox-fp16"


class CosyVoiceVC:
    """Voice conversion using mlx-audio TTS models with voice cloning.

    This provides zero-shot voice conversion by synthesizing text with a
    target speaker's voice characteristics extracted from reference audio.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        verbose: bool = True,
    ):
        if load_tts_model is None:
            raise ImportError(
                "mlx-audio is required for CosyVoice VC: "
                "pip install mlx-audio[tts] or pip install mlx-vc[cosyvoice]"
            )

        self.model_name = model_name
        self.verbose = verbose
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.verbose:
                print(f"Loading model: {self.model_name}")
            self._model = load_tts_model(self.model_name)
        return self._model

    @property
    def sample_rate(self) -> int:
        return getattr(self.model, "sample_rate", 24000)

    def convert(
        self,
        text: str,
        ref_audio: Union[str, np.ndarray],
        ref_audio_sr: Optional[int] = None,
        exaggeration: float = 0.1,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        speed: float = 1.0,
        lang_code: str = "en",
        **kwargs,
    ) -> np.ndarray:
        """Convert text to speech using the reference speaker's voice.

        Args:
            text: Text to synthesize.
            ref_audio: Path to reference audio file or numpy array.
            ref_audio_sr: Sample rate of ref_audio (if numpy array).
            exaggeration: Emotion/style exaggeration (0-1).
            cfg_weight: Classifier-free guidance weight.
            temperature: Sampling temperature.
            speed: Speech speed multiplier.
            lang_code: Language code.

        Returns:
            Generated audio as numpy array.
        """
        # Load reference audio if path
        if isinstance(ref_audio, str):
            from mlx_vc.audio_io import load_audio

            ref_audio = load_audio(ref_audio, sample_rate=self.sample_rate)

        # Convert to mlx array (mlx-audio expects mlx arrays for ref_audio)
        import mlx.core as mx

        if isinstance(ref_audio, np.ndarray):
            ref_audio = mx.array(ref_audio)

        # Collect all audio chunks
        chunks = []
        for result in self.model.generate(
            text=text,
            ref_audio=ref_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            speed=speed,
            lang_code=lang_code,
            verbose=self.verbose,
            **kwargs,
        ):
            audio = result.audio
            if hasattr(audio, "tolist"):
                # Convert mlx array to numpy
                audio = np.array(audio, dtype=np.float32)
            chunks.append(audio.flatten())

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks)

    def stream(
        self,
        text: str,
        ref_audio: Union[str, np.ndarray],
        **kwargs,
    ) -> Generator[np.ndarray, None, None]:
        """Stream voice conversion results chunk by chunk.

        Args:
            text: Text to synthesize.
            ref_audio: Path to reference audio or numpy array.
            **kwargs: Additional arguments passed to convert().

        Yields:
            Audio chunks as numpy arrays.
        """
        if isinstance(ref_audio, str):
            from mlx_vc.audio_io import load_audio

            ref_audio = load_audio(ref_audio, sample_rate=self.sample_rate)

        import mlx.core as mx

        if isinstance(ref_audio, np.ndarray):
            ref_audio = mx.array(ref_audio)

        for result in self.model.generate(
            text=text,
            ref_audio=ref_audio,
            verbose=self.verbose,
            **kwargs,
        ):
            audio = result.audio
            if hasattr(audio, "tolist"):
                audio = np.array(audio, dtype=np.float32)
            yield audio.flatten()
