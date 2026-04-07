"""Audio I/O utilities for loading and saving audio files."""

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def load_audio(
    path: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> np.ndarray:
    """Load an audio file and return as numpy array.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate for resampling.
        mono: If True, convert to mono.

    Returns:
        Audio samples as float32 numpy array, shape (samples,) if mono.
    """
    if librosa is None:
        raise ImportError("librosa is required: pip install librosa")

    audio, _ = librosa.load(path, sr=sample_rate, mono=mono)
    return audio.astype(np.float32)


def save_audio(
    path: str,
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> None:
    """Save audio to a WAV file.

    Args:
        path: Output file path.
        audio: Audio samples as numpy array.
        sample_rate: Sample rate in Hz.
    """
    import soundfile as sf

    sf.write(path, audio, sample_rate)
