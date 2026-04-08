"""Tests for audio I/O: load, save, round-trip integrity."""

import os
import tempfile

import numpy as np
import pytest

from mlx_vc.audio_io import load_audio, save_audio


def test_save_and_load_roundtrip():
    """Save audio to WAV then load it back — should be nearly identical."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    original = 0.5 * np.sin(2 * np.pi * 440 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    try:
        save_audio(path, original, sample_rate=sr)
        loaded = load_audio(path, sample_rate=sr)

        assert loaded.shape == original.shape
        # WAV is 16-bit, so some quantization error expected
        np.testing.assert_allclose(loaded, original, atol=1e-3)
    finally:
        os.unlink(path)


def test_load_resamples():
    """Loading at a different sample rate should resample."""
    sr_orig = 44100
    sr_target = 16000
    t = np.linspace(0, 1.0, sr_orig, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    try:
        save_audio(path, audio, sample_rate=sr_orig)
        loaded = load_audio(path, sample_rate=sr_target)

        # Should be resampled to target SR
        expected_samples = int(1.0 * sr_target)
        assert abs(len(loaded) - expected_samples) <= 1
    finally:
        os.unlink(path)


def test_load_nonexistent_raises():
    """Loading a file that doesn't exist should raise an error."""
    with pytest.raises(Exception):
        load_audio("/nonexistent/path/audio.wav")
