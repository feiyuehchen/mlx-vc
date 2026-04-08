"""Tests for the backend subprocess runner."""

import os
import tempfile

import numpy as np
import pytest

from mlx_vc.audio_io import save_audio
from mlx_vc.backend import BACKENDS, run_backend


def test_backends_registry_complete():
    """All expected backends should be registered."""
    expected = ["seed-vc", "openvoice", "knn-vc", "meanvc"]
    for name in expected:
        assert name in BACKENDS, f"Missing backend: {name}"


def test_backends_have_required_fields():
    """Each backend must have script, sample_rate, description."""
    for name, cfg in BACKENDS.items():
        assert "script" in cfg, f"{name} missing 'script'"
        assert "sample_rate" in cfg, f"{name} missing 'sample_rate'"
        assert "description" in cfg, f"{name} missing 'description'"


def test_backend_scripts_exist():
    """All backend inference scripts should exist on disk."""
    from pathlib import Path

    script_dir = Path(__file__).parent.parent / "backends"
    for name, cfg in BACKENDS.items():
        script = script_dir / cfg["script"]
        assert script.exists(), f"Backend script missing: {script}"


def test_run_backend_unknown_raises():
    """Running an unknown backend should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        run_backend("nonexistent-model", source="a.wav", reference="b.wav")


def test_run_backend_missing_source_raises():
    """Running with a nonexistent source file should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        run_backend(
            "openvoice",
            source="/nonexistent/source.wav",
            reference="/nonexistent/ref.wav",
        )
