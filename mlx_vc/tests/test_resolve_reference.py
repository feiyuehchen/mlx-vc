"""Tests for the WS reference path resolver."""

import os
import tempfile

import pytest

pytest.importorskip("fastapi", reason="install with `pip install -e .[server]`")

from mlx_vc.server import MLX_VC_REF_DIR, _resolve_reference  # noqa: E402


def test_absolute_path_returned_as_is():
    """Existing absolute paths are returned unchanged."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        assert _resolve_reference(path) == path
    finally:
        os.unlink(path)


def test_absolute_path_nonexistent_returns_none():
    """A non-existent absolute path should return None."""
    assert _resolve_reference("/nonexistent/abs/path.wav") is None


def test_filename_resolved_under_ref_dir():
    """A bare filename should be looked up under MLX_VC_REF_DIR."""
    if not MLX_VC_REF_DIR or not os.path.isdir(MLX_VC_REF_DIR):
        pytest.skip("MLX_VC_REF_DIR not set or missing")
    # Stage a temp WAV inside MLX_VC_REF_DIR, resolve by bare filename.
    fname = "_test_resolve_reference.wav"
    path = os.path.join(MLX_VC_REF_DIR, fname)
    with open(path, "wb") as f:
        f.write(b"RIFF")
    try:
        assert _resolve_reference(fname) == path
    finally:
        os.unlink(path)


def test_unknown_filename_returns_none():
    assert _resolve_reference("zzz_does_not_exist_zzz.wav") is None


def test_path_traversal_blocked():
    """Filenames with .. or / should be rejected to prevent escape."""
    assert _resolve_reference("../etc/passwd") is None
    assert _resolve_reference("subdir/file.wav") is None


def test_empty_string_returns_none():
    assert _resolve_reference("") is None
    assert _resolve_reference(None) is None
