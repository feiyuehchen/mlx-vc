"""Tests for the WS reference path resolver."""

import os
import tempfile

import pytest

from mlx_vc.server import _resolve_reference, MLX_VC_REF_DIR


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
    # Use a known file from the demo data dir
    expected = os.path.join(MLX_VC_REF_DIR, "professor_ref.wav")
    if os.path.exists(expected):
        assert _resolve_reference("professor_ref.wav") == expected


def test_unknown_filename_returns_none():
    assert _resolve_reference("zzz_does_not_exist_zzz.wav") is None


def test_path_traversal_blocked():
    """Filenames with .. or / should be rejected to prevent escape."""
    assert _resolve_reference("../etc/passwd") is None
    assert _resolve_reference("subdir/file.wav") is None


def test_empty_string_returns_none():
    assert _resolve_reference("") is None
    assert _resolve_reference(None) is None
