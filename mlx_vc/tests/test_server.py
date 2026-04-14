"""Tests for the FastAPI server."""

import io

from fastapi.testclient import TestClient

from mlx_vc.server import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert "openvoice" in models
    assert "seed-vc" in models
    assert "knn-vc" in models


def test_convert_unknown_model():
    """Requesting an unknown model should return 400."""
    fake_wav = io.BytesIO(b"RIFF" + b"\x00" * 100)
    r = client.post(
        "/v1/audio/convert",
        data={"model": "nonexistent"},
        files={
            "source": ("src.wav", fake_wav, "audio/wav"),
            "reference": ("ref.wav", fake_wav, "audio/wav"),
        },
    )
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


def test_convert_missing_files():
    """Missing source/reference files should return 422."""
    r = client.post("/v1/audio/convert", data={"model": "openvoice"})
    assert r.status_code == 422


def test_batch_unknown_model_rejected():
    """Batch should reject unknown model names."""
    fake_wav = io.BytesIO(b"RIFF" + b"\x00" * 100)
    r = client.post(
        "/v1/audio/convert/batch",
        data={"models": "openvoice,nonexistent"},
        files={
            "source": ("src.wav", fake_wav, "audio/wav"),
            "reference": ("ref.wav", fake_wav, "audio/wav"),
        },
    )
    assert r.status_code == 400
    assert "Unknown models" in r.json()["detail"]


def test_job_not_found():
    """GET non-existent job should 404."""
    r = client.get("/v1/jobs/nonexistent_id")
    assert r.status_code == 404


def test_job_result_not_found():
    """GET result for non-existent job should 404."""
    r = client.get("/v1/jobs/nonexistent/result/openvoice")
    assert r.status_code == 404
