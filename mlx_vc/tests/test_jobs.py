"""Unit tests for mlx_vc.jobs (JobManager + dispatch)."""

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlx_vc.audio_io import save_audio
from mlx_vc.jobs import ETA_SECONDS, JOB_TMP_ROOT, JobManager, TaskState, get_manager


def _make_wav(path: str, duration_s: float = 1.0, sr: int = 22050) -> None:
    """Generate a small sine wav for testing."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    save_audio(path, audio, sample_rate=sr)


def test_eta_seconds_has_all_known_models():
    """Every model the demo cares about should have an ETA hint."""
    expected = {"openvoice", "seed-vc", "knn-vc", "cosyvoice", "meanvc", "rvc"}
    assert expected.issubset(set(ETA_SECONDS.keys()))


def test_taskstate_default_status():
    t = TaskState(model="openvoice")
    assert t.status == "queued"
    assert t.elapsed_s == 0.0
    assert t.error is None


def test_create_job_assigns_unique_id():
    mgr = JobManager()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
        _make_wav(src.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref:
            _make_wav(ref.name)
            j1 = mgr.create_job(src.name, ref.name, ["openvoice"])
            j2 = mgr.create_job(src.name, ref.name, ["openvoice"])
            assert j1.job_id != j2.job_id
            assert j1.tmp_dir.exists()
            assert j2.tmp_dir.exists()
            mgr.cleanup_job(j1.job_id)
            mgr.cleanup_job(j2.job_id)
        os.unlink(ref.name)
    os.unlink(src.name)


def test_create_job_initializes_tasks():
    mgr = JobManager()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
        _make_wav(src.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref:
            _make_wav(ref.name)
            job = mgr.create_job(src.name, ref.name, ["openvoice", "seed-vc", "knn-vc"])
            assert set(job.tasks.keys()) == {"openvoice", "seed-vc", "knn-vc"}
            for task in job.tasks.values():
                assert task.status == "queued"
                assert task.eta_s > 0
            mgr.cleanup_job(job.job_id)
        os.unlink(ref.name)
    os.unlink(src.name)


def test_get_manager_returns_singleton():
    a = get_manager()
    b = get_manager()
    assert a is b


def test_invoke_unknown_model_raises():
    """Dispatching an unknown model name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        JobManager._invoke_model("not-a-real-model", "src", "ref", "out", "text")


def test_invoke_cosyvoice_uses_in_process(monkeypatch):
    """Dispatching cosyvoice should NOT call run_backend."""
    called = {"backend": False, "cosyvoice": False}

    def fake_run_backend(*args, **kwargs):
        called["backend"] = True

    class FakeCosyVoice:
        sample_rate = 24000

        def __init__(self, **kwargs):
            pass

        def convert(self, **kwargs):
            called["cosyvoice"] = True
            return np.zeros(100, dtype=np.float32)

    import mlx_vc.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "run_backend", fake_run_backend)
    monkeypatch.setitem(
        __import__("sys").modules,
        "mlx_vc.models.cosyvoice",
        type("M", (), {"CosyVoiceVC": FakeCosyVoice}),
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
        out_path = out.name
    try:
        JobManager._invoke_model(
            "cosyvoice", "src.wav", "ref.wav", out_path, "Hello world"
        )
        assert called["cosyvoice"] is True
        assert called["backend"] is False
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_cleanup_removes_tmp_dir():
    mgr = JobManager()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
        _make_wav(src.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref:
            _make_wav(ref.name)
            job = mgr.create_job(src.name, ref.name, ["openvoice"])
            tmp_dir = job.tmp_dir
            assert tmp_dir.exists()
            mgr.cleanup_job(job.job_id)
            assert not tmp_dir.exists()
            assert mgr.get_job(job.job_id) is None
        os.unlink(ref.name)
    os.unlink(src.name)


def test_get_nonexistent_job_returns_none():
    mgr = JobManager()
    assert mgr.get_job("does-not-exist") is None
