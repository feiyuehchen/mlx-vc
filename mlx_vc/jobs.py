"""In-memory job manager for batch voice conversion runs.

Used by `mlx_vc.server` to coordinate running multiple VC models on the
same source/reference inputs. Models run serialized via a shared semaphore
so they don't fight over the GPU.
"""

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from mlx_vc.audio_io import save_audio
from mlx_vc.backend import BACKENDS, run_backend


# Global semaphore: only one model inference at a time (single GPU)
MODEL_LOCK = asyncio.Semaphore(1)

# Job temp dir (cleaned on shutdown)
JOB_TMP_ROOT = Path("/tmp/mlx_vc_jobs")
JOB_TMP_ROOT.mkdir(parents=True, exist_ok=True)


# Expected speed tier (seconds, rough estimates for a 10s input on M-series)
ETA_SECONDS = {
    "openvoice": 1,
    "knn-vc": 4,
    "cosyvoice": 8,
    "meanvc": 5,
    "rvc": 5,
    "seed-vc": 20,
}


@dataclass
class TaskState:
    model: str
    status: str = "queued"  # queued | running | done | error
    output_path: Optional[str] = None
    elapsed_s: float = 0.0
    eta_s: float = 0.0
    error: Optional[str] = None
    _start_time: float = 0.0


@dataclass
class Job:
    job_id: str
    source_path: str
    reference_path: str
    text: str = "Welcome to the demo."  # for CosyVoice
    tasks: Dict[str, TaskState] = field(default_factory=dict)
    tmp_dir: Optional[Path] = None
    created_at: float = field(default_factory=time.time)


class JobManager:
    """In-memory job manager."""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create_job(
        self,
        source_path: str,
        reference_path: str,
        models: List[str],
        text: str = None,
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        tmp_dir = JOB_TMP_ROOT / job_id
        tmp_dir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            source_path=source_path,
            reference_path=reference_path,
            text=text or "Welcome to the demo.",
            tmp_dir=tmp_dir,
        )
        for model in models:
            job.tasks[model] = TaskState(
                model=model, eta_s=ETA_SECONDS.get(model, 10)
            )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def run_job(self, job: Job) -> None:
        """Schedule all tasks. Fast models run first."""
        ordered_models = sorted(
            job.tasks.keys(), key=lambda m: ETA_SECONDS.get(m, 10)
        )
        # Spawn one async task per model — they all wait on MODEL_LOCK
        await asyncio.gather(
            *[self._run_task(job, model) for model in ordered_models],
            return_exceptions=True,
        )

    async def _run_task(self, job: Job, model: str) -> None:
        task = job.tasks[model]

        async with MODEL_LOCK:
            task.status = "running"
            task._start_time = time.monotonic()

            try:
                output_path = str(job.tmp_dir / f"{model}.wav")
                await asyncio.to_thread(
                    self._invoke_model,
                    model,
                    job.source_path,
                    job.reference_path,
                    output_path,
                    job.text,
                )
                task.output_path = output_path
                task.status = "done"
            except Exception as e:
                task.status = "error"
                task.error = str(e)
            finally:
                task.elapsed_s = time.monotonic() - task._start_time

    @staticmethod
    def _invoke_model(
        model: str,
        source: str,
        reference: str,
        output: str,
        text: str,
    ) -> None:
        """Dispatch a single model run. Runs in a thread.

        Fast in-process path:
          - openvoice: reuse the realtime singleton (no model reload)
          - cosyvoice: in-process via mlx-audio
        Subprocess path (model loads each call):
          - seed-vc, knn-vc, meanvc, rvc
        """
        if model == "openvoice":
            # Use the persistent OpenVoiceSession singleton — avoids the
            # ~10s model load that the subprocess backend incurs each call.
            import librosa
            from mlx_vc.realtime import get_session

            session = get_session()
            session.set_reference(reference)

            src_audio, _ = librosa.load(source, sr=session.sr)
            converted = session.convert_chunk(src_audio, sample_rate=session.sr)
            save_audio(output, converted, sample_rate=session.output_sr)
            return

        if model == "cosyvoice":
            # CosyVoice is in-process, takes text not audio
            from mlx_vc.models.cosyvoice import CosyVoiceVC

            vc = CosyVoiceVC(verbose=False)
            audio = vc.convert(text=text, ref_audio=reference)
            save_audio(output, audio, sample_rate=vc.sample_rate)
            return

        if model in BACKENDS:
            # Subprocess backend (seed-vc, knn-vc, meanvc, rvc)
            run_backend(model, source=source, reference=reference, output=output)
            return

        raise ValueError(f"Unknown model: {model}")

    def cleanup_job(self, job_id: str) -> None:
        """Remove a job and its temp files."""
        job = self.jobs.pop(job_id, None)
        if job and job.tmp_dir and job.tmp_dir.exists():
            shutil.rmtree(job.tmp_dir, ignore_errors=True)

    def cleanup_all(self) -> None:
        """Clean up everything (called on shutdown)."""
        for job_id in list(self.jobs.keys()):
            self.cleanup_job(job_id)
        if JOB_TMP_ROOT.exists():
            shutil.rmtree(JOB_TMP_ROOT, ignore_errors=True)


# Module-level singleton
_MANAGER = JobManager()


def get_manager() -> JobManager:
    return _MANAGER
