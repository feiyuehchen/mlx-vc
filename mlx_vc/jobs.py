"""In-memory job manager for batch voice conversion runs.

Used by `mlx_vc.server` to coordinate running multiple VC models on the
same source/reference inputs. Models run serialized via a shared semaphore
so they don't fight over the GPU.
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from mlx_vc.audio_io import save_audio
from mlx_vc.backend import BACKENDS, run_backend

log = logging.getLogger(__name__)

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
    "sesame": 6,
    "outetts": 8,
    "dia": 6,
    "meanvc": 5,
    "rvc": 5,
    "freevc": 5,
    "freevc-s": 5,
    "seed-vc": 20,
}


# Model-name → HF repo mapping for mlx-audio TTS-clone models.
# Each of these goes through the same `CosyVoiceVC` wrapper (text+ref_audio
# voice cloning) so source audio is Whisper-transcribed before synthesis.
_TTS_CLONE_MODELS = {
    "cosyvoice": "mlx-community/chatterbox-fp16",  # Resemble AI Chatterbox
    "sesame": "mlx-community/csm-1b",  # Sesame CSM-1B
    "outetts": "OuteAI/Llama-OuteTTS-1.0-1B",  # OuteTTS 1.0
    "dia": "mlx-community/Dia-1.6B",  # Nari Labs Dia
}

# Per-model maximum reference duration (seconds). None means "no trim".
# Sesame CSM's max_seq_len is tight — 60s of reference blows past 923
# frames. OuteTTS warns at 20s then degrades; Dia handles 20s.
_TTS_CLONE_MAX_REF_SEC = {
    "sesame": 10,
    "outetts": 12,
    "dia": 15,
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


def _knn_vc_extra_refs(reference: str) -> List[str]:
    """Find sibling '*_3min.wav' and '*_clean.wav' files next to `reference`.

    Returns list of extra reference paths (may be empty).  Used so kNN-VC
    gets a 3-minute reference pool even when the caller passes the short
    default reference.
    """
    ref_path = Path(reference).resolve()
    parent = ref_path.parent
    stem = ref_path.stem
    # Strip common suffixes to find the speaker prefix
    for suffix in ("_ref", "_src", "_10s", "_clean", "_3min"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Only use demucs-denoised siblings, never the raw long recording
    # (too noisy + too slow to process into a matching pool).
    candidates = [
        parent / f"{stem}_ref_3min.wav",
        parent / f"{stem}_ref_clean.wav",
        parent / f"{stem}_3min.wav",
    ]
    return [str(c) for c in candidates if c.exists() and c != ref_path]


def _trim_reference(path: str, max_seconds: float) -> str:
    """Return a path to a reference audio clipped to `max_seconds`.

    Uses a deterministic cache path under JOB_TMP_ROOT keyed by the source
    path + duration so repeated calls don't re-encode.  Returns the original
    path if the file is already shorter than `max_seconds`.
    """
    import librosa
    import soundfile as sf

    try:
        dur = librosa.get_duration(path=path)
    except Exception:
        return path
    if dur <= max_seconds:
        return path

    cache_dir = JOB_TMP_ROOT / "_refs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{Path(path).stem}_{int(max_seconds)}s.wav"
    out_path = cache_dir / cache_key
    if out_path.exists():
        return str(out_path)

    wav, sr = librosa.load(path, sr=None, mono=True)
    n = int(sr * max_seconds)
    sf.write(str(out_path), wav[:n], sr)
    return str(out_path)


def _transcribe_source(source: str) -> str:
    """Transcribe a source audio file to text using Whisper (small model).

    Used by the CosyVoice path so the output preserves the source content
    rather than synthesizing a fixed demo sentence.
    """
    try:
        import whisper

        model = whisper.load_model("small")
        result = model.transcribe(source, fp16=False)
        text = result.get("text", "").strip()
        if text:
            return text
    except Exception as e:
        log.warning(
            "[cosyvoice] transcription failed (%s), falling back to demo text", e
        )
    return "Welcome to the demo."


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
            job.tasks[model] = TaskState(model=model, eta_s=ETA_SECONDS.get(model, 10))
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def run_job(self, job: Job) -> None:
        """Schedule all tasks. Fast models run first."""
        ordered_models = sorted(job.tasks.keys(), key=lambda m: ETA_SECONDS.get(m, 10))
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

        if model in _TTS_CLONE_MODELS:
            # mlx-audio TTS + voice cloning path. Source is transcribed with
            # Whisper, then synthesized in the target voice.  Different
            # entries here use different mlx-audio models (Chatterbox,
            # Sesame CSM, OuteTTS, Dia, ...).
            if not text or text == "Welcome to the demo.":
                text = _transcribe_source(source)
            from mlx_vc.models.cosyvoice import CosyVoiceVC

            hf_model = _TTS_CLONE_MODELS[model]
            vc = CosyVoiceVC(model_name=hf_model, verbose=False)

            # Several mlx-audio TTS models have strict max reference length
            # (Sesame: ~10s context, OuteTTS: 15s, Dia: 20s).  Longer
            # references trigger ValueErrors or quality regressions.  Trim
            # to a safe window here (first N seconds of reference).
            effective_ref = reference
            max_ref_seconds = _TTS_CLONE_MAX_REF_SEC.get(model)
            if max_ref_seconds is not None:
                effective_ref = _trim_reference(reference, max_ref_seconds)

            # Sesame CSM requires ref_text alongside ref_audio. Transcribe
            # the trimmed reference so ref_text matches the audio actually passed.
            extra = {}
            if model == "sesame":
                extra["ref_text"] = _transcribe_source(effective_ref)

            audio = vc.convert(text=text, ref_audio=effective_ref, **extra)
            save_audio(output, audio, sample_rate=vc.sample_rate)
            return

        if model in BACKENDS:
            # Subprocess backend (seed-vc, knn-vc, meanvc, rvc).
            extra_kwargs = {}
            if model == "knn-vc":
                # kNN-VC quality scales with reference pool size; if a longer
                # sibling reference is available (e.g. <name>_ref_3min.wav
                # sitting next to <name>_ref.wav), feed it as extra_refs.
                extra_kwargs["extra_references"] = _knn_vc_extra_refs(reference)
            run_backend(
                model,
                source=source,
                reference=reference,
                output=output,
                **extra_kwargs,
            )
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
