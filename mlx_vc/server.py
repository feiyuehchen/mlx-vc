"""FastAPI server for voice conversion.

Provides HTTP endpoints so other applications (web, mobile, scripts)
can use voice conversion without running Python directly.

Start: python -m mlx_vc.server
Then:  curl -X POST http://localhost:8000/v1/audio/convert \
         -F "source=@my_voice.wav" -F "reference=@target.wav" \
         --output converted.wav
"""

import argparse
import asyncio
import io
import os
import struct
import tempfile
import time
from typing import List, Optional

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from mlx_vc.audio_io import load_audio, save_audio
from mlx_vc.backend import BACKENDS, run_backend
from mlx_vc.generate import AVAILABLE_MODELS
from mlx_vc.jobs import JOB_TMP_ROOT, get_manager

app = FastAPI(
    title="mlx-vc",
    description="Voice conversion API for Apple Silicon",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MLX_VC_ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API docs."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """List available VC models and their capabilities."""
    models = {}
    for name, info in AVAILABLE_MODELS.items():
        models[name] = {
            "description": info["description"],
            "default_repo": info.get("default_repo"),
        }
    return {"models": models}


@app.post("/v1/audio/convert")
async def convert_audio(
    source: UploadFile = File(..., description="Source audio file"),
    reference: UploadFile = File(..., description="Reference speaker audio"),
    model: str = Form("openvoice", description="Model name"),
):
    """Convert source audio to match reference speaker's voice.

    Returns the converted audio as a WAV file.
    """
    if model not in BACKENDS and model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {list(AVAILABLE_MODELS.keys())}",
        )

    # Save uploaded files to temp
    src_path = _save_upload(source, "src")
    ref_path = _save_upload(reference, "ref")
    out_path = tempfile.mktemp(suffix=".wav", prefix="mlx_vc_out_")

    try:
        t0 = time.time()

        if model in BACKENDS:
            # Subprocess backend
            run_backend(model, source=src_path, reference=ref_path, output=out_path)
        else:
            # In-process model (CosyVoice, etc.)
            from mlx_vc.generate import get_vc_model

            vc = get_vc_model(model, verbose=False)
            audio = vc.convert(source_audio=src_path, ref_audio=ref_path)
            save_audio(out_path, audio, sample_rate=vc.sample_rate)

        elapsed = time.time() - t0

        # Read output and return as WAV
        with open(out_path, "rb") as f:
            wav_bytes = f.read()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Processing-Time": f"{elapsed:.3f}s",
                "X-Model": model,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [src_path, ref_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)


def _save_upload(upload: UploadFile, prefix: str, dest_dir: str = None) -> str:
    """Save an uploaded file to a temp path."""
    if dest_dir is not None:
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{prefix}_{upload.filename or 'audio.wav'}")
        with open(path, "wb") as f:
            f.write(upload.file.read())
        return path
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=f"mlx_vc_{prefix}_")
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


# ============================================================================
# Batch endpoints — run multiple models on the same input
# ============================================================================

@app.post("/v1/audio/convert/batch")
async def convert_batch(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    models: str = Form("openvoice,seed-vc,knn-vc,cosyvoice"),
    text: Optional[str] = Form(None),
):
    """Run multiple VC models on the same source/reference, in the background.

    Returns a job_id that can be polled via /v1/jobs/{job_id}.
    """
    requested = [m.strip() for m in models.split(",") if m.strip()]
    valid_models = set(BACKENDS.keys()) | {"cosyvoice"}
    invalid = [m for m in requested if m not in valid_models]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown models: {invalid}. Valid: {sorted(valid_models)}",
        )

    manager = get_manager()
    # Pre-create job_id then save uploads under its tmp dir
    import uuid

    job_id = uuid.uuid4().hex[:12]
    tmp_dir = JOB_TMP_ROOT / job_id
    src_path = _save_upload(source, "src", str(tmp_dir))
    ref_path = _save_upload(reference, "ref", str(tmp_dir))

    job = manager.create_job(
        source_path=src_path,
        reference_path=ref_path,
        models=requested,
        text=text,
    )
    # Override the auto-generated job_id and tmp_dir
    manager.jobs.pop(job.job_id, None)
    job.job_id = job_id
    job.tmp_dir = tmp_dir
    manager.jobs[job_id] = job

    # Spawn the run in background
    asyncio.create_task(manager.run_job(job))

    return {
        "job_id": job_id,
        "models": requested,
        "tasks": [
            {"model": m, "status": "queued", "eta_s": job.tasks[m].eta_s}
            for m in requested
        ],
    }


@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status of a batch job."""
    job = get_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    tasks = []
    for model, task in job.tasks.items():
        elapsed = task.elapsed_s
        if task.status == "running" and task._start_time:
            elapsed = time.monotonic() - task._start_time
        tasks.append(
            {
                "model": model,
                "status": task.status,
                "elapsed_s": round(elapsed, 2),
                "eta_s": task.eta_s,
                "error": task.error,
                "result_url": (
                    f"/v1/jobs/{job_id}/result/{model}"
                    if task.status == "done"
                    else None
                ),
            }
        )

    all_done = all(t["status"] in ("done", "error") for t in tasks)
    return {"job_id": job_id, "tasks": tasks, "all_done": all_done}


@app.get("/v1/jobs/{job_id}/result/{model}")
async def get_job_result(job_id: str, model: str):
    """Stream the converted WAV for a completed task."""
    job = get_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    task = job.tasks.get(model)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Model not in job: {model}")
    if task.status != "done" or not task.output_path:
        raise HTTPException(
            status_code=409, detail=f"Task not ready: {task.status}"
        )
    return FileResponse(task.output_path, media_type="audio/wav")


# ============================================================================
# WebSocket: real-time voice conversion via OpenVoice
# ============================================================================

# Reference audio resolution: search this dir for filenames sent by clients.
# Set MLX_VC_REF_DIR to override.
MLX_VC_REF_DIR = os.environ.get(
    "MLX_VC_REF_DIR", "/Users/fychen/research/ADP/data/ece472course"
)


# Uploaded references (live for the server's lifetime, cleaned on shutdown)
UPLOAD_REF_DIR = "/tmp/mlx_vc_uploads/refs"
os.makedirs(UPLOAD_REF_DIR, exist_ok=True)


def _resolve_reference(ref: str) -> Optional[str]:
    """Resolve a reference identifier into an absolute path on the server.

    Accepts:
      - Absolute path: returned as-is if it exists
      - Bare filename: looked up under MLX_VC_REF_DIR, then UPLOAD_REF_DIR
    """
    if not ref:
        return None
    # Absolute path?
    if os.path.isabs(ref) and os.path.exists(ref):
        return ref
    # Bare filename — block path traversal then look up under known dirs
    if "/" in ref or ".." in ref:
        return None
    for d in (MLX_VC_REF_DIR, UPLOAD_REF_DIR):
        candidate = os.path.join(d, ref)
        if os.path.exists(candidate):
            return candidate
    return None


@app.post("/v1/audio/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    """Save an uploaded reference WAV; return its server-side filename.

    The frontend can later use this filename in WS init messages or
    batch requests. Files live in UPLOAD_REF_DIR until server shutdown.
    """
    import uuid

    # Generate a unique filename keeping the original extension
    suffix = ".wav"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
        if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            suffix = ext

    filename = f"upload_{uuid.uuid4().hex[:12]}{suffix}"
    path = os.path.join(UPLOAD_REF_DIR, filename)
    with open(path, "wb") as out:
        out.write(file.file.read())

    return {"filename": filename, "path": path}


@app.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket):
    """Stream microphone audio in, get converted audio out.

    Protocol:
      C->S text: {"type":"init","reference":"<filename or abs path>","sample_rate":16000,"tau":0.3}
      S->C text: {"type":"ready"} or {"type":"error","message":...}
      C->S binary: Float32 PCM at sample_rate (mono)
      S->C binary: Float32 PCM (converted, at OpenVoice 22050Hz)
      C->S text: {"type":"stop"}

    Bare filenames are resolved under MLX_VC_REF_DIR (default
    /Users/fychen/research/ADP/data/ece472course).
    """
    await websocket.accept()

    from mlx_vc.realtime import get_session

    session = None
    sample_rate = 16000
    tau = 0.3

    try:
        # Wait for init message
        init_msg = await websocket.receive_json()
        if init_msg.get("type") != "init":
            await websocket.send_json({"type": "error", "message": "Expected init"})
            return

        ref_input = init_msg.get("reference")
        sample_rate = int(init_msg.get("sample_rate", 16000))
        tau = float(init_msg.get("tau", 0.3))

        ref_path = _resolve_reference(ref_input)
        if ref_path is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        f"Reference not found: '{ref_input}'. "
                        f"Searched under MLX_VC_REF_DIR={MLX_VC_REF_DIR}"
                    ),
                }
            )
            return

        # Load OpenVoice session (singleton)
        session = await asyncio.to_thread(get_session)
        await asyncio.to_thread(session.set_reference, ref_path)

        await websocket.send_json(
            {"type": "ready", "model": "openvoice", "output_sr": session.output_sr}
        )

        # Audio loop
        block_count = 0
        while True:
            msg = await websocket.receive()
            if "text" in msg and msg["text"]:
                import json

                payload = json.loads(msg["text"])
                if payload.get("type") == "stop":
                    break
                continue

            if "bytes" not in msg or msg["bytes"] is None:
                continue

            # Decode Float32 PCM
            raw = msg["bytes"]
            audio = np.frombuffer(raw, dtype=np.float32).copy()
            if len(audio) == 0:
                continue

            # Skip silent blocks (energy gate)
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 0.005:
                # Send silence at output sr
                out_len = int(len(audio) * session.output_sr / sample_rate)
                silence = np.zeros(out_len, dtype=np.float32)
                await websocket.send_bytes(silence.tobytes())
                continue

            t0 = time.monotonic()
            converted = await asyncio.to_thread(
                session.convert_chunk, audio, sample_rate, tau
            )
            latency_ms = (time.monotonic() - t0) * 1000

            await websocket.send_bytes(converted.astype(np.float32).tobytes())

            block_count += 1
            if block_count % 10 == 0:
                await websocket.send_json(
                    {"type": "stats", "latency_ms": round(latency_ms, 1)}
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.on_event("shutdown")
async def cleanup_on_shutdown():
    """Clean up job temp files when server stops."""
    get_manager().cleanup_all()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="mlx-vc API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    print(f"Starting mlx-vc server at http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"Models: {list(AVAILABLE_MODELS.keys())}")

    uvicorn.run(
        "mlx_vc.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
