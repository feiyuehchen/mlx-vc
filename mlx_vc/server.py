"""FastAPI server for voice conversion.

Provides HTTP endpoints so other applications (web, mobile, scripts)
can use voice conversion without running Python directly.

Start: python -m mlx_vc.server
Then:  curl -X POST http://localhost:8000/v1/audio/convert \
         -F "source=@my_voice.wav" -F "reference=@target.wav" \
         --output converted.wav
"""

import argparse
import io
import os
import tempfile
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from mlx_vc.audio_io import load_audio, save_audio
from mlx_vc.backend import BACKENDS, run_backend
from mlx_vc.generate import AVAILABLE_MODELS

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


def _save_upload(upload: UploadFile, prefix: str) -> str:
    """Save an uploaded file to a temp path."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=f"mlx_vc_{prefix}_")
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


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
