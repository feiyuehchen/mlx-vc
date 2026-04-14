#!/usr/bin/env python3
"""Speed benchmark for mlx-vc batch endpoint and OpenVoice realtime path.

Measures:
  1. OpenVoice in-process: cold (first call) vs warm (subsequent)
  2. OpenVoice WS-style chunk inference (300ms blocks)
  3. Seed-VC subprocess batch
  4. kNN-VC subprocess batch
  5. CosyVoice in-process

Run with the mlx-vc venv after starting `python -m mlx_vc.server`:
    python benchmarks/bench_models.py [--server-port 8000]
"""

import argparse
import statistics
import time
from pathlib import Path

import requests


SOURCE = "/Users/fychen/research/ADP/data/ece472course/professor_src_10s.wav"
REFERENCE = "/Users/fychen/research/ADP/data/ece472course/professor_ref.wav"


def fmt_ms(s: float) -> str:
    return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


def run_batch(server: str, model: str) -> dict:
    """POST a single-model batch and poll until done. Returns timing dict."""
    t_start = time.monotonic()
    with open(SOURCE, "rb") as f1, open(REFERENCE, "rb") as f2:
        r = requests.post(
            f"{server}/v1/audio/convert/batch",
            files={
                "source": ("src.wav", f1, "audio/wav"),
                "reference": ("ref.wav", f2, "audio/wav"),
            },
            data={"models": model},
        )
    if r.status_code != 200:
        return {"model": model, "error": r.text, "wall_s": 0}

    job_id = r.json()["job_id"]

    # Poll
    while True:
        time.sleep(0.2)
        s = requests.get(f"{server}/v1/jobs/{job_id}").json()
        if s["all_done"]:
            wall_s = time.monotonic() - t_start
            task = s["tasks"][0]
            return {
                "model": model,
                "wall_s": wall_s,
                "infer_s": task["elapsed_s"],
                "status": task["status"],
                "error": task.get("error"),
            }


def bench_openvoice_in_process():
    """Direct OpenVoiceSession benchmark (no HTTP, no subprocess)."""
    print("\n[1] OpenVoice in-process (no HTTP)")
    print("-" * 50)
    import librosa
    import numpy as np
    from mlx_vc.realtime import OpenVoiceSession

    s = OpenVoiceSession()

    t = time.monotonic()
    s.load()
    print(f"  load():           {fmt_ms(time.monotonic() - t)}")

    t = time.monotonic()
    s.set_reference(REFERENCE)
    print(f"  set_reference():  {fmt_ms(time.monotonic() - t)}")

    src, _ = librosa.load(SOURCE, sr=s.sr)

    # Cold convert (first inference)
    t = time.monotonic()
    out = s.convert_chunk(src, sample_rate=s.sr)
    cold = time.monotonic() - t
    print(f"  convert (cold):   {fmt_ms(cold)} ({len(src)/s.sr:.1f}s audio)")

    # Warm converts
    times = []
    for _ in range(5):
        t = time.monotonic()
        s.convert_chunk(src, sample_rate=s.sr)
        times.append(time.monotonic() - t)
    print(
        f"  convert (warm):   "
        f"{fmt_ms(statistics.mean(times))} avg "
        f"(min {fmt_ms(min(times))}, max {fmt_ms(max(times))})"
    )

    # Realtime chunk size (300ms @ 16kHz)
    chunk = np.random.randn(int(0.3 * 16000)).astype(np.float32) * 0.1
    times = []
    for _ in range(10):
        t = time.monotonic()
        s.convert_chunk(chunk, sample_rate=16000)
        times.append(time.monotonic() - t)
    avg_ms = statistics.mean(times) * 1000
    print(
        f"  300ms chunk @16k: {avg_ms:.1f}ms avg "
        f"(min {min(times)*1000:.0f}ms, max {max(times)*1000:.0f}ms)"
    )
    rtf = avg_ms / 300
    print(f"  -> RTF {rtf:.2f} ({'real-time capable' if rtf < 1 else 'TOO SLOW'})")


def bench_batch_endpoint(server: str):
    """Benchmark each model via the batch endpoint."""
    print("\n[2] Batch endpoint (HTTP)")
    print("-" * 50)

    results = []
    for model in ["openvoice", "openvoice", "knn-vc", "cosyvoice", "seed-vc"]:
        print(f"  Running {model}... ", end="", flush=True)
        r = run_batch(server, model)
        if "error" in r and r["error"]:
            print(f"ERROR: {r['error'][:80]}")
            continue
        if r.get("status") == "error":
            print(f"FAILED: {r.get('error')}")
            continue
        print(
            f"wall={fmt_ms(r['wall_s'])}, "
            f"infer={fmt_ms(r['infer_s'])}"
        )
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--skip-server", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("mlx-vc speed benchmark")
    print("=" * 50)
    print(f"Source:    {Path(SOURCE).name} (10s)")
    print(f"Reference: {Path(REFERENCE).name}")

    bench_openvoice_in_process()

    if not args.skip_server:
        server = f"http://127.0.0.1:{args.server_port}"
        try:
            requests.get(f"{server}/health", timeout=2)
        except Exception:
            print(f"\n  Server not running at {server}, skipping HTTP bench.")
            print(f"  Start it with: python -m mlx_vc.server --port {args.server_port}")
            return

        bench_batch_endpoint(server)

    print("\n" + "=" * 50)
    print("done.")


if __name__ == "__main__":
    main()
