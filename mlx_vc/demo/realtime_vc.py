#!/usr/bin/env python3
"""Real-time voice conversion demo using Seed-VC.

Microphone input -> Seed-VC (XLSR-tiny, 10 diffusion steps) -> Speaker output

Usage:
    python -m mlx_vc.demo.realtime_vc --reference speaker.wav
    python -m mlx_vc.demo.realtime_vc --reference speaker.wav --list-devices
    python -m mlx_vc.demo.realtime_vc --reference speaker.wav --input-device 1 --output-device 2
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd


def list_devices():
    """List available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        kind = []
        if d["max_input_channels"] > 0:
            kind.append("IN")
        if d["max_output_channels"] > 0:
            kind.append("OUT")
        marker = " <-default" if i in sd.default.device else ""
        print(f"  [{i}] {d['name']} ({'/'.join(kind)}, {int(d['default_samplerate'])}Hz){marker}")
    print()


class RealtimeVC:
    """Real-time voice conversion using Seed-VC."""

    def __init__(
        self,
        reference_path: str,
        block_time: float = 0.5,
        crossfade_time: float = 0.05,
        extra_time: float = 2.5,
        diffusion_steps: int = 10,
        inference_cfg_rate: float = 0.7,
        max_prompt_length: float = 3.0,
        input_device=None,
        output_device=None,
    ):
        self.reference_path = reference_path
        self.block_time = block_time
        self.crossfade_time = crossfade_time
        self.extra_time = extra_time
        self.diffusion_steps = diffusion_steps
        self.inference_cfg_rate = inference_cfg_rate
        self.max_prompt_length = max_prompt_length
        self.input_device = input_device
        self.output_device = output_device

        self.model_set = None
        self.running = False
        self.sr = 22050  # model sample rate

    def _load_models(self):
        """Load Seed-VC real-time model (XLSR-tiny)."""
        print("Loading Seed-VC real-time model...")

        seed_vc_ref = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "seed-vc-ref"
        )
        seed_vc_ref = os.path.abspath(seed_vc_ref)

        if not os.path.exists(seed_vc_ref):
            raise RuntimeError(
                f"Seed-VC reference repo not found at {seed_vc_ref}. "
                "Clone it: git clone https://github.com/Plachtaa/seed-vc.git seed-vc-ref"
            )

        sys.path.insert(0, seed_vc_ref)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(seed_vc_ref, "checkpoints", "hf_cache"))

        # Patch BigVGAN
        self._patch_bigvgan()

        import torch
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device

        # Import real-time gui's load_models but with our device
        import importlib
        rt_module = importlib.import_module("real-time-gui")
        rt_module.device = device

        from types import SimpleNamespace
        args = SimpleNamespace(
            checkpoint_path=None,
            config_path=None,
            fp16=False,
        )
        self.model_set = rt_module.load_models(args)
        self.custom_infer = rt_module.custom_infer

        mel_fn_args = self.model_set[-1]
        self.sr = mel_fn_args["sampling_rate"]
        self.hop_length = mel_fn_args["hop_size"]

        # Load reference audio
        import librosa
        ref_audio = librosa.load(self.reference_path, sr=self.sr)[0]
        self.reference_wav = ref_audio
        self.reference_wav_name = self.reference_path

        print(f"Model loaded. SR={self.sr}, hop={self.hop_length}")
        print(f"Reference: {self.reference_path} ({len(ref_audio)/self.sr:.1f}s)")

    def _patch_bigvgan(self):
        """Patch BigVGAN for newer huggingface_hub."""
        try:
            from modules.bigvgan import bigvgan
            original_fp = bigvgan.BigVGAN._from_pretrained.__func__

            @classmethod
            def patched_fp(cls, *, proxies=None, resume_download=False, **kwargs):
                return original_fp(cls, proxies=proxies, resume_download=resume_download, **kwargs)

            bigvgan.BigVGAN._from_pretrained = patched_fp
        except Exception:
            pass

    def run(self):
        """Start the real-time VC loop."""
        import torch
        import torchaudio.transforms as tat

        self._load_models()

        block_samples = int(self.block_time * self.sr)
        extra_samples = int(self.extra_time * self.sr)
        crossfade_samples = int(self.crossfade_time * self.sr)
        # Total buffer: extra context + block + some tail
        total_samples = extra_samples + block_samples + int(2.0 * self.sr)

        # Audio buffers
        input_buffer = np.zeros(total_samples, dtype=np.float32)
        output_queue = deque()
        lock = threading.Lock()

        # Resampler for device SR -> model SR
        device_sr = int(sd.query_devices(self.input_device, "input")["default_samplerate"])
        if device_sr != self.sr:
            resample_in = tat.Resample(device_sr, self.sr).to(self.device)
            resample_out = tat.Resample(self.sr, device_sr).to(self.device)
        else:
            resample_in = resample_out = None

        print(f"\nDevice SR: {device_sr}Hz, Model SR: {self.sr}Hz")
        print(f"Block: {self.block_time}s, Extra context: {self.extra_time}s")
        print(f"Diffusion steps: {self.diffusion_steps}")
        print("\n" + "=" * 50)
        print("  REAL-TIME VOICE CONVERSION ACTIVE")
        print("  Press Ctrl+C to stop")
        print("=" * 50 + "\n")

        self.running = True
        prev_output = None

        def audio_callback(indata, outdata, frames, t, status):
            if status:
                print(f"Audio status: {status}")

            # Collect input
            mono_in = indata[:, 0].copy()
            with lock:
                input_buffer[:-len(mono_in)] = input_buffer[len(mono_in):]
                input_buffer[-len(mono_in):] = mono_in

            # Play output if available
            if output_queue:
                chunk = output_queue.popleft()
                if len(chunk) >= frames:
                    outdata[:, 0] = chunk[:frames]
                else:
                    outdata[:frames, 0] = 0
                    outdata[:len(chunk), 0] = chunk
            else:
                outdata[:] = 0

        # Processing thread
        def process_loop():
            nonlocal prev_output

            while self.running:
                t0 = time.time()

                with lock:
                    chunk = input_buffer.copy()

                # Check if there's actual audio (simple energy gate)
                rms = np.sqrt(np.mean(chunk[-block_samples:] ** 2))
                if rms < 0.005:
                    # Silence - output silence
                    silence = np.zeros(int(self.block_time * device_sr), dtype=np.float32)
                    output_queue.append(silence)
                    time.sleep(self.block_time * 0.5)
                    continue

                # Resample to model SR if needed
                with torch.no_grad():
                    chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
                    if resample_in is not None:
                        chunk_16k = resample_in(chunk_tensor)
                    else:
                        chunk_16k = chunk_tensor

                    # Convert block_frame_16k to 16kHz for content encoder
                    block_frame_16k = torchaudio.functional.resample(
                        chunk_16k, self.sr, 16000
                    )

                    skip_head = int(self.extra_time * 50)  # in 20ms frames
                    skip_tail = int(2.0 * 50)
                    return_length = int(self.block_time * 50)

                    try:
                        output = self.custom_infer(
                            self.model_set,
                            self.reference_wav,
                            self.reference_wav_name,
                            block_frame_16k,
                            int(self.block_time * 16000),
                            skip_head,
                            skip_tail,
                            return_length,
                            self.diffusion_steps,
                            self.inference_cfg_rate,
                            self.max_prompt_length,
                        )

                        # Crossfade with previous
                        output_np = output.cpu().numpy()
                        if prev_output is not None and crossfade_samples > 0:
                            fade_len = min(crossfade_samples, len(output_np), len(prev_output))
                            fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
                            fade_in = np.cos(np.linspace(np.pi / 2, 0, fade_len)) ** 2
                            output_np[:fade_len] = (
                                output_np[:fade_len] * fade_in
                                + prev_output[-fade_len:] * fade_out
                            )
                        prev_output = output_np.copy()

                        # Resample to device SR
                        if resample_out is not None:
                            output_tensor = torch.from_numpy(output_np).float().to(self.device)
                            output_device = resample_out(output_tensor).cpu().numpy()
                        else:
                            output_device = output_np

                        output_queue.append(output_device)

                    except Exception as e:
                        print(f"Inference error: {e}")
                        silence = np.zeros(int(self.block_time * device_sr), dtype=np.float32)
                        output_queue.append(silence)

                elapsed = time.time() - t0
                rtf = elapsed / self.block_time
                print(f"\rBlock: {self.block_time:.2f}s | Inference: {elapsed:.3f}s | RTF: {rtf:.2f} | Queue: {len(output_queue)}  ", end="", flush=True)

                # Don't process faster than real-time
                wait = self.block_time - elapsed
                if wait > 0:
                    time.sleep(wait * 0.5)

        # Need torchaudio for the process loop
        import torchaudio

        # Start audio stream
        try:
            with sd.Stream(
                samplerate=device_sr,
                blocksize=int(self.block_time * device_sr),
                device=(self.input_device, self.output_device),
                channels=1,
                dtype="float32",
                callback=audio_callback,
            ):
                proc_thread = threading.Thread(target=process_loop, daemon=True)
                proc_thread.start()

                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.running = False
            time.sleep(0.5)
            print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time voice conversion using Seed-VC"
    )
    parser.add_argument(
        "--reference", type=str, required=True,
        help="Path to reference speaker audio",
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
    parser.add_argument("--block-time", type=float, default=0.5, help="Block size in seconds (default: 0.5)")
    parser.add_argument("--diffusion-steps", type=int, default=10, help="Diffusion steps (default: 10)")
    parser.add_argument("--cfg-rate", type=float, default=0.7, help="CFG rate (default: 0.7)")
    parser.add_argument("--extra-time", type=float, default=2.5, help="Extra context time in seconds (default: 2.5)")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    vc = RealtimeVC(
        reference_path=args.reference,
        block_time=args.block_time,
        diffusion_steps=args.diffusion_steps,
        inference_cfg_rate=args.cfg_rate,
        extra_time=args.extra_time,
        input_device=args.input_device,
        output_device=args.output_device,
    )
    vc.run()


if __name__ == "__main__":
    main()
