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

        self.sr = 22050
        self.hop_length = 256

    def _setup_seed_vc(self):
        """Set up Seed-VC path and patches."""
        seed_vc_ref = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "seed-vc-ref"
        )
        seed_vc_ref = os.path.abspath(seed_vc_ref)

        if not os.path.exists(seed_vc_ref):
            raise RuntimeError(
                f"Seed-VC reference repo not found at {seed_vc_ref}. "
                "Clone it: git clone https://github.com/Plachtaa/seed-vc.git seed-vc-ref"
            )

        if seed_vc_ref not in sys.path:
            sys.path.insert(0, seed_vc_ref)
        os.environ.setdefault(
            "HF_HUB_CACHE",
            os.path.join(seed_vc_ref, "checkpoints", "hf_cache"),
        )
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # Patch BigVGAN for newer huggingface_hub
        try:
            from modules.bigvgan import bigvgan

            original_fp = bigvgan.BigVGAN._from_pretrained.__func__

            @classmethod
            def patched_fp(cls, *, proxies=None, resume_download=False, **kwargs):
                return original_fp(
                    cls, proxies=proxies, resume_download=resume_download, **kwargs
                )

            bigvgan.BigVGAN._from_pretrained = patched_fp
        except Exception:
            pass

    def _load_models(self):
        """Load Seed-VC models using inference.py's load_models."""
        import torch

        self._setup_seed_vc()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print("Loading Seed-VC models...")
        from types import SimpleNamespace

        args = SimpleNamespace(
            checkpoint=None,
            config=None,
            f0_condition=False,
            fp16=False,
        )
        from inference import load_models

        (
            self.model,
            self.semantic_fn,
            self.f0_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.mel_fn,
            self.mel_fn_args,
        ) = load_models(args)

        self.sr = self.mel_fn_args["sampling_rate"]
        self.hop_length = self.mel_fn_args["hop_size"]

        # Prepare reference audio embeddings
        self._prepare_reference()
        print(f"Model loaded. SR={self.sr}, hop={self.hop_length}")

    def _prepare_reference(self):
        """Pre-compute reference speaker embeddings."""
        import librosa
        import torch
        import torchaudio

        ref_audio = librosa.load(self.reference_path, sr=self.sr)[0]
        ref_audio = ref_audio[: int(self.sr * self.max_prompt_length)]

        ref_tensor = torch.from_numpy(ref_audio).float().to(self.device)
        ref_16k = torchaudio.functional.resample(ref_tensor, self.sr, 16000)

        with torch.no_grad():
            S_ref = self.semantic_fn(ref_16k.unsqueeze(0))
            self.ref_mel = self.mel_fn(ref_tensor.unsqueeze(0))
            target2_lengths = torch.LongTensor([self.ref_mel.size(2)]).to(self.device)
            self.prompt_condition = self.model.length_regulator(
                S_ref, ylens=target2_lengths, n_quantizers=3, f0=None
            )[0]

            feat = torchaudio.compliance.kaldi.fbank(
                ref_16k.unsqueeze(0),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            self.style = self.campplus_model(feat.unsqueeze(0))

        print(f"Reference: {self.reference_path} ({len(ref_audio)/self.sr:.1f}s)")

    def _infer_block(self, input_16k_tensor):
        """Run VC on one block of audio.

        Args:
            input_16k_tensor: [samples] at 16kHz on device

        Returns:
            output waveform as numpy array at self.sr
        """
        import torch
        import torchaudio

        with torch.no_grad():
            S_alt = self.semantic_fn(input_16k_tensor.unsqueeze(0))

            # Skip initial context (content encoder / DiT difference)
            ce_dit_frames = int(self.extra_time * 50)
            S_alt = S_alt[:, ce_dit_frames:]

            skip_head_frames = int(self.extra_time * 50)
            return_frames = int(self.block_time * 50)
            skip_tail_frames = int(2.0 * 50)
            total_frames = skip_head_frames + return_frames + skip_tail_frames - ce_dit_frames

            target_lengths = torch.LongTensor(
                [int(total_frames / 50 * self.sr / self.hop_length)]
            ).to(self.device)

            cond = self.model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]

            cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

            vc_target = self.model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.device),
                self.ref_mel,
                self.style,
                None,
                n_timesteps=self.diffusion_steps,
                inference_cfg_rate=self.inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.ref_mel.size(-1) :]

            vc_wave = self.vocoder_fn(vc_target.float()).squeeze()

        # Extract the return portion
        output_len = int(return_frames * self.sr / 50)
        tail_len = int(skip_tail_frames * self.sr / 50)
        if tail_len > 0 and tail_len < len(vc_wave):
            output = vc_wave[-output_len - tail_len : -tail_len]
        else:
            output = vc_wave[-output_len:]

        return output.cpu().numpy()

    def run(self):
        """Start the real-time VC loop."""
        import torch
        import torchaudio

        self._load_models()

        # Audio parameters
        device_sr = int(
            sd.query_devices(self.input_device, "input")["default_samplerate"]
        )
        block_device_samples = int(self.block_time * device_sr)
        total_context_s = self.extra_time + self.block_time + 2.0
        total_context_samples = int(total_context_s * device_sr)

        # Resamplers
        resample_to_16k = torchaudio.transforms.Resample(device_sr, 16000).to(
            self.device
        )
        resample_to_device = torchaudio.transforms.Resample(self.sr, device_sr).to(
            self.device
        )

        # Buffers
        input_buffer = np.zeros(total_context_samples, dtype=np.float32)
        output_queue = deque()
        lock = threading.Lock()

        print(f"\nDevice SR: {device_sr}Hz, Model SR: {self.sr}Hz")
        print(f"Block: {self.block_time}s, Context: {self.extra_time}s")
        print(f"Diffusion steps: {self.diffusion_steps}")
        print("\n" + "=" * 50)
        print("  REAL-TIME VOICE CONVERSION ACTIVE")
        print("  Press Ctrl+C to stop")
        print("=" * 50 + "\n")

        self.running = True
        prev_tail = None
        crossfade_samples = int(self.crossfade_time * self.sr)

        def audio_callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"  Audio: {status}")

            mono = indata[:, 0].copy()
            with lock:
                input_buffer[:-len(mono)] = input_buffer[len(mono) :]
                input_buffer[-len(mono) :] = mono

            if output_queue:
                chunk = output_queue.popleft()
                n = min(len(chunk), frames)
                outdata[:n, 0] = chunk[:n]
                outdata[n:, 0] = 0
            else:
                outdata[:] = 0

        def process_loop():
            nonlocal prev_tail

            while self.running:
                t0 = time.time()

                with lock:
                    raw = input_buffer.copy()

                # Energy gate
                block_start = len(raw) - block_device_samples
                rms = np.sqrt(np.mean(raw[block_start:] ** 2))
                if rms < 0.003:
                    output_queue.append(
                        np.zeros(block_device_samples, dtype=np.float32)
                    )
                    time.sleep(self.block_time * 0.8)
                    continue

                try:
                    raw_tensor = torch.from_numpy(raw).float().to(self.device)
                    input_16k = resample_to_16k(raw_tensor)

                    output = self._infer_block(input_16k)

                    # Crossfade
                    if prev_tail is not None and crossfade_samples > 0:
                        n = min(crossfade_samples, len(output), len(prev_tail))
                        fade_in = np.cos(np.linspace(np.pi / 2, 0, n)) ** 2
                        fade_out = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
                        output[:n] = output[:n] * fade_in + prev_tail[-n:] * fade_out
                    prev_tail = output.copy()

                    # Resample to device SR
                    out_tensor = torch.from_numpy(output).float().to(self.device)
                    out_device = resample_to_device(out_tensor).cpu().numpy()
                    output_queue.append(out_device)

                except Exception as e:
                    print(f"\n  Error: {e}")
                    output_queue.append(
                        np.zeros(block_device_samples, dtype=np.float32)
                    )

                elapsed = time.time() - t0
                rtf = elapsed / self.block_time
                sym = "OK" if rtf < 1.0 else "SLOW"
                print(
                    f"\r  RTF: {rtf:.2f} ({sym}) | "
                    f"Infer: {elapsed:.3f}s | "
                    f"Queue: {len(output_queue)}  ",
                    end="",
                    flush=True,
                )

                wait = self.block_time - elapsed
                if wait > 0:
                    time.sleep(wait * 0.5)

        try:
            with sd.Stream(
                samplerate=device_sr,
                blocksize=block_device_samples,
                device=(self.input_device, self.output_device),
                channels=1,
                dtype="float32",
                callback=audio_callback,
            ):
                proc = threading.Thread(target=process_loop, daemon=True)
                proc.start()
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
        "--reference",
        type=str,
        required=True,
        help="Path to reference speaker audio",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit"
    )
    parser.add_argument(
        "--input-device", type=int, default=None, help="Input device index"
    )
    parser.add_argument(
        "--output-device", type=int, default=None, help="Output device index"
    )
    parser.add_argument(
        "--block-time",
        type=float,
        default=0.5,
        help="Block size in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=10,
        help="Diffusion steps (default: 10)",
    )
    parser.add_argument(
        "--cfg-rate",
        type=float,
        default=0.7,
        help="CFG rate (default: 0.7)",
    )
    parser.add_argument(
        "--extra-time",
        type=float,
        default=2.5,
        help="Extra context in seconds (default: 2.5)",
    )

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
