#!/usr/bin/env python3
"""Real-time voice conversion demo.

Microphone -> Denoise -> OpenVoice V2 (tone color conversion) -> Speaker/BlackHole

OpenVoice V2 is used because it's extremely fast (RTF ~0.04),
enabling sub-300ms latency for real-time voice conversion.

Usage:
    python -m mlx_vc.demo.realtime_vc --reference speaker.wav
    python -m mlx_vc.demo.realtime_vc --reference speaker.wav --discord
    python -m mlx_vc.demo.realtime_vc --reference speaker.wav --list-devices
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
        bh = " [BlackHole]" if "blackhole" in d["name"].lower() else ""
        print(
            f"  [{i}] {d['name']} ({'/'.join(kind)}, "
            f"{int(d['default_samplerate'])}Hz){marker}{bh}"
        )
    print()


def find_blackhole_device():
    """Find BlackHole virtual audio device index."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if "blackhole" in d["name"].lower() and d["max_output_channels"] > 0:
            return i
    raise RuntimeError(
        "BlackHole not found. Install it first:\n"
        "  brew install blackhole-2ch\n"
        "Then restart your Mac."
    )


class RealtimeVC:
    """Real-time voice conversion using OpenVoice V2.

    OpenVoice is a VITS-based tone color converter. It's fast enough
    for real-time (RTF ~0.04) with sub-300ms block sizes.
    """

    def __init__(
        self,
        reference_path: str,
        block_time: float = 0.3,
        tau: float = 0.3,
        input_device=None,
        output_device=None,
    ):
        self.reference_path = reference_path
        self.block_time = block_time
        self.tau = tau
        self.input_device = input_device
        self.output_device = output_device
        self.monitor_device = None

        self.sr = 22050  # OpenVoice operates at 22050Hz
        self.converter = None
        self.tgt_se = None

    def _setup_openvoice(self):
        """Load OpenVoice V2 model and extract reference speaker embedding."""
        import torch

        seed_vc_ref = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "seed-vc-ref"
        )
        seed_vc_ref = os.path.abspath(seed_vc_ref)
        if seed_vc_ref not in sys.path:
            sys.path.insert(0, seed_vc_ref)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Checkpoint paths
        ckpt_dir = os.path.join(
            seed_vc_ref, "modules", "openvoice", "checkpoints_v2", "converter"
        )
        config_path = os.path.join(ckpt_dir, "config.json")
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")

        # Download if needed
        if not os.path.exists(ckpt_path):
            from huggingface_hub import hf_hub_download
            import shutil

            print("Downloading OpenVoice V2 checkpoint...")
            dl = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth")
            os.makedirs(ckpt_dir, exist_ok=True)
            shutil.copy2(dl, ckpt_path)
        if not os.path.exists(config_path):
            from huggingface_hub import hf_hub_download
            import shutil

            dl = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/config.json")
            os.makedirs(ckpt_dir, exist_ok=True)
            shutil.copy2(dl, config_path)

        print("Loading OpenVoice V2...")
        from modules.openvoice.api import ToneColorConverter

        self.converter = ToneColorConverter(config_path, device=str(self.device))
        self.converter.load_ckpt(ckpt_path)

        # Pre-extract reference speaker embedding (done once)
        import librosa

        ref_audio, _ = librosa.load(self.reference_path, sr=self.sr)
        ref_tensor = torch.FloatTensor(ref_audio).to(self.device)
        ref_len = len(ref_audio)

        with torch.no_grad():
            self.tgt_se = self.converter.extract_se(
                [ref_tensor], [ref_len]
            )

        print(f"Reference: {self.reference_path} ({len(ref_audio)/self.sr:.1f}s)")
        print(f"Target SE shape: {self.tgt_se.shape}")

    def _convert_chunk(self, audio_np: np.ndarray) -> np.ndarray:
        """Convert a single chunk of audio using OpenVoice.

        Args:
            audio_np: float32 numpy array at self.sr

        Returns:
            Converted audio as float32 numpy array
        """
        import torch

        audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0).to(self.device)
        audio_len = torch.LongTensor([len(audio_np)]).to(self.device)

        with torch.no_grad():
            # Extract source speaker embedding from this chunk
            src_se = self.converter.extract_se(
                [audio_tensor.squeeze(0)], [audio_len.item()]
            )
            # Convert tone color
            converted = self.converter.convert(
                audio_tensor, audio_len, src_se, self.tgt_se, tau=self.tau
            )

        return converted.squeeze().cpu().numpy()

    def run(self):
        """Start the real-time VC loop."""
        import torch
        import torchaudio

        self._setup_openvoice()

        # Device sample rates
        in_sr = int(
            sd.query_devices(self.input_device, "input")["default_samplerate"]
        )
        out_sr = int(
            sd.query_devices(self.output_device, "output")["default_samplerate"]
        )
        mon_sr = None
        if self.monitor_device is not None:
            mon_sr = int(
                sd.query_devices(self.monitor_device, "output")["default_samplerate"]
            )

        in_block = int(self.block_time * in_sr)
        out_block = int(self.block_time * out_sr)

        # Resamplers
        resample_in_to_model = torchaudio.transforms.Resample(in_sr, self.sr)
        resample_model_to_out = torchaudio.transforms.Resample(self.sr, out_sr)
        resample_model_to_mon = (
            torchaudio.transforms.Resample(self.sr, mon_sr) if mon_sr else None
        )

        # Buffers
        input_buffer = np.zeros(int(self.block_time * 2 * in_sr), dtype=np.float32)
        output_queue = deque(maxlen=3)
        monitor_queue = deque(maxlen=3)
        lock = threading.Lock()

        # Noise profiling
        print("  Profiling noise (stay quiet 0.5s)...")
        in_stream_profile = sd.InputStream(
            samplerate=in_sr, device=self.input_device, channels=1, dtype="float32"
        )
        in_stream_profile.start()
        time.sleep(0.6)
        noise_data, _ = in_stream_profile.read(int(0.5 * in_sr))
        in_stream_profile.stop()
        in_stream_profile.close()
        noise_profile = noise_data[:, 0].copy()
        noise_rms = np.sqrt(np.mean(noise_profile ** 2))
        gate_threshold = max(noise_rms * 4.0, 0.003)
        print(f"  Noise floor: {noise_rms:.5f}, gate: {gate_threshold:.5f}")

        print(f"\n  Input:  device {self.input_device}, {in_sr}Hz")
        print(f"  Output: device {self.output_device}, {out_sr}Hz")
        if mon_sr:
            print(f"  Monitor: device {self.monitor_device}, {mon_sr}Hz")
        print(f"  Model: OpenVoice V2 @ {self.sr}Hz, block={self.block_time}s")
        print("\n" + "=" * 50)
        print("  REAL-TIME VOICE CONVERSION (OpenVoice V2)")
        print("  Press Ctrl+C to stop")
        print("=" * 50 + "\n")

        self.running = True

        def in_callback(indata, frames, time_info, status):
            mono = indata[:, 0].copy()
            with lock:
                input_buffer[:-len(mono)] = input_buffer[len(mono):]
                input_buffer[-len(mono):] = mono

        def out_callback(outdata, frames, time_info, status):
            if output_queue:
                chunk = output_queue.popleft()
                n = min(len(chunk), frames)
                outdata[:n, 0] = chunk[:n]
                outdata[n:, 0] = 0
            else:
                outdata[:] = 0

        def mon_callback(outdata, frames, time_info, status):
            if monitor_queue:
                chunk = monitor_queue.popleft()
                n = min(len(chunk), frames)
                outdata[:n, 0] = chunk[:n]
                outdata[n:, 0] = 0
            else:
                outdata[:] = 0

        def process_loop():
            import noisereduce as nr

            while self.running:
                t0 = time.time()

                with lock:
                    raw = input_buffer[-in_block:].copy()

                # Energy gate
                rms = np.sqrt(np.mean(raw ** 2))
                if rms < gate_threshold:
                    output_queue.append(np.zeros(out_block, dtype=np.float32))
                    if mon_sr:
                        monitor_queue.append(
                            np.zeros(int(self.block_time * mon_sr), dtype=np.float32)
                        )
                    time.sleep(self.block_time * 0.5)
                    continue

                try:
                    # Denoise
                    raw_clean = nr.reduce_noise(
                        y=raw, sr=in_sr, y_noise=noise_profile,
                        stationary=True, prop_decrease=0.8,
                    )

                    # Resample to model SR
                    raw_tensor = torch.from_numpy(raw_clean).float()
                    model_audio = resample_in_to_model(raw_tensor).numpy()

                    # Voice conversion (OpenVoice — very fast)
                    converted = self._convert_chunk(model_audio)

                    # Resample to output SR
                    conv_tensor = torch.from_numpy(converted).float()
                    out_audio = resample_model_to_out(conv_tensor).numpy()
                    output_queue.append(out_audio)

                    if resample_model_to_mon is not None:
                        mon_audio = resample_model_to_mon(conv_tensor).numpy()
                        monitor_queue.append(mon_audio)

                except Exception as e:
                    print(f"\n  Error: {e}")
                    output_queue.append(np.zeros(out_block, dtype=np.float32))

                elapsed = time.time() - t0
                latency_ms = elapsed * 1000
                print(
                    f"\r  Latency: {latency_ms:.0f}ms | "
                    f"Queue: {len(output_queue)}  ",
                    end="",
                    flush=True,
                )

                # Wait for next block
                wait = self.block_time - elapsed
                if wait > 0:
                    time.sleep(wait)

        # Start streams
        streams = []
        try:
            streams.append(
                sd.InputStream(
                    samplerate=in_sr, blocksize=in_block,
                    device=self.input_device, channels=1,
                    dtype="float32", callback=in_callback,
                )
            )
            streams.append(
                sd.OutputStream(
                    samplerate=out_sr, blocksize=out_block,
                    device=self.output_device, channels=1,
                    dtype="float32", callback=out_callback,
                )
            )
            if mon_sr:
                streams.append(
                    sd.OutputStream(
                        samplerate=mon_sr,
                        blocksize=int(self.block_time * mon_sr),
                        device=self.monitor_device, channels=1,
                        dtype="float32", callback=mon_callback,
                    )
                )

            for s in streams:
                s.start()

            proc = threading.Thread(target=process_loop, daemon=True)
            proc.start()

            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.running = False
            for s in streams:
                s.stop()
                s.close()
            time.sleep(0.3)
            print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time voice conversion (OpenVoice V2)"
    )
    parser.add_argument(
        "--reference", type=str, required=True,
        help="Path to target speaker audio",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit",
    )
    parser.add_argument(
        "--input-device", type=int, default=None, help="Input device index",
    )
    parser.add_argument(
        "--output-device", type=int, default=None, help="Output device index",
    )
    parser.add_argument(
        "--block-time", type=float, default=0.3,
        help="Block size in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.3,
        help="Style control: 0=more target, 1=more source (default: 0.3)",
    )
    parser.add_argument(
        "--discord", action="store_true",
        help="Discord mode: output to BlackHole, monitor to headphones",
    )
    parser.add_argument(
        "--monitor", type=int, default=None,
        help="Monitor device index (hear yourself)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    output_device = args.output_device
    monitor_device = args.monitor
    if args.discord:
        try:
            bh = find_blackhole_device()
            output_device = bh
            print(f"Discord mode: output -> BlackHole (device {bh})")
            if monitor_device is None:
                default_out = sd.default.device[1]
                if default_out != bh:
                    monitor_device = default_out
                    print(f"Monitor: device {default_out}")
        except RuntimeError as e:
            print(f"Error: {e}")
            return

    vc = RealtimeVC(
        reference_path=args.reference,
        block_time=args.block_time,
        tau=args.tau,
        input_device=args.input_device,
        output_device=output_device,
    )
    vc.monitor_device = monitor_device
    vc.run()


if __name__ == "__main__":
    main()
