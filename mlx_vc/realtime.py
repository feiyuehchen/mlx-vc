"""Real-time OpenVoice V2 session — reusable building block.

Provides a stateless `OpenVoiceSession` class for converting audio chunks
without any audio I/O. Use this from WebSocket handlers, real-time demos,
or anywhere that needs low-latency tone color conversion.

The underlying ToneColorConverter is loaded once at module-level singleton
(via `get_session()`) so subsequent uses are instant.
"""

import os
import sys
import threading
from typing import Optional

import numpy as np

# Module-level singleton + lock
_SESSION_LOCK = threading.Lock()
_SESSION: Optional["OpenVoiceSession"] = None


def get_session() -> "OpenVoiceSession":
    """Return a process-wide OpenVoiceSession singleton (lazy-loaded)."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = OpenVoiceSession()
            _SESSION.load()
    return _SESSION


class OpenVoiceSession:
    """OpenVoice V2 tone color converter, ready for chunk-by-chunk inference.

    Usage:
        session = OpenVoiceSession()
        session.load()
        session.set_reference("speaker.wav")  # extract target speaker embedding once
        out = session.convert_chunk(audio_np, sample_rate=16000)
    """

    def __init__(self):
        self.sr = 22050  # OpenVoice operates at 22050Hz internally
        self.converter = None
        self.tgt_se = None
        self.reference_path = None
        self.device = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load OpenVoice ToneColorConverter (download checkpoint if needed)."""
        if self._loaded:
            return

        import torch

        seed_vc_ref = os.path.join(os.path.dirname(__file__), "..", "..", "seed-vc-ref")
        seed_vc_ref = os.path.abspath(seed_vc_ref)
        if seed_vc_ref not in sys.path:
            sys.path.insert(0, seed_vc_ref)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        ckpt_dir = os.path.join(
            seed_vc_ref, "modules", "openvoice", "checkpoints_v2", "converter"
        )
        config_path = os.path.join(ckpt_dir, "config.json")
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")

        if not os.path.exists(ckpt_path) or not os.path.exists(config_path):
            import shutil

            from huggingface_hub import hf_hub_download

            os.makedirs(ckpt_dir, exist_ok=True)
            if not os.path.exists(ckpt_path):
                dl = hf_hub_download(
                    "myshell-ai/OpenVoiceV2", "converter/checkpoint.pth"
                )
                shutil.copy2(dl, ckpt_path)
            if not os.path.exists(config_path):
                dl = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/config.json")
                shutil.copy2(dl, config_path)

        from modules.openvoice.api import ToneColorConverter

        self.converter = ToneColorConverter(config_path, device=str(self.device))
        self.converter.load_ckpt(ckpt_path)
        self._loaded = True

    def set_reference(self, ref_path: str) -> None:
        """Pre-extract target speaker embedding from a reference WAV file.

        Mirrors OpenVoice's official se_extractor.get_se(): splits the
        reference into ~10s chunks, drops silent ones, averages embeddings.
        Much more robust than a single large segment for noisy/long refs.
        """
        if not self._loaded:
            self.load()

        if self.reference_path == ref_path and self.tgt_se is not None:
            return  # already loaded

        import librosa
        import torch

        ref_audio, _ = librosa.load(ref_path, sr=self.sr)

        # Split into ~10s chunks, filter low-energy
        target_len = int(10.0 * self.sr)
        if len(ref_audio) <= target_len:
            chunks = [ref_audio]
        else:
            n = max(1, len(ref_audio) // target_len)
            step = len(ref_audio) // n
            chunks = []
            for i in range(n):
                c = ref_audio[i * step : (i + 1) * step]
                rms = float(np.sqrt(np.mean(c**2)))
                if rms >= 0.008:
                    chunks.append(c)
            if not chunks:
                chunks = [ref_audio]  # fallback

        tensors = [torch.FloatTensor(c).to(self.device) for c in chunks]
        lens = [int(c.shape[0]) for c in chunks]

        with torch.no_grad():
            self.tgt_se = self.converter.extract_se(tensors, lens)
        self.reference_path = ref_path

    def convert_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
        tau: float = 0.3,
    ) -> np.ndarray:
        """Convert a single audio chunk to the loaded reference voice.

        Args:
            audio: float32 numpy array (mono)
            sample_rate: Input sample rate. If different from 22050, will resample.
            tau: Style control (0=more target, 1=more source)

        Returns:
            Converted float32 numpy array at 22050Hz.
        """
        if not self._loaded or self.tgt_se is None:
            raise RuntimeError("Call load() and set_reference() before convert_chunk()")

        import torch
        import torchaudio

        with self._lock:
            # Resample if needed
            if sample_rate is not None and sample_rate != self.sr:
                audio_t = torch.from_numpy(audio).float()
                resampler = torchaudio.transforms.Resample(sample_rate, self.sr)
                audio_t = resampler(audio_t)
                audio_np = audio_t.numpy()
            else:
                audio_np = audio

            audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0).to(self.device)
            audio_len = torch.LongTensor([len(audio_np)]).to(self.device)

            with torch.no_grad():
                src_se = self.converter.extract_se(
                    [audio_tensor.squeeze(0)], [audio_len.item()]
                )
                converted = self.converter.convert(
                    audio_tensor, audio_len, src_se, self.tgt_se, tau=tau
                )

            return converted.squeeze().cpu().numpy()

    @property
    def output_sr(self) -> int:
        return self.sr
