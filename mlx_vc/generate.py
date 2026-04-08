"""Unified CLI for voice conversion."""

import argparse
import time
from pathlib import Path

import numpy as np


AVAILABLE_MODELS = {
    "cosyvoice": {
        "class": "mlx_vc.models.cosyvoice.CosyVoiceVC",
        "description": "CosyVoice3/Chatterbox zero-shot voice cloning (via mlx-audio)",
        "default_repo": "mlx-community/chatterbox-fp16",
    },
    "seed-vc": {
        "class": "mlx_vc.models.seed_vc.SeedVC",
        "description": "Seed-VC zero-shot voice conversion (Whisper + DiT + BigVGAN)",
        "default_repo": "Plachta/Seed-VC",
    },
    "openvoice": {
        "class": "mlx_vc.models.openvoice.OpenVoiceVC",
        "description": "OpenVoice V2 tone color conversion (VITS-based, multilingual)",
        "default_repo": "myshell-ai/OpenVoiceV2",
    },
    "rvc": {
        "class": "mlx_vc.models.rvc.RVCVC",
        "description": "RVC retrieval-based VC (requires per-speaker fine-tuned model)",
        "default_repo": None,
    },
}


def get_vc_model(model_type: str, model_repo: str = None, **kwargs):
    """Load a VC model by type.

    Args:
        model_type: One of the keys in AVAILABLE_MODELS.
        model_repo: HuggingFace repo or local path (overrides default).
        **kwargs: Additional arguments for model init.

    Returns:
        Initialized VC model instance.
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(AVAILABLE_MODELS.keys())}"
        )

    info = AVAILABLE_MODELS[model_type]
    module_path, class_name = info["class"].rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    repo = model_repo or info["default_repo"]
    return cls(model_name=repo, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="MLX Voice Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic voice conversion with CosyVoice
  mlx_vc.generate --model cosyvoice --text "Hello world" --ref_audio speaker.wav

  # Save to specific path
  mlx_vc.generate --model cosyvoice --text "Hello" --ref_audio ref.wav --output out.wav

  # Use a specific HuggingFace model
  mlx_vc.generate --model cosyvoice --model_repo mlx-community/chatterbox-fp16 \\
    --text "Hello" --ref_audio ref.wav
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cosyvoice",
        choices=list(AVAILABLE_MODELS.keys()),
        help="VC model type (default: cosyvoice)",
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default=None,
        help="HuggingFace repo ID or local path (overrides model default)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize with the target voice",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Path to reference audio (target speaker voice)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (default: 1.0)",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load model
    vc = get_vc_model(
        model_type=args.model,
        model_repo=args.model_repo,
        verbose=args.verbose,
    )

    # Run conversion
    print(f"Converting with {args.model}...")
    print(f"  Text: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print(f"  Reference: {args.ref_audio}")

    start = time.time()
    audio = vc.convert(
        text=args.text,
        ref_audio=args.ref_audio,
        temperature=args.temperature,
        speed=args.speed,
        lang_code=args.lang_code,
    )
    elapsed = time.time() - start

    duration = len(audio) / vc.sample_rate
    print(f"  Generated {duration:.2f}s audio in {elapsed:.2f}s")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from mlx_vc.audio_io import save_audio

    save_audio(str(output_path), audio, sample_rate=vc.sample_rate)
    print(f"  Saved to {output_path}")

    # Optionally play
    if args.play:
        try:
            import sounddevice as sd

            sd.play(audio, samplerate=vc.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"  Playback failed: {e}")


if __name__ == "__main__":
    main()
