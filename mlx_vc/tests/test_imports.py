"""Basic import tests to verify package structure."""


def test_version():
    from mlx_vc import __version__

    assert __version__ == "0.1.0"


def test_generate_module():
    from mlx_vc.generate import AVAILABLE_MODELS, get_vc_model

    assert "cosyvoice" in AVAILABLE_MODELS


def test_audio_io():
    from mlx_vc.audio_io import load_audio, save_audio

    assert callable(load_audio)
    assert callable(save_audio)


def test_utils():
    from mlx_vc.utils import get_model_path, load_config

    assert callable(get_model_path)
    assert callable(load_config)
