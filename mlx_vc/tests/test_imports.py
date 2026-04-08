"""Basic import tests to verify package structure."""


def test_version():
    from mlx_vc import __version__

    assert __version__ == "0.1.0"


def test_generate_module():
    from mlx_vc.generate import AVAILABLE_MODELS, get_vc_model

    assert "cosyvoice" in AVAILABLE_MODELS
    assert "seed-vc" in AVAILABLE_MODELS
    assert "openvoice" in AVAILABLE_MODELS
    assert "rvc" in AVAILABLE_MODELS


def test_backend_registry():
    from mlx_vc.backend import BACKENDS

    assert "seed-vc" in BACKENDS
    assert "openvoice" in BACKENDS


def test_openvoice_import():
    from mlx_vc.models.openvoice import OpenVoiceVC

    assert callable(OpenVoiceVC)


def test_rvc_import():
    from mlx_vc.models.rvc import RVCVC

    assert callable(RVCVC)


def test_knnvc_import():
    from mlx_vc.models.knn_vc import KnnVC

    assert callable(KnnVC)


def test_meanvc_import():
    from mlx_vc.models.meanvc import MeanVC

    assert callable(MeanVC)


def test_all_backends():
    from mlx_vc.backend import BACKENDS

    assert len(BACKENDS) >= 3
    for name in ["seed-vc", "openvoice", "knn-vc", "meanvc"]:
        assert name in BACKENDS


def test_audio_io():
    from mlx_vc.audio_io import load_audio, save_audio

    assert callable(load_audio)
    assert callable(save_audio)


def test_utils():
    from mlx_vc.utils import get_model_path, load_config

    assert callable(get_model_path)
    assert callable(load_config)
