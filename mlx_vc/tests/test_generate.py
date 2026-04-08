"""Tests for the unified CLI / generate module."""

from mlx_vc.generate import AVAILABLE_MODELS, get_vc_model


def test_all_models_registered():
    """All expected model types should be in the registry."""
    expected = ["cosyvoice", "seed-vc", "openvoice", "rvc", "knn-vc", "meanvc"]
    for name in expected:
        assert name in AVAILABLE_MODELS, f"Missing model: {name}"


def test_model_entries_have_required_fields():
    """Each model entry must have class, description."""
    for name, info in AVAILABLE_MODELS.items():
        assert "class" in info, f"{name} missing 'class'"
        assert "description" in info, f"{name} missing 'description'"


def test_get_vc_model_unknown_raises():
    """Loading an unknown model type should raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown model type"):
        get_vc_model("nonexistent-model")


def test_model_wrappers_have_convert_method():
    """Each model wrapper class should have a convert() method."""
    from mlx_vc.models.cosyvoice import CosyVoiceVC
    from mlx_vc.models.seed_vc import SeedVC
    from mlx_vc.models.openvoice import OpenVoiceVC
    from mlx_vc.models.knn_vc import KnnVC
    from mlx_vc.models.meanvc import MeanVC
    from mlx_vc.models.rvc import RVCVC

    for cls in [CosyVoiceVC, SeedVC, OpenVoiceVC, KnnVC, MeanVC, RVCVC]:
        assert hasattr(cls, "convert"), f"{cls.__name__} missing convert()"


def test_model_wrappers_have_model_info():
    """Each model wrapper should expose model_info property."""
    from mlx_vc.models.seed_vc import SeedVC
    from mlx_vc.models.openvoice import OpenVoiceVC
    from mlx_vc.models.knn_vc import KnnVC
    from mlx_vc.models.meanvc import MeanVC

    for cls in [SeedVC, OpenVoiceVC, KnnVC, MeanVC]:
        instance = cls(verbose=False)
        info = instance.model_info
        assert isinstance(info, dict)
        assert "name" in info
        assert "type" in info
        assert "sr" in info
