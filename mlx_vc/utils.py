"""Shared utilities for model loading and weight management."""

import json
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def get_model_path(
    path_or_repo: str,
    revision: Optional[str] = None,
) -> Path:
    """Resolve a model path or download from HuggingFace Hub.

    Args:
        path_or_repo: Local path or HuggingFace repo ID.
        revision: Optional git revision for HuggingFace downloads.

    Returns:
        Path to the local model directory.
    """
    model_path = Path(path_or_repo)
    if model_path.exists():
        return model_path

    model_path = Path(
        snapshot_download(
            repo_id=path_or_repo,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model"],
        )
    )
    return model_path


def load_config(model_path: Path) -> dict:
    """Load model config from a JSON file.

    Args:
        model_path: Path to the model directory.

    Returns:
        Config dictionary.
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_path}")

    with open(config_path, "r") as f:
        return json.load(f)
