"""Convert Seed-VC PyTorch checkpoints to MLX safetensors format."""

import argparse
import re
from pathlib import Path

import mlx.core as mx
import numpy as np


def remap_key(key: str) -> str:
    """Remap PyTorch state dict key to MLX model key.

    Handles the model dict structure: model['cfm'], model['length_regulator'], etc.
    """
    # cfm.estimator.* -> cfm.estimator.*
    # The main transformer is inside cfm.estimator.transformer

    # AdaptiveLayerNorm: project_layer stays, norm.weight stays
    # RMSNorm: weight stays

    # Conv1d weights: PyTorch [O, I, K] -> MLX [O, K, I]
    # (handled separately in convert_weights)

    return key


def convert_torch_to_mlx(checkpoint_path: str, output_path: str):
    """Convert a Seed-VC PyTorch checkpoint to MLX format.

    Args:
        checkpoint_path: Path to .pth file
        output_path: Path to output directory
    """
    import torch

    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Seed-VC checkpoint structure: {'net': {component: state_dict, ...}, ...}
    # Or direct state dict
    if "net" in state_dict:
        state_dict = state_dict["net"]

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each component
    for component_name in ["cfm", "length_regulator"]:
        if component_name not in state_dict:
            print(f"  Skipping {component_name} (not found)")
            continue

        comp_dict = state_dict[component_name]
        mlx_dict = {}

        for key, value in comp_dict.items():
            np_val = value.numpy().astype(np.float32) if hasattr(value, "numpy") else np.array(value, dtype=np.float32)

            # Transpose Conv1d weights: PyTorch [O, I, K] -> MLX [O, K, I]
            if "conv" in key and "weight" in key and np_val.ndim == 3:
                np_val = np.transpose(np_val, (0, 2, 1))
                print(f"  Transposed conv weight: {component_name}.{key} {np_val.shape}")

            mlx_dict[key] = mx.array(np_val)

        # Save as safetensors
        out_file = output_dir / f"{component_name}.safetensors"
        mx.save_safetensors(str(out_file), mlx_dict)
        print(f"  Saved {component_name}: {len(mlx_dict)} tensors -> {out_file}")

    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert Seed-VC weights to MLX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    convert_torch_to_mlx(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
