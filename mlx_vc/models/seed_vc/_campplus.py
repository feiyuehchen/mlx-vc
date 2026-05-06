"""CAMPPlus speaker encoder loader.

Loads the CAMPPlus model from the original checkpoint.
"""

import sys
from pathlib import Path


def load_campplus(device=None):
    """Load CAMPPlus model from Seed-VC reference.

    Falls back to a simple speaker encoder if the full model isn't available.
    """
    import torch

    # Try to load from seed-vc-ref
    seed_vc_path = Path(__file__).parent.parent.parent.parent.parent / "seed-vc-ref"
    if seed_vc_path.exists():
        sys.path.insert(0, str(seed_vc_path))
        try:
            from modules.campplus.DTDNN import CAMPPlus

            ckpt_path = seed_vc_path / "campplus_cn_common.bin"
            if not ckpt_path.exists():
                from huggingface_hub import hf_hub_download

                ckpt_path = hf_hub_download("funasr/campplus", "campplus_cn_common.bin")

            model = CAMPPlus(feat_dim=80, embedding_size=192)
            model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
            model.eval()
            if device is not None:
                model = model.to(device)
            return model
        except Exception as e:
            print(f"Warning: Could not load full CAMPPlus model: {e}")
        finally:
            if str(seed_vc_path) in sys.path:
                sys.path.remove(str(seed_vc_path))

    # Fallback: simple stats pooling
    print("Using simplified speaker encoder (stats pooling)")
    return _SimpleSpeakerEncoder(device=device)


class _SimpleSpeakerEncoder:
    """Fallback speaker encoder using mean/std pooling."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, x):
        import torch

        mean = x.mean(dim=1)
        std = x.std(dim=1)
        return torch.cat([mean, std], dim=-1)[:, :192]
