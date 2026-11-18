"""Check that a cortical adaptor gives back its input in bf16 (the vision block 17 case).

    python -m src.tools.check_adaptor_round_trip <weights dir>   # a released checkpoint

Loads vision block 17's W from the checkpoint, builds a bf16 stream with two massive-activation
dims (~8,500, as the vision tower carries), and reports the worst per-token cosine between the
adaptor's output and its input. Lossless means > 0.9999.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from src.models.qwen2_5_omni import CorticalAdaptor

KEY = "visual.cortical_adaptors.17.W"


def main(weights: str) -> int:
    W = None
    for f in sorted(Path(weights).glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            if KEY in fh.keys():
                W = fh.get_tensor(KEY)
    if W is None:
        raise SystemExit(f"{KEY} not found in {weights}")
    ad = CorticalAdaptor(SimpleNamespace(hidden_size=W.shape[0], cortical_dim=W.shape[1]))
    ad.W.data = W
    ad = ad.to(torch.bfloat16).eval()
    torch.manual_seed(0)
    X = torch.randn(64, W.shape[0])
    X[:4, 849] = X[:4, 1109] = 8500.0
    X = X.to(torch.bfloat16)
    with torch.no_grad():
        X_hat, _ = ad(X)
    cos = torch.nn.functional.cosine_similarity(X_hat.float(), X.float(), dim=1)
    print(f"condition number {torch.linalg.cond(W.float()):.0f}, worst per-token cosine {cos.min():.6f}")
    return 0 if cos.min() > 0.9999 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
