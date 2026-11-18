**Vision block 17's adaptor is not lossless in bf16, and the error grows through blocks 18–31**

`CorticalAdaptor` relies on `X @ W @ pinv(W) == X`. In bf16 (the default for `load_topo_omni` on CUDA, and for training) this fails at vision block 17. Its released W has condition number 548 (other blocks 2–9): one singular value is 0.0136, while the others are ≥ 1.3. On tokens carrying the vision stream's massive activations (dims 849 and 1109, up to ~9,500), bf16 rounding of Z is amplified ~73× by `pinv(W)`.

Repro (released code and weights; any few photos):

```python
import os, sys; os.environ["TOPO_SHEET_LAYOUT"] = "tiled32"
import torch, torch.nn.functional as F
from PIL import Image
from src.core.model_loading import load_topo_omni

model, proc, _ = load_topo_omni(device="cuda")                 # bf16 by default
ad, xs = model.visual.cortical_adaptors[17], []
ad.register_forward_pre_hook(lambda m, a: xs.append(a[0]))
inp = proc.image_processor(images=[Image.open(p).convert("RGB").resize((224, 224)) for p in sys.argv[1:]],
                           return_tensors="pt")
with torch.no_grad():
    model.get_image_features(inp["pixel_values"].cuda(), inp["image_grid_thw"].cuda())
    X = xs[0]; X32, W = X.float(), ad.W.float()
    out = {"bf16": ad(X)[0].float(), "fp32": X32 @ W @ torch.linalg.pinv(W)}
hot, keep = (X32.abs() > 1000).any(1), X32.abs().amax(0) <= 1000
for k, Xh in out.items():
    c = F.cosine_similarity(Xh[:, keep], X32[:, keep], dim=1)
    print(k, "cos(X_hat, X), ordinary dims: massive tokens", c[hot].median().item(), "others", c[~hot].median().item())
```

On 4 ImageNet val images: bf16 gives 0.36 on the massive tokens and 0.998 on the others; fp32 gives 1.000 on both.

Measured on 64 ImageNet val images (torch 2.9.1, transformers 4.57.6):

| | block 17 | block 30 |
|---|---|---|
| vision stream, release vs fp32 round trip: median per-token cosine, massive dims excluded | 0.997 | 0.81 |
| same for plain Qwen2.5-Omni-3B, bf16 vs fp32 | 0.9997 | 0.999 |
| cortical units Z, release vs fp32 round trip: median per-unit r across images | 0.998 | 0.86 |

Doing only block 17's round trip in fp32 restores everything to bf16 noise level (≥ 0.99 at every block). With the fix, the vision stream matches plain Qwen.

**Possible fixes (no retraining):** do the round trip in float32, or return `X` unchanged. `X_hat == X` holds exactly when W has full row rank, so the reconstruction only adds rounding error.

**Not measured:** whether the published maps or behaviour change. The units from block 18 onward do change. We also don't know when W17 collapsed during training: nothing guards W's conditioning, and the intermediate tiled32 checkpoints referenced in the eval configs (`step0`, `ckpt292` task-only, `snap438`) would show it. Were the paper's features extracted in bf16?
