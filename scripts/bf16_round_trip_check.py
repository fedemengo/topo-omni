"""Is the bf16 adaptor round trip a real bug on real images? Evidence for docs/bf16-round-trip/.

    IMAGENET_SHARD=<ILSVRC/imagenet-1k on HF: data/validation-00000-of-00014.parquet; first 64 images used> \
        CUDA_VISIBLE_DEVICES=<gpu> python scripts/bf16_round_trip_check.py

TOPO_WEIGHTS: the release (default: the HF repo, or $TOPO_OMNI_MODEL); plain Qwen comes from the HF cache.

1. block 17's real input: are there massive activations, and does the bf16 round trip lose X?
2. the vision stream after every block: the release (bf16 round trips) vs fp32 round trips at
   every block, and vs fp32 at block 17 only; reference: plain Qwen2.5-Omni-3B, bf16 vs fp32.
3. the cortical units Z per image: release vs fix, against split-half measurement noise.
Writes outputs/bf16_check/{results.json, propagation.png, cortical_units.png}. `--plot-only` redraws the figure.
"""
import io, json, os, sys, types
from pathlib import Path

os.environ["TOPO_SHEET_LAYOUT"] = "tiled32"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch, torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
from src.core.model_loading import load_topo_omni

WEIGHTS = os.environ.get("TOPO_WEIGHTS", os.environ.get("TOPO_OMNI_MODEL", "epfl-neuroai/topo-omni"))
SHARD = os.environ.get("IMAGENET_SHARD")
N_IMAGES, MASSIVE = 64, 1000.0
OUT = ROOT / "outputs" / "bf16_check"
bf16, fp32 = torch.bfloat16, torch.float32


def images():
    """First N_IMAGES of the shard (classes are mixed), shorter side 256, centre crop 224."""
    out = []
    for rec in pq.read_table(SHARD).slice(0, N_IMAGES).to_pylist():
        im = Image.open(io.BytesIO(rec["image"]["bytes"])).convert("RGB")
        s = 256 / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        l, t = (im.width - 224) // 2, (im.height - 224) // 2
        out.append(im.crop((l, t, l + 224, t + 224)))
    return out


def pinv32(W):
    with torch.autocast(device_type=W.device.type, enabled=False):
        return torch.linalg.pinv(W.float())


def release_forward(self, X):   # CorticalAdaptor.forward on main (600c685): bf16 throughout
    Z = self.z_hook(X @ self.W)
    return Z @ self._pinv_bf16, Z


def fp32_forward(self, X):      # CorticalAdaptor.forward of fd0f9e1
    Z = self.z_hook(X.float() @ self.W.float())
    return (Z @ self._pinv_fp32).to(X.dtype), Z.to(X.dtype)


def bypass_forward(self, X):    # plain Qwen: no adaptor in the stream
    return X, X.new_zeros(X.shape[0], self.cortical_dim)


@torch.no_grad()
def run(imgs, qwen, dtype, fp32_blocks=()):
    """Stream after every vision block (float32, CPU), block 17's adaptor input, and every block's
    cortical units Z averaged over each image's tokens: (32, images, 4608)."""
    model, proc, _ = load_topo_omni(model=None if qwen else WEIGHTS, baseline=qwen, device="cuda", dtype=dtype)
    ads, out, x17, z = model.visual.cortical_adaptors, [[] for _ in range(32)], [], [[] for _ in range(32)]
    n_tok = []
    for b, ad in enumerate(ads):
        P = pinv32(ad.W)
        ad._pinv_bf16, ad._pinv_fp32 = P.to(ad.W.dtype), P
        f = bypass_forward if qwen else fp32_forward if b in fp32_blocks else release_forward
        ad.forward = types.MethodType(f, ad)
        ad.register_forward_hook(lambda m, a, o, b=b: out[b].append(o[0].float().cpu()))
        ad.z_hook.register_forward_hook(lambda m, a, o, b=b: z[b].extend(t.mean(0).cpu() for t in o.float().split(n_tok[0])))
    ads[17].register_forward_pre_hook(lambda m, a: x17.append(a[0].float().cpu()))
    for i in range(0, len(imgs), 16):
        inp = proc.image_processor(images=imgs[i:i + 16], return_tensors="pt")
        n_tok[:] = [inp["image_grid_thw"].prod(1).tolist()]
        model.get_image_features(inp["pixel_values"].cuda(), inp["image_grid_thw"].cuda())
    W17 = ads[17].W.float().cpu()
    del model; torch.cuda.empty_cache()
    return torch.stack([torch.cat(o) for o in out]), torch.cat(x17), W17, (torch.stack([torch.stack(v) for v in z]) if z[0] else None)   # plain Qwen: no units


def corr_upper(Z):
    """Unit-unit correlation matrix over images, upper triangle (units with no variance -> nan)."""
    C = torch.corrcoef(Z.T.double())
    i, j = torch.triu_indices(*C.shape, 1)
    return C[i, j]


def pearson(a, b):
    ok = a.isfinite() & b.isfinite()
    return torch.corrcoef(torch.stack([a[ok], b[ok]]))[0, 1].item()


def unit_agreement(Za, Zb):
    """Za, Zb: (images, units). Agreement of unit-unit correlation structure, and per unit the
    correlation of its responses across images."""
    per_unit = F.cosine_similarity(Za - Za.mean(0), Zb - Zb.mean(0), dim=0)   # = Pearson r per unit
    per_unit = per_unit[(Za.std(0) > 0) & (Zb.std(0) > 0)]          # drop dead units
    return dict(unit_unit_corr_agreement=pearson(corr_upper(Za), corr_upper(Zb)), per_unit_r=stats(per_unit))


def stats(c):
    return dict(median=c.median().item(), p5=c.quantile(0.05).item(), min=c.min().item())


LABELS = {
    "release_vs_fixed": "Topo-Omni release (bf16 round trips) vs fp32 round trips at every block",
    "fixed17_vs_fixed": "fp32 round trip at block 17 only vs fp32 round trips at every block",
    "qwen_bf16_vs_fp32": "plain Qwen2.5-Omni-3B, no adaptors: bf16 vs fp32 (harmless-bf16 reference)",
    "fixed_vs_qwen_bf16": "fp32 round trips at every block vs plain Qwen2.5-Omni-3B, both bf16",
}


def plot(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, key, zoom in zip(axs, ("all", "no_massive", "no_massive"), (False, False, True)):
        for (name, rows), color in zip(res["propagation"].items(), ("C3", "C0", "C2", "C1")):
            ax.plot([r[key]["median"] for r in rows], color=color, marker="o", ms=3, label=f"{LABELS[name]}: median")
            ax.plot([r[key]["p5"] for r in rows], color=color, ls="--", label=f"{LABELS[name]}: 5th percentile")
        ax.axvline(17, color="grey", lw=0.5)
        dims = "all dims" if key == "all" else f"without massive dims {res['massive_dims_anywhere']}"
        if zoom:
            ax.set_ylim(0.97, 1.0005); dims = "zoom on the middle panel"
        ax.set(title=f"per-token cosine, {dims}", xlabel="vision block (stream after the block's adaptor)", ylabel="cosine")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(f"Topo-Omni vision stream: bf16 vs fp32 adaptor round trip, {res['n_images']} ImageNet val images")
    fig.tight_layout(rect=(0, 0.15, 1, 1)); fig.savefig(OUT / "propagation.png", dpi=120)

    cu = res["cortical_units"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axs[0]
    ax.plot([r["unit_unit_corr_agreement"] for r in cu["release_vs_fixed"]], "o-", ms=3, color="C3",
            label="release vs fp32 round trips at every block (same images)")
    ax.plot([r["unit_unit_corr_agreement"] for r in cu["fixed17_vs_fixed"]], "o-", ms=3, color="C0",
            label="fp32 at block 17 only vs fp32 at every block (same images)")
    ax.plot(cu["split_half_fixed"], ls="--", color="grey",
            label=f"image-sampling noise: fp32 round trips, images 1-{res['n_images']//2} vs the other half")
    ax.plot(cu["split_half_release"], ls=":", color="grey",
            label=f"image-sampling noise: release, images 1-{res['n_images']//2} vs the other half")
    ax.set(title="unit-unit correlation structure: agreement (Pearson r of upper triangles)",
           xlabel="vision block", ylabel="agreement (r)")
    ax.legend(fontsize=8)
    ax = axs[1]
    for key, color, lab in (("release_vs_fixed", "C3", "release vs fp32 round trips at every block"),
                            ("fixed17_vs_fixed", "C0", "fp32 at block 17 only vs fp32 at every block")):
        ax.plot([r["per_unit_r"]["median"] for r in cu[key]], "o-", ms=3, color=color, label=f"{lab}: median unit")
        ax.plot([r["per_unit_r"]["p5"] for r in cu[key]], "--", color=color, label=f"{lab}: 5th percentile unit")
    ax.set(title="per unit: correlation of its responses across images", xlabel="vision block", ylabel="Pearson r")
    ax.legend(fontsize=8)
    for ax in axs: ax.axvline(17, color="grey", lw=0.5)
    fig.suptitle(f"Topo-Omni vision cortical units (4608 per block, mean over tokens), {res['n_images']} ImageNet val images")
    fig.tight_layout(); fig.savefig(OUT / "cortical_units.png", dpi=120)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = images()
    release, X, W, Zrel = run(imgs, qwen=False, dtype=bf16)
    fixed, _, _, Zfix = run(imgs, qwen=False, dtype=bf16, fp32_blocks=range(32))
    fixed17, _, _, Zfix17 = run(imgs, qwen=False, dtype=bf16, fp32_blocks=(17,))
    qwen_bf16, *_ = run(imgs, qwen=True, dtype=bf16)
    qwen_fp32, *_ = run(imgs, qwen=True, dtype=fp32)

    # 1. block 17 on its real input (captured from the release)
    mx = X.abs().amax(0)
    massive = (mx > MASSIVE).nonzero().flatten()
    keep = mx <= MASSIVE
    hot = (X.abs() > MASSIVE).any(1)
    Xb = ((X.to(bf16) @ W.to(bf16)) @ pinv32(W).to(bf16)).float()
    Xf = (X @ W) @ pinv32(W)
    blk17 = {}
    for name, Xh in (("bf16", Xb), ("fp32", Xf)):
        c_all, c_rest = F.cosine_similarity(Xh, X, dim=1), F.cosine_similarity(Xh[:, keep], X[:, keep], dim=1)
        err = (Xh[:, keep] - X[:, keep]).norm(dim=1) / X[:, keep].norm(dim=1)
        blk17[name] = {sub: dict(cos_all=stats(c_all[m]), cos_no_massive=stats(c_rest[m]), rel_err_no_massive=stats(err[m]))
                       for sub, m in (("all_tokens", torch.ones_like(hot)), ("massive_tokens", hot), ("other_tokens", ~hot))
                       if m.any()}
    top = mx.topk(5)
    res = dict(n_images=len(imgs), n_tokens=X.shape[0], torch=torch.__version__,
               block17_input=dict(top5_dims=top.indices.tolist(), top5_max_abs=top.values.tolist(),
                                  massive_dims=massive.tolist(), share_tokens_massive=hot.float().mean().item()),
               block17_round_trip=blk17)

    # 2. propagation: per block, per-token cosine; "no massive" drops dims > MASSIVE anywhere in `fixed`
    keep_all = fixed.abs().amax((0, 1)) <= MASSIVE
    res["massive_dims_anywhere"] = (~keep_all).nonzero().flatten().tolist()
    prop = {}
    for name, (s, t) in {"release_vs_fixed": (release, fixed), "fixed17_vs_fixed": (fixed17, fixed),
                          "qwen_bf16_vs_fp32": (qwen_bf16, qwen_fp32)}.items():
        prop[name] = [dict(block=b, all=stats(F.cosine_similarity(s[b], t[b], dim=1)),
                           no_massive=stats(F.cosine_similarity(s[b][:, keep_all], t[b][:, keep_all], dim=1)))
                      for b in range(32)]
    res["propagation"] = prop
    prop["fixed_vs_qwen_bf16"] = [dict(block=b, all=stats(F.cosine_similarity(fixed[b], qwen_bf16[b], dim=1)),
                                       no_massive=stats(F.cosine_similarity(fixed[b][:, keep_all], qwen_bf16[b][:, keep_all], dim=1)))
                                  for b in range(32)]

    # 3. the cortical units (what the paper's maps are built from), per image, per block
    h = len(imgs) // 2
    res["cortical_units"] = dict(
        release_vs_fixed=[unit_agreement(Zrel[b], Zfix[b]) for b in range(32)],
        fixed17_vs_fixed=[unit_agreement(Zfix17[b], Zfix[b]) for b in range(32)],
        # measurement noise: same model, two disjoint halves of the images
        split_half_fixed=[pearson(corr_upper(Zfix[b][:h]), corr_upper(Zfix[b][h:])) for b in range(32)],
        split_half_release=[pearson(corr_upper(Zrel[b][:h]), corr_upper(Zrel[b][h:])) for b in range(32)],
    )
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    plot(res)
    print(json.dumps({k: v for k, v in res.items() if k != "propagation"}, indent=1))
    for name, rows in prop.items():
        print(name, "block 30 median cos all/no-massive:", rows[30]["all"]["median"], rows[30]["no_massive"]["median"])
    for name, rows in res["cortical_units"].items():
        print(name, "block 30:", rows[30])


if __name__ == "__main__":
    if "--plot-only" in sys.argv:   # redraw from results.json, no GPU
        plot(json.loads((OUT / "results.json").read_text()))
    else:
        main()
