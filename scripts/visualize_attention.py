"""Attention-rollout visualisation for the SMT (vit_model_img_ts) model.

Rebuilds the model with its attention hook enabled, runs one image+time-series
sample, rolls out the attention across all transformer layers, and overlays the
CLS-to-image-patch attention on the input sky image.

    python scripts/visualize_attention.py --config configs/smt.yaml \
        --ckpt outputs/smt.pt --index 0 --out outputs/attention.png

If no --ckpt is given the model runs with random weights (the map is then just
a sanity check of the plumbing, not a trained explanation).
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rollout(attentions, discard_ratio=0.0):
    """Abnar & Zuidema (2020) attention rollout.

    attentions: list of [B, heads, N, N] tensors (one per layer).
    Returns the [N, N] rolled-out attention for the first sample.
    """
    import torch
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attn in attentions:
            a = attn[0].mean(0)                      # average over heads -> [N, N]
            a = a + torch.eye(a.size(-1))            # add residual/identity
            a = a / a.sum(dim=-1, keepdim=True)      # row-normalise
            result = a @ result
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smt.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--index", type=int, default=0, help="test-sample index to explain")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="outputs/attention.png")
    args_cli = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smt.config import load_config
    from smt.preprocess import ensure_ghi_index
    from smt.data_provider import DataGenerator_ViLT
    from smt.models.vit_model_collection import vit_model_img_ts, Layer_scale_init_Block

    args = load_config(args_cli.config, {"data_root": args_cli.data_root})
    args.ts_data = ensure_ghi_index(args.ts_data, args.latitude, args.longitude, args.altitude)
    assert args.model_skeleton == "vit_model_img_ts", \
        "attention rollout is defined for the SMT model (vit_model_img_ts)"
    device = torch.device("cpu")

    ds = DataGenerator_ViLT(
        image=args.image, image_time=args.image_time, ts_data=args.ts_data,
        horizon=args.horizon, window=args.window, flag="test",
        data_flag=args.data_flag, indices=args.indices, special_test=args.special_test,
        image_token=True, ts_token=True)

    img_size = (ds.pixel_values.shape[2], ds.pixel_values.shape[3])
    ts_size = (ds.rawdat.shape[1], ds.rawdat.shape[2])
    model = vit_model_img_ts(
        img_size=img_size, patch_size=16, in_chans=ds.pixel_values.shape[1],
        ts_shape=ts_size, embed_dim=args.embed_dim, depth=args.depth_transformer,
        num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True,
        attn_drop_rate=args.attn_drop, drop_rate=args.drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block, patch_mode=args.patch_mode,
        attention_visualization=True).to(device)

    if args_cli.ckpt:
        model.load_state_dict(torch.load(args_cli.ckpt, map_location=device))
        print(f"loaded {args_cli.ckpt}")
    else:
        print("WARNING: no --ckpt, using random weights")
    model.eval()

    # one sample: raw uint8 image for display + normalised tensors for the model
    t = ds.times[args_cli.index]
    img_idx = int(np.where(ds.image_times == t)[0][0])
    raw = ds.pixel_values[img_idx].transpose(1, 2, 0)             # C,H,W -> H,W,C uint8
    x = torch.from_numpy(ds.pixel_values[img_idx][None].astype(np.float32) / 255.0)
    ts = torch.from_numpy(ds.rawdat[args_cli.index][None]).float()

    with torch.no_grad():
        _, attn = model(x, ts)

    roll = rollout(attn)                                          # [N, N]
    n_patches = model.patch_embed.num_patches
    gh, gw = model.patch_embed.grid_size
    cls_to_patch = roll[0, 1:1 + n_patches].reshape(gh, gw).numpy()
    cls_to_patch = (cls_to_patch - cls_to_patch.min()) / (np.ptp(cls_to_patch) + 1e-8)

    # upsample the patch grid to the image size
    heat = np.kron(cls_to_patch, np.ones((img_size[0] // gh, img_size[1] // gw)))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(raw); ax[0].set_title(f"sky image\n{str(t)}"); ax[0].axis("off")
    ax[1].imshow(cls_to_patch, cmap="jet"); ax[1].set_title(f"CLS attention\n(patch grid {gh}x{gw})"); ax[1].axis("off")
    ax[2].imshow(raw); ax[2].imshow(heat, cmap="jet", alpha=0.5)
    ax[2].set_title(f"overlay (patch_mode={args.patch_mode})"); ax[2].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    fig.savefig(args_cli.out, dpi=130)
    print(f"saved {args_cli.out}")


if __name__ == "__main__":
    main()
