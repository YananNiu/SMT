"""Attention visualisation for the SMT (vit_model_img_ts) model.

Follows Attention_visualization.ipynb: rebuilds the model with its attention
hook enabled, runs one image+time-series sample, and produces two CAM overlays
of the [CLS]->image-patch attention on the sky image:

  * last layer  -- attention of the final transformer block (heads averaged)
  * all layers  -- attention rollout across every block (Abnar & Zuidema, 2020)

    python scripts/visualize_attention.py --config configs/smt.yaml \
        --ckpt outputs/smt.pt --index 0 --out outputs/attention.png

Without --ckpt the model runs with random weights (a plumbing sanity check,
not a trained explanation).
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_rollout_attention(layer_matrices):
    """Attention rollout (as in Attention_visualization.ipynb).

    layer_matrices: list of [N, N] head-averaged attention matrices, one per
    block. Adds the residual/identity to each layer and multiplies them
    together (no row re-normalisation), returning the [N, N] joint attention.
    """
    import torch
    n = layer_matrices[0].shape[-1]
    eye = torch.eye(n)
    mats = [m + eye for m in layer_matrices]
    joint = mats[0]
    for m in mats[1:]:
        joint = m @ joint
    return joint


def patch_heatmap(cls_vec, gh, gw, img_size, F):
    """CLS->patch vector -> normalised [0,1] heat map at image resolution."""
    import numpy as np
    m = cls_vec.reshape(1, 1, gh, gw)
    m = F.interpolate(m, size=img_size, mode="bilinear", align_corners=False)[0, 0].numpy()
    return (m - m.min()) / (np.ptp(m) + 1e-8)


def cam_overlay(raw01, heat, cm):
    """show_cam_on_image (Attention_visualization.ipynb): additive CAM.

    JET colormap of the attention added onto the image, then normalised.
    """
    heat_rgb = cm.jet(heat)[..., :3]
    cam = heat_rgb + raw01
    return cam / cam.max()


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
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from smt.config import load_config
    from smt.preprocess import ensure_ghi_index
    from smt.data_provider import DataGenerator_ViLT
    from smt.models.vit_model_collection import vit_model_img_ts, Layer_scale_init_Block

    args = load_config(args_cli.config, {"data_root": args_cli.data_root})
    args.ts_data = ensure_ghi_index(args.ts_data, args.latitude, args.longitude, args.altitude)
    assert args.model_skeleton == "vit_model_img_ts", \
        "attention visualisation is defined for the SMT model (vit_model_img_ts)"
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
        _, attn = model(x, ts)          # attn: list (per block) of [1, heads, N, N]

    n_patches = model.patch_embed.num_patches
    gh, gw = model.patch_embed.grid_size
    # per-block, average over heads -> [N, N]
    per_layer = [a[0].mean(0) for a in attn]

    # last layer: [CLS] row over the image patches (drop [CLS] and the ts token)
    cls_last = per_layer[-1][0, 1:1 + n_patches]
    # all layers: attention rollout, then the same [CLS]->image-patch slice
    cls_roll = compute_rollout_attention(per_layer)[0, 1:1 + n_patches]

    raw01 = raw.astype(np.float32) / 255.0
    heat_last = patch_heatmap(cls_last, gh, gw, img_size, F)
    heat_roll = patch_heatmap(cls_roll, gh, gw, img_size, F)
    cam_last = cam_overlay(raw01, heat_last, cm)
    cam_roll = cam_overlay(raw01, heat_roll, cm)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(raw); ax[0].set_title(f"sky image\n{str(t)}"); ax[0].axis("off")
    ax[1].imshow(cam_last); ax[1].set_title("last-layer attention"); ax[1].axis("off")
    ax[2].imshow(cam_roll)
    ax[2].set_title(f"all-layer rollout (patch_mode={args.patch_mode})"); ax[2].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    fig.savefig(args_cli.out, dpi=130)
    print(f"saved {args_cli.out}")


if __name__ == "__main__":
    main()
