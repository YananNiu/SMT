"""Train a model from a YAML config.

    python scripts/train.py --config configs/smt.yaml
    python scripts/train.py --config configs/smt.yaml --epochs 3 --out outputs/smt.pt

Every hyper-parameter comes from the YAML (see smt/config.py). A few knobs can
be overridden on the command line for quick toy runs. wandb is stubbed out by
default (set --wandb to enable real logging).
"""
import argparse
import os
import sys
import types
import contextlib

# make the repo root importable no matter where we run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_wandb():
    m = types.ModuleType("wandb")
    m.log = lambda *a, **k: None

    @contextlib.contextmanager
    def _init(*a, **k):
        yield types.SimpleNamespace(config=types.SimpleNamespace())
    m.init = _init
    sys.modules["wandb"] = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="override data_root")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs_max")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="checkpoint path (default outputs/<config>.pt)")
    ap.add_argument("--wandb", action="store_true", help="use real wandb instead of the stub")
    args_cli = ap.parse_args()

    if not args_cli.wandb:
        os.environ.setdefault("WANDB_MODE", "disabled")
        _stub_wandb()

    import numpy as np
    import torch
    import wandb
    from smt.config import load_config
    from smt.preprocess import ensure_ghi_index
    from smt.data_provider import DataGenerator_ViLT, DataGenerator_ViLT_2img
    from smt.engine import make_vilt, run_model
    from smt.train_val_test import train_oneEpoch, evaluate_oneEpoch

    overrides = dict(data_root=args_cli.data_root,
                     epochs_max=args_cli.epochs,
                     epochs=args_cli.epochs,
                     batch_size=args_cli.batch_size,
                     seed=args_cli.seed)
    args = load_config(args_cli.config, overrides)
    # make sure the clear-sky / GHI-index columns exist (compute from coords if not)
    args.ts_data = ensure_ghi_index(args.ts_data, args.latitude, args.longitude, args.altitude)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | model: {args.model_skeleton} | patch: {args.patch_mode} "
          f"| img_tok={args.image_token} ts_tok={args.ts_token}")

    out = args_cli.out or os.path.join(
        "outputs", os.path.splitext(os.path.basename(args_cli.config))[0] + ".pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # 2-camera / 2-image models use the two-image data loader
    Generator = (DataGenerator_ViLT_2img
                 if args.model_skeleton in ("CNNLSTM_2camera", "vit_model_2img_ts")
                 else DataGenerator_ViLT)

    with wandb.init(mode="disabled" if not args_cli.wandb else None):
        packed = make_vilt(Generator, args, device)
        model, train_data, valid_data, test_data, criterion, evaluateL2, optim, sched, best_val = packed
        print(f"samples  train/val/test = "
              f"{len(train_data.times)}/{len(valid_data.times)}/{len(test_data.times)}")
        run_model(model, train_oneEpoch, evaluate_oneEpoch,
                  train_data, valid_data, test_data, criterion, evaluateL2,
                  optim, sched, best_val, args, out, device)
    print(f"\nbest checkpoint saved to {out}")


if __name__ == "__main__":
    main()
