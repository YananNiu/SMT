"""Evaluate a trained model, or the smart-persistence baseline, on the test split.

    # smart-persistence baseline (no model, no training needed)
    python scripts/evaluate.py --config configs/persistence.yaml --persistence

    # a trained checkpoint
    python scripts/evaluate.py --config configs/smt.yaml --ckpt outputs/smt.pt

Reports RMSE, RSE (RMSE / std of target) and Pearson correlation, and writes
per-sample predictions to outputs/<name>_pred.csv.
"""
import argparse
import os
import sys
import types
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def metrics(pred, true):
    import numpy as np
    pred, true = np.asarray(pred), np.asarray(true)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rse = float(rmse / true.std(ddof=1))
    denom = pred.std() * true.std()
    corr = float(((pred - pred.mean()) * (true - true.mean())).mean() / denom) if denom > 0 else float("nan")
    return rmse, rse, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", default=None, help="trained checkpoint to evaluate")
    ap.add_argument("--persistence", action="store_true",
                    help="evaluate the smart-persistence baseline instead of a model")
    ap.add_argument("--out", default=None)
    args_cli = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    m = types.ModuleType("wandb"); m.log = lambda *a, **k: None
    @contextlib.contextmanager
    def _i(*a, **k): yield types.SimpleNamespace(config=types.SimpleNamespace())
    m.init = _i; sys.modules["wandb"] = m

    import numpy as np
    import pandas as pd
    import torch
    from smt.config import load_config
    from smt.preprocess import ensure_ghi_index
    from smt.data_provider import DataGenerator_ViLT
    from smt.engine import make_vilt
    from smt.train_val_test import test_oneEpoch

    args = load_config(args_cli.config, {"data_root": args_cli.data_root})
    args.ts_data = ensure_ghi_index(args.ts_data, args.latitude, args.longitude, args.altitude)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = os.path.splitext(os.path.basename(args_cli.config))[0]
    out = args_cli.out or os.path.join("outputs", f"{name}_pred.csv")
    os.makedirs("outputs", exist_ok=True)

    if args_cli.persistence:
        # smart persistence = clear-sky[t+h] * (GHI[t] / clear-sky[t]); computed
        # for every sample by the data provider as `smart_index`.
        ds = DataGenerator_ViLT(
            image=args.image, image_time=args.image_time, ts_data=args.ts_data,
            horizon=args.horizon, window=args.window, flag="test",
            data_flag=args.data_flag, indices=args.indices, special_test=args.special_test,
            image_token=False, ts_token=True)
        pred, true, t = ds.smart_index, ds.labels, ds.times
        tag = "smart-persistence"
    else:
        assert args_cli.ckpt, "provide --ckpt or --persistence"
        packed = make_vilt(DataGenerator_ViLT, args, device)
        model, _, _, test_data = packed[0], packed[1], packed[2], packed[3]
        model.load_state_dict(torch.load(args_cli.ckpt, map_location=device))
        df = test_oneEpoch(test_data, model=model, evaluateL2=torch.nn.MSELoss(reduction="sum"),
                           batch_size=args.batch_size, device=device)
        pred, true, t = df["predict"].values, df["test"].values, df["time"].values
        tag = os.path.basename(args_cli.ckpt)

    rmse, rse, corr = metrics(pred, true)
    print(f"[{tag}]  n={len(true)}  RMSE={rmse:.4f}  RSE={rse:.4f}  Corr={corr:.4f}")
    pd.DataFrame({"time": t, "predict": pred, "true": true}).to_csv(out, index=False)
    print(f"predictions -> {out}")


if __name__ == "__main__":
    main()
