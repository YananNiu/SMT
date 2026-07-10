"""Run an SMT ablation group: train + evaluate each config, tabulate the metrics.

    python scripts/run_ablation.py --group patch    --epochs 5
    python scripts/run_ablation.py --group branch   --epochs 5
    python scripts/run_ablation.py --group all      --epochs 5

Groups:
    patch   -- patch-shape ablation   (square / row / column)
    branch  -- SMT branch ablation    (image-only / ts-only / image+ts / +reference)
    all     -- everything in configs/ablation/

Writes outputs/ablation_<group>.csv with RMSE / RSE / Corr per config.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ABL = os.path.join(ROOT, "configs", "ablation")

GROUPS = {
    "patch": ["smt_patch_square", "smt_patch_row", "smt_patch_column"],
    "branch": ["smt_branch_ts_only", "smt_branch_img_only", "smt_branch_img_ts"],
}


def configs_for(group):
    if group == "all":
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(ABL, "*.yaml")))
    return GROUPS[group]


def run(cmd):
    print("  $", " ".join(cmd[len(ROOT):] if False else cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS) + ["all"], default="patch")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--data-root", default="data/toy")
    args = ap.parse_args()

    rows = []
    for name in configs_for(args.group):
        cfg = os.path.join("configs", "ablation", f"{name}.yaml")
        ckpt = os.path.join("outputs", f"{name}.pt")
        print(f"\n=== {name} ===")
        t = run([sys.executable, "scripts/train.py", "--config", cfg,
                 "--epochs", str(args.epochs), "--data-root", args.data_root,
                 "--out", ckpt])
        if t.returncode != 0:
            print(t.stderr[-800:]); rows.append((name, "FAIL", "", "")); continue
        e = run([sys.executable, "scripts/evaluate.py", "--config", cfg,
                 "--ckpt", ckpt, "--data-root", args.data_root])
        m = re.search(r"RMSE=([\d.]+)\s+RSE=([\d.]+)\s+Corr=([-\d.nan]+)", e.stdout)
        if m:
            rmse, rse, corr = m.groups()
            print(f"  RMSE={rmse}  RSE={rse}  Corr={corr}")
            rows.append((name, rmse, rse, corr))
        else:
            print(e.stdout[-400:]); rows.append((name, "?", "", ""))

    import csv
    out = os.path.join("outputs", f"ablation_{args.group}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "RMSE", "RSE", "Corr"]); w.writerows(rows)

    print("\n" + "=" * 56)
    print(f"{'config':<34}{'RMSE':>8}{'RSE':>7}{'Corr':>8}")
    print("-" * 56)
    for name, rmse, rse, corr in rows:
        print(f"{name:<34}{rmse:>8}{rse:>7}{corr:>8}")
    print("=" * 56)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
