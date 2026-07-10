"""Turn a raw-GHI csv into a trainable one by adding the clear-sky / GHI-index
columns, computed from the site coordinates via base_model.add_clearsky_columns.

    python tools/compute_ghi_index.py \
        --csv raw_ghi.csv --lat 46.630858 --lon 8.580525 --out ghi_with_index.csv

Input csv must have at least `time` and `ghi` columns. Output adds:
    ghi_clear_sky, day, GHI_daily_max_clearsky, GHI_percent_wrt_max
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--altitude", type=float, default=0)
    ap.add_argument("--model", default="haurwitz", choices=["haurwitz", "ineichen", "simplified_solis"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    from smt.base_model import add_clearsky_columns

    df = pd.read_csv(a.csv)
    df = add_clearsky_columns(df, a.lat, a.lon, altitude=a.altitude, model=a.model)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  rows={len(df)}  cols={list(df.columns)}")


if __name__ == "__main__":
    main()
