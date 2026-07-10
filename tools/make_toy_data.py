"""Build a tiny toy dataset from the full WACV arrays.

The published repo ships a ~3-day slice of the ANT site so the whole pipeline
runs on a laptop in minutes. This is how that slice was produced; point
`--src` at the full `WACV_data/` to regenerate a bigger one.

The image array (`{site}_X_224_all.npy`) and time array
(`{site}_time_224_all.npy`) are index-aligned; we mask both with the same
time window so they stay aligned. The GHI csv is sliced by time (with a
look-back margin so the first sample still has its 24 h history) and
re-indexed 0..N-1 (the data provider addresses it with `.loc`).

    python tools/make_toy_data.py \
        --src /path/to/WACV_data --dst data/toy --site ANT \
        --img-start 2023-06-01 --img-end 2023-06-04 \
        --csv-margin-days 2
"""
import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="full WACV_data directory")
    ap.add_argument("--dst", default="data/toy")
    ap.add_argument("--site", default="ANT")
    ap.add_argument("--img-start", default="2023-06-01")
    ap.add_argument("--img-end", default="2023-06-04")
    ap.add_argument("--img-stride", type=int, default=1,
                    help="keep every Nth image (2 => 20-min cadence, halves the size)")
    ap.add_argument("--csv-margin-days", type=int, default=2,
                    help="extra csv history kept before img-start (>= window/144 days)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    s = args.site

    # --- images + their timestamps (index-aligned) ---
    times = np.load(os.path.join(args.src, f"{s}_time_224_all.npy"),
                    allow_pickle=True).astype("datetime64[s]")
    t = pd.to_datetime(times)
    mask = np.asarray((t >= pd.Timestamp(args.img_start)) & (t < pd.Timestamp(args.img_end)))
    keep = np.where(mask)[0][:: args.img_stride]
    print(f"images kept: {len(keep)}  ({args.img_start} .. {args.img_end}, stride {args.img_stride})")

    X = np.load(os.path.join(args.src, f"{s}_X_224_all.npy"), mmap_mode="r")
    X_small = np.ascontiguousarray(X[keep])
    np.save(os.path.join(args.dst, f"{s}_X_224_all.npy"), X_small)
    np.save(os.path.join(args.dst, f"{s}_time_224_all.npy"), times[keep])
    mb = X_small.nbytes / 1e6
    print(f"saved {s}_X_224_all.npy  shape={X_small.shape}  ({mb:.0f} MB)")

    # --- GHI time series (sliced by time, re-indexed) ---
    df = pd.read_csv(os.path.join(args.src, f"ghi_{s}_pure_scaled.csv"))
    df["time"] = pd.to_datetime(df["time"])
    csv_start = pd.Timestamp(args.img_start) - pd.Timedelta(days=args.csv_margin_days)
    csv_end = pd.Timestamp(args.img_end) + pd.Timedelta(days=1)
    df = df[(df["time"] >= csv_start) & (df["time"] < csv_end)].reset_index(drop=True)
    df.to_csv(os.path.join(args.dst, f"ghi_{s}_pure_scaled.csv"), index=False)
    print(f"saved ghi_{s}_pure_scaled.csv  rows={len(df)}  "
          f"({df['time'].min()} .. {df['time'].max()})")

    # --- suggest a test split covering roughly the last 1/3 of the images ---
    img_t = t[mask]
    split = img_t[int(len(img_t) * 0.66)].floor("D")
    print("\nsuggested config `indices` (test period):")
    print(f"  indices: ['{split:%Y-%m-%d %H:%M}', '{args.img_end} 23:59']")


if __name__ == "__main__":
    main()
