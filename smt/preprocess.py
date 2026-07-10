"""Make sure a time-series csv has the columns the data provider needs.

The pipeline forecasts (and builds its smart-persistence reference from) the
**GHI index** = clear-sky index, which needs the clear-sky columns
``ghi_clear_sky`` / ``GHI_daily_max_clearsky`` / ``GHI_percent_wrt_max``.

  * ``target: ghi_index`` -> the model target is ``GHI_percent_wrt_max``.
  * ``target: ghi``       -> the model target is the raw ``ghi``.

Either way those clear-sky columns must exist. If the csv only has raw ``ghi``
(no clear-sky columns), we compute them from the site coordinates using
``smt.base_model.add_clearsky_columns`` and cache the enriched csv.
"""
import os
import pandas as pd

from smt.base_model import add_clearsky_columns

REQUIRED = ["ghi_clear_sky", "GHI_daily_max_clearsky", "GHI_percent_wrt_max"]


def ensure_ghi_index(csv_path, latitude=None, longitude=None, altitude=0,
                     cache_dir="outputs/_cache"):
    """Return a csv path that is guaranteed to contain the clear-sky columns.

    If `csv_path` already has them it is returned unchanged; otherwise they are
    computed from (latitude, longitude) and a new enriched csv is written to
    `cache_dir` and its path returned.
    """
    df = pd.read_csv(csv_path, nrows=5)
    if all(c in df.columns for c in REQUIRED):
        return csv_path

    if latitude is None or longitude is None:
        raise ValueError(
            f"{csv_path} is missing {REQUIRED} and no coordinates were given. "
            "Set `latitude`/`longitude` in the config (or use target: ghi_index "
            "with a pre-computed csv) so the GHI index can be derived via "
            "base_model.add_clearsky_columns().")
    if "ghi" not in df.columns:
        raise ValueError(f"{csv_path} needs a raw 'ghi' column to compute the GHI index.")

    full = pd.read_csv(csv_path)
    full = add_clearsky_columns(full, latitude, longitude, altitude)
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir,
                       os.path.basename(csv_path).replace(".csv", "_with_index.csv"))
    full.to_csv(out, index=False)
    print(f"[preprocess] computed GHI-index columns from coords "
          f"({latitude}, {longitude}) -> {out}")
    return out
