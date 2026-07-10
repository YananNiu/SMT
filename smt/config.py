"""YAML -> args (SimpleNamespace).

All experiment hyper-parameters live in `configs/*.yaml`. A config is a flat
mapping whose keys are exactly the attribute names the training engine
(`smt.engine`) and data provider (`smt.data_provider`) read off `args`.

On top of the raw YAML we:
  * fill in defaults (so configs stay short and only override what matters),
  * build the data paths from `data_root` + `site` when they are not given
    explicitly (e.g. `{data_root}/{site}_X_224_all.npy`),
  * keep `epochs` and `epochs_max` in sync (timm's scheduler reads `epochs`,
    the training loop reads `epochs_max`).
"""
from types import SimpleNamespace
import os
import yaml

# ---------------------------------------------------------------------------
# Defaults. A YAML file only needs to specify what it changes.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # --- data ---
    data_root="data/toy",
    site="ANT",
    image=None, image_time=None, ts_data=None,          # built from data_root/site if None
    image1=None, image_time1=None, image2=None, image_time2=None,  # 2-camera only
    horizon=12,          # prediction lead time in 10-min steps (12 => 2 h ahead)
    window=24 * 6,       # look-back length for the time series (144 => 24 h)
    meteo=False,
    # target: what the model predicts.
    #   ghi_index -> GHI_percent_wrt_max (clear-sky index, recommended / more stable)
    #   ghi       -> raw irradiance (W/m^2)
    # Either way the clear-sky columns are needed; if the csv lacks them they are
    # computed from (latitude, longitude) via base_model.add_clearsky_columns.
    target="ghi_index",
    data_flag=None,      # derived from `target` unless set explicitly
    latitude=None, longitude=None, altitude=0,   # site coordinates for the GHI index
    indices=None,        # [test_start, test_end]; required (special_test split)
    special_test=True,
    img_num=None,

    # --- which branches are active (the SMT ablation knobs) ---
    model_skeleton="vit_model_img_ts",
    image_token=True,
    ts_token=True,
    smart_token=False,

    # --- transformer ---
    depth_transformer=3,
    embed_dim=192,
    num_heads=6,
    drop_rate=0.0,
    attn_drop=0.1,
    patch_mode="square",   # square | row | column  (patch-shape ablation)

    # --- LSTNet (used by lstnet + CNNLSTM_LSTNet) ---
    highway_window=12, skip=0, dropout=0.2,
    hidCNN=100, hidRNN=100, CNN_kernel=6, hidSkip=5,
    output_fun="ReLU",

    # --- training ---
    epochs_max=50, patience=5, batch_size=32, seed=0,

    # --- optimizer (timm create_optimizer) ---
    opt="adamw", opt_eps=1e-8, opt_betas=None, clip_grad=None,
    momentum=0.9, weight_decay=0.05,

    # --- lr scheduler (timm create_scheduler) ---
    sched="cosine", lr=5e-4, warmup_lr=5e-5, warmup_epochs=3, min_lr=5e-5,
    epochs=50, lr_cycle_decay=0.5, lr_cycle_limit=5,
)


def _build_paths(cfg):
    """Fill image / image_time / ts_data from data_root + site when omitted."""
    root, site = cfg["data_root"], cfg["site"]
    if cfg.get("image") is None:
        cfg["image"] = os.path.join(root, f"{site}_X_224_all.npy")
    if cfg.get("image_time") is None:
        cfg["image_time"] = os.path.join(root, f"{site}_time_224_all.npy")
    if cfg.get("ts_data") is None:
        cfg["ts_data"] = os.path.join(root, f"ghi_{site}_pure_scaled.csv")
    return cfg


def load_config(path, overrides=None):
    """Load a YAML config into an args namespace.

    `overrides` is an optional dict (e.g. from CLI flags) applied last.
    """
    cfg = dict(DEFAULTS)
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    cfg.update(user)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    cfg = _build_paths(cfg)

    # resolve the target column: `target` -> `data_flag` (unless data_flag is explicit)
    TARGET_TO_COL = {"ghi_index": "GHI_percent_wrt_max", "ghi": "ghi"}
    if cfg.get("data_flag") is None:
        if cfg["target"] not in TARGET_TO_COL:
            raise ValueError(f"{path}: target must be one of {list(TARGET_TO_COL)}, got {cfg['target']!r}")
        cfg["data_flag"] = TARGET_TO_COL[cfg["target"]]

    # keep the two epoch knobs consistent (scheduler vs. loop)
    if "epochs" not in user and "epochs_max" in (user or {}):
        cfg["epochs"] = cfg["epochs_max"]
    cfg["epochs_max"] = cfg.get("epochs_max", cfg["epochs"])

    if cfg["indices"] is None:
        raise ValueError(
            f"{path}: 'indices' (test period [start, end]) must be set — "
            "it defines the special_test train/val/test split."
        )
    return SimpleNamespace(**cfg)
