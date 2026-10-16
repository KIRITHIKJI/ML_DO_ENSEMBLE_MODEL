"""
Shared plumbing for every model in step 3, so a baseline and a gradient
boosting machine are fed exactly the same data in exactly the same way.

Nothing here touches the test split. `load_split("test")` exists but is not
called by any step-3 script.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import to_class

CFG = load_config()
P = CFG["paths"]
M = CFG["models"]
HORIZONS = CFG["horizons"]
DO = CFG["target"]["column"]
CLASS_NAMES = CFG["target"]["class_names"]
SEED = CFG["seed"]

FEATURE_GROUPS = ["water", "weather", "air", "time", "station"]


def feature_columns():
    """The 211 feature columns, in a fixed order, from feature_groups.json."""
    with open(project_path(P["feature_groups"]), encoding="utf-8") as f:
        groups = json.load(f)
    return [c for g in FEATURE_GROUPS for c in groups[g]]


def load_split(name, columns=None):
    """Read data/processed/<name>.parquet ('train', 'val' or 'test')."""
    return pd.read_parquet(project_path(P[name]), columns=columns)


def target_name(task, h):
    """y_reg_24 or y_cls_24."""
    return f"{'y_reg' if task == 'regression' else 'y_cls'}_{h}"


def get_xy(df, task, h, features, drop_missing_target=True):
    """
    Features and target for one task and horizon.

    For training we drop rows whose target is NaN (there is nothing to learn
    from them). For validation we keep every row, predict for all of them,
    and let the metrics skip the NaN ones.
    """
    y = pd.to_numeric(df[target_name(task, h)], errors="coerce")
    if drop_missing_target:
        keep = y.notna().to_numpy()
        return df.loc[keep, features], y[keep], keep
    return df[features], y, np.ones(len(df), dtype=bool)


def persistence(df, task):
    """
    The naive forecast every model has to beat: "DO will be what it is now".
    For regression that is the current DO reading; for classification, the
    class of the current DO reading.
    """
    now = df[DO].to_numpy(dtype="float64")
    return now if task == "regression" else to_class(now)


def balanced_sample_weight(y):
    """
    Weights that make the three classes count equally, the same rule sklearn
    uses for class_weight="balanced": w = n / (n_classes * count_of_class).
    Needed for XGBoost, which has no class_weight argument.
    """
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    weight_of = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([weight_of[v] for v in y])


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def _dir(key):
    path = project_path(M["dirs"][key])
    os.makedirs(path, exist_ok=True)
    return path


def save_predictions(model_name, task, h, df, y_pred, y_proba=None):
    """
    Validation predictions, one file per model / task / horizon. The later
    ensemble step builds on exactly these files, so they always carry the
    station and time columns needed to line them up.
    """
    out = pd.DataFrame({
        "station": df["station"].to_numpy(),
        "time": df["time"].to_numpy(),
        "y_true": pd.to_numeric(df[target_name(task, h)], errors="coerce").to_numpy(),
        "y_pred": np.asarray(y_pred, dtype="float32"),
    })
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype="float32")
        for i, name in enumerate(CLASS_NAMES):
            out[f"proba_{name}"] = proba[:, i]
    path = os.path.join(_dir("val_predictions"), f"{model_name}_{task}_h{h}.parquet")
    out.to_parquet(path, index=False)
    return path


def save_importance(model_name, task, h, features, importances):
    """The most useful features, as judged by the model itself."""
    top = (pd.DataFrame({"feature": features, "importance": importances})
           .sort_values("importance", ascending=False)
           .head(M["top_importances"]).reset_index(drop=True))
    path = os.path.join(_dir("val_importance"), f"{model_name}_{task}_h{h}.csv")
    top.to_csv(path, index=False)
    return top


def save_model(model, model_name, task, h, how="joblib"):
    """Save a fitted model so later steps do not have to retrain it."""
    folder = _dir("ml_models")
    if how == "joblib":
        import joblib
        path = os.path.join(folder, f"{model_name}_{task}_h{h}.joblib")
        joblib.dump(model, path, compress=3)
    elif how == "xgboost":
        path = os.path.join(folder, f"{model_name}_{task}_h{h}.json")
        model.save_model(path)
    elif how == "lightgbm":
        path = os.path.join(folder, f"{model_name}_{task}_h{h}.txt")
        model.booster_.save_model(path)
    return path


def update_metrics_csv(task, rows):
    """
    Add these rows to results/val/metrics_<task>.csv, replacing any earlier
    rows for the same model and horizon. That way baselines.py and
    train_ml.py can be run separately, in any order, as often as we like.
    """
    path = os.path.join(_dir("val_metrics"), f"metrics_{task}.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        keys = set(zip(new["model"], new["horizon"]))
        old = old[~old.apply(lambda r: (r["model"], r["horizon"]) in keys, axis=1)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["horizon", "model"]).reset_index(drop=True)
    new.to_csv(path, index=False)
    return path


class Timer:
    """`with Timer() as t:` ... then t.seconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0

    def __str__(self):
        return f"{self.seconds:.1f}s"
