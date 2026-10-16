"""
STEP 3.1 - The metrics every model in this project is judged by.

One place for all of them, so a baseline, a tree model and a neural network
are always scored in exactly the same way.

Regression (DO in mg/L):
    R2    - share of the variance explained
    RMSE  - root mean squared error, in mg/L
    MAE   - mean absolute error, in mg/L
    NSE   - Nash-Sutcliffe efficiency, the standard score in hydrology.
            It is the same formula as R2 against the mean of the observed
            values, so NSE = 1 is perfect and NSE = 0 is no better than
            always predicting the observed mean.
    skill_vs_persistence - how much better than "DO will stay as it is now":
            1 - RMSE_model / RMSE_persistence. Above 0 means better.

Classification (Low / Medium / Good):
    accuracy, balanced accuracy, macro precision / recall / F1,
    ROC-AUC (one-vs-rest, macro), per-class precision / recall / F1,
    and the confusion matrix.

Rows whose target is NaN for that horizon are always dropped first: a
forecast cannot be scored against a reading that does not exist.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.config import load_config

CFG = load_config()
CLASS_NAMES = CFG["target"]["class_names"]          # ["Low", "Medium", "Good"]
N_CLASSES = len(CLASS_NAMES)


def as_float(x):
    """
    Anything (list, numpy array, pandas Series) to a plain float64 array.
    The class targets are stored as pandas' nullable Int8, whose pd.NA does
    not survive a direct numpy cast, so it is converted here instead.
    """
    import pandas as pd
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.astype("float64").to_numpy()
    return np.asarray(x, dtype="float64")


def valid_mask(y_true, *others):
    """
    Which rows can be scored: the target must exist, and so must the
    prediction. Returns a boolean array.
    """
    ok = ~np.isnan(as_float(y_true))
    for o in others:
        if o is None:
            continue
        a = as_float(o)
        if a.ndim == 1:
            ok &= ~np.isnan(a)
        else:
            ok &= ~np.isnan(a).any(axis=1)
    return ok


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, y_persistence=None):
    """
    Score a DO forecast. `y_persistence` is the "no change" forecast (the DO
    reading at time t); pass it to get skill_vs_persistence.
    """
    ok = valid_mask(y_true, y_pred, y_persistence)
    y = as_float(y_true)[ok]
    p = as_float(y_pred)[ok]
    n = len(y)
    if n == 0:
        return {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan,
                "nse": np.nan, "skill_vs_persistence": np.nan}

    errors = p - y
    sse = float(np.sum(errors ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))         # variance of the truth

    rmse = float(np.sqrt(sse / n))
    mae = float(np.mean(np.abs(errors)))
    # R2 and NSE are the same formula here; both are reported because the ML
    # literature asks for R2 and the hydrology literature for NSE.
    r2 = 1 - sse / sst if sst > 0 else np.nan
    nse = r2

    skill = np.nan
    if y_persistence is not None:
        q = as_float(y_persistence)[ok]
        rmse_p = float(np.sqrt(np.mean((q - y) ** 2)))
        skill = 1 - rmse / rmse_p if rmse_p > 0 else np.nan

    return {"n": n, "r2": r2, "rmse": rmse, "mae": mae, "nse": nse,
            "skill_vs_persistence": skill}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_proba=None):
    """
    Score a Low / Medium / Good forecast.

    `y_proba` is an (n, 3) array of class probabilities and is only needed
    for ROC-AUC. A persistence classifier has no probabilities, so its
    roc_auc comes back as NaN.
    """
    ok = valid_mask(y_true, y_pred, y_proba)
    y = as_float(y_true)[ok].astype(int)
    p = as_float(y_pred)[ok].astype(int)
    labels = list(range(N_CLASSES))
    if len(y) == 0:
        empty = {f"{m}_{c}": np.nan for c in CLASS_NAMES
                 for m in ["precision", "recall", "f1"]}
        return {"n": 0, "accuracy": np.nan, "balanced_accuracy": np.nan,
                "precision_macro": np.nan, "recall_macro": np.nan,
                "f1_macro": np.nan, "roc_auc_ovr_macro": np.nan,
                "confusion_matrix": np.zeros((N_CLASSES, N_CLASSES), int),
                **empty}

    out = {
        "n": len(y),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        # zero_division=0: a class a model never predicts scores 0, not an error
        "precision_macro": float(precision_score(y, p, labels=labels,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, p, labels=labels,
                                           average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, p, labels=labels,
                                   average="macro", zero_division=0)),
    }

    # per class, in the order Low, Medium, Good
    per = {m: f(y, p, labels=labels, average=None, zero_division=0)
           for m, f in [("precision", precision_score), ("recall", recall_score),
                        ("f1", f1_score)]}
    for i, name in enumerate(CLASS_NAMES):
        for m in per:
            out[f"{m}_{name}"] = float(per[m][i])

    # ROC-AUC needs probabilities and at least two classes actually present
    out["roc_auc_ovr_macro"] = np.nan
    if y_proba is not None and len(np.unique(y)) > 1:
        proba = as_float(y_proba)[ok]
        try:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y, proba, multi_class="ovr", average="macro",
                              labels=labels))
        except ValueError:
            pass        # not all classes present: leave it as NaN

    out["confusion_matrix"] = confusion_matrix(y, p, labels=labels)
    return out


def confusion_frame(cm):
    """The confusion matrix as a labelled table (rows = true, cols = predicted)."""
    import pandas as pd
    return pd.DataFrame(cm,
                        index=[f"true {c}" for c in CLASS_NAMES],
                        columns=[f"pred {c}" for c in CLASS_NAMES])


def to_class(do_values):
    """
    Turn DO in mg/L into the class code: 0 Low, 1 Medium, 2 Good.
    NaN stays NaN. Used by the persistence baseline and when checking targets.
    """
    low = CFG["target"]["thresholds"]["low"]
    good = CFG["target"]["thresholds"]["good"]
    v = as_float(do_values)
    out = np.full(v.shape, np.nan)
    out[v < low] = 0
    out[(v >= low) & (v <= good)] = 1
    out[v > good] = 2
    return out
