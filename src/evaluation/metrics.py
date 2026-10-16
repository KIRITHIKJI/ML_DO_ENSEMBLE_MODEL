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


# ---------------------------------------------------------------------------
# STEP 6 - early-warning metrics
#
# Overall accuracy hides the thing this project exists for. Most hours the
# river simply stays where it is, so a model that never predicts a change
# still scores well. These metrics look only at the hours that matter: the
# ones where the class CHANGES, and in particular where water that is not
# currently Low becomes Low.
# ---------------------------------------------------------------------------
def early_warning_metrics(class_now, y_true, y_pred):
    """
    class_now  the class at time t (0 Low, 1 Medium, 2 Good)
    y_true     the true class at time t + h
    y_pred     the predicted class at time t + h

    Returns:
      n_transitions        rows where the class changes between t and t+h
      transition_accuracy  accuracy on those rows only
      n_onsets             rows where it is NOT Low now and IS Low at t+h
      onset_recall         share of those the model predicted Low
      onset_precision      of the rows that are not Low now and are
                           predicted Low, the share that really are Low
      false_alarm_rate     of the rows that are not Low now and stay not
                           Low, the share wrongly predicted Low

    A persistence forecast scores 0 on onset recall and 0 on transition
    accuracy by definition: it predicts "the same as now", so it never
    predicts a change and is wrong on every row where one happens.
    """
    low = 0
    ok = valid_mask(y_true, y_pred, class_now)
    now = as_float(class_now)[ok].astype(int)
    true = as_float(y_true)[ok].astype(int)
    pred = as_float(y_pred)[ok].astype(int)

    changed = true != now
    not_low_now = now != low
    onset = not_low_now & (true == low)          # the water turns dangerous
    stays_safe = not_low_now & (true != low)
    predicted_low = not_low_now & (pred == low)

    def share(numerator, denominator):
        n = int(denominator.sum())
        return (float(numerator.sum()) / n) if n else float("nan")

    return {
        "n_transitions": int(changed.sum()),
        "transition_accuracy": share(changed & (pred == true), changed),
        "n_onsets": int(onset.sum()),
        "onset_recall": share(onset & (pred == low), onset),
        "onset_precision": share(predicted_low & (true == low), predicted_low),
        "false_alarm_rate": share(stays_safe & (pred == low), stays_safe),
    }


def threshold_weight(y_value, alpha, sigma, centre=None):
    """
    How much an error at this DO value matters, for the step-6 ensemble:

        w(y) = 1 + alpha * exp( -(y - threshold)^2 / (2 * sigma^2) )

    It is largest right at the Low threshold (5 mg/L), where being wrong
    turns a warning into a miss, and falls to 1 far away from it, where
    being wrong costs nothing but a number in a table. alpha = 0 switches
    the whole idea off.
    """
    centre = CFG["target"]["thresholds"]["low"] if centre is None else centre
    y = as_float(y_value)
    return 1.0 + alpha * np.exp(-((y - centre) ** 2) / (2.0 * sigma ** 2))


def weighted_rmse(y_true, y_pred, alpha, sigma):
    """RMSE with each row weighted by threshold_weight: the tuning score."""
    ok = valid_mask(y_true, y_pred)
    y, p = as_float(y_true)[ok], as_float(y_pred)[ok]
    if not len(y):
        return np.nan
    w = threshold_weight(y, alpha, sigma)
    return float(np.sqrt(np.sum(w * (y - p) ** 2) / np.sum(w)))


def danger_zone_rmse(y_true, y_pred, low=3.0, high=7.0):
    """
    RMSE over the rows that actually matter operationally: true DO between
    3 and 7 mg/L, the band where a river crosses into trouble.
    """
    ok = valid_mask(y_true, y_pred)
    y, p = as_float(y_true)[ok], as_float(y_pred)[ok]
    band = (y >= low) & (y <= high)
    if not band.any():
        return np.nan
    return float(np.sqrt(np.mean((y[band] - p[band]) ** 2)))


def brier_score(y_true_class, proba):
    """
    Brier score per row: the squared distance between the predicted
    probabilities and a 1-of-3 indicator of the truth. 0 is perfect, 2 is
    as wrong as possible. Used as the ensemble's classification error.
    """
    y = as_float(y_true_class)
    p = as_float(proba)
    out = np.full(len(y), np.nan)
    ok = ~np.isnan(y) & ~np.isnan(p).any(axis=1)
    target = np.zeros_like(p[ok])
    target[np.arange(ok.sum()), y[ok].astype(int)] = 1.0
    out[ok] = ((p[ok] - target) ** 2).sum(axis=1)
    return out
