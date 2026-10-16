"""
STEP 5b - class-prior calibration, and the val-A / val-B split of 2024.

Why the split: step 5 chose settings on 2024 and reported on 2024, so its
numbers are the optimistic end of the range. Here the validation year is cut
in half - the first half (val-A) decides the loss weight and the calibration
factors, the second half (val-B) is only ever read to report. Nothing in
2025 is touched either way.

What calibration does: a classifier that is trained with balanced class
weights tends to over-predict the rare class. Multiplying the three
probabilities by one factor each, renormalising and taking the argmax moves
the decision boundaries without retraining anything. The factors are a
plain grid search on val-A, per model and per horizon.

This changes the class decision only. The regression output is untouched.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import load_config

CFG = load_config()
VS = CFG["validation_split"]
FIX = CFG["dl_cls_fix"]
N_CLASSES = len(CFG["target"]["class_names"])


# ---------------------------------------------------------------------------
# The two halves of 2024
# ---------------------------------------------------------------------------
def val_a_mask(times):
    """True for rows in the first half of 2024 (used to CHOOSE settings)."""
    t = pd.to_datetime(pd.Series(np.asarray(times)).reset_index(drop=True))
    return ((t >= VS["val_a_start"]) & (t <= f"{VS['val_a_end']} 23:59:59")).to_numpy()


def val_b_mask(times):
    """True for rows in the second half of 2024 (used only to REPORT)."""
    t = pd.to_datetime(pd.Series(np.asarray(times)).reset_index(drop=True))
    return ((t >= VS["val_b_start"]) & (t <= f"{VS['val_b_end']} 23:59:59")).to_numpy()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def apply_calibration(proba, factors):
    """
    Multiply each class probability by its factor and renormalise.
    The argmax is what changes; renormalising keeps the numbers readable as
    probabilities.
    """
    scaled = np.asarray(proba, dtype="float64") * np.asarray(factors, dtype="float64")
    total = scaled.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return scaled / total


def fit_calibration(y_true, proba, grid=None, metric=None):
    """
    Find the factors that maximise macro F1 on the rows GIVEN TO THIS
    FUNCTION. The caller is responsible for passing val-A rows only - that
    is what makes the val-B numbers honest, and it is what the test checks.

    Returns (factors, best_score, baseline_score). Rows whose target is
    missing are skipped.
    """
    grid = grid or FIX["calibration_grid"]
    y = np.asarray(pd.Series(y_true).astype("float64"))
    p = np.asarray(proba, dtype="float64")
    ok = ~np.isnan(y) & ~np.isnan(p).any(axis=1)
    y, p = y[ok].astype(int), p[ok]
    labels = list(range(N_CLASSES))
    if len(y) == 0:
        return (1.0,) * N_CLASSES, float("nan"), float("nan")

    def score(pred):
        return float(f1_score(y, pred, labels=labels, average="macro",
                              zero_division=0))

    baseline = score(p.argmax(axis=1))
    best, best_score = (1.0,) * N_CLASSES, baseline
    for factors in itertools.product(grid, repeat=N_CLASSES):
        # renormalising cannot change which class is largest, so for the
        # search it is enough to scale and take the argmax
        s = score((p * np.asarray(factors)).argmax(axis=1))
        if s > best_score + 1e-9:
            best, best_score = factors, s
    return tuple(float(f) for f in best), best_score, baseline
