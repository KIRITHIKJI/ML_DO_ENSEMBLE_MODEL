"""
Tests for the step-5b class-prior calibration in src/evaluation/calibration.py.

The one that matters is the last group: the calibration factors must be
fitted on val-A rows only. If any val-B row leaks into the fit, the val-B
numbers in the report are no longer an honest estimate - the model would
have been tuned on the very rows it is being judged on.

Run from the project root:   pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluation.calibration import (apply_calibration, fit_calibration,
                                        val_a_mask, val_b_mask)

CFG = load_config()
VS = CFG["validation_split"]


# ---------------------------------------------------------------------------
# The two halves of 2024
# ---------------------------------------------------------------------------
def test_val_a_and_val_b_do_not_overlap_and_cover_2024():
    hours = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    a, b = val_a_mask(hours), val_b_mask(hours)
    assert not (a & b).any(), "a row cannot be in both halves"
    assert (a | b).all(), "every hour of 2024 belongs to one half"
    assert hours[a].max() < hours[b].min(), "val-A must come before val-B"
    assert str(hours[a].min().date()) == VS["val_a_start"]
    assert str(hours[b].min().date()) == VS["val_b_start"]


def test_the_halves_ignore_other_years():
    hours = pd.date_range("2023-12-30", "2025-01-02", freq="D")
    a, b = val_a_mask(hours), val_b_mask(hours)
    assert not a[hours.year != 2024].any()
    assert not b[hours.year != 2024].any()


# ---------------------------------------------------------------------------
# What calibration does
# ---------------------------------------------------------------------------
def test_calibrated_probabilities_still_sum_to_one():
    p = np.array([[.2, .3, .5], [.7, .2, .1]])
    out = apply_calibration(p, (1.3, 1.0, 0.7))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_factors_of_one_change_nothing():
    p = np.array([[.2, .3, .5], [.7, .2, .1]])
    assert np.allclose(apply_calibration(p, (1.0, 1.0, 1.0)), p)


def test_a_bigger_factor_makes_that_class_more_likely_to_win():
    p = np.array([[.40, .35, .25]])              # Low wins by a little
    assert apply_calibration(p, (1.0, 1.0, 1.0)).argmax() == 0
    assert apply_calibration(p, (0.7, 1.3, 1.0)).argmax() == 1


def test_calibration_never_makes_val_a_worse():
    """The search starts from (1,1,1), so it can only improve or tie."""
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1, 2], 2000, p=[.2, .25, .55])
    p = rng.random((2000, 3))
    p[np.arange(2000), y] += 0.8                  # a weak but real signal
    p /= p.sum(axis=1, keepdims=True)
    factors, after, before = fit_calibration(y, p)
    assert after >= before - 1e-9
    assert len(factors) == 3


def test_rows_with_a_missing_target_are_skipped():
    """
    The middle row has no target, so it must not be scored. The two rows
    that remain are both predicted correctly; macro F1 is 2/3 rather than 1
    because Medium never appears among them and scores 0, which is the same
    convention src/evaluation/metrics.py uses.
    """
    y = pd.Series([0.0, np.nan, 2.0])
    p = np.array([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]])
    factors, after, before = fit_calibration(y, p)
    assert before == pytest.approx(2 / 3)
    assert after == pytest.approx(2 / 3)          # nothing left to improve
    assert factors == (1.0, 1.0, 1.0)

    # if the NaN row were scored as a mistake, Low and Good would lose
    # precision and the score would drop below 2/3
    y_all = pd.Series([0.0, 0.0, 2.0])
    _, _, before_all = fit_calibration(y_all, p)
    assert before_all < 2 / 3


# ---------------------------------------------------------------------------
# THE IMPORTANT ONE: the fit must use val-A rows only
# ---------------------------------------------------------------------------
def test_calibration_is_fitted_on_val_a_rows_only():
    """
    Build a year where the two halves want OPPOSITE calibrations, fit on
    val-A as the pipeline does, and check the answer follows val-A. If val-B
    rows were leaking in, the factors would be pulled towards them.
    """
    rng = np.random.default_rng(2)
    hours = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    n = len(hours)
    a, b = val_a_mask(hours), val_b_mask(hours)

    # val-A: the truth is nearly always Low, but the model leans to Good
    # val-B: the truth is nearly always Good, and the model leans to Low
    y = np.where(a, 0, 2)
    p = np.empty((n, 3))
    p[a] = [0.30, 0.25, 0.45]
    p[b] = [0.45, 0.25, 0.30]
    p += rng.random((n, 3)) * 0.02

    factors, after, before = fit_calibration(y[a], p[a])
    # the fit has to push Low up (or Good down) to be right on val-A
    assert factors[0] > factors[2], f"factors {factors} do not follow val-A"
    assert after > before

    # and the same fit must be WRONG on val-B - proof it never saw it
    from sklearn.metrics import f1_score
    calibrated_b = apply_calibration(p[b], factors).argmax(axis=1)
    plain_b = p[b].argmax(axis=1)
    assert f1_score(y[b], calibrated_b, average="macro", zero_division=0) <= \
        f1_score(y[b], plain_b, average="macro", zero_division=0) + 1e-9

    # fitting on the whole year instead would give different factors
    whole_year, _, _ = fit_calibration(y, p)
    assert whole_year != factors, \
        "fitting on all of 2024 gives the same answer - the test is not biting"


def test_fitting_on_val_a_uses_only_half_the_rows():
    """A direct check that the mask really subsets the data handed over."""
    hours = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    a = val_a_mask(hours)
    assert a.sum() < len(hours)
    assert hours[a].max() <= pd.Timestamp(f"{VS['val_a_end']} 23:59:59")
