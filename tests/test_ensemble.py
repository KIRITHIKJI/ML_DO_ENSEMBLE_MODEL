"""
Tests for the ATAE ensemble in src/models/ensemble.py.

The first one is the one that matters: the weight used at time t must not
depend on anything that only becomes knowable after t. An ensemble that
weights its members by how well they turned out to do is not a forecast,
it is a report, and it would look spectacular on validation and fall over
in production.

Run from the project root:   pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluation.metrics import threshold_weight
from src.models import ensemble as E

CFG = load_config()
LOW = CFG["target"]["thresholds"]["low"]
N_MODELS = 3


def toy(n_hours=400, n_models=N_MODELS, station="A", start="2024-01-01"):
    """One station, hourly, with three members and a known truth."""
    times = pd.date_range(start, periods=n_hours, freq="h")
    keys = pd.DataFrame({"station": station, "time": times})
    rng = np.random.default_rng(0)
    y_reg = 6 + np.sin(np.arange(n_hours) / 4) + rng.normal(0, .2, n_hours)
    y_cls = np.where(y_reg < LOW, 0, np.where(y_reg <= 7, 1, 2)).astype(float)
    # model 0 is good, model 1 is mediocre, model 2 is poor
    reg = np.column_stack([y_reg + rng.normal(0, s, n_hours)
                           for s in (0.1, 0.5, 1.5)])
    proba = np.zeros((n_hours, n_models, 3))
    for m in range(n_models):
        proba[:, m, :] = 0.2
        proba[np.arange(n_hours), m, y_cls.astype(int)] = 0.6
    proba /= proba.sum(axis=2, keepdims=True)
    return keys, y_reg, y_cls, reg, proba


# ---------------------------------------------------------------------------
# 1. No leakage
# ---------------------------------------------------------------------------
def test_a_huge_future_error_does_not_change_the_weight_now():
    """
    Wreck one model's prediction at target times that have NOT happened yet
    at the hour we check. Its weight at that hour must be untouched.
    """
    h = 24
    keys, y_reg, y_cls, reg, proba = toy()
    cfg = dict(alpha=5.0, sigma=1.0, window=168, temperature=0.5)

    _, w_before = E.atae("regression", keys, y_reg, y_cls, reg, proba,
                         cfg["alpha"], cfg["sigma"], cfg["window"],
                         cfg["temperature"], h)

    check_row = 200                      # the hour we inspect
    t_check = keys["time"].iloc[check_row]
    # rows whose target time is still in the future at t_check
    target_times = pd.to_datetime(keys["time"]) + pd.Timedelta(hours=h)
    future = (target_times > t_check).to_numpy()
    assert future.sum() > 0

    wrecked = reg.copy()
    wrecked[future, 0] = -9999.0         # model 0 becomes catastrophic, later
    _, w_after = E.atae("regression", keys, y_reg, y_cls, wrecked, proba,
                        cfg["alpha"], cfg["sigma"], cfg["window"],
                        cfg["temperature"], h)

    assert np.allclose(w_before[check_row], w_after[check_row]), \
        "the weight at t moved when only the future changed"
    # and every row at or before t_check is equally untouched
    assert np.allclose(w_before[:check_row + 1], w_after[:check_row + 1])


def test_the_weight_does_change_when_the_known_past_changes():
    """The opposite check: if the test above passes trivially, it is useless."""
    h = 24
    keys, y_reg, y_cls, reg, proba = toy()
    _, w_before = E.atae("regression", keys, y_reg, y_cls, reg, proba,
                         5.0, 1.0, 168, 0.5, h)
    check_row = 200
    t_check = keys["time"].iloc[check_row]
    target_times = pd.to_datetime(keys["time"]) + pd.Timedelta(hours=h)
    past = (target_times <= t_check).to_numpy()

    wrecked = reg.copy()
    wrecked[past, 0] = -9999.0           # model 0 was catastrophic, and we know
    _, w_after = E.atae("regression", keys, y_reg, y_cls, wrecked, proba,
                        5.0, 1.0, 168, 0.5, h)
    assert w_after[check_row, 0] < w_before[check_row, 0] - 1e-6, \
        "a known past disaster should cost that model weight"


def test_errors_outside_the_window_are_forgotten():
    """A disaster long ago must not still be held against a model."""
    h = 6
    keys, y_reg, y_cls, reg, proba = toy(n_hours=800)
    wrecked = reg.copy()
    wrecked[:50, 0] = -9999.0            # a disaster in the first 50 hours
    window = 72
    _, w = E.atae("regression", keys, y_reg, y_cls, wrecked, proba,
                  0.0, 1.0, window, 0.5, h)
    late = 700                           # far outside the 72-hour window
    assert w[late, 0] == pytest.approx(w[late].max(), abs=1e-6), \
        "the long-forgotten disaster is still being punished"


# ---------------------------------------------------------------------------
# 2. Weights are a probability distribution
# ---------------------------------------------------------------------------
def test_weights_are_non_negative_and_sum_to_one():
    keys, y_reg, y_cls, reg, proba = toy()
    for task in ["regression", "classification"]:
        for temperature in [0.25, 1.0]:
            _, w = E.atae(task, keys, y_reg, y_cls, reg, proba, 5.0, 1.0,
                          168, temperature, 24)
            assert (w >= 0).all(), "a negative weight makes no sense"
            assert np.allclose(w.sum(axis=1), 1.0), "weights must sum to 1"
            assert not np.isnan(w).any()


def test_rows_with_no_history_get_equal_weights():
    """At the very start nothing is known, so no model may be preferred."""
    keys, y_reg, y_cls, reg, proba = toy()
    _, w = E.atae("regression", keys, y_reg, y_cls, reg, proba, 5.0, 1.0,
                  168, 0.5, 48)
    assert np.allclose(w[0], 1.0 / reg.shape[1]), \
        "the first hour cannot know anything yet"


# ---------------------------------------------------------------------------
# 3. With nothing to distinguish them, ATAE is the simple average
# ---------------------------------------------------------------------------
def test_equal_errors_and_alpha_zero_give_the_simple_average():
    keys, y_reg, y_cls, reg, proba = toy()
    # make every model identical, so their errors are identical too
    same = np.column_stack([reg[:, 0]] * reg.shape[1])
    same_proba = np.stack([proba[:, 0, :]] * proba.shape[1], axis=1)

    pred, w = E.atae("regression", keys, y_reg, y_cls, same, same_proba,
                     0.0, 1.0, 168, 0.5, 24)
    avg, _ = E.simple_average("regression", same, same_proba)
    assert np.allclose(w, 1.0 / reg.shape[1]), \
        f"identical models must get identical weights, got {w[100]}"
    assert np.allclose(pred, avg)

    pred_c, w_c = E.atae("classification", keys, y_reg, y_cls, same,
                         same_proba, 0.0, 1.0, 168, 0.5, 24)
    avg_c, _ = E.simple_average("classification", same, same_proba)
    assert np.allclose(pred_c, avg_c)


def test_combine_is_a_weighted_mean():
    reg = np.array([[1.0, 3.0]])
    proba = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    w = np.array([[0.25, 0.75]])
    assert E.combine("regression", w, reg, proba)[0] == pytest.approx(2.5)
    assert np.allclose(E.combine("classification", w, reg, proba)[0],
                       [0.25, 0.0, 0.75])


# ---------------------------------------------------------------------------
# 4. The threshold weight w(y)
# ---------------------------------------------------------------------------
def test_threshold_weight_peaks_at_the_low_threshold():
    ys = np.linspace(0, 15, 301)
    w = threshold_weight(ys, alpha=5.0, sigma=1.0)
    assert ys[np.argmax(w)] == pytest.approx(LOW, abs=0.05)
    assert w.max() == pytest.approx(1.0 + 5.0)


def test_threshold_weight_is_one_far_from_the_threshold():
    assert threshold_weight(LOW + 12, 5.0, 1.0) == pytest.approx(1.0, abs=1e-6)
    assert threshold_weight(LOW - 12, 5.0, 1.0) == pytest.approx(1.0, abs=1e-6)


def test_alpha_zero_switches_the_threshold_idea_off():
    ys = np.array([0.0, 5.0, 9.0, 20.0])
    assert np.allclose(threshold_weight(ys, 0.0, 1.0), 1.0)


def test_a_wider_sigma_spreads_the_weight():
    narrow = threshold_weight(LOW + 2, 5.0, 0.5)
    wide = threshold_weight(LOW + 2, 5.0, 2.0)
    assert wide > narrow


def test_threshold_weighting_shifts_weight_towards_the_model_that_is_good_near_five():
    """
    Two models: one accurate everywhere except near 5 mg/L, one accurate
    only near 5. With alpha = 0 the first should win; with a large alpha the
    second should gain.
    """
    n = 600
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    keys = pd.DataFrame({"station": "A", "time": times})
    rng = np.random.default_rng(3)
    y = rng.uniform(2, 10, n)
    near = np.abs(y - LOW) < 1.0
    a = y + np.where(near, rng.normal(0, 2.0, n), rng.normal(0, 0.1, n))
    b = y + np.where(near, rng.normal(0, 0.1, n), rng.normal(0, 1.0, n))
    reg = np.column_stack([a, b])
    proba = np.full((n, 2, 3), 1 / 3)
    y_cls = np.where(y < LOW, 0, np.where(y <= 7, 1, 2)).astype(float)

    _, w_off = E.atae("regression", keys, y, y_cls, reg, proba, 0.0, 1.0,
                      336, 0.5, 6)
    _, w_on = E.atae("regression", keys, y, y_cls, reg, proba, 10.0, 0.5,
                     336, 0.5, 6)
    late = slice(400, None)
    assert w_on[late, 1].mean() > w_off[late, 1].mean(), \
        "threshold weighting did not favour the model that is good near 5"
