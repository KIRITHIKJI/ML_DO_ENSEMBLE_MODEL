"""
Tests for step 7: the cross-fitting protocol, the CAE inputs and the
warning threshold.

The protocol is the whole point of step 7, so most of these check the
protocol rather than the model: a method that learns from 2024 must never
be scored on rows it learned from, and nothing it uses at time t may come
from after t.

Run from the project root:   pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models import context_ensemble as CAE
from src.models import crossfit as X
from src.models import run_step7 as R

CFG = load_config()
CV = CFG["crossfit"]
LOW = CFG["target"]["thresholds"]["low"]
YEAR = CFG["split"]["val_years"][0]


@pytest.fixture(scope="module")
def year_times():
    """Every hour of the validation year."""
    return pd.Series(pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31 23:00",
                                   freq="h"))


# ---------------------------------------------------------------------------
# 1. Cross-fitting: no row is predicted by a model that could have seen it
# ---------------------------------------------------------------------------
def test_every_row_is_held_out_exactly_once(year_times):
    seen = np.zeros(len(year_times), dtype=int)
    for k in X.all_folds():
        _, test = X.split_indices(year_times, k)
        seen[test] += 1
    assert (seen == 1).all(), "every hour must be predicted exactly once"


def test_training_rows_never_overlap_the_held_out_fold(year_times):
    for k in X.all_folds():
        train, test = X.split_indices(year_times, k)
        assert not set(train) & set(test), f"fold {k} trains on its own rows"


def test_training_rows_respect_the_purge(year_times):
    """
    No training row may lie within the purge of the held-out fold - that is
    what stops a 48 h target or a 336 h error window reaching into it.
    """
    purge = pd.Timedelta(days=int(CV["purge_days"]))
    for k in X.all_folds():
        train, _ = X.split_indices(year_times, k)
        start, end = X.fold_bounds(k)
        t = year_times.iloc[train]
        assert (t.lt(start - purge) | t.gt(end + purge)).all(), \
            f"fold {k} has training rows inside the purge window"


def test_the_purge_is_wide_enough_for_the_longest_horizon_and_window(year_times):
    """The purge has to cover both leaks, not just the obvious one."""
    purge_hours = int(CV["purge_days"]) * 24
    assert purge_hours >= max(CFG["horizons"]), "purge shorter than the horizon"
    assert purge_hours >= CFG["context_ensemble"]["error_window_hours"], \
        "purge shorter than the error window"


def test_a_target_from_a_training_row_cannot_land_in_the_held_out_fold(year_times):
    max_h = max(CFG["horizons"])
    for k in X.all_folds():
        train, _ = X.split_indices(year_times, k)
        start, end = X.fold_bounds(k)
        targets = year_times.iloc[train] + pd.Timedelta(hours=max_h)
        inside = ((targets >= start) & (targets <= end)).sum()
        assert inside == 0, f"fold {k}: {inside} training targets land in it"


def test_inner_split_is_time_ordered(year_times):
    """Early stopping must use LATER hours, not a random scatter of them."""
    train, _ = X.split_indices(year_times, 3)
    inner_train, inner_valid = X.inner_split(year_times, train)
    assert len(inner_valid) > 0
    assert year_times.iloc[inner_train].max() <= year_times.iloc[inner_valid].min()


# ---------------------------------------------------------------------------
# 2. CAE inputs at t use only data up to t
# ---------------------------------------------------------------------------
def toy_members(n=600, n_models=3, station="A"):
    times = pd.date_range(f"{YEAR}-01-01", periods=n, freq="h")
    keys = pd.DataFrame({"station": station, "time": times})
    rng = np.random.default_rng(0)
    y_reg = 6 + np.sin(np.arange(n) / 5) + rng.normal(0, .3, n)
    y_cls = np.where(y_reg < LOW, 0, np.where(y_reg <= 7, 1, 2)).astype(float)
    reg = np.column_stack([y_reg + rng.normal(0, s, n) for s in (.2, .6, 1.2)])
    proba = np.full((n, n_models, 3), 1 / 3)
    for m in range(n_models):
        proba[np.arange(n), m, y_cls.astype(int)] = 0.6
    proba /= proba.sum(axis=2, keepdims=True)
    ctx = pd.DataFrame(
        {c: rng.normal(0, 1, n).astype("float32")
         for c in CFG["context_ensemble"]["context_features"]})
    return keys, y_reg, y_cls, reg, proba, ctx


def test_cae_inputs_at_t_ignore_everything_after_t():
    """
    The recent-error features are the only part of the input that looks
    backwards over time, so they are the part that could leak. Wreck the
    members at target times that have not happened yet and the whole input
    row at t must be unchanged.
    """
    h = 24
    keys, y_reg, y_cls, reg, proba, ctx = toy_members()
    before, _ = CAE.build_features("regression", keys, y_reg, y_cls, reg,
                                   proba, h, ctx)
    row = 400
    t_check = keys["time"].iloc[row]
    # Rows made AFTER t. A member's own forecast made at t is an input at t
    # and is supposed to be there; what must not reach t is anything a
    # later hour produced or revealed.
    future = (pd.to_datetime(keys["time"]) > t_check).to_numpy()
    assert future.sum() > 0

    wrecked = reg.copy()
    wrecked[future, 0] = -9999.0
    after, _ = CAE.build_features("regression", keys, y_reg, y_cls, wrecked,
                                  proba, h, ctx)

    # The feature ROW at t must be identical. (Rows shortly before t also
    # had their predictions overwritten - their own targets are still in the
    # future - and those rows' own `pred_*` inputs are expected to change.
    # What must not change is anything feeding the forecast made AT t.)
    pd.testing.assert_series_equal(before.iloc[row], after.iloc[row],
                                   check_names=False)

    # in particular the recent-error features, the only ones that look back
    # over time and so the only ones that could leak
    err_cols = [c for c in before.columns if c.startswith("recent_err_")]
    assert np.allclose(before.loc[row, err_cols], after.loc[row, err_cols]), \
        "a recent-error feature at t moved when only the future changed"


def test_cae_recent_error_features_do_change_with_the_known_past():
    """If the test above passes because nothing ever changes, it is useless."""
    h = 24
    keys, y_reg, y_cls, reg, proba, ctx = toy_members()
    before, names = CAE.build_features("regression", keys, y_reg, y_cls, reg,
                                       proba, h, ctx)
    row = 400
    t_check = keys["time"].iloc[row]
    past = ((pd.to_datetime(keys["time"]) + pd.Timedelta(hours=h))
            <= t_check).to_numpy()
    wrecked = reg.copy()
    wrecked[past, 0] = -9999.0
    after, _ = CAE.build_features("regression", keys, y_reg, y_cls, wrecked,
                                  proba, h, ctx)
    col = names.index(f"recent_err_{CAE.MEMBERS[0]}")
    assert after.iloc[row, col] > before.iloc[row, col], \
        "a known past disaster should show up in the recent-error feature"


def test_cae_has_no_station_id_input():
    keys, y_reg, y_cls, reg, proba, ctx = toy_members()
    for task in ["regression", "classification"]:
        _, names = CAE.build_features(task, keys, y_reg, y_cls, reg, proba,
                                      24, ctx)
        assert not any("station_id" in n for n in names), \
            f"{task} CAE inputs contain station_id"
        assert not any(n == "station" for n in names)
    assert CFG["context_ensemble"]["use_station_id"] is False


def test_cae_feature_count_matches_the_design():
    keys, y_reg, y_cls, reg, proba, ctx = toy_members()
    n_models = reg.shape[1]
    n_ctx = len(CFG["context_ensemble"]["context_features"])
    f_r, _ = CAE.build_features("regression", keys, y_reg, y_cls, reg, proba,
                                24, ctx)
    f_c, _ = CAE.build_features("classification", keys, y_reg, y_cls, reg,
                                proba, 24, ctx)
    # members + 2 disagreement + recent errors + context
    assert f_r.shape[1] == n_models + 2 + n_models + n_ctx
    assert f_c.shape[1] == 3 * n_models + 2 + n_models + n_ctx


# ---------------------------------------------------------------------------
# 3. tau is chosen without the held-out fold
# ---------------------------------------------------------------------------
def test_tau_is_chosen_without_the_held_out_fold():
    """
    Build a year where the held-out fold wants a very different threshold
    from the rest. The tau chosen on the training rows must follow the
    training rows, not the fold.
    """
    times = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31 23:00", freq="h")
    n = len(times)
    rng = np.random.default_rng(5)
    folds = X.fold_of(pd.Series(times))
    held = folds == 2

    class_now = np.full(n, 2.0)                  # never Low right now
    y_cls = np.zeros(n)
    proba = np.zeros((n, 3))
    # training rows: onsets come with a clearly high P(Low)
    y_cls[~held] = np.where(rng.random((~held).sum()) < .3, 0, 2)
    proba[~held, 0] = np.where(y_cls[~held] == 0, 0.8, 0.05)
    # held-out fold: the model is badly calibrated, onsets look like 0.2
    y_cls[held] = np.where(rng.random(held.sum()) < .3, 0, 2)
    proba[held, 0] = np.where(y_cls[held] == 0, 0.2, 0.01)
    proba[:, 2] = 1 - proba[:, 0]

    train, test = X.split_indices(pd.Series(times), 2)
    tau, recall, rate = R.choose_tau(class_now, y_cls, proba, train)
    assert tau is not None
    # the training rows separate at 0.8, so tau should sit well above the
    # 0.2 the held-out fold would have wanted
    assert tau > 0.3, f"tau {tau} looks like it was fitted on the held-out fold"

    tau_if_cheating, _, _ = R.choose_tau(class_now, y_cls, proba, test)
    assert tau_if_cheating is not None and tau_if_cheating < tau, \
        "the test is not biting: the fold wants the same tau anyway"


def test_the_warning_rule_respects_its_false_alarm_budget():
    rng = np.random.default_rng(6)
    n = 20000
    class_now = np.full(n, 2.0)
    y_cls = np.where(rng.random(n) < .1, 0.0, 2.0)
    p_low = np.clip(rng.normal(np.where(y_cls == 0, .5, .1), .2), 0, 1)
    proba = np.column_stack([p_low, np.zeros(n), 1 - p_low])
    rows = np.arange(n)
    tau, recall, rate = R.choose_tau(class_now, y_cls, proba, rows)
    assert tau is not None
    assert rate <= CFG["warning_rule"]["max_false_alarm_rate"] + 1e-9


def test_apply_warning_issues_low_exactly_when_p_low_reaches_tau():
    #                 P(Low) P(Med) P(Good)
    proba = np.array([[.30, .30, .40],      # Good is the better non-Low
                      [.10, .80, .10],      # Medium is
                      [.55, .20, .25]])     # Good is
    got = R.apply_warning(proba, 0.30)
    assert got.tolist() == [0, 1, 0], "P(Low) == tau must still warn"
    got = R.apply_warning(proba, 0.60)
    # below tau the rule picks the better of Medium and Good, never Low
    assert got.tolist() == [2, 1, 2]
    assert 0 not in R.apply_warning(proba, 0.99).tolist()
