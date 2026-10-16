"""
Tests for the feature building in src/features/build_features.py.

The important ones are the leakage tests: they are what stops a model from
quietly cheating. They work on a small synthetic station, so they are fast
and do not need the downloaded data.

Run from the project root:   pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import build_features as B

CFG = load_config()
WATER_VARS = CFG["features"]["water_vars"]
WEATHER_VARS = CFG["download"]["weather_vars"]
AIR_VARS = CFG["download"]["air_vars"]
MASK = CFG["cleaning"]["mask_suffix"]
HOURS = 400


def fake_station(seed=0):
    """
    One synthetic station on a complete hourly index, in the same shape as a
    slice of data/interim/hourly.parquet.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=HOURS, freq="h")
    df = pd.DataFrame(index=idx)
    # a daily cycle plus noise, so DO crosses the Low threshold sometimes
    hour = idx.hour.to_numpy()
    df["do_mgl"] = 6 + 2.5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, .4, HOURS)
    df["water_temp_c"] = 18 + 3 * np.sin(2 * np.pi * hour / 24)
    df["spec_cond_uscm"] = 300 + rng.normal(0, 10, HOURS)
    df["ph"] = 7.2 + rng.normal(0, .1, HOURS)
    df["turbidity_fnu"] = np.abs(rng.normal(8, 3, HOURS))
    df["discharge_cfs"] = np.abs(rng.normal(500, 50, HOURS))
    for v in WATER_VARS:
        df[v + MASK] = 0
    for v in WEATHER_VARS:
        df[v] = rng.normal(10, 2, HOURS)
    df["precipitation"] = np.where(rng.random(HOURS) < .1, rng.random(HOURS) * 5, 0)
    for v in AIR_VARS:
        df[v] = rng.normal(20, 5, HOURS)
    return df


def build(df, station="A", station_id=0, lat=30.0, lon=-82.0):
    return B.build_station(df, station, station_id, lat, lon).set_index("time")


# ---------------------------------------------------------------------------
# 2.5 no leakage: nothing after t may change a feature at t
# ---------------------------------------------------------------------------
def test_features_at_t_ignore_everything_after_t():
    df = fake_station()
    t = df.index[200]
    before = build(df)

    # scramble every value strictly after t
    altered = df.copy()
    after = altered.index > t
    for c in altered.columns:
        if c.endswith(MASK):
            continue
        altered.loc[after, c] = altered.loc[after, c] * -3.0 + 99.0
    after_build = build(altered)

    feature_cols = [c for c in before.columns
                    if not c.startswith(("y_reg_", "y_cls_", "station"))]
    row_before = before.loc[t, feature_cols]
    row_after = after_build.loc[t, feature_cols]
    changed = [c for c in feature_cols
               if not (pd.isna(row_before[c]) and pd.isna(row_after[c]))
               and not np.isclose(float(row_before[c]), float(row_after[c]),
                                  equal_nan=True)]
    assert changed == [], f"these features looked into the future: {changed}"


def test_every_row_up_to_t_is_unchanged_not_just_t():
    """The same check for all rows at or before t, not only the single row."""
    df = fake_station(seed=1)
    cut = 150
    t = df.index[cut]
    altered = df.copy()
    altered.loc[altered.index > t, "do_mgl"] += 50.0

    a, b = build(df), build(altered)
    cols = [c for c in a.columns if not c.startswith(("y_reg_", "y_cls_"))]
    pd.testing.assert_frame_equal(a.iloc[:cut + 1][cols], b.iloc[:cut + 1][cols])


# ---------------------------------------------------------------------------
# 2.5 per station: one station's features never depend on another's data
# ---------------------------------------------------------------------------
def test_gap_filling_does_not_use_future_values():
    """
    The cleaning step fills gaps of up to 3 h. Carrying the last observed
    value forward can only use the past; interpolating between both sides of
    a gap would not. Changing the reading AFTER a gap must leave the filled
    values inside that gap untouched.
    """
    from src.data import clean as C

    idx = pd.date_range("2023-01-01", periods=12, freq="h")
    base = pd.DataFrame({"do_mgl": [6.0, 6.5, np.nan, np.nan, 8.0,
                                    7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]},
                        index=idx)
    filled = C.fill_short_gaps(base.copy(), ["do_mgl"])["do_mgl"]

    # the gap is filled with the last value BEFORE it, not a blend
    assert filled.iloc[2] == pytest.approx(6.5)
    assert filled.iloc[3] == pytest.approx(6.5)

    # changing the value after the gap must not move the filled values
    moved = base.copy()
    moved.loc[idx[4], "do_mgl"] = 200.0
    filled_moved = C.fill_short_gaps(moved, ["do_mgl"])["do_mgl"]
    assert filled_moved.iloc[2] == pytest.approx(filled.iloc[2])
    assert filled_moved.iloc[3] == pytest.approx(filled.iloc[3])


def test_gaps_longer_than_the_limit_stay_missing():
    from src.data import clean as C

    limit = CFG["cleaning"]["max_gap_fill_hours"]
    idx = pd.date_range("2023-01-01", periods=12, freq="h")
    col = pd.Series([6.0] + [np.nan] * 6 + [7.0] * 5, index=idx)
    filled = C.fill_short_gaps(pd.DataFrame({"do_mgl": col}), ["do_mgl"])["do_mgl"]
    # only the first `limit` hours of the long gap are carried forward
    assert filled.iloc[1:1 + limit].notna().all()
    assert filled.iloc[1 + limit:7].isna().all()


def test_cleaning_keeps_negative_discharge():
    """Negative discharge is the tide pushing water upstream, not a fault."""
    from src.data import clean as C

    idx = pd.date_range("2023-01-01", periods=4, freq="h")
    df = pd.DataFrame({"discharge_cfs": [500.0, -500.0, -12000.0, 300.0]},
                      index=idx)
    out, removed = C.apply_ranges(df.copy())
    assert out["discharge_cfs"].notna().all()
    assert "discharge_cfs" not in removed


def test_station_a_features_do_not_depend_on_station_b():
    a = fake_station(seed=2)
    b = fake_station(seed=3)
    panel = pd.concat([a.assign(station="A"), b.assign(station="B")])

    def features_for_a(p):
        g = p[p["station"] == "A"].drop(columns="station")
        return build(g, station="A")

    before = features_for_a(panel)

    altered = panel.copy()
    is_b = altered["station"] == "B"
    for c in altered.columns:
        if c == "station" or c.endswith(MASK):
            continue
        altered.loc[is_b, c] = altered.loc[is_b, c] * 10.0 + 1000.0

    pd.testing.assert_frame_equal(before, features_for_a(altered))


# ---------------------------------------------------------------------------
# 2.5 targets
# ---------------------------------------------------------------------------
def test_y_reg_6_is_the_do_six_hours_later():
    df = fake_station(seed=4)
    f = build(df)
    for i in [100, 200, 300]:
        t = df.index[i]
        assert f.loc[t, "y_reg_6"] == pytest.approx(df["do_mgl"].iloc[i + 6])


def test_y_reg_is_nan_when_the_future_do_was_missing():
    """An interpolated DO value may be used as an input, never as a target."""
    df = fake_station(seed=5)
    i = 100
    df.loc[df.index[i + 6], "do_mgl" + MASK] = 1       # was filled, not observed
    f = build(df)
    assert pd.isna(f.loc[df.index[i], "y_reg_6"])
    assert pd.isna(f.loc[df.index[i], "y_cls_6"])


def test_y_cls_follows_the_configured_thresholds():
    low = CFG["target"]["thresholds"]["low"]
    good = CFG["target"]["thresholds"]["good"]
    df = fake_station(seed=6)
    f = build(df)
    y, c = f["y_reg_6"], f["y_cls_6"]
    ok = y.notna()
    assert (c[ok & (y < low)] == 0).all()
    assert (c[ok & (y >= low) & (y <= good)] == 1).all()
    assert (c[ok & (y > good)] == 2).all()


# ---------------------------------------------------------------------------
# 2.5 split: no target may cross a split boundary
# ---------------------------------------------------------------------------
def test_no_target_time_crosses_a_split_boundary():
    """
    Read the real split files and check that the furthest target time
    (t + 48 h) of every train row is still in a train year, and likewise
    for validation.
    """
    import os
    from src.config import project_path
    paths = {k: project_path(CFG["paths"][k]) for k in ["train", "val", "test"]}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("run `python -m src.features.build_features` first")

    max_h = max(CFG["horizons"])
    years = {"train": set(CFG["split"]["train_years"]),
             "val": set(CFG["split"]["val_years"]),
             "test": set(CFG["split"]["test_years"])}
    for name, path in paths.items():
        t = pd.read_parquet(path, columns=["time"])["time"]
        assert set(t.dt.year.unique()) <= years[name], \
            f"{name} contains rows from the wrong year"
        far = (t + pd.Timedelta(hours=max_h)).dt.year
        if name == "train":
            assert not far.isin(years["val"]).any(), \
                "a train target lands in the validation year"
        if name == "val":
            assert not far.isin(years["test"]).any(), \
                "a validation target lands in the test year"


def test_splits_do_not_share_any_station_hour():
    import os
    from src.config import project_path
    paths = {k: project_path(CFG["paths"][k]) for k in ["train", "val", "test"]}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("run `python -m src.features.build_features` first")
    seen = {}
    for name, path in paths.items():
        d = pd.read_parquet(path, columns=["station", "time"])
        seen[name] = set(zip(d["station"], d["time"]))
    assert not seen["train"] & seen["val"]
    assert not seen["val"] & seen["test"]
    assert not seen["train"] & seen["test"]


# ---------------------------------------------------------------------------
# a few sanity checks on the hand-written features
# ---------------------------------------------------------------------------
def test_hours_since_true_counts_backwards_only():
    idx = pd.date_range("2023-01-01", periods=10, freq="h")
    flag = pd.Series([False, True, False, False, True, False, False, False, False, False],
                     index=idx)
    got = B.hours_since_true(flag, cap=720).tolist()
    assert got == [720, 0, 1, 2, 0, 1, 2, 3, 4, 5]


def test_solubility_matches_published_values():
    """Benson-Krause solubility: ~14.6 mg/L at 0 C, 11.3 at 10 C, 8.26 at 25 C."""
    got = B.do_solubility(np.array([0.0, 10.0, 25.0]))
    assert got[0] == pytest.approx(14.62, abs=0.05)
    assert got[1] == pytest.approx(11.29, abs=0.05)
    assert got[2] == pytest.approx(8.26, abs=0.05)


def test_rolling_windows_include_the_current_hour():
    df = fake_station(seed=7)
    f = build(df)
    i = 100
    t = df.index[i]
    window = df["do_mgl"].iloc[i - 5:i + 1]          # 6 hours ending at t
    assert f.loc[t, "do_mgl_roll6_mean"] == pytest.approx(window.mean())
    assert f.loc[t, "do_mgl_roll6_max"] == pytest.approx(window.max())
