"""
Tests for the sequence dataset in src/models/sequences.py.

The leakage questions are the same as in step 2, but the shape of the data
is different: here a "row" is a 48-hour window, so the thing to prove is
that a window ending at t contains nothing after t and nothing from another
station.

Run from the project root:   pytest -q
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path
from src.models import sequences as S

CFG = load_config()
SEQ_LEN = S.SEQ_LEN


def toy_data(n_hours=200):
    """
    Two stations whose arrays are easy to recognise: station A holds the
    numbers 0, 1, 2, ... and station B holds 10000, 10001, ... so any mixing
    of the two is obvious.
    """
    a = np.tile(np.arange(n_hours, dtype="float32")[:, None], (1, S.N_CHANNELS))
    b = a + 10_000
    arrays = {"A": a, "B": b}
    station_ids = {"A": 0, "B": 1}
    # one row per station, both ending at hour 150
    index = np.array([[0, 150], [1, 150]], dtype="int64")
    meta = pd.DataFrame({"station": ["A", "B"],
                         "time": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                         "do_now": [1.0, 2.0],
                         "y_reg_24": [5.0, 6.0], "y_cls_24": [1, 1]})
    return S.SequenceData(arrays, station_ids, index, meta, None, None)


# ---------------------------------------------------------------------------
# A window ends at t: nothing after t may be in it
# ---------------------------------------------------------------------------
def test_window_ends_at_t_and_looks_backwards():
    data = toy_data()
    x, _ = data.batch([0])
    assert x.shape == (1, SEQ_LEN, S.N_CHANNELS)
    # the last row of the window is hour t itself
    assert x[0, -1, 0] == pytest.approx(150.0)
    # the first row is t - 47
    assert x[0, 0, 0] == pytest.approx(150.0 - SEQ_LEN + 1)
    # and every hour in between is in order, all <= t
    assert (x[0, :, 0] <= 150.0).all()
    assert np.allclose(np.diff(x[0, :, 0]), 1.0)


def test_changing_data_after_t_does_not_change_the_window():
    data = toy_data()
    before, _ = data.batch([0])
    # scribble over everything after hour t
    data._stack[0][151:] = -999.0
    after, _ = data.batch([0])
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# One window, one station
# ---------------------------------------------------------------------------
def test_windows_never_mix_stations():
    data = toy_data()
    x, codes = data.batch([0, 1])
    assert (x[0] < 1_000).all(), "station A window picked up station B data"
    assert (x[1] >= 10_000).all(), "station B window picked up station A data"
    assert codes.tolist() == [0, 1]


def test_changing_station_b_leaves_station_a_alone():
    data = toy_data()
    before, _ = data.batch([0])
    data._stack[1][:] = -1.0           # wreck station B entirely
    after, _ = data.batch([0])
    assert np.array_equal(before, after)


def test_batch_order_matches_the_row_ids_asked_for():
    """The gather is done per station, so the rows must still come back in order."""
    data = toy_data()
    x, codes = data.batch([1, 0])
    assert (x[0] >= 10_000).all()      # station B first, as asked
    assert (x[1] < 1_000).all()
    assert codes.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# Forward fill: past only
# ---------------------------------------------------------------------------
def test_fill_past_only_never_fills_backwards():
    idx = pd.date_range("2023-01-01", periods=6, freq="h")
    df = pd.DataFrame({"do_mgl": [np.nan, 5.0, np.nan, np.nan, 8.0, np.nan]},
                      index=idx)
    out = S.fill_past_only(df)["do_mgl"]
    assert np.isnan(out.iloc[0]), "a leading NaN must not be filled from later data"
    assert out.iloc[2] == 5.0 and out.iloc[3] == 5.0     # carried forward
    assert out.iloc[5] == 8.0


# ---------------------------------------------------------------------------
# The real data: the scaler is fitted on train, and the rows match step 3
# ---------------------------------------------------------------------------
def _have_processed():
    return all(os.path.exists(project_path(CFG["paths"][k]))
               for k in ["hourly", "train", "val"])


@pytest.fixture(scope="module")
def two_station_data():
    if not _have_processed():
        pytest.skip("run step 2 first")
    stations = sorted(pd.read_csv(project_path(CFG["paths"]["stations"]),
                                  dtype={"site_no": str})["site_no"])[:2]
    train = S.prepare("train", stations=stations)
    val = S.prepare("val", scaler=train.scaler, medians=train.medians,
                    stations=stations)
    return train, val, stations


def test_the_scaler_is_fitted_on_train_only(two_station_data):
    train, val, _ = two_station_data
    # validation reuses the very same fitted object - nothing is refitted
    assert val.scaler is train.scaler
    assert val.medians is train.medians

    # and fitting it again on train alone gives the same numbers
    from sklearn.preprocessing import StandardScaler
    again = StandardScaler().fit(
        np.concatenate([train.arrays[s] * train.scaler.scale_
                        + train.scaler.mean_ for s in train.arrays]))
    assert np.allclose(again.mean_, train.scaler.mean_, atol=1e-4)


def test_scaled_train_data_has_mean_zero(two_station_data):
    train, _, _ = two_station_data
    stacked = np.concatenate([train.arrays[s] for s in train.arrays])
    # constant channels (a station with no turbidity) stay at 0 and are fine
    assert np.nanmax(np.abs(stacked.mean(axis=0))) < 1e-3


def test_dl_and_ml_rows_are_the_same(two_station_data):
    """
    The deep models must be scored on exactly the rows the tree models were
    scored on, or the comparison in the report is meaningless.
    """
    _, val, stations = two_station_data
    ml = pd.read_parquet(project_path(CFG["paths"]["val"]),
                         columns=["station", "time"])
    ml = ml[ml["station"].isin(stations)]
    dl_keys = set(zip(val.meta["station"], val.meta["time"]))
    ml_keys = set(zip(ml["station"], ml["time"]))
    assert dl_keys == ml_keys, (
        f"{len(ml_keys - dl_keys)} rows only in the ML split, "
        f"{len(dl_keys - ml_keys)} only in the DL split")


def test_targets_line_up_with_the_processed_file(two_station_data):
    _, val, stations = two_station_data
    ml = pd.read_parquet(project_path(CFG["paths"]["val"]),
                         columns=["station", "time", "y_reg_24"])
    ml = ml[ml["station"].isin(stations)]
    merged = val.meta[["station", "time", "y_reg_24"]].merge(
        ml, on=["station", "time"], suffixes=("_dl", "_ml"))
    assert len(merged) == len(val.meta)
    both = merged.dropna(subset=["y_reg_24_dl", "y_reg_24_ml"])
    assert np.allclose(both["y_reg_24_dl"], both["y_reg_24_ml"])
