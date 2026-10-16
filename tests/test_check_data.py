"""
Small checks for the low-DO event definition in src/data/check_data.py.

An event = DO drops below 5 mg/L after at least 24 h at or above 5 mg/L,
so one long sag counts once instead of once per hour.

Run from the project root:   pytest -q
"""

import numpy as np
import pandas as pd

from src.data.check_data import class_shares, count_events

IDX = pd.date_range("2023-01-01", periods=200, freq="h")


def series(value=8.0):
    """200 hours of DO, all 'Good' by default."""
    return pd.Series(value, index=IDX)


def test_no_low_hours_means_no_events():
    assert count_events(series()) == 0


def test_one_sag_is_one_event():
    s = series()
    s[50:60] = 3.0
    assert count_events(s) == 1


def test_two_sags_far_apart_are_two_events():
    s = series()
    s[50:60] = 3.0
    s[90:95] = 3.0          # 30 recovered hours in between
    assert count_events(s) == 2


def test_two_sags_close_together_are_one_event():
    s = series()
    s[50:60] = 3.0
    s[65:70] = 3.0          # only 5 recovered hours, not a new event
    assert count_events(s) == 1


def test_low_from_the_first_hour_counts():
    s = series()
    s[0:10] = 3.0
    assert count_events(s) == 1


def test_a_data_gap_is_not_a_recovery():
    s = series()
    s[50:60] = 3.0
    s[60:85] = np.nan       # sensor offline, not proof the river recovered
    s[85:90] = 3.0
    assert count_events(s) == 1


def test_class_shares_ignore_missing_hours():
    s = pd.Series([3.0, 3.0, 6.0, 8.0, np.nan], index=IDX[:5])
    shares, n_obs = class_shares(s)
    assert n_obs == 4
    assert shares["Low"] == 50.0
    assert shares["Medium"] == 25.0
    assert shares["Good"] == 25.0
