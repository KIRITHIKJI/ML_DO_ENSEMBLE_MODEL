"""
Tests for step 12b - the stale-data guard, and early warning vs ongoing low.

Two things a live forecast must not do.

It must not build a prediction on a reading from days ago. Every input the
models take is a lag or a rolling statistic of the DO the sensor last sent,
so a stale reading yields a confident answer about a river nobody has looked
at since - and that answer is indistinguishable from a fresh one.

And it must not present "the water is already below 5 mg/L" as an early
warning. An operator can read that off the sensor; counting it as a warning
flatters the system at exactly the point where it claims its value.

The fixtures here come from `tests/test_deploy.py`, which already builds a
forecast file carrying one station of each alert type.

Run from the project root:   pytest -q tests/test_deploy_12b.py
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path
from src.deploy import common as D

# the shared forecast fixture and its constants
from tests.test_deploy import (SCHEMA_12B, STALE_STATION,  # noqa: F401
                               client, fixture_forecast)

CFG = load_config()
HORIZONS = CFG["horizons"]
DO = CFG["target"]["column"]
ALERT = D.RUN["alert_types"]


# ===========================================================================
# The configured limit
# ===========================================================================
def test_the_stale_limit_is_configured_and_sane():
    limit = float(D.RUN["max_data_age_hours"])
    assert limit > 0
    assert limit <= min(HORIZONS), (
        "a reading older than the shortest horizon would make the "
        "forecast about an hour that has already passed")
    assert D.RUN["stale_status"] and D.RUN["fresh_status"]
    assert set(ALERT) == {"early_warning", "ongoing_low", "none", "stale"}


def test_the_limit_is_read_from_the_config_not_hard_coded():
    src = open(project_path("src", "deploy", "run.py"),
               encoding="utf-8").read()
    assert 'RUN["max_data_age_hours"]' in src
    assert "max_data_age_hours" in open(
        project_path("config_deploy.yaml"), encoding="utf-8").read()


# ===========================================================================
# alert_type, on hand-made rows: all four cases
# ===========================================================================
@pytest.mark.parametrize("class_now,warning,stale,expected", [
    (2.0, True, False, "early_warning"),    # Good now, rule fires
    (1.0, True, False, "early_warning"),    # Medium now, rule fires
    (0.0, True, False, "ongoing_low"),      # already Low, rule fires
    (2.0, False, False, "none"),            # rule does not fire
    (1.0, False, False, "none"),
    (0.0, False, False, "none"),            # already Low, rule silent
    (2.0, True, True, "stale"),             # stale beats everything
    (0.0, True, True, "stale"),
    (1.0, False, True, "stale"),
])
def test_alert_type_on_hand_made_rows(class_now, warning, stale, expected):
    from src.deploy import run as RUNNER
    assert RUNNER.classify_alert(class_now, warning, stale) == expected


def test_every_alert_type_is_reachable():
    """All four names are produced by the classifier, not just declared."""
    from src.deploy import run as RUNNER
    produced = {
        RUNNER.classify_alert(2.0, True),
        RUNNER.classify_alert(0.0, True),
        RUNNER.classify_alert(2.0, False),
        RUNNER.classify_alert(2.0, True, True),
    }
    assert produced == set(ALERT.values())


def test_a_warning_with_an_unknown_current_class_is_an_early_warning():
    """
    Between hiding a warning and over-stating one, a warning system says
    something. A firing rule with no known current class is reported as an
    early warning rather than swallowed.
    """
    from src.deploy import run as RUNNER
    assert RUNNER.classify_alert(np.nan, True) == "early_warning"
    assert RUNNER.classify_alert(None, True) == "early_warning"
    assert RUNNER.classify_alert(np.nan, False) == "none"


# ===========================================================================
# Fix 1: stale data must not produce a forecast
# ===========================================================================
def test_split_by_age_keeps_the_limit_inclusive():
    from src.deploy import run as RUNNER
    issued = pd.Timestamp("2026-10-18 12:00")
    limit = float(D.RUN["max_data_age_hours"])
    rows = pd.DataFrame({
        "station": ["fresh", "borderline", "just_over", "dead"],
        "time": [issued - pd.Timedelta(hours=1),
                 issued - pd.Timedelta(hours=limit),
                 issued - pd.Timedelta(hours=limit + 0.5),
                 issued - pd.Timedelta(days=40)],
        DO: [7.0, 7.0, 7.0, 7.0]})
    fresh, stale, ages = RUNNER.split_by_age(rows, issued)
    assert list(fresh["station"]) == ["fresh", "borderline"], \
        "a reading exactly at the limit is still usable"
    assert list(stale["station"]) == ["just_over", "dead"]
    assert ages[0] == pytest.approx(1.0)
    assert ages[-1] == pytest.approx(40 * 24)


def test_a_stale_station_gets_no_forecast_and_no_warning():
    """
    THE test for fix 1. The stale rows must come back with no predicted DO,
    no P(Low), no tau, `warning` False, and the outage in `status`.
    """
    from src.deploy import run as RUNNER
    issued = pd.Timestamp("2026-10-18 12:00")
    limit = float(D.RUN["max_data_age_hours"])
    stale_in = pd.DataFrame({
        "station": ["OLD", "DEAD"],
        "time": [issued - pd.Timedelta(hours=limit + 1),
                 issued - pd.Timedelta(days=45)],
        DO: [6.0, 4.0]})

    frame = RUNNER.stale_frame(stale_in, None, issued, "v-test")
    assert len(frame) == 2 * len(HORIZONS)
    assert set(frame["station"]) == {"OLD", "DEAD"}
    assert frame["pred_do_mgl"].isna().all(), \
        "a stale station must not be given a predicted DO"
    assert frame["p_low"].isna().all(), \
        "a stale station must not be given a probability of Low"
    assert frame["tau"].isna().all()
    assert not frame["warning"].astype(bool).any(), \
        "a stale station must never raise a warning"
    assert (frame["status"] == D.RUN["stale_status"]).all()
    assert (frame["alert_type"] == ALERT["stale"]).all()
    assert (frame["data_age_hours"] > limit).all()
    dead = frame[frame["station"] == "DEAD"]
    assert dead["data_age_hours"].iloc[0] == pytest.approx(45 * 24)
    # the age is reported even though nothing was predicted
    assert frame["data_age_hours"].notna().all()


def test_forecast_never_hands_a_stale_row_to_the_models(monkeypatch):
    """
    The guard runs BEFORE the sequence is built, so a dead sensor costs no
    GPU work. With every station stale, the models must not be touched at
    all - the stand-ins here raise if they are.
    """
    from src.deploy import run as RUNNER

    def explode(*a, **k):                       # pragma: no cover
        raise AssertionError("the models were called for a stale station")

    monkeypatch.setattr(RUNNER, "Forecaster", explode)
    monkeypatch.setattr(RUNNER.S, "sequences_for", explode)

    issued = pd.Timestamp("2026-10-18 12:00")
    rows = pd.DataFrame({
        "station": ["DEAD1", "DEAD2"],
        "time": [issued - pd.Timedelta(days=5),
                 issued - pd.Timedelta(days=9)],
        DO: [6.0, 6.0]})
    frame, info = RUNNER.forecast(rows, None, None, issued)

    assert len(frame) == 2 * len(HORIZONS)
    assert frame["pred_do_mgl"].isna().all()
    assert not frame["warning"].astype(bool).any()
    assert list(frame.columns)[:len(SCHEMA_12B)] == SCHEMA_12B
    reasons = {i["station"]: i["reason"] for i in info}
    assert set(reasons) == {"DEAD1", "DEAD2"}
    for why in reasons.values():
        assert "no forecast issued" in why
        assert "limit" in why


def test_todays_three_stale_stations_are_detected():
    """
    The stations the brief named, against the live window actually on disk:
    03353415 (~1069 h), 01484080 (~149 h) and 08048000 (~45 h) must all come
    out stale, and at least one station must still be fresh - a guard that
    rejects everything is not a guard.
    """
    from src.deploy import run as RUNNER
    import os
    path = D.dpath(D.DATA["features"], mode="live")
    if not os.path.exists(path):
        pytest.skip("no live window on disk")
    features = pd.read_parquet(path, columns=["station", "time", DO])
    rows = RUNNER.latest_rows(features)
    issued = pd.Timestamp(rows["time"].max())
    fresh, stale, _ = RUNNER.split_by_age(rows, issued)
    stale_set = set(stale["station"])
    for station in ("03353415", "01484080", "08048000"):
        if station in set(rows["station"]):
            assert station in stale_set, \
                (f"{station} should be stale: its latest reading is "
                 f"{(issued - pd.Timestamp(rows.loc[rows['station'] == station, 'time'].iloc[0])).total_seconds() / 3600:.0f} h "
                 f"old, over the {float(D.RUN['max_data_age_hours']):g} h limit")
    assert len(fresh), "every station came out stale - check the limit"


# ===========================================================================
# Fix 2: the run summary and the counts
# ===========================================================================
def test_the_run_summary_counts_every_alert_type(fixture_forecast):
    from src.deploy import run as RUNNER
    frame, _ = fixture_forecast
    counts = RUNNER.counts_by_alert(frame)
    assert set(counts) == set(ALERT.values())
    assert counts["early_warning"]["stations"] == 1
    assert counts["ongoing_low"]["stations"] == 1
    assert counts["stale"]["stations"] == 1
    assert counts["none"]["stations"] == 1
    assert sum(c["rows"] for c in counts.values()) == len(frame), \
        "every row must fall into exactly one alert type"


def test_the_summary_prints_the_counts_and_names_the_stale(fixture_forecast,
                                                           capsys):
    from src.deploy import run as RUNNER
    frame, path = fixture_forecast
    RUNNER.summarise(frame, [], path, path, len(frame))
    out = capsys.readouterr().out
    assert "BY ALERT TYPE" in out
    for name in ALERT.values():
        assert name in out, f"the summary does not count {name}"
    assert "EARLY WARNINGS" in out
    assert "SENSOR OFFLINE" in out
    assert STALE_STATION in out, "the summary does not name the stale station"


# ===========================================================================
# Fix 1 and 2 in the API
# ===========================================================================
def test_warnings_excludes_stale_stations(client):
    """
    THE test for the API half of fix 1. The stale station must not appear in
    /warnings under any query, and the response must say it was held back
    rather than leaving it silently missing.
    """
    c, _ = client
    body = c.get("/warnings").json()
    stations = {w["station"] for w in body["warnings"]}
    assert STALE_STATION not in stations, \
        "/warnings returned a station with no prediction"
    assert body["excluded_stale_stations"] == [STALE_STATION]
    for h in HORIZONS:
        one = c.get(f"/warnings?horizon={h}").json()
        assert STALE_STATION not in {w["station"] for w in one["warnings"]}
    assert all(w["alert_type"] != ALERT["stale"] for w in body["warnings"])


def test_warnings_can_be_narrowed_to_early_warnings(client):
    c, _ = client
    body = c.get("/warnings?alert_type=early_warning").json()
    assert {w["station"] for w in body["warnings"]} == {"02110701"}
    assert body["n_stations_early_warning"] == 1
    assert body["n_stations_ongoing_low"] == 0
    ongoing = c.get("/warnings?alert_type=ongoing_low").json()
    assert {w["station"] for w in ongoing["warnings"]} == {"02110550"}
    # stale is never a warning, so asking for it is a bad request
    bad = c.get("/warnings?alert_type=stale")
    assert bad.status_code == 400
    assert "stale" in bad.json()["detail"]


def test_forecast_for_a_stale_station_returns_the_status(client):
    c, _ = client
    body = c.get(f"/forecast/{STALE_STATION}").json()
    assert body["stale"] is True
    assert body["status"] == D.RUN["stale_status"]
    assert body["any_warning"] is False
    assert body["alert_types"] == [ALERT["stale"]]
    assert "no forecast was issued" in body["detail"]
    assert body["data_age_hours"] == 1069.0
    for h in body["horizons"]:
        assert h["pred_do_mgl"] is None or pd.isna(h["pred_do_mgl"])
        assert h["warning"] in (False, 0)


def test_health_names_the_stale_stations(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["stations_stale"] == [STALE_STATION]
    assert body["stations_forecast"] < body["stations"]
    assert body["stations_early_warning"] == 1
    assert body["stations_ongoing_low"] == 1


# ===========================================================================
# Fix 1 and 2 in the monitoring
# ===========================================================================
def _log_row(station, observed, h, **over):
    valid = observed + pd.Timedelta(hours=h)
    row = {"station": station, "issued_at": observed, "horizon": h,
           "valid_at": valid, "pred_do_mgl": 7.0, "p_low": 0.01,
           "tau": 0.24, "warning": False, "model_version": "v",
           "weather_source": "era5_archive", "data_age_hours": 1.0,
           "status": D.RUN["fresh_status"], "alert_type": ALERT["none"],
           "observed_at": observed, "do_now_mgl": 7.0, "class_now": 2.0}
    row.update(over)
    return row


def test_the_monitor_does_not_score_stale_rows(monkeypatch):
    """
    A stale row has no prediction. Left in, every one would count as a
    correct "no warning" and the false-alarm rate would be diluted by the
    outage itself.
    """
    from src.deploy import monitor as M
    t0 = pd.Timestamp("2026-10-01 00:00")
    rows, truth = [], []
    for i in range(60):
        observed = t0 + pd.Timedelta(hours=i)
        for h in HORIZONS:
            rows.append(_log_row("A", observed, h))
            rows.append(_log_row(
                "B", observed, h, pred_do_mgl=np.nan, p_low=np.nan,
                tau=np.nan, data_age_hours=900.0,
                status=D.RUN["stale_status"], alert_type=ALERT["stale"]))
            for s in ("A", "B"):
                truth.append({"station": s,
                              "valid_at": observed + pd.Timedelta(hours=h),
                              "y_true_reg": 7.2})
    log = pd.DataFrame(rows)
    monkeypatch.setattr(M, "log", lambda: log)
    monkeypatch.setattr(
        M, "observed",
        lambda: pd.DataFrame(truth).drop_duplicates(["station", "valid_at"]))

    d = M.joined(now=pd.Timestamp("2026-10-20"))
    assert set(d["station"]) == {"A"}, \
        "the stale station was scored although it has no prediction"
    assert d.attrs["n_stale_dropped"] == len(log) // 2
    perf = M.performance(d, now=pd.Timestamp("2026-10-20"))
    assert "B" not in set(perf.loc[perf["station"] != "ALL", "station"])


def test_the_monitor_scores_early_warnings_separately(monkeypatch):
    """
    A station that has been under 5 mg/L for weeks produces a long run of
    warnings that only repeat the sensor. The early-warning columns must
    count those as non-warnings, so what is left is the question the system
    is actually built for.
    """
    from src.deploy import monitor as M
    t0 = pd.Timestamp("2026-10-01 00:00")
    rows, truth = [], []
    for i in range(80):
        observed = t0 + pd.Timedelta(hours=i)
        for h in HORIZONS:
            rows.append(_log_row(
                "LOW", observed, h, pred_do_mgl=4.0, p_low=0.95,
                warning=True, alert_type=ALERT["ongoing_low"],
                do_now_mgl=4.2, class_now=0.0))
            truth.append({"station": "LOW",
                          "valid_at": observed + pd.Timedelta(hours=h),
                          "y_true_reg": 4.1})
    log = pd.DataFrame(rows)
    monkeypatch.setattr(M, "log", lambda: log)
    monkeypatch.setattr(
        M, "observed",
        lambda: pd.DataFrame(truth).drop_duplicates(["station", "valid_at"]))
    d = M.joined(now=pd.Timestamp("2026-10-20"))
    perf = M.performance(d, now=pd.Timestamp("2026-10-20"))
    overall = perf[(perf["station"] == "ALL")
                   & (perf["window"] != "all_time")]
    assert len(overall) == len(HORIZONS)
    for r in overall.itertuples():
        assert r.n_ongoing_low > 0, "the ongoing-low rows were not counted"
        assert r.n_early_warning == 0, \
            "an already-Low hour must not be counted as an early warning"
        # every logged hour was already Low, so there is no early-warning
        # population at all, and the columns say so rather than inventing one
        assert r.n_rows_early == 0
    for col in ("n_early_warning", "n_ongoing_low", "onset_recall_early",
                "false_alarm_rate_early", "onset_precision_early",
                "n_onsets_early", "n_rows_early"):
        assert col in perf.columns, f"monitor.csv lost {col}"


def test_a_real_early_warning_is_scored_as_one(monkeypatch):
    """
    The other side of the same partition: safe water, the rule fires, and
    the water really does go Low. That must register as an early warning
    with a real onset recall.
    """
    from src.deploy import monitor as M
    t0 = pd.Timestamp("2026-10-01 00:00")
    rows, truth = [], []

    # The truth has to be a function of `valid_at` alone, because that is
    # the key it is joined on: two different (issued_at, horizon) pairs can
    # land on the same hour, and they must not disagree about what the
    # oxygen did. Here the river goes Low every fourth hour of the day, and
    # the system warns exactly then.
    def goes_low(valid_at):
        return pd.Timestamp(valid_at).hour % 4 == 0

    for i in range(80):
        observed = t0 + pd.Timedelta(hours=i)
        for h in HORIZONS:
            valid = observed + pd.Timedelta(hours=h)
            onset = goes_low(valid)
            rows.append(_log_row(
                "RIVER", observed, h, pred_do_mgl=4.5 if onset else 7.0,
                p_low=0.9 if onset else 0.01, warning=onset,
                alert_type=ALERT["early_warning"] if onset else ALERT["none"],
                do_now_mgl=7.0, class_now=2.0))
            truth.append({"station": "RIVER", "valid_at": valid,
                          "y_true_reg": 4.1 if onset else 7.2})
    log = pd.DataFrame(rows)
    monkeypatch.setattr(M, "log", lambda: log)
    monkeypatch.setattr(
        M, "observed",
        lambda: pd.DataFrame(truth).drop_duplicates(["station", "valid_at"]))
    d = M.joined(now=pd.Timestamp("2026-10-20"))
    perf = M.performance(d, now=pd.Timestamp("2026-10-20"))
    overall = perf[(perf["station"] == "ALL")
                   & (perf["window"] != "all_time")]
    for r in overall.itertuples():
        assert r.n_early_warning > 0
        assert r.n_ongoing_low == 0
        assert r.n_onsets_early > 0
        assert r.onset_recall_early == pytest.approx(1.0), \
            "every onset was warned about, so recall should be 1"
        assert r.false_alarm_rate_early == pytest.approx(0.0)
        assert r.onset_precision_early == pytest.approx(1.0)
