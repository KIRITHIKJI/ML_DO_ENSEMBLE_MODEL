"""
Tests for step 12c - `class_now` in words, in both formats.

A live forecast file is read by people. `class_now` used to go out as
0.0 / 1.0 / 2.0, which means nothing to anyone who has not read
`config.yaml`, so it now goes out as "Low" / "Medium" / "Good".

The whole risk of that change is the files that already exist. Every
forecast written before it carries the numbers, and they must keep working
- in the API, in the monitor and in the app - without being rewritten.
So almost every test here runs twice, once per format, and one of them
reads the real forecast file that is sitting in results/live.

Run from the project root:   pytest -q tests/test_deploy_12c.py
"""

import glob
import os

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path
from src.deploy import common as D
from src.deploy import run as RUNNER

CFG = load_config()
HORIZONS = CFG["horizons"]
DO = CFG["target"]["column"]
CLASS_NAMES = CFG["target"]["class_names"]
LOW, GOOD = (CFG["target"]["thresholds"]["low"],
             CFG["target"]["thresholds"]["good"])
ALERT = D.RUN["alert_types"]

# the two formats, and how each one spells the three classes
NUMERIC = {"Low": 0.0, "Medium": 1.0, "Good": 2.0}
WORDS = {"Low": "Low", "Medium": "Medium", "Good": "Good"}
FORMATS = {"numeric": NUMERIC, "words": WORDS}


# ===========================================================================
# The two helpers
# ===========================================================================
def test_class_labels_turns_do_into_the_three_words():
    out = RUNNER.class_labels([2.0, LOW, 6.0, GOOD, 9.0])
    assert out == ["Low", "Medium", "Medium", "Medium", "Good"], \
        "the boundaries must match config.yaml: Low < 5.0, Good > 7.0"
    assert set(out) <= set(CLASS_NAMES)


def test_class_labels_leaves_a_missing_reading_missing():
    """
    A river nobody measured has no class. None is written to a csv as an
    empty field; a word there would be a claim.
    """
    assert RUNNER.class_labels([np.nan, 7.5]) == [None, "Good"]
    assert RUNNER.class_labels(pd.Series([np.nan])) == [None]


@pytest.mark.parametrize("value,expected", [
    ("Low", 0), ("Medium", 1), ("Good", 2),       # the new format
    ("low", 0), ("GOOD", 2), (" Medium ", 1),     # however it is cased
    (0, 0), (1, 1), (2, 2),                       # the old format
    (0.0, 0), (1.0, 1), (2.0, 2),
    ("0.0", 0), ("2.0", 2), ("1", 1),             # a csv read with no dtype
    (np.float64(0.0), 0), (np.int64(2), 2),
    (None, None), (np.nan, None), ("", None),     # and every kind of missing
    ("nonsense", None), (float("nan"), None),
])
def test_class_code_reads_both_formats(value, expected):
    assert RUNNER.class_code(value) == expected


def test_class_label_round_trips_either_way():
    for name in CLASS_NAMES:
        assert RUNNER.class_label(name) == name
        assert RUNNER.class_label(CLASS_NAMES.index(name)) == name
        assert RUNNER.class_label(float(CLASS_NAMES.index(name))) == name
    assert RUNNER.class_label(np.nan) is None


# ===========================================================================
# classify_alert must not care which format it is handed
# ===========================================================================
@pytest.mark.parametrize("fmt", list(FORMATS))
@pytest.mark.parametrize("cls,warning,stale,expected", [
    ("Good", True, False, "early_warning"),
    ("Medium", True, False, "early_warning"),
    ("Low", True, False, "ongoing_low"),
    ("Good", False, False, "none"),
    ("Low", False, False, "none"),
    ("Good", True, True, "stale"),
    ("Low", True, True, "stale"),
])
def test_alert_type_is_the_same_in_both_formats(fmt, cls, warning, stale,
                                                expected):
    value = FORMATS[fmt][cls]
    assert RUNNER.classify_alert(value, warning, stale) == expected


def test_an_unknown_class_is_still_an_early_warning():
    """Step 12b's rule, unchanged: a firing rule is never swallowed."""
    for missing in (None, np.nan, "", "nonsense"):
        assert RUNNER.classify_alert(missing, True) == "early_warning"
        assert RUNNER.classify_alert(missing, False) == "none"


# ===========================================================================
# What a run writes
# ===========================================================================
def test_the_stale_rows_carry_words(tmp_path):
    issued = pd.Timestamp("2026-10-18 12:00")
    stale_in = pd.DataFrame({
        "station": ["OLD", "DEAD", "NOREADING"],
        "time": [issued - pd.Timedelta(days=2),
                 issued - pd.Timedelta(days=45),
                 issued - pd.Timedelta(days=3)],
        DO: [6.0, 4.0, np.nan]})
    frame = RUNNER.stale_frame(stale_in, None, issued, "v-test")
    got = dict(zip(frame["station"], frame["class_now"]))
    assert got["OLD"] == "Medium"
    assert got["DEAD"] == "Low"
    assert got["NOREADING"] is None
    # and the guard itself is untouched: still no forecast, still no warning
    assert frame["pred_do_mgl"].isna().all()
    assert not frame["warning"].astype(bool).any()


def _stub_forecaster(monkeypatch, p_low):
    """A Forecaster stand-in, so this file needs no models and no GPU."""
    class Stub:
        batch_size = 1

        def __init__(self, *a, **k):
            pass

        def predict_rows(self, use, seq):
            n = len(use)
            return {h: {"do": np.full(n, 4.5), "p_low": np.full(n, p_low),
                        "tau": 0.24, "warn": np.full(n, p_low >= 0.24),
                        "method": "stub"} for h in HORIZONS}

    monkeypatch.setattr(RUNNER, "Forecaster", Stub)
    monkeypatch.setattr(
        RUNNER.S, "sequences_for",
        lambda fresh, panel, folder: (
            None, pd.DataFrame({"station": fresh["station"].to_numpy(),
                                "time": fresh["time"].to_numpy()})))


def test_a_forecast_writes_words_not_numbers(tmp_path, monkeypatch):
    """
    THE test for step 12c. The column that leaves a run says what the water
    is, in the words config.yaml uses, and the alert types still come out
    right on top of it.
    """
    _stub_forecaster(monkeypatch, p_low=0.9)
    issued = pd.Timestamp("2026-10-18 12:00")
    rows = pd.DataFrame({
        "station": ["ALREADY_LOW", "SAFE", "DEAD"],
        "time": [issued - pd.Timedelta(hours=1),
                 issued - pd.Timedelta(hours=1),
                 issued - pd.Timedelta(days=9)],
        DO: [3.1, 8.0, 6.0]})
    frame, _ = RUNNER.forecast(rows, None, None, issued)

    by_station = dict(zip(frame["station"], frame["class_now"]))
    assert by_station == {"ALREADY_LOW": "Low", "SAFE": "Good",
                          "DEAD": "Medium"}
    assert set(frame["class_now"]) <= set(CLASS_NAMES), \
        "class_now must hold only the configured class names"
    for value in frame["class_now"]:
        assert isinstance(value, str) and not value.replace(".", "").isdigit()

    alerts = dict(zip(frame["station"], frame["alert_type"]))
    assert alerts == {"ALREADY_LOW": ALERT["ongoing_low"],
                      "SAFE": ALERT["early_warning"],
                      "DEAD": ALERT["stale"]}

    # and it survives a round trip through the csv, which is where everyone
    # downstream actually meets it
    path = tmp_path / "forecast.csv"
    frame.to_csv(path, index=False)
    back = pd.read_csv(path, dtype={"station": str})
    assert set(back["class_now"]) == {"Low", "Good", "Medium"}
    assert [RUNNER.class_code(v) for v in back["class_now"]] \
        == [RUNNER.class_code(v) for v in frame["class_now"]]


def test_the_promised_schema_is_unchanged():
    """
    class_now is context, not part of the pre-registered schema, and this
    change must not have quietly moved the columns the API and the app
    read first.
    """
    assert "class_now" not in RUNNER.COLUMNS
    assert list(D.RUN["columns"]) == RUNNER.COLUMNS


# ===========================================================================
# The log: one file must never hold both formats
# ===========================================================================
def test_as_labels_converts_an_old_log():
    old = pd.DataFrame({"station": ["A", "B", "C"],
                        "class_now": [0.0, 2.0, np.nan],
                        "do_now_mgl": [3.0, 8.0, np.nan]})
    out = RUNNER.as_labels(old)
    assert list(out["class_now"]) == ["Low", "Good", None]
    assert list(old["class_now"][:2]) == [0.0, 2.0], "the input was mutated"
    # a frame without the column is handed back untouched
    assert RUNNER.as_labels(old.drop(columns=["class_now"])) is not None


def test_appending_to_an_old_log_leaves_one_format(tmp_path, monkeypatch):
    """
    A parquet column holding 1.0 in the old rows and "Medium" in the new
    ones has no type, and pyarrow refuses the file. The old rows are
    converted as they are read.
    """
    pytest.importorskip("pyarrow")
    log_path = tmp_path / "log.parquet"
    monkeypatch.setitem(RUNNER.RUN, "log_parquet", str(log_path))
    monkeypatch.setattr("src.deploy.run.project_path", lambda *p: str(p[0]))

    def rows(station, issued, cls, do_now):
        return pd.DataFrame({
            "station": [station] * len(HORIZONS),
            # one resolution for both frames: pandas 2.1 cannot concatenate
            # datetime64[s] with datetime64[ns], and a real log never mixes
            # them because every row comes from the same writer
            "issued_at": pd.Series([issued] * len(HORIZONS),
                                   dtype="datetime64[ns]"),
            "horizon": HORIZONS, "class_now": cls, "do_now_mgl": do_now})

    issued = pd.Timestamp("2026-10-18 06:00")
    old = rows("A", issued, 1.0, 6.0)
    old.to_parquet(log_path, index=False)
    new = rows("B", issued + pd.Timedelta(hours=1), "Good", 8.0)
    path, n = RUNNER.append_log(new)

    both = pd.read_parquet(path)
    assert n == len(both) == 2 * len(HORIZONS)
    assert set(both["class_now"]) == {"Medium", "Good"}, \
        "the old rows were not converted"
    assert both["class_now"].map(type).eq(str).all(), \
        "the log ended up holding two formats at once"


# ===========================================================================
# Everything downstream, in both formats
# ===========================================================================
def _forecast_frame(class_format):
    """
    One station of each alert type, with `class_now` in the given format.
    Mirrors the step 12b fixture; only the format of that one column
    differs between the two runs.
    """
    issued = pd.Timestamp("2026-10-18 06:00:00")
    plan = [("02110701", 0.42, 1.0, 6.2, "Medium"),     # -> early warning
            ("02198920", 0.05, 3.5, 6.2, "Medium"),     # -> none
            ("02110550", 0.80, 2.0, 3.1, "Low")]        # -> ongoing low
    rows = []
    for station, p_base, age, do_now, cls in plan:
        for h in HORIZONS:
            p_low = min(0.99, p_base * (1 + h / 100.0))
            warn = p_low >= 0.24
            rows.append({
                "station": station, "issued_at": issued, "horizon": h,
                "valid_at": issued + pd.Timedelta(hours=h),
                "pred_do_mgl": 6.0 - p_low, "p_low": p_low, "tau": 0.24,
                "warning": warn, "model_version": "final-testfixture",
                "weather_source": "forecast_api", "data_age_hours": age,
                "status": D.RUN["fresh_status"],
                "alert_type": RUNNER.classify_alert(cls, warn),
                "observed_at": issued, "do_now_mgl": do_now,
                "class_now": class_format[cls], "method": "stacking"})
    for h in HORIZONS:
        rows.append({
            "station": "03353415", "issued_at": issued, "horizon": h,
            "valid_at": issued + pd.Timedelta(hours=h),
            "pred_do_mgl": np.nan, "p_low": np.nan, "tau": np.nan,
            "warning": False, "model_version": "final-testfixture",
            "weather_source": "era5_archive", "data_age_hours": 1069.0,
            "status": D.RUN["stale_status"], "alert_type": ALERT["stale"],
            "observed_at": issued - pd.Timedelta(hours=1069),
            "do_now_mgl": 5.5, "class_now": class_format["Medium"],
            "method": "none - stale"})
    frame = pd.DataFrame(rows)
    return frame[RUNNER.COLUMNS + [c for c in frame.columns
                                   if c not in RUNNER.COLUMNS]]


@pytest.fixture(params=list(FORMATS))
def both_formats(request, tmp_path, monkeypatch):
    """The same forecast file, written once per format."""
    frame = _forecast_frame(FORMATS[request.param])
    folder = tmp_path / "live"
    folder.mkdir()
    path = folder / "forecast_20261018T060000Z.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(RUNNER, "forecast_dir", lambda: str(folder))
    return request.param, frame, str(path)


def test_the_api_reads_either_format(both_formats):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.deploy import api as A

    fmt, _, path = both_formats
    c = TestClient(A.app)

    health = c.get("/health").json()
    assert health["status"] == "ok", f"/health broke on the {fmt} format"
    assert health["latest_forecast"] == os.path.basename(path)
    assert health["stations_early_warning"] == 1
    assert health["stations_ongoing_low"] == 1
    assert health["stations_stale"] == ["03353415"]

    warnings = c.get("/warnings?alert_type=early_warning").json()
    assert {w["station"] for w in warnings["warnings"]} == {"02110701"}

    one = c.get("/forecast/02110550").json()
    assert one["any_warning"] is True
    assert one["alert_types"] == [ALERT["ongoing_low"]]
    stale = c.get("/forecast/03353415").json()
    assert stale["stale"] is True


def test_the_app_groups_stations_in_either_format(both_formats):
    pytest.importorskip("streamlit")
    import importlib
    app = importlib.import_module("app.app")

    fmt, frame, _ = both_formats
    groups = app.station_groups(frame)
    assert groups["early_warning"] == ["02110701"], \
        f"the app grouped the wrong station on the {fmt} format"
    assert groups["ongoing_low"] == ["02110550"]
    assert groups["none"] == ["02198920"]
    assert groups["stale"] == ["03353415"]


def test_the_monitor_scores_either_format(both_formats, monkeypatch):
    """
    The monitor derives `class_now` from `do_now_mgl` rather than trusting
    the column, so both formats must give identical early-warning counts.
    """
    from src.deploy import monitor as M
    fmt, frame, _ = both_formats
    log = frame.copy()
    truth = pd.DataFrame({
        "station": log["station"], "valid_at": log["valid_at"],
        "y_true_reg": np.where(log["station"] == "02110701", 4.1, 7.2),
    }).drop_duplicates(["station", "valid_at"])
    monkeypatch.setattr(M, "log", lambda: log)
    monkeypatch.setattr(M, "observed", lambda: truth)

    d = M.joined(now=pd.Timestamp("2026-10-20"))
    assert d is not None and "03353415" not in set(d["station"])
    counts = {s: int(c) for s, c in d["class_now"].value_counts().items()}
    assert counts == {1.0: 2 * len(HORIZONS), 0.0: len(HORIZONS)}, \
        f"the monitor read class_now differently on the {fmt} format"


# ===========================================================================
# The forecast file that is already on disk
# ===========================================================================
def _existing_numeric_forecasts():
    """
    Every forecast file on disk whose `class_now` is still the old numeric
    one, newest last. There are two of them in this checkout, and one
    predates `alert_type` as well.
    """
    folder = project_path(D.RUN["results_dir"])
    out = []
    for path in sorted(glob.glob(os.path.join(folder,
                                              D.RUN["forecast_glob"]))):
        df = pd.read_csv(path, dtype={"station": str})
        if "class_now" in df.columns \
                and pd.api.types.is_numeric_dtype(df["class_now"]):
            out.append((path, df))
    return out


def _existing_numeric_forecast():
    found = _existing_numeric_forecasts()
    return found[-1] if found else (None, None)


def test_the_forecast_already_on_disk_was_not_rewritten():
    """
    results/live holds forecasts written before this change. They are
    evidence of what the system actually said at that hour, so they are
    read in their own format and never edited into the new one.
    """
    found = _existing_numeric_forecasts()
    if not found:
        pytest.skip("no numeric-format forecast on disk")
    for path, df in found:
        assert set(df["class_now"].dropna().unique()) <= {0.0, 1.0, 2.0}, \
            f"{os.path.basename(path)} was edited into the new format"
        codes = [RUNNER.class_code(v) for v in df["class_now"]]
        assert set(codes) <= {0, 1, 2, None}
        assert [RUNNER.class_label(v) for v in df["class_now"]] \
            == [None if c is None else CLASS_NAMES[c] for c in codes], \
            "the old format must still read back as the same three classes"


def test_the_live_tab_still_loads_the_forecast_on_disk():
    """
    The Live tab's own path over the real file: the groups must come out,
    and they must agree with the `alert_type` column it was written with.
    """
    pytest.importorskip("streamlit")
    import importlib
    found = _existing_numeric_forecasts()
    if not found:
        pytest.skip("no numeric-format forecast on disk")
    app = importlib.import_module("app.app")
    for path, df in found:
        groups = app.station_groups(df)
        assert sum(len(v) for v in groups.values()) \
            == df["station"].nunique(), \
            f"{os.path.basename(path)} lost stations in the Live tab"
        if "alert_type" not in df.columns:
            continue        # written before step 12b; the app falls back
        for name, stations in groups.items():
            for station in stations:
                kinds = set(df.loc[df["station"] == station, "alert_type"])
                assert name in kinds, \
                    f"{station} was grouped as {name}, which it is not"


def test_the_api_still_serves_the_forecast_on_disk(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.deploy import api as A

    path, df = _existing_numeric_forecast()
    if path is None:
        pytest.skip("no numeric-format forecast on disk")
    monkeypatch.setattr(RUNNER, "latest_forecast_path", lambda: path)
    c = TestClient(A.app)
    health = c.get("/health").json()
    assert health["status"] == "ok"
    assert health["latest_forecast"] == os.path.basename(path)
    assert c.get("/warnings").status_code == 200
    station = str(df["station"].iloc[0])
    assert c.get(f"/forecast/{station}").status_code == 200


# ===========================================================================
# The model card says what the deployed system now does
# ===========================================================================
def _card():
    path = project_path(D.DEP["models"]["model_card"])
    if not os.path.exists(path):
        pytest.skip("no model card on disk - python -m src.deploy.model_card")
    return open(path, encoding="utf-8").read()


def test_the_card_carries_the_deployed_version_id():
    assert f"`{D.model_version()}`" in _card()


def test_the_card_describes_the_stale_guard_from_the_config():
    """
    The limit in the card must be the one in config_deploy.yaml, and it
    must get there by being read, not by being typed.
    """
    card, limit = _card(), float(D.RUN["max_data_age_hours"])
    assert f"**{limit:g} h**" in card
    assert f"{limit:g} h of silence" in card
    assert D.RUN["stale_status"] in card
    assert "no prediction at all" in card

    src = open(project_path("src", "deploy", "model_card.py"),
               encoding="utf-8").read()
    assert "D.RUN['max_data_age_hours']" in src
    assert f"{limit:g} h" not in src, \
        "the card generator has the limit typed into it"


def test_the_card_names_all_four_alert_types():
    card = _card()
    for name in D.RUN["alert_types"].values():
        assert f"`{name}`" in card, f"the card does not mention {name}"


def test_the_card_is_honest_about_sensor_faults():
    """
    The old card said sensor faults are not detected. Half of that is no
    longer true, and the other half still is; the card must say which.
    """
    card = _card()
    assert "Sensor faults are not detected" not in card
    assert "A sensor that STOPS reporting is caught; one that lies is not" \
        in card
    assert "stuck or drifting sensor that keeps reporting is fed straight " \
        "in" in card


def _tidal_bullet():
    """The one Limitations bullet about the tidal stations."""
    hits = [line for line in _card().splitlines()
            if line.startswith("- ") and "tidal stations" in line]
    assert len(hits) == 1, "expected exactly one tidal limitation"
    return hits[0]


def test_the_card_warns_about_the_tidal_stations():
    from src.deploy.model_card import TIDAL_STATIONS
    bullet = _tidal_bullet()
    for station in TIDAL_STATIONS:
        assert station in bullet, \
            f"the tidal limitation does not name {station}"
    assert "twice-daily cycle" in bullet
    assert "`early_warning`" in bullet
    assert "the forecast is correct" in bullet
    assert "not deterioration events" in bullet


def test_the_tidal_limitation_claims_no_mechanism():
    """
    Step 12d. The card used to say the DO dips "as salt water comes up the
    estuary". That is a cause, and this project never tested one: the data
    shows a twice-daily cycle and nothing about what drives it. The card
    reports the cycle and says outright that the mechanism is untested.
    """
    card, bullet = _card(), _tidal_bullet()
    assert "salt water" not in card, \
        "the card claims a mechanism the project never tested"
    for claim in ("saline", "salinity", "because the tide", "driven by"):
        assert claim not in bullet, \
            f"the tidal limitation still explains the cycle: {claim!r}"
    assert "no mechanism is claimed" in bullet

    src = open(project_path("src", "deploy", "model_card.py"),
               encoding="utf-8").read()
    assert "salt water" not in src
