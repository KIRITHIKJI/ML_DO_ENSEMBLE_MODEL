"""
Tests for step 12 - the deployment layer.

What these tests are for, in order of how much they matter:

  * the 2026 test was pre-registered. `git log` must show
    reports/plan_2026.md committed before results/live_2026/ existed, and
    the scorer must refuse a window past the date the plan fixed. Without
    that, "we scored the deployed models on an unseen year" is a claim, not
    a fact.
  * the paper's evidence is untouched. config.yaml is byte-identical to the
    freeze commit, and nothing under models/stage2 or results/test is
    written by any deployment module.
  * a production run survives a station failing. One offline sensor, one
    API refusal or one missing column must cost that station and nothing
    else.
  * the forecast csv has the schema config_deploy.yaml promises, because
    the API and the app both read it.
  * the promotion gate refuses a worse model. That gate is the only thing
    standing between an automated retrain and a silently degraded warning
    system.
  * the parallel bootstrap gives the same intervals as the serial one.

Run from the project root:   pytest -q tests/test_deploy.py
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config, project_path
from src.deploy import boot as B
from src.deploy import common as D

CFG = load_config()
HORIZONS = CFG["horizons"]
ROOT = project_path(".")


def git(*args):
    """git, from the project root, as text. Skips the test if git is not here."""
    try:
        out = subprocess.run(("git",) + args, cwd=ROOT, text=True,
                             capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover
        pytest.skip(f"git is not usable here: {e}")
    if out.returncode != 0:
        pytest.skip(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


# ===========================================================================
# 1. The configs
# ===========================================================================
def test_config_deploy_exists_and_is_separate_from_config():
    """
    Every deployment setting lives in config_deploy.yaml. The test is that
    the file is there, that it has the blocks the modules read, and that it
    does not quietly redefine the science - horizons, thresholds, the seed
    and the split belong to config.yaml and must not appear here.
    """
    path = project_path("config_deploy.yaml")
    assert os.path.exists(path), "config_deploy.yaml is missing"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert set(raw) == {"deploy"}, \
        "config_deploy.yaml must have exactly one top-level key, `deploy`"
    dep = raw["deploy"]
    for block in ("models", "test_2026", "data", "fetch", "run", "monitor",
                  "retrain", "api", "bootstrap", "compute", "walk_forward"):
        assert block in dep, f"config_deploy.yaml is missing `{block}`"
    forbidden = {"horizons", "split", "seed", "target", "period",
                 "pass_criteria", "final_system", "warning_rule"}
    assert not (forbidden & set(dep)), \
        (f"these belong to config.yaml and must not be redefined in "
         f"config_deploy.yaml: {sorted(forbidden & set(dep))}")


def test_config_yaml_is_byte_identical_to_the_freeze_commit():
    """
    config.yaml is the paper's evidence. Step 12 adds a file beside it and
    changes nothing in it. Compared against the commit that introduced
    FROZEN.md, so this keeps holding however many steps come later.
    """
    freeze = git("log", "--diff-filter=A", "--format=%H", "--",
                 "FROZEN.md").strip().splitlines()
    assert freeze, "no commit introduces FROZEN.md"
    frozen_text = git("show", f"{freeze[0]}:config.yaml")
    current = open(project_path("config.yaml"), encoding="utf-8").read()
    assert current == frozen_text, \
        "config.yaml differs from the version committed at the freeze"


def test_the_deployment_never_writes_into_the_paper_s_folders():
    """
    Read the deployment modules and check that every output path they use
    is under data/live, results/live, results/live_2026, models/candidate or
    models/archive. A path into data/raw, data/processed, models/stage2 or
    results/test would mean the deployment can overwrite the evidence.
    """
    protected = ("data/raw", "data/processed", "models/stage2",
                 "models/ml", "models/dl", "models/ensemble",
                 CFG["final_test"]["results_dir"].replace("\\", "/"))
    allowed_out = ("data/live", "results/live", "results/live_2026",
                   "models/candidate", "models/archive", "models/final")
    dep = D.DEP
    paths = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and ("/" in node) and not node.startswith(
                ("http://", "https://")):
            paths.append(node.replace("\\", "/"))

    walk(dep)
    for p in paths:
        if p.startswith(allowed_out) or p.startswith("reports/"):
            continue
        assert not p.startswith(protected), \
            f"config_deploy.yaml points an output at the paper's data: {p}"


# ===========================================================================
# 2. Pre-registration
# ===========================================================================
def test_plan_2026_was_committed_before_any_2026_result_existed():
    """
    THE test for the 2026 claim. The commit that adds reports/plan_2026.md
    must come strictly before the first commit that touches
    results/live_2026/ - and, if the results have not been committed yet,
    the plan must still be committed. Order is taken from the commit list
    itself, not from timestamps, which can be rewritten.
    """
    order = git("log", "--format=%H", "--reverse").strip().splitlines()
    position = {h: i for i, h in enumerate(order)}

    plan_commits = git("log", "--diff-filter=A", "--format=%H", "--",
                       "reports/plan_2026.md").strip().splitlines()
    assert plan_commits, ("reports/plan_2026.md has never been committed; "
                          "the 2026 test is not pre-registered")
    plan_at = min(position[h] for h in plan_commits if h in position)

    results_commits = git("log", "--format=%H", "--",
                          D.TEST26["results_dir"]).strip().splitlines()
    results_commits = [h for h in results_commits if h in position]
    if not results_commits:
        return          # the plan is committed and the results are not yet
    first_result = min(position[h] for h in results_commits)
    assert plan_at < first_result, (
        "results/live_2026/ was committed before (or in the same commit as) "
        "reports/plan_2026.md - the plan must come first and alone")


def test_the_plan_is_committed_on_its_own():
    """
    The pre-registration commit touches the plan and nothing else. A commit
    that also carried results would leave it unclear which came first.
    """
    plan_commits = git("log", "--diff-filter=A", "--format=%H", "--",
                       "reports/plan_2026.md").strip().splitlines()
    assert plan_commits
    files = git("show", "--name-only", "--format=", plan_commits[0]
                ).strip().splitlines()
    files = [f.strip() for f in files if f.strip()]
    assert files == ["reports/plan_2026.md"], \
        f"the pre-registration commit also touched: {files}"


def test_the_plan_records_the_criteria_it_is_judged_by():
    text = open(project_path(D.TEST26["plan_file"]), encoding="utf-8").read()
    budget = CFG["final_test"]["target_false_alarm_rate"]
    for needed in ["2026-01-01", str(D.TEST26["end_frozen"]),
                   f"{budget:.0%}", "onset recall", "RMSE", "PASS",
                   "PARTIAL", "FAIL", "degraded", "holds"]:
        assert needed in text, f"plan_2026.md does not mention {needed!r}"


def test_the_scorer_refuses_a_window_past_the_pre_registered_end():
    """
    The plan fixed the end of the test period. Handing the scorer a later
    window must stop the run, not quietly extend the test to a month that
    happens to look better.
    """
    from src.deploy import score_2026 as S
    frozen = pd.Timestamp(D.TEST26["end_frozen"])
    S.check_plan(frozen + pd.Timedelta("23h"))            # allowed
    with pytest.raises(SystemExit):
        S.check_plan(frozen + pd.Timedelta(days=7))


def test_the_era5_cutoff_keeps_the_forecast_model_out_of_the_test():
    """
    Open-Meteo's archive endpoint serves the last few days from a forecast
    model. The history window must stop `era5_lag_days` behind today, or
    the 2026 test would quietly contain forecast weather.
    """
    from src.deploy import fetch_live as F
    now = pd.Timestamp("2026-06-15 09:00")
    assert F.era5_cutoff(now) == pd.Timestamp("2026-06-10")
    start, end, fetch_start, water_end = F.window("history", now)
    assert start == pd.Timestamp(D.TEST26["start"])
    assert end <= pd.Timestamp("2026-06-10 23:00")
    assert fetch_start < start, "history needs a pre-roll for the long windows"
    assert water_end > end, "water must reach past the window for the targets"


# ===========================================================================
# 3. A station failing does not stop a run
# ===========================================================================
def test_one_station_failing_does_not_stop_the_fetch(monkeypatch, tmp_path):
    """
    Three stations, the middle one's sensor offline. The run must finish,
    keep the other two, and log the failure with a reason.
    """
    from src.deploy import fetch_live as F

    sites = ["AAA", "BBB", "CCC"]
    table = pd.DataFrame({"site_no": sites, "state": ["FL"] * 3,
                          "name": sites, "lat": [29.0] * 3,
                          "lon": [-82.0] * 3})
    monkeypatch.setattr(D, "stations_table", lambda: table)
    monkeypatch.setattr(F.D, "stations_table", lambda: table)

    calls = []

    def fake_fetch(site, lat, lon, mode, win):
        calls.append(site)
        if site == "BBB":
            raise F.Skip("the DO sensor reported nothing in the window")
        if site == "CCC":
            raise RuntimeError("Open-Meteo returned HTTP 503")
        return {"station": site, "seconds": 0.1, "note": "ok"}

    monkeypatch.setattr(F, "fetch_station", fake_fetch)
    monkeypatch.setattr(
        F, "build_panel",
        lambda mode, stations, win, workers=None, prefer="era5": (
            pd.DataFrame({"station": stations, "time": [pd.Timestamp("2026-01-01")] * len(stations)}),
            pd.DataFrame({"station": stations, "time": [pd.Timestamp("2026-01-01")] * len(stations)}),
            pd.DataFrame({"station": stations, "time": [pd.Timestamp("2026-01-01")] * len(stations),
                          "weather_source": ["era5_archive"] * len(stations),
                          "air_source": ["era5_archive"] * len(stations)}),
            pd.DataFrame(columns=["station", "stage", "reason"])))

    written = {}
    monkeypatch.setattr(F.D, "dpath",
                        lambda *parts, **fmt: str(tmp_path / "_".join(
                            [p.format(**fmt) if fmt else p
                             for p in parts]).replace("/", "_")))

    out = F.run("live", now=pd.Timestamp("2026-06-15 09:00"))

    assert sorted(calls) == sites, "not every station was attempted"
    assert out["stations"] == ["AAA"], \
        f"the healthy station should be the only one kept, got {out['stations']}"
    reasons = {s["station"]: s["reason"] for s in out["skipped"]}
    assert set(reasons) == {"BBB", "CCC"}
    assert "DO sensor" in reasons["BBB"]
    assert "503" in reasons["CCC"]


def test_a_skip_is_a_reason_not_a_crash():
    """`Skip` carries the reason, so the caller can log it."""
    from src.deploy import fetch_live as F
    e = F.Skip("required column do_mgl missing from the USGS reply")
    assert isinstance(e, Exception)
    assert "do_mgl" in str(e)


def test_missing_ranges_only_asks_for_what_the_cache_lacks():
    """
    A re-run must not re-download a window it already has. An interior hole
    is a real gap in the source and is deliberately not re-requested every
    hour.
    """
    from src.deploy import fetch_live as F
    idx = pd.date_range("2026-01-01", "2026-03-01", freq="h")
    have = pd.DataFrame({"x": 1.0}, index=idx)
    assert F.missing_ranges(have, pd.Timestamp("2026-01-01"),
                            pd.Timestamp("2026-03-01")) == []
    assert F.missing_ranges(None, pd.Timestamp("2026-01-01"),
                            pd.Timestamp("2026-03-01")) == [
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-01"))]
    gaps = F.missing_ranges(have, pd.Timestamp("2026-01-01"),
                            pd.Timestamp("2026-04-01"))
    assert len(gaps) == 1 and gaps[0][1] == pd.Timestamp("2026-04-01")


# ===========================================================================
# 4. The forecast csv schema
# ===========================================================================
REQUIRED_COLUMNS = ["station", "issued_at", "horizon", "valid_at",
                    "pred_do_mgl", "p_low", "tau", "warning",
                    "model_version", "weather_source", "data_age_hours"]
# step 12b added two columns to the end of the same schema: whether a
# prediction was made at all, and what kind of alert the row is.
SCHEMA_12B = REQUIRED_COLUMNS + ["status", "alert_type"]
STALE_STATION = "03353415"


def test_the_configured_schema_is_the_one_the_brief_asked_for():
    assert list(D.RUN["columns"]) == SCHEMA_12B, \
        "run.columns in config_deploy.yaml is not the agreed forecast schema"
    # the original eleven are unchanged and still come first, so anything
    # reading the step-12 schema by position still works
    assert list(D.RUN["columns"])[:len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS


@pytest.fixture
def fixture_forecast(tmp_path, monkeypatch):
    """
    A small forecast file in the real schema, in a temporary results folder.
    Both the csv-schema test and the API tests use it, so neither needs a
    GPU or a live network.
    """
    from src.deploy import run as RUNNER
    issued = pd.Timestamp("2026-10-18 06:00:00")
    rows = []
    # one station of each kind, so every test that reads this fixture sees
    # all four alert types: an early warning, an ongoing low, a quiet
    # station and a dead sensor.
    plan = [("02110701", 0.42, 1.0, 6.2, 1.0),    # Medium now -> early
            ("02198920", 0.05, 3.5, 6.2, 1.0),    # Medium now -> none
            ("02110550", 0.80, 2.0, 3.1, 0.0)]    # already Low -> ongoing
    for station, p_base, age, do_now, cls_now in plan:
        for h in HORIZONS:
            tau = 0.24
            # p_base scales with the horizon rather than having a constant
            # added, so the quiet station stays quiet at 48 h too
            p_low = min(0.99, p_base * (1 + h / 100.0))
            warn = p_low >= tau
            rows.append({
                "station": station, "issued_at": issued, "horizon": h,
                "valid_at": issued.floor("h") + pd.Timedelta(hours=h),
                "pred_do_mgl": 6.0 - p_low, "p_low": p_low, "tau": tau,
                "warning": warn, "model_version": "final-testfixture",
                "weather_source": "forecast_api", "data_age_hours": age,
                "status": D.RUN["fresh_status"],
                "alert_type": RUNNER.classify_alert(cls_now, warn),
                "observed_at": issued.floor("h"), "do_now_mgl": do_now,
                "class_now": cls_now, "method": "stacking"})
    # and a station whose sensor stopped reporting: no prediction at all
    for h in HORIZONS:
        rows.append({
            "station": STALE_STATION, "issued_at": issued, "horizon": h,
            "valid_at": issued.floor("h") + pd.Timedelta(hours=h),
            "pred_do_mgl": np.nan, "p_low": np.nan, "tau": np.nan,
            "warning": False, "model_version": "final-testfixture",
            "weather_source": "era5_archive", "data_age_hours": 1069.0,
            "status": D.RUN["stale_status"],
            "alert_type": D.RUN["alert_types"]["stale"],
            "observed_at": issued.floor("h") - pd.Timedelta(hours=1069),
            "do_now_mgl": 5.5, "class_now": 1.0, "method": "none - stale"})
    frame = pd.DataFrame(rows)
    folder = tmp_path / "live"
    folder.mkdir()
    path = folder / "forecast_20261018T060000Z.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(RUNNER, "forecast_dir", lambda: str(folder))
    return frame, str(path)


def test_the_forecast_csv_has_the_promised_schema(fixture_forecast):
    frame, path = fixture_forecast
    on_disk = pd.read_csv(path, dtype={"station": str})
    for col in REQUIRED_COLUMNS:
        assert col in on_disk.columns, f"the forecast csv has no {col!r}"
    assert list(on_disk.columns)[:len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS, \
        "the promised columns must come first, in order"
    assert set(on_disk["horizon"]) == set(HORIZONS)
    assert on_disk["model_version"].notna().all(), \
        "every row must carry the model version id"
    assert on_disk["weather_source"].notna().all(), \
        "every row must say where its weather came from"
    assert (on_disk["data_age_hours"] >= 0).all()
    for col in ("status", "alert_type"):
        assert on_disk[col].notna().all(), f"every row needs a {col}"
    assert set(on_disk["alert_type"]) <= set(D.RUN["alert_types"].values())

    # the stale rows carry the outage instead of numbers
    stale = on_disk[on_disk["alert_type"] == D.RUN["alert_types"]["stale"]]
    assert len(stale), "the fixture has no stale station"
    assert stale["pred_do_mgl"].isna().all()
    assert stale["p_low"].isna().all()
    assert not stale["warning"].astype(bool).any()
    assert (stale["status"] == D.RUN["stale_status"]).all()

    # and where a prediction WAS made, the warning flag agrees with the rule
    live = on_disk.drop(stale.index)
    assert live["p_low"].between(0, 1).all()
    rule = live["p_low"] >= live["tau"]
    assert (live["warning"].astype(bool) == rule).all(), \
        "the warning column does not match P(Low) >= tau"


def test_the_model_version_is_a_hash_of_the_model_files(tmp_path):
    """
    The version id in every forecast row must change when a model file
    changes and not otherwise - that is what makes a forecast traceable.
    """
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "a.bin").write_bytes(b"one")
    v1 = D.model_version(str(folder))
    assert v1.startswith(D.DEP["models"]["version_prefix"] + "-")
    assert len(v1.split("-")[1]) == int(D.DEP["models"]["version_digest_chars"])
    assert D.model_version(str(folder)) == v1, "the id is not stable"
    (folder / "MODEL_CARD.md").write_text("documenting it changes nothing")
    assert D.model_version(str(folder)) == v1, \
        "the model card must not change the version id"
    (folder / "a.bin").write_bytes(b"two")
    assert D.model_version(str(folder)) != v1, \
        "a changed model file must change the version id"


def test_the_log_does_not_double_count_a_repeated_run(tmp_path, monkeypatch):
    """
    An hourly schedule that fires twice for the same hour must replace that
    hour's rows, not append them again - the monitoring reads this file.
    """
    from src.deploy import run as RUNNER
    path = tmp_path / "log.parquet"
    monkeypatch.setitem(RUNNER.RUN, "log_parquet", str(path))
    monkeypatch.setattr(RUNNER, "project_path", lambda *p: str(path))
    frame = pd.DataFrame({
        "station": ["A"] * len(HORIZONS), "issued_at": pd.Timestamp("2026-10-18"),
        "horizon": HORIZONS, "valid_at": pd.Timestamp("2026-10-18"),
        "pred_do_mgl": 6.0, "p_low": 0.1, "tau": 0.24, "warning": False,
        "model_version": "v", "weather_source": "era5_archive",
        "data_age_hours": 1.0})
    _, n1 = RUNNER.append_log(frame)
    _, n2 = RUNNER.append_log(frame)
    assert n1 == n2 == len(HORIZONS), \
        f"re-running the same hour doubled the log: {n1} then {n2}"


# ===========================================================================
# 5. The promotion gate
# ===========================================================================
def _metrics(rmse, far):
    return {h: {"rmse": rmse, "false_alarm_rate": far} for h in HORIZONS}


def test_the_gate_refuses_a_worse_model():
    """THE test for automated retraining. A worse candidate is refused."""
    from src.deploy import retrain as R
    old = _metrics(0.67, 0.015)
    worse_rmse = _metrics(0.70, 0.015)
    worse_far = _metrics(0.67, 0.019)
    for cand, what in ((worse_rmse, "RMSE"), (worse_far, "false alarms")):
        promote, rows = R.gate(old, cand)
        assert not promote, f"the gate promoted a model with worse {what}"
        assert any(not r["passes"] for r in rows)


def test_the_gate_accepts_a_better_model():
    from src.deploy import retrain as R
    promote, rows = R.gate(_metrics(0.67, 0.015), _metrics(0.64, 0.012))
    assert promote
    assert all(r["passes"] for r in rows)


def test_the_gate_needs_every_horizon_to_pass():
    """
    Better on average is not good enough: a system that got worse at 48 h
    is worse at 48 h, and the operator who reads the 48 h card would be
    the one paying for it.
    """
    from src.deploy import retrain as R
    old = _metrics(0.67, 0.015)
    cand = _metrics(0.64, 0.012)
    cand[HORIZONS[-1]] = {"rmse": 0.64, "false_alarm_rate": 0.021}
    promote, rows = R.gate(old, cand)
    assert not promote
    failed = [r for r in rows if not r["passes"]]
    assert {r["horizon"] for r in failed} == {HORIZONS[-1]}


def test_the_gate_refuses_what_it_could_not_measure():
    """A candidate with no held-out measurement is refused, not promoted."""
    from src.deploy import retrain as R
    old = _metrics(0.67, 0.015)
    cand = {h: {"rmse": np.nan, "false_alarm_rate": np.nan} for h in HORIZONS}
    promote, _ = R.gate(old, cand)
    assert not promote


def test_the_gate_is_exactly_as_strict_as_the_configured_tolerance():
    from src.deploy import retrain as R
    tol = dict(D.RETRAIN["gate_tolerance"])
    assert tol["rmse"] == 0.0 and tol["false_alarm_rate"] == 0.0, \
        "the shipped gate must have no slack"
    old = _metrics(0.67, 0.015)
    assert R.gate(old, _metrics(0.67, 0.015))[0], "equal must pass"
    assert not R.gate(old, _metrics(0.67 + 1e-6, 0.015))[0], \
        "worse by anything must fail with a zero tolerance"


def test_the_holdout_is_months_the_candidate_did_not_train_on():
    from src.deploy import retrain as R
    cutoff, holdout_start = R.cutoff_and_holdout("2026-10-13 23:00")
    assert holdout_start < cutoff
    months = int(D.RETRAIN["holdout_months"])
    assert (cutoff - holdout_start).days >= 28 * months
    train = R.training_table(holdout_start)
    assert (pd.to_datetime(train["time"]) < holdout_start).all(), \
        "the candidate's training rows reach into the held-out months"


def test_promotion_archives_the_old_version_first(tmp_path):
    """
    The models that produced every past forecast must stay recoverable; the
    version id in a forecast row is the key to the archive folder.
    """
    from src.deploy import retrain as R
    deployed = tmp_path / "final"
    candidate = tmp_path / "candidate"
    archive = tmp_path / "archive"
    deployed.mkdir()
    candidate.mkdir()
    (deployed / "m.bin").write_bytes(b"old")
    (candidate / "m.bin").write_bytes(b"new")
    old_version = D.model_version(str(deployed))
    info = R.promote(str(candidate), str(deployed), str(archive))
    assert info["archived_version"] == old_version
    assert (archive / old_version / "m.bin").read_bytes() == b"old"
    assert (deployed / "m.bin").read_bytes() == b"new"
    assert info["new_version"] != old_version


# ===========================================================================
# 6. The API
# ===========================================================================
@pytest.fixture
def client(fixture_forecast):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.deploy import api as A
    return TestClient(A.app), fixture_forecast


def test_health_reports_the_latest_run(client):
    c, (frame, path) = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["latest_forecast"] == os.path.basename(path)
    assert body["stations"] == frame["station"].nunique()
    assert sorted(body["horizons"]) == sorted(HORIZONS)
    assert body["model_version"] == ["final-testfixture"]
    assert isinstance(body["age_minutes"], float)
    # the early warning and the ongoing low are warned; the quiet station
    # and the dead sensor are not
    assert body["stations_warned"] == 2
    assert body["stations_early_warning"] == 1
    assert body["stations_ongoing_low"] == 1
    # the stale station is named, not merely absent
    assert body["stations_stale"] == [STALE_STATION]
    assert body["stations_forecast"] == 3
    assert body["max_data_age_hours"] == 1069.0
    assert body["max_data_age_hours_forecast"] == 3.5,         "the stale station's age must not be reported as a forecast age"
    assert body["max_data_age_hours_allowed"] == float(
        D.RUN["max_data_age_hours"])


def test_warnings_returns_only_the_stations_under_warning(client):
    c, (frame, _) = client
    r = c.get("/warnings")
    assert r.status_code == 200
    body = r.json()
    assert body["n_stations"] == 2
    assert {w["station"] for w in body["warnings"]} == {"02110701",
                                                        "02110550"}
    for w in body["warnings"]:
        assert w["warning"] is True
        for col in SCHEMA_12B:
            assert col in w, f"/warnings dropped {col}"
    # and it can be narrowed to one horizon
    one = c.get(f"/warnings?horizon={HORIZONS[0]}").json()
    assert one["n_rows"] == 2
    assert c.get("/warnings?horizon=7").status_code == 400


def test_forecast_for_one_station_has_every_horizon(client):
    c, (frame, _) = client
    r = c.get("/forecast/02198920")
    assert r.status_code == 200
    body = r.json()
    assert body["station"] == "02198920"
    assert [h["horizon"] for h in body["horizons"]] == sorted(HORIZONS)
    assert body["any_warning"] is False
    assert body["model_version"] == "final-testfixture"
    assert body["data_age_hours"] == 3.5
    assert body["stale"] is False
    assert body["status"] == D.RUN["fresh_status"]
    assert body["any_early_warning"] is False
    for h in body["horizons"]:
        for col in SCHEMA_12B:
            assert col in h


def test_an_unknown_station_is_a_404_not_an_invention(client):
    c, _ = client
    r = c.get("/forecast/99999999")
    assert r.status_code == 404
    assert "not in the latest forecast" in r.json()["detail"]


def test_the_api_says_so_when_there_is_no_forecast(tmp_path, monkeypatch):
    """
    With no forecast on disk, /health must say `no_forecast` and the other
    two must return 503 rather than an empty answer that looks like "all
    clear".
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.deploy import api as A
    from src.deploy import run as RUNNER
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(RUNNER, "forecast_dir", lambda: str(empty))
    c = TestClient(A.app)
    assert c.get("/health").json()["status"] == "no_forecast"
    assert c.get("/warnings").status_code == 503
    assert c.get("/forecast/02110701").status_code == 503


def test_the_api_never_trains_or_fetches():
    """
    The API is read-only by construction. Its source must not import the
    training, fetching or retraining modules.
    """
    src = open(project_path("src", "deploy", "api.py"), encoding="utf-8").read()
    for forbidden in ("stage2", "train_dl", "fetch_live", "retrain",
                      ".fit(", "to_csv", "to_parquet"):
        assert forbidden not in src, \
            f"src/deploy/api.py mentions {forbidden!r} - it must only read"


# ===========================================================================
# 7. The parallel bootstrap
# ===========================================================================
def _toy_predictions(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    stations = [f"S{i:02d}" for i in range(6)]
    keys = pd.DataFrame({
        "station": rng.choice(stations, n),
        "time": pd.Timestamp("2026-01-01") + pd.to_timedelta(
            rng.integers(0, 24 * 90, n), unit="h")})
    y_reg = rng.normal(6.5, 1.5, n)
    y_cls = np.where(y_reg < 5, 0, np.where(y_reg <= 7, 1, 2)).astype(float)
    reg = y_reg + rng.normal(0, 0.6, n)
    cls = np.where(reg < 5, 0, np.where(reg <= 7, 1, 2)).astype(float)
    class_now = np.roll(y_cls, 1)
    return keys, y_reg, y_cls, class_now, {"reg": reg, "cls": cls}


def test_the_parallel_bootstrap_gives_identical_intervals():
    """
    The whole point of spawning one generator per resample: splitting the
    work across cores must not move a single digit of the interval.
    """
    keys, y_reg, y_cls, class_now, a = _toy_predictions()
    kw = dict(n=120, seed=7)
    serial = B.bootstrap(keys, y_reg, y_cls, class_now, a,
                         parallel=False, **kw)
    par = B.bootstrap(keys, y_reg, y_cls, class_now, a,
                      parallel=True, n_jobs=-1, batch_size=17, **kw)
    assert set(serial) == set(par) == set(B.METRICS)
    for m in B.METRICS:
        for k in ("mean", "lo", "hi", "n"):
            assert serial[m][k] == par[m][k], (
                f"parallel and serial disagree on {m}.{k}: "
                f"{serial[m][k]} vs {par[m][k]}")


def test_the_parallel_bootstrap_is_identical_for_a_difference_too():
    keys, y_reg, y_cls, class_now, a = _toy_predictions(seed=1)
    b = {"reg": a["reg"] + 0.15, "cls": a["cls"]}
    kw = dict(n=80, seed=11)
    serial = B.bootstrap(keys, y_reg, y_cls, class_now, a, b,
                         parallel=False, **kw)
    par = B.bootstrap(keys, y_reg, y_cls, class_now, a, b,
                      parallel=True, batch_size=9, **kw)
    for m in B.METRICS:
        assert serial[m]["mean"] == par[m]["mean"]
        assert serial[m]["lo"] == par[m]["lo"]
        assert serial[m]["hi"] == par[m]["hi"]


def test_the_bootstrap_block_is_the_station_week_the_project_uses():
    """The blocks come from the project's own helper, not a new one."""
    import inspect
    from src.evaluation import step6_report
    assert B.station_week_blocks is step6_report.station_week_blocks
    src = inspect.getsource(B)
    assert "early_warning_metrics" in src and "fast_rmse" in src, \
        "the bootstrap must use the project's own metric code"
    assert B.n_resamples() == CFG["ensemble"]["bootstrap"]["n_resamples"]


# ===========================================================================
# 8. The deployed models and the frozen rule
# ===========================================================================
def test_the_deployed_models_are_models_final_with_the_frozen_thresholds():
    from src.models.predict_service import STAGE_DIRS, load_thresholds
    assert D.DEP["models"]["stage"] == "final"
    assert STAGE_DIRS["final"] == CFG["final_test"]["stage3_dir"]
    taus = load_thresholds()
    for h in HORIZONS:
        key = f"{CFG['final_system']}_h{h}"
        assert key in taus, f"no frozen threshold for {key}"


def test_the_deployment_reuses_the_project_s_own_code():
    """
    Step 12 is a layer on top, not a second implementation. Each module must
    import the project's own fetchers, cleaning, features, sequences,
    metrics and prediction service rather than rolling its own.
    """
    expect = {
        "fetch_live.py": ["from src.data import download",
                          "from src.data import clean",
                          "from src.features.build_features import "
                          "build_station"],
        "score_2026.py": ["from src.evaluation.metrics import",
                          "from src.evaluation.test_2025 import",
                          "from src.models.predict_service import",
                          "from src.models import run_step7"],
        "run.py": ["from src.models.predict_service import"],
        "monitor.py": ["from src.evaluation.metrics import"],
        "retrain.py": ["from src.evaluation.metrics import"],
    }
    for name, needles in expect.items():
        src = open(project_path("src", "deploy", name),
                   encoding="utf-8").read()
        for needle in needles:
            assert needle in src, f"src/deploy/{name} does not reuse {needle}"


def test_no_deployment_module_fits_a_model_on_2026():
    """
    The deployed models are used as they are. Only retrain.py is allowed to
    fit anything, and only through stage2.py's frozen settings.
    """
    for name in ("fetch_live.py", "score_2026.py", "run.py", "monitor.py",
                 "api.py", "walk_forward.py", "boot.py"):
        src = open(project_path("src", "deploy", name),
                   encoding="utf-8").read()
        for forbidden in (".fit(", "train_model(", "fit_tabular("):
            assert forbidden not in src, \
                f"src/deploy/{name} fits a model; only retrain.py may"


# ===========================================================================
# 9. The 2026 results, when they exist
# ===========================================================================
def results_2026(name):
    path = project_path(D.TEST26["results_dir"], name)
    return pd.read_csv(path, dtype={"station": str}) \
        if os.path.exists(path) else None


def test_the_2026_window_is_the_pre_registered_one():
    summary = project_path(D.TEST26["results_dir"], "summary.json")
    if not os.path.exists(summary):
        pytest.skip("step 12.3 has not been run on this checkout")
    with open(summary, encoding="utf-8") as f:
        s = json.load(f)
    lo, hi = pd.Timestamp(s["window"][0]), pd.Timestamp(s["window"][1])
    assert lo >= pd.Timestamp(D.TEST26["start"])
    assert hi <= pd.Timestamp(D.TEST26["end_frozen"]) + pd.Timedelta("23h")
    assert s["model_folder"] == "final"
    assert s["verdict"] in ("PASS", "PARTIAL", "FAIL")


def test_the_2026_metrics_used_the_frozen_taus():
    cls = results_2026("metrics_classification.csv")
    if cls is None:
        pytest.skip("step 12.3 has not been run on this checkout")
    from src.models.predict_service import tau_for
    for h in HORIZONS:
        row = cls[(cls["method"] == CFG["final_system"])
                  & (cls["horizon"] == h) & (cls["rule"] == "warn")]
        assert len(row) == 1
        # the csv round-trip costs the last bit of the float
        assert float(row["tau"].iloc[0]) == pytest.approx(
            float(tau_for(CFG["final_system"], h)), abs=1e-9), \
            f"the 2026 scoring did not use the frozen tau at {h} h"


def test_the_2026_predictions_contain_no_hour_outside_2026():
    import glob
    files = glob.glob(os.path.join(project_path(D.TEST26["results_dir"]),
                                   "pred_*_h*.parquet"))
    if not files:
        pytest.skip("step 12.3 has not been run on this checkout")
    for path in files:
        t = pd.read_parquet(path, columns=["time"])["time"]
        assert t.min() >= pd.Timestamp(D.TEST26["start"]), path
        assert t.max() <= pd.Timestamp(D.TEST26["end_frozen"]) \
            + pd.Timedelta("23h"), path


def test_the_walk_forward_table_marks_2024_as_not_clean():
    path = project_path(D.DEP["walk_forward"]["out_csv"])
    if not os.path.exists(path):
        pytest.skip("the walk-forward table has not been built")
    df = pd.read_csv(path)
    assert set(df["test"].astype(str)) >= {"2024", "2025", "2026"}
    note = df[df["test"].astype(str) == "2024"]["note"].iloc[0]
    assert "not clean" in str(note).lower(), \
        "the 2024 row must carry its warning: it was the tuning year"


def test_the_model_card_names_the_deployed_version():
    path = project_path(D.DEP["models"]["model_card"])
    if not os.path.exists(path):
        pytest.skip("the model card has not been written")
    text = open(path, encoding="utf-8").read()
    assert D.model_version() in text, \
        ("the model card's version id does not match models/final as it is "
         "on disk")
    for needed in ["Intended use", "Limitations", "Training data"]:
        assert needed.lower() in text.lower(), \
            f"the model card has no {needed!r} section"


# ===========================================================================
# 10. Monitoring
#
# A live run forecasts the future, so on a fresh deployment there is nothing
# to score yet. These tests therefore drive the scoring path with a
# synthetic log and a synthetic truth, which is the only way to check that
# the rolling window, the onset metrics and the two drift flags behave
# before a month of real forecasts exists.
# ===========================================================================
def _synthetic_log(n_hours=200, far=0.0, station="02110701"):
    """
    A logged forecast per hour at every horizon, with the truth attached.
    `far` is the share of safe hours deliberately warned about, so a test
    can put the false-alarm rate either side of the promise.
    """
    rng = np.random.default_rng(3)
    t0 = pd.Timestamp("2026-10-01 00:00")
    rows = []
    for i in range(n_hours):
        observed = t0 + pd.Timedelta(hours=i)
        do_now = 7.5 if i % 20 else 4.2
        for h in HORIZONS:
            truth = 4.1 if (i + h) % 20 == 0 else 7.4
            safe_now = do_now >= 5.0
            onset = safe_now and truth < 5.0
            warn = bool(onset or (safe_now and rng.random() < far))
            rows.append({
                "station": station, "issued_at": observed,
                "horizon": h, "valid_at": observed + pd.Timedelta(hours=h),
                "pred_do_mgl": truth + rng.normal(0, 0.3), "p_low": 0.9
                if warn else 0.01, "tau": 0.24, "warning": warn,
                "model_version": "v", "weather_source": "era5_archive",
                "data_age_hours": 1.0, "observed_at": observed,
                "do_now_mgl": do_now, "class_now": 1.0 if safe_now else 0.0,
                "y_true_reg": truth})
    log = pd.DataFrame(rows)
    truth = log[["station", "valid_at", "y_true_reg"]].drop_duplicates()
    return log.drop(columns=["y_true_reg"]), truth


def _joined(monkeypatch, far=0.0):
    from src.deploy import monitor as M
    log, truth = _synthetic_log(far=far)
    monkeypatch.setattr(M, "log", lambda: log)
    monkeypatch.setattr(
        M, "observed",
        lambda: truth.rename(columns={"y_true_reg": "y_true_reg"}))
    return M, M.joined(now=pd.Timestamp("2026-10-20 00:00"))


def test_the_monitor_scores_the_forecasts_the_truth_has_caught_up_with(
        monkeypatch):
    M, d = _joined(monkeypatch)
    assert d is not None and len(d)
    assert d["y_true_reg"].notna().any(), \
        "the monitor attached no truth to the logged forecasts"
    perf = M.performance(d, now=pd.Timestamp("2026-10-20 00:00"))
    overall = perf[(perf["station"] == "ALL") & (perf["window"] != "all_time")]
    assert len(overall) == len(HORIZONS)
    judged = overall[overall["judged"]]
    assert len(judged), "nothing was judged although there are scored rows"
    assert (judged["rmse"] > 0).all()
    assert judged["onset_recall"].notna().all()
    # a per-station row exists as well as the overall one
    assert (perf["station"] == "02110701").any()


def test_the_monitor_flags_performance_drift_only_when_the_promise_breaks(
        monkeypatch):
    """
    The flag is the 2% promise, over the rolling window. A clean run must
    not be flagged, and one crying wolf must be.
    """
    from src.deploy import monitor as M
    clean = _joined(monkeypatch, far=0.0)[1]
    perf = M.performance(clean, now=pd.Timestamp("2026-10-20 00:00"))
    overall = perf[(perf["station"] == "ALL") & (perf["window"] != "all_time")]
    assert not overall["flag_performance_drift"].any(), \
        "a run inside the promise was flagged as drifting"

    noisy = _joined(monkeypatch, far=0.25)[1]
    perf = M.performance(noisy, now=pd.Timestamp("2026-10-20 00:00"))
    overall = perf[(perf["station"] == "ALL") & (perf["window"] != "all_time")]
    assert overall["flag_performance_drift"].any(), \
        ("a false-alarm rate far above the promise was not flagged as "
         "performance drift")


def test_the_monitor_window_and_limits_come_from_the_config():
    from src.deploy import monitor as M
    assert M.MON["window_days"] == D.MON["window_days"]
    assert float(M.MON["max_false_alarm_rate"]) == \
        float(CFG["warning_rule"]["max_false_alarm_rate"]), \
        ("the monitoring limit must be the same 2% the system promises in "
         "config.yaml, not a second copy of it")


def test_data_drift_compares_against_the_training_range(monkeypatch,
                                                        tmp_path):
    """
    A feature far outside anything the models saw in training must be
    flagged; the same feature inside its range must not be.
    """
    from src.deploy import monitor as M
    ranges = pd.DataFrame([{"feature": "do_mgl", "train_min": 0.0,
                            "train_max": 20.0, "train_mean": 7.0}])
    monkeypatch.setattr(M, "training_ranges", lambda features=None: ranges)

    def write(values):
        path = tmp_path / "features_live.parquet"
        pd.DataFrame({
            "station": "A",
            "time": pd.Timestamp("2026-10-18") - pd.to_timedelta(
                range(len(values)), unit="h"),
            "do_mgl": values}).to_parquet(path, index=False)
        monkeypatch.setattr(M.D, "dpath",
                            lambda *a, **k: str(path))
        return M.data_drift(now=pd.Timestamp("2026-10-18 01:00"))

    ok = write([7.0] * 100)
    assert not ok["flag_data_drift"].any(), \
        "an in-range feature was flagged as drifting"
    bad = write([7.0] * 50 + [250.0] * 50)
    assert bad["flag_data_drift"].all(), \
        "a feature far outside the training range was not flagged"
    assert bad["share_out_of_range"].iloc[0] == pytest.approx(0.5)


def test_the_monitor_only_scores_hours_the_sensor_really_reported():
    """
    A value the cleaning step carried forward over a short gap is a fine
    input but is not something to score a forecast against - the same rule
    step 2 used for the training targets.
    """
    src = open(project_path("src", "deploy", "monitor.py"),
               encoding="utf-8").read()
    assert 'mask_suffix' in src and '== 0' in src, \
        ("monitor.py must drop hours whose DO was filled rather than "
         "measured")


# ===========================================================================
# 11. Cleaning a window that is not 2023-2025
#
# A regression test for a real bug. `clean.clean_station` grew an `index`
# argument so the deployment could clean a 2026 window with the project's
# own code - but the weather and air tables were still being reindexed onto
# the module-level 2023-2025 spine. The concat then unioned the two
# indices, and every weather and air feature came out NaN for every hour
# the water actually covered. The 2026 test ran water-only and nobody was
# told.
#
# The symptom was visible from the results: the live-data penalty came out
# as exactly zero at every horizon, because weather the model never reads
# cannot change its answer. That is what these tests check for directly.
# ===========================================================================
def _tiny_station_csvs(folder, index):
    """One station's three raw files over an arbitrary window."""
    water = pd.DataFrame({
        "time": index,
        "do_mgl": np.linspace(6.0, 7.0, len(index)),
        "water_temp_c": 20.0, "spec_cond_uscm": 300.0, "ph": 7.2,
        "turbidity_fnu": 5.0, "discharge_cfs": 100.0})
    weather = pd.DataFrame({"time": index, **{
        v: float(i + 1) for i, v in enumerate(
            CFG["download"]["weather_vars"])}})
    air = pd.DataFrame({"time": index, **{
        v: float(i + 1) for i, v in enumerate(CFG["download"]["air_vars"])}})
    paths = []
    for name, df in (("water", water), ("weather", weather), ("air", air)):
        p = os.path.join(str(folder), f"{name}.csv")
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


def test_cleaning_a_window_outside_2023_2025_keeps_the_weather_and_air(
        tmp_path):
    """
    The bug itself. Clean a 2026 window and check that the weather and air
    columns are present for those hours - not NaN, and not on some other
    index.
    """
    from src.data import clean as CLEAN
    index = pd.date_range("2026-05-01", "2026-05-10 23:00", freq="h")
    water, weather, air = _tiny_station_csvs(tmp_path, index)
    out, dropped, _, _ = CLEAN.clean_station("TEST", water, weather, air,
                                             index=index, drop_variables=[])
    assert len(out) == len(index), \
        (f"cleaning returned {len(out)} rows for a {len(index)}-hour window - "
         f"the weather/air index was unioned in instead of aligned")
    assert out.index.equals(index)
    for v in CFG["download"]["weather_vars"]:
        assert out[v].notna().all(), f"weather column {v} came out empty"
    for v in CFG["download"]["air_vars"]:
        assert out[v].notna().all(), f"air column {v} came out empty"
    assert out["do_mgl"].notna().all()


def test_cleaning_the_default_window_is_unchanged(tmp_path):
    """
    The fix must not move the paper's own cleaning. With no `index`, the
    station still lands on the 2023-2025 spine the project has always used.
    """
    from src.data import clean as CLEAN
    index = pd.date_range("2023-01-01", "2023-01-05 23:00", freq="h")
    water, weather, air = _tiny_station_csvs(tmp_path, index)
    out, _, _, _ = CLEAN.clean_station("TEST", water, weather, air)
    assert out.index.equals(CLEAN.FULL_INDEX), \
        "the default cleaning window changed"
    assert out.loc[index, "temperature_2m"].notna().all()


def test_the_2026_features_actually_contain_weather_and_air():
    """
    The end-to-end version of the same check, on the real 2026 table. If
    these are empty, the 2026 result is a water-only result wearing a
    multimodal label.
    """
    path = D.dpath(D.DATA["features"], mode="history")
    if not os.path.exists(path):
        pytest.skip("the 2026 window has not been built on this checkout")
    import json as _json
    with open(project_path(CFG["paths"]["feature_groups"]),
              encoding="utf-8") as f:
        groups = _json.load(f)
    want = ["station", "time"] + groups["weather"][:12] + groups["air"][:8]
    df = pd.read_parquet(path, columns=want)
    for col in want[2:]:
        share = float(df[col].notna().mean())
        assert share > 0.5, \
            (f"{col} is {1 - share:.0%} missing in the 2026 feature table - "
             f"the weather/air join is broken, so the models are being fed "
             f"water only")


def test_the_live_station_codes_are_the_ones_the_models_were_trained_with():
    """
    The deep members hold one embedding row per station, indexed by the
    integer code `build_features` assigned from the sorted station list. If
    the deployment recomputed that code from a different list, every live
    forecast would be made with another station's embedding - and nothing
    would look wrong.
    """
    train = pd.read_parquet(project_path(CFG["paths"]["test"]),
                            columns=["station", "station_id"])
    truth = train.groupby("station")["station_id"].first().astype(int).to_dict()
    mine = D.station_codes()
    assert mine == truth, \
        ("the deployment's station codes differ from training's: "
         + str({k: (truth[k], mine.get(k)) for k in truth
                if mine.get(k) != truth[k]}))

    path = D.dpath(D.DATA["features"], mode="history")
    if not os.path.exists(path):
        return
    live = pd.read_parquet(path, columns=["station", "station_id"])
    live = live.groupby("station")["station_id"].first().astype(int).to_dict()
    assert all(live[s] == truth[s] for s in live), \
        "the 2026 feature table carries different station codes than training"


def test_the_tree_members_predict_on_every_cpu_core():
    """
    The trees must stay on the CPU - their predictions have to match the
    saved models exactly - but they should use all of it. `n_jobs` travels
    inside the saved model, from config.yaml, so this checks the models on
    disk rather than a setting that might not be applied.

    XGBoost is the exception and needs no setting: it defaults to every
    available core, and `nthread` is left unset in the saved booster.
    """
    import joblib
    folder = D.deployed_folder()
    if not os.path.isdir(folder):
        pytest.skip("models/final is not on this checkout")
    assert int(CFG["models"]["random_forest"]["n_jobs"]) == -1
    assert int(CFG["models"]["lightgbm"]["n_jobs"]) == -1
    assert int(CFG["models"]["xgboost"]["n_jobs"]) == -1
    assert D.tree_n_jobs() == -1, \
        "config_deploy.yaml asks the trees to use fewer than all cores"
    h = HORIZONS[0]
    rf = joblib.load(os.path.join(folder,
                                  f"random_forest_regression_h{h}.joblib"))
    assert rf.named_steps["model"].n_jobs == -1, \
        "the saved random forest would predict on one core"
    lgb = joblib.load(os.path.join(folder, f"lightgbm_regression_h{h}.joblib"))
    assert lgb["model"].get_params().get("n_jobs") == -1, \
        "the saved LightGBM would predict on one core"


def test_the_deep_members_are_asked_for_the_gpu_and_say_so_when_it_is_missing():
    """
    A silent fall back to the CPU for the deep members is the failure this
    project refuses. Every entry point that runs them must call
    `require_cuda_or_explain`, and the message must point at the README's
    install line.
    """
    for name in ("score_2026.py", "run.py", "retrain.py"):
        src = open(project_path("src", "deploy", name),
                   encoding="utf-8").read()
        assert "require_cuda_or_explain" in src, \
            f"src/deploy/{name} can run the deep models on the CPU silently"
        assert "--allow-cpu" in src, \
            f"src/deploy/{name} has no explicit way to accept the CPU"
    helptext = open(project_path("src", "deploy", "common.py"),
                    encoding="utf-8").read()
    assert "download.pytorch.org/whl/cu121" in helptext, \
        "the CPU warning does not tell the reader how to install CUDA torch"
    assert D.COMPUTE["deep_device"] in ("auto", "cuda", "cpu")
    assert D.COMPUTE["cudnn_benchmark"] is True, \
        "cudnn.benchmark should be on for repeated inference"
