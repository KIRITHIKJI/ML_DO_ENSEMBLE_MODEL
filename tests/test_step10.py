"""
Tests for step 10: the additional design checks.

Step 10 is allowed to ADD analyses and nothing else. These tests say so in
code: the frozen system is untouched, no file FROZEN.md hashes is written
by step 10, the re-scored subsets agree with the headline where they are
supposed to be identical, and the rebuilt stage-1 chain reproduces the
numbers step 8 already recorded.

Run from the project root:   pytest -q
"""

import os
import re

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path

CFG = load_config()
FT = CFG["final_test"]
HORIZONS = CFG["horizons"]
MEMBERS = CFG["ensemble"]["members"]
RESULTS = project_path(FT["results_dir"])
PLOTS = project_path(FT["plots_dir"])

STEP10_FILES = ["step10_pipeline.csv", "step10_dropout.csv",
                "step10_seed_spread.csv", "step10_method_gaps.csv",
                "step10_error_correlation_h24.csv"]


def step10(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        pytest.skip("run `python -m src.evaluation.step10_checks` first")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Step 10 changed nothing
# ---------------------------------------------------------------------------
def test_the_final_system_is_still_the_frozen_one():
    """Step 10 is analysis. The deployed choice stays what step 7 made it."""
    assert CFG["final_system"] == "stacking"


def test_step10_writes_nothing_that_frozen_md_hashes():
    """
    Every path FROZEN.md pins must be outside the set of files step 10
    produces. If step 10 ever overwrote one of them, the 2025 test would no
    longer be the test FROZEN.md describes.
    """
    with open(project_path(FT["frozen_file"]), encoding="utf-8") as f:
        frozen = set(re.findall(r"^\| (models/[^|]+) \| `[0-9a-f]{64}` \|",
                                f.read(), flags=re.M))
    assert frozen, "no hashes recorded in FROZEN.md"
    written = {os.path.join(FT["results_dir"], n).replace("\\", "/")
               for n in STEP10_FILES}
    written |= {os.path.join(FT["plots_dir"],
                             "error_correlation_h24.png").replace("\\", "/")}
    written |= {os.path.join(FT["results_dir"],
                             f"pred_stacking_h{h}_stage1.parquet"
                             ).replace("\\", "/") for h in HORIZONS}
    clash = {p.strip() for p in frozen} & written
    assert not clash, f"step 10 would overwrite frozen files: {clash}"
    # and all of them live under results/, which FROZEN.md never pins
    assert all(p.strip().startswith("models/") for p in frozen)


def test_step10_did_not_touch_the_warning_thresholds():
    """
    The thresholds are the system's only tunable at decision time. Step 10
    must read them, never rewrite them, so the ones on disk must still be
    the ones the step-8 predictions were made with.
    """
    import json
    with open(project_path(FT["thresholds_file"]), encoding="utf-8") as f:
        taus = json.load(f)
    cls = pd.read_csv(os.path.join(RESULTS,
                                   "metrics_classification_stage2.csv"))
    for h in HORIZONS:
        used = cls[(cls.method == "stacking") & (cls.horizon == h)
                   & (cls.rule == "warn")]["tau"].iloc[0]
        assert np.isclose(float(used), taus[f"stacking_h{h}"]["tau"]), \
            f"the tau on disk for h{h} is not the one the test used"


# ---------------------------------------------------------------------------
# 10.1  the consistent pipeline
# ---------------------------------------------------------------------------
def test_101_reproduces_the_step8_stage1_metrics():
    """
    The stage-1 rows step 10 rebuilt are pure inference with the frozen
    2023-only models, so they must land on the stage-1 metrics step 8
    already wrote. This is the check that the rebuild is the same chain.
    """
    from src.evaluation.step10_checks import check_stage1_matches_step8
    if not all(os.path.exists(os.path.join(
            RESULTS, f"pred_stacking_h{h}_stage1.parquet"))
            for h in HORIZONS):
        pytest.skip("run `python -m src.evaluation.step10_checks` first")
    rows = {h: pd.read_parquet(os.path.join(
        RESULTS, f"pred_stacking_h{h}_stage1.parquet")) for h in HORIZONS}
    problems = check_stage1_matches_step8(rows)
    assert not problems, "the rebuilt stage-1 chain drifted: " + "; ".join(problems)


def test_101_covers_every_horizon_with_a_real_interval():
    d = step10("step10_pipeline.csv")
    assert sorted(d["horizon"]) == sorted(HORIZONS)
    for metric in ["rmse", "f1_macro", "onset_recall", "false_alarm_rate"]:
        lo, hi = d[f"{metric}_lo"], d[f"{metric}_hi"]
        assert (lo <= d[f"{metric}_diff"]).all() and \
               (d[f"{metric}_diff"] <= hi).all(), \
            f"the {metric} difference falls outside its own CI"
        assert (hi > lo).all(), f"a degenerate CI for {metric}"


def test_101_headline_column_matches_the_step8_tables():
    """The stage-2 side of the comparison must be the published headline."""
    d = step10("step10_pipeline.csv").set_index("horizon")
    reg = pd.read_csv(os.path.join(RESULTS, "metrics_regression_stage2.csv"))
    cls = pd.read_csv(os.path.join(RESULTS,
                                   "metrics_classification_stage2.csv"))
    for h in HORIZONS:
        want_r = reg[(reg.method == "stacking") & (reg.horizon == h)].iloc[0]
        want_c = cls[(cls.method == "stacking") & (cls.horizon == h)
                     & (cls.rule == "warn")].iloc[0]
        assert np.isclose(d.loc[h, "rmse_stage2"], want_r["rmse"], atol=5e-4)
        assert np.isclose(d.loc[h, "f1_macro_stage2"], want_c["f1_macro"],
                          atol=5e-4)
        assert np.isclose(d.loc[h, "onset_recall_stage2"],
                          want_c["onset_recall"], atol=5e-4)


# ---------------------------------------------------------------------------
# 10.2  station dropout
# ---------------------------------------------------------------------------
def test_102_headline_subset_equals_the_published_numbers():
    """
    Re-scoring is only meaningful if the un-restricted subset reproduces
    the published table exactly. If it does not, the scorer differs from
    step 8's and none of the other rows can be compared with the headline.
    """
    d = step10("step10_dropout.csv")
    reg = pd.read_csv(os.path.join(RESULTS, "metrics_regression_stage2.csv"))
    cls = pd.read_csv(os.path.join(RESULTS,
                                   "metrics_classification_stage2.csv"))
    head = d[d["subset"].str.startswith("headline")].set_index("horizon")
    assert len(head) == len(HORIZONS)
    for h in HORIZONS:
        want_r = reg[(reg.method == "stacking") & (reg.horizon == h)].iloc[0]
        want_c = cls[(cls.method == "stacking") & (cls.horizon == h)
                     & (cls.rule == "warn")].iloc[0]
        assert np.isclose(head.loc[h, "rmse"], want_r["rmse"], atol=1e-6)
        assert np.isclose(head.loc[h, "f1_macro"], want_c["f1_macro"],
                          atol=1e-6)
        assert np.isclose(head.loc[h, "onset_recall"], want_c["onset_recall"],
                          atol=1e-6)
        assert np.isclose(head.loc[h, "false_alarm_rate"],
                          want_c["false_alarm_rate"], atol=1e-6)


def test_102_the_subsets_really_are_smaller():
    d = step10("step10_dropout.csv")
    for h in HORIZONS:
        g = d[d["horizon"] == h].set_index("subset")
        head = g[g.index.str.startswith("headline")]["n"].iloc[0]
        assert (g["n"] <= head).all(), "a subset has more rows than the whole"
        assert (g["n"] < head).sum() == 2, \
            "both dropout subsets should be strictly smaller"


def test_102_false_alarms_stay_inside_the_promise():
    """The 2% promise has to survive the sensitivity analysis to mean much."""
    d = step10("step10_dropout.csv")
    worst = float(d["false_alarm_rate"].max())
    assert worst <= FT["target_false_alarm_rate"], \
        f"a subset breaks the false-alarm budget: {worst:.4f}"


# ---------------------------------------------------------------------------
# 10.3  seed spread
# ---------------------------------------------------------------------------
def test_103_seed_spread_covers_both_deep_models():
    d = step10("step10_seed_spread.csv")
    assert set(d["model"]) == {"bilstm", "cnn_lstm"}
    assert (d["rmse_std"] > 0).all() and (d["f1_std"] > 0).all()
    for r in d.itertuples():
        assert len(r.seeds.split(";")) == len(CFG["dl_search"]["final_seeds"])


def test_103_gaps_are_read_from_the_published_tables():
    d = step10("step10_method_gaps.csv")
    reg = pd.read_csv(os.path.join(RESULTS, "metrics_regression_stage2.csv"))
    reg24 = reg[reg.horizon == 24].set_index("method")
    want = abs(reg24.loc["stacking", "rmse"] - reg24.loc["lightgbm", "rmse"])
    got = d.set_index("comparison").loc[
        "stacking vs best single (lightgbm)", "rmse_gap"]
    assert np.isclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# 10.4  error correlation
# ---------------------------------------------------------------------------
def test_104_correlation_matrix_is_well_formed():
    d = step10("step10_error_correlation_h24.csv")
    d = d.set_index(d.columns[0])
    assert list(d.columns) == MEMBERS and list(d.index) == MEMBERS
    m = d.to_numpy()
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m, m.T), "a correlation matrix must be symmetric"
    assert ((m >= -1.0) & (m <= 1.0)).all()


def test_104_errors_are_pred_minus_true():
    """The sign convention the report states, checked against the rows."""
    h = 24
    d = pd.read_parquet(os.path.join(RESULTS, f"pred_stacking_h{h}.parquet"))
    e = d["pred_reg"].to_numpy() - d["y_true_reg"].to_numpy()
    ok = ~np.isnan(e)
    rmse = float(np.sqrt(np.mean(e[ok] ** 2)))
    reg = pd.read_csv(os.path.join(RESULTS, "metrics_regression_stage2.csv"))
    want = reg[(reg.method == "stacking") & (reg.horizon == h)]["rmse"].iloc[0]
    assert np.isclose(rmse, want, atol=1e-6)


def test_104_heatmap_was_written():
    path = os.path.join(PLOTS, "error_correlation_h24.png")
    if not os.path.exists(path):
        pytest.skip("run `python -m src.evaluation.step10_checks` first")
    assert os.path.getsize(path) > 10_000


# ---------------------------------------------------------------------------
# 10.5  the report and the app
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report():
    path = project_path("reports", "final_results.md")
    if not os.path.exists(path):
        pytest.skip("run `python -m src.evaluation.final_report` first")
    return open(path, encoding="utf-8").read()


def test_the_report_has_the_two_new_sections(report):
    assert "## Design choices and how they were checked" in report
    assert "## Scope and future work" in report
    assert "## Limitations" not in report, \
        "the old Limitations section should have been replaced"


def test_the_report_labels_the_new_work_as_analysis_only(report):
    sec = report.split("## Design choices and how they were checked")[1]
    sec = sec.split("## Scope and future work")[0]
    assert "additional analyses" in sec.lower()
    assert "analysis only" in sec.lower()
    for needed in ["train/apply mismatch", "2025-11-19", "correlat",
                   "warning rule", "seed"]:
        assert needed.lower() in sec.lower(), f"section 10.5 lost {needed}"


def test_the_scope_section_says_what_is_future_work(report):
    sec = report.split("## Scope and future work")[1]
    for needed in ["monitored stations", "USGS", "CPCB", "One test year",
                   "ATAE"]:
        assert needed in sec, f"the scope section lost {needed}"


def test_the_app_reads_the_report_rather_than_hard_coding_it():
    src = open(project_path("app", "app.py"), encoding="utf-8").read()
    assert "report_section(" in src
    assert "## Design choices and how they were checked" in src
    assert "## Scope and future work" in src
