"""
Tests for step 8: the freeze and the 2025 test.

These are the tests that protect the headline claim of the whole project -
that 2025 was scored once, against a system fixed in advance. If any of
them fail, the final numbers mean nothing.

Run from the project root:   pytest -q
"""

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path

CFG = load_config()
FT = CFG["final_test"]
HORIZONS = CFG["horizons"]
TEST_YEARS = CFG["split"]["test_years"]


def have(path):
    return os.path.exists(project_path(path))


# ---------------------------------------------------------------------------
# FROZEN.md
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def frozen_text():
    if not have(FT["frozen_file"]):
        pytest.skip("run `python -m src.models.freeze` first")
    with open(project_path(FT["frozen_file"]), encoding="utf-8") as f:
        return f.read()


def test_frozen_file_records_the_essentials(frozen_text):
    for needed in ["## The split", "## Members and settings",
                   "## Rounds and epochs", "## Warning thresholds",
                   "## Ablation thresholds", "## Pass criteria",
                   "## File hashes"]:
        assert needed in frozen_text, f"FROZEN.md is missing {needed}"
    assert re.search(r"at commit `[0-9a-f]{40}`", frozen_text), \
        "FROZEN.md does not record a git commit"


def test_frozen_hashes_match_the_files_on_disk(frozen_text):
    """
    Every model file the test reads must still be byte-for-byte what was
    frozen. A mismatch means something was retrained after the freeze.
    """
    rows = re.findall(r"^\| (models/[^|]+) \| `([0-9a-f]{64})` \|",
                      frozen_text, flags=re.M)
    assert len(rows) > 0, "no hashes recorded"
    checked, missing, changed = 0, [], []
    for rel, expected in rows:
        path = project_path(rel.strip())
        if not os.path.exists(path):
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            changed.append(rel)
        checked += 1
    assert not changed, f"these files changed after the freeze: {changed}"
    assert not missing, f"these frozen files are gone: {missing}"
    assert checked >= 50, f"only {checked} files verified"


# ---------------------------------------------------------------------------
# Nothing in the freeze or stage 2 used 2025
# ---------------------------------------------------------------------------
def test_warning_thresholds_were_chosen_without_2025():
    """
    The thresholds come from the step-7 out-of-fold files, which only ever
    contain validation-year rows. This checks that directly.
    """
    if not have(FT["thresholds_file"]):
        pytest.skip("run the freeze first")
    oof = project_path(CFG["step7"]["oof_predictions"])
    for h in HORIZONS:
        path = os.path.join(oof, f"stacking_classification_h{h}.parquet")
        if not os.path.exists(path):
            continue
        years = set(pd.read_parquet(path, columns=["time"])["time"].dt.year)
        assert years <= set(CFG["split"]["val_years"]), \
            f"the out-of-fold file for h={h} contains {years}"


def test_thresholds_cover_every_method_and_horizon():
    if not have(FT["thresholds_file"]):
        pytest.skip("run the freeze first")
    with open(project_path(FT["thresholds_file"]), encoding="utf-8") as f:
        taus = json.load(f)
    for method in ["stacking", "simple_average", "lightgbm", "xgboost"]:
        for h in HORIZONS:
            assert f"{method}_h{h}" in taus, f"no threshold for {method} h{h}"


def test_the_selection_budget_is_tighter_than_the_promise():
    """
    Step 7 found a tau tuned to the limit overshoots out of sample, so the
    budget used to CHOOSE must be below the budget the system promises.
    """
    assert FT["select_false_alarm_rate"] < FT["target_false_alarm_rate"]
    assert np.isclose(FT["select_false_alarm_rate"],
                      FT["target_false_alarm_rate"] - FT["safety_margin"])


def test_stage2_training_rows_contain_no_2025():
    """Stage 2 trains on 2023 + 2024. If 2025 crept in, the test is void."""
    from src.models.stage2 import STAGE_YEARS
    assert "test" not in STAGE_YEARS[2]
    years = set()
    for split in STAGE_YEARS[2]:
        t = pd.read_parquet(project_path(CFG["paths"][split]), columns=["time"])
        years |= set(t["time"].dt.year.unique())
    assert not (years & set(TEST_YEARS)), \
        f"stage 2 would train on {years & set(TEST_YEARS)}"


def test_stage3_is_the_only_thing_that_uses_2025():
    """Stage 3 is for the app and is never scored; it may use everything."""
    from src.models.stage2 import STAGE_YEARS
    assert "test" in STAGE_YEARS[3]


def test_ablation_thresholds_were_chosen_on_2024():
    if not have(FT["ablation_thresholds_file"]):
        pytest.skip("run the freeze first")
    with open(project_path(FT["ablation_thresholds_file"]), encoding="utf-8") as f:
        abl = json.load(f)
    for s in FT["ablation_sets"]:
        for h in HORIZONS:
            key = f"{s}_h{h}"
            assert key in abl, f"no ablation threshold for {key}"
            # the recorded recall comes from 2024, so it must be a real number
            assert not np.isnan(abl[key]["onset_recall_2024"])


# ---------------------------------------------------------------------------
# The models the test and the app need
# ---------------------------------------------------------------------------
def test_stage2_models_exist_for_every_member_and_horizon():
    folder = project_path(FT["stage2_dir"])
    if not os.path.isdir(folder):
        pytest.skip("run `python -m src.models.stage2` first")
    files = set(os.listdir(folder))
    for name in ["random_forest", "linear"]:
        for task in ["regression", "classification"]:
            for h in HORIZONS:
                assert f"{name}_{task}_h{h}.joblib" in files, \
                    f"missing {name}_{task}_h{h}"
    for h in HORIZONS:
        assert f"xgboost_regression_h{h}.json" in files
        assert f"lightgbm_classification_h{h}.joblib" in files
    for name in ["bilstm", "cnn_lstm"]:
        for seed in CFG["dl_search"]["final_seeds"]:
            assert f"{name}_seed{seed}.pt" in files, \
                f"missing {name}_seed{seed}.pt"
    assert "dl_scaler.joblib" in files


def test_stage3_files_exist_for_the_app():
    folder = project_path(FT["stage3_dir"])
    if not os.path.isdir(folder):
        pytest.skip("run `python -m src.models.stage2 --stage 3` first")
    files = set(os.listdir(folder))
    assert any(f.startswith("lightgbm_") for f in files)
    assert any(f.startswith("bilstm_seed") for f in files)
    assert "dl_scaler.joblib" in files
    # and the stacking meta-models the app needs on top of them
    ens = project_path(CFG["context_ensemble"]["model_dir"])
    for task in ["regression", "classification"]:
        for h in HORIZONS:
            assert os.path.exists(
                os.path.join(ens, f"stacking_{task}_h{h}_final.joblib")), \
                f"no stacking meta-model for {task} h{h}"


# ---------------------------------------------------------------------------
# The test results themselves
# ---------------------------------------------------------------------------
def test_test_predictions_cover_only_2025():
    folder = project_path(FT["results_dir"])
    path = os.path.join(folder, "pred_stacking_h24.parquet")
    if not os.path.exists(path):
        pytest.skip("run `python -m src.evaluation.test_2025` first")
    d = pd.read_parquet(path, columns=["time"])
    assert set(d["time"].dt.year.unique()) <= set(TEST_YEARS)


def test_the_warning_rule_used_the_frozen_tau():
    """The predictions on disk must match the frozen threshold, not a new one."""
    folder = project_path(FT["results_dir"])
    path = os.path.join(folder, "pred_stacking_h24.parquet")
    if not os.path.exists(path) or not have(FT["thresholds_file"]):
        pytest.skip("run the test first")
    with open(project_path(FT["thresholds_file"]), encoding="utf-8") as f:
        tau = json.load(f)["stacking_h24"]["tau"]
    d = pd.read_parquet(path)
    from src.models.run_step7 import apply_warning
    expected = apply_warning(d[[f"proba_{c}" for c in
                                CFG["target"]["class_names"]]].to_numpy(), tau)
    assert (expected == d["pred_warn"].to_numpy()).all(), \
        "the saved warnings do not match the frozen threshold"


def test_lead_times_are_within_the_window():
    path = os.path.join(project_path(FT["results_dir"]), "lead_times.csv")
    if not os.path.exists(path):
        pytest.skip("run the test first")
    d = pd.read_csv(path)
    warned = d[d["warned"]]
    assert (warned["lead_hours"] > 0).all(), "a lead time must be positive"
    assert (warned["lead_hours"] <= FT["lead_time_window_hours"]).all(), \
        "a lead time longer than the search window is impossible"
