"""
Tests for src/evaluation/metrics.py, on small examples worked out by hand.

If these are wrong, every number in every later report is wrong, so they are
checked against values computed on paper rather than against sklearn.

Run from the project root:   pytest -q
"""

import numpy as np
import pytest

from src.evaluation.metrics import (classification_metrics, regression_metrics,
                                    to_class)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def test_perfect_regression():
    y = [5.0, 6.0, 7.0, 8.0]
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert m["nse"] == 1.0
    assert m["n"] == 4


def test_regression_numbers_by_hand():
    """
    truth       4, 6, 8, 10   (mean 7)
    prediction  5, 5, 9,  9   (errors +1, -1, +1, -1)
    RMSE = sqrt(4/4) = 1, MAE = 1
    SST = 9 + 1 + 1 + 9 = 20, SSE = 4  ->  R2 = 1 - 4/20 = 0.8
    """
    m = regression_metrics([4.0, 6.0, 8.0, 10.0], [5.0, 5.0, 9.0, 9.0])
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(0.8)
    assert m["nse"] == pytest.approx(0.8)


def test_predicting_the_mean_gives_r2_zero():
    y = [4.0, 6.0, 8.0, 10.0]
    m = regression_metrics(y, [7.0] * 4)
    assert m["r2"] == pytest.approx(0.0)
    assert m["nse"] == pytest.approx(0.0)


def test_skill_vs_persistence():
    """
    truth        4, 6, 8, 10
    model        5, 5, 9,  9   -> RMSE 1
    persistence  2, 8, 6, 12   -> errors -2, +2, -2, +2 -> RMSE 2
    skill = 1 - 1/2 = 0.5
    """
    m = regression_metrics([4.0, 6.0, 8.0, 10.0], [5.0, 5.0, 9.0, 9.0],
                           y_persistence=[2.0, 8.0, 6.0, 12.0])
    assert m["skill_vs_persistence"] == pytest.approx(0.5)


def test_skill_is_zero_when_the_model_equals_persistence():
    y = [4.0, 6.0, 8.0]
    p = [5.0, 5.0, 9.0]
    m = regression_metrics(y, p, y_persistence=p)
    assert m["skill_vs_persistence"] == pytest.approx(0.0)


def test_rows_with_a_missing_target_are_skipped():
    """The NaN row must be dropped, not treated as a zero."""
    m = regression_metrics([4.0, np.nan, 8.0, 10.0], [5.0, 1000.0, 9.0, 9.0])
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(1.0)


def test_missing_predictions_are_skipped_too():
    m = regression_metrics([4.0, 6.0, 8.0], [5.0, np.nan, 9.0])
    assert m["n"] == 2


def test_empty_input_gives_nan_not_a_crash():
    m = regression_metrics([np.nan, np.nan], [1.0, 2.0])
    assert m["n"] == 0
    assert np.isnan(m["rmse"])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_perfect_classification():
    y = [0, 1, 2, 0, 1, 2]
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["recall_Low"] == 1.0
    assert (np.diag(m["confusion_matrix"]) == [2, 2, 2]).all()


def test_classification_numbers_by_hand():
    """
    truth       0 0 0 0  1 1  2 2 2 2
    prediction  0 0 0 1  1 2  2 2 2 0
    accuracy = 7/10 = 0.7

    Low:    predicted 5 times (4 true-0 rows minus one, plus the last row);
            tp = 3, fp = 1 -> precision 3/4 = 0.75, recall 3/4 = 0.75
    Medium: tp = 1, fp = 1 -> precision 0.5, recall 1/2 = 0.5
    Good:   tp = 3, fp = 1 -> precision 0.75, recall 3/4 = 0.75
    balanced accuracy = mean recall = (0.75 + 0.5 + 0.75)/3 = 2/3
    """
    y = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    p = [0, 0, 0, 1, 1, 2, 2, 2, 2, 0]
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["precision_Low"] == pytest.approx(0.75)
    assert m["recall_Low"] == pytest.approx(0.75)
    assert m["precision_Medium"] == pytest.approx(0.5)
    assert m["recall_Medium"] == pytest.approx(0.5)
    assert m["recall_Good"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["f1_Low"] == pytest.approx(0.75)        # P = R = 0.75


def test_confusion_matrix_layout_is_true_by_predicted():
    y = [0, 0, 1]
    p = [0, 1, 1]
    cm = classification_metrics(y, p)["confusion_matrix"]
    assert cm[0, 0] == 1       # true Low, predicted Low
    assert cm[0, 1] == 1       # true Low, predicted Medium
    assert cm[1, 1] == 1
    assert cm.shape == (3, 3)  # always 3x3, even when a class is absent


def test_a_class_that_is_never_predicted_scores_zero_not_an_error():
    y = [0, 1, 2]
    p = [0, 1, 1]               # Good never predicted
    m = classification_metrics(y, p)
    assert m["precision_Good"] == 0.0
    assert m["recall_Good"] == 0.0
    assert not np.isnan(m["f1_macro"])


def test_roc_auc_is_one_for_confident_correct_probabilities():
    y = [0, 1, 2]
    proba = [[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]]
    m = classification_metrics(y, [0, 1, 2], proba)
    assert m["roc_auc_ovr_macro"] == pytest.approx(1.0)


def test_roc_auc_is_nan_without_probabilities():
    m = classification_metrics([0, 1, 2], [0, 1, 2])
    assert np.isnan(m["roc_auc_ovr_macro"])


def test_classification_skips_missing_targets():
    y = [0, np.nan, 2]
    p = [0, 1, 2]
    m = classification_metrics(y, p)
    assert m["n"] == 2
    assert m["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# to_class, used by the persistence baseline
# ---------------------------------------------------------------------------
def test_to_class_uses_the_configured_thresholds():
    got = to_class([4.9, 5.0, 6.0, 7.0, 7.1, np.nan])
    assert got[0] == 0                 # below 5 -> Low
    assert got[1] == 1                 # exactly 5 -> Medium
    assert got[2] == 1
    assert got[3] == 1                 # exactly 7 -> Medium
    assert got[4] == 2                 # above 7 -> Good
    assert np.isnan(got[5])
