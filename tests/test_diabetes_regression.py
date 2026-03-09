"""
Regression tests based on the Pima Indians Diabetes dataset.

These tests validate that the apar package reproduces the published results
from the notebook in the original ApplicabilityArea-ApAr repository. The
test fixture (uTP80Odds3489_diabetes.json) contains pre-computed ROC curves
for 5 models (Logistic Regression, Decision Trees, XGBoost, Random Forest,
SVM) each trained at 8 different cost ratios, along with the expected ApAr
values computed by the original code.

Utility parameters used in the paper for this dataset:
    u_test = -5
    u_tp   = 80
    u_fn   = 0
    u_tn   = 95  (100 + u_test)
    u_fp   = 90  (overridden by cost_ratio in applicability_area)
    prior  = 0.3489  (prevalence of diabetes among female Pima Indians)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from apar import applicability_area

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "uTP80Odds3489_diabetes.json"

# Utility parameters matching the notebook
U_TEST = -5
U_TP = 80
U_FN = 0
U_TN = 100 + U_TEST  # = 95
U_FP_BASE = 90        # overridden per-row by cost_ratio
PRIOR = 0.3489


@pytest.fixture(scope="module")
def diabetes_results() -> pd.DataFrame:
    """Load the pre-computed diabetes results from the notebook."""
    return pd.read_json(FIXTURE_PATH)


# ---------------------------------------------------------------------------
# Test every model × cost_ratio combination
# ---------------------------------------------------------------------------

def _row_ids(df: pd.DataFrame) -> list[str]:
    """Generate readable test IDs like 'LogisticRegression_cost1.0'."""
    ids = []
    for _, row in df.iterrows():
        name = row["classifiers"].replace(" ", "")
        cost = row["costRatio"]
        ids.append(f"{name}_cost{cost}")
    return ids


class TestDiabetesApAr:
    """Validate ApAr values against all 40 model×cost combinations."""

    def test_fixture_exists(self):
        assert FIXTURE_PATH.exists(), (
            f"Test fixture not found at {FIXTURE_PATH}. "
            "Copy uTP80Odds3489_diabetes.json into the tests/ directory."
        )

    def test_fixture_has_expected_rows(self, diabetes_results):
        # 5 models × 8 cost ratios = 40 rows
        assert len(diabetes_results) == 40

    @pytest.mark.parametrize("row_idx", range(40))
    def test_apar_matches_published(self, diabetes_results, row_idx):
        """Each row's ApAr should match the original notebook's result."""
        row = diabetes_results.iloc[row_idx]

        tpr = np.array(row["tpr"])
        fpr = np.array(row["fpr"])
        thresholds = np.array(row["thresholds"])
        cost_ratio = row["costRatio"]
        expected_apar = row["applicableArea"]

        result = applicability_area(
            tpr=tpr,
            fpr=fpr,
            thresholds=thresholds,
            u_tn=U_TN,
            u_tp=U_TP,
            u_fn=U_FN,
            u_fp=U_FP_BASE,
            u_test=U_TEST,
            cost_ratio=cost_ratio,
            prior=PRIOR,
        )

        assert result["apar"] == pytest.approx(expected_apar, abs=0.005), (
            f"Row {row_idx} ({row['classifiers']}, cost={cost_ratio}): "
            f"got {result['apar']}, expected {expected_apar}"
        )


# ---------------------------------------------------------------------------
# Specific high-value regression checks from the paper
# ---------------------------------------------------------------------------

class TestDiabetesKeyFindings:
    """Test specific findings highlighted in the paper/notebook."""

    def test_xgboost_cost1_highest_apar(self, diabetes_results):
        """At cost_ratio=1, xgBoost had the highest ApAr (0.286)."""
        subset = diabetes_results[diabetes_results["costRatio"] == 1.0]
        best = subset.loc[subset["applicableArea"].idxmax()]
        assert best["classifiers"] == "xgBoost"
        assert best["applicableArea"] == pytest.approx(0.286, abs=0.005)

    def test_xgboost_cost1_beats_logistic_by_apar(self, diabetes_results):
        """xgBoost has higher ApAr than Logistic Regression at cost=1,
        even though Logistic Regression has higher AUC."""
        subset = diabetes_results[diabetes_results["costRatio"] == 1.0]
        lr = subset[subset["classifiers"] == "Logistic Regression"].iloc[0]
        xgb = subset[subset["classifiers"] == "xgBoost"].iloc[0]

        # Logistic Regression has higher AUC
        assert lr["auc"] > xgb["auc"]
        # But xgBoost has higher ApAr
        assert xgb["applicableArea"] > lr["applicableArea"]

    def test_high_cost_ratios_yield_zero_apar(self, diabetes_results):
        """At very high cost ratios (10+), all models have ApAr = 0."""
        high_cost = diabetes_results[diabetes_results["costRatio"] >= 10.0]
        assert (high_cost["applicableArea"] == 0.0).all()

    def test_apar_decreases_with_cost_ratio(self, diabetes_results):
        """For each model, ApAr should generally decrease as cost ratio
        increases (i.e. as FP becomes less costly relative to FN)."""
        for model_name in diabetes_results["classifiers"].unique():
            subset = diabetes_results[
                diabetes_results["classifiers"] == model_name
            ].sort_values("costRatio")
            apars = subset["applicableArea"].values
            # Check monotonically non-increasing (allowing ties at 0)
            for i in range(len(apars) - 1):
                assert apars[i] >= apars[i + 1] - 0.01, (
                    f"{model_name}: ApAr increased from cost "
                    f"{subset['costRatio'].values[i]} to "
                    f"{subset['costRatio'].values[i+1]}"
                )

    def test_decision_trees_worst_apar_at_cost1(self, diabetes_results):
        """Decision Trees had the lowest ApAr at cost_ratio=1."""
        subset = diabetes_results[diabetes_results["costRatio"] == 1.0]
        worst = subset.loc[subset["applicableArea"].idxmin()]
        assert worst["classifiers"] == "Decision Trees"
        assert worst["applicableArea"] == pytest.approx(0.116, abs=0.005)

    def test_logistic_regression_cost1_apar(self, diabetes_results):
        """Logistic Regression at cost=1 should have ApAr = 0.252."""
        row = diabetes_results[
            (diabetes_results["classifiers"] == "Logistic Regression")
            & (diabetes_results["costRatio"] == 1.0)
        ].iloc[0]
        assert row["applicableArea"] == pytest.approx(0.252, abs=0.005)

    def test_svm_cost2_apar(self, diabetes_results):
        """SVM at cost=2 should have ApAr = 0.160."""
        row = diabetes_results[
            (diabetes_results["classifiers"] == "SVM")
            & (diabetes_results["costRatio"] == 2.0)
        ].iloc[0]
        assert row["applicableArea"] == pytest.approx(0.160, abs=0.005)

    def test_random_forest_cost1_apar(self, diabetes_results):
        """Random Forest at cost=1 should have ApAr = 0.217."""
        row = diabetes_results[
            (diabetes_results["classifiers"] == "Random Forest")
            & (diabetes_results["costRatio"] == 1.0)
        ].iloc[0]
        assert row["applicableArea"] == pytest.approx(0.217, abs=0.005)
