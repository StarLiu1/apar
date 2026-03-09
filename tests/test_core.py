"""Tests for the apar package."""

import numpy as np
import pytest


def test_import():
    """Package imports successfully."""
    import apar
    assert hasattr(apar, "__version__")
    assert hasattr(apar, "applicability_area")


def test_treat_all():
    from apar import treat_all
    # At p=0, expected utility = u_fp
    assert treat_all(0.0, u_fp=0.3, u_tp=0.8) == pytest.approx(0.3)
    # At p=1, expected utility = u_tp
    assert treat_all(1.0, u_fp=0.3, u_tp=0.8) == pytest.approx(0.8)


def test_treat_none():
    from apar import treat_none
    # At p=0, expected utility = u_tn
    assert treat_none(0.0, u_fn=0.0, u_tn=1.0) == pytest.approx(1.0)
    # At p=1, expected utility = u_fn
    assert treat_none(1.0, u_fn=0.0, u_tn=1.0) == pytest.approx(0.0)


def test_test_utility():
    from apar import test_utility
    # Perfect test (sens=1, spec=1) should equal treat_all at p=1
    result = test_utility(
        1.0, sensitivity=1.0, specificity=1.0,
        u_tn=1.0, u_tp=0.8, u_fn=0.0, u_fp=0.3,
    )
    assert result == pytest.approx(0.8)


def test_compute_thresholds():
    from apar import compute_thresholds
    result = compute_thresholds(
        sensitivity=0.9, specificity=0.85,
        u_tn=1.0, u_tp=0.8, u_fn=0.0, u_fp=0.6,
    )
    assert "pL" in result
    assert "pStar" in result
    assert "pU" in result
    # pL should be less than pU for a decent test
    if result["pL"] is not None and result["pU"] is not None:
        assert result["pL"] <= result["pU"]


def test_applicability_area_basic():
    from apar import applicability_area
    # Simple synthetic ROC curve
    fpr = np.array([0.0, 0.1, 0.3, 0.5, 1.0])
    tpr = np.array([0.0, 0.5, 0.8, 0.9, 1.0])
    thresholds = np.array([1.0, 0.8, 0.5, 0.3, 0.0])

    result = applicability_area(
        tpr=tpr, fpr=fpr, thresholds=thresholds,
        u_tn=1.0, u_tp=0.8, u_fn=0.0, u_fp=0.6,
    )

    assert "apar" in result
    assert 0.0 <= result["apar"] <= 1.0
    assert "pL" in result
    assert "pU" in result
    assert "thresholds" in result


def test_applicability_area_with_cost_ratio():
    from apar import applicability_area
    fpr = np.array([0.0, 0.1, 0.3, 0.5, 1.0])
    tpr = np.array([0.0, 0.5, 0.8, 0.9, 1.0])
    thresholds = np.array([1.0, 0.8, 0.5, 0.3, 0.0])

    result = applicability_area(
        tpr=tpr, fpr=fpr, thresholds=thresholds,
        u_tn=1.0, u_tp=0.8, u_fn=0.0, u_fp=0.0,
        cost_ratio=2.0,
    )
    assert 0.0 <= result["apar"] <= 1.0


def test_applicability_area_with_prior():
    from apar import applicability_area
    fpr = np.array([0.0, 0.1, 0.3, 0.5, 1.0])
    tpr = np.array([0.0, 0.5, 0.8, 0.9, 1.0])
    thresholds = np.array([1.0, 0.8, 0.5, 0.3, 0.0])

    result = applicability_area(
        tpr=tpr, fpr=fpr, thresholds=thresholds,
        u_tn=1.0, u_tp=0.8, u_fn=0.0, u_fp=0.6,
        prior=0.3,
    )
    assert isinstance(result["prior_in_range"], bool)


def test_vectorized_treat_all():
    from apar import treat_all
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = treat_all(p, u_fp=0.3, u_tp=0.8)
    assert result.shape == (5,)
    assert result[0] == pytest.approx(0.3)
    assert result[-1] == pytest.approx(0.8)
