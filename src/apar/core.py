"""
Core functions for computing the Applicability Area (ApAr).

This module implements the decision-analytic utility-based framework for
evaluating predictive models. The framework considers three options in a
binary classification problem:

- **Treat All**: treat all patients as if they have the target condition
- **Treat None**: treat none, as if everyone is free of the target condition
- **Test**: use a predictive model (test) to identify those with and without
  the target condition

The theoretical basis can be found in Chapter 9 of *Medical Decision Making*
by Sox et al., and in the Kassirer-Pauker framework.

Reference:
    Liu S, Wei S, Lehmann HP. Applicability Area: A novel utility-based
    approach for evaluating predictive models, beyond discrimination.
    AMIA Annu Symp Proc. 2024 Jan 11;2023:494-503.
"""

from __future__ import annotations

import numpy as np
import sympy as sy
from numpy.typing import ArrayLike
from typing import Optional


# ---------------------------------------------------------------------------
# Utility line functions
# ---------------------------------------------------------------------------

def treat_all(
    p: float | ArrayLike,
    u_fp: float,
    u_tp: float,
) -> float | np.ndarray:
    """Expected utility for the 'treat all' strategy.

    Parameters
    ----------
    p : float or array-like
        Prior probability of the target condition.
    u_fp : float
        Utility of a false positive outcome.
    u_tp : float
        Utility of a true positive outcome.

    Returns
    -------
    float or ndarray
        Expected utility.
    """
    p = np.asarray(p)
    return p * u_tp + (1 - p) * u_fp


def treat_none(
    p: float | ArrayLike,
    u_fn: float,
    u_tn: float,
) -> float | np.ndarray:
    """Expected utility for the 'treat none' strategy.

    Parameters
    ----------
    p : float or array-like
        Prior probability of the target condition.
    u_fn : float
        Utility of a false negative outcome.
    u_tn : float
        Utility of a true negative outcome.

    Returns
    -------
    float or ndarray
        Expected utility.
    """
    p = np.asarray(p)
    return p * u_fn + (1 - p) * u_tn


def test_utility(
    p: float | ArrayLike,
    sensitivity: float,
    specificity: float,
    u_tn: float,
    u_tp: float,
    u_fn: float,
    u_fp: float,
    u_test: float = 0.0,
) -> float | np.ndarray:
    """Expected utility for the 'test' (use the model) strategy.

    Parameters
    ----------
    p : float or array-like
        Prior probability of the target condition.
    sensitivity : float
        Sensitivity (true positive rate) of the test.
    specificity : float
        Specificity (1 - false positive rate) of the test.
    u_tn : float
        Utility of a true negative outcome.
    u_tp : float
        Utility of a true positive outcome.
    u_fn : float
        Utility of a false negative outcome.
    u_fp : float
        Utility of a false positive outcome.
    u_test : float, optional
        Disutility (cost) of the test itself. Default is 0.

    Returns
    -------
    float or ndarray
        Expected utility.
    """
    p = np.asarray(p)
    return (
        p * sensitivity * u_tp
        + p * (1 - sensitivity) * u_fn
        + (1 - p) * (1 - specificity) * u_fp
        + (1 - p) * specificity * u_tn
        + u_test
    )


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def compute_thresholds(
    sensitivity: float,
    specificity: float,
    u_tn: float,
    u_tp: float,
    u_fn: float,
    u_fp: float,
    u_test: float = 0.0,
) -> dict[str, float]:
    """Compute the three probability thresholds (pL, pStar, pU).

    These thresholds are the intersections of the three expected-utility
    lines (treat all, treat none, test) and define the ranges of prior
    probability for which each strategy is optimal.

    Parameters
    ----------
    sensitivity : float
        Sensitivity (true positive rate) of the test.
    specificity : float
        Specificity (1 - false positive rate) of the test.
    u_tn : float
        Utility of a true negative outcome.
    u_tp : float
        Utility of a true positive outcome.
    u_fn : float
        Utility of a false negative outcome.
    u_fp : float
        Utility of a false positive outcome.
    u_test : float, optional
        Disutility (cost) of the test itself. Default is 0.

    Returns
    -------
    dict
        Dictionary with keys ``'pL'``, ``'pStar'``, ``'pU'`` containing the
        lower threshold, treatment threshold, and upper threshold respectively.
        A value of ``None`` indicates the threshold does not exist.
    """
    x = sy.symbols("x")

    # Upper threshold: test vs treat all
    pU_solutions = sy.solve(
        treat_all(x, u_fp, u_tp)
        - test_utility(x, sensitivity, specificity, u_tn, u_tp, u_fn, u_fp, u_test),
        x,
    )

    # Treatment threshold: treat all vs treat none
    pStar_solutions = sy.solve(
        treat_all(x, u_fp, u_tp) - treat_none(x, u_fn, u_tn), x
    )

    # Lower threshold: treat none vs test
    pL_solutions = sy.solve(
        treat_none(x, u_fn, u_tn)
        - test_utility(x, sensitivity, specificity, u_tn, u_tp, u_fn, u_fp, u_test),
        x,
    )

    def _extract(solutions):
        if len(solutions) == 0:
            return None
        val = float(solutions[0])
        if val > 1:
            return 1.0
        if val < 0:
            return 0.0
        return val

    return {
        "pL": _extract(pL_solutions),
        "pStar": _extract(pStar_solutions),
        "pU": _extract(pU_solutions),
    }


def compute_thresholds_over_roc(
    tpr: ArrayLike,
    fpr: ArrayLike,
    u_tn: float,
    u_tp: float,
    u_fn: float,
    u_fp: float,
    u_test: float = 0.0,
) -> dict[str, list[float | None]]:
    """Compute pL, pStar, pU for every operating point on the ROC curve.

    Parameters
    ----------
    tpr : array-like
        True positive rates (sensitivities) at each ROC operating point.
    fpr : array-like
        False positive rates (1 - specificities) at each ROC operating point.
    u_tn, u_tp, u_fn, u_fp : float
        Utilities for each outcome.
    u_test : float, optional
        Disutility (cost) of the test. Default is 0.

    Returns
    -------
    dict
        Dictionary with keys ``'pL'``, ``'pStar'``, ``'pU'``, each mapping
        to a list of threshold values (one per ROC operating point).
    """
    tpr = np.asarray(tpr).ravel()
    fpr = np.asarray(fpr).ravel()

    pLs, pStars, pUs = [], [], []
    for t, f in zip(tpr, fpr):
        thresholds = compute_thresholds(t, 1 - f, u_tn, u_tp, u_fn, u_fp, u_test)
        pLs.append(thresholds["pL"])
        pStars.append(thresholds["pStar"])
        pUs.append(thresholds["pU"])

    return {"pL": pLs, "pStar": pStars, "pU": pUs}


# ---------------------------------------------------------------------------
# Internal helpers for boundary adjustment
# ---------------------------------------------------------------------------

def _fill_boundary(prior_list: list, is_lower: bool) -> list:
    """Replace ``None`` values at head/tail of a boundary list."""
    result = list(prior_list)
    n = len(result)
    mid = n / 2
    for i in range(n):
        if result[i] is None:
            if is_lower:
                result[i] = 1.0 if i < mid else 0.0
            else:
                result[i] = 0.0
    return result


def _smooth_boundary(prior_list: list) -> list:
    """Apply heuristic smoothing to handle isolated boundary anomalies."""
    result = list(prior_list)
    n = len(result)
    mid = n / 2
    for i in range(n):
        if i >= 3 and i < n - 1:
            if i < mid:
                if (
                    result[i] == 1
                    and result[i - 1] < result[i - 2] < result[i - 3]
                ):
                    result[i] = 0
                elif (
                    result[i] == 0
                    and result[i - 1] > result[i - 2] > result[i - 3]
                ):
                    result[i] = 1
        if i == n - 1 and i > 0:
            if result[i - 1] != 0 and result[i] == 0:
                result[i] = result[i - 1]
    return result


def _eq_line(x, x0, x1, y0, y1):
    """Value of the line through (x0, y0) and (x1, y1) at x."""
    slope = (y1 - y0) / (x1 - x0)
    return slope * (x - x0) + y0


# ---------------------------------------------------------------------------
# Main ApAr computation
# ---------------------------------------------------------------------------

def applicability_area(
    tpr: ArrayLike,
    fpr: ArrayLike,
    thresholds: ArrayLike,
    u_tn: float,
    u_tp: float,
    u_fn: float,
    u_fp: float,
    u_test: float = 0.0,
    cost_ratio: Optional[float] = None,
    prior: Optional[float] = None,
) -> dict:
    """Compute the Applicability Area (ApAr) for a predictive model.

    The ApAr quantifies the range of prior probabilities and classification
    cutoffs for which a predictive model has positive utility over the
    alternatives of treating all or treating none.

    Parameters
    ----------
    tpr : array-like
        True positive rates from ``sklearn.metrics.roc_curve``.
    fpr : array-like
        False positive rates from ``sklearn.metrics.roc_curve``.
    thresholds : array-like
        Classification thresholds from ``sklearn.metrics.roc_curve``.
    u_tn : float
        Utility of a true negative (typically the highest, e.g. 1.0).
    u_tp : float
        Utility of a true positive (second highest).
    u_fn : float
        Utility of a false negative (lowest, e.g. 0.0).
    u_fp : float
        Utility of a false positive (third highest). If ``cost_ratio`` is
        provided, ``u_fp`` is computed from the cost ratio and this parameter
        is ignored.
    u_test : float, optional
        Disutility (cost) of the test. Default is 0.
    cost_ratio : float, optional
        Asymmetric cost ratio. If provided, ``u_fp`` is recomputed as:
        ``u_fp = u_tn - (u_tp - u_fn) / cost_ratio``.
    prior : float, optional
        A specific prior probability of interest. If given, the result
        includes whether this prior falls within the model's applicable range.

    Returns
    -------
    dict
        - ``'apar'`` (float): The Applicability Area value in [0, 1].
        - ``'best_cutoff_index'`` (int): Index of the threshold with the
          largest range of applicable priors.
        - ``'prior_in_range'`` (bool): Whether ``prior`` is within the
          applicable range (only meaningful if ``prior`` was given).
        - ``'pL'`` (list): Lower prior boundaries per cutoff.
        - ``'pU'`` (list): Upper prior boundaries per cutoff.
        - ``'thresholds'`` (ndarray): Adjusted classification thresholds.

    Examples
    --------
    >>> from sklearn.metrics import roc_curve
    >>> from apar import applicability_area
    >>> fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    >>> result = applicability_area(
    ...     tpr, fpr, thresholds,
    ...     u_tn=1.0, u_tp=0.8, u_fn=0.0, u_fp=0.6,
    ... )
    >>> print(f"ApAr = {result['apar']}")
    """
    tpr = np.asarray(tpr).ravel()
    fpr = np.asarray(fpr).ravel()
    thresholds = np.asarray(thresholds).ravel().copy()

    # Recompute u_fp from cost ratio if provided
    if cost_ratio is not None:
        u_fp = u_tn - (u_tp - u_fn) * (1 / cost_ratio)

    # Compute pL and pU over the ROC
    roc_thresholds = compute_thresholds_over_roc(
        tpr, fpr, u_tn, u_tp, u_fn, u_fp, u_test
    )
    pLs = list(roc_thresholds["pL"])
    pUs = list(roc_thresholds["pU"])

    # Adjust classification thresholds
    thresholds = np.where(thresholds > 1, 1, thresholds)

    # Fill and smooth boundaries
    pLs = _fill_boundary(pLs, is_lower=True)
    pLs = _smooth_boundary(pLs)
    pUs = _fill_boundary(pUs, is_lower=False)
    pUs = _smooth_boundary(pUs)

    # Handle edge case where last threshold is 0
    if thresholds[-1] == 0:
        thresholds[-1] = 0.0001
        thresholds = np.append(thresholds, 0)
        pLs.insert(0, pLs[0] if len(pLs) > 1 else 0)
        pUs.insert(0, pUs[0] if len(pUs) > 1 else 0)

    # Reverse thresholds to be in ascending order
    thresholds = thresholds[::-1]

    # Compute applicability area using trapezoidal integration
    area = 0.0
    best_range = 0.0
    best_index = -1

    for i in range(len(pLs) - 1):
        if i + 1 >= len(thresholds):
            break

        pL_i, pU_i = pLs[i], pUs[i]
        pL_next, pU_next = pLs[i + 1], pUs[i + 1]

        if pL_i is None or pU_i is None or pL_next is None or pU_next is None:
            continue

        if pL_i < pU_i and pL_next < pU_next:
            # Both endpoints have positive range
            range_prior = pU_i - pL_i
            if range_prior > best_range:
                best_range = range_prior
                best_index = i
            avg_range = (range_prior + (pU_next - pL_next)) / 2
            area += abs(avg_range) * abs(thresholds[i + 1] - thresholds[i])

        elif pL_i > pU_i and pL_next < pU_next:
            # Crossing into positive range
            x0, x1 = thresholds[i], thresholds[i + 1]
            if x0 != x1:
                x_sym = sy.symbols("x")
                x_int = sy.solve(
                    _eq_line(x_sym, x0, x1, pL_i, pL_next)
                    - _eq_line(x_sym, x0, x1, pU_i, pU_next),
                    x_sym,
                )
                if x_int:
                    avg_range = (0 + (pU_next - pL_next)) / 2
                    area += abs(avg_range) * abs(thresholds[i + 1] - float(x_int[0]))

        elif pL_i < pU_i and pL_next > pU_next:
            # Crossing out of positive range
            x0, x1 = thresholds[i], thresholds[i + 1]
            if x0 != x1:
                x_sym = sy.symbols("x")
                x_int = sy.solve(
                    _eq_line(x_sym, x0, x1, pL_i, pL_next)
                    - _eq_line(x_sym, x0, x1, pU_i, pU_next),
                    x_sym,
                )
                if x_int:
                    avg_range = (0 + (pU_i - pL_i)) / 2
                    area += abs(avg_range) * abs(float(x_int[0]) - thresholds[i + 1])

    area = round(float(area), 3)
    area = min(area, 1.0)

    # Check if specified prior is within range
    prior_in_range = False
    if prior is not None:
        for pL_i, pU_i in zip(pLs, pUs):
            if pL_i is not None and pU_i is not None and pL_i < prior < pU_i:
                prior_in_range = True
                break

    return {
        "apar": area,
        "best_cutoff_index": best_index,
        "prior_in_range": prior_in_range,
        "pL": pLs,
        "pU": pUs,
        "thresholds": thresholds,
    }
