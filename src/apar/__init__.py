"""
ApAr: Applicability Area for evaluating predictive models.

A decision-analytic utility-based approach to evaluating predictive models
that communicates the range of prior probability and test cutoffs for which
the model has positive utility.

Reference:
    Liu S, Wei S, Lehmann HP. Applicability Area: A novel utility-based
    approach for evaluating predictive models, beyond discrimination.
    AMIA Annu Symp Proc. 2024 Jan 11;2023:494-503.
    PMID: 38222359
"""

from apar._version import __version__
from apar.core import (
    treat_all,
    treat_none,
    test_utility,
    compute_thresholds,
    compute_thresholds_over_roc,
    applicability_area,
)
from apar.plotting import plot_applicability_area, plot_utility_lines

__all__ = [
    "__version__",
    "treat_all",
    "treat_none",
    "test_utility",
    "compute_thresholds",
    "compute_thresholds_over_roc",
    "applicability_area",
    "plot_applicability_area",
    "plot_utility_lines",
]
