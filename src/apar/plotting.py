"""
Plotting utilities for visualizing the Applicability Area.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install apar[plot]"
        )


def plot_applicability_area(
    result: dict,
    ax: Optional[object] = None,
    title: Optional[str] = None,
    cmap: str = "RdYlGn",
    figsize: tuple = (8, 6),
) -> object:
    """Plot the Applicability Area diagram.

    Displays the upper (pU) and lower (pL) prior boundaries as a function
    of classification threshold, with the applicable region shaded.

    Parameters
    ----------
    result : dict
        Output from :func:`apar.applicability_area`.
    ax : matplotlib Axes, optional
        Axes to plot on. If ``None``, a new figure is created.
    title : str, optional
        Title for the plot.
    cmap : str, optional
        Colormap name for shading. Default is ``'RdYlGn'``.
    figsize : tuple, optional
        Figure size if creating a new figure. Default is ``(8, 6)``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    _check_matplotlib()

    pLs = np.array(result["pL"], dtype=float)
    pUs = np.array(result["pU"], dtype=float)
    thresholds = np.array(result["thresholds"], dtype=float)

    # Ensure arrays match in length (take the minimum)
    n = min(len(pLs), len(pUs), len(thresholds))
    pLs, pUs, thresholds = pLs[:n], pUs[:n], thresholds[:n]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the boundaries
    ax.plot(thresholds, pUs, color="tab:blue", linewidth=2, label="Upper bound (pU)")
    ax.plot(thresholds, pLs, color="tab:orange", linewidth=2, label="Lower bound (pL)")

    # Shade the applicable region
    applicable = pUs > pLs
    ax.fill_between(
        thresholds,
        pLs,
        pUs,
        where=applicable,
        alpha=0.3,
        color="green",
        label=f"ApAr = {result['apar']:.3f}",
    )

    ax.set_xlabel("Classification Threshold (Cutoff)", fontsize=12)
    ax.set_ylabel("Prior Probability", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Applicability Area (ApAr = {result['apar']:.3f})", fontsize=14)

    return ax


def plot_utility_lines(
    sensitivity: float,
    specificity: float,
    u_tn: float,
    u_tp: float,
    u_fn: float,
    u_fp: float,
    u_test: float = 0.0,
    ax: Optional[object] = None,
    figsize: tuple = (8, 5),
) -> object:
    """Plot the three expected-utility lines (treat all, treat none, test).

    This recreates the Kassirer-Pauker diagram for a single operating
    point on the ROC curve.

    Parameters
    ----------
    sensitivity, specificity : float
        Test characteristics at the chosen operating point.
    u_tn, u_tp, u_fn, u_fp : float
        Utilities for each outcome.
    u_test : float, optional
        Disutility of the test. Default is 0.
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size if creating a new figure. Default is ``(8, 5)``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    _check_matplotlib()

    from apar.core import test_utility, treat_all, treat_none

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    p = np.linspace(0, 1, 200)

    eu_all = treat_all(p, u_fp, u_tp)
    eu_none = treat_none(p, u_fn, u_tn)
    eu_test = test_utility(p, sensitivity, specificity, u_tn, u_tp, u_fn, u_fp, u_test)

    ax.plot(p, eu_all, label="Treat All", linewidth=2)
    ax.plot(p, eu_none, label="Treat None", linewidth=2)
    ax.plot(p, eu_test, label="Test", linewidth=2, linestyle="--")

    ax.set_xlabel("Prior Probability of Disease", fontsize=12)
    ax.set_ylabel("Expected Utility", fontsize=12)
    ax.set_title("Kassirer-Pauker Utility Lines", fontsize=14)
    ax.legend(loc="best")
    ax.set_xlim(0, 1)

    return ax
