# ApAr: Applicability Area

[![PyPI version](https://badge.fury.io/py/apar.svg)](https://pypi.org/project/apar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**A decision-analytic utility-based approach to evaluating predictive models, beyond discrimination.**

ApAr communicates the range of prior probability and test cutoffs for which a predictive model has positive utility — larger ApAr values suggest broader potential use of the model.

## Installation

```bash
pip install apar
```

For plotting support:

```bash
pip install apar[plot]
```

## Quick Start

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import numpy as np
from apar import applicability_area

# Example: binary classification
X, y = load_diabetes(return_X_y=True)
y_binary = (y > np.median(y)).astype(int)

model = LogisticRegression(max_iter=1000).fit(X, y_binary)
y_scores = model.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y_binary, y_scores)

# Compute ApAr
result = applicability_area(
    tpr=tpr,
    fpr=fpr,
    thresholds=thresholds,
    u_tn=1.0,    # utility of true negative (highest)
    u_tp=0.8,    # utility of true positive
    u_fn=0.0,    # utility of false negative (lowest)
    u_fp=0.6,    # utility of false positive
)

print(f"Applicability Area: {result['apar']}")
```

## Visualize

```python
from apar import plot_applicability_area

plot_applicability_area(result, title="My Model's Applicability Area")
```

## Key Concepts

The framework considers three clinical strategies in a binary classification problem:

| Strategy | Description |
|---|---|
| **Treat All** | Treat all patients as if they have the condition |
| **Treat None** | Treat no one — assume everyone is free of the condition |
| **Test** | Use the predictive model to decide who to treat |

ApAr integrates the range of prior probabilities over the ROC curve where the **Test** strategy has the highest expected utility. This goes beyond AUROC by incorporating the decision context (utilities/costs) into model evaluation.

## API Reference

### Core Functions

- **`applicability_area(tpr, fpr, thresholds, u_tn, u_tp, u_fn, u_fp, ...)`** — Compute the ApAr metric.
- **`compute_thresholds(sensitivity, specificity, u_tn, u_tp, u_fn, u_fp)`** — Get the pL/pStar/pU thresholds for a single operating point.
- **`compute_thresholds_over_roc(tpr, fpr, u_tn, u_tp, u_fn, u_fp)`** — Thresholds for every ROC operating point.
- **`treat_all(p, u_fp, u_tp)`** — Expected utility of the "treat all" strategy.
- **`treat_none(p, u_fn, u_tn)`** — Expected utility of the "treat none" strategy.
- **`test_utility(p, sensitivity, specificity, u_tn, u_tp, u_fn, u_fp)`** — Expected utility of the "test" strategy.

### Plotting

- **`plot_applicability_area(result)`** — Plot the ApAr diagram with shaded applicable region.
- **`plot_utility_lines(sensitivity, specificity, u_tn, u_tp, u_fn, u_fp)`** — Plot the Kassirer-Pauker utility lines.

## Citation

If you use ApAr in your research, please cite:

```bibtex
@article{liu2023applicability,
  title={Applicability Area: A novel utility-based approach for evaluating predictive models, beyond discrimination},
  author={Liu, Star and Wei, Shixiong and Lehmann, Harold P},
  journal={AMIA Annual Symposium Proceedings},
  volume={2023},
  pages={494--503},
  year={2024},
  publisher={American Medical Informatics Association}
}
```

> Liu S, Wei S, Lehmann HP. Applicability Area: A novel utility-based approach for evaluating predictive models, beyond discrimination. *AMIA Annu Symp Proc.* 2024 Jan 11;2023:494-503. PMID: [38222359](https://pubmed.ncbi.nlm.nih.gov/38222359/)

## License

MIT License. See [LICENSE](LICENSE) for details.
