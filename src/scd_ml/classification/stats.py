"""Statistical comparison of cross-validated classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats


def corrected_paired_ttest(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_train: float,
    n_test: float,
) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores.

    The variance term ``1/k + n_test/n_train`` corrects for the overlap between
    training sets of different folds. ``n_train`` and ``n_test`` are the average
    partition sizes of the folds.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("paired scores must be 1-D arrays of equal length >= 2")
    differences = a - b
    k = len(differences)
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    corrected = (1.0 / k + n_test / n_train) * variance
    if corrected == 0:
        statistic = 0.0 if mean == 0 else float(np.sign(mean) * np.inf)
        p_value = 1.0 if mean == 0 else 0.0
    else:
        statistic = mean / np.sqrt(corrected)
        p_value = float(2 * stats.t.sf(abs(statistic), df=k - 1))
    return {"mean_difference": mean, "t_statistic": float(statistic), "p_value": p_value}


def friedman_test(scores: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Friedman test over models (keys) evaluated on the same folds (values)."""
    if len(scores) < 3:
        raise ValueError("the Friedman test needs at least three models")
    statistic, p_value = stats.friedmanchisquare(*(np.asarray(v) for v in scores.values()))
    return {"statistic": float(statistic), "p_value": float(p_value)}
