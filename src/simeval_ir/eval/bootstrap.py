"""Bootstrap resampling utilities."""

from typing import Callable

import numpy as np


def bootstrap_ci(
    data: np.ndarray | list,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: Input data array.
        statistic: Function to compute the statistic (default: mean).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    data = np.asarray(data)
    n = len(data)

    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    if n == 1:
        val = statistic(data)
        return (val, val, val)

    rng = np.random.default_rng(random_state)

    # Compute bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    point_estimate = statistic(data)

    return (float(point_estimate), float(ci_lower), float(ci_upper))


def paired_bootstrap_test(
    x: np.ndarray | list,
    y: np.ndarray | list,
    n_bootstrap: int = 10000,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Two-sided paired bootstrap hypothesis test.

    Tests whether the mean of x is significantly different from y.

    Args:
        x: First sample.
        y: Second sample (paired with x).
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed.

    Returns:
        Tuple of (observed_difference, p_value).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length for paired test")

    n = len(x)
    diff = x - y
    observed_diff = np.mean(diff)

    rng = np.random.default_rng(random_state)

    # Bootstrap under null hypothesis (mean diff = 0)
    centered_diff = diff - np.mean(diff)
    count = 0

    for _ in range(n_bootstrap):
        sample = rng.choice(centered_diff, size=n, replace=True)
        if abs(np.mean(sample)) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_bootstrap + 1)

    return (float(observed_diff), float(p_value))
