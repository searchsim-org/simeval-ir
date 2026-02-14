"""Maximum Mean Discrepancy (MMD) metric for behavioral realism."""

from typing import Literal

import numpy as np

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel between two sets of samples.

    Args:
        X: First set of samples (n_samples_X, n_features).
        Y: Second set of samples (n_samples_Y, n_features).
        gamma: Kernel bandwidth parameter.

    Returns:
        Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    # Compute pairwise squared distances
    X_sqnorms = np.sum(X ** 2, axis=1)
    Y_sqnorms = np.sum(Y ** 2, axis=1)
    dists_sq = X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :] - 2 * X @ Y.T
    return np.exp(-gamma * dists_sq)


def compute_mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Compute squared Maximum Mean Discrepancy between two samples.

    Uses an unbiased estimator with RBF kernel.

    Args:
        X: Samples from first distribution (n_X, d).
        Y: Samples from second distribution (n_Y, d).
        gamma: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        Squared MMD value.
    """
    n_X = len(X)
    n_Y = len(Y)

    if n_X < 2 or n_Y < 2:
        return float("nan")

    # Median heuristic for bandwidth selection
    if gamma is None:
        combined = np.vstack([X, Y])
        pairwise_dists = np.sum(
            (combined[:, np.newaxis] - combined[np.newaxis, :]) ** 2,
            axis=2
        )
        median_dist = np.median(pairwise_dists[pairwise_dists > 0])
        gamma = 1.0 / median_dist if median_dist > 0 else 1.0

    # Compute kernel matrices
    K_XX = _rbf_kernel(X, X, gamma)
    K_YY = _rbf_kernel(Y, Y, gamma)
    K_XY = _rbf_kernel(X, Y, gamma)

    # Unbiased MMD^2 estimator
    # E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
    # with diagonal elements removed for unbiased estimation

    # Remove diagonal for unbiased estimation
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd_sq = (
        np.sum(K_XX) / (n_X * (n_X - 1))
        - 2 * np.sum(K_XY) / (n_X * n_Y)
        + np.sum(K_YY) / (n_Y * (n_Y - 1))
    )

    return float(max(0, mmd_sq))  # Clamp to non-negative


@metric("mmd")
class MMD(BaseMetric):
    """Maximum Mean Discrepancy between session embedding distributions.

    MMD is a kernel-based distance between distributions that captures
    higher-order moments. Lower values indicate more similar distributions.

    Requires either pre-computed embeddings or an embedder to compute them.
    """

    name = "mmd"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Maximum Mean Discrepancy on session embeddings"

    def compute(
        self,
        real: list[InteractionSession] | None = None,
        sim: list[InteractionSession] | None = None,
        real_embeddings: np.ndarray | None = None,
        sim_embeddings: np.ndarray | None = None,
        embedder=None,
        gamma: float | None = None,
        **kwargs,
    ) -> MetricResult:
        """Compute MMD between real and simulated session distributions.

        Args:
            real: List of real sessions (optional if embeddings provided).
            sim: List of simulated sessions (optional if embeddings provided).
            real_embeddings: Pre-computed embeddings for real sessions.
            sim_embeddings: Pre-computed embeddings for simulated sessions.
            embedder: Optional embedder to compute embeddings from sessions.
            gamma: RBF kernel bandwidth. If None, uses median heuristic.

        Returns:
            MetricResult with MMD value (lower = more similar).
        """
        # Get embeddings
        if real_embeddings is None or sim_embeddings is None:
            if embedder is None:
                raise ValueError(
                    "Either provide embeddings directly or provide sessions + embedder."
                )
            if real is None or sim is None:
                raise ValueError("Sessions required when embedder is provided.")

            real_embeddings = embedder.embed_sessions(real)
            sim_embeddings = embedder.embed_sessions(sim)

        # Compute MMD
        mmd_squared = compute_mmd_squared(real_embeddings, sim_embeddings, gamma)
        mmd = np.sqrt(mmd_squared) if not np.isnan(mmd_squared) else float("nan")

        return MetricResult(
            name=self.name,
            value=mmd,
            meta={
                "mmd_squared": mmd_squared,
                "n_real": len(real_embeddings),
                "n_sim": len(sim_embeddings),
                "embedding_dim": real_embeddings.shape[1] if len(real_embeddings) > 0 else 0,
                "gamma": gamma,
            },
        )
