"""Fréchet Distance metric for session embeddings."""

from typing import Literal

import numpy as np
from scipy import linalg

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet Distance between two multivariate Gaussians.

    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

    Args:
        mu1: Mean of first distribution.
        sigma1: Covariance of first distribution.
        mu2: Mean of second distribution.
        sigma2: Covariance of second distribution.
        eps: Small value for numerical stability.

    Returns:
        Fréchet Distance value.
    """
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical errors might give complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Check for nans
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    tr_covmean = np.trace(covmean)

    fd = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(fd, 0.0)  # Ensure non-negative due to numerical errors


@metric("session_fd")
class SessionFD(BaseMetric):
    """Fréchet Distance between session embedding distributions.

    Computes FD between embeddings of real and simulated sessions.
    Lower values indicate more similar session distributions.

    Requires session embeddings to be computed first via an embedder.
    """

    name = "session_fd"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Fréchet Distance between session embedding distributions"

    def compute(
        self,
        real: list[InteractionSession] | None = None,
        sim: list[InteractionSession] | None = None,
        real_embeddings: np.ndarray | None = None,
        sim_embeddings: np.ndarray | None = None,
        embedder=None,
        **kwargs,
    ) -> MetricResult:
        """Compute Fréchet Distance between session distributions.

        Args:
            real: List of real sessions (optional if embeddings provided).
            sim: List of simulated sessions (optional if embeddings provided).
            real_embeddings: Pre-computed embeddings for real sessions.
            sim_embeddings: Pre-computed embeddings for simulated sessions.
            embedder: Optional embedder to compute embeddings from sessions.

        Returns:
            MetricResult with FD value.
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

        # Compute statistics
        mu_real = np.mean(real_embeddings, axis=0)
        sigma_real = np.cov(real_embeddings, rowvar=False)

        mu_sim = np.mean(sim_embeddings, axis=0)
        sigma_sim = np.cov(sim_embeddings, rowvar=False)

        # Handle edge case of 1D covariance
        if sigma_real.ndim == 0:
            sigma_real = sigma_real.reshape(1, 1)
            sigma_sim = sigma_sim.reshape(1, 1)

        # Compute FD
        fd = compute_frechet_distance(mu_real, sigma_real, mu_sim, sigma_sim)

        return MetricResult(
            name=self.name,
            value=fd,
            meta={
                "embedding_dim": real_embeddings.shape[1],
                "n_real": len(real_embeddings),
                "n_sim": len(sim_embeddings),
            },
        )
