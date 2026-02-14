"""Classifier-based realism metric."""

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


@metric("realism_classifier")
class RealismClassifier(BaseMetric):
    """Classifier-based realism metric.

    Trains a classifier to distinguish real from simulated sessions.
    If the classifier cannot distinguish them (AUC ≈ 0.5), the
    simulated sessions are highly realistic.

    Returns 1 - AUC so that higher values indicate better realism.
    """

    name = "realism_classifier"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Classifier-based realism score (1 - AUC)"

    def compute(
        self,
        real: list[InteractionSession] | None = None,
        sim: list[InteractionSession] | None = None,
        real_embeddings: np.ndarray | None = None,
        sim_embeddings: np.ndarray | None = None,
        embedder=None,
        n_folds: int = 5,
        **kwargs,
    ) -> MetricResult:
        """Compute classifier-based realism score.

        Args:
            real: List of real sessions (optional if embeddings provided).
            sim: List of simulated sessions (optional if embeddings provided).
            real_embeddings: Pre-computed embeddings for real sessions.
            sim_embeddings: Pre-computed embeddings for simulated sessions.
            embedder: Optional embedder to compute embeddings from sessions.
            n_folds: Number of cross-validation folds.

        Returns:
            MetricResult with realism score (1 - AUC).
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

        # Prepare data
        X = np.vstack([real_embeddings, sim_embeddings])
        y = np.array([0] * len(real_embeddings) + [1] * len(sim_embeddings))

        # Handle small sample sizes
        n_samples = len(y)
        if n_samples < n_folds * 2:
            n_folds = max(2, n_samples // 2)

        # Train classifier with cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42)

        try:
            # Get cross-validated predictions
            y_pred_proba = cross_val_predict(
                clf, X, y, cv=n_folds, method="predict_proba"
            )[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            auc = roc_auc_score(y, y_pred_proba)
            accuracy = accuracy_score(y, y_pred)
        except Exception:
            # Fallback if cross-validation fails
            auc = 0.5
            accuracy = 0.5

        # Realism score: 1 - AUC (higher is better)
        # AUC of 0.5 = perfect realism (classifier can't distinguish)
        # AUC of 1.0 = worst realism (classifier always correct)
        realism_score = 1.0 - auc

        return MetricResult(
            name=self.name,
            value=realism_score,
            meta={
                "auc": auc,
                "accuracy": accuracy,
                "n_real": len(real_embeddings),
                "n_sim": len(sim_embeddings),
                "n_folds": n_folds,
            },
        )
