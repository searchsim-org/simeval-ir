"""RATE-style tester reliability estimation."""

from typing import Literal

import numpy as np

from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


@metric("rate")
class RATE(BaseMetric):
    """RATE-style tester reliability estimation.

    Estimates the reliability of each tester based on agreement
    with other testers, weighted by their own reliability.
    """

    name = "rate"
    objective: Literal["tester"] = "tester"
    granularity: Literal["system"] = "system"
    scenarios = {"T", "C"}
    description = "RATE-style tester reliability estimation"

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """Initialize RATE.

        Args:
            max_iterations: Maximum iterations for convergence.
            tolerance: Convergence tolerance.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute(
        self,
        tester_scores: dict[str, dict[str, float]],
        **kwargs,
    ) -> MetricResult:
        """Compute RATE reliability for each tester.

        Args:
            tester_scores: Dict of {tester_id: {system_id: score}}.

        Returns:
            MetricResult with mean reliability and per-tester reliabilities.
        """
        tester_ids = list(tester_scores.keys())
        n_testers = len(tester_ids)

        if n_testers < 2:
            return MetricResult(
                name=self.name,
                value=1.0,
                meta={"warning": "Need at least 2 testers"},
            )

        # Get common systems
        all_systems = [set(tester_scores[t].keys()) for t in tester_ids]
        common_systems = set.intersection(*all_systems)

        if len(common_systems) < 2:
            return MetricResult(
                name=self.name,
                value=0.5,
                meta={"warning": "Not enough common systems"},
            )

        system_list = sorted(common_systems)

        # Build score matrix: testers x systems
        score_matrix = np.array([
            [tester_scores[t][s] for s in system_list]
            for t in tester_ids
        ])

        # Initialize reliabilities uniformly
        reliabilities = np.ones(n_testers) / n_testers

        # Iterative refinement
        for iteration in range(self.max_iterations):
            old_reliabilities = reliabilities.copy()

            # Compute weighted consensus scores
            weights = reliabilities / reliabilities.sum()
            consensus = np.average(score_matrix, axis=0, weights=weights)

            # Update reliabilities based on agreement with consensus
            for i in range(n_testers):
                corr = np.corrcoef(score_matrix[i], consensus)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                reliabilities[i] = max(0.0, corr)

            # Normalize
            if reliabilities.sum() > 0:
                reliabilities = reliabilities / reliabilities.sum()

            # Check convergence
            if np.max(np.abs(reliabilities - old_reliabilities)) < self.tolerance:
                break

        # Convert back to raw reliability scores (0-1)
        per_tester = {
            tester_ids[i]: float(reliabilities[i] * n_testers)
            for i in range(n_testers)
        }

        # Mean reliability
        mean_rel = float(np.mean(list(per_tester.values())))

        return MetricResult(
            name=self.name,
            value=mean_rel,
            per_item=per_tester,
            meta={
                "n_testers": n_testers,
                "n_systems": len(system_list),
                "iterations": iteration + 1,
            },
        )
