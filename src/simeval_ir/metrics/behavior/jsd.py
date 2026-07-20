"""Jensen-Shannon Divergence metric for action type distributions."""

from collections import Counter
from typing import Literal

import numpy as np
from scipy.spatial.distance import jensenshannon

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


@metric("jsd_action_types")
class JSDActionTypes(BaseMetric):
    """Jensen-Shannon Divergence between action type distributions.

    Computes JSD between the distribution of event types in real
    vs simulated sessions. Lower values indicate more similar
    action patterns.

    Range: [0, 1] where 0 = identical distributions, 1 = maximally different.
    """

    name = "jsd_action_types"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Jensen-Shannon divergence between action type distributions"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute JSD between action type distributions.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.

        Returns:
            MetricResult with JSD value (0-1).
        """
        self.validate_sessions(real)
        self.validate_sessions(sim)

        # Count action types
        real_counts = self._count_action_types(real)
        sim_counts = self._count_action_types(sim)

        # Get all action types
        all_types = sorted(set(real_counts.keys()) | set(sim_counts.keys()))

        if not all_types:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "No action types found"},
            )

        # Create probability distributions
        total_real = sum(real_counts.values())
        total_sim = sum(sim_counts.values())

        p = np.array([real_counts.get(t, 0) / total_real for t in all_types])
        q = np.array([sim_counts.get(t, 0) / total_sim for t in all_types])

        # Add small epsilon to avoid zero probabilities
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()

        # Compute JSD
        jsd = jensenshannon(p, q)

        return MetricResult(
            name=self.name,
            value=float(jsd),
            meta={
                "action_types": all_types,
                "real_distribution": {t: real_counts.get(t, 0) for t in all_types},
                "sim_distribution": {t: sim_counts.get(t, 0) for t in all_types},
            },
        )

    def _count_action_types(
        self,
        sessions: list[InteractionSession],
    ) -> dict[str, int]:
        """Count action types across sessions."""
        counts: Counter[str] = Counter()
        for session in sessions:
            for event in session.events:
                counts[event.type.value] += 1
        return dict(counts)
