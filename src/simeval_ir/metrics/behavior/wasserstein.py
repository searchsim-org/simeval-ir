"""Wasserstein distance metric for behavioral realism."""

from typing import Literal

import numpy as np
from scipy import stats

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


@metric("wasserstein_session_length")
class WassersteinSessionLength(BaseMetric):
    """Wasserstein distance between session length distributions.

    The Wasserstein distance (Earth Mover's Distance) measures the minimum
    cost of transforming one distribution into another. Lower values indicate
    more similar distributions.
    """

    name = "wasserstein_session_length"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Wasserstein distance for session length distributions"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute Wasserstein distance between session length distributions.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.

        Returns:
            MetricResult with Wasserstein distance (lower = more similar).
        """
        real_lengths = np.array([len(s) for s in real], dtype=float)
        sim_lengths = np.array([len(s) for s in sim], dtype=float)

        if len(real_lengths) < 2 or len(sim_lengths) < 2:
            return MetricResult(
                name=self.name,
                value=float("nan"),
                meta={"warning": "Insufficient data"},
            )

        # Compute 1D Wasserstein distance (Earth Mover's Distance)
        distance = stats.wasserstein_distance(real_lengths, sim_lengths)

        return MetricResult(
            name=self.name,
            value=float(distance),
            meta={
                "mean_real": float(np.mean(real_lengths)),
                "mean_sim": float(np.mean(sim_lengths)),
                "std_real": float(np.std(real_lengths)),
                "std_sim": float(np.std(sim_lengths)),
                "n_real": len(real_lengths),
                "n_sim": len(sim_lengths),
            },
        )


@metric("wasserstein_click_depth")
class WassersteinClickDepth(BaseMetric):
    """Wasserstein distance between click depth distributions.

    Compares the distribution of click ranks between real and simulated sessions.
    """

    name = "wasserstein_click_depth"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T"}
    description = "Wasserstein distance for click depth distributions"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute Wasserstein distance between click depth distributions.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.

        Returns:
            MetricResult with Wasserstein distance.
        """
        real_depths = self._extract_click_depths(real)
        sim_depths = self._extract_click_depths(sim)

        if len(real_depths) < 2 or len(sim_depths) < 2:
            return MetricResult(
                name=self.name,
                value=float("nan"),
                meta={"warning": "Insufficient click data"},
            )

        distance = stats.wasserstein_distance(real_depths, sim_depths)

        return MetricResult(
            name=self.name,
            value=float(distance),
            meta={
                "mean_real": float(np.mean(real_depths)),
                "mean_sim": float(np.mean(sim_depths)),
                "n_real": len(real_depths),
                "n_sim": len(sim_depths),
            },
        )

    def _extract_click_depths(
        self,
        sessions: list[InteractionSession],
    ) -> np.ndarray:
        """Extract click depths (ranks) from sessions."""
        depths = []
        for session in sessions:
            for click in session.get_clicks():
                if click.rank is not None:
                    depths.append(click.rank)
        return np.array(depths, dtype=float) if depths else np.array([])
