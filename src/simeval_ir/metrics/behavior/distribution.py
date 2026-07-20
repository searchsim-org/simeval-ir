"""Distribution comparison metrics."""

from typing import Literal

import numpy as np
from scipy import stats

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


@metric("timing_distribution")
class TimingDistribution(BaseMetric):
    """Compare timing distributions between real and simulated sessions.

    Uses Kolmogorov-Smirnov test to compare inter-event time distributions.
    """

    name = "timing_distribution"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["turn"] = "turn"
    scenarios = {"T", "C"}
    description = "KS statistic for inter-event timing distributions"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute KS statistic between timing distributions.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.

        Returns:
            MetricResult with KS statistic (lower = more similar).
        """
        real_times = self._extract_inter_event_times(real)
        sim_times = self._extract_inter_event_times(sim)

        if len(real_times) < 2 or len(sim_times) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Insufficient timing data"},
            )

        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(real_times, sim_times)

        return MetricResult(
            name=self.name,
            value=float(ks_stat),
            meta={
                "p_value": float(p_value),
                "n_real": len(real_times),
                "n_sim": len(sim_times),
                "mean_real": float(np.mean(real_times)),
                "mean_sim": float(np.mean(sim_times)),
            },
        )

    def _extract_inter_event_times(
        self,
        sessions: list[InteractionSession],
    ) -> np.ndarray:
        """Extract inter-event times from sessions."""
        times = []
        for session in sessions:
            timestamps = [
                e.timestamp
                for e in session.events
                if e.timestamp is not None
            ]
            timestamps = sorted(timestamps)
            for i in range(1, len(timestamps)):
                delta = timestamps[i] - timestamps[i - 1]
                if delta > 0:
                    times.append(delta)
        return np.array(times)


@metric("session_length_distribution")
class SessionLengthDistribution(BaseMetric):
    """Compare session length distributions.

    Uses Kolmogorov-Smirnov test to compare the distribution of
    session lengths (number of events).
    """

    name = "session_length_distribution"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "KS statistic for session length distributions"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute KS statistic between session length distributions.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.

        Returns:
            MetricResult with KS statistic.
        """
        real_lengths = np.array([len(s) for s in real])
        sim_lengths = np.array([len(s) for s in sim])

        if len(real_lengths) < 2 or len(sim_lengths) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Insufficient data"},
            )

        ks_stat, p_value = stats.ks_2samp(real_lengths, sim_lengths)

        return MetricResult(
            name=self.name,
            value=float(ks_stat),
            meta={
                "p_value": float(p_value),
                "mean_real": float(np.mean(real_lengths)),
                "mean_sim": float(np.mean(sim_lengths)),
                "std_real": float(np.std(real_lengths)),
                "std_sim": float(np.std(sim_lengths)),
            },
        )
