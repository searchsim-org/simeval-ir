"""Kolmogorov--Smirnov two-sample tests on session-level distributions.

Two metrics share the same shape; only the per-session feature changes:
``ks_session_length`` and ``ks_click_depth``. They report the supremum
distance between the empirical CDFs together with the two-sided p-value
returned by :func:`scipy.stats.ks_2samp`. KS is one of the realism
metrics most commonly cited in click-model evaluation work and is a
strict marginal-distribution check.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import EventType, MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


def _session_lengths(sessions: list[InteractionSession]) -> list[int]:
    return [len(s.events) for s in sessions]


def _click_depths(sessions: list[InteractionSession]) -> list[int]:
    """Per-click rank, flattened across sessions; sessions with no clicks
    contribute nothing (an empty-click session has no click-depth)."""
    out: list[int] = []
    for s in sessions:
        for ev in s.events:
            if ev.type == EventType.CLICK and ev.clicked_items:
                for c in ev.clicked_items:
                    if c.rank is not None:
                        out.append(int(c.rank))
    return out


def _ks(real_x: list[int | float], sim_x: list[int | float]) -> tuple[float, float]:
    if not real_x or not sim_x:
        return float("nan"), float("nan")
    res = stats.ks_2samp(np.asarray(real_x), np.asarray(sim_x))
    return float(res.statistic), float(res.pvalue)


@metric("ks_session_length")
class KSSessionLength(BaseMetric):
    """Two-sample KS statistic on session-length distributions."""

    name = "ks_session_length"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Kolmogorov--Smirnov distance on session length"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        self.validate_sessions(real)
        self.validate_sessions(sim)
        d, p = _ks(_session_lengths(real), _session_lengths(sim))
        return MetricResult(
            name=self.name, value=d,
            meta={"p_value": p, "n_real": len(real), "n_sim": len(sim)},
        )


@metric("ks_click_depth")
class KSClickDepth(BaseMetric):
    """Two-sample KS statistic on per-click rank (click-depth) distribution."""

    name = "ks_click_depth"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T"}
    description = "Kolmogorov--Smirnov distance on click depth"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        self.validate_sessions(real)
        self.validate_sessions(sim)
        real_d = _click_depths(real)
        sim_d = _click_depths(sim)
        d, p = _ks(real_d, sim_d)
        return MetricResult(
            name=self.name, value=d,
            meta={"p_value": p,
                  "n_real_clicks": len(real_d),
                  "n_sim_clicks": len(sim_d)},
        )
