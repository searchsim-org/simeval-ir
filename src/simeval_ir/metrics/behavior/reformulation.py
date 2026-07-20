"""Reformulation similarity metrics for SimEval-IR.

Measures how similar query reformulation patterns are between real and simulated sessions.
This metric captures sequential query dynamics that marginal distributions miss.
"""

from collections import Counter
from typing import Literal

import numpy as np
from scipy.spatial.distance import jensenshannon

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import EventType, MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


def _extract_query_pairs(sessions: list[InteractionSession]) -> list[tuple[str, str]]:
    """Extract consecutive query pairs from sessions."""
    pairs = []
    for session in sessions:
        queries = [
            e.query for e in session.events
            if e.type == EventType.QUERY_ISSUED and e.query
        ]
        for i in range(len(queries) - 1):
            pairs.append((queries[i], queries[i + 1]))
    return pairs


def _classify_reformulation(q1: str, q2: str) -> str:
    """Classify the type of reformulation between two queries.

    Returns one of:
    - 'specialization': q2 adds terms to q1
    - 'generalization': q2 removes terms from q1
    - 'replacement': q2 replaces terms in q1
    - 'repetition': q1 and q2 are identical
    - 'new': q2 shares no terms with q1
    """
    tokens1 = set(q1.lower().split())
    tokens2 = set(q2.lower().split())

    if tokens1 == tokens2:
        return "repetition"

    overlap = tokens1 & tokens2
    if not overlap:
        return "new"

    added = tokens2 - tokens1
    removed = tokens1 - tokens2

    if added and not removed:
        return "specialization"
    elif removed and not added:
        return "generalization"
    else:
        return "replacement"


def _compute_reformulation_distribution(
    pairs: list[tuple[str, str]]
) -> dict[str, float]:
    """Compute distribution over reformulation types."""
    if not pairs:
        return {}

    counts = Counter(_classify_reformulation(q1, q2) for q1, q2 in pairs)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


@metric("reformulation_similarity")
class ReformulationSimilarity(BaseMetric):
    """Reformulation pattern similarity between real and simulated sessions.

    Computes Jensen-Shannon Divergence between the distribution of
    reformulation types (specialization, generalization, replacement, etc.)
    in real vs simulated sessions.

    Lower values indicate more similar reformulation patterns.
    """

    name = "reformulation_similarity"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T"}  # Only for traditional search
    description = "JSD between reformulation type distributions"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute reformulation similarity.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.

        Returns:
            MetricResult with JSD value (lower = more similar).
        """
        # Extract query pairs
        real_pairs = _extract_query_pairs(real)
        sim_pairs = _extract_query_pairs(sim)

        if not real_pairs or not sim_pairs:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={
                    "warning": "Insufficient query pairs for comparison",
                    "n_real_pairs": len(real_pairs),
                    "n_sim_pairs": len(sim_pairs),
                },
            )

        # Compute distributions
        real_dist = _compute_reformulation_distribution(real_pairs)
        sim_dist = _compute_reformulation_distribution(sim_pairs)

        # Get all reformulation types
        all_types = sorted(
            {"specialization", "generalization", "replacement", "repetition", "new"}
        )

        # Create probability vectors
        eps = 1e-10
        p = np.array([real_dist.get(t, 0) + eps for t in all_types])
        q = np.array([sim_dist.get(t, 0) + eps for t in all_types])
        p = p / p.sum()
        q = q / q.sum()

        # Compute JSD
        jsd = float(jensenshannon(p, q))

        return MetricResult(
            name=self.name,
            value=jsd,
            meta={
                "reformulation_types": all_types,
                "real_distribution": real_dist,
                "sim_distribution": sim_dist,
                "n_real_pairs": len(real_pairs),
                "n_sim_pairs": len(sim_pairs),
            },
        )


@metric("jsd_click_depth")
class JSDClickDepth(BaseMetric):
    """Jensen-Shannon Divergence between click depth distributions.

    Click depth is the rank/position of clicked documents.
    This metric compares the distribution of click positions.

    Lower values indicate more similar click patterns.
    """

    name = "jsd_click_depth"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T"}  # Only for traditional search
    description = "Jensen-Shannon divergence between click depth distributions"

    def __init__(self, max_depth: int = 20):
        """Initialize JSD Click Depth metric.

        Args:
            max_depth: Maximum depth to consider (positions beyond this are grouped).
        """
        self.max_depth = max_depth

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute JSD between click depth distributions.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.

        Returns:
            MetricResult with JSD value.
        """
        real_depths = self._extract_click_depths(real)
        sim_depths = self._extract_click_depths(sim)

        if not real_depths or not sim_depths:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={
                    "warning": "No clicks found",
                    "n_real_clicks": len(real_depths),
                    "n_sim_clicks": len(sim_depths),
                },
            )

        # Create histograms
        bins = list(range(1, self.max_depth + 2))  # 1 to max_depth+1
        real_hist, _ = np.histogram(
            [min(d, self.max_depth) for d in real_depths],
            bins=bins,
            density=True
        )
        sim_hist, _ = np.histogram(
            [min(d, self.max_depth) for d in sim_depths],
            bins=bins,
            density=True
        )

        # Add smoothing
        eps = 1e-10
        real_hist = real_hist + eps
        sim_hist = sim_hist + eps
        real_hist = real_hist / real_hist.sum()
        sim_hist = sim_hist / sim_hist.sum()

        # Compute JSD
        jsd = float(jensenshannon(real_hist, sim_hist))

        return MetricResult(
            name=self.name,
            value=jsd,
            meta={
                "max_depth": self.max_depth,
                "n_real_clicks": len(real_depths),
                "n_sim_clicks": len(sim_depths),
                "mean_real_depth": float(np.mean(real_depths)),
                "mean_sim_depth": float(np.mean(sim_depths)),
            },
        )

    def _extract_click_depths(
        self,
        sessions: list[InteractionSession],
    ) -> list[int]:
        """Extract click depths (ranks) from sessions."""
        depths = []
        for session in sessions:
            for event in session.events:
                if event.clicked_items:
                    for click in event.clicked_items:
                        if click.rank > 0:
                            depths.append(click.rank)
        return depths
