"""Session-level nDCG metrics."""

from typing import Literal

import numpy as np

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import EventType, MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute DCG@k.

    Args:
        relevances: List of relevance scores in rank order.
        k: Cutoff position.

    Returns:
        DCG@k value.
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = np.array(relevances)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Compute nDCG@k.

    Args:
        relevances: List of relevance scores in rank order.
        k: Cutoff position.

    Returns:
        nDCG@k value (0-1).
    """
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


@metric("session_ndcg")
class SessionNDCG(BaseMetric):
    """Session-level nDCG metric.

    Computes nDCG across all queries in a session, aggregating
    relevance judgments from ranked items.
    """

    name = "session_ndcg"
    objective: Literal["evaluation"] = "evaluation"
    granularity: Literal["session"] = "session"
    scenarios = {"T"}
    description = "Session-level nDCG aggregated across queries"

    def __init__(self, k: int = 10):
        """Initialize with cutoff.

        Args:
            k: Rank cutoff for nDCG computation.
        """
        self.k = k

    def compute(
        self,
        sessions: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute session-level nDCG.

        Args:
            sessions: List of sessions with relevance judgments.

        Returns:
            MetricResult with mean session nDCG.
        """
        self.validate_sessions(sessions)

        session_scores: dict[str, float] = {}

        for session in sessions:
            query_ndcgs = []

            for event in session.events:
                if event.type != EventType.SERP_VIEW:
                    continue
                if not event.ranked_items:
                    continue

                # Get relevances in rank order
                items = sorted(event.ranked_items, key=lambda x: x.rank)
                relevances = [
                    float(item.judged_relevance or 0) for item in items
                ]

                if relevances:
                    ndcg = ndcg_at_k(relevances, self.k)
                    query_ndcgs.append(ndcg)

            if query_ndcgs:
                session_scores[session.session_id] = float(np.mean(query_ndcgs))

        if not session_scores:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "No relevance judgments found"},
            )

        mean_score = float(np.mean(list(session_scores.values())))

        return MetricResult(
            name=self.name,
            value=mean_score,
            per_item=session_scores,
            meta={"k": self.k, "n_sessions": len(session_scores)},
        )
