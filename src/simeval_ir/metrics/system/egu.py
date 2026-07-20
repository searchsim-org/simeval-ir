"""Expected Global Utility (EGU) metric."""

from typing import Literal

import numpy as np

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import EventType, MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


@metric("egu")
class EGU(BaseMetric):
    """Expected Global Utility metric.

    EGU models the expected utility a user gains from a search session,
    accounting for the probability of continuing the session and
    the utility gained from each interaction.
    """

    name = "egu"
    objective: Literal["evaluation"] = "evaluation"
    granularity: Literal["session"] = "session"
    scenarios = {"T"}
    description = "Expected Global Utility for search sessions"

    def __init__(
        self,
        max_gain: float = 1.0,
        continue_prob: float = 0.8,
    ):
        """Initialize EGU metric.

        Args:
            max_gain: Maximum utility gain per relevant document.
            continue_prob: Probability of continuing after each action.
        """
        self.max_gain = max_gain
        self.continue_prob = continue_prob

    def compute(
        self,
        sessions: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        """Compute EGU for sessions.

        Args:
            sessions: List of sessions with relevance judgments.

        Returns:
            MetricResult with mean EGU.
        """
        self.validate_sessions(sessions)

        session_scores: dict[str, float] = {}

        for session in sessions:
            utility = 0.0
            prob_reaching = 1.0
            actions_seen = 0

            for event in session.events:
                # Decay probability of reaching this point
                if actions_seen > 0:
                    prob_reaching *= self.continue_prob

                if event.type == EventType.CLICK and event.clicked_items:
                    for click in event.clicked_items:
                        # Get relevance from meta if available
                        rel = click.meta.get("relevance", 0)
                        if rel > 0:
                            gain = self.max_gain * (rel / 3.0)  # Normalize
                            utility += prob_reaching * gain

                elif event.type == EventType.SERP_VIEW and event.ranked_items:
                    for item in event.ranked_items:
                        if item.judged_relevance and item.judged_relevance > 0:
                            # Assume click probability decreases with rank
                            click_prob = 1.0 / (1 + item.rank)
                            gain = self.max_gain * (item.judged_relevance / 3.0)
                            utility += prob_reaching * click_prob * gain

                actions_seen += 1

            session_scores[session.session_id] = utility

        if not session_scores:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "No sessions processed"},
            )

        mean_score = float(np.mean(list(session_scores.values())))

        return MetricResult(
            name=self.name,
            value=mean_score,
            per_item=session_scores,
            meta={
                "max_gain": self.max_gain,
                "continue_prob": self.continue_prob,
                "n_sessions": len(session_scores),
            },
        )
