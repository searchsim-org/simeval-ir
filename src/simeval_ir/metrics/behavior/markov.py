"""Markov-chain transition divergence between real and simulated sessions.

Each session is treated as a string of action types (the unigram view is
already covered by ``jsd_action_types``). This metric goes one step
further and compares the bigram transition matrices: how often does a
``QUERY`` follow a ``CLICK``, a ``CLICK`` follow a ``SERP_VIEW``, and so
on. We add a single absorbing ``$END`` symbol so that the trailing
transition out of the last event is well defined.

The reported value is the Jensen--Shannon divergence between the two
flattened transition matrices (rows are conditional distributions
``P(next | current)``). Lower is better, the same convention as the
other behaviour distances.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

import numpy as np
from scipy.spatial.distance import jensenshannon

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


_END = "$END"


def _transition_matrix(
    sessions: list[InteractionSession],
) -> tuple[dict[str, dict[str, int]], set[str]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    states: set[str] = set()
    for s in sessions:
        types = [e.type.value for e in s.events]
        if not types:
            continue
        for cur, nxt in zip(types, types[1:] + [_END]):
            counts[cur][nxt] += 1
            states.add(cur)
            states.add(nxt)
    return counts, states


@metric("markov_transition_jsd")
class MarkovTransitionJSD(BaseMetric):
    """JS divergence between action-bigram transition matrices."""

    name = "markov_transition_jsd"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Jensen--Shannon divergence on action-bigram transition matrices"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        self.validate_sessions(real)
        self.validate_sessions(sim)
        real_t, real_states = _transition_matrix(real)
        sim_t, sim_states = _transition_matrix(sim)
        states = sorted(real_states | sim_states)
        if len(states) <= 1:
            return MetricResult(name=self.name, value=0.0,
                                meta={"warning": "no transitions"})

        eps = 1e-10
        # Flatten conditional distributions row by row, then compute one
        # combined JSD over the full transition matrix (a row that is empty
        # in both real and sim contributes nothing).
        p_vec, q_vec = [], []
        for cur in states:
            r_row = real_t.get(cur, Counter())
            s_row = sim_t.get(cur, Counter())
            r_total = sum(r_row.values()) or 0
            s_total = sum(s_row.values()) or 0
            if r_total == 0 and s_total == 0:
                continue
            for nxt in states:
                rp = (r_row.get(nxt, 0) + eps) / (r_total + eps * len(states))
                sp = (s_row.get(nxt, 0) + eps) / (s_total + eps * len(states))
                p_vec.append(rp)
                q_vec.append(sp)
        p = np.array(p_vec)
        q = np.array(q_vec)
        p = p / p.sum()
        q = q / q.sum()
        jsd = float(jensenshannon(p, q))

        return MetricResult(
            name=self.name, value=jsd,
            meta={"states": states,
                  "n_real_sessions": len(real),
                  "n_sim_sessions": len(sim)},
        )
