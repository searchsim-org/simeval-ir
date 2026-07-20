"""Action-sequence edit distance (normalised Levenshtein) between
matched real/simulated sessions.

Each session is encoded as a string of action types (one symbol per
event). For each real session, we find the simulated session whose
event count is closest and compute the Levenshtein edit distance
between the two action strings, normalised by the maximum length. The
reported value is the mean normalised distance across all real
sessions. Lower is better.

Relies only on the standard library; we implement the textbook
two-row dynamic programme so we don't pull in a new dependency.
"""

from __future__ import annotations

import bisect
from typing import Literal

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


def _action_string(s: InteractionSession) -> str:
    # Use first character of each event-type value as the alphabet.
    # Different EventType values start with different characters in our
    # schema (q, s, c, d, u, y), so this is unambiguous and keeps the
    # strings short.
    return "".join(e.type.value[0] for e in s.events)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = cur
    return prev[-1]


@metric("action_sequence_edit")
class ActionSequenceEdit(BaseMetric):
    """Mean normalised Levenshtein distance over action strings.

    For each real session, the closest-by-length simulated session is
    paired with it and the normalised edit distance is recorded. The
    pairing is greedy by length, not optimal-transport: the cost of an
    optimal pairing is dominated by the choice of feature, not the
    matching scheme, on the corpora we ship with.
    """

    name = "action_sequence_edit"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Mean normalised Levenshtein on action-type sequences"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        **kwargs,
    ) -> MetricResult:
        self.validate_sessions(real)
        self.validate_sessions(sim)
        real_strs = [_action_string(s) for s in real]
        sim_strs = [_action_string(s) for s in sim]
        if not real_strs or not sim_strs:
            return MetricResult(name=self.name, value=0.0,
                                meta={"warning": "empty input"})

        # Sort sim by length to do O(log n) closest-length lookups.
        sim_by_len = sorted(sim_strs, key=len)
        sim_lengths = [len(s) for s in sim_by_len]
        n = 0
        total = 0.0
        for r in real_strs:
            idx = bisect.bisect_left(sim_lengths, len(r))
            cands = []
            if idx < len(sim_by_len):
                cands.append(sim_by_len[idx])
            if idx > 0:
                cands.append(sim_by_len[idx - 1])
            if not cands:
                continue
            best = min(_levenshtein(r, c) for c in cands)
            denom = max(len(r), max(len(c) for c in cands), 1)
            total += best / denom
            n += 1
        value = total / n if n else 0.0
        return MetricResult(
            name=self.name, value=value,
            meta={"n_pairs": n, "n_real": len(real), "n_sim": len(sim)},
        )
