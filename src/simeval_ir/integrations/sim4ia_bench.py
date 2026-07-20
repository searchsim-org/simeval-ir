"""Sim4IA-Bench (SIGIR 2025 micro-shared task) integration.

Loads the Sim4IA-Bench Task-A and Task-B release files into the
canonical SimEval-IR session schema. Sim4IA-Bench is the dockerised
SimIIR-3 wrapper distributed with the Sim4IA workshop; its Task-A
predetermined-query CSV ships one (session, query, SERP, click) row per
interaction. We reconstruct sessions by grouping rows by (user, topic)
and sorting by timestamp.

Typical use::

    from simeval_ir.integrations.sim4ia_bench import load_task_a
    sessions = list(load_task_a("simiir/predetermined_queries_Task_A.csv"))
"""

from __future__ import annotations

import ast
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from simeval_ir.core.session import (
    Click,
    Event,
    InteractionSession,
    RankedItem,
)
from simeval_ir.core.types import EventType, Role, SessionType


# --------------------------------------------------------------------- #
# Task-A: per-row CSV with (session_id, user_id, topic_id, query, ts,
#         serp[json list], session_hash, clicks[python literal of tuples])
# --------------------------------------------------------------------- #

def _parse_serp(s: str) -> list[str]:
    s = (s or "").strip()
    try:
        return [str(x) for x in ast.literal_eval(s)]
    except Exception:
        return []


def _parse_clicks(s: str):
    s = (s or "[]").strip()
    try:
        ns = {"datetime": datetime}
        return list(eval(s, ns, ns))  # noqa: S307 -- input is local CSV
    except Exception:
        return []


def load_task_a(csv_path: Path | str) -> Iterator[InteractionSession]:
    """Yield :class:`InteractionSession` objects from a Task-A CSV file."""
    csv_path = Path(csv_path)
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 8:
                continue
            try:
                rows.append({
                    "row_id": int(r[0]),
                    "user_id": r[1],
                    "topic_id": r[2],
                    "query": r[3],
                    "timestamp": datetime.fromisoformat(r[4]),
                    "serp": _parse_serp(r[5]),
                    "hash": r[6],
                    "clicks": _parse_clicks(r[7]),
                })
            except Exception:
                continue

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["user_id"], r["topic_id"]), []).append(r)

    for (user, topic), interactions in grouped.items():
        interactions.sort(key=lambda x: x["timestamp"])
        events: list[Event] = []
        eid = 0
        if not interactions:
            continue
        base_t = interactions[0]["timestamp"]
        for it in interactions:
            ts = (it["timestamp"] - base_t).total_seconds()
            eid += 1
            events.append(Event(
                event_id=f"sim4ia-{user}-{topic}-q{eid}",
                type=EventType.QUERY_ISSUED,
                timestamp=ts,
                role=Role.USER,
                query=it["query"],
                meta={"row_id": it["row_id"], "session_hash": it["hash"]},
            ))
            ranked_items = [
                RankedItem(doc_id=d, rank=i + 1)
                for i, d in enumerate(it["serp"][:10])
            ]
            if ranked_items:
                eid += 1
                events.append(Event(
                    event_id=f"sim4ia-{user}-{topic}-s{eid}",
                    type=EventType.SERP_VIEW,
                    timestamp=ts + 0.5,
                    role=Role.SYSTEM,
                    ranked_items=ranked_items,
                ))
                rank_lookup = {d: i + 1 for i, d in enumerate(it["serp"][:10])}
                clicks = []
                for c in it["clicks"]:
                    if not c:
                        continue
                    doc_id = str(c[0])
                    rank = rank_lookup.get(doc_id, 0)
                    clicks.append(Click(doc_id=doc_id, rank=rank))
                if clicks:
                    eid += 1
                    events.append(Event(
                        event_id=f"sim4ia-{user}-{topic}-c{eid}",
                        type=EventType.CLICK,
                        timestamp=ts + 1.0,
                        role=Role.USER,
                        clicked_items=clicks,
                    ))
        if events:
            yield InteractionSession(
                session_id=f"sim4ia-{user}-{topic}",
                dataset_id="sim4ia-bench-task-a",
                session_type=SessionType.SEARCH,
                events=events,
                user_id=user,
                topic_id=topic,
                domain="academic-search",
                meta={"source": "Sim4IA-Bench", "task": "A"},
            )


# --------------------------------------------------------------------- #
# Task-B: JSON file mapping {topic_id: [utterance, utterance, ...]}
# --------------------------------------------------------------------- #

def load_task_b(json_path: Path | str) -> Iterator[InteractionSession]:
    """Yield conversational :class:`InteractionSession` objects for Task-B."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    for topic_id, utterances in data.items():
        events: list[Event] = []
        for turn_idx, utt in enumerate(utterances, start=1):
            events.append(Event(
                event_id=f"sim4ia-task-b-{topic_id}-t{turn_idx}",
                type=EventType.USER_UTTERANCE,
                timestamp=float(turn_idx),
                turn_id=turn_idx,
                role=Role.USER,
                utterance=utt,
                query=utt,
            ))
        if events:
            yield InteractionSession(
                session_id=f"sim4ia-task-b-{topic_id}",
                dataset_id="sim4ia-bench-task-b",
                session_type=SessionType.CONVERSATIONAL,
                events=events,
                topic_id=str(topic_id),
                domain="academic-search",
                meta={"source": "Sim4IA-Bench", "task": "B"},
            )
