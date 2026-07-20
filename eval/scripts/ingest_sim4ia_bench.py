#!/usr/bin/env python
"""Ingest Sim4IA-Bench Task A evaluation sessions into the SimEval-IR
canonical schema.

Source: ``simiir/evaluation_sessions_Task_A.csv`` from the Sim4IA-Bench
SIGIR 2025 micro-shared task release. Each row is a single (session, query,
serp, clicks) record:

    session_id, user_id, topic_id, query, timestamp, serp(json list of doc_ids),
    session_hash, clicks(json list of (doc_id, datetime, click_type))

Sessions are reconstructed by grouping rows by session_id (and ordering by
timestamp). Click rank is recovered by looking up the clicked doc in the
SERP. The output is a JSONL file in the canonical SimEval-IR schema.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from simeval_ir.core.session import (  # noqa: E402
    InteractionSession, Event, Click, RankedItem,
)
from simeval_ir.core.types import EventType, Role, SessionType  # noqa: E402


def _parse_serp(s: str) -> list[str]:
    s = s.strip()
    try:
        return [str(x) for x in ast.literal_eval(s)]
    except Exception:
        return []


def _parse_clicks(s: str) -> list[tuple[str, datetime, str]]:
    """The CSV stores clicks as a Python literal of tuples."""
    s = s.strip() or "[]"
    try:
        # The data uses datetime.datetime(...) which is not JSON-safe; use
        # a small eval namespace
        ns = {"datetime": __import__("datetime").datetime}
        return list(eval(s, ns, ns))
    except Exception:
        return []


def _session_to_dict(s: InteractionSession) -> dict:
    return {
        "schema_version": "v1",
        "session_id": s.session_id,
        "dataset_id": s.dataset_id,
        "session_type": s.session_type.value,
        "user_id": s.user_id,
        "topic_id": s.topic_id,
        "domain": s.domain,
        "events": [
            {
                "event_id": e.event_id,
                "type": e.type.value,
                "timestamp": e.timestamp,
                "turn_id": e.turn_id,
                "role": e.role.value if e.role else None,
                "query": e.query,
                "utterance": e.utterance,
                "ranked_items": [
                    {"doc_id": r.doc_id, "rank": r.rank,
                     "score": r.score, "judged_relevance": r.judged_relevance}
                    for r in (e.ranked_items or [])
                ] if e.ranked_items else None,
                "clicked_items": [
                    {"doc_id": c.doc_id, "rank": c.rank,
                     "dwell_time": c.dwell_time, "meta": c.meta}
                    for c in (e.clicked_items or [])
                ] if e.clicked_items else None,
                "dialogue_act": e.dialogue_act,
                "response_to": e.response_to,
                "meta": e.meta,
            }
            for e in s.events
        ],
        "meta": s.meta,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True,
                   help="Path to evaluation_sessions_Task_A.csv")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    print(f"Parsing {args.csv}")
    rows: list[dict] = []
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        for r in reader:
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
    print(f"  parsed {len(rows)} interactions")

    # Group by (user_id, topic_id) -> session
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["user_id"], r["topic_id"]), []).append(r)

    print(f"  reconstructed {len(by_key)} unique (user, topic) sessions")

    sessions: list[InteractionSession] = []
    for (user, topic), interactions in by_key.items():
        interactions.sort(key=lambda x: x["timestamp"])
        events: list[Event] = []
        eid = 0
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
            ranked_items = [RankedItem(doc_id=d, rank=i + 1)
                            for i, d in enumerate(it["serp"][:10])]
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
                    if not c or len(c) < 1:
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
            sessions.append(InteractionSession(
                session_id=f"sim4ia-{user}-{topic}",
                dataset_id="sim4ia-bench-task-a",
                session_type=SessionType.SEARCH,
                events=events,
                user_id=user,
                topic_id=topic,
                domain="academic-search",
                meta={"source": "Sim4IA-Bench", "task": "A"},
            ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in sessions:
            f.write(json.dumps(_session_to_dict(s)) + "\n")
    print(f"  -> {args.output} ({len(sessions)} sessions)")


if __name__ == "__main__":
    main()
