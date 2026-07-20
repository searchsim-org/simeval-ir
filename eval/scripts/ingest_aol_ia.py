#!/usr/bin/env python
"""Ingest the FULL AOL-IA dataset (via ir_datasets) into the SimEval-IR
canonical JSONL session format.

The AOL-IA release (10M unique queries, 36.4M qlog entries, 19.4M qrels) is a
full reconstruction of the AOL Query Log with rebuilt URL clicks and a frozen
document store. We sessionise per user_id with a configurable inactivity
timeout, mirror the AOL adapter's loss-accounting behaviour, and emit one
SimEval-IR session per JSONL line.

The output of this script is what B1 / B2 / B3 use as the "real" reference
corpus. No synthetic or excerpt data is involved.

Example:
    python ingest_aol_ia.py --output data/aol_ia/sessions.jsonl --max-sessions 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from simeval_ir.core.session import (  # noqa: E402
    InteractionSession,
    Event,
    Click,
    RankedItem,
)
from simeval_ir.core.types import EventType, Role, SessionType  # noqa: E402
from simeval_ir.datasets.io import save_sessions_to_jsonl  # noqa: E402

import ir_datasets  # noqa: E402


def _emit_session(user_id: str, session_idx: int,
                  events_data: list[dict]) -> InteractionSession | None:
    """Convert a list of (time, query, query_id, items) into a session."""
    if not events_data:
        return None

    events: list[Event] = []
    base_time = events_data[0]["time"]
    counter = 0

    for ev in events_data:
        ts = (ev["time"] - base_time).total_seconds()

        # query
        counter += 1
        events.append(Event(
            event_id=f"aol-{user_id}-{session_idx}-q{counter}",
            type=EventType.QUERY_ISSUED,
            timestamp=ts,
            role=Role.USER,
            query=ev["query"],
            meta={"query_id": ev["query_id"], "query_orig": ev["query_orig"]},
        ))

        # SERP_VIEW: synthesise a SERP from the items list (clicked + non-clicked
        # at the recorded ranks)
        if ev["items"]:
            counter += 1
            ranked = [
                RankedItem(doc_id=it.doc_id,
                           rank=int(it.rank) if str(it.rank).isdigit() else 1,
                           score=None,
                           judged_relevance=int(it.clicked))
                for it in ev["items"]
            ]
            ranked.sort(key=lambda r: r.rank)
            events.append(Event(
                event_id=f"aol-{user_id}-{session_idx}-s{counter}",
                type=EventType.SERP_VIEW,
                timestamp=ts + 0.5,
                role=Role.SYSTEM,
                ranked_items=ranked,
            ))

            clicks = [
                Click(doc_id=it.doc_id,
                      rank=int(it.rank) if str(it.rank).isdigit() else 1)
                for it in ev["items"] if it.clicked
            ]
            if clicks:
                counter += 1
                events.append(Event(
                    event_id=f"aol-{user_id}-{session_idx}-c{counter}",
                    type=EventType.CLICK,
                    timestamp=ts + 1.0,
                    role=Role.USER,
                    clicked_items=clicks,
                ))

    return InteractionSession(
        session_id=f"aol-{user_id}-{session_idx:04d}",
        dataset_id="aol-ia",
        session_type=SessionType.SEARCH,
        events=events,
        user_id=user_id,
        domain="web",
        meta={"adapter": "aol-ia", "n_queries": len(events_data)},
    )


def ingest(output_path: Path, max_sessions: int = 20000,
           min_session_len: int = 2, timeout_minutes: int = 30,
           start_user: int = 0):
    """Stream AOL-IA qlogs, sessionise, write JSONL.

    Sessions of length < min_session_len are discarded.
    """
    timeout = timedelta(minutes=timeout_minutes)
    ds = ir_datasets.load("aol-ia")
    print(f"AOL-IA loaded. Total qlog entries: {ds.qlogs_count():,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_emitted = 0
    n_dropped_short = 0
    n_users = 0
    current_user: str | None = None
    current_session: list[dict] = []
    current_idx = 0
    last_time: datetime | None = None
    skipped_users = 0

    written = open(output_path, "w")
    try:
        for log in ds.qlogs_iter():
            user = log.user_id
            t = log.time

            if user != current_user:
                # Emit pending session
                if current_session and len(current_session) >= min_session_len:
                    sess = _emit_session(current_user, current_idx, current_session)
                    if sess is not None:
                        written.write(json.dumps(_session_to_dict(sess)) + "\n")
                        n_emitted += 1
                elif current_session:
                    n_dropped_short += 1
                if current_user is not None:
                    n_users += 1
                if n_emitted >= max_sessions:
                    break
                current_user = user
                current_idx = 0
                current_session = []
                last_time = None
                if n_users < start_user:
                    skipped_users += 1
                    # We still need to advance the iterator; no need to record
                    # events for skipped users
                    continue

            if n_users < start_user:
                continue

            # Same user: check timeout
            if last_time is not None and (t - last_time) > timeout:
                if len(current_session) >= min_session_len:
                    sess = _emit_session(user, current_idx, current_session)
                    if sess is not None:
                        written.write(json.dumps(_session_to_dict(sess)) + "\n")
                        n_emitted += 1
                        if n_emitted >= max_sessions:
                            break
                else:
                    n_dropped_short += 1
                current_idx += 1
                current_session = []

            current_session.append({
                "time": t, "query": log.query, "query_id": log.query_id,
                "query_orig": log.query_orig, "items": log.items,
            })
            last_time = t

        # Emit trailing session
        if current_session and len(current_session) >= min_session_len and n_emitted < max_sessions:
            sess = _emit_session(current_user, current_idx, current_session)
            if sess is not None:
                written.write(json.dumps(_session_to_dict(sess)) + "\n")
                n_emitted += 1
    finally:
        written.close()

    print(f"\nIngestion complete:")
    print(f"  sessions emitted: {n_emitted:,}")
    print(f"  sessions dropped (< {min_session_len} queries): {n_dropped_short:,}")
    print(f"  users seen: {n_users:,}")
    print(f"  output: {output_path}")
    return n_emitted


def _session_to_dict(s: InteractionSession) -> dict:
    """Serialise InteractionSession into the SimEval-IR JSONL schema."""
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-sessions", type=int, default=20000)
    p.add_argument("--min-session-len", type=int, default=2)
    p.add_argument("--timeout-minutes", type=int, default=30)
    args = p.parse_args()

    ingest(args.output, max_sessions=args.max_sessions,
           min_session_len=args.min_session_len,
           timeout_minutes=args.timeout_minutes)


if __name__ == "__main__":
    main()
