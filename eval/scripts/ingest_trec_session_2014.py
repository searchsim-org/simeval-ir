#!/usr/bin/env python
"""Ingest the FULL TREC Session Track 2014 dataset into the SimEval-IR
canonical JSONL session format.

Source: NIST TREC Session Track 2014 release
    sessiontrack2014.xml   -- 1257 sessions, 3645 interactions, 1041 clicks
    judgments.txt          -- 16949 graded relevance judgements

Each `<session>` becomes one InteractionSession. Each `<interaction>` produces
a QUERY_ISSUED -> SERP_VIEW -> (optional) CLICK chain. Click dwell time is
computed from `endtime - starttime` recorded in the XML. Doc ids use the
`clueweb12id` field for SERP results and `docno` for clicks (the two are
identical in the 2014 release). Topics from the XML's `<topic>/<desc>` block
are recorded as session.topic_id and session.meta["topic_desc"].
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
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


def _safe_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_results(interaction: ET.Element, qrels_lookup: dict) -> list[RankedItem]:
    items = []
    for r in interaction.findall("results/result"):
        rank = int(r.get("rank", "0") or "0")
        cw = (r.findtext("clueweb12id") or "").strip()
        url = (r.findtext("url") or "").strip()
        title = (r.findtext("title") or "").strip()
        snippet = (r.findtext("snippet") or "").strip()
        doc_id = cw or url
        if not doc_id:
            continue
        rel = qrels_lookup.get(doc_id)
        meta = {}
        if url:
            meta["url"] = url
        if title:
            meta["title"] = title
        if snippet:
            meta["snippet"] = snippet
        items.append(RankedItem(
            doc_id=doc_id, rank=rank, score=None,
            judged_relevance=rel, meta=meta,
        ))
    items.sort(key=lambda r: r.rank)
    return items


def _parse_clicks(interaction: ET.Element, t_offset: float) -> list[Click]:
    clicks = []
    clicked = interaction.find("clicked")
    if clicked is None:
        return clicks
    for c in clicked.findall("click"):
        rank = int((c.findtext("rank") or "0") or "0")
        doc_id = (c.findtext("docno") or "").strip()
        if not doc_id:
            continue
        st = _safe_float(c.get("starttime"))
        et = _safe_float(c.get("endtime"))
        dwell = (et - st) if (st is not None and et is not None) else None
        clicks.append(Click(doc_id=doc_id, rank=rank, dwell_time=dwell))
    return clicks


def _build_session(s_elem: ET.Element, qrels_lookup: dict[int, dict[str, int]]) -> InteractionSession | None:
    session_num = s_elem.get("num")
    user_id = s_elem.get("userid")
    s_starttime = _safe_float(s_elem.get("starttime")) or 0.0

    topic_elem = s_elem.find("topic")
    topic_id = topic_elem.get("num") if topic_elem is not None else None
    topic_desc = (topic_elem.findtext("desc") or "").strip() if topic_elem is not None else None

    topic_qrels = (qrels_lookup.get(int(topic_id)) or {}) if topic_id and topic_id.isdigit() else {}

    events: list[Event] = []
    counter = 0
    for inter in s_elem.findall("interaction"):
        i_start = _safe_float(inter.get("starttime")) or 0.0
        rel_t = i_start  # relative timestamp (already in seconds from session start)

        q_text = (inter.findtext("query") or "").strip()
        if not q_text:
            continue

        counter += 1
        events.append(Event(
            event_id=f"trec-s14-{session_num}-q{counter}",
            type=EventType.QUERY_ISSUED,
            timestamp=rel_t,
            role=Role.USER,
            query=q_text,
            meta={
                "interaction_num": inter.get("num"),
                "type": inter.get("type"),
            },
        ))

        ranked_items = _parse_results(inter, topic_qrels)
        if ranked_items:
            counter += 1
            events.append(Event(
                event_id=f"trec-s14-{session_num}-s{counter}",
                type=EventType.SERP_VIEW,
                timestamp=rel_t + 0.5,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            ))

        clicks = _parse_clicks(inter, rel_t)
        if clicks:
            counter += 1
            t_click = rel_t + 1.0
            events.append(Event(
                event_id=f"trec-s14-{session_num}-c{counter}",
                type=EventType.CLICK,
                timestamp=t_click,
                role=Role.USER,
                clicked_items=clicks,
            ))

    if not events:
        return None

    return InteractionSession(
        session_id=f"trec-s14-{session_num}",
        dataset_id="trec-session-2014",
        session_type=SessionType.SEARCH,
        events=events,
        user_id=user_id,
        topic_id=topic_id,
        domain="web",
        meta={
            "topic_desc": topic_desc,
            "n_interactions": len(s_elem.findall("interaction")),
            "session_starttime": s_starttime,
        },
    )


def _load_qrels(path: Path) -> dict[int, dict[str, int]]:
    """TREC Session 2014 qrels file format: ``topic 0 docno relevance``."""
    out: dict[int, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                topic = int(parts[0])
                doc = parts[2]
                rel = int(parts[3])
            except (ValueError, IndexError):
                continue
            out.setdefault(topic, {})[doc] = rel
    return out


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
                     "score": r.score, "judged_relevance": r.judged_relevance,
                     "meta": r.meta}
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
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    print(f"Parsing qrels: {args.qrels}")
    qrels = _load_qrels(args.qrels)
    print(f"  topics with qrels: {len(qrels)}; total judgements: "
          f"{sum(len(d) for d in qrels.values())}")

    print(f"Parsing XML: {args.xml}")
    tree = ET.parse(args.xml)
    root = tree.getroot()
    sessions = root.findall("session")
    print(f"  found {len(sessions)} <session> elements")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_emitted = 0
    n_with_clicks = 0
    n_events_total = 0
    judged_serp_items = 0
    total_serp_items = 0

    with open(args.output, "w") as f:
        for s in sessions:
            sess = _build_session(s, qrels)
            if sess is None:
                continue
            f.write(json.dumps(_session_to_dict(sess)) + "\n")
            n_emitted += 1
            for ev in sess.events:
                n_events_total += 1
                if ev.type == EventType.CLICK:
                    n_with_clicks += 1
                if ev.ranked_items:
                    for r in ev.ranked_items:
                        total_serp_items += 1
                        if r.judged_relevance is not None:
                            judged_serp_items += 1

    print(f"\nIngestion complete.")
    print(f"  sessions emitted: {n_emitted}")
    print(f"  total events: {n_events_total}")
    print(f"  click events: {n_with_clicks}")
    print(f"  SERP items: {total_serp_items} ({judged_serp_items} judged, "
          f"{100*judged_serp_items/max(1,total_serp_items):.1f}%)")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
