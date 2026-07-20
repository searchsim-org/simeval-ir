#!/usr/bin/env python
"""Drive PBM, DBN, heuristic and LLM-style simulators on the AOL-IA query
stream so that they share the same queries, the same per-query observed
ranks, and the same number of sessions as the real AOL reference corpus.

For each real AOL session we keep the structure (number of queries, query
text, observed SERP positions) and only re-sample the *click* events using
the simulator's click model, parameterised by the click probabilities from
the click-model literature.

This makes the simulator outputs strictly comparable to the real AOL
sessions: any difference observed by B1 is in the click pattern, not in
the query distribution or session length.

The "judged relevance" used by the click models is binary -- 1 for ranks
that were clicked in the real log (treated as positive), 0 otherwise --
because AOL has no editorial judgements.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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
from simeval_ir.datasets.io import save_sessions_to_jsonl, load_sessions_from_jsonl  # noqa: E402


# --- Click models (literature parameters) -----------------------------------

# PBM (Craswell et al., 2008): independent examination per rank * attractiveness
PBM_EXAM = [0.94, 0.84, 0.71, 0.55, 0.43, 0.32, 0.24, 0.18, 0.13, 0.10]
PBM_ATTR = [0.05, 0.55]      # P(attract | rel)

# DBN (Chapelle and Zhang, 2009): cascade with satisfaction
DBN_ATTR = [0.08, 0.65]
DBN_SAT  = [0.05, 0.55]
DBN_GAMMA = 0.20             # P(continue | satisfied)

# Heuristic: pure rank-based, ignores relevance
HEUR_PROB = [0.45, 0.27, 0.16, 0.10, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01]


def _pbm_clicks(rng: random.Random, serp: list[RankedItem]) -> list[Click]:
    out = []
    for it in serp:
        rel = it.judged_relevance or 0
        e = PBM_EXAM[min(it.rank - 1, 9)]
        a = PBM_ATTR[min(rel, 1)]
        if rng.random() < e * a:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(3.0, rng.gauss(20 + 8 * rel, 6))))
    return out


def _dbn_clicks(rng: random.Random, serp: list[RankedItem]) -> list[Click]:
    out = []
    for it in serp:
        rel = it.judged_relevance or 0
        if rng.random() < DBN_ATTR[min(rel, 1)]:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(3.0, rng.gauss(25 + 10 * rel, 7))))
            if rng.random() < DBN_SAT[min(rel, 1)] and rng.random() > DBN_GAMMA:
                break
    return out


def _heur_clicks(rng: random.Random, serp: list[RankedItem]) -> list[Click]:
    out = []
    for it in serp:
        if rng.random() < HEUR_PROB[min(it.rank - 1, 9)]:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(3.0, rng.uniform(5, 25))))
    return out


def _llm_clicks(rng: random.Random, serp: list[RankedItem]) -> list[Click]:
    """Verbose LLM-style: examines every rank, click probability scales with
    relevance but with a high baseline ('over-clicker')."""
    out = []
    for it in serp:
        rel = it.judged_relevance or 0
        e = max(0.30, 0.92 - 0.06 * (it.rank - 1))
        a = 0.18 + 0.40 * rel
        if rng.random() < e * a:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(8.0, rng.gauss(40 + 15 * rel, 12))))
    return out


CLICK_MODELS = {
    "simiir-pbm": _pbm_clicks,
    "simiir-dbn": _dbn_clicks,
    "heuristic":  _heur_clicks,
    "llm-sim":    _llm_clicks,
}


def _llm_query_paraphrase(q: str | None, rng: random.Random) -> str | None:
    """Mild verbose paraphrase to reflect typical LLM-simulator output."""
    if not q:
        return q
    if rng.random() < 0.35:
        prefixes = ["how do i find ", "tell me about ", "search for ",
                    "i'm looking for ", "what is "]
        return rng.choice(prefixes) + q
    return q


def simulate(real_path: Path, out_dir: Path, sim_name: str, seed: int):
    click_fn = CLICK_MODELS[sim_name]
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sim_name}.jsonl"

    n_sessions = 0
    sessions = []

    for real_session in load_sessions_from_jsonl(real_path):
        new_events = []
        counter = 0
        for ev in real_session.events:
            if ev.type == EventType.QUERY_ISSUED:
                counter += 1
                q = ev.query
                if sim_name == "llm-sim":
                    q = _llm_query_paraphrase(q, rng)
                new_events.append(Event(
                    event_id=f"{sim_name}-{real_session.session_id}-q{counter}",
                    type=EventType.QUERY_ISSUED,
                    timestamp=ev.timestamp,
                    role=Role.USER,
                    query=q,
                    meta={"src_event_id": ev.event_id},
                ))
            elif ev.type == EventType.SERP_VIEW and ev.ranked_items:
                counter += 1
                new_events.append(Event(
                    event_id=f"{sim_name}-{real_session.session_id}-s{counter}",
                    type=EventType.SERP_VIEW,
                    timestamp=ev.timestamp,
                    role=Role.SYSTEM,
                    ranked_items=ev.ranked_items,
                ))
                clicks = click_fn(rng, ev.ranked_items)
                if clicks:
                    counter += 1
                    new_events.append(Event(
                        event_id=f"{sim_name}-{real_session.session_id}-c{counter}",
                        type=EventType.CLICK,
                        timestamp=(ev.timestamp or 0) + 1.0,
                        role=Role.USER,
                        clicked_items=clicks,
                    ))
            # CLICK events from the real session are intentionally dropped --
            # the simulator generates its own clicks above
        sessions.append(InteractionSession(
            session_id=f"{sim_name}-{real_session.session_id}",
            dataset_id=f"aol-ia.{sim_name}",
            session_type=SessionType.SEARCH,
            events=new_events,
            user_id=real_session.user_id,
            topic_id=real_session.topic_id,
            domain=real_session.domain,
            meta={"simulator": sim_name, "src_session_id": real_session.session_id},
        ))
        n_sessions += 1

    save_sessions_to_jsonl(sessions, out_file)
    print(f"{sim_name}: wrote {n_sessions} sessions -> {out_file}")
    return n_sessions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--simulators", type=str,
                   default="simiir-pbm,simiir-dbn,heuristic,llm-sim")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for i, name in enumerate([s.strip() for s in args.simulators.split(",")]):
        simulate(args.real, args.output, name, seed=args.seed + 17 * i)


if __name__ == "__main__":
    main()
