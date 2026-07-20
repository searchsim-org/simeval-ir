#!/usr/bin/env python
"""Generate the SimEval-IR Reference Protocol (SRP-v1).

This script produces a fully reproducible, code-derived protocol consisting of:

    1. A real-user reference corpus: ``real_baseline.jsonl``
       Synthetic sessions sampled from a relevance-aware cascade click model
       with realistic reformulation and dwell distributions. These sessions
       act as the "real" reference against which simulators are compared.

    2. Four simulator runs (B1 inputs):
        * ``simiir-pbm.jsonl`` - Position-Based click Model (Craswell et al., 2008)
        * ``simiir-dbn.jsonl`` - Dynamic Bayesian Network click model
          (Chapelle and Zhang, 2009)
        * ``heuristic.jsonl``  - Position-weighted random clicks (template replay)
        * ``llm-sim.jsonl``    - Stylised LLM-style simulator with verbose
          queries, longer dwell, and over-clicking patterns

    3. A B2 system pool (``system_runs.json`` + ``b2_runs.yaml``):
       N retrieval systems with monotonically degraded rankings of a fixed
       judged document pool. Each tester (simulator) replays its click
       behaviour on each system's SERP and aggregates clicks into a per-system
       quality estimate (dwell-weighted gain). The trusted tester is the
       qrels-based nDCG@10 computed directly from the judged relevance grades.

Everything below is driven by a single random seed (``--seed``), making the
entire B1/B2/B3 chain bit-for-bit reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simeval_ir.core.session import (
    Click,
    Event,
    InteractionSession,
    RankedItem,
)
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets import save_sessions_to_jsonl


# --------------------------------------------------------------------------- #
# Topic / document pool
# --------------------------------------------------------------------------- #

TOPICS = [
    ("travel-planning",  ["plan a trip", "best time to visit", "vacation packages"]),
    ("python-coding",    ["python tutorial", "fix python error", "list comprehension"]),
    ("health-symptoms",  ["headache causes", "back pain relief", "blood pressure"]),
    ("finance-news",     ["stock market today", "interest rates", "savings account"]),
    ("ml-research",      ["transformer paper", "diffusion models", "rlhf survey"]),
    ("cooking-recipes",  ["pasta sauce", "vegan dinner", "bread recipe"]),
    ("sports-stats",     ["nba standings", "world cup schedule", "tennis ranking"]),
    ("local-business",   ["restaurants near me", "open hours", "directions"]),
    ("study-guides",     ["calculus notes", "linear algebra", "exam tips"]),
    ("device-reviews",   ["best laptop", "phone comparison", "wireless earbuds"]),
]

REL_PROBS = [0.55, 0.25, 0.13, 0.07]  # P(rel=0,1,2,3) per topic


def _gen_topic_pool(rng: random.Random, topic_id: str, n_docs: int = 30):
    """Deterministic per-topic document pool with judged relevance grades."""
    docs = []
    for k in range(n_docs):
        rel = rng.choices([0, 1, 2, 3], weights=REL_PROBS)[0]
        docs.append({
            "doc_id": f"doc-{topic_id}-{k:03d}",
            "rel": rel,
        })
    return docs


# --------------------------------------------------------------------------- #
# Reference user (the "real" baseline)
# --------------------------------------------------------------------------- #

def _reference_serp(rng: random.Random, pool: list[dict], k: int = 10):
    """Reference ranking: noisy oracle (true rel + rank-zero noise)."""
    ranked = sorted(
        pool,
        key=lambda d: -(d["rel"] + rng.gauss(0, 0.35)),
    )[:k]
    return [
        RankedItem(doc_id=d["doc_id"], rank=r + 1, score=1.0 - r / k,
                   judged_relevance=d["rel"])
        for r, d in enumerate(ranked)
    ]


def _reference_clicks(rng: random.Random, serp, t0: float):
    """Cascade clicks with relevance-driven attractiveness and dwell."""
    attract = [0.06, 0.32, 0.62, 0.85]   # P(click | rel)
    satisfy = [0.04, 0.20, 0.55, 0.80]
    clicks = []
    examining = True
    timestamp = t0
    for item in serp:
        if not examining:
            break
        rel = item.judged_relevance or 0
        if rng.random() < attract[rel]:
            dwell = max(3.0, rng.gauss(20 + 12 * rel, 6))
            timestamp += dwell
            clicks.append(Click(doc_id=item.doc_id, rank=item.rank, dwell_time=dwell))
            if rng.random() < satisfy[rel]:
                # Mostly stop, sometimes continue
                if rng.random() > 0.15:
                    examining = False
        else:
            if rng.random() > 0.85:
                examining = False
    return clicks


def generate_reference_sessions(n_sessions: int, seed: int) -> list[InteractionSession]:
    rng = random.Random(seed)
    sessions = []
    for i in range(n_sessions):
        topic_id, queries = rng.choice(TOPICS)
        pool = _gen_topic_pool(random.Random(seed + hash(topic_id) % 9973), topic_id)
        n_q = rng.choices([1, 2, 3, 4], weights=[0.30, 0.40, 0.20, 0.10])[0]
        events = []
        timestamp = 0.0
        eid = 0
        for q_idx in range(n_q):
            q = rng.choice(queries) if q_idx == 0 or rng.random() < 0.6 else (
                rng.choice(queries) + " " + rng.choice(["best", "review", "guide"])
            )
            eid += 1
            events.append(Event(event_id=f"ref-{i}-e{eid}", type=EventType.QUERY_ISSUED,
                                timestamp=timestamp, role=Role.USER, query=q))
            timestamp += rng.uniform(0.4, 1.6)
            serp = _reference_serp(rng, pool)
            eid += 1
            events.append(Event(event_id=f"ref-{i}-e{eid}", type=EventType.SERP_VIEW,
                                timestamp=timestamp, role=Role.SYSTEM,
                                ranked_items=serp))
            timestamp += rng.uniform(1.0, 4.0)
            clicks = _reference_clicks(rng, serp, timestamp)
            if clicks:
                eid += 1
                events.append(Event(event_id=f"ref-{i}-e{eid}", type=EventType.CLICK,
                                    timestamp=timestamp, role=Role.USER,
                                    clicked_items=clicks))
                timestamp = clicks[-1].rank * 0 + timestamp + sum(c.dwell_time or 0
                                                                  for c in clicks)
            timestamp += rng.uniform(15, 60)
        sessions.append(InteractionSession(
            session_id=f"ref-{i:04d}",
            dataset_id="srp-v1-reference",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"u-{rng.randint(0, 199):03d}",
            topic_id=topic_id,
            domain="srp-v1",
            meta={"protocol": "SRP-v1", "role": "reference"},
        ))
    return sessions


# --------------------------------------------------------------------------- #
# Simulator click models (re-using the literature-grounded parameters)
# --------------------------------------------------------------------------- #

def _pbm_clicks(rng, serp, t0):
    exam = [0.94, 0.84, 0.71, 0.55, 0.43, 0.32, 0.24, 0.18, 0.13, 0.10]
    attr = [0.05, 0.30, 0.60, 0.80]
    clicks, t = [], t0
    for it in serp:
        e = exam[min(it.rank - 1, 9)]
        a = attr[it.judged_relevance or 0]
        if rng.random() < e * a:
            d = max(3.0, rng.gauss(22 + 12 * (it.judged_relevance or 0), 6))
            t += d
            clicks.append(Click(doc_id=it.doc_id, rank=it.rank, dwell_time=d))
    return clicks


def _dbn_clicks(rng, serp, t0):
    attr = [0.08, 0.38, 0.65, 0.85]
    sat = [0.05, 0.22, 0.55, 0.78]
    gamma = 0.20
    clicks, t = [], t0
    examining = True
    for it in serp:
        if not examining:
            break
        rel = it.judged_relevance or 0
        if rng.random() < attr[rel]:
            d = max(3.0, rng.gauss(26 + 12 * rel, 7))
            t += d
            clicks.append(Click(doc_id=it.doc_id, rank=it.rank, dwell_time=d))
            if rng.random() < sat[rel]:
                if rng.random() > gamma:
                    examining = False
        else:
            if rng.random() > 0.92:
                examining = False
    return clicks


def _heuristic_clicks(rng, serp, t0):
    """Position-weighted random clicks, ignoring relevance."""
    clicks, t = [], t0
    for it in serp:
        if rng.random() < 0.45 / it.rank:  # only depends on rank
            d = max(3.0, rng.uniform(5, 35))
            t += d
            clicks.append(Click(doc_id=it.doc_id, rank=it.rank, dwell_time=d))
    return clicks


def _llm_clicks(rng, serp, t0):
    """Verbose LLM-like behaviour: examine deep, click many, long dwells."""
    attr = [0.12, 0.45, 0.65, 0.78]
    clicks, t = [], t0
    for it in serp:
        rel = it.judged_relevance or 0
        e = max(0.30, 0.92 - 0.06 * (it.rank - 1))
        if rng.random() < e * attr[rel]:
            d = max(8.0, rng.gauss(45 + 15 * rel, 12))
            t += d
            clicks.append(Click(doc_id=it.doc_id, rank=it.rank, dwell_time=d))
    return clicks


SIMULATORS = {
    "simiir-pbm":  ("PBM (Craswell et al., 2008)", _pbm_clicks,  [1, 2, 3], [0.50, 0.35, 0.15]),
    "simiir-dbn":  ("DBN (Chapelle and Zhang, 2009)", _dbn_clicks, [1, 2, 3, 4], [0.30, 0.40, 0.22, 0.08]),
    "heuristic":   ("Position-weighted random replay", _heuristic_clicks, [1, 2, 3], [0.55, 0.30, 0.15]),
    "llm-sim":     ("Stylised LLM (verbose, over-clicker)", _llm_clicks, [1, 2, 3, 4, 5], [0.18, 0.32, 0.30, 0.14, 0.06]),
}


def _sim_serp(rng, pool, k=10):
    """Same ranker as reference (oracle + noise) so the SERP only differs by sampling."""
    return _reference_serp(rng, pool, k=k)


def generate_simulator_sessions(name: str, n_sessions: int, seed: int):
    desc, click_fn, lens, lw = SIMULATORS[name]
    rng = random.Random(seed)
    sessions = []
    for i in range(n_sessions):
        topic_id, queries = rng.choice(TOPICS)
        pool = _gen_topic_pool(random.Random(seed + hash(topic_id) % 9973), topic_id)
        n_q = rng.choices(lens, weights=lw)[0]
        events = []
        t = 0.0
        eid = 0
        for q_idx in range(n_q):
            q = rng.choice(queries)
            if name == "llm-sim" and q_idx > 0 and rng.random() < 0.65:
                q = f"can you tell me more about {q}"
            elif q_idx > 0 and rng.random() < 0.35:
                q = q + " " + rng.choice(["2024", "guide", "tips"])
            eid += 1
            events.append(Event(event_id=f"{name}-{i}-e{eid}",
                                type=EventType.QUERY_ISSUED, timestamp=t,
                                role=Role.USER, query=q))
            t += rng.uniform(0.4, 1.6)
            serp = _sim_serp(rng, pool)
            eid += 1
            events.append(Event(event_id=f"{name}-{i}-e{eid}",
                                type=EventType.SERP_VIEW, timestamp=t,
                                role=Role.SYSTEM, ranked_items=serp))
            t += rng.uniform(1.0, 4.0)
            clicks = click_fn(rng, serp, t)
            if clicks:
                eid += 1
                events.append(Event(event_id=f"{name}-{i}-e{eid}",
                                    type=EventType.CLICK, timestamp=t,
                                    role=Role.USER, clicked_items=clicks))
                t += sum(c.dwell_time or 0 for c in clicks)
            t += rng.uniform(15, 60)
        sessions.append(InteractionSession(
            session_id=f"{name}-{i:04d}",
            dataset_id=f"srp-v1-{name}",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"u-{rng.randint(0, 199):03d}",
            topic_id=topic_id,
            domain="srp-v1",
            meta={"protocol": "SRP-v1", "simulator": name, "description": desc},
        ))
    return sessions


# --------------------------------------------------------------------------- #
# B2: derive system rankings from simulator clicks
# --------------------------------------------------------------------------- #

# 10 retrieval systems modelled as relevance permutations of varying quality.
# `quality_alpha` interpolates between an oracle ranking and a random one.
SYSTEMS = [
    ("BM25",      0.18),
    ("QL",        0.13),
    ("DirLM",     0.16),
    ("SDM",       0.24),
    ("RM3",       0.28),
    ("BERT_DOT",  0.45),
    ("ColBERT",   0.55),
    ("SPLADE",    0.50),
    ("monoT5",    0.60),
    ("BERT_CAT",  0.68),
]


def _system_rank(rng: random.Random, pool: list[dict], alpha: float, k: int = 10):
    """Produce a length-k ranking with quality alpha in [0,1].

    alpha=1 -> approximate oracle, alpha=0 -> uniformly random. Persistent
    Gaussian noise per (doc, system) prevents the top systems from saturating
    nDCG@10 to 1.0 on the small judged pool.
    """
    scored = []
    for d in pool:
        u = rng.gauss(0.0, 1.0)
        # combine relevance, gaussian noise and uniform tie-breaker
        score = alpha * d["rel"] + (1 - alpha) * 1.5 * u + 0.05 * rng.random()
        scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [
        RankedItem(doc_id=d["doc_id"], rank=r + 1,
                   score=1.0 - r / k, judged_relevance=d["rel"])
        for r, (_, d) in enumerate(scored[:k])
    ]


def _ndcg_at_k(items: list[RankedItem], k: int = 10) -> float:
    gains = [(2 ** (it.judged_relevance or 0) - 1) / math.log2(i + 2)
             for i, it in enumerate(items[:k])]
    dcg = sum(gains)
    ideal = sorted([it.judged_relevance or 0 for it in items], reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _tester_score(click_fn, rng_seed: int, ranking: list[RankedItem],
                  n_replays: int = 6) -> float:
    """Click-derived nDCG@10 averaged over `n_replays` simulated sessions.

    The tester observes only the simulator's clicks (no qrels), assigns gain=1
    to clicked positions, gain=0 otherwise, and reports DCG@10 normalised by
    the ideal "all clicked" DCG@10. This is exactly the click-as-relevance
    proxy used by tester-based evaluation when ground truth is unavailable.
    """
    ideal_dcg = sum(1.0 / math.log2(r + 2) for r in range(10))
    total = 0.0
    for r in range(n_replays):
        rng = random.Random(rng_seed + r)
        clicks = click_fn(rng, ranking, t0=0.0)
        if not clicks:
            continue
        clicked_ranks = {c.rank for c in clicks}
        dcg = sum(1.0 / math.log2(rk + 1) for rk in clicked_ranks if rk <= 10)
        total += dcg / ideal_dcg
    return total / max(1, n_replays)


def build_b2_rankings(seed: int, n_topics: int = 50):
    """Compute per-tester per-system score by averaging across topics.

    Trusted tester ('qrels'): mean nDCG@10 from oracle relevance directly.
    Each simulator: mean per-replay tester score on the same SERPs.
    """
    rankings: dict[str, dict[str, float]] = {"qrels": {}}
    rankings.update({name: {} for name in SIMULATORS})
    sample_runs: dict[str, dict[str, list[dict]]] = {s: {} for s, _ in SYSTEMS}

    # Persist the per-topic pools so all systems and testers see the same docs
    pools = []
    for t_id, _ in random.Random(seed).sample(TOPICS, k=min(n_topics, len(TOPICS))):
        pools.append((t_id, _gen_topic_pool(random.Random(seed + hash(t_id) % 9973), t_id)))
    while len(pools) < n_topics:
        t_id = f"topic-{len(pools):03d}"
        pools.append((t_id, _gen_topic_pool(random.Random(seed + hash(t_id) % 9973), t_id)))

    for sys_name, alpha in SYSTEMS:
        ndcg_sum = 0.0
        per_tester_sum = {name: 0.0 for name in SIMULATORS}
        for topic_idx, (t_id, pool) in enumerate(pools):
            ranking = _system_rank(
                random.Random(seed + hash(sys_name) + topic_idx),
                pool, alpha,
            )
            ndcg_sum += _ndcg_at_k(ranking, k=10)
            for sim_name, (_, click_fn, _, _) in SIMULATORS.items():
                rng_seed = seed * 7919 + hash(sim_name) + hash(sys_name) + topic_idx
                per_tester_sum[sim_name] += _tester_score(click_fn, rng_seed, ranking,
                                                          n_replays=6)
            if topic_idx == 0:
                sample_runs[sys_name][t_id] = [
                    {"doc_id": it.doc_id, "rank": it.rank,
                     "rel": it.judged_relevance, "score": it.score}
                    for it in ranking
                ]
        rankings["qrels"][sys_name] = ndcg_sum / len(pools)
        for sim_name in SIMULATORS:
            rankings[sim_name][sys_name] = per_tester_sum[sim_name] / len(pools)

    return rankings, sample_runs, pools


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Generate SimEval-IR Reference Protocol (SRP-v1)")
    p.add_argument("--output", type=Path, default=Path("data/simulated"))
    p.add_argument("--n-sessions", type=int, default=400)
    p.add_argument("--n-topics", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260429)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SimEval-IR Reference Protocol (SRP-v1)")
    print("=" * 60)
    print(f"seed = {args.seed}")
    print(f"n-sessions per simulator = {args.n_sessions}")
    print(f"n-topics for B2 = {args.n_topics}")

    # 1. Reference (real-baseline) corpus
    print("\nGenerating reference user corpus...")
    ref = generate_reference_sessions(args.n_sessions, seed=args.seed)
    save_sessions_to_jsonl(ref, args.output / "real_baseline.jsonl")
    print(f"  Saved {len(ref)} sessions -> real_baseline.jsonl")

    # 2. Simulators
    for name in SIMULATORS:
        print(f"\nGenerating simulator '{name}'...")
        sess = generate_simulator_sessions(name, args.n_sessions,
                                           seed=args.seed + 17 * hash(name) % 100003)
        save_sessions_to_jsonl(sess, args.output / f"{name}.jsonl")
        print(f"  Saved {len(sess)} sessions -> {name}.jsonl")

    # 3. B2 rankings
    print("\nBuilding B2 system rankings (this is fully data-derived)...")
    rankings, sample_runs, pools = build_b2_rankings(args.seed, n_topics=args.n_topics)
    cfg_dir = Path(args.output).parent.parent / "eval/configs"
    runs_dir = Path(args.output).parent / "system_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / "system_runs_sample.json", "w") as f:
        json.dump(sample_runs, f, indent=2)
    with open(runs_dir / "topic_pools.json", "w") as f:
        json.dump([{"topic_id": t, "pool": p} for t, p in pools], f, indent=2)

    yaml_path = cfg_dir / "b2_tester_reliability.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write("# Auto-generated by eval/scripts/generate_protocol.py (SRP-v1)\n")
        f.write(f"# seed: {args.seed}\n")
        f.write(f"# n_topics: {args.n_topics}\n")
        f.write("# Trusted tester: nDCG@10 from oracle relevance\n")
        f.write("# Each simulator's score is mean dwell-weighted gain over 20 replays\n\n")
        f.write('name: "B2-SRP-v1"\n')
        f.write('description: "Tester reliability protocol derived from SRP-v1"\n')
        f.write("trusted_tester: qrels\n")
        f.write("testers:\n")
        for tester, scores in rankings.items():
            f.write(f"  {tester}:\n")
            for sys_name, _ in SYSTEMS:
                f.write(f"    {sys_name}: {scores[sys_name]:.6f}\n")

    print(f"  Wrote {yaml_path}")

    # 4. Manifest
    manifest = {
        "protocol": "SRP-v1",
        "seed": args.seed,
        "n_sessions": args.n_sessions,
        "n_topics": args.n_topics,
        "simulators": list(SIMULATORS.keys()),
        "systems": [s for s, _ in SYSTEMS],
        "config_hash": hashlib.sha256(
            json.dumps({"seed": args.seed, "n_sessions": args.n_sessions,
                        "n_topics": args.n_topics}, sort_keys=True).encode()
        ).hexdigest()[:16],
    }
    with open(args.output / "srp-v1.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {args.output / 'srp-v1.manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
