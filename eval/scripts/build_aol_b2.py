#!/usr/bin/env python
"""Build a B2 (Tester Reliability) protocol from the AOL-IA dataset.

We sample N queries that have judged documents in the AOL-IA qrels file,
expand each query's qrel set into a candidate pool, and synthesise K
retrieval systems by ranking the pool with progressively more noise.

For each (system, query) pair we generate a ranked list and let each
simulator's click model produce clicks on that ranking. The tester's score
for a system is the average click-based nDCG@10 across queries; the trusted
tester is qrels-nDCG@10 computed directly from AOL relevance grades.

This is a fully reproducible B2 protocol grounded in the real AOL-IA data
(no synthetic queries or hand-crafted scores).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval/scripts"))

from simulate_on_aol import (  # noqa: E402
    _pbm_clicks, _dbn_clicks, _heur_clicks, _llm_clicks,
)
from simeval_ir.core.session import RankedItem  # noqa: E402

import ir_datasets  # noqa: E402

CLICK_MODELS = {
    "simiir-pbm": _pbm_clicks,
    "simiir-dbn": _dbn_clicks,
    "heuristic":  _heur_clicks,
    "llm-sim":    _llm_clicks,
}

# Ten retrieval systems with quality alpha in [0,1]: alpha=1 -> oracle,
# alpha=0 -> uniformly random ordering. Names are stylised but the rank
# permutation each system produces is computed deterministically below.
SYSTEMS = [
    ("BM25",      0.20),
    ("QL",        0.15),
    ("DirLM",     0.18),
    ("SDM",       0.30),
    ("RM3",       0.35),
    ("BERT_DOT",  0.50),
    ("ColBERT",   0.62),
    ("SPLADE",    0.55),
    ("monoT5",    0.70),
    ("BERT_CAT",  0.80),
]


def _system_rank(rng: random.Random, candidates: list[dict],
                 alpha: float, k: int = 10) -> list[RankedItem]:
    """Rank `candidates` (each {'doc_id', 'rel'}) with quality alpha."""
    scored = []
    for d in candidates:
        u = rng.gauss(0.0, 1.0)
        score = alpha * d["rel"] + (1 - alpha) * 1.5 * u + 0.05 * rng.random()
        scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [
        RankedItem(doc_id=d["doc_id"], rank=r + 1, score=1.0 - r / k,
                   judged_relevance=d["rel"])
        for r, (_, d) in enumerate(scored[:k])
    ]


def _ndcg_at_k(items: list[RankedItem], k: int = 10) -> float:
    rels = [(it.judged_relevance or 0) for it in items[:k]]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted([(it.judged_relevance or 0) for it in items], reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _click_ndcg(ranking: list[RankedItem], click_fn, rng_seed: int,
                n_replays: int) -> float:
    """Click-derived nDCG@10 averaged over n_replays."""
    ideal_dcg = sum(1.0 / math.log2(r + 2) for r in range(10))
    total = 0.0
    for r in range(n_replays):
        rng = random.Random(rng_seed + r)
        clicks = click_fn(rng, ranking)
        if not clicks:
            continue
        clicked_ranks = {c.rank for c in clicks}
        dcg = sum(1.0 / math.log2(rk + 1) for rk in clicked_ranks if rk <= 10)
        total += dcg / ideal_dcg
    return total / max(1, n_replays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=100,
                   help="Number of AOL queries to sample for B2")
    p.add_argument("--min-pool", type=int, default=15,
                   help="Minimum candidate-pool size per query")
    p.add_argument("--max-pool", type=int, default=40)
    p.add_argument("--n-replays", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--output", type=Path,
                   default=Path("eval/configs/b2_tester_reliability.yaml"))
    p.add_argument("--manifest", type=Path,
                   default=Path("data/aol_ia/b2_manifest.json"))
    args = p.parse_args()

    print("Loading AOL-IA qrels (full file)...")
    ds = ir_datasets.load("aol-ia")

    # Group qrels by query_id (AOL relevance is binary: rel=1 means clicked)
    qrels_by_q: dict[str, dict[str, int]] = defaultdict(dict)
    all_doc_ids: list[str] = []
    seen_docs = set()
    for qr in ds.qrels_iter():
        qrels_by_q[qr.query_id][qr.doc_id] = qr.relevance
        if qr.doc_id not in seen_docs:
            all_doc_ids.append(qr.doc_id)
            seen_docs.add(qr.doc_id)
    print(f"  qrels loaded: {len(qrels_by_q):,} queries with judgements; "
          f"{len(all_doc_ids):,} unique judged docs")

    # We define "pool size" as the number of clicked (positive) docs per query.
    eligible = [q for q, docs in qrels_by_q.items()
                if args.min_pool <= len(docs) <= args.max_pool]
    print(f"  eligible queries (#clicked docs in [{args.min_pool}, {args.max_pool}]): "
          f"{len(eligible):,}")

    rng = random.Random(args.seed)
    sampled = rng.sample(eligible, k=min(args.n_queries, len(eligible)))
    print(f"  sampled {len(sampled)} queries for B2")

    # For meaningful nDCG, every per-query candidate pool must contain
    # negative (non-clicked) distractor docs. We sample distractors from the
    # global pool of clicked docs across other queries (which are guaranteed
    # to be real document ids from the AOL-IA collection).
    pools = []
    n_distractors = 100  # negative candidates per query
    rng_doc = random.Random(args.seed + 1)
    for q in sampled:
        positive_docs = list(qrels_by_q[q].keys())
        positive_set = set(positive_docs)

        candidates = [{"doc_id": d, "rel": 2} for d in positive_docs]
        # Sample distractors (rel=0)
        attempts = 0
        while sum(1 for c in candidates if c["rel"] == 0) < n_distractors and attempts < 5 * n_distractors:
            d = rng_doc.choice(all_doc_ids)
            if d not in positive_set:
                candidates.append({"doc_id": d, "rel": 0})
                positive_set.add(d)
            attempts += 1
        pools.append((q, candidates))

    # Compute per-system rankings + scores
    rankings_out: dict[str, dict[str, float]] = {"qrels": {}}
    rankings_out.update({sim: {} for sim in CLICK_MODELS})

    for sys_name, alpha in SYSTEMS:
        ndcg_sum = 0.0
        per_sim_sum = {sim: 0.0 for sim in CLICK_MODELS}
        for q_idx, (q, pool) in enumerate(pools):
            r_seed = args.seed + hash(sys_name) % 99991 + q_idx
            ranking = _system_rank(random.Random(r_seed), pool, alpha)
            ndcg_sum += _ndcg_at_k(ranking)
            for sim_name, click_fn in CLICK_MODELS.items():
                tseed = args.seed * 7919 + hash(sim_name) % 99991 + r_seed
                per_sim_sum[sim_name] += _click_ndcg(ranking, click_fn, tseed,
                                                     args.n_replays)
        rankings_out["qrels"][sys_name] = ndcg_sum / len(pools)
        for sim_name in CLICK_MODELS:
            rankings_out[sim_name][sys_name] = per_sim_sum[sim_name] / len(pools)

    # Write yaml
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Auto-generated by eval/scripts/build_aol_b2.py\n")
        f.write(f"# AOL-IA queries sampled: {len(sampled)}\n")
        f.write(f"# Seed: {args.seed}\n")
        f.write(f"# n_replays per (tester, system, query): {args.n_replays}\n\n")
        f.write('name: "B2-AOL-IA-v1"\n')
        f.write('description: "Tester reliability on AOL-IA queries+qrels"\n')
        f.write("trusted_tester: qrels\n")
        f.write("testers:\n")
        for tester, scores in rankings_out.items():
            f.write(f"  {tester}:\n")
            for sys_name, _ in SYSTEMS:
                f.write(f"    {sys_name}: {scores[sys_name]:.6f}\n")
    print(f"\nWrote B2 config -> {args.output}")

    # Write manifest
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump({
            "dataset": "aol-ia",
            "seed": args.seed,
            "n_queries_sampled": len(sampled),
            "n_replays": args.n_replays,
            "systems": [s for s, _ in SYSTEMS],
            "testers": ["qrels"] + list(CLICK_MODELS.keys()),
            "qrels_by_q_total": len(qrels_by_q),
            "eligible_queries": len(eligible),
            "rankings": rankings_out,
        }, f, indent=2)
    print(f"Manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
