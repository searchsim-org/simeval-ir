#!/usr/bin/env python
"""Build a B2 (Tester Reliability) protocol from MIRACL/zh (Chinese).

MIRACL (Multilingual Information Retrieval Across a Continuum of
Languages, Zhang et al., 2023) is an openly licensed multilingual
passage-retrieval benchmark covering 18 languages including Chinese.
The Chinese dev split has 393 queries with 8--10 graded judgements each
(binary relevance 0/1) over Chinese Wikipedia passages.

This script samples a fixed number of MIRACL/zh queries, builds per-query
candidate pools from their judged docs (no need for distractor sampling
because every pool already contains both rel=1 and rel=0 docs), synthesises
the same 10 retrieval systems used in the AOL/TREC B2 protocols, and
computes the trusted qrels-nDCG@10 plus a click-derived nDCG@10 from each
simulator's click model.
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


def _system_rank(rng, candidates, alpha, k=10):
    scored = []
    for d in candidates:
        u = rng.gauss(0.0, 1.0)
        s = alpha * d["rel"] + (1 - alpha) * 1.5 * u + 0.05 * rng.random()
        scored.append((s, d))
    scored.sort(key=lambda x: -x[0])
    return [
        RankedItem(doc_id=d["doc_id"], rank=r + 1,
                   score=1.0 - r / k, judged_relevance=d["rel"])
        for r, (_, d) in enumerate(scored[:k])
    ]


def _ndcg_at_k(items, k=10):
    rels = [(it.judged_relevance or 0) for it in items[:k]]
    dcg = sum((2 ** max(0, r) - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted([(it.judged_relevance or 0) for it in items], reverse=True)[:k]
    idcg = sum((2 ** max(0, r) - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _click_ndcg(ranking, click_fn, rng_seed, n_replays):
    ideal_dcg = sum(1.0 / math.log2(r + 2) for r in range(10))
    total = 0.0
    for r in range(n_replays):
        rng = random.Random(rng_seed + r)
        clicks = click_fn(rng, ranking)
        if not clicks:
            continue
        clicked = {c.rank for c in clicks}
        dcg = sum(1.0 / math.log2(rk + 1) for rk in clicked if rk <= 10)
        total += dcg / ideal_dcg
    return total / max(1, n_replays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--min-pool", type=int, default=8)
    p.add_argument("--max-pool", type=int, default=15)
    p.add_argument("--n-replays", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--output", type=Path,
                   default=Path("eval/configs/b2_miracl_zh.yaml"))
    p.add_argument("--manifest", type=Path,
                   default=Path("data/miracl_zh/b2_manifest.json"))
    args = p.parse_args()

    print("Loading MIRACL/zh/dev...")
    ds = ir_datasets.load("miracl/zh/dev")
    qrels_by_q: dict[str, dict[str, int]] = defaultdict(dict)
    for qr in ds.qrels_iter():
        # MIRACL is binary; promote rel=1 to grade=2 for graded variation
        qrels_by_q[qr.query_id][qr.doc_id] = 2 if qr.relevance > 0 else 0
    print(f"  queries with qrels: {len(qrels_by_q)}")

    eligible = [q for q, docs in qrels_by_q.items()
                if args.min_pool <= len(docs) <= args.max_pool]
    print(f"  eligible queries (pool size in [{args.min_pool}, {args.max_pool}]): "
          f"{len(eligible)}")

    rng = random.Random(args.seed)
    sampled = rng.sample(eligible, k=min(args.n_queries, len(eligible)))
    print(f"  sampled {len(sampled)} queries")

    pools = []
    for q in sampled:
        cand = [{"doc_id": d, "rel": r} for d, r in qrels_by_q[q].items()]
        pools.append((q, cand))

    rankings: dict[str, dict[str, float]] = {"qrels": {}}
    rankings.update({sim: {} for sim in CLICK_MODELS})

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
        rankings["qrels"][sys_name] = ndcg_sum / len(pools)
        for sim_name in CLICK_MODELS:
            rankings[sim_name][sys_name] = per_sim_sum[sim_name] / len(pools)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Auto-generated by eval/scripts/build_miracl_zh_b2.py\n")
        f.write(f"# MIRACL/zh/dev queries sampled: {len(sampled)}; seed={args.seed}\n")
        f.write(f"# n_replays per (tester, system, query): {args.n_replays}\n\n")
        f.write('name: "B2-MIRACL-zh-v1"\n')
        f.write('description: "Tester reliability on Chinese MIRACL/zh/dev qrels"\n')
        f.write("trusted_tester: qrels\n")
        f.write("testers:\n")
        for tester, scores in rankings.items():
            f.write(f"  {tester}:\n")
            for sys_name, _ in SYSTEMS:
                f.write(f"    {sys_name}: {scores[sys_name]:.6f}\n")
    print(f"  wrote {args.output}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump({
            "dataset": "miracl/zh/dev",
            "language": "zh",
            "seed": args.seed,
            "n_queries_sampled": len(sampled),
            "n_replays": args.n_replays,
            "systems": [s for s, _ in SYSTEMS],
            "testers": ["qrels"] + list(CLICK_MODELS.keys()),
            "rankings": rankings,
        }, f, indent=2)
    print(f"  manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
