#!/usr/bin/env python
"""Build a B2 (Tester Reliability) protocol from TREC Session Track 2014.

Source: NIST TREC Session 2014 judgments.txt (graded relevance, scale -2..4).
We sample N topics that have a sufficient number of judged docs, build the
candidate pool from the judged docs (preserving graded relevance), and
synthesise 10 retrieval systems by ranking the pool with quality $\\alpha$.
Each tester replays its click model on each system's SERP; the trusted
tester is qrels-nDCG@10 computed directly from the graded relevance.

Output: a YAML config that ``run_b2.py`` can consume.
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


def _system_rank(rng: random.Random, candidates, alpha: float, k: int = 10):
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


def _load_qrels(path: Path) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                topic, doc, rel = int(parts[0]), parts[2], int(parts[3])
            except (ValueError, IndexError):
                continue
            # Treat NIST -2 (junk/spam) as 0
            out[topic][doc] = max(0, rel)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--n-topics", type=int, default=40,
                   help="How many topics to sample (max ~51 for TREC S14)")
    p.add_argument("--min-judged", type=int, default=80)
    p.add_argument("--max-judged", type=int, default=1000)
    p.add_argument("--n-replays", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--output", type=Path,
                   default=Path("eval/configs/b2_trec_s14.yaml"))
    p.add_argument("--manifest", type=Path,
                   default=Path("data/trec_session2014/b2_manifest.json"))
    args = p.parse_args()

    print(f"Loading qrels: {args.qrels}")
    qrels = _load_qrels(args.qrels)
    print(f"  {len(qrels)} topics; total judgements: {sum(len(d) for d in qrels.values())}")

    eligible = [(t, d) for t, d in qrels.items()
                if args.min_judged <= len(d) <= args.max_judged]
    print(f"  eligible topics with [{args.min_judged}, {args.max_judged}] judgements: "
          f"{len(eligible)}")

    rng = random.Random(args.seed)
    sampled = rng.sample(eligible, k=min(args.n_topics, len(eligible)))
    print(f"  sampled {len(sampled)} topics")

    # candidate pools (already include both rel>0 and rel=0 docs from qrels)
    pools = []
    for topic, judged in sampled:
        cand = [{"doc_id": d, "rel": r} for d, r in judged.items()]
        # We have plenty of rel=0 docs in TREC qrels; no need to add distractors
        pools.append((topic, cand))

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
                per_sim_sum[sim_name] += _click_ndcg(
                    ranking, click_fn, tseed, args.n_replays)
        rankings["qrels"][sys_name] = ndcg_sum / len(pools)
        for sim_name in CLICK_MODELS:
            rankings[sim_name][sys_name] = per_sim_sum[sim_name] / len(pools)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Auto-generated by eval/scripts/build_trec_s14_b2.py\n")
        f.write(f"# topics sampled: {len(sampled)}; seed={args.seed}\n")
        f.write(f"# n_replays per (tester, system, topic): {args.n_replays}\n\n")
        f.write('name: "B2-TREC-Session-2014-v1"\n')
        f.write('description: "Tester reliability on TREC Session Track 2014 qrels"\n')
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
            "dataset": "trec-session-2014",
            "qrels_path": str(args.qrels),
            "seed": args.seed,
            "n_topics_sampled": len(sampled),
            "topics": [t for t, _ in sampled],
            "n_replays": args.n_replays,
            "systems": [s for s, _ in SYSTEMS],
            "testers": ["qrels"] + list(CLICK_MODELS.keys()),
            "rankings": rankings,
        }, f, indent=2)
    print(f"  manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
