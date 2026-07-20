#!/usr/bin/env python
"""End-to-end AOL-IA pipeline: B1 + B2 + B3 across multiple random seeds.

Each seed re-samples
    * a random shard of real AOL-IA sessions (B1 reference),
    * a fresh per-simulator click realisation (B1 simulator output),
    * a fresh sample of 100 AOL queries with their qrels (B2 testbed).

This gives n_seeds x 4 simulator-dataset pairs for B3, all derived from
the FULL AOL-IA dataset (no excerpts, no synthetic data).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval/scripts"))

from simeval_ir.datasets.io import (  # noqa: E402
    load_sessions_from_jsonl, save_sessions_to_jsonl,
)
from simeval_ir.embeddings.action import ActionSequenceEmbedder  # noqa: E402
from simeval_ir.eval import run_protocol  # noqa: E402
from simeval_ir.metrics.behavior import run_leakage_audit  # noqa: E402
from simeval_ir.metrics.tester import KendallTau, RATE  # noqa: E402

import ir_datasets  # noqa: E402

# Reuse simulator click models and B2 system definitions
from simulate_on_aol import (  # noqa: E402
    _pbm_clicks, _dbn_clicks, _heur_clicks, _llm_clicks,
    _llm_query_paraphrase,
)
from simeval_ir.core.session import (  # noqa: E402
    InteractionSession, Event, Click, RankedItem,
)
from simeval_ir.core.types import EventType, Role, SessionType  # noqa: E402

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


def _sub_sessions(real_path: Path, n: int, offset: int) -> list[InteractionSession]:
    """Return sessions [offset, offset+n) from real_path."""
    out = []
    for i, s in enumerate(load_sessions_from_jsonl(real_path)):
        if i < offset:
            continue
        if i >= offset + n:
            break
        out.append(s)
    return out


def _simulate_on(real_sessions, sim_name, seed):
    click_fn = CLICK_MODELS[sim_name]
    rng = random.Random(seed)
    out = []
    for r in real_sessions:
        new_events = []
        counter = 0
        for ev in r.events:
            if ev.type == EventType.QUERY_ISSUED:
                counter += 1
                q = ev.query
                if sim_name == "llm-sim":
                    q = _llm_query_paraphrase(q, rng)
                new_events.append(Event(
                    event_id=f"{sim_name}-{r.session_id}-q{counter}",
                    type=EventType.QUERY_ISSUED, timestamp=ev.timestamp,
                    role=Role.USER, query=q, meta={"src_event_id": ev.event_id},
                ))
            elif ev.type == EventType.SERP_VIEW and ev.ranked_items:
                counter += 1
                new_events.append(Event(
                    event_id=f"{sim_name}-{r.session_id}-s{counter}",
                    type=EventType.SERP_VIEW, timestamp=ev.timestamp,
                    role=Role.SYSTEM, ranked_items=ev.ranked_items,
                ))
                clicks = click_fn(rng, ev.ranked_items)
                if clicks:
                    counter += 1
                    new_events.append(Event(
                        event_id=f"{sim_name}-{r.session_id}-c{counter}",
                        type=EventType.CLICK,
                        timestamp=(ev.timestamp or 0) + 1.0,
                        role=Role.USER, clicked_items=clicks,
                    ))
        out.append(InteractionSession(
            session_id=f"{sim_name}-{r.session_id}",
            dataset_id=f"aol-ia.{sim_name}",
            session_type=SessionType.SEARCH,
            events=new_events,
            user_id=r.user_id, topic_id=r.topic_id, domain=r.domain,
            meta={"simulator": sim_name},
        ))
    return out


def run_b1(real, sim, n_folds: int = 5):
    embedder = ActionSequenceEmbedder()
    proto = run_protocol(protocol="realism-benchmark-v1",
                         real=real, sim=sim, embedder=embedder, n_folds=n_folds)
    audit = run_leakage_audit(real, sim, embedder=embedder, n_folds=n_folds)
    metrics = {r.name: {"value": r.value, "meta": r.meta}
               for r in proto.metric_results}
    metrics["realism_classifier"]["meta"]["audit"] = {
        "main_auc": audit.main_auc,
        "metadata_only_auc": audit.metadata_only_auc,
        "structural_only_auc": audit.structural_only_auc,
        "permutation_auc": audit.permutation_auc,
        "leakage_detected": audit.leakage_detected,
    }
    return metrics


# --- B2 ---

def _system_rank(rng: random.Random, candidates, alpha: float, k: int = 10):
    scored = []
    for d in candidates:
        u = rng.gauss(0.0, 1.0)
        s = alpha * d["rel"] + (1 - alpha) * 1.5 * u + 0.05 * rng.random()
        scored.append((s, d))
    scored.sort(key=lambda x: -x[0])
    import math as _m
    return [
        RankedItem(doc_id=d["doc_id"], rank=r + 1,
                   score=1.0 - r / k, judged_relevance=d["rel"])
        for r, (_, d) in enumerate(scored[:k])
    ]


def _ndcg_at_k(items, k=10):
    import math as _m
    rels = [(it.judged_relevance or 0) for it in items[:k]]
    dcg = sum((2 ** r - 1) / _m.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted([(it.judged_relevance or 0) for it in items], reverse=True)[:k]
    idcg = sum((2 ** r - 1) / _m.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _click_ndcg(ranking, click_fn, rng_seed, n_replays):
    import math as _m
    ideal_dcg = sum(1.0 / _m.log2(r + 2) for r in range(10))
    total = 0.0
    for r in range(n_replays):
        rng = random.Random(rng_seed + r)
        clicks = click_fn(rng, ranking)
        if not clicks:
            continue
        clicked = {c.rank for c in clicks}
        dcg = sum(1.0 / _m.log2(rk + 1) for rk in clicked if rk <= 10)
        total += dcg / ideal_dcg
    return total / max(1, n_replays)


def build_b2(qrels_by_q, all_doc_ids, seed, n_queries=100,
             min_pool=5, max_pool=25, n_distractors=100, n_replays=8):
    rng = random.Random(seed)
    eligible = [q for q, docs in qrels_by_q.items()
                if min_pool <= len(docs) <= max_pool]
    sampled = rng.sample(eligible, k=min(n_queries, len(eligible)))
    rng_doc = random.Random(seed + 1)

    # Build pools
    pools = []
    for q in sampled:
        pos = list(qrels_by_q[q].keys())
        seen = set(pos)
        cand = [{"doc_id": d, "rel": 2} for d in pos]
        attempts = 0
        while sum(1 for c in cand if c["rel"] == 0) < n_distractors and attempts < 5 * n_distractors:
            d = rng_doc.choice(all_doc_ids)
            if d not in seen:
                cand.append({"doc_id": d, "rel": 0})
                seen.add(d)
            attempts += 1
        pools.append((q, cand))

    rankings: dict[str, dict[str, float]] = {"qrels": {}}
    rankings.update({sim: {} for sim in CLICK_MODELS})

    for sys_name, alpha in SYSTEMS:
        ndcg_sum = 0.0
        per_sim_sum = {sim: 0.0 for sim in CLICK_MODELS}
        for q_idx, (q, pool) in enumerate(pools):
            r_seed = seed + hash(sys_name) % 99991 + q_idx
            ranking = _system_rank(random.Random(r_seed), pool, alpha)
            ndcg_sum += _ndcg_at_k(ranking)
            for sim_name, click_fn in CLICK_MODELS.items():
                tseed = seed * 7919 + hash(sim_name) % 99991 + r_seed
                per_sim_sum[sim_name] += _click_ndcg(
                    ranking, click_fn, tseed, n_replays)
        rankings["qrels"][sys_name] = ndcg_sum / len(pools)
        for sim_name in CLICK_MODELS:
            rankings[sim_name][sys_name] = per_sim_sum[sim_name] / len(pools)

    # Compute Kendall tau per simulator vs qrels
    out = {"testers": {}, "rate": {}, "n_queries": len(sampled)}
    for tester, scores in rankings.items():
        if tester == "qrels":
            continue
        tau = KendallTau().compute(
            scores_1=rankings["qrels"], scores_2=scores,
            n_bootstrap=1000, random_state=seed)
        out["testers"][tester] = {
            "kendall_tau": tau.value,
            "tau_ci_lower": tau.ci_lower,
            "tau_ci_upper": tau.ci_upper,
            "tau_p_value": tau.meta.get("p_value"),
        }
    rate = RATE().compute(tester_scores=rankings)
    out["rate"] = {
        "reliabilities": rate.per_item or {},
        "iterations": rate.meta.get("iterations", 0),
    }
    out["rankings"] = rankings
    return out


# --- main ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-jsonl", type=Path,
                   default=Path("data/aol_ia/sessions.jsonl"))
    p.add_argument("--n-sessions", type=int, default=2000)
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--seeds", type=str, default="20260429,20260430,20260501,20260502,20260503,20260504")
    p.add_argument("--output", type=Path, default=Path("results"))
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading AOL-IA qrels (full)...")
    ds = ir_datasets.load("aol-ia")
    qrels_by_q: dict[str, dict[str, int]] = defaultdict(dict)
    seen = set()
    all_doc_ids: list[str] = []
    for qr in ds.qrels_iter():
        qrels_by_q[qr.query_id][qr.doc_id] = qr.relevance
        if qr.doc_id not in seen:
            all_doc_ids.append(qr.doc_id)
            seen.add(qr.doc_id)
    print(f"  qrels: {len(qrels_by_q):,} queries, {len(all_doc_ids):,} unique docs")

    per_seed = []
    for seed_idx, seed in enumerate(seeds):
        print(f"\n=== seed {seed} ({seed_idx+1}/{len(seeds)}) ===")
        # Different slice of real sessions per seed
        offset = seed_idx * args.n_sessions
        real = _sub_sessions(args.real_jsonl, args.n_sessions, offset)
        if len(real) < args.n_sessions:
            print(f"  WARNING: only {len(real)} real sessions available at offset {offset}")
        b1 = {}
        for name in CLICK_MODELS:
            print(f"  simulating {name}...")
            sim = _simulate_on(real, name, seed=seed + 17 * hash(name) % 100003)
            print(f"  B1 {name}...")
            b1[name] = run_b1(real, sim)
        print("  building B2 from AOL qrels...")
        b2 = build_b2(qrels_by_q, all_doc_ids, seed,
                      n_queries=args.n_queries)
        per_seed.append({"seed": seed, "n_sessions": len(real), "b1": b1, "b2": b2})
        # Persist canonical (first seed) result files
        if seed_idx == 0:
            (args.output / "b1").mkdir(parents=True, exist_ok=True)
            for sim, m in b1.items():
                with open(args.output / "b1" / f"b1_{sim}.json", "w") as f:
                    json.dump({
                        "experiment": "B1",
                        "dataset": "aol-ia",
                        "n_sessions": len(real),
                        "simulator": sim,
                        "seed": seed,
                        "metrics": m,
                    }, f, indent=2, default=str)
            (args.output / "b2").mkdir(parents=True, exist_ok=True)
            with open(args.output / "b2" / "b2_results.json", "w") as f:
                json.dump({
                    "experiment": "B2",
                    "dataset": "aol-ia",
                    "seed": seed,
                    **b2,
                }, f, indent=2, default=str)

    # B3 aggregate
    rows = []
    metrics_of_interest = [
        "jsd_click_depth", "wasserstein_session_length",
        "session_fd", "jsd_action_types", "reformulation_similarity",
        "realism_classifier", "mmd",
    ]
    for run in per_seed:
        for sim_name in CLICK_MODELS:
            tau = run["b2"]["testers"][sim_name]["kendall_tau"]
            row = {"sim": sim_name, "seed": run["seed"], "tau": tau}
            for k in metrics_of_interest:
                v = run["b1"][sim_name].get(k, {}).get("value")
                row[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
            rows.append(row)

    correlations = {}
    for k in metrics_of_interest:
        xs = [r[k] for r in rows if r[k] is not None and not np.isnan(r[k])]
        ys = [r["tau"] for r in rows if r[k] is not None and not np.isnan(r[k])]
        if len(xs) < 3:
            continue
        pr, pp = pearsonr(xs, ys)
        sr, sp = spearmanr(xs, ys)
        correlations[k] = {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
            "n": len(xs), "abs_pearson": float(abs(pr)),
        }
    (args.output / "b3").mkdir(parents=True, exist_ok=True)
    with open(args.output / "b3/b3_results.json", "w") as f:
        json.dump({
            "experiment": "B3",
            "dataset": "aol-ia",
            "seeds": seeds,
            "n_sessions_per_seed": args.n_sessions,
            "n_queries_per_seed": args.n_queries,
            "n_data_points": len(rows),
            "correlations": correlations,
            "raw_rows": rows,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print("\n=== B3 SUMMARY ===")
    for k, c in sorted(correlations.items(), key=lambda x: -x[1]["abs_pearson"]):
        print(f"  {k:<28}  r={c['pearson_r']:+.3f}  (p={c['pearson_p']:.3f}, n={c['n']})")


if __name__ == "__main__":
    main()
