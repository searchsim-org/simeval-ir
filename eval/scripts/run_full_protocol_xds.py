#!/usr/bin/env python
"""Cross-dataset SimEval-IR protocol runner.

For each (dataset, seed) pair, run:
    * B1 on a shard of real sessions vs each simulator's clicks-on-the-same-SERPs
    * B2 on a fresh sample of qrels-eligible queries with 10 synthesised systems
    * Collect per-(simulator, dataset, seed) data points for B3 aggregation

Datasets supported:
    aol-ia            -- via ir_datasets, 36.4M qlog entries
    trec-session-2014 -- via NIST sessiontrack2014.xml + judgments.txt

Each (dataset, seed) contributes 4 simulator-dataset-seed rows. With
``--seeds-per-dataset 6`` this gives 4 * 2 * 6 = 48 rows for B3.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval/scripts"))

from simeval_ir.datasets.io import load_sessions_from_jsonl  # noqa: E402
from simeval_ir.embeddings.action import ActionSequenceEmbedder  # noqa: E402
from simeval_ir.eval import run_protocol  # noqa: E402
from simeval_ir.metrics.behavior import run_leakage_audit  # noqa: E402
from simeval_ir.metrics.tester import KendallTau, RATE  # noqa: E402

import ir_datasets  # noqa: E402

from simulate_on_aol import (  # noqa: E402
    _pbm_clicks, _dbn_clicks, _heur_clicks, _llm_clicks,
    _llm_query_paraphrase,
)
from simeval_ir.core.session import (  # noqa: E402
    InteractionSession, Event, RankedItem,
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


# ----------- session sub-sampling ---------------------------------------- #

def _sub_sessions(real_path: Path, n: int, offset: int) -> list[InteractionSession]:
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
            dataset_id=f"{r.dataset_id}.{sim_name}",
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


# ----------- B2 helpers -------------------------------------------------- #

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


def build_b2_aol(qrels_by_q, all_doc_ids, seed,
                 n_queries=100, min_pool=5, max_pool=25,
                 n_distractors=100, n_replays=8):
    rng = random.Random(seed)
    eligible = [q for q, docs in qrels_by_q.items()
                if min_pool <= len(docs) <= max_pool]
    sampled = rng.sample(eligible, k=min(n_queries, len(eligible)))
    rng_doc = random.Random(seed + 1)

    pools = []
    for q in sampled:
        pos = list(qrels_by_q[q].keys())
        seen = set(pos)
        # AOL grades all clicked docs as 2 to introduce variation; distractors are 0
        cand = [{"doc_id": d, "rel": 2} for d in pos]
        attempts = 0
        while sum(1 for c in cand if c["rel"] == 0) < n_distractors and attempts < 5 * n_distractors:
            d = rng_doc.choice(all_doc_ids)
            if d not in seen:
                cand.append({"doc_id": d, "rel": 0})
                seen.add(d)
            attempts += 1
        pools.append((q, cand))

    return _b2_score(pools, seed, n_replays)


def build_b2_trec_s14(qrels_by_topic, seed,
                      n_topics=40, min_judged=80, max_judged=1000,
                      n_replays=8):
    rng = random.Random(seed)
    eligible = [(t, d) for t, d in qrels_by_topic.items()
                if min_judged <= len(d) <= max_judged]
    sampled = rng.sample(eligible, k=min(n_topics, len(eligible)))
    pools = []
    for topic, judged in sampled:
        cand = [{"doc_id": d, "rel": r} for d, r in judged.items()]
        pools.append((topic, cand))
    return _b2_score(pools, seed, n_replays)


def _b2_score(pools, seed, n_replays):
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

    out = {"testers": {}, "rate": {}, "n_queries": len(pools)}
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


# ----------- main -------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default="trec-s14,aol-ia",
                   help="Comma-separated dataset list")
    p.add_argument("--aol-real-jsonl", type=Path,
                   default=Path("data/aol_ia/sessions.jsonl"))
    p.add_argument("--trec-s14-jsonl", type=Path,
                   default=Path("data/trec_session2014/sessions.jsonl"))
    p.add_argument("--trec-s14-qrels", type=Path,
                   default=Path("data/trec_session2014/judgments.txt"))
    p.add_argument("--n-sessions-aol", type=int, default=2000)
    p.add_argument("--n-queries-aol", type=int, default=100)
    p.add_argument("--n-topics-trec", type=int, default=40)
    p.add_argument("--seeds", type=str,
                   default="20260429,20260430,20260501,20260502,20260503,20260504")
    p.add_argument("--output", type=Path, default=Path("results_xds"))
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    aol_qrels = None
    aol_doc_ids = None
    trec_qrels = None

    if "aol-ia" in datasets:
        print("Loading AOL-IA qrels...")
        ds = ir_datasets.load("aol-ia")
        aol_qrels = defaultdict(dict)
        seen = set()
        aol_doc_ids = []
        for qr in ds.qrels_iter():
            aol_qrels[qr.query_id][qr.doc_id] = qr.relevance
            if qr.doc_id not in seen:
                aol_doc_ids.append(qr.doc_id)
                seen.add(qr.doc_id)
        print(f"  {len(aol_qrels):,} queries; {len(aol_doc_ids):,} unique docs")

    if "trec-s14" in datasets:
        print("Loading TREC Session 2014 qrels...")
        trec_qrels = defaultdict(dict)
        with open(args.trec_s14_qrels) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    trec_qrels[int(parts[0])][parts[2]] = max(0, int(parts[3]))
        print(f"  topics: {len(trec_qrels)}; total: {sum(len(v) for v in trec_qrels.values())}")

    rows = []
    per_run = []

    for ds_name in datasets:
        for seed_idx, seed in enumerate(seeds):
            print(f"\n=== {ds_name}, seed {seed} ({seed_idx+1}/{len(seeds)}) ===")
            if ds_name == "aol-ia":
                offset = seed_idx * args.n_sessions_aol
                real = _sub_sessions(args.aol_real_jsonl,
                                     args.n_sessions_aol, offset)
                if not real:
                    print(f"  WARN: no sessions at offset {offset}")
                    continue
                b1 = {}
                for name in CLICK_MODELS:
                    print(f"  simulating {name}...")
                    sim = _simulate_on(real, name,
                                       seed=seed + 17 * hash(name) % 100003)
                    print(f"  B1 {name}...")
                    b1[name] = run_b1(real, sim)
                print("  building B2 (AOL)...")
                b2 = build_b2_aol(aol_qrels, aol_doc_ids, seed,
                                  n_queries=args.n_queries_aol)
            elif ds_name == "trec-s14":
                # subsample sessions: split TREC's 1257 sessions into 6 disjoint shards
                all_sessions = list(load_sessions_from_jsonl(args.trec_s14_jsonl))
                shard = 1257 // max(1, len(seeds))
                start = seed_idx * shard
                end = start + shard
                real = all_sessions[start:end]
                if not real:
                    print(f"  WARN: empty shard")
                    continue
                b1 = {}
                for name in CLICK_MODELS:
                    print(f"  simulating {name} ({len(real)} sessions)...")
                    sim = _simulate_on(real, name,
                                       seed=seed + 17 * hash(name) % 100003)
                    print(f"  B1 {name}...")
                    b1[name] = run_b1(real, sim)
                print("  building B2 (TREC S14)...")
                b2 = build_b2_trec_s14(trec_qrels, seed,
                                       n_topics=args.n_topics_trec)
            else:
                continue

            per_run.append({"dataset": ds_name, "seed": seed,
                            "n_real_sessions": len(real),
                            "b1": b1, "b2": b2})

            for sim_name in CLICK_MODELS:
                tau = b2["testers"][sim_name]["kendall_tau"]
                row = {"dataset": ds_name, "sim": sim_name, "seed": seed,
                       "tau": tau,
                       "n_real_sessions": len(real),
                       "n_b2_queries": b2["n_queries"]}
                for k in ("jsd_click_depth", "wasserstein_session_length",
                          "session_fd", "jsd_action_types",
                          "reformulation_similarity", "realism_classifier",
                          "mmd"):
                    v = b1[sim_name].get(k, {}).get("value")
                    row[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
                rows.append(row)

    # B3 aggregation
    metrics_of_interest = [
        "jsd_click_depth", "wasserstein_session_length",
        "session_fd", "jsd_action_types", "reformulation_similarity",
        "realism_classifier", "mmd",
    ]
    correlations = {}
    for k in metrics_of_interest:
        xs = [r[k] for r in rows if r.get(k) is not None]
        ys = [r["tau"] for r in rows if r.get(k) is not None]
        if len(xs) < 3:
            continue
        pr, pp = pearsonr(xs, ys)
        sr, sp = spearmanr(xs, ys)
        correlations[k] = {"pearson_r": float(pr), "pearson_p": float(pp),
                           "spearman_rho": float(sr), "spearman_p": float(sp),
                           "n": len(xs), "abs_pearson": float(abs(pr))}

    # Per-dataset correlations
    per_dataset = {}
    for ds_name in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds_name]
        ds_corr = {}
        for k in metrics_of_interest:
            xs = [r[k] for r in ds_rows if r.get(k) is not None]
            ys = [r["tau"] for r in ds_rows if r.get(k) is not None]
            if len(xs) < 3:
                continue
            pr, pp = pearsonr(xs, ys)
            ds_corr[k] = {"pearson_r": float(pr), "pearson_p": float(pp),
                          "n": len(xs), "abs_pearson": float(abs(pr))}
        per_dataset[ds_name] = ds_corr

    out_dir = args.output / "b3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "b3_results.json", "w") as f:
        json.dump({
            "experiment": "B3 cross-dataset",
            "datasets": datasets,
            "seeds": seeds,
            "n_data_points": len(rows),
            "correlations_pooled": correlations,
            "correlations_per_dataset": per_dataset,
            "raw_rows": rows,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    # Persist canonical seed B1/B2 per dataset (first seed)
    for ds_name in datasets:
        canonical = next((r for r in per_run if r["dataset"] == ds_name), None)
        if not canonical:
            continue
        d_dir = args.output / ds_name.replace("/", "_")
        (d_dir / "b1").mkdir(parents=True, exist_ok=True)
        for sim, m in canonical["b1"].items():
            with open(d_dir / "b1" / f"b1_{sim}.json", "w") as f:
                json.dump({"experiment": "B1", "dataset": ds_name,
                           "seed": canonical["seed"],
                           "n_sessions": canonical["n_real_sessions"],
                           "simulator": sim, "metrics": m}, f, indent=2, default=str)
        (d_dir / "b2").mkdir(parents=True, exist_ok=True)
        with open(d_dir / "b2" / "b2_results.json", "w") as f:
            json.dump({"experiment": "B2", "dataset": ds_name,
                       "seed": canonical["seed"],
                       **canonical["b2"]}, f, indent=2, default=str)

    print("\n=== B3 (POOLED across datasets) ===")
    for k, c in sorted(correlations.items(), key=lambda x: -x[1]["abs_pearson"]):
        print(f"  {k:<28} r={c['pearson_r']:+.3f}  (p={c['pearson_p']:.3f}, n={c['n']})")
    for ds_name, ds_c in per_dataset.items():
        print(f"\n=== B3 ({ds_name}) ===")
        for k, c in sorted(ds_c.items(), key=lambda x: -x[1]["abs_pearson"]):
            print(f"  {k:<28} r={c['pearson_r']:+.3f}  (p={c['pearson_p']:.3f}, n={c['n']})")


if __name__ == "__main__":
    main()
