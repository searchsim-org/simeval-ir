#!/usr/bin/env python
"""B2 (tester reliability) for the real-LLM simulator outputs.

The B2 protocol scores N retrieval systems by replaying each tester's clicks
on each system's SERP. The five existing testers (qrels, simiir-pbm,
simiir-dbn, heuristic, llm-sim) carry a fixed click model. To evaluate the
real-LLM simulator as a B2 tester without spending O(systems * queries *
replays) extra API calls, we fit a per-(rank, relevance-bin) click profile
from the B1 LLM-real sessions and use that as the tester's click function.
This is faithful to what the LLM actually clicked under realistic SERPs and
keeps B2 evaluation tractable.

Output is written into ``results/<dataset>/b2/b2_results.json``, merging
new testers (``llm-gpt5nano``, ``llm-qwen3next``) with the testers that
already exist in that file. The trusted reference (``qrels``) is left
intact.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from simeval_ir.core.session import Click, RankedItem  # noqa: E402
from simeval_ir.core.types import EventType  # noqa: E402
from simeval_ir.datasets.io import load_sessions_from_jsonl  # noqa: E402
from simeval_ir.metrics.tester import KendallTau, RATE  # noqa: E402

import ir_datasets  # noqa: E402


SYSTEMS = [
    ("BM25", 0.20), ("QL", 0.15), ("DirLM", 0.18), ("SDM", 0.30),
    ("RM3", 0.35), ("BERT_DOT", 0.50), ("ColBERT", 0.62),
    ("SPLADE", 0.55), ("monoT5", 0.70), ("BERT_CAT", 0.80),
]


def fit_click_profile(sim_path: Path) -> dict:
    """Empirical P(click | rank, rel_bin) from an LLM-real JSONL.

    rel_bin is ``"rel"`` if the SERP item has judged_relevance > 0,
    else ``"nonrel"``. Returns a profile usable by ``make_click_fn``.
    """
    counts = defaultdict(lambda: [0, 0])  # (rank, bin) -> [clicks, impressions]
    dwells = []
    for s in load_sessions_from_jsonl(sim_path):
        last_serp = None
        for ev in s.events:
            if ev.type == EventType.SERP_VIEW and ev.ranked_items:
                last_serp = ev.ranked_items
                for it in ev.ranked_items[:10]:
                    rb = "rel" if (it.judged_relevance or 0) > 0 else "nonrel"
                    counts[(it.rank, rb)][1] += 1
            elif ev.type == EventType.CLICK and last_serp is not None:
                rank_to_rel = {it.rank: (it.judged_relevance or 0)
                               for it in last_serp}
                for c in (ev.clicked_items or []):
                    rb = "rel" if rank_to_rel.get(c.rank, 0) > 0 else "nonrel"
                    counts[(c.rank, rb)][0] += 1
                    if c.dwell_time:
                        dwells.append(c.dwell_time)
                last_serp = None  # don't double-count
    p = {}
    for (rank, rb), (clicks, imps) in counts.items():
        p[(rank, rb)] = (clicks + 0.5) / (imps + 1.0)
    return {"p_click": p,
            "median_dwell": (sorted(dwells)[len(dwells)//2]
                             if dwells else 25.0)}


def make_click_fn(profile: dict):
    p_click = profile["p_click"]
    median_dwell = profile["median_dwell"]

    def _fn(rng: random.Random, serp: list[RankedItem]):
        out = []
        for it in serp[:10]:
            rb = "rel" if (it.judged_relevance or 0) > 0 else "nonrel"
            p = p_click.get((it.rank, rb), p_click.get((it.rank, "nonrel"), 0.0))
            if rng.random() < p:
                out.append(Click(doc_id=it.doc_id, rank=it.rank,
                                 dwell_time=max(3.0,
                                                rng.gauss(median_dwell, 4.0))))
        return out
    return _fn


# ------- Existing tester click models (mirrored from simulate_on_aol.py) -- #

PBM_EXAM = [0.94, 0.84, 0.71, 0.55, 0.43, 0.32, 0.24, 0.18, 0.13, 0.10]
PBM_ATTR = [0.05, 0.55]
DBN_ATTR = [0.08, 0.65]
DBN_SAT = [0.05, 0.55]
DBN_GAMMA = 0.20
HEUR_PROB = [0.45, 0.27, 0.16, 0.10, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01]


def pbm(rng, serp):
    out = []
    for it in serp:
        rel = it.judged_relevance or 0
        e = PBM_EXAM[min(it.rank - 1, 9)]
        a = PBM_ATTR[min(rel, 1)]
        if rng.random() < e * a:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(3.0, rng.gauss(20 + 8 * rel, 6))))
    return out


def dbn(rng, serp):
    out = []
    for it in serp:
        rel = it.judged_relevance or 0
        if rng.random() < DBN_ATTR[min(rel, 1)]:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(3.0, rng.gauss(25 + 10 * rel, 7))))
            if rng.random() < DBN_SAT[min(rel, 1)] and rng.random() > DBN_GAMMA:
                break
    return out


def heur(rng, serp):
    out = []
    for it in serp:
        if rng.random() < HEUR_PROB[min(it.rank - 1, 9)]:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(3.0, rng.uniform(5, 25))))
    return out


def llm_sim(rng, serp):
    # Verbose LLM-style: examines every rank, click probability scales with
    # relevance but with a high baseline ('over-clicker'). Same parameters
    # as eval/scripts/simulate_on_aol.py:_llm_clicks.
    out = []
    for it in serp:
        rel = it.judged_relevance or 0
        e = max(0.30, 0.92 - 0.06 * (it.rank - 1))
        a = 0.18 + 0.40 * rel
        if rng.random() < e * a:
            out.append(Click(doc_id=it.doc_id, rank=it.rank,
                             dwell_time=max(8.0, rng.gauss(40 + 15 * rel, 12))))
    return out


# ------- B2 score core --------------------------------------------------- #

def _system_rank(rng, candidates, alpha, k=10):
    scored = []
    for d in candidates:
        u = rng.gauss(0.0, 1.0)
        s = alpha * d["rel"] + (1 - alpha) * 1.5 * u + 0.05 * rng.random()
        scored.append((s, d))
    scored.sort(key=lambda x: -x[0])
    return [
        RankedItem(doc_id=d["doc_id"], rank=r+1, score=1.0 - r/k,
                   judged_relevance=d["rel"])
        for r, (_, d) in enumerate(scored[:k])
    ]


def _ndcg(items, k=10):
    rels = [(it.judged_relevance or 0) for it in items[:k]]
    dcg = sum((2**max(0, r) - 1)/math.log2(i+2) for i, r in enumerate(rels))
    ideal = sorted([(it.judged_relevance or 0) for it in items],
                   reverse=True)[:k]
    idcg = sum((2**max(0, r) - 1)/math.log2(i+2) for i, r in enumerate(ideal))
    return dcg/idcg if idcg > 0 else 0.0


def _click_ndcg(ranking, click_fn, rng_seed, n_replays):
    ideal_dcg = sum(1.0/math.log2(r+2) for r in range(10))
    total = 0.0
    for r in range(n_replays):
        rng = random.Random(rng_seed + r)
        clicks = click_fn(rng, ranking)
        if not clicks:
            continue
        clicked = {c.rank for c in clicks}
        dcg = sum(1.0/math.log2(rk+1) for rk in clicked if rk <= 10)
        total += dcg/ideal_dcg
    return total/max(1, n_replays)


def _b2_score(pools, click_fns, seed, n_replays):
    rankings = {"qrels": {}}
    rankings.update({name: {} for name in click_fns})
    for sys_name, alpha in SYSTEMS:
        ndcg_sum = 0.0
        per_t = {name: 0.0 for name in click_fns}
        for q_idx, (q, pool) in enumerate(pools):
            rseed = seed + hash(sys_name) % 99991 + q_idx
            ranking = _system_rank(random.Random(rseed), pool, alpha)
            ndcg_sum += _ndcg(ranking)
            for tname, fn in click_fns.items():
                tseed = seed*7919 + hash(tname) % 99991 + rseed
                per_t[tname] += _click_ndcg(ranking, fn, tseed, n_replays)
        rankings["qrels"][sys_name] = ndcg_sum/len(pools)
        for tname in click_fns:
            rankings[tname][sys_name] = per_t[tname]/len(pools)
    return rankings


def build_b2_aol(qrels_by_q, doc_ids, click_fns, seed,
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
        cand = [{"doc_id": d, "rel": 2} for d in pos]
        attempts = 0
        while sum(1 for c in cand if c["rel"] == 0) < n_distractors and attempts < 5*n_distractors:
            d = rng_doc.choice(doc_ids)
            if d not in seen:
                cand.append({"doc_id": d, "rel": 0})
                seen.add(d)
            attempts += 1
        pools.append((q, cand))
    rankings = _b2_score(pools, click_fns, seed, n_replays)
    return rankings, len(pools)


def build_b2_trec(qrels_by_topic, click_fns, seed,
                  n_topics=40, min_judged=80, max_judged=1000, n_replays=8):
    rng = random.Random(seed)
    eligible = [(t, d) for t, d in qrels_by_topic.items()
                if min_judged <= len(d) <= max_judged]
    sampled = rng.sample(eligible, k=min(n_topics, len(eligible)))
    pools = []
    for topic, judged in sampled:
        cand = [{"doc_id": d, "rel": r} for d, r in judged.items()]
        pools.append((topic, cand))
    rankings = _b2_score(pools, click_fns, seed, n_replays)
    return rankings, len(pools)


def aggregate(rankings, seed):
    out = {"trusted": "qrels", "testers": {}}
    trusted = rankings["qrels"]
    for tname, scores in rankings.items():
        if tname == "qrels":
            continue
        tau = KendallTau().compute(scores_1=trusted, scores_2=scores,
                                   n_bootstrap=1000, random_state=seed)
        out["testers"][tname] = {
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
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--trec-qrels", type=Path,
                   default=Path("data/trec_session2014/judgments.txt"))
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--n-replays", type=int, default=8)
    p.add_argument("--llm-real-dir", type=Path,
                   default=Path("data/llm_real"))
    args = p.parse_args()

    LLM_PAIRS = {
        "trec-s14": [
            ("llm-gpt5nano", "trec-s14_gpt5nano.jsonl"),
            ("llm-qwen3next", "trec-s14_qwen3next.jsonl"),
        ],
        # AOL-IA SERPs are doc-id-only (no titles or snippets), so a
        # prompt-driven simulator has nothing to react to. Skip it here
        # and run the LLM-real B2 on TREC Session 2014 only.
        "aol-ia": [],
    }
    base_click_fns = {
        "simiir-pbm": pbm, "simiir-dbn": dbn,
        "heuristic": heur, "llm-sim": llm_sim,
    }

    # AOL B2
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
    print(f"  {len(aol_qrels)} queries, {len(aol_doc_ids)} docs")

    # Fit per-LLM-real profile from AOL sessions
    aol_click_fns = dict(base_click_fns)
    for tname, fname in LLM_PAIRS["aol-ia"]:
        path = args.llm_real_dir / fname
        if not path.exists():
            print(f"  skip {tname}: {path} missing")
            continue
        prof = fit_click_profile(path)
        print(f"  fit {tname} on AOL: median_dwell={prof['median_dwell']:.1f}, "
              f"profile_keys={len(prof['p_click'])}")
        aol_click_fns[tname] = make_click_fn(prof)

    print("\nB2 (AOL)...")
    rankings, n_q = build_b2_aol(aol_qrels, aol_doc_ids, aol_click_fns,
                                  seed=args.seed, n_queries=100,
                                  n_replays=args.n_replays)
    out = aggregate(rankings, seed=args.seed)
    out["dataset"] = "aol-ia"
    out["n_queries"] = n_q
    out["seed"] = args.seed
    out["rankings"] = rankings
    aol_out = args.output_root / "aol-ia/b2/b2_results.json"
    aol_out.parent.mkdir(parents=True, exist_ok=True)
    aol_out.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> {aol_out}")
    for t, s in out["testers"].items():
        print(f"    {t}: tau={s['kendall_tau']:.3f}")

    # TREC B2
    print("\nLoading TREC qrels...")
    trec_qrels = defaultdict(dict)
    with open(args.trec_qrels) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                trec_qrels[int(parts[0])][parts[2]] = max(0, int(parts[3]))
    print(f"  {len(trec_qrels)} topics")

    trec_click_fns = dict(base_click_fns)
    for tname, fname in LLM_PAIRS["trec-s14"]:
        path = args.llm_real_dir / fname
        if not path.exists():
            print(f"  skip {tname}: {path} missing")
            continue
        prof = fit_click_profile(path)
        print(f"  fit {tname} on TREC: median_dwell={prof['median_dwell']:.1f}, "
              f"profile_keys={len(prof['p_click'])}")
        trec_click_fns[tname] = make_click_fn(prof)

    print("\nB2 (TREC)...")
    rankings, n_q = build_b2_trec(trec_qrels, trec_click_fns,
                                   seed=args.seed, n_topics=40,
                                   n_replays=args.n_replays)
    out = aggregate(rankings, seed=args.seed)
    out["dataset"] = "trec-s14"
    out["n_queries"] = n_q
    out["seed"] = args.seed
    out["rankings"] = rankings
    trec_out = args.output_root / "trec-s14/b2/b2_results.json"
    trec_out.parent.mkdir(parents=True, exist_ok=True)
    trec_out.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> {trec_out}")
    for t, s in out["testers"].items():
        print(f"    {t}: tau={s['kendall_tau']:.3f}")


if __name__ == "__main__":
    main()
