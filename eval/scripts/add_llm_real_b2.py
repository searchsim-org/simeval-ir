#!/usr/bin/env python
"""Add LLM-real testers to an existing TREC Session 2014 B2 result without
re-running the four base testers.

The script reproduces ``run_full_protocol_xds.py``'s B2 SERP synthesis
verbatim (same qrels, same seed, same RNG sequencing), runs the LLM-real
click profiles on the resulting SERPs, computes Kendall's tau against the
qrels-nDCG ranking already in the file, and merges the new tester rows in.

Run AFTER ``run_full_protocol_xds.py`` (which produces the canonical
4-tester ``results/trec-s14/b2/b2_results.json``) and AFTER
``simeval generate-llm-sim`` (which produces the LLM-real session JSONLs).
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
sys.path.insert(0, str(ROOT / "eval/scripts"))

from simeval_ir.core.session import Click, RankedItem  # noqa: E402
from simeval_ir.core.types import EventType  # noqa: E402
from simeval_ir.datasets.io import load_sessions_from_jsonl  # noqa: E402
from simeval_ir.metrics.tester import KendallTau, RATE  # noqa: E402

# Reuse the exact xds helpers so the SERPs are bit-identical
from run_full_protocol_xds import (  # noqa: E402
    SYSTEMS, _system_rank, _ndcg_at_k, _click_ndcg, build_b2_trec_s14,
)


def fit_click_profile(sim_path: Path) -> dict:
    counts = defaultdict(lambda: [0, 0])
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
                last_serp = None
    p = {}
    for (rank, rb), (clicks, imps) in counts.items():
        p[(rank, rb)] = (clicks + 0.5) / (imps + 1.0)
    return {"p_click": p,
            "median_dwell": (sorted(dwells)[len(dwells)//2]
                             if dwells else 25.0)}


def make_click_fn(profile: dict):
    p_click = profile["p_click"]
    md = profile["median_dwell"]

    def _fn(rng: random.Random, serp):
        out = []
        for it in serp[:10]:
            rb = "rel" if (it.judged_relevance or 0) > 0 else "nonrel"
            p = p_click.get((it.rank, rb),
                            p_click.get((it.rank, "nonrel"), 0.0))
            if rng.random() < p:
                out.append(Click(doc_id=it.doc_id, rank=it.rank,
                                 dwell_time=max(3.0, rng.gauss(md, 4.0))))
        return out
    return _fn


def replay_for_tester(qrels_by_topic, click_fn, seed,
                      n_topics=40, min_judged=80, max_judged=1000,
                      n_replays=8):
    """Reproduce build_b2_trec_s14's SERP loop for one tester."""
    rng = random.Random(seed)
    eligible = [(t, d) for t, d in qrels_by_topic.items()
                if min_judged <= len(d) <= max_judged]
    sampled = rng.sample(eligible, k=min(n_topics, len(eligible)))
    pools = []
    for topic, judged in sampled:
        cand = [{"doc_id": d, "rel": r} for d, r in judged.items()]
        pools.append((topic, cand))
    per_sys = {}
    for sys_name, alpha in SYSTEMS:
        s_sum = 0.0
        for q_idx, (q, pool) in enumerate(pools):
            r_seed = seed + hash(sys_name) % 99991 + q_idx
            ranking = _system_rank(random.Random(r_seed), pool, alpha)
            tseed = seed * 7919 + hash("__llm_real_proxy__") % 99991 + r_seed
            s_sum += _click_ndcg(ranking, click_fn, tseed, n_replays)
        per_sys[sys_name] = s_sum / len(pools)
    return per_sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b2-path", type=Path,
                   default=Path("results/trec-s14/b2/b2_results.json"))
    p.add_argument("--qrels", type=Path,
                   default=Path("data/trec_session2014/judgments.txt"))
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--llm-real-dir", type=Path, default=Path("data/llm_real"))
    args = p.parse_args()

    b2 = json.loads(args.b2_path.read_text())
    qrels_per_topic = b2.get("rankings", {}).get("qrels", {})
    if not qrels_per_topic:
        sys.exit("Existing B2 file has no qrels rankings to compare against.")

    print("Loading TREC qrels...")
    qrels = defaultdict(dict)
    with open(args.qrels) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qrels[int(parts[0])][parts[2]] = max(0, int(parts[3]))

    LLM_FILES = [("llm-gpt5nano", "trec-s14_gpt5nano.jsonl"),
                 ("llm-qwen3next", "trec-s14_qwen3next.jsonl")]

    for tname, fname in LLM_FILES:
        sim_path = args.llm_real_dir / fname
        if not sim_path.exists():
            print(f"  skip {tname}: {sim_path} missing")
            continue
        print(f"\nFitting {tname} click profile from {sim_path.name}...")
        profile = fit_click_profile(sim_path)
        print(f"  median_dwell={profile['median_dwell']:.1f}, "
              f"profile_keys={len(profile['p_click'])}")
        per_sys = replay_for_tester(qrels, make_click_fn(profile),
                                     seed=args.seed)
        b2.setdefault("rankings", {})[tname] = per_sys

        # Kendall tau vs qrels (using the qrels rankings already in the file)
        tau = KendallTau().compute(scores_1=qrels_per_topic, scores_2=per_sys,
                                   n_bootstrap=1000, random_state=args.seed)
        b2.setdefault("testers", {})[tname] = {
            "kendall_tau": tau.value,
            "tau_ci_lower": tau.ci_lower,
            "tau_ci_upper": tau.ci_upper,
            "tau_p_value": tau.meta.get("p_value"),
        }
        print(f"  {tname}: tau={tau.value:+.3f}  "
              f"CI=[{tau.ci_lower:+.3f}, {tau.ci_upper:+.3f}]")

    # Recompute RATE with all testers (including new ones)
    rate = RATE().compute(tester_scores=b2["rankings"])
    b2["rate"] = {"reliabilities": rate.per_item or {},
                  "iterations": rate.meta.get("iterations", 0)}

    args.b2_path.write_text(json.dumps(b2, indent=2, default=str))
    print(f"\nUpdated {args.b2_path}")
    print("\nFinal RATE reliabilities:")
    for k, v in sorted(b2["rate"]["reliabilities"].items(),
                       key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:.3f}")


if __name__ == "__main__":
    main()
