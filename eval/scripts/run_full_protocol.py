#!/usr/bin/env python
"""End-to-end SimEval-IR Reference Protocol runner (B1 + B2 + B3).

Executes the full SRP-v1 pipeline over multiple random seeds, treating each
seed as an independent topic split. The result is a B3 analysis with
n = n_seeds * n_simulators data points instead of just n_simulators.

Usage:
    python run_full_protocol.py --output results --seeds 20260429,20260430,20260501
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval/scripts"))

from generate_protocol import (  # noqa: E402
    SIMULATORS,
    SYSTEMS,
    build_b2_rankings,
    generate_reference_sessions,
    generate_simulator_sessions,
)

from simeval_ir.datasets import load_sessions_from_jsonl, save_sessions_to_jsonl  # noqa: E402
from simeval_ir.eval import run_protocol  # noqa: E402
from simeval_ir.embeddings.action import ActionSequenceEmbedder  # noqa: E402
from simeval_ir.metrics.behavior import run_leakage_audit  # noqa: E402
from simeval_ir.metrics.tester import KendallTau, RATE  # noqa: E402

from scipy.stats import pearsonr, spearmanr  # noqa: E402


def run_b1_one(real, sim, sim_name: str, n_folds: int = 5):
    embedder = ActionSequenceEmbedder()
    b1 = run_protocol(protocol="realism-benchmark-v1", real=real, sim=sim,
                      embedder=embedder, n_folds=n_folds)
    audit = run_leakage_audit(real, sim, embedder=embedder, n_folds=n_folds)
    metrics = {r.name: {"value": r.value, "meta": r.meta} for r in b1.metric_results}
    metrics["realism_classifier"]["meta"]["audit"] = {
        "main_auc": audit.main_auc,
        "metadata_only_auc": audit.metadata_only_auc,
        "structural_only_auc": audit.structural_only_auc,
        "permutation_auc": audit.permutation_auc,
        "leakage_detected": audit.leakage_detected,
    }
    return metrics


def run_b2_from_rankings(rankings: dict[str, dict[str, float]], trusted: str = "qrels",
                         n_bootstrap: int = 1000, seed: int = 42):
    out = {"trusted": trusted, "testers": {}}
    trusted_scores = rankings[trusted]
    for tester, scores in rankings.items():
        if tester == trusted:
            continue
        tau = KendallTau().compute(scores_1=trusted_scores, scores_2=scores,
                                   n_bootstrap=n_bootstrap, random_state=seed)
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
    return out


def run_one_seed(seed: int, n_sessions: int, n_topics: int, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- seed={seed} (n_sessions={n_sessions}, n_topics={n_topics}) ---")

    # Generate / cache sessions
    ref = generate_reference_sessions(n_sessions, seed=seed)
    save_sessions_to_jsonl(ref, work_dir / "real_baseline.jsonl")

    sim_sessions = {}
    for name in SIMULATORS:
        sess = generate_simulator_sessions(name, n_sessions,
                                           seed=seed + 17 * hash(name) % 100003)
        save_sessions_to_jsonl(sess, work_dir / f"{name}.jsonl")
        sim_sessions[name] = sess

    # B1 per simulator
    b1 = {}
    for name, sessions in sim_sessions.items():
        print(f"  B1 [{name}]...")
        b1[name] = run_b1_one(ref, sessions, sim_name=name)

    # B2 from rankings
    print("  B2: building rankings + Kendall...")
    rankings, _, _ = build_b2_rankings(seed, n_topics=n_topics)
    b2 = run_b2_from_rankings(rankings, trusted="qrels", n_bootstrap=1000, seed=seed)

    return {"seed": seed, "n_sessions": n_sessions, "n_topics": n_topics,
            "b1": b1, "b2": b2, "rankings": rankings}


def aggregate_b3(per_seed: list[dict]):
    """Across seeds, build (realism_metric, tester_tau) pairs per simulator."""
    metrics_of_interest = [
        ("jsd_click_depth",            "JS(click)"),
        ("wasserstein_session_length", "W(len)"),
        ("jsd_action_types",           "JS(action)"),
        ("reformulation_similarity",   "Reform"),
        ("realism_classifier",         "Classifier (audited)"),
        ("session_fd",                 "Frechet"),
        ("mmd",                        "MMD"),
    ]
    # Collect per (sim, seed) data points
    rows = []  # list of dicts
    for run in per_seed:
        seed = run["seed"]
        for sim_name in SIMULATORS:
            tau = run["b2"]["testers"][sim_name]["kendall_tau"]
            row = {"sim": sim_name, "seed": seed, "tau": tau}
            for key, _ in metrics_of_interest:
                v = run["b1"][sim_name].get(key, {}).get("value")
                row[key] = v
            rows.append(row)

    correlations = {}
    for key, label in metrics_of_interest:
        xs = np.array([r[key] for r in rows if r[key] is not None and not np.isnan(r[key])])
        ys = np.array([r["tau"] for r in rows if r[key] is not None and not np.isnan(r[key])])
        if len(xs) < 3:
            continue
        pr, pp = pearsonr(xs, ys)
        sr, sp = spearmanr(xs, ys)
        correlations[key] = {
            "label": label,
            "n": int(len(xs)),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
            "abs_pearson": float(abs(pr)),
        }
    return rows, correlations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str,
                   default="20260429,20260430,20260501,20260502")
    p.add_argument("--n-sessions", type=int, default=400)
    p.add_argument("--n-topics", type=int, default=25)
    p.add_argument("--output", type=Path, default=Path("results"))
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SimEval-IR Full Protocol Runner")
    print("=" * 60)
    print(f"Seeds: {seeds}")

    per_seed = []
    for seed in seeds:
        wd = args.output / f"per_seed/seed_{seed}/data"
        run = run_one_seed(seed, args.n_sessions, args.n_topics, work_dir=wd)
        per_seed.append(run)
        with open(args.output / f"per_seed/seed_{seed}/result.json", "w") as f:
            json.dump(run, f, indent=2, default=str)

    rows, correlations = aggregate_b3(per_seed)

    out = {
        "experiment": "SRP-v1 full pipeline",
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "n_sessions": args.n_sessions,
        "n_topics": args.n_topics,
        "n_simulators": len(SIMULATORS),
        "n_data_points": len(rows),
        "b3_correlations": correlations,
        "raw_rows": rows,
    }
    out_dir = args.output / "b3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "b3_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Persist a single B1/B2 directory using the FIRST seed (canonical)
    canonical = per_seed[0]
    (args.output / "b1").mkdir(parents=True, exist_ok=True)
    for name, m in canonical["b1"].items():
        out_file = args.output / "b1" / f"b1_{name}.json"
        with open(out_file, "w") as f:
            json.dump({
                "experiment": "B1",
                "timestamp": canonical.get("seed"),
                "simulator": name,
                "seed": canonical["seed"],
                "n_sessions": canonical["n_sessions"],
                "metrics": m,
            }, f, indent=2, default=str)

    (args.output / "b2").mkdir(parents=True, exist_ok=True)
    with open(args.output / "b2/b2_results.json", "w") as f:
        json.dump({
            "experiment": "B2",
            "seed": canonical["seed"],
            "n_topics": canonical["n_topics"],
            **canonical["b2"],
            "rankings": canonical["rankings"],
        }, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("B3 SUMMARY (across all seeds)")
    print("=" * 60)
    print(f"n data points = {len(rows)}")
    for key, c in sorted(correlations.items(), key=lambda x: -x[1]["abs_pearson"]):
        print(f"  {c['label']:<24} r = {c['pearson_r']:+.3f}  (p={c['pearson_p']:.3f}, n={c['n']})")
    print("\nFull results: ", out_dir / "b3_results.json")


if __name__ == "__main__":
    main()
