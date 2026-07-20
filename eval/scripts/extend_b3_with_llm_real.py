#!/usr/bin/env python
"""Append LLM-real (gpt-5-nano, qwen3-next) data points to the existing
B3 results JSON and recompute pooled and per-dataset correlations.

Inputs:
    results/b3/b3_results.json                 (existing 48-row result)
    results/<dataset>/b1/b1_llm-<model>.json   (4 files; from run_b1_llm_real.py)
    results/<dataset>/b2/b2_results.json       (per-dataset, contains llm-real testers)

Output:
    results/b3/b3_results_llmreal.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import pearsonr, spearmanr


METRIC_KEYS = [
    "jsd_click_depth", "wasserstein_session_length",
    "session_fd", "jsd_action_types",
    "reformulation_similarity", "realism_classifier", "mmd",
]
LLM_PAIRS = [
    ("trec-s14", "llm-gpt5nano"), ("trec-s14", "llm-qwen3next"),
    ("aol-ia",  "llm-gpt5nano"), ("aol-ia",  "llm-qwen3next"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path,
                   default=Path("results/b3/b3_results_llmreal.json"))
    args = p.parse_args()

    base = json.loads(
        (args.results_root / "b3" / "b3_results.json").read_text())
    rows = list(base.get("raw_rows", []))

    # Add 4 LLM-real rows
    for ds, sim in LLM_PAIRS:
        b1_path = args.results_root / ds / "b1" / f"b1_{sim}.json"
        b2_path = args.results_root / ds / "b2" / "b2_results.json"
        if not b1_path.exists() or not b2_path.exists():
            print(f"  [skip] missing {b1_path} or {b2_path}")
            continue
        b1 = json.loads(b1_path.read_text())
        b2 = json.loads(b2_path.read_text())
        tau = b2.get("testers", {}).get(sim, {}).get("kendall_tau")
        if tau is None:
            print(f"  [skip] no tau for {ds}/{sim}")
            continue
        row = {"dataset": ds, "sim": sim, "seed": b2.get("seed"),
               "tau": float(tau),
               "n_real_sessions": b1.get("n_sessions"),
               "n_b2_queries": b2.get("n_queries")}
        for k in METRIC_KEYS:
            v = b1["metrics"].get(k, {}).get("value") if isinstance(
                b1.get("metrics"), dict) else None
            row[k] = float(v) if isinstance(v, (int, float)) and not (
                isinstance(v, float) and math.isnan(v)) else None
        rows.append(row)
        print(f"  + {ds}/{sim}  tau={tau:+.3f}  jsd_click={row['jsd_click_depth']:.3f}")

    def correlations(rs):
        out = {}
        for k in METRIC_KEYS:
            xs = [r[k] for r in rs if r.get(k) is not None]
            ys = [r["tau"] for r in rs if r.get(k) is not None]
            if len(xs) < 3:
                continue
            pr, pp = pearsonr(xs, ys)
            sr, sp = spearmanr(xs, ys)
            out[k] = {"pearson_r": float(pr), "pearson_p": float(pp),
                      "spearman_rho": float(sr), "spearman_p": float(sp),
                      "n": len(xs), "abs_pearson": float(abs(pr))}
        return out

    pooled = correlations(rows)
    per_dataset = {}
    for ds in {r["dataset"] for r in rows}:
        per_dataset[ds] = correlations([r for r in rows if r["dataset"] == ds])

    out = {
        "experiment": "B3 cross-dataset (with LLM-real)",
        "datasets": sorted({r["dataset"] for r in rows}),
        "n_data_points": len(rows),
        "n_simulators_total": len({r["sim"] for r in rows}),
        "correlations_pooled": pooled,
        "correlations_per_dataset": per_dataset,
        "raw_rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.out}  (n={len(rows)}, sims={len({r['sim'] for r in rows})})")
    print("\nPOOLED correlations:")
    for k, c in sorted(pooled.items(), key=lambda x: -x[1]["abs_pearson"]):
        print(f"  {k:<28} r={c['pearson_r']:+.3f}  (p={c['pearson_p']:.3f}, "
              f"n={c['n']})")
    for ds, dc in per_dataset.items():
        print(f"\n{ds.upper()} correlations:")
        for k, c in sorted(dc.items(), key=lambda x: -x[1]["abs_pearson"]):
            print(f"  {k:<28} r={c['pearson_r']:+.3f}  "
                  f"(p={c['pearson_p']:.3f}, n={c['n']})")


if __name__ == "__main__":
    main()
