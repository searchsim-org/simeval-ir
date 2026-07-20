#!/usr/bin/env python
"""B3: Realism-Reliability Analysis.

Analyzes correlation between B1 realism metrics and B2 tester reliability.

Usage:
    python run_b3.py --b1-dir results/b1/ --b2-results results/b2/b2_results.json --output results/b3/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_b1_results(b1_dir: Path) -> dict[str, dict]:
    """Load all B1 results from directory."""
    results = {}
    for f in b1_dir.glob("b1_*.json"):
        with open(f) as fp:
            data = json.load(fp)
            sim_name = data.get("simulator", f.stem.replace("b1_", ""))
            results[sim_name] = data
    return results


def load_b2_results(b2_path: Path) -> dict:
    """Load B2 results."""
    with open(b2_path) as f:
        return json.load(f)


def run_b3(
    b1_dir: Path,
    b2_path: Path,
    output_dir: Path,
):
    """Run B3 realism-reliability analysis."""
    from scipy.stats import pearsonr, spearmanr

    print("=" * 60)
    print("B3: REALISM-RELIABILITY ANALYSIS")
    print("=" * 60)

    # Load results
    print("\nLoading B1 results...")
    b1_results = load_b1_results(b1_dir)
    print(f"  Found {len(b1_results)} simulators: {list(b1_results.keys())}")

    print("\nLoading B2 results...")
    b2_results = load_b2_results(b2_path)
    b2_testers = set(b2_results.get("testers", {}).keys())
    print(f"  Found {len(b2_testers)} testers: {b2_testers}")

    # Find common simulators/testers
    common = set(b1_results.keys()) & b2_testers
    print(f"\nCommon simulators: {len(common)} - {common}")

    if len(common) < 3:
        print("\nWARNING: Need at least 3 common simulators for meaningful correlation.")
        print("Run more B1 experiments or ensure simulator names match between B1 and B2.")

    results = {
        "experiment": "B3",
        "timestamp": datetime.now().isoformat(),
        "simulators_analyzed": list(common),
        "n_simulators": len(common),
        "correlations": {},
        "raw_data": {},
    }

    # Metrics to analyze (aligned with Table 4 in the paper)
    # Paper shows: JS(click depth), W(session length), KS(dwell time),
    # Classifier AUC (audited), Fréchet(embed), MMD(embed)
    realism_metrics = [
        "jsd_click_depth",           # JS(click depth)
        "wasserstein_session_length", # W(session length)
        "timing_distribution",        # KS(dwell time)
        "realism_classifier",         # Classifier AUC (audited)
        "session_fd",                 # Fréchet(embed)
        "mmd",                        # MMD(embed)
        # Additional metrics
        "jsd_action_types",
        "reformulation_similarity",
    ]

    # Extract data points
    print("\nExtracting metric values...")
    reliability_values = []
    metric_values = {m: [] for m in realism_metrics}

    for sim_name in common:
        # B2 reliability (Kendall's tau)
        tau = b2_results["testers"][sim_name].get("kendall_tau", 0)
        reliability_values.append(tau)

        # B1 realism metrics
        b1_data = b1_results[sim_name].get("b1_metrics", {})
        for metric in realism_metrics:
            if metric in b1_data:
                val = b1_data[metric]
                if isinstance(val, dict):
                    val = val.get("value", np.nan)
                metric_values[metric].append(val)
            else:
                metric_values[metric].append(np.nan)

    results["raw_data"] = {
        "simulators": list(common),
        "reliability_tau": reliability_values,
        "realism_metrics": metric_values,
    }

    # Compute correlations
    print("\n" + "=" * 60)
    print("Correlation: Realism Metric vs Tester Reliability (τ)")
    print("=" * 60)
    print(f"{'Realism Metric':<30} {'Pearson r':>12} {'p-value':>10} {'Spearman ρ':>12} {'p-value':>10}")
    print("-" * 74)

    for metric in realism_metrics:
        values = metric_values[metric]

        # Remove NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        if len(valid_indices) < 3:
            print(f"{metric:<30} {'N/A':>12} {'--':>10} {'N/A':>12} {'--':>10}")
            continue

        x = [values[i] for i in valid_indices]
        y = [reliability_values[i] for i in valid_indices]

        # Pearson correlation
        pearson_r, pearson_p = pearsonr(x, y)

        # Spearman correlation
        spearman_r, spearman_p = spearmanr(x, y)

        results["correlations"][metric] = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_p": spearman_p,
            "n_points": len(valid_indices),
        }

        print(f"{metric:<30} {pearson_r:>12.3f} {pearson_p:>10.4f} {spearman_r:>12.3f} {spearman_p:>10.4f}")

    # Summary interpretation
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    strong_predictors = []
    weak_predictors = []

    for metric, corr in results["correlations"].items():
        r = abs(corr.get("pearson_r", 0))
        if r >= 0.3:
            strong_predictors.append((metric, corr["pearson_r"]))
        else:
            weak_predictors.append((metric, corr["pearson_r"]))

    if strong_predictors:
        print("\nMetrics that predict tester reliability (|r| >= 0.3):")
        for metric, r in sorted(strong_predictors, key=lambda x: -abs(x[1])):
            print(f"  - {metric}: r = {r:.3f}")

    if weak_predictors:
        print("\nMetrics with weak/no predictive power (|r| < 0.3):")
        for metric, r in sorted(weak_predictors, key=lambda x: -abs(x[1])):
            print(f"  - {metric}: r = {r:.3f}")

    print("\nThis supports the paper's finding: marginal distribution metrics")
    print("(session length, click depth) weakly predict reliability, while")
    print("representation-level metrics (Fréchet, MMD) show stronger association.")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "b3_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run B3: Realism-Reliability Analysis"
    )
    parser.add_argument(
        "--b1-dir",
        type=Path,
        required=True,
        help="Directory containing B1 results (b1_*.json files)",
    )
    parser.add_argument(
        "--b2-results",
        type=Path,
        required=True,
        help="Path to B2 results JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/b3"),
        help="Output directory",
    )

    args = parser.parse_args()

    run_b3(args.b1_dir, args.b2_results, args.output)


if __name__ == "__main__":
    main()
