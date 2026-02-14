#!/usr/bin/env python
"""B2: Tester Reliability Experiment.

Computes agreement between simulator-induced system rankings and trusted evaluators.

Usage:
    python run_b2.py --rankings rankings.json --trusted qrels --output results/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simeval_ir.metrics.tester import KendallTau, SpearmanRho, PearsonCorr, RATE


def load_rankings(path: Path) -> dict[str, dict[str, float]]:
    """Load system rankings from JSON or YAML file.

    Expected format:
    {
        "qrels": {"sys_A": 0.45, "sys_B": 0.52, ...},
        "simiir_pbm": {"sys_A": 0.42, "sys_B": 0.48, ...},
        ...
    }

    Or YAML with 'testers' key containing the rankings.
    """
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(f)
            # YAML config has 'testers' key
            return data.get("testers", data)
        else:
            return json.load(f)


def run_b2(
    rankings: dict[str, dict[str, float]],
    trusted_name: str,
    output_dir: Path,
):
    """Run B2 tester reliability experiment."""
    print("=" * 60)
    print("B2: TESTER RELIABILITY EXPERIMENT")
    print("=" * 60)

    if trusted_name not in rankings:
        raise ValueError(f"Trusted tester '{trusted_name}' not found in rankings")

    trusted_scores = rankings[trusted_name]
    print(f"\nTrusted tester: {trusted_name}")
    print(f"Systems evaluated: {len(trusted_scores)}")

    results = {
        "experiment": "B2",
        "timestamp": datetime.now().isoformat(),
        "trusted_tester": trusted_name,
        "n_systems": len(trusted_scores),
        "testers": {},
    }

    # Evaluate each tester against trusted
    print("\nTester Agreement with Trusted Evaluator:")
    print("-" * 50)
    print(f"{'Tester':<20} {'Kendall τ':>12} {'Spearman ρ':>12} {'Pearson r':>12}")
    print("-" * 50)

    for tester_name, tester_scores in rankings.items():
        if tester_name == trusted_name:
            continue

        # Compute agreement metrics
        tau_metric = KendallTau()
        rho_metric = SpearmanRho()
        pearson_metric = PearsonCorr()

        tau_result = tau_metric.compute(
            scores_1=trusted_scores, scores_2=tester_scores
        )
        rho_result = rho_metric.compute(
            scores_1=trusted_scores, scores_2=tester_scores
        )
        pearson_result = pearson_metric.compute(
            scores_1=trusted_scores, scores_2=tester_scores
        )

        results["testers"][tester_name] = {
            "kendall_tau": tau_result.value,
            "tau_p_value": tau_result.meta.get("p_value"),
            "spearman_rho": rho_result.value,
            "rho_p_value": rho_result.meta.get("p_value"),
            "pearson_r": pearson_result.value,
            "pearson_p_value": pearson_result.meta.get("p_value"),
        }

        print(f"{tester_name:<20} {tau_result.value:>12.3f} {rho_result.value:>12.3f} {pearson_result.value:>12.3f}")

    # RATE-style aggregation
    if len(rankings) > 2:
        print("\n" + "=" * 50)
        print("RATE-Style Reliability Aggregation")
        print("=" * 50)

        rate_metric = RATE()
        rate_result = rate_metric.compute(tester_scores=rankings)

        reliabilities = rate_result.per_item or {}
        iterations = rate_result.meta.get("iterations", 0)
        results["rate"] = {
            "reliabilities": reliabilities,
            "converged": iterations < rate_metric.max_iterations,
            "iterations": iterations,
        }

        print("\nPer-tester reliability weights:")
        for tester, weight in sorted(reliabilities.items(), key=lambda x: -x[1]):
            print(f"  {tester}: {weight:.3f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "b2_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run B2: Tester Reliability Experiment"
    )
    parser.add_argument(
        "--rankings", "-r",
        type=Path,
        required=True,
        help="JSON file with system rankings from different testers",
    )
    parser.add_argument(
        "--trusted", "-t",
        type=str,
        default="qrels",
        help="Name of the trusted tester (default: qrels)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/b2"),
        help="Output directory",
    )

    args = parser.parse_args()

    rankings = load_rankings(args.rankings)
    run_b2(rankings, args.trusted, args.output)


if __name__ == "__main__":
    main()
