#!/usr/bin/env python
"""B2: Tester Reliability Experiment.

Computes agreement between simulator-induced system rankings and trusted evaluators.
Includes leave-one-out sensitivity analysis and bootstrap confidence intervals.

Usage:
    python run_b2.py --rankings rankings.yaml --trusted qrels --output results/
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simeval_ir.core.session import SCHEMA_VERSION
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
            return data.get("testers", data)
        else:
            return json.load(f)


def _compute_config_hash(rankings: dict, trusted_name: str) -> str:
    """Compute hash of the experiment configuration."""
    config_str = json.dumps({"rankings": rankings, "trusted": trusted_name}, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def run_b2(
    rankings: dict[str, dict[str, float]],
    trusted_name: str,
    output_dir: Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
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
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "config_hash": _compute_config_hash(rankings, trusted_name),
            "random_seed": seed,
            "n_bootstrap": n_bootstrap,
        },
        "trusted_tester": trusted_name,
        "n_systems": len(trusted_scores),
        "testers": {},
    }

    # Evaluate each tester against trusted
    print("\nTester Agreement with Trusted Evaluator:")
    print("-" * 70)
    print(f"{'Tester':<20} {'Kendall τ':>10} {'95% CI':>20} {'Spearman ρ':>12} {'Pearson r':>12}")
    print("-" * 70)

    for tester_name, tester_scores in rankings.items():
        if tester_name == trusted_name:
            continue

        # Compute agreement metrics with bootstrap CIs
        tau_result = KendallTau().compute(
            scores_1=trusted_scores, scores_2=tester_scores,
            n_bootstrap=n_bootstrap, random_state=seed,
        )
        rho_result = SpearmanRho().compute(
            scores_1=trusted_scores, scores_2=tester_scores,
            n_bootstrap=n_bootstrap, random_state=seed,
        )
        pearson_result = PearsonCorr().compute(
            scores_1=trusted_scores, scores_2=tester_scores,
            n_bootstrap=n_bootstrap, random_state=seed,
        )

        ci_str = ""
        if tau_result.ci_lower is not None and tau_result.ci_upper is not None:
            ci_str = f"[{tau_result.ci_lower:.2f}, {tau_result.ci_upper:.2f}]"

        results["testers"][tester_name] = {
            "kendall_tau": tau_result.value,
            "tau_ci_lower": tau_result.ci_lower,
            "tau_ci_upper": tau_result.ci_upper,
            "tau_p_value": tau_result.meta.get("p_value"),
            "spearman_rho": rho_result.value,
            "rho_ci_lower": rho_result.ci_lower,
            "rho_ci_upper": rho_result.ci_upper,
            "rho_p_value": rho_result.meta.get("p_value"),
            "pearson_r": pearson_result.value,
            "pearson_ci_lower": pearson_result.ci_lower,
            "pearson_ci_upper": pearson_result.ci_upper,
            "pearson_p_value": pearson_result.meta.get("p_value"),
        }

        print(f"{tester_name:<20} {tau_result.value:>10.3f} {ci_str:>20} {rho_result.value:>12.3f} {pearson_result.value:>12.3f}")

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

    # Leave-one-out sensitivity analysis
    if len(rankings) > 3:
        print("\n" + "=" * 50)
        print("Leave-One-Out Sensitivity Analysis")
        print("=" * 50)

        sensitivity = {}
        # Get the full RATE ranking for comparison
        full_rate = RATE()
        full_result = full_rate.compute(tester_scores=rankings)
        full_reliabilities = full_result.per_item or {}

        # Determine system ranking from full aggregation
        all_systems = sorted(set().union(*[set(v.keys()) for v in rankings.values()]))
        full_weights = {t: full_reliabilities.get(t, 0) for t in rankings}
        w_sum = sum(full_weights.values())
        if w_sum > 0:
            full_weights = {t: w / w_sum for t, w in full_weights.items()}

        full_system_scores = {}
        for sys_name in all_systems:
            weighted_sum = 0
            weight_total = 0
            for tester, scores in rankings.items():
                if sys_name in scores:
                    w = full_weights.get(tester, 0)
                    weighted_sum += w * scores[sys_name]
                    weight_total += w
            if weight_total > 0:
                full_system_scores[sys_name] = weighted_sum / weight_total

        full_top_system = max(full_system_scores, key=full_system_scores.get) if full_system_scores else None

        tester_names = [t for t in rankings if t != trusted_name]
        for leave_out in tester_names:
            reduced = {t: s for t, s in rankings.items() if t != leave_out}
            if len(reduced) < 2:
                continue

            loo_rate = RATE()
            loo_result = loo_rate.compute(tester_scores=reduced)
            loo_reliabilities = loo_result.per_item or {}

            # Compute system ranking without this tester
            loo_weights = {t: loo_reliabilities.get(t, 0) for t in reduced}
            w_sum = sum(loo_weights.values())
            if w_sum > 0:
                loo_weights = {t: w / w_sum for t, w in loo_weights.items()}

            loo_system_scores = {}
            for sys_name in all_systems:
                weighted_sum = 0
                weight_total = 0
                for tester, scores in reduced.items():
                    if sys_name in scores:
                        w = loo_weights.get(tester, 0)
                        weighted_sum += w * scores[sys_name]
                        weight_total += w
                if weight_total > 0:
                    loo_system_scores[sys_name] = weighted_sum / weight_total

            loo_top = max(loo_system_scores, key=loo_system_scores.get) if loo_system_scores else None

            sensitivity[leave_out] = {
                "reliabilities_without": loo_reliabilities,
                "top_system_changes": full_top_system != loo_top,
                "top_system_full": full_top_system,
                "top_system_loo": loo_top,
            }

            change_str = " *TOP CHANGED*" if full_top_system != loo_top else ""
            print(f"  Without {leave_out}: top={loo_top}{change_str}")

        results["sensitivity"] = sensitivity

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
        help="JSON/YAML file with system rankings from different testers",
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
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CIs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    rankings = load_rankings(args.rankings)
    run_b2(rankings, args.trusted, args.output, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
