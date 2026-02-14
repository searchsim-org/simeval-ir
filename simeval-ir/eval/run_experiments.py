#!/usr/bin/env python
"""Main experiment runner for SimEval-IR paper experiments.

Usage:
    python run_experiments.py --config configs/b1_tripclick.yaml
    python run_experiments.py --experiment b1 --dataset tripclick
    python run_experiments.py --experiment all --all-datasets
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simeval_ir.datasets import get_adapter, load_sessions_from_jsonl
from simeval_ir.eval import run_protocol, BenchmarkConfig
from simeval_ir.embeddings.action import ActionSequenceEmbedder
from simeval_ir.metrics.behavior import run_leakage_audit
from simeval_ir.metrics.tester import KendallTau, SpearmanRho, RATE


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_real_sessions(config: dict) -> list:
    """Load real sessions based on configuration."""
    real_config = config["real_data"]
    adapter_name = real_config.get("adapter")
    data_path = Path(real_config["path"])
    split = real_config.get("split", "all")
    max_sessions = real_config.get("max_sessions")

    if adapter_name:
        adapter = get_adapter(adapter_name)()
        sessions = list(adapter.load_sessions(data_path, split))
    elif data_path.suffix == ".jsonl":
        sessions = list(load_sessions_from_jsonl(data_path))
    elif data_path.suffix == ".json":
        from simeval_ir.datasets import load_sessions_from_json
        sessions = load_sessions_from_json(data_path)
    else:
        raise ValueError(f"Cannot determine format for {data_path}")

    if max_sessions:
        sessions = sessions[:max_sessions]

    return sessions


def load_simulated_sessions(sim_config: dict) -> list:
    """Load simulated sessions."""
    data_path = Path(sim_config["path"])

    if data_path.suffix == ".jsonl":
        return list(load_sessions_from_jsonl(data_path))
    elif data_path.suffix == ".json":
        from simeval_ir.datasets import load_sessions_from_json
        return load_sessions_from_json(data_path)
    else:
        # Try adapter
        adapter_name = sim_config.get("adapter")
        if adapter_name:
            adapter = get_adapter(adapter_name)()
            return list(adapter.load_sessions(data_path))
        raise ValueError(f"Cannot determine format for {data_path}")


def run_b1_experiment(config: dict, output_dir: Path) -> dict:
    """Run B1: Behavioral Realism experiments."""
    print("\n" + "=" * 60)
    print("B1: BEHAVIORAL REALISM EXPERIMENT")
    print("=" * 60)

    # Load real sessions
    print("\nLoading real sessions...")
    real_sessions = load_real_sessions(config)
    print(f"  Loaded {len(real_sessions)} real sessions")

    # Initialize embedder
    embedder = ActionSequenceEmbedder()

    results = {
        "experiment": "B1",
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "real_sessions": len(real_sessions),
        "simulators": {},
    }

    # Evaluate each simulator
    simulators = config.get("simulators", [])
    for sim_config in simulators:
        sim_name = sim_config["name"]
        print(f"\nEvaluating simulator: {sim_name}")

        try:
            sim_sessions = load_simulated_sessions(sim_config)
            print(f"  Loaded {len(sim_sessions)} simulated sessions")

            # Run B1 protocol
            print("  Running realism-benchmark-v1...")
            b1_result = run_protocol(
                protocol="realism-benchmark-v1",
                real=real_sessions,
                sim=sim_sessions,
                embedder=embedder,
                n_folds=config.get("n_folds", 5),
            )

            # Extract metrics
            metrics = {}
            for r in b1_result.metric_results:
                metrics[r.name] = {
                    "value": r.value,
                    "meta": r.meta,
                }

            # Run leakage audit
            print("  Running leakage audit...")
            audit = run_leakage_audit(
                real_sessions,
                sim_sessions,
                embedder=embedder,
                n_folds=config.get("n_folds", 5),
            )

            results["simulators"][sim_name] = {
                "sessions": len(sim_sessions),
                "b1_metrics": metrics,
                "leakage_audit": {
                    "main_auc": audit.main_auc,
                    "metadata_only_auc": audit.metadata_only_auc,
                    "structural_only_auc": audit.structural_only_auc,
                    "permutation_auc": audit.permutation_auc,
                    "leakage_detected": audit.leakage_detected,
                    "realism_score": audit.realism_score,
                },
            }

            # Print summary
            print(f"  Results for {sim_name}:")
            for name, data in metrics.items():
                print(f"    {name}: {data['value']:.4f}")
            print(f"    Leakage detected: {audit.leakage_detected}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results["simulators"][sim_name] = {"error": str(e)}

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "b1_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    return results


def run_b2_experiment(config: dict, output_dir: Path) -> dict:
    """Run B2: Tester Reliability experiments."""
    print("\n" + "=" * 60)
    print("B2: TESTER RELIABILITY EXPERIMENT")
    print("=" * 60)

    results = {
        "experiment": "B2",
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "testers": {},
    }

    # Load system rankings from different testers
    testers_config = config.get("testers", {})

    # Trusted tester (typically qrels-based)
    trusted_name = config.get("trusted_tester", "qrels")
    trusted_scores = testers_config.get(trusted_name, {})

    if not trusted_scores:
        print("WARNING: No trusted tester scores provided")
        return results

    results["trusted_tester"] = trusted_name

    # Compute agreement for each tester
    for tester_name, scores in testers_config.items():
        if tester_name == trusted_name:
            continue

        print(f"\nEvaluating tester: {tester_name}")

        # Compute Kendall's tau
        tau_metric = KendallTau()
        tau_result = tau_metric.compute(
            rankings={"trusted": trusted_scores, tester_name: scores}
        )

        # Compute Spearman's rho
        rho_metric = SpearmanRho()
        rho_result = rho_metric.compute(
            rankings={"trusted": trusted_scores, tester_name: scores}
        )

        results["testers"][tester_name] = {
            "kendall_tau": tau_result.value,
            "tau_meta": tau_result.meta,
            "spearman_rho": rho_result.value,
            "rho_meta": rho_result.meta,
        }

        print(f"  Kendall's tau: {tau_result.value:.3f}")
        print(f"  Spearman's rho: {rho_result.value:.3f}")

    # RATE-style aggregation
    if len(testers_config) > 2:
        print("\nRunning RATE aggregation...")
        rate_metric = RATE()
        rate_result = rate_metric.compute(rankings=testers_config)
        results["rate"] = {
            "reliabilities": rate_result.meta.get("reliabilities", {}),
            "aggregated_ranking": rate_result.meta.get("aggregated_ranking", {}),
        }
        print(f"  Reliabilities: {results['rate']['reliabilities']}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "b2_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    return results


def run_b3_experiment(config: dict, output_dir: Path) -> dict:
    """Run B3: Realism-Reliability Analysis."""
    print("\n" + "=" * 60)
    print("B3: REALISM-RELIABILITY ANALYSIS")
    print("=" * 60)

    from scipy.stats import pearsonr, spearmanr

    results = {
        "experiment": "B3",
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "correlations": {},
    }

    # Load B1 and B2 results
    b1_path = Path(config.get("b1_results", "results/b1/b1_results.json"))
    b2_path = Path(config.get("b2_results", "results/b2/b2_results.json"))

    if not b1_path.exists() or not b2_path.exists():
        print("ERROR: B1 and B2 results required. Run those experiments first.")
        return results

    with open(b1_path) as f:
        b1_results = json.load(f)
    with open(b2_path) as f:
        b2_results = json.load(f)

    # Get common simulators
    b1_sims = set(b1_results.get("simulators", {}).keys())
    b2_testers = set(b2_results.get("testers", {}).keys())
    common = b1_sims & b2_testers

    if len(common) < 3:
        print(f"WARNING: Only {len(common)} common simulators/testers. Need >= 3 for correlation.")

    print(f"\nAnalyzing {len(common)} simulators: {common}")

    # Extract B1 metrics and B2 tau for each simulator
    realism_metrics = ["jsd_action_types", "session_fd", "wasserstein_session_length", "mmd", "realism_classifier"]

    for metric_name in realism_metrics:
        realism_values = []
        reliability_values = []

        for sim in common:
            b1_data = b1_results["simulators"].get(sim, {})
            b2_data = b2_results["testers"].get(sim, {})

            if "b1_metrics" in b1_data and metric_name in b1_data["b1_metrics"]:
                realism_val = b1_data["b1_metrics"][metric_name]
                if isinstance(realism_val, dict):
                    realism_val = realism_val.get("value", 0)
                realism_values.append(realism_val)

                tau_val = b2_data.get("kendall_tau", 0)
                reliability_values.append(tau_val)

        if len(realism_values) >= 3:
            corr, p_value = pearsonr(realism_values, reliability_values)
            results["correlations"][metric_name] = {
                "pearson_r": corr,
                "p_value": p_value,
                "n_points": len(realism_values),
            }
            print(f"  {metric_name} vs tau: r={corr:.3f} (p={p_value:.3f})")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "b3_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run SimEval-IR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=["b1", "b2", "b3", "all"],
        default="b1",
        help="Which experiment to run (default: b1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run on all supported datasets",
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration for testing
        config = {
            "name": "default-experiment",
            "real_data": {
                "path": "./data/real/",
                "adapter": "synthetic",
                "split": "search",
            },
            "simulators": [
                {"name": "synthetic-test", "path": "./data/simulated/", "adapter": "synthetic"},
            ],
            "n_folds": 5,
            "seed": 42,
        }

    output_dir = args.output / config.get("name", "experiment")

    # Run experiments
    if args.experiment in ("b1", "all"):
        run_b1_experiment(config, output_dir / "b1")

    if args.experiment in ("b2", "all"):
        run_b2_experiment(config, output_dir / "b2")

    if args.experiment in ("b3", "all"):
        run_b3_experiment(config, output_dir / "b3")

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
