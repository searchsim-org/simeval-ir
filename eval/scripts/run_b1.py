#!/usr/bin/env python
"""B1: Behavioral Realism Experiment.

Compares simulated sessions against real sessions using multiple realism metrics.

Usage:
    python run_b1.py --real-data data/tripclick/ --sim-data data/simulated.jsonl --output results/
    python run_b1.py --real-adapter tripclick --real-path data/tripclick/ \
                     --sim-path data/simiir_output.jsonl --output results/
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simeval_ir.core.session import SCHEMA_VERSION
from simeval_ir.datasets import get_adapter, load_sessions_from_jsonl, load_sessions_from_json
from simeval_ir.eval import run_protocol
from simeval_ir.embeddings.action import ActionSequenceEmbedder
from simeval_ir.metrics.behavior import run_leakage_audit


def load_sessions(path: Path, adapter_name: str | None = None, split: str = "all"):
    """Load sessions from path using appropriate method."""
    if adapter_name:
        adapter = get_adapter(adapter_name)()
        return list(adapter.load_sessions(path, split))
    elif path.suffix == ".jsonl":
        return list(load_sessions_from_jsonl(path))
    elif path.suffix == ".json":
        return load_sessions_from_json(path)
    elif path.is_dir():
        # Try to find files
        jsonl_files = list(path.glob("*.jsonl"))
        if jsonl_files:
            sessions = []
            for f in jsonl_files:
                sessions.extend(load_sessions_from_jsonl(f))
            return sessions
    raise ValueError(f"Cannot determine how to load sessions from {path}")


def run_b1(
    real_path: Path,
    sim_path: Path,
    output_dir: Path,
    real_adapter: str | None = None,
    sim_adapter: str | None = None,
    split: str = "all",
    n_folds: int = 5,
    sim_name: str = "simulator",
    seed: int = 42,
):
    """Run B1 behavioral realism experiment."""
    print("=" * 60)
    print("B1: BEHAVIORAL REALISM EXPERIMENT")
    print("=" * 60)

    # Load sessions
    print(f"\nLoading real sessions from {real_path}...")
    real_sessions = load_sessions(real_path, real_adapter, split)
    print(f"  Loaded {len(real_sessions)} real sessions")

    print(f"\nLoading simulated sessions from {sim_path}...")
    sim_sessions = load_sessions(sim_path, sim_adapter, split)
    print(f"  Loaded {len(sim_sessions)} simulated sessions")

    # Initialize embedder
    embedder = ActionSequenceEmbedder()

    # Run B1 protocol
    print("\nRunning realism-benchmark-v1 protocol...")
    b1_result = run_protocol(
        protocol="realism-benchmark-v1",
        real=real_sessions,
        sim=sim_sessions,
        embedder=embedder,
        n_folds=n_folds,
    )

    # Extract metrics
    metrics = {}
    print("\nB1 Metrics:")
    print("-" * 40)
    for r in b1_result.metric_results:
        metrics[r.name] = {
            "value": r.value,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "meta": r.meta,
        }
        ci_str = ""
        if r.ci_lower is not None and r.ci_upper is not None:
            ci_str = f" [{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
        print(f"  {r.name}: {r.value:.4f}{ci_str}")

    # Run leakage audit
    print("\nRunning leakage audit...")
    print("-" * 40)
    audit = run_leakage_audit(
        real_sessions,
        sim_sessions,
        embedder=embedder,
        n_folds=n_folds,
    )

    leakage_results = {
        "main_auc": audit.main_auc,
        "metadata_only_auc": audit.metadata_only_auc,
        "masked_feature_auc": audit.masked_feature_auc,
        "structural_only_auc": audit.structural_only_auc,
        "permutation_auc": audit.permutation_auc,
        "leakage_detected": audit.leakage_detected,
        "realism_score": audit.realism_score if not audit.leakage_detected else None,
    }

    print(f"  Main classifier AUC: {audit.main_auc:.3f}")
    print(f"  Metadata-only AUC:   {audit.metadata_only_auc:.3f}")
    print(f"  Structural-only AUC: {audit.structural_only_auc:.3f}")
    print(f"  Permutation AUC:     {audit.permutation_auc:.3f}")
    print(f"  Leakage detected:    {audit.leakage_detected}")

    if audit.leakage_detected:
        print("\n  WARNING: Leakage detected! Classifier may be exploiting artifacts.")
    else:
        print(f"\n  Valid realism score: {audit.realism_score:.3f}")

    # Compute config hash
    config = {
        "real_path": str(real_path),
        "sim_path": str(sim_path),
        "real_adapter": real_adapter,
        "sim_adapter": sim_adapter,
        "n_folds": n_folds,
        "seed": seed,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Compile results
    results = {
        "experiment": "B1",
        "timestamp": datetime.now().isoformat(),
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "config_hash": config_hash,
            "random_seed": seed,
        },
        "config": config,
        "data": {
            "real_sessions": len(real_sessions),
            "sim_sessions": len(sim_sessions),
        },
        "simulator": sim_name,
        "b1_metrics": metrics,
        "leakage_audit": leakage_results,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"b1_{sim_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run B1: Behavioral Realism Experiment"
    )
    parser.add_argument(
        "--real-path", "-r",
        type=Path,
        required=True,
        help="Path to real session data",
    )
    parser.add_argument(
        "--sim-path", "-s",
        type=Path,
        required=True,
        help="Path to simulated session data",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/b1"),
        help="Output directory",
    )
    parser.add_argument(
        "--real-adapter",
        type=str,
        help="Adapter name for real data (e.g., tripclick, trec-session)",
    )
    parser.add_argument(
        "--sim-adapter",
        type=str,
        help="Adapter name for simulated data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Data split to use (train, dev, test, all)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds for classifier metrics",
    )
    parser.add_argument(
        "--sim-name",
        type=str,
        default="simulator",
        help="Name for the simulator (used in output filename)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    run_b1(
        real_path=args.real_path,
        sim_path=args.sim_path,
        output_dir=args.output,
        real_adapter=args.real_adapter,
        sim_adapter=args.sim_adapter,
        split=args.split,
        n_folds=args.n_folds,
        sim_name=args.sim_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
