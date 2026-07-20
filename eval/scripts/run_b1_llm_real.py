#!/usr/bin/env python
"""B1 (behavioural realism) for the real-LLM simulator outputs.

For each (dataset, llm_model) pair, loads the matching real-session shard
and the pre-generated LLM-real sessions, then runs the realism-benchmark-v1
protocol plus the leakage audit. Writes one JSON per pair into
``results/<dataset>/b1/b1_llm-<model>.json`` so the existing B3 aggregator
just picks them up.

Run after ``simeval generate-llm-sim`` has produced the JSONL files in
``data/llm_real/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from simeval_ir.datasets.io import load_sessions_from_jsonl  # noqa: E402
from simeval_ir.embeddings.action import ActionSequenceEmbedder  # noqa: E402
from simeval_ir.eval import run_protocol  # noqa: E402
from simeval_ir.metrics.behavior import run_leakage_audit  # noqa: E402


PAIRS = [
    # (dataset_label, real_jsonl, n_real_shard, llm_jsonl, sim_label)
    ("trec-s14", "data/trec_session2014/sessions.jsonl", 209,
     "data/llm_real/trec-s14_gpt5nano.jsonl", "llm-gpt5nano"),
    ("trec-s14", "data/trec_session2014/sessions.jsonl", 209,
     "data/llm_real/trec-s14_qwen3next.jsonl", "llm-qwen3next"),
    ("aol-ia", "data/aol_ia/sessions.jsonl", 2000,
     "data/llm_real/aol-ia_gpt5nano.jsonl", "llm-gpt5nano"),
    ("aol-ia", "data/aol_ia/sessions.jsonl", 2000,
     "data/llm_real/aol-ia_qwen3next.jsonl", "llm-qwen3next"),
]


def run_pair(dataset, real_path, n_real, sim_path, sim_label, out_dir):
    real = list(load_sessions_from_jsonl(Path(real_path)))[:n_real]
    sim = list(load_sessions_from_jsonl(Path(sim_path)))
    print(f"\n[{dataset}/{sim_label}]  real={len(real)}  sim={len(sim)}")
    embedder = ActionSequenceEmbedder()
    proto = run_protocol("realism-benchmark-v1", real=real, sim=sim,
                         embedder=embedder, n_folds=5)
    audit = run_leakage_audit(real, sim, embedder=embedder, n_folds=5)
    metrics = {r.name: {"value": r.value, "meta": r.meta}
               for r in proto.metric_results}
    metrics["realism_classifier"]["meta"]["audit"] = {
        "main_auc": audit.main_auc,
        "metadata_only_auc": audit.metadata_only_auc,
        "structural_only_auc": audit.structural_only_auc,
        "permutation_auc": audit.permutation_auc,
        "leakage_detected": audit.leakage_detected,
    }
    payload = {
        "experiment": "B1",
        "dataset": dataset,
        "n_sessions": len(real),
        "simulator": sim_label,
        "metrics": metrics,
    }
    out_path = Path(out_dir) / dataset / "b1" / f"b1_{sim_label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  -> {out_path}")
    for k, v in metrics.items():
        if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
            print(f"    {k}: {v['value']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results"))
    args = p.parse_args()
    for ds, real, n, sim, label in PAIRS:
        if not Path(sim).exists():
            print(f"  [skip] {sim} missing")
            continue
        run_pair(ds, real, n, sim, label, args.results_root)


if __name__ == "__main__":
    main()
