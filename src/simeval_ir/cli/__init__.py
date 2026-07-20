"""Command-line interface for SimEval-IR."""

import json
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from simeval_ir import __version__
from simeval_ir.datasets import list_adapters, load_sessions_from_jsonl
from simeval_ir.metrics import list_metrics, get_metric
from simeval_ir.metrics.behavior import run_leakage_audit
from simeval_ir.metrics.tester import KendallTau, SpearmanRho, PearsonCorr, RATE
from simeval_ir.embeddings.action import ActionSequenceEmbedder
from simeval_ir.eval import run_protocol


console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """SimEval-IR: evaluate simulated search and conversational sessions."""
    pass


# --------------------------------------------------------------------------
# Discovery commands
# --------------------------------------------------------------------------

@main.command("list-metrics")
@click.option("--objective", "-o",
              type=click.Choice(["behavior", "evaluation", "tester"]))
@click.option("--scenario", "-s", type=click.Choice(["T", "C"]))
def cmd_list_metrics(objective, scenario):
    """List available metrics."""
    metrics = list_metrics(objective=objective, scenario=scenario)

    table = Table(title="Available Metrics")
    table.add_column("Name", style="cyan")
    table.add_column("Objective", style="green")
    table.add_column("Granularity", style="yellow")
    table.add_column("Scenarios", style="magenta")

    for name in metrics:
        metric_cls = get_metric(name)
        table.add_row(
            name,
            metric_cls.objective,
            metric_cls.granularity,
            ", ".join(metric_cls.scenarios),
        )

    console.print(table)


@main.command("list-protocols")
def cmd_list_protocols():
    """List available evaluation protocols."""
    from simeval_ir.eval.protocols import PROTOCOLS

    table = Table(title="Available Protocols")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Metrics", style="green")

    for name, protocol in PROTOCOLS.items():
        table.add_row(
            name,
            protocol.description,
            ", ".join(protocol.metrics),
        )

    console.print(table)


@main.command("list-datasets")
def cmd_list_datasets():
    """List registered dataset adapters."""
    adapters = list_adapters()

    if not adapters:
        console.print("[yellow]No dataset adapters registered.[/yellow]")
        return

    table = Table(title="Registered Dataset Adapters")
    table.add_column("Name", style="cyan")

    for name in adapters:
        table.add_row(name)

    console.print(table)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load(path: str | Path):
    """Load a JSONL session file and return the list of sessions."""
    return list(load_sessions_from_jsonl(Path(path)))


def _save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _safe(v):
    """Convert NaN to None for JSON output."""
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


# --------------------------------------------------------------------------
# Generic protocol runner (any registered protocol by name)
# --------------------------------------------------------------------------

@main.command("run-protocol")
@click.option("--protocol", "-p", required=True, help="Protocol name")
@click.option("--real", "-r", type=click.Path(exists=True),
              help="Path to real sessions (JSONL)")
@click.option("--sim", "-s", type=click.Path(exists=True),
              help="Path to simulated sessions (JSONL)")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def cmd_run_protocol(protocol, real, sim, output):
    """Run a named evaluation protocol."""
    console.print(f"[bold]Running protocol: {protocol}[/bold]")

    real_sessions = _load(real) if real else None
    sim_sessions = _load(sim) if sim else None
    if real_sessions is not None:
        console.print(f"  real: {len(real_sessions)} sessions")
    if sim_sessions is not None:
        console.print(f"  sim:  {len(sim_sessions)} sessions")

    result = run_protocol(protocol, real=real_sessions, sim=sim_sessions,
                          embedder=ActionSequenceEmbedder())

    table = Table(title=f"Protocol Results: {protocol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")
    for r in result.metric_results:
        ok = not r.meta.get("error")
        v = _safe(r.value)
        table.add_row(r.name,
                      f"{v:.4f}" if isinstance(v, float) else "N/A",
                      "OK" if ok else "ERR")
    console.print(table)

    if output:
        _save_json(output, {
            "protocol": protocol,
            "metrics": [
                {"name": r.name, "value": _safe(r.value), "meta": r.meta}
                for r in result.metric_results
            ],
        })
        console.print(f"[green]Results saved to {output}[/green]")


# --------------------------------------------------------------------------
# B1, B2, B3 named runners (the "one command" form referenced in the paper)
# --------------------------------------------------------------------------

@main.command("run-b1")
@click.option("--real", "-r", type=click.Path(exists=True), required=True)
@click.option("--sim",  "-s", type=click.Path(exists=True), required=True)
@click.option("--output", "-o", type=click.Path(), required=True)
@click.option("--n-folds", default=5, type=int,
              help="CV folds for the classifier metric")
def cmd_run_b1(real, sim, output, n_folds):
    """B1: behavioural realism (real vs simulated sessions)."""
    real_s, sim_s = _load(real), _load(sim)
    console.print(f"[bold]B1: realism[/bold]  real={len(real_s)} sim={len(sim_s)}")

    embedder = ActionSequenceEmbedder()
    res = run_protocol("realism-benchmark-v1", real=real_s, sim=sim_s,
                       embedder=embedder)
    audit = run_leakage_audit(real_s, sim_s, embedder=embedder, n_folds=n_folds)

    payload = {
        "experiment": "B1",
        "n_real": len(real_s),
        "n_sim":  len(sim_s),
        "metrics": {r.name: {"value": _safe(r.value), "meta": r.meta}
                    for r in res.metric_results},
        "leakage_audit": {
            "main_auc": audit.main_auc,
            "metadata_only_auc": audit.metadata_only_auc,
            "structural_only_auc": audit.structural_only_auc,
            "permutation_auc": audit.permutation_auc,
            "leakage_detected": audit.leakage_detected,
        },
    }
    _save_json(output, payload)
    console.print(f"[green]Wrote {output}[/green]")


@main.command("run-b2")
@click.option("--rankings", "-r", type=click.Path(exists=True), required=True,
              help="YAML or JSON file with per-tester {system: score} maps")
@click.option("--trusted", "-t", default="qrels",
              help="Name of the trusted tester key")
@click.option("--output", "-o", type=click.Path(), required=True)
@click.option("--n-bootstrap", default=1000, type=int)
@click.option("--seed", default=42, type=int)
def cmd_run_b2(rankings, trusted, output, n_bootstrap, seed):
    """B2: tester reliability (Kendall/Spearman/Pearson + RATE + LOO)."""
    rankings_path = Path(rankings)
    if rankings_path.suffix in (".yaml", ".yml"):
        import yaml  # optional dep handled lazily
        data = yaml.safe_load(rankings_path.read_text())
        scores = data.get("testers", data)
    else:
        scores = json.loads(rankings_path.read_text())

    if trusted not in scores:
        raise click.ClickException(
            f"Trusted tester '{trusted}' not in rankings file")
    trusted_scores = scores[trusted]
    console.print(f"[bold]B2: tester reliability[/bold]  trusted={trusted}  "
                  f"systems={len(trusted_scores)}")

    out = {"experiment": "B2", "trusted_tester": trusted,
           "n_systems": len(trusted_scores), "testers": {}}
    for tester, sc in scores.items():
        if tester == trusted:
            continue
        tau = KendallTau().compute(scores_1=trusted_scores, scores_2=sc,
                                   n_bootstrap=n_bootstrap, random_state=seed)
        rho = SpearmanRho().compute(scores_1=trusted_scores, scores_2=sc,
                                    n_bootstrap=n_bootstrap, random_state=seed)
        pr = PearsonCorr().compute(scores_1=trusted_scores, scores_2=sc,
                                   n_bootstrap=n_bootstrap, random_state=seed)
        out["testers"][tester] = {
            "kendall_tau": tau.value, "tau_ci": [tau.ci_lower, tau.ci_upper],
            "spearman_rho": rho.value, "pearson_r":  pr.value,
        }

    if len(scores) > 2:
        rate = RATE().compute(tester_scores=scores)
        out["rate"] = {"reliabilities": rate.per_item or {},
                       "iterations": rate.meta.get("iterations", 0)}

    _save_json(output, out)
    console.print(f"[green]Wrote {output}[/green]")


@main.command("run-b3")
@click.option("--b1-dir", type=click.Path(exists=True), required=True,
              help="Directory with one B1 result JSON per simulator "
                   "(filename must contain the simulator name).")
@click.option("--b2-results", type=click.Path(exists=True), required=True,
              help="Path to a B2 results JSON.")
@click.option("--output", "-o", type=click.Path(), required=True)
def cmd_run_b3(b1_dir, b2_results, output):
    """B3: realism vs reliability correlation analysis."""
    from scipy.stats import pearsonr, spearmanr

    b1 = {}
    for p in Path(b1_dir).glob("*.json"):
        d = json.loads(p.read_text())
        sim = d.get("simulator") or p.stem.replace("b1_", "")
        b1[sim] = d.get("metrics") or d.get("b1_metrics", {})
    b2 = json.loads(Path(b2_results).read_text())

    common = sorted(set(b1) & set(b2.get("testers", {})))
    console.print(f"[bold]B3: realism vs reliability[/bold]  sims={common}")

    rows = []
    metric_keys = ["jsd_click_depth", "wasserstein_session_length",
                   "session_fd", "jsd_action_types",
                   "reformulation_similarity", "realism_classifier", "mmd"]
    for sim in common:
        row = {"sim": sim,
               "tau": b2["testers"][sim].get("kendall_tau", float("nan"))}
        for k in metric_keys:
            v = b1[sim].get(k, {})
            row[k] = v.get("value") if isinstance(v, dict) else v
        rows.append(row)

    correlations = {}
    for k in metric_keys:
        xs = [r[k] for r in rows if r[k] is not None and not np.isnan(r[k])]
        ys = [r["tau"] for r in rows if r[k] is not None and not np.isnan(r[k])]
        if len(xs) < 3:
            continue
        pr, pp = pearsonr(xs, ys)
        sr, sp = spearmanr(xs, ys)
        correlations[k] = {"pearson_r": float(pr), "pearson_p": float(pp),
                           "spearman_rho": float(sr), "spearman_p": float(sp),
                           "n": len(xs)}

    _save_json(output, {"experiment": "B3",
                        "simulators": common,
                        "correlations": correlations,
                        "raw_rows": rows})
    console.print(f"[green]Wrote {output}[/green]")


# --------------------------------------------------------------------------
# Backwards-compatible alias kept for the README
# --------------------------------------------------------------------------

@main.command("generate-llm-sim")
@click.option("--real", "-r", type=click.Path(exists=True), required=True,
              help="JSONL file of real reference sessions to drive")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="JSONL file to write simulated sessions to")
@click.option("--model", "-m", required=True,
              help="Model id (e.g., gpt-4o-mini, meta-llama/...)")
@click.option("--api-key-env", default="OPENAI_API_KEY",
              help="Env var holding the API key (default: OPENAI_API_KEY)")
@click.option("--base-url", default=None,
              help="OpenAI-compatible base URL. "
                   "Use $CUSTOM_LLM_ENDPOINT for the custom endpoint.")
@click.option("--temperature", default=0.0, type=float)
@click.option("--max-queries", default=5, type=int,
              help="Max queries per session")
@click.option("--limit", default=0, type=int,
              help="Stop after this many sessions (0=all)")
@click.option("--offset", default=0, type=int,
              help="Skip this many real sessions before starting (default 0)")
@click.option("--concurrency", default=1, type=int,
              help="Number of concurrent sessions (thread pool); each "
                   "session is still serial inside.")
@click.option("--seed", default=None, type=int)
def cmd_generate_llm_sim(real, output, model, api_key_env, base_url,
                         temperature, max_queries, limit, offset,
                         concurrency, seed):
    """Drive a real LLM as a search simulator over real reference sessions.

    The simulator never sees relevance judgements; it only sees the same
    URL/title/snippet text a real user would. Provenance (model name,
    temperature, prompt-template hash, endpoint) is logged into the
    output sessions' metadata.
    """
    import os
    from simeval_ir.datasets.io import save_sessions_to_jsonl
    from simeval_ir.simulators.llm_real import LLMSimulator

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise click.ClickException(
            f"Env var {api_key_env} is not set. "
            f"Set it (or use --api-key-env) before running.")

    real_sessions = list(load_sessions_from_jsonl(Path(real)))
    if offset:
        real_sessions = real_sessions[offset:]
    if limit:
        real_sessions = real_sessions[:limit]

    console.print(
        f"[bold]LLM-real[/bold]  model={model} endpoint={base_url or 'OpenAI default'} "
        f"temperature={temperature} sessions={len(real_sessions)} "
        f"concurrency={concurrency}"
    )
    sim = LLMSimulator(model=model, api_key=api_key, base_url=base_url,
                       temperature=temperature, seed=seed)
    out_sessions = list(sim.simulate_from(real_sessions, max_queries=max_queries,
                                          concurrency=concurrency))
    save_sessions_to_jsonl(out_sessions, Path(output))
    meta = sim.last_run_meta.to_dict() if hasattr(sim, "last_run_meta") else {}
    console.print(
        f"[green]Wrote {len(out_sessions)} sessions[/green]  "
        f"calls={meta.get('api_calls', '?')}"
    )
    # Sidecar provenance file
    side = Path(output).with_suffix(Path(output).suffix + ".meta.json")
    _save_json(side, {"simulator": "llm-real",
                      "n_sessions": len(out_sessions),
                      "run_meta": meta})
    console.print(f"[green]Wrote provenance -> {side}[/green]")


@main.command("eval-behavior")
@click.option("--real", "-r", type=click.Path(exists=True), required=True)
@click.option("--sim",  "-s", type=click.Path(exists=True), required=True)
@click.option("--metrics", "-m",
              default="jsd_action_types,session_length_distribution",
              help="Comma-separated metric names")
@click.option("--output", "-o", type=click.Path())
def cmd_eval_behavior(real, sim, metrics, output):
    """Compute a custom subset of B1 metrics."""
    real_s, sim_s = _load(real), _load(sim)
    console.print(f"real={len(real_s)} sim={len(sim_s)}")
    out = []
    embedder = ActionSequenceEmbedder()
    table = Table(title="Behavioural Realism")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="green")
    for name in [m.strip() for m in metrics.split(",")]:
        try:
            r = get_metric(name)().compute(real=real_s, sim=sim_s, embedder=embedder)
            out.append({"name": r.name, "value": _safe(r.value),
                        "ci_lower": r.ci_lower, "ci_upper": r.ci_upper,
                        "meta": r.meta})
            v = _safe(r.value)
            table.add_row(r.name, f"{v:.4f}" if isinstance(v, float) else "N/A")
        except Exception as e:
            console.print(f"[red]{name}: {e}[/red]")
    console.print(table)
    if output:
        _save_json(output, {"metrics": out})
        console.print(f"[green]Wrote {output}[/green]")


if __name__ == "__main__":
    main()
