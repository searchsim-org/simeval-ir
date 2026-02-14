"""Command-line interface for SimEval-IR."""

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from simeval_ir import __version__
from simeval_ir.datasets import list_adapters, load_sessions_from_jsonl
from simeval_ir.metrics import list_metrics, get_metric
from simeval_ir.eval import list_protocols, run_protocol


console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """SimEval-IR: Evaluate simulated search and conversational sessions."""
    pass


@main.command("list-metrics")
@click.option("--objective", "-o", type=click.Choice(["behavior", "evaluation", "tester"]))
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
        console.print("Install simeval-ir-datasets or register custom adapters.")
        return

    table = Table(title="Registered Dataset Adapters")
    table.add_column("Name", style="cyan")

    for name in adapters:
        table.add_row(name)

    console.print(table)


@main.command("eval-behavior")
@click.option("--real", "-r", type=click.Path(exists=True), required=True, help="Path to real sessions (JSONL)")
@click.option("--sim", "-s", type=click.Path(exists=True), required=True, help="Path to simulated sessions (JSONL)")
@click.option("--metrics", "-m", default="jsd_action_types,session_length_distribution", help="Comma-separated metrics")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def cmd_eval_behavior(real, sim, metrics, output):
    """Evaluate behavioral realism of simulated sessions."""
    console.print(f"[bold]Loading sessions...[/bold]")

    real_sessions = list(load_sessions_from_jsonl(Path(real)))
    sim_sessions = list(load_sessions_from_jsonl(Path(sim)))

    console.print(f"  Real: {len(real_sessions)} sessions")
    console.print(f"  Simulated: {len(sim_sessions)} sessions")

    metric_names = [m.strip() for m in metrics.split(",")]
    results = []

    table = Table(title="Behavioral Realism Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric_name in metric_names:
        try:
            metric_cls = get_metric(metric_name)
            metric = metric_cls()
            result = metric.compute(real=real_sessions, sim=sim_sessions)
            results.append(result)
            table.add_row(result.name, f"{result.value:.4f}")
        except Exception as e:
            console.print(f"[red]Error computing {metric_name}: {e}[/red]")

    console.print(table)

    if output:
        output_data = {
            "metrics": [
                {
                    "name": r.name,
                    "value": r.value,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                    "meta": r.meta,
                }
                for r in results
            ]
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


@main.command("run-protocol")
@click.option("--protocol", "-p", required=True, help="Protocol name")
@click.option("--real", "-r", type=click.Path(exists=True), help="Path to real sessions")
@click.option("--sim", "-s", type=click.Path(exists=True), help="Path to simulated sessions")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def cmd_run_protocol(protocol, real, sim, output):
    """Run a named evaluation protocol."""
    console.print(f"[bold]Running protocol: {protocol}[/bold]")

    real_sessions = None
    sim_sessions = None

    if real:
        real_sessions = list(load_sessions_from_jsonl(Path(real)))
        console.print(f"  Real: {len(real_sessions)} sessions")

    if sim:
        sim_sessions = list(load_sessions_from_jsonl(Path(sim)))
        console.print(f"  Simulated: {len(sim_sessions)} sessions")

    result = run_protocol(protocol, real=real_sessions, sim=sim_sessions)

    table = Table(title=f"Protocol Results: {protocol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    for metric_result in result.metric_results:
        status = "✓" if not metric_result.meta.get("error") else "✗"
        value_str = f"{metric_result.value:.4f}" if not np.isnan(metric_result.value) else "N/A"
        table.add_row(metric_result.name, value_str, status)

    console.print(table)

    if output:
        output_data = {
            "protocol": protocol,
            "metrics": [
                {
                    "name": r.name,
                    "value": r.value if not np.isnan(r.value) else None,
                    "meta": r.meta,
                }
                for r in result.metric_results
            ]
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        console.print(f"[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    import numpy as np  # Ensure numpy is available for nan checks
    main()
