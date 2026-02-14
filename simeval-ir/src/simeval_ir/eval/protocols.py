"""Named evaluation protocols."""

from dataclasses import dataclass, field
from typing import Any

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult
from simeval_ir.metrics import get_metric


@dataclass
class Protocol:
    """Definition of an evaluation protocol."""

    name: str
    description: str
    metrics: list[str]
    scenarios: list[str] = field(default_factory=lambda: ["T", "C"])


@dataclass
class ProtocolResult:
    """Result of running a protocol."""

    protocol_name: str
    metric_results: list[MetricResult]
    meta: dict[str, Any] = field(default_factory=dict)


# Named evaluation protocols
PROTOCOLS: dict[str, Protocol] = {
    "realism-benchmark-v1": Protocol(
        name="realism-benchmark-v1",
        description="Standard behavioral realism benchmark (B1)",
        metrics=[
            "jsd_action_types",
            "jsd_click_depth",
            "session_fd",
            "realism_classifier",
            "reformulation_similarity",
            "session_length_distribution",
            "wasserstein_session_length",
            "mmd",
        ],
        scenarios=["T", "C"],
    ),
    "tester-reliability-v1": Protocol(
        name="tester-reliability-v1",
        description="Tester reliability assessment protocol (B2)",
        metrics=["kendall_tau", "spearman_rho", "rate"],
        scenarios=["T", "C"],
    ),
    "realism-reliability-v1": Protocol(
        name="realism-reliability-v1",
        description="Realism-reliability analysis benchmark (B3) - analyzes correlation between B1 realism metrics and B2 reliability metrics",
        metrics=[
            "jsd_action_types",
            "jsd_click_depth",
            "session_fd",
            "realism_classifier",
            "reformulation_similarity",
            "wasserstein_session_length",
            "mmd",
            "kendall_tau",
            "spearman_rho",
        ],
        scenarios=["T", "C"],
    ),
    "session-effectiveness-v1": Protocol(
        name="session-effectiveness-v1",
        description="Session-level system effectiveness",
        metrics=["session_ndcg", "egu"],
        scenarios=["T"],
    ),
}


def run_protocol(
    protocol: str | Protocol,
    real: list[InteractionSession] | None = None,
    sim: list[InteractionSession] | None = None,
    embedder=None,
    **kwargs,
) -> ProtocolResult:
    """Run a named evaluation protocol.

    Args:
        protocol: Protocol name or Protocol object.
        real: Real sessions (for behavior protocols).
        sim: Simulated sessions (for behavior protocols).
        embedder: Optional embedder for embedding-based metrics.
        **kwargs: Additional arguments passed to metrics.

    Returns:
        ProtocolResult with all metric results.
    """
    if isinstance(protocol, str):
        if protocol not in PROTOCOLS:
            available = ", ".join(PROTOCOLS.keys())
            raise ValueError(
                f"Unknown protocol '{protocol}'. Available: {available}"
            )
        protocol = PROTOCOLS[protocol]

    results = []
    for metric_name in protocol.metrics:
        try:
            metric_cls = get_metric(metric_name)
            metric = metric_cls()

            # Determine which arguments to pass based on metric type
            if metric.objective == "behavior":
                result = metric.compute(
                    real=real,
                    sim=sim,
                    embedder=embedder,
                    **kwargs,
                )
            elif metric.objective == "evaluation":
                result = metric.compute(
                    sessions=real or sim,
                    **kwargs,
                )
            else:
                result = metric.compute(**kwargs)

            results.append(result)
        except Exception as e:
            # Record error but continue with other metrics
            results.append(MetricResult(
                name=metric_name,
                value=float("nan"),
                meta={"error": str(e)},
            ))

    return ProtocolResult(
        protocol_name=protocol.name,
        metric_results=results,
    )


def list_protocols() -> list[str]:
    """List available protocol names.

    Returns:
        List of protocol names.
    """
    return list(PROTOCOLS.keys())
