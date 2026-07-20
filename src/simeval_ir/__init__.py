"""SimEval-IR: a unified framework for evaluating user simulators and search sessions.

The public surface exposed at the top level is the minimum a user needs
to write code against the toolkit:

    >>> from simeval_ir import (
    ...     InteractionSession, Event, EventType, RankedItem, Click,
    ...     load_sessions_from_jsonl, save_sessions_to_jsonl,
    ...     get_metric, list_metrics,
    ...     run_protocol, list_protocols,
    ... )
    >>> sessions = list(load_sessions_from_jsonl("real.jsonl"))
    >>> result = run_protocol("realism-benchmark-v1", real=sessions, sim=sim)

Heavier optional integrations (SimIIR, Sim4IA-Bench, PyTerrier) live in
``simeval_ir.integrations`` and are imported lazily on demand.
"""

from simeval_ir.core.session import (
    Click,
    Event,
    InteractionSession,
    RankedItem,
)
from simeval_ir.core.types import (
    EventType,
    MetricResult,
    Role,
    SessionType,
)

# Pull in the metric registry submodules so @metric decorators run.
from simeval_ir.metrics import (  # noqa: F401  (side-effect: register metrics)
    BaseMetric,
    METRIC_REGISTRY,
    get_metric,
    list_metrics,
    register_metric,
)
from simeval_ir.datasets import (  # noqa: F401  (side-effect: register adapters)
    DatasetAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    load_sessions_from_json,
    load_sessions_from_jsonl,
    save_sessions_to_json,
    save_sessions_to_jsonl,
)
from simeval_ir.eval import list_protocols, run_protocol

__version__ = "0.1.0"

__all__ = [
    # Core data models
    "InteractionSession",
    "Event",
    "RankedItem",
    "Click",
    # Types
    "SessionType",
    "EventType",
    "Role",
    "MetricResult",
    # Metric registry
    "BaseMetric",
    "METRIC_REGISTRY",
    "get_metric",
    "list_metrics",
    "register_metric",
    # Dataset adapters and I/O
    "DatasetAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "load_sessions_from_json",
    "load_sessions_from_jsonl",
    "save_sessions_to_json",
    "save_sessions_to_jsonl",
    # Protocol runner
    "list_protocols",
    "run_protocol",
    # Version
    "__version__",
]
