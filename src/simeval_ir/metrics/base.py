"""Base metric class for SimEval-IR.

This module provides the abstract base class for all metrics,
along with the metric registry for discovery and lookup.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult


class BaseMetric(ABC):
    """Abstract base class for all metrics.

    Metrics are organized by:
    - objective: "behavior" (realism), "evaluation" (system), "tester" (reliability)
    - granularity: "turn", "session", "system"
    - scenarios: which session types are supported ("T", "C")

    Example:
        >>> class MyMetric(BaseMetric):
        ...     name = "my_metric"
        ...     objective = "behavior"
        ...     granularity = "session"
        ...     scenarios = {"T", "C"}
        ...
        ...     def compute(self, real, sim, **kwargs):
        ...         # Compute metric
        ...         return MetricResult(name=self.name, value=0.5)
    """

    name: str
    objective: Literal["behavior", "evaluation", "tester"]
    granularity: Literal["turn", "session", "system"]
    scenarios: set[Literal["T", "C"]]
    description: str = ""

    @abstractmethod
    def compute(self, **kwargs) -> MetricResult:
        """Compute the metric.

        The signature varies by metric type:
        - Behavior metrics: compute(real, sim, **kwargs)
        - Evaluation metrics: compute(sessions, qrels, **kwargs)
        - Tester metrics: compute(testers, **kwargs)

        Returns:
            MetricResult with computed value and optional CIs.
        """
        ...

    def validate_sessions(
        self,
        sessions: list[InteractionSession],
        check_type: bool = True,
    ) -> None:
        """Validate that sessions are compatible with this metric.

        Args:
            sessions: Sessions to validate.
            check_type: Whether to check session types against scenarios.

        Raises:
            ValueError: If sessions are incompatible.
        """
        if not sessions:
            raise ValueError("No sessions provided.")

        if check_type:
            for s in sessions:
                session_scenario = (
                    "T" if s.session_type.value == "search" else "C"
                )
                if session_scenario not in self.scenarios:
                    raise ValueError(
                        f"Session {s.session_id} has type {s.session_type.value}, "
                        f"but metric {self.name} only supports scenarios {self.scenarios}."
                    )


# Global metric registry
METRIC_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_metric(name: str, metric_cls: type[BaseMetric]) -> None:
    """Register a metric.

    Args:
        name: Name to register the metric under.
        metric_cls: The metric class.
    """
    METRIC_REGISTRY[name] = metric_cls


def get_metric(name: str) -> type[BaseMetric]:
    """Get a registered metric by name.

    Args:
        name: Name of the metric.

    Returns:
        The metric class.

    Raises:
        KeyError: If the metric is not registered.
    """
    if name not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY.keys())) or "none"
        raise KeyError(
            f"Metric '{name}' not found. Available metrics: {available}."
        )
    return METRIC_REGISTRY[name]


def list_metrics(
    objective: Literal["behavior", "evaluation", "tester"] | None = None,
    granularity: Literal["turn", "session", "system"] | None = None,
    scenario: Literal["T", "C"] | None = None,
) -> list[str]:
    """List registered metrics with optional filtering.

    Args:
        objective: Filter by objective type.
        granularity: Filter by granularity level.
        scenario: Filter by supported scenario.

    Returns:
        List of metric names matching the filters.
    """
    result = []
    for name, cls in METRIC_REGISTRY.items():
        if objective is not None and cls.objective != objective:
            continue
        if granularity is not None and cls.granularity != granularity:
            continue
        if scenario is not None and scenario not in cls.scenarios:
            continue
        result.append(name)
    return sorted(result)


# Decorator for convenient registration
def metric(name: str):
    """Decorator to register a metric.

    Example:
        >>> @metric("my_metric")
        ... class MyMetric(BaseMetric):
        ...     pass
    """

    def decorator(cls: type[BaseMetric]) -> type[BaseMetric]:
        register_metric(name, cls)
        return cls

    return decorator
