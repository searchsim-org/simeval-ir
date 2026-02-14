"""Metrics module for SimEval-IR."""

from simeval_ir.metrics.base import (
    METRIC_REGISTRY,
    BaseMetric,
    get_metric,
    list_metrics,
    metric,
    register_metric,
)

__all__ = [
    "BaseMetric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "metric",
    "METRIC_REGISTRY",
]
