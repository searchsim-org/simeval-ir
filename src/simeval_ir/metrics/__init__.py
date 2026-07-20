"""Metrics module for SimEval-IR.

Importing this package auto-registers every shipped metric (B1 realism,
B2 tester reliability, system effectiveness) so that ``get_metric(name)``
resolves out of the box.
"""

from simeval_ir.metrics.base import (
    METRIC_REGISTRY,
    BaseMetric,
    get_metric,
    list_metrics,
    metric,
    register_metric,
)

# Importing the submodules below populates METRIC_REGISTRY via the @metric
# decorator. This is what makes get_metric("jsd_click_depth") work after a
# bare `from simeval_ir.metrics import get_metric`.
from simeval_ir.metrics import behavior  # noqa: F401  (registers B1 metrics)
from simeval_ir.metrics import tester    # noqa: F401  (registers B2 metrics)
from simeval_ir.metrics import system    # noqa: F401  (registers session-level metrics)

__all__ = [
    "BaseMetric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "metric",
    "METRIC_REGISTRY",
]
