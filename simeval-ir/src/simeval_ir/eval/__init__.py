"""Evaluation protocols and utilities for SimEval-IR."""

from simeval_ir.eval.protocols import (
    PROTOCOLS,
    Protocol,
    ProtocolResult,
    run_protocol,
    list_protocols,
)
from simeval_ir.eval.bootstrap import bootstrap_ci
from simeval_ir.eval.config import (
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    MetricConfig,
    run_benchmark,
)

__all__ = [
    "PROTOCOLS",
    "Protocol",
    "ProtocolResult",
    "run_protocol",
    "list_protocols",
    "bootstrap_ci",
    "BenchmarkConfig",
    "BenchmarkResult",
    "DatasetConfig",
    "MetricConfig",
    "run_benchmark",
]
