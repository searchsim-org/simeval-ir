"""
SimEval-IR: A unified framework for evaluating simulated search and conversational sessions.

This package provides:
- Standard data models for interaction sessions (search and conversational)
- Behavioral realism metrics (FD, JSD, classifier-based)
- System effectiveness metrics (session nDCG, EGU, conversational metrics)
- Tester reliability metrics (Kendall τ, RATE)
- Integration with SimIIR and PyTerrier
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
    # Version
    "__version__",
]
