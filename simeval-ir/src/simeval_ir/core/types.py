"""Type definitions and enums for SimEval-IR."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionType(str, Enum):
    """Type of interaction session."""

    SEARCH = "search"
    CONVERSATIONAL = "conversational"
    MIXED = "mixed"


class EventType(str, Enum):
    """Type of event within a session.

    Note: Aliases are provided to match paper terminology (Section 3.2):
    - QUERY = QUERY_ISSUED
    - CONV_USER = USER_UTTERANCE
    - CONV_SYSTEM = SYSTEM_UTTERANCE
    """

    # Search events
    QUERY_ISSUED = "query_issued"
    SERP_VIEW = "serp_view"
    CLICK = "click"
    DWELL = "dwell"
    BOOKMARK = "bookmark"
    NAVIGATION = "navigation"

    # Conversational events
    USER_UTTERANCE = "user_utterance"
    SYSTEM_UTTERANCE = "system_utterance"

    # Feedback events
    FEEDBACK = "feedback"
    RATING = "rating"

    # Recommendation events
    RECOMMENDATION = "recommendation"

    # Other
    OTHER = "other"


# Paper terminology aliases (Section 3.2 schema)
# These allow using either naming convention
EventType.QUERY = EventType.QUERY_ISSUED
EventType.CONV_USER = EventType.USER_UTTERANCE
EventType.CONV_SYSTEM = EventType.SYSTEM_UTTERANCE


class Role(str, Enum):
    """Role of the actor in an event."""

    USER = "user"
    SYSTEM = "system"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


@dataclass
class MetricResult:
    """Result of a metric computation.

    Attributes:
        name: Name of the metric.
        value: Aggregated metric value.
        ci_lower: Lower bound of confidence interval (optional).
        ci_upper: Upper bound of confidence interval (optional).
        per_item: Per-session or per-system results (optional).
        meta: Additional metadata.
    """

    name: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    per_item: dict[str, float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.ci_lower is not None and self.ci_upper is not None:
            return f"{self.name}: {self.value:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return f"{self.name}: {self.value:.4f}"


@dataclass
class DatasetMetadata:
    """Metadata about a dataset.

    Attributes:
        name: Dataset name.
        dataset_type: Set of type flags (T=traditional, C=conversational, A=annotated).
        domain: Domain of the dataset (e.g., "web", "health", "academic").
        description: Short description.
        size: Number of sessions (if known).
        languages: List of languages in the dataset.
        url: URL to dataset documentation or download.
    """

    name: str
    dataset_type: set[str]
    domain: str
    description: str = ""
    size: int | None = None
    languages: list[str] = field(default_factory=lambda: ["en"])
    url: str | None = None
