"""Core session data models for SimEval-IR.

This module defines the unified data model that works for both
traditional search sessions and conversational interactions.
"""

from dataclasses import dataclass, field
from typing import Any

from simeval_ir.core.types import EventType, Role, SessionType

# Schema version for compatibility tracking (as mentioned in paper Section 3.2)
SCHEMA_VERSION = "v1"


@dataclass
class RankedItem:
    """A document/item in a ranked list.

    Attributes:
        doc_id: Unique identifier for the document.
        rank: Position in the ranking (1-indexed).
        score: Retrieval score (optional).
        judged_relevance: Relevance judgment if available (optional).
    """

    doc_id: str
    rank: int
    score: float | None = None
    judged_relevance: int | None = None


@dataclass
class Click:
    """A click event on a ranked item.

    Attributes:
        doc_id: Identifier of the clicked document.
        rank: Position of the clicked document.
        dwell_time: Time spent on the document in seconds (optional).
        meta: Additional click metadata.
    """

    doc_id: str
    rank: int
    dwell_time: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """An event within an interaction session.

    This is the fundamental unit of interaction, representing
    queries, clicks, utterances, and other actions.

    Attributes:
        event_id: Unique identifier for this event.
        type: Type of event (query, click, utterance, etc.).
        timestamp: Time in seconds from session start (optional).
        turn_id: Conversational turn index (optional).
        role: Actor role (user, system, environment).
        query: Search query text (for search events).
        utterance: Full utterance text (for conversational events).
        ranked_items: List of ranked items shown to user (optional).
        clicked_items: List of clicked items (optional).
        dialogue_act: Dialogue act label (optional, e.g., ASK, CLARIFY).
        response_to: Event ID this is responding to (optional).
        meta: Additional event metadata.
    """

    event_id: str
    type: EventType
    timestamp: float | None = None
    turn_id: int | None = None
    role: Role | None = None
    query: str | None = None
    utterance: str | None = None
    ranked_items: list[RankedItem] | None = None
    clicked_items: list[Click] | None = None
    dialogue_act: str | None = None
    response_to: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def is_user_action(self) -> bool:
        """Check if this event is a user action."""
        return self.role == Role.USER or self.type in {
            EventType.QUERY_ISSUED,
            EventType.CLICK,
            EventType.USER_UTTERANCE,
            EventType.BOOKMARK,
            EventType.RATING,
        }

    def is_system_action(self) -> bool:
        """Check if this event is a system action."""
        return self.role == Role.SYSTEM or self.type in {
            EventType.SERP_VIEW,
            EventType.SYSTEM_UTTERANCE,
            EventType.RECOMMENDATION,
        }

    def has_text(self) -> bool:
        """Check if this event has text content."""
        return self.query is not None or self.utterance is not None

    def get_text(self) -> str | None:
        """Get the text content of this event."""
        return self.query or self.utterance


@dataclass
class Turn:
    """A conversational turn (user + system exchange).

    Convenience wrapper for grouping related events in a conversation.

    Attributes:
        turn_id: Turn index (0-indexed).
        user_event: User's event in this turn (optional).
        system_event: System's event in this turn (optional).
        other_events: Additional events in this turn (clicks, etc.).
    """

    turn_id: int
    user_event: Event | None = None
    system_event: Event | None = None
    other_events: list[Event] = field(default_factory=list)

    def all_events(self) -> list[Event]:
        """Get all events in this turn, ordered."""
        events = []
        if self.user_event:
            events.append(self.user_event)
        if self.system_event:
            events.append(self.system_event)
        events.extend(self.other_events)
        return events


@dataclass
class InteractionSession:
    """A complete interaction session.

    This is the top-level data structure representing a user's
    interaction with a system, whether search-based, conversational,
    or mixed.

    Attributes:
        session_id: Unique identifier for this session.
        dataset_id: Identifier of the source dataset.
        session_type: Type of session (search, conversational, mixed).
        events: Ordered list of events in the session.
        user_id: User identifier (optional).
        topic_id: Topic or task identifier (optional).
        domain: Domain of the session (optional).
        meta: Additional session metadata.
    """

    session_id: str
    dataset_id: str
    session_type: SessionType
    events: list[Event]
    user_id: str | None = None
    topic_id: str | None = None
    domain: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of events in the session."""
        return len(self.events)

    def __iter__(self):
        """Iterate over events in the session."""
        return iter(self.events)

    def get_queries(self) -> list[str]:
        """Extract all queries from the session."""
        return [
            e.query
            for e in self.events
            if e.type == EventType.QUERY_ISSUED and e.query is not None
        ]

    def get_utterances(self, role: Role | None = None) -> list[str]:
        """Extract all utterances from the session.

        Args:
            role: Filter by role (USER, SYSTEM, etc.). If None, return all.

        Returns:
            List of utterance texts.
        """
        utterances = []
        for e in self.events:
            if e.type in {EventType.USER_UTTERANCE, EventType.SYSTEM_UTTERANCE}:
                if role is None or e.role == role:
                    if e.utterance is not None:
                        utterances.append(e.utterance)
        return utterances

    def get_clicks(self) -> list[Click]:
        """Extract all clicks from the session."""
        clicks = []
        for e in self.events:
            if e.clicked_items:
                clicks.extend(e.clicked_items)
        return clicks

    def get_turns(self) -> list[Turn]:
        """Group events into conversational turns.

        Returns:
            List of Turn objects, one per turn in the conversation.
        """
        turns_dict: dict[int, Turn] = {}

        for event in self.events:
            turn_id = event.turn_id
            if turn_id is None:
                continue

            if turn_id not in turns_dict:
                turns_dict[turn_id] = Turn(turn_id=turn_id)

            turn = turns_dict[turn_id]
            if event.type == EventType.USER_UTTERANCE:
                turn.user_event = event
            elif event.type == EventType.SYSTEM_UTTERANCE:
                turn.system_event = event
            else:
                turn.other_events.append(event)

        return [turns_dict[k] for k in sorted(turns_dict.keys())]

    def num_queries(self) -> int:
        """Count the number of queries in the session."""
        return sum(1 for e in self.events if e.type == EventType.QUERY_ISSUED)

    def num_clicks(self) -> int:
        """Count the number of clicks in the session."""
        return sum(
            len(e.clicked_items) for e in self.events if e.clicked_items is not None
        )

    def num_turns(self) -> int:
        """Count the number of conversational turns."""
        turn_ids = {e.turn_id for e in self.events if e.turn_id is not None}
        return len(turn_ids)

    def duration(self) -> float | None:
        """Calculate the session duration in seconds.

        Returns:
            Duration in seconds, or None if timestamps are not available.
        """
        timestamps = [e.timestamp for e in self.events if e.timestamp is not None]
        if len(timestamps) < 2:
            return None
        return max(timestamps) - min(timestamps)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to a dictionary for serialization."""
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "dataset_id": self.dataset_id,
            "session_type": self.session_type.value,
            "user_id": self.user_id,
            "topic_id": self.topic_id,
            "domain": self.domain,
            "events": [
                {
                    "event_id": e.event_id,
                    "type": e.type.value,
                    "timestamp": e.timestamp,
                    "turn_id": e.turn_id,
                    "role": e.role.value if e.role else None,
                    "query": e.query,
                    "utterance": e.utterance,
                    "ranked_items": (
                        [
                            {
                                "doc_id": r.doc_id,
                                "rank": r.rank,
                                "score": r.score,
                                "judged_relevance": r.judged_relevance,
                            }
                            for r in e.ranked_items
                        ]
                        if e.ranked_items
                        else None
                    ),
                    "clicked_items": (
                        [
                            {
                                "doc_id": c.doc_id,
                                "rank": c.rank,
                                "dwell_time": c.dwell_time,
                                "meta": c.meta,
                            }
                            for c in e.clicked_items
                        ]
                        if e.clicked_items
                        else None
                    ),
                    "dialogue_act": e.dialogue_act,
                    "response_to": e.response_to,
                    "meta": e.meta,
                }
                for e in self.events
            ],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionSession":
        """Create a session from a dictionary.

        Args:
            data: Dictionary representation of a session.

        Returns:
            InteractionSession instance.

        Note:
            The schema_version field is validated but not stored in the session.
            Future versions may use this for migration.
        """
        # Validate schema version if present
        data_version = data.get("schema_version", "v1")
        if data_version != SCHEMA_VERSION:
            import warnings
            warnings.warn(
                f"Session schema version mismatch: expected {SCHEMA_VERSION}, "
                f"got {data_version}. Data may not load correctly.",
                UserWarning,
            )

        events = []
        for e_data in data.get("events", []):
            ranked_items = None
            if e_data.get("ranked_items"):
                ranked_items = [
                    RankedItem(
                        doc_id=r["doc_id"],
                        rank=r["rank"],
                        score=r.get("score"),
                        judged_relevance=r.get("judged_relevance"),
                    )
                    for r in e_data["ranked_items"]
                ]

            clicked_items = None
            if e_data.get("clicked_items"):
                clicked_items = [
                    Click(
                        doc_id=c["doc_id"],
                        rank=c["rank"],
                        dwell_time=c.get("dwell_time"),
                        meta=c.get("meta", {}),
                    )
                    for c in e_data["clicked_items"]
                ]

            event = Event(
                event_id=e_data["event_id"],
                type=EventType(e_data["type"]),
                timestamp=e_data.get("timestamp"),
                turn_id=e_data.get("turn_id"),
                role=Role(e_data["role"]) if e_data.get("role") else None,
                query=e_data.get("query"),
                utterance=e_data.get("utterance"),
                ranked_items=ranked_items,
                clicked_items=clicked_items,
                dialogue_act=e_data.get("dialogue_act"),
                response_to=e_data.get("response_to"),
                meta=e_data.get("meta", {}),
            )
            events.append(event)

        return cls(
            session_id=data["session_id"],
            dataset_id=data["dataset_id"],
            session_type=SessionType(data["session_type"]),
            events=events,
            user_id=data.get("user_id"),
            topic_id=data.get("topic_id"),
            domain=data.get("domain"),
            meta=data.get("meta", {}),
        )
