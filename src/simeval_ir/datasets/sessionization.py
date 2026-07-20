"""Configurable sessionization for raw event streams.

Provides inactivity-based session boundary detection with
dataset-typed defaults as described in the paper (Section 4).
"""

from simeval_ir.core.session import InteractionSession, Event
from simeval_ir.core.types import SessionType


# Dataset-typed defaults (Section 4)
SESSIONIZATION_DEFAULTS = {
    "web": 30 * 60,       # 30-minute inactivity gap for web logs
    "academic": 60 * 60,  # 60-minute gap for academic search
    "health": 30 * 60,    # 30-minute gap for health search
    "conversational": 60 * 60,  # 60-minute gap for conversations
}


def sessionize_events(
    events: list[Event],
    timeout_seconds: float | None = None,
    domain: str = "web",
    dataset_id: str = "unknown",
    session_type: SessionType = SessionType.SEARCH,
    user_id: str | None = None,
) -> list[InteractionSession]:
    """Group raw events into sessions by inactivity gap.

    Args:
        events: List of events sorted by timestamp.
        timeout_seconds: Inactivity threshold in seconds. If None,
            uses the dataset-typed default based on domain.
        domain: Domain type for default timeout lookup.
        dataset_id: Dataset identifier for created sessions.
        session_type: Type of sessions to create.
        user_id: Optional user identifier.

    Returns:
        List of InteractionSession objects.
    """
    if not events:
        return []

    if timeout_seconds is None:
        timeout_seconds = SESSIONIZATION_DEFAULTS.get(domain, 30 * 60)

    # Sort events by timestamp
    timed_events = [e for e in events if e.timestamp is not None]
    untimed_events = [e for e in events if e.timestamp is None]
    timed_events.sort(key=lambda e: e.timestamp)

    if not timed_events:
        # All events lack timestamps - put in one session
        return [
            InteractionSession(
                session_id=f"{dataset_id}-session-0",
                dataset_id=dataset_id,
                session_type=session_type,
                events=untimed_events,
                user_id=user_id,
                domain=domain,
                meta={"sessionization_timeout": timeout_seconds},
            )
        ]

    sessions = []
    current_events = [timed_events[0]]

    for event in timed_events[1:]:
        prev_ts = current_events[-1].timestamp
        if event.timestamp - prev_ts > timeout_seconds:
            # New session
            sessions.append(
                InteractionSession(
                    session_id=f"{dataset_id}-session-{len(sessions)}",
                    dataset_id=dataset_id,
                    session_type=session_type,
                    events=current_events,
                    user_id=user_id,
                    domain=domain,
                    meta={"sessionization_timeout": timeout_seconds},
                )
            )
            current_events = [event]
        else:
            current_events.append(event)

    # Last session
    if current_events:
        sessions.append(
            InteractionSession(
                session_id=f"{dataset_id}-session-{len(sessions)}",
                dataset_id=dataset_id,
                session_type=session_type,
                events=current_events,
                user_id=user_id,
                domain=domain,
                meta={"sessionization_timeout": timeout_seconds},
            )
        )

    return sessions
