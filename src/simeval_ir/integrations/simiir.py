"""SimIIR integration for SimEval-IR.

This module provides utilities for loading and converting
SimIIR log files to InteractionSession objects.
"""

import json
from pathlib import Path
from typing import Iterator

from simeval_ir.core.session import Click, Event, InteractionSession, RankedItem
from simeval_ir.core.types import EventType, Role, SessionType


def parse_simiir_log(path: Path) -> Iterator[InteractionSession]:
    """Parse SimIIR log files into InteractionSession objects.

    Supports both JSON and JSONL formats from SimIIR.

    Args:
        path: Path to log file or directory.

    Yields:
        InteractionSession objects.
    """
    path = Path(path)

    if path.is_dir():
        # Process all log files in directory
        for log_file in path.glob("*.json"):
            yield from _parse_log_file(log_file)
        for log_file in path.glob("*.jsonl"):
            yield from _parse_log_file(log_file)
    else:
        yield from _parse_log_file(path)


def _parse_log_file(path: Path) -> Iterator[InteractionSession]:
    """Parse a single SimIIR log file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSONL format first
    if content.startswith("{"):
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line:
                data = json.loads(line)
                yield _convert_simiir_session(data)
    else:
        # Try JSON array format
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                yield _convert_simiir_session(item)
        else:
            yield _convert_simiir_session(data)


def _convert_simiir_session(data: dict) -> InteractionSession:
    """Convert a SimIIR log entry to an InteractionSession.

    This handles the common SimIIR log format with actions like:
    - QUERY: query reformulation
    - SERP: viewing search results
    - DOC: examining/clicking a document
    - MARK: marking a document as relevant
    """
    session_id = data.get("session_id", data.get("id", "unknown"))
    user_id = data.get("user_id")
    topic_id = data.get("topic_id", data.get("topic"))

    events = []
    event_idx = 0

    for action in data.get("actions", data.get("log", [])):
        action_type = action.get("action", action.get("type", "")).upper()
        timestamp = action.get("time", action.get("timestamp"))

        if action_type in ("QUERY", "QUERY_REFORMULATION"):
            event = Event(
                event_id=f"{session_id}_e{event_idx}",
                type=EventType.QUERY_ISSUED,
                timestamp=float(timestamp) if timestamp else None,
                role=Role.USER,
                query=action.get("query", action.get("text", "")),
                meta={"original_action": action_type},
            )
            events.append(event)

        elif action_type == "SERP":
            ranked_items = None
            if "results" in action:
                ranked_items = [
                    RankedItem(
                        doc_id=r.get("doc_id", r.get("docid", str(i))),
                        rank=r.get("rank", i + 1),
                        score=r.get("score"),
                        judged_relevance=r.get("relevance"),
                    )
                    for i, r in enumerate(action["results"])
                ]

            event = Event(
                event_id=f"{session_id}_e{event_idx}",
                type=EventType.SERP_VIEW,
                timestamp=float(timestamp) if timestamp else None,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            )
            events.append(event)

        elif action_type in ("DOC", "CLICK", "EXAMINE"):
            doc_id = action.get("doc_id", action.get("docid", "unknown"))
            rank = action.get("rank", action.get("position", 0))
            dwell_time = action.get("dwell_time", action.get("time_spent"))

            click = Click(
                doc_id=doc_id,
                rank=int(rank),
                dwell_time=float(dwell_time) if dwell_time else None,
                meta={
                    "relevance": action.get("relevance"),
                    "judged": action.get("judged", False),
                },
            )

            event = Event(
                event_id=f"{session_id}_e{event_idx}",
                type=EventType.CLICK,
                timestamp=float(timestamp) if timestamp else None,
                role=Role.USER,
                clicked_items=[click],
            )
            events.append(event)

        elif action_type in ("MARK", "SAVE", "BOOKMARK"):
            event = Event(
                event_id=f"{session_id}_e{event_idx}",
                type=EventType.BOOKMARK,
                timestamp=float(timestamp) if timestamp else None,
                role=Role.USER,
                meta={
                    "doc_id": action.get("doc_id", action.get("docid")),
                    "relevance": action.get("relevance"),
                },
            )
            events.append(event)

        else:
            # Generic event for other action types
            event = Event(
                event_id=f"{session_id}_e{event_idx}",
                type=EventType.OTHER,
                timestamp=float(timestamp) if timestamp else None,
                role=Role.UNKNOWN,
                meta={"original_action": action_type, "data": action},
            )
            events.append(event)

        event_idx += 1

    return InteractionSession(
        session_id=str(session_id),
        dataset_id="simiir",
        session_type=SessionType.SEARCH,
        events=events,
        user_id=str(user_id) if user_id else None,
        topic_id=str(topic_id) if topic_id else None,
        domain="web",
        meta=data.get("meta", {}),
    )


def from_simiir_logs(log_paths: list[Path] | Path) -> list[InteractionSession]:
    """Convenience wrapper for loading multiple SimIIR logs.

    Args:
        log_paths: Single path or list of paths to log files/directories.

    Returns:
        List of InteractionSession objects.
    """
    if isinstance(log_paths, Path):
        log_paths = [log_paths]

    sessions = []
    for path in log_paths:
        sessions.extend(parse_simiir_log(Path(path)))

    return sessions
