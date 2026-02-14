"""I/O utilities for loading and saving sessions.

This module provides functions to load and save InteractionSession
objects from/to JSON and JSONL files, enabling dataset-agnostic
usage of the package.
"""

import json
from pathlib import Path
from typing import Iterator

from simeval_ir.core.session import InteractionSession


def load_sessions_from_json(path: Path) -> list[InteractionSession]:
    """Load sessions from a JSON file.

    The JSON file should contain a list of session objects.

    Args:
        path: Path to the JSON file.

    Returns:
        List of InteractionSession objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "sessions" in data:
        # Handle wrapped format: {"sessions": [...]}
        sessions_data = data["sessions"]
    elif isinstance(data, list):
        sessions_data = data
    else:
        raise ValueError(
            f"Invalid JSON format. Expected list or {{'sessions': [...]}}."
        )

    return [InteractionSession.from_dict(s) for s in sessions_data]


def load_sessions_from_jsonl(path: Path) -> Iterator[InteractionSession]:
    """Load sessions from a JSONL file.

    Each line should be a JSON object representing a session.

    Args:
        path: Path to the JSONL file.

    Yields:
        InteractionSession objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield InteractionSession.from_dict(data)


def save_sessions_to_json(
    sessions: list[InteractionSession],
    path: Path,
    pretty: bool = True,
) -> None:
    """Save sessions to a JSON file.

    Args:
        sessions: List of sessions to save.
        path: Output file path.
        pretty: Whether to pretty-print the JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"sessions": [s.to_dict() for s in sessions]}
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)


def save_sessions_to_jsonl(
    sessions: Iterator[InteractionSession] | list[InteractionSession],
    path: Path,
) -> int:
    """Save sessions to a JSONL file.

    Args:
        sessions: Sessions to save (can be iterator or list).
        path: Output file path.

    Returns:
        Number of sessions saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for session in sessions:
            f.write(json.dumps(session.to_dict(), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count
