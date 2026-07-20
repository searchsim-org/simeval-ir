"""AOL Query Log adapter.

Converts AOL Query Log data to the SimEval-IR canonical format.
The AOL dataset contains ~20M queries from ~650K users collected over 3 months in 2006.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from simeval_ir.core.session import (
    InteractionSession,
    Event,
    Click,
)
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets.base import DatasetAdapter, adapter
from simeval_ir.datasets.adapters.trec_session import LossAccountingManifest


@adapter("aol")
class AOLAdapter(DatasetAdapter):
    """Adapter for AOL Query Log data.

    The AOL dataset format has columns:
    AnonID, Query, QueryTime, ItemRank, ClickURL

    Sessions are constructed using a configurable inactivity timeout.
    """

    name = "aol"
    dataset_type: set[Literal["T", "C", "A"]] = {"T"}
    domain = "web"
    description = "AOL Query Log (2006)"
    url = "https://jeffhuang.com/search_query_logs/"
    languages = ["en"]

    def __init__(self, session_timeout_minutes: int = 30):
        """Initialize AOL adapter.

        Args:
            session_timeout_minutes: Minutes of inactivity to start new session.
        """
        self.session_timeout_minutes = session_timeout_minutes
        self._manifest: LossAccountingManifest | None = None

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from AOL Query Log files.

        Args:
            data_path: Path to directory with user-ct-test-collection-*.txt files
                      or a single combined file.
            split: Split to load (ignored for AOL, loads all).

        Yields:
            InteractionSession objects.
        """
        self._manifest = LossAccountingManifest(
            source_format="aol-query-log",
            fields_mapped=[
                "anon_id", "query", "query_time", "item_rank", "click_url"
            ],
            fields_lost=[
                "serp_results",  # Full SERP not available
                "dwell_time",  # Not recorded
                "document_content",  # URLs only
            ],
            transformation_notes=[
                f"Sessions segmented using {self.session_timeout_minutes}min inactivity timeout",
                "Click ranks extracted from ItemRank column",
                "URLs used as document IDs (no content available)",
            ],
        )

        if data_path.is_file():
            data_files = [data_path]
        else:
            data_files = sorted(data_path.glob("user-ct-test-collection-*.txt"))
            if not data_files:
                data_files = sorted(data_path.glob("*.txt"))

        # Group queries by user
        user_queries: dict[str, list[dict]] = {}

        for data_file in data_files:
            try:
                with open(data_file, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f, delimiter="\t")
                    next(reader, None)  # Skip header

                    for row in reader:
                        if len(row) < 3:
                            continue

                        anon_id = row[0]
                        query = row[1]
                        query_time = row[2]
                        item_rank = row[3] if len(row) > 3 else None
                        click_url = row[4] if len(row) > 4 else None

                        if anon_id not in user_queries:
                            user_queries[anon_id] = []

                        user_queries[anon_id].append({
                            "query": query,
                            "time": query_time,
                            "rank": item_rank,
                            "url": click_url,
                        })
            except Exception as e:
                self._manifest.transformation_notes.append(
                    f"Error reading {data_file.name}: {str(e)}"
                )

        # Convert to sessions per user
        for user_id, queries in user_queries.items():
            # Sort by time
            queries.sort(key=lambda x: x["time"])

            # Segment into sessions
            sessions = self._segment_sessions(user_id, queries)
            for session in sessions:
                self._manifest.sessions_processed += 1
                yield session

    def _segment_sessions(
        self,
        user_id: str,
        queries: list[dict],
    ) -> Iterator[InteractionSession]:
        """Segment user queries into sessions based on time gaps."""
        if not queries:
            return

        timeout_seconds = self.session_timeout_minutes * 60
        session_queries = []
        last_time = None
        session_idx = 0

        for q in queries:
            try:
                current_time = datetime.strptime(
                    q["time"], "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                continue

            if last_time is not None:
                gap = (current_time - last_time).total_seconds()
                if gap > timeout_seconds and session_queries:
                    # Emit current session
                    session = self._build_session(
                        user_id, session_idx, session_queries
                    )
                    if session:
                        yield session
                    session_queries = []
                    session_idx += 1

            session_queries.append({**q, "datetime": current_time})
            last_time = current_time

        # Emit final session
        if session_queries:
            session = self._build_session(user_id, session_idx, session_queries)
            if session:
                yield session

    def _build_session(
        self,
        user_id: str,
        session_idx: int,
        queries: list[dict],
    ) -> InteractionSession | None:
        """Build a session from a list of queries."""
        if not queries:
            return None

        events = []
        event_counter = 0
        base_time = queries[0]["datetime"]

        current_query = None
        current_clicks = []

        for q in queries:
            timestamp = (q["datetime"] - base_time).total_seconds()

            # New query
            if q["query"] != current_query:
                # Emit pending clicks for previous query
                if current_clicks:
                    event_counter += 1
                    events.append(Event(
                        event_id=f"aol-{user_id}-{session_idx}-e{event_counter}",
                        type=EventType.CLICK,
                        timestamp=events[-1].timestamp + 0.5 if events else 0,
                        role=Role.USER,
                        clicked_items=current_clicks,
                    ))
                    current_clicks = []

                current_query = q["query"]
                event_counter += 1
                events.append(Event(
                    event_id=f"aol-{user_id}-{session_idx}-e{event_counter}",
                    type=EventType.QUERY_ISSUED,
                    timestamp=timestamp,
                    role=Role.USER,
                    query=current_query,
                ))

            # Record click if present
            if q.get("url") and q["url"].strip():
                rank = 0
                if q.get("rank"):
                    try:
                        rank = int(q["rank"])
                    except ValueError:
                        pass

                current_clicks.append(Click(
                    doc_id=q["url"],
                    rank=rank,
                ))

        # Emit final clicks
        if current_clicks:
            event_counter += 1
            events.append(Event(
                event_id=f"aol-{user_id}-{session_idx}-e{event_counter}",
                type=EventType.CLICK,
                timestamp=events[-1].timestamp + 0.5 if events else 0,
                role=Role.USER,
                clicked_items=current_clicks,
            ))

        if not events:
            return None

        return InteractionSession(
            session_id=f"aol-{user_id}-{session_idx}",
            dataset_id="aol",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=user_id,
            domain="web",
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains AOL files."""
        if data_path.is_file():
            return data_path.suffix == ".txt"
        return any(data_path.glob("*.txt"))

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
