"""Sogou Query Log adapter.

Converts Sogou search log data to the SimEval-IR canonical format.
Sogou provides Chinese web search query logs with click data.
"""

from pathlib import Path
from typing import Iterator, Literal
from datetime import datetime

from simeval_ir.core.session import (
    InteractionSession,
    Event,
    Click,
)
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets.base import DatasetAdapter, adapter
from simeval_ir.datasets.adapters.trec_session import LossAccountingManifest


@adapter("sogou")
class SogouAdapter(DatasetAdapter):
    """Adapter for Sogou Query Log data.

    Sogou dataset format (tab-separated):
    AccessTime [Query] DestinationURL UserID [ClickOrder] [ClickRank]

    The exact format varies by release year.
    """

    name = "sogou"
    dataset_type: set[Literal["T", "C", "A"]] = {"T"}
    domain = "web"
    description = "Sogou Query Log (Chinese Web Search)"
    url = "https://www.sogou.com/labs/resource/list_querylog.php"
    languages = ["zh"]

    def __init__(self, session_timeout_minutes: int = 30):
        """Initialize Sogou adapter.

        Args:
            session_timeout_minutes: Minutes of inactivity to segment sessions.
        """
        self.session_timeout_minutes = session_timeout_minutes
        self._manifest: LossAccountingManifest | None = None

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from Sogou Query Log files.

        Args:
            data_path: Path to Sogou data file(s).
            split: Split to load (ignored for Sogou, loads all).

        Yields:
            InteractionSession objects.
        """
        self._manifest = LossAccountingManifest(
            source_format="sogou-query-log",
            fields_mapped=[
                "access_time", "query", "destination_url", "user_id",
                "click_order", "click_rank"
            ],
            fields_lost=[
                "serp_results",  # Full SERP not available
                "dwell_time",  # Not recorded
                "query_suggestions",  # Not available
            ],
            transformation_notes=[
                "Chinese text preserved without translation",
                f"Sessions segmented using {self.session_timeout_minutes}min inactivity timeout",
                "Click ranks extracted when available",
            ],
        )

        if data_path.is_file():
            data_files = [data_path]
        else:
            data_files = list(data_path.glob("*.txt")) + list(data_path.glob("*.log"))

        # Group by user
        user_queries: dict[str, list[dict]] = {}

        for data_file in data_files:
            try:
                # Try different encodings common for Chinese text
                for encoding in ["utf-8", "gb18030", "gbk", "gb2312"]:
                    try:
                        with open(data_file, "r", encoding=encoding) as f:
                            for line in f:
                                record = self._parse_line(line)
                                if record:
                                    user_id = record["user_id"]
                                    if user_id not in user_queries:
                                        user_queries[user_id] = []
                                    user_queries[user_id].append(record)
                        break  # Success, exit encoding loop
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                self._manifest.transformation_notes.append(
                    f"Error reading {data_file.name}: {str(e)}"
                )

        # Convert to sessions
        for user_id, queries in user_queries.items():
            queries.sort(key=lambda x: x.get("time", ""))
            yield from self._segment_sessions(user_id, queries)

    def _parse_line(self, line: str) -> dict | None:
        """Parse a single line from Sogou log."""
        parts = line.strip().split("\t")
        if len(parts) < 4:
            return None

        # Common format: AccessTime [Query] DestinationURL UserID [ClickOrder] [ClickRank]
        access_time = parts[0]
        query = parts[1].strip("[]") if len(parts) > 1 else ""
        dest_url = parts[2] if len(parts) > 2 else ""
        user_id = parts[3] if len(parts) > 3 else ""

        click_order = None
        click_rank = None
        if len(parts) > 4:
            try:
                click_order = int(parts[4])
            except ValueError:
                pass
        if len(parts) > 5:
            try:
                click_rank = int(parts[5])
            except ValueError:
                pass

        if not query or not user_id:
            return None

        return {
            "time": access_time,
            "query": query,
            "url": dest_url,
            "user_id": user_id,
            "click_order": click_order,
            "click_rank": click_rank,
        }

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
            current_time = self._parse_time(q.get("time", ""))

            if current_time and last_time:
                gap = (current_time - last_time).total_seconds()
                if gap > timeout_seconds and session_queries:
                    session = self._build_session(user_id, session_idx, session_queries)
                    if session:
                        self._manifest.sessions_processed += 1
                        yield session
                    session_queries = []
                    session_idx += 1

            session_queries.append({**q, "datetime": current_time})
            if current_time:
                last_time = current_time

        # Emit final session
        if session_queries:
            session = self._build_session(user_id, session_idx, session_queries)
            if session:
                self._manifest.sessions_processed += 1
                yield session

    def _parse_time(self, time_str: str) -> datetime | None:
        """Parse Sogou timestamp format."""
        formats = [
            "%Y%m%d%H%M%S",
            "%Y-%m-%d %H:%M:%S",
            "%H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        return None

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
        base_time = queries[0].get("datetime")

        current_query = None
        current_clicks = []

        for q in queries:
            timestamp = 0.0
            if q.get("datetime") and base_time:
                timestamp = (q["datetime"] - base_time).total_seconds()

            # New query
            if q["query"] != current_query:
                # Emit pending clicks
                if current_clicks:
                    event_counter += 1
                    events.append(Event(
                        event_id=f"sogou-{user_id}-{session_idx}-e{event_counter}",
                        type=EventType.CLICK,
                        timestamp=events[-1].timestamp + 0.5 if events else 0,
                        role=Role.USER,
                        clicked_items=current_clicks,
                    ))
                    current_clicks = []

                current_query = q["query"]
                event_counter += 1
                events.append(Event(
                    event_id=f"sogou-{user_id}-{session_idx}-e{event_counter}",
                    type=EventType.QUERY_ISSUED,
                    timestamp=timestamp,
                    role=Role.USER,
                    query=current_query,
                ))

            # Record click
            if q.get("url") and q["url"].strip():
                rank = q.get("click_rank") or 0
                current_clicks.append(Click(
                    doc_id=q["url"],
                    rank=rank,
                ))

        # Emit final clicks
        if current_clicks:
            event_counter += 1
            events.append(Event(
                event_id=f"sogou-{user_id}-{session_idx}-e{event_counter}",
                type=EventType.CLICK,
                timestamp=events[-1].timestamp + 0.5 if events else 0,
                role=Role.USER,
                clicked_items=current_clicks,
            ))

        if not events:
            return None

        return InteractionSession(
            session_id=f"sogou-{user_id}-{session_idx}",
            dataset_id="sogou",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=user_id,
            domain="web",
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains Sogou files."""
        if data_path.is_file():
            return data_path.suffix in {".txt", ".log"}
        return any(data_path.glob("*.txt")) or any(data_path.glob("*.log"))

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
