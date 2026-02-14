"""TripClick adapter.

Converts TripClick health search data to the SimEval-IR canonical format.
TripClick is a large-scale dataset of click logs from the Trip medical search engine.
"""

import json
from pathlib import Path
from typing import Iterator, Literal

from simeval_ir.core.session import (
    InteractionSession,
    Event,
    RankedItem,
    Click,
)
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets.base import DatasetAdapter, adapter
from simeval_ir.datasets.adapters.trec_session import LossAccountingManifest


@adapter("tripclick")
class TripClickAdapter(DatasetAdapter):
    """Adapter for TripClick health search data.

    TripClick provides:
    - Click logs with queries and clicked documents
    - Document collection
    - Relevance judgments (qrels)
    """

    name = "tripclick"
    dataset_type: set[Literal["T", "C", "A"]] = {"T"}
    domain = "health"
    description = "TripClick Health Search (2021)"
    url = "https://tripdatabase.github.io/tripclick/"
    languages = ["en"]

    def __init__(self, include_qrels: bool = True):
        """Initialize TripClick adapter.

        Args:
            include_qrels: Whether to include relevance judgments in ranked items.
        """
        self.include_qrels = include_qrels
        self._manifest: LossAccountingManifest | None = None
        self._qrels: dict[str, dict[str, int]] = {}

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from TripClick data.

        Args:
            data_path: Path to TripClick directory containing:
                      - clicks/ or sessions/ directory
                      - qrels/ directory (optional)
            split: "train", "dev", "test", or "all"

        Yields:
            InteractionSession objects.
        """
        self._manifest = LossAccountingManifest(
            source_format="tripclick",
            fields_mapped=[
                "session_id", "query", "clicked_docs", "impressions", "qrels"
            ],
            fields_lost=[
                "timestamps",  # Not consistently available
                "dwell_time",  # Not recorded
                "user_demographics",  # Not available
            ],
            transformation_notes=[
                "Sessions reconstructed from query-click pairs",
                "Relevance judgments merged when available",
            ],
        )

        # Load qrels if available
        qrels_path = data_path / "qrels"
        if qrels_path.exists() and self.include_qrels:
            self._load_qrels(qrels_path, split)

        # Determine which files to load
        sessions_path = data_path / "sessions"
        if not sessions_path.exists():
            sessions_path = data_path / "clicks"
        if not sessions_path.exists():
            sessions_path = data_path

        # Load based on split
        if split == "all":
            patterns = ["*.jsonl", "*.json", "*.tsv"]
        else:
            patterns = [f"*{split}*.jsonl", f"*{split}*.json", f"*{split}*.tsv"]

        for pattern in patterns:
            for file_path in sessions_path.glob(pattern):
                if file_path.suffix == ".jsonl":
                    yield from self._load_jsonl(file_path)
                elif file_path.suffix == ".json":
                    yield from self._load_json(file_path)
                elif file_path.suffix == ".tsv":
                    yield from self._load_tsv(file_path)

    def _load_qrels(self, qrels_path: Path, split: str) -> None:
        """Load relevance judgments."""
        for qrels_file in qrels_path.glob("*.txt"):
            with open(qrels_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        query_id, _, doc_id, rel = parts[:4]
                        if query_id not in self._qrels:
                            self._qrels[query_id] = {}
                        self._qrels[query_id][doc_id] = int(rel)

    def _load_jsonl(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load sessions from JSONL format."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    session = self._parse_session(data, line_num)
                    if session:
                        self._manifest.sessions_processed += 1
                        yield session
                except json.JSONDecodeError:
                    self._manifest.sessions_skipped += 1

    def _load_json(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load sessions from JSON format."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            sessions_data = data
        elif isinstance(data, dict) and "sessions" in data:
            sessions_data = data["sessions"]
        else:
            sessions_data = [data]

        for idx, session_data in enumerate(sessions_data):
            session = self._parse_session(session_data, idx)
            if session:
                self._manifest.sessions_processed += 1
                yield session

    def _load_tsv(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load sessions from TSV format (query-click pairs)."""
        # Group by session/query
        sessions: dict[str, list[dict]] = {}

        with open(file_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                row = dict(zip(header, parts))
                session_id = row.get("session_id") or row.get("query_id") or row.get("qid")

                if session_id:
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(row)

        for session_id, rows in sessions.items():
            session = self._parse_tsv_session(session_id, rows)
            if session:
                self._manifest.sessions_processed += 1
                yield session

    def _parse_session(self, data: dict, idx: int) -> InteractionSession | None:
        """Parse a session from dictionary data."""
        session_id = data.get("session_id") or data.get("id") or f"tripclick-{idx}"
        query = data.get("query") or data.get("q")
        query_id = data.get("query_id") or data.get("qid")

        if not query:
            return None

        events = []
        event_counter = 0

        # Query event
        event_counter += 1
        events.append(Event(
            event_id=f"{session_id}-e{event_counter}",
            type=EventType.QUERY_ISSUED,
            timestamp=0.0,
            role=Role.USER,
            query=query,
        ))

        # SERP/impressions
        impressions = data.get("impressions") or data.get("docs") or data.get("results")
        if impressions:
            ranked_items = []
            for rank, doc in enumerate(impressions, start=1):
                if isinstance(doc, str):
                    doc_id = doc
                else:
                    doc_id = doc.get("doc_id") or doc.get("docno") or doc.get("id")

                rel = None
                if query_id and query_id in self._qrels:
                    rel = self._qrels[query_id].get(doc_id)

                ranked_items.append(RankedItem(
                    doc_id=doc_id,
                    rank=rank,
                    judged_relevance=rel,
                ))

            event_counter += 1
            events.append(Event(
                event_id=f"{session_id}-e{event_counter}",
                type=EventType.SERP_VIEW,
                timestamp=0.1,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            ))

        # Clicks
        clicks_data = data.get("clicks") or data.get("clicked_docs")
        if clicks_data:
            clicks = []
            for click in clicks_data:
                if isinstance(click, str):
                    doc_id = click
                    rank = 0
                else:
                    doc_id = click.get("doc_id") or click.get("docno")
                    rank = click.get("rank") or click.get("position") or 0

                clicks.append(Click(doc_id=doc_id, rank=rank))

            if clicks:
                event_counter += 1
                events.append(Event(
                    event_id=f"{session_id}-e{event_counter}",
                    type=EventType.CLICK,
                    timestamp=1.0,
                    role=Role.USER,
                    clicked_items=clicks,
                ))

        return InteractionSession(
            session_id=f"tripclick-{session_id}",
            dataset_id="tripclick",
            session_type=SessionType.SEARCH,
            events=events,
            topic_id=query_id,
            domain="health",
        )

    def _parse_tsv_session(
        self,
        session_id: str,
        rows: list[dict],
    ) -> InteractionSession | None:
        """Parse session from TSV rows."""
        if not rows:
            return None

        query = rows[0].get("query") or rows[0].get("q")
        query_id = rows[0].get("query_id") or rows[0].get("qid")

        if not query:
            return None

        events = []

        # Query event
        events.append(Event(
            event_id=f"{session_id}-e1",
            type=EventType.QUERY_ISSUED,
            timestamp=0.0,
            role=Role.USER,
            query=query,
        ))

        # Collect clicks
        clicks = []
        for row in rows:
            doc_id = row.get("doc_id") or row.get("docno") or row.get("clicked_doc")
            if doc_id:
                rank = int(row.get("rank") or row.get("position") or 0)
                clicks.append(Click(doc_id=doc_id, rank=rank))

        if clicks:
            events.append(Event(
                event_id=f"{session_id}-e2",
                type=EventType.CLICK,
                timestamp=1.0,
                role=Role.USER,
                clicked_items=clicks,
            ))

        return InteractionSession(
            session_id=f"tripclick-{session_id}",
            dataset_id="tripclick",
            session_type=SessionType.SEARCH,
            events=events,
            topic_id=query_id,
            domain="health",
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains TripClick files."""
        if data_path.is_file():
            return data_path.suffix in {".json", ".jsonl", ".tsv"}
        return (
            any(data_path.glob("*.jsonl")) or
            any(data_path.glob("*.json")) or
            any(data_path.glob("*.tsv")) or
            (data_path / "sessions").exists() or
            (data_path / "clicks").exists()
        )

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
