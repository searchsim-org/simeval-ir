"""TianGong-ST adapter.

Converts TianGong-ST session search data to the SimEval-IR canonical format.
TianGong-ST is a Chinese session search dataset from Sogou with rich behavioral signals.
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


@adapter("tiangong-st")
class TianGongSTAdapter(DatasetAdapter):
    """Adapter for TianGong-ST session search data.

    TianGong-ST provides:
    - Search sessions with multiple queries
    - Click and skip annotations
    - Relevance judgments
    - Rich behavioral features (dwell time, viewport, etc.)
    """

    name = "tiangong-st"
    dataset_type: set[Literal["T", "C", "A"]] = {"T"}
    domain = "web"
    description = "TianGong-ST Chinese Session Search"
    url = "http://www.thuir.cn/tiangong-st/"
    languages = ["zh"]

    def __init__(self):
        self._manifest: LossAccountingManifest | None = None
        self._qrels: dict[str, dict[str, int]] = {}

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from TianGong-ST data.

        Args:
            data_path: Path to TianGong-ST directory.
            split: "train", "dev", "test", or "all"

        Yields:
            InteractionSession objects.
        """
        self._manifest = LossAccountingManifest(
            source_format="tiangong-st",
            fields_mapped=[
                "session_id", "query", "query_time", "serp", "clicks",
                "dwell_time", "relevance_labels", "satisfaction"
            ],
            fields_lost=[
                "viewport_data",  # Complex viewport tracking not converted
                "scroll_events",  # Detailed scroll data not converted
                "mouse_movements",  # Fine-grained mouse data not converted
            ],
            transformation_notes=[
                "Chinese text preserved without translation",
                "Dwell time converted to seconds",
                "Session satisfaction labels included in metadata",
            ],
        )

        # Load qrels if available
        for qrels_file in data_path.glob("*qrels*"):
            self._load_qrels(qrels_file)

        # Load session files
        session_files = []
        if split == "all":
            session_files = list(data_path.glob("*.json")) + list(data_path.glob("*.jsonl"))
        else:
            session_files = list(data_path.glob(f"*{split}*.json")) + list(data_path.glob(f"*{split}*.jsonl"))

        for session_file in session_files:
            if "qrels" in session_file.name.lower():
                continue
            yield from self._load_session_file(session_file)

    def _load_qrels(self, qrels_file: Path) -> None:
        """Load relevance judgments."""
        if qrels_file.suffix == ".json":
            with open(qrels_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._qrels = data
        else:
            with open(qrels_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        query_id, _, doc_id, rel = parts[:4]
                        if query_id not in self._qrels:
                            self._qrels[query_id] = {}
                        self._qrels[query_id][doc_id] = int(rel)

    def _load_session_file(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load sessions from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            session = self._parse_session(data)
                            if session:
                                self._manifest.sessions_processed += 1
                                yield session
                        except json.JSONDecodeError:
                            self._manifest.sessions_skipped += 1
            else:
                data = json.load(f)
                sessions = data if isinstance(data, list) else data.get("sessions", [data])
                for session_data in sessions:
                    session = self._parse_session(session_data)
                    if session:
                        self._manifest.sessions_processed += 1
                        yield session

    def _parse_session(self, data: dict) -> InteractionSession | None:
        """Parse a session from dictionary data."""
        session_id = data.get("session_id") or data.get("sid") or data.get("id")
        if session_id is None:
            return None

        session_id = str(session_id)
        queries = data.get("queries") or data.get("interactions") or []

        if not queries and "query" in data:
            queries = [data]

        events = []
        event_counter = 0
        base_timestamp = 0.0

        for query_data in queries:
            query_text = query_data.get("query") or query_data.get("q") or query_data.get("text")
            if not query_text:
                continue

            query_id = query_data.get("query_id") or query_data.get("qid")
            timestamp = query_data.get("timestamp") or query_data.get("time") or base_timestamp

            if isinstance(timestamp, str):
                try:
                    timestamp = float(timestamp)
                except ValueError:
                    timestamp = base_timestamp

            # Query event
            event_counter += 1
            events.append(Event(
                event_id=f"tg-{session_id}-e{event_counter}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=query_text,
            ))

            # SERP event
            serp = query_data.get("serp") or query_data.get("results") or query_data.get("docs")
            if serp:
                ranked_items = []
                for rank, doc in enumerate(serp, start=1):
                    if isinstance(doc, str):
                        doc_id = doc
                        score = None
                    else:
                        doc_id = doc.get("doc_id") or doc.get("docno") or doc.get("url") or doc.get("id")
                        score = doc.get("score")

                    if doc_id:
                        rel = None
                        if query_id and query_id in self._qrels:
                            rel = self._qrels[query_id].get(str(doc_id))

                        ranked_items.append(RankedItem(
                            doc_id=str(doc_id),
                            rank=rank,
                            score=float(score) if score else None,
                            judged_relevance=rel,
                        ))

                if ranked_items:
                    event_counter += 1
                    events.append(Event(
                        event_id=f"tg-{session_id}-e{event_counter}",
                        type=EventType.SERP_VIEW,
                        timestamp=timestamp + 0.1,
                        role=Role.SYSTEM,
                        ranked_items=ranked_items,
                    ))

            # Click events
            clicks_data = query_data.get("clicks") or query_data.get("clicked") or []
            if clicks_data:
                clicks = []
                for click in clicks_data:
                    if isinstance(click, (str, int)):
                        doc_id = str(click)
                        rank = 0
                        dwell = None
                    else:
                        doc_id = click.get("doc_id") or click.get("docno") or click.get("url")
                        rank = click.get("rank") or click.get("position") or 0
                        dwell = click.get("dwell") or click.get("dwell_time") or click.get("reading_time")

                    if doc_id:
                        clicks.append(Click(
                            doc_id=str(doc_id),
                            rank=int(rank),
                            dwell_time=float(dwell) if dwell else None,
                        ))

                if clicks:
                    event_counter += 1
                    events.append(Event(
                        event_id=f"tg-{session_id}-e{event_counter}",
                        type=EventType.CLICK,
                        timestamp=timestamp + 1.0,
                        role=Role.USER,
                        clicked_items=clicks,
                    ))

            base_timestamp = timestamp + 60.0

        if not events:
            return None

        # Extract session-level metadata
        satisfaction = data.get("satisfaction") or data.get("sat")
        topic_id = data.get("topic_id") or data.get("topic")

        return InteractionSession(
            session_id=f"tiangong-{session_id}",
            dataset_id="tiangong-st",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=data.get("user_id") or data.get("uid"),
            topic_id=str(topic_id) if topic_id else None,
            domain="web",
            meta={"satisfaction": satisfaction} if satisfaction else {},
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains TianGong-ST files."""
        return (
            any(data_path.glob("*.json")) or
            any(data_path.glob("*.jsonl"))
        )

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
