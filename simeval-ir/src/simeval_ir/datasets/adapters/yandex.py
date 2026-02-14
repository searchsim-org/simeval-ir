"""Yandex Relevance Prediction Challenge adapter.

Converts Yandex click log data to the SimEval-IR canonical format.
The Yandex dataset contains search sessions with click data for relevance prediction.
"""

import gzip
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


@adapter("yandex")
class YandexAdapter(DatasetAdapter):
    """Adapter for Yandex Relevance Prediction Challenge data.

    The Yandex dataset provides:
    - Search sessions with queries and clicks
    - Document features (not converted)
    - Query features (not converted)

    Format: Session data in TSV with SessionID, TimePassed, TypeOfAction, QueryID/URLID
    """

    name = "yandex"
    dataset_type: set[Literal["T", "C", "A"]] = {"T"}
    domain = "web"
    description = "Yandex Relevance Prediction Challenge"
    url = "https://www.kaggle.com/c/yandex-personalized-web-search-challenge"
    languages = ["ru"]

    def __init__(self, session_timeout_minutes: int = 30):
        """Initialize Yandex adapter.

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
        """Load sessions from Yandex data.

        Args:
            data_path: Path to Yandex train/test file(s).
            split: "train", "test", or "all"

        Yields:
            InteractionSession objects.
        """
        self._manifest = LossAccountingManifest(
            source_format="yandex-relevance",
            fields_mapped=[
                "session_id", "time_passed", "action_type", "query_id",
                "region_id", "url_id", "domain_id"
            ],
            fields_lost=[
                "query_terms",  # Only term IDs, not text
                "url_content",  # Not available
                "document_features",  # Complex feature vectors not converted
            ],
            transformation_notes=[
                "Query IDs used instead of query text (anonymized)",
                "URL IDs used as document identifiers",
                "Time passed converted to cumulative timestamps",
                "Russian web search domain",
            ],
        )

        if data_path.is_file():
            data_files = [data_path]
        else:
            data_files = []
            if split in ("all", "train"):
                data_files.extend(data_path.glob("*train*"))
            if split in ("all", "test"):
                data_files.extend(data_path.glob("*test*"))
            if not data_files:
                data_files = list(data_path.glob("*.txt")) + list(data_path.glob("*.gz"))

        for data_file in data_files:
            yield from self._load_file(data_file)

    def _load_file(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load sessions from a single file."""
        # Handle gzipped files
        if file_path.suffix == ".gz":
            opener = gzip.open
            mode = "rt"
        else:
            opener = open
            mode = "r"

        current_session_id = None
        current_events = []
        current_timestamp = 0.0
        event_counter = 0
        current_user_id = None
        current_serp = []

        with opener(file_path, mode, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue

                session_id = parts[0]
                time_passed = int(parts[1]) if parts[1].isdigit() else 0
                action_type = parts[2]

                # New session
                if session_id != current_session_id:
                    if current_session_id and current_events:
                        session = self._build_session(
                            current_session_id,
                            current_events,
                            current_user_id,
                        )
                        if session:
                            self._manifest.sessions_processed += 1
                            yield session

                    current_session_id = session_id
                    current_events = []
                    current_timestamp = 0.0
                    event_counter = 0
                    current_serp = []
                    current_user_id = parts[3] if len(parts) > 3 and action_type == "M" else None

                current_timestamp += time_passed / 1000.0  # Convert ms to seconds

                if action_type == "Q":
                    # Query: Q SerPID QueryID RegionID Terms...
                    if len(parts) >= 5:
                        query_id = parts[4]
                        region_id = parts[5] if len(parts) > 5 else None

                        event_counter += 1
                        current_events.append(Event(
                            event_id=f"yandex-{session_id}-e{event_counter}",
                            type=EventType.QUERY_ISSUED,
                            timestamp=current_timestamp,
                            role=Role.USER,
                            query=f"[QueryID:{query_id}]",
                            meta={"query_id": query_id, "region_id": region_id},
                        ))
                        current_serp = []

                elif action_type == "S":
                    # SERP: S SerPID URLID
                    if len(parts) >= 4:
                        url_id = parts[3]
                        current_serp.append(url_id)

                elif action_type == "C":
                    # Click: C SerPID URLID
                    if len(parts) >= 4:
                        url_id = parts[3]

                        # Add SERP if we have pending results
                        if current_serp:
                            event_counter += 1
                            ranked_items = [
                                RankedItem(doc_id=uid, rank=r+1)
                                for r, uid in enumerate(current_serp)
                            ]
                            current_events.append(Event(
                                event_id=f"yandex-{session_id}-e{event_counter}",
                                type=EventType.SERP_VIEW,
                                timestamp=current_timestamp - 0.1,
                                role=Role.SYSTEM,
                                ranked_items=ranked_items,
                            ))
                            current_serp = []

                        # Add click
                        rank = 0
                        for r, uid in enumerate(current_serp):
                            if uid == url_id:
                                rank = r + 1
                                break

                        event_counter += 1
                        current_events.append(Event(
                            event_id=f"yandex-{session_id}-e{event_counter}",
                            type=EventType.CLICK,
                            timestamp=current_timestamp,
                            role=Role.USER,
                            clicked_items=[Click(doc_id=url_id, rank=rank)],
                        ))

        # Emit final session
        if current_session_id and current_events:
            session = self._build_session(
                current_session_id,
                current_events,
                current_user_id,
            )
            if session:
                self._manifest.sessions_processed += 1
                yield session

    def _build_session(
        self,
        session_id: str,
        events: list[Event],
        user_id: str | None,
    ) -> InteractionSession | None:
        """Build a session from events."""
        if not events:
            return None

        return InteractionSession(
            session_id=f"yandex-{session_id}",
            dataset_id="yandex",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=user_id,
            domain="web",
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains Yandex files."""
        if data_path.is_file():
            return data_path.suffix in {".txt", ".gz", ".tsv"}
        return (
            any(data_path.glob("*.txt")) or
            any(data_path.glob("*.gz")) or
            any(data_path.glob("*.tsv"))
        )

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
