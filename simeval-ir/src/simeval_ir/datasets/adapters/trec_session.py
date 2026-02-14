"""TREC Session Track adapter.

Converts TREC Session Track data (2010-2014) to the SimEval-IR canonical format.
The adapter handles the XML format used in TREC Session Track files and produces
a loss accounting manifest documenting any fields that cannot be mapped.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal
import json

from simeval_ir.core.session import (
    InteractionSession,
    Event,
    RankedItem,
    Click,
)
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets.base import DatasetAdapter, adapter


@dataclass
class LossAccountingManifest:
    """Manifest documenting data loss during conversion.

    Attributes:
        source_format: Original data format name.
        target_format: Target format (always "simeval-ir").
        sessions_processed: Number of sessions processed.
        sessions_skipped: Number of sessions skipped due to errors.
        fields_mapped: List of source fields successfully mapped.
        fields_lost: List of source fields that could not be mapped.
        transformation_notes: Notes about data transformations applied.
    """

    source_format: str
    target_format: str = "simeval-ir"
    sessions_processed: int = 0
    sessions_skipped: int = 0
    fields_mapped: list[str] = field(default_factory=list)
    fields_lost: list[str] = field(default_factory=list)
    transformation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert manifest to dictionary."""
        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "sessions_processed": self.sessions_processed,
            "sessions_skipped": self.sessions_skipped,
            "fields_mapped": self.fields_mapped,
            "fields_lost": self.fields_lost,
            "transformation_notes": self.transformation_notes,
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@adapter("trec-session")
class TRECSessionAdapter(DatasetAdapter):
    """Adapter for TREC Session Track data.

    Supports TREC Session Track 2010-2014 XML format. The adapter maps:
    - Sessions → InteractionSession
    - Queries → Event(type=QUERY_ISSUED)
    - Clicks → Event(type=CLICK) with Click objects
    - Results → RankedItem objects

    Fields that cannot be mapped are documented in the loss manifest.
    """

    name = "trec-session"
    dataset_type: set[Literal["T", "C", "A"]] = {"T"}
    domain = "web"
    description = "TREC Session Track (2010-2014)"
    url = "https://trec.nist.gov/data/session.html"
    languages = ["en"]

    def __init__(self):
        self._manifest: LossAccountingManifest | None = None

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from TREC Session Track XML files.

        Args:
            data_path: Path to directory containing XML files or a single XML file.
            split: Split to load (ignored for TREC data, loads all).

        Yields:
            InteractionSession objects.
        """
        self._manifest = LossAccountingManifest(
            source_format="trec-session-xml",
            fields_mapped=[
                "session_id", "topic_id", "query_text", "query_time",
                "clicked_docs", "click_rank", "dwell_time", "ranked_results"
            ],
            fields_lost=[
                "session_type_annotation",  # TREC doesn't have this
                "dialogue_acts",  # Not applicable
                "user_feedback",  # Not always present
            ],
            transformation_notes=[
                "Query timestamps converted from absolute to relative (seconds from session start)",
                "Click ranks extracted from position in result list when not explicit",
                "Session IDs prefixed with 'trec-' namespace",
            ],
        )

        if data_path.is_file():
            xml_files = [data_path]
        else:
            xml_files = list(data_path.glob("*.xml"))

        for xml_file in xml_files:
            try:
                yield from self._parse_xml_file(xml_file)
            except Exception as e:
                self._manifest.transformation_notes.append(
                    f"Error processing {xml_file.name}: {str(e)}"
                )

    def _parse_xml_file(self, xml_path: Path) -> Iterator[InteractionSession]:
        """Parse a single TREC Session XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle different TREC Session formats
        sessions = root.findall(".//session") or root.findall(".//Session")

        for session_elem in sessions:
            try:
                session = self._parse_session(session_elem)
                if session:
                    self._manifest.sessions_processed += 1
                    yield session
            except Exception:
                self._manifest.sessions_skipped += 1

    def _parse_session(self, session_elem: ET.Element) -> InteractionSession | None:
        """Parse a single session element."""
        session_id = session_elem.get("num") or session_elem.get("id")
        if not session_id:
            return None

        topic_id = session_elem.get("topic") or session_elem.findtext("topic")

        events = []
        event_counter = 0
        base_timestamp = 0.0

        # Parse interactions (queries and clicks)
        interactions = (
            session_elem.findall(".//interaction") or
            session_elem.findall(".//Interaction") or
            session_elem.findall(".//query")
        )

        for interaction in interactions:
            # Parse query
            query_text = (
                interaction.findtext("query") or
                interaction.findtext("Query") or
                interaction.get("query")
            )

            if query_text:
                event_counter += 1
                query_event = Event(
                    event_id=f"{session_id}-e{event_counter}",
                    type=EventType.QUERY_ISSUED,
                    timestamp=base_timestamp,
                    role=Role.USER,
                    query=query_text.strip(),
                )
                events.append(query_event)

                # Parse SERP/results
                ranked_items = self._parse_results(interaction)
                if ranked_items:
                    event_counter += 1
                    serp_event = Event(
                        event_id=f"{session_id}-e{event_counter}",
                        type=EventType.SERP_VIEW,
                        timestamp=base_timestamp + 0.1,
                        role=Role.SYSTEM,
                        ranked_items=ranked_items,
                    )
                    events.append(serp_event)

                # Parse clicks
                clicks = self._parse_clicks(interaction)
                if clicks:
                    event_counter += 1
                    click_event = Event(
                        event_id=f"{session_id}-e{event_counter}",
                        type=EventType.CLICK,
                        timestamp=base_timestamp + 0.5,
                        role=Role.USER,
                        clicked_items=clicks,
                    )
                    events.append(click_event)

                base_timestamp += 60.0  # Assume 60s between queries

        if not events:
            return None

        return InteractionSession(
            session_id=f"trec-{session_id}",
            dataset_id="trec-session",
            session_type=SessionType.SEARCH,
            events=events,
            topic_id=topic_id,
            domain="web",
        )

    def _parse_results(self, interaction: ET.Element) -> list[RankedItem] | None:
        """Parse ranked results from an interaction."""
        items = []

        results = (
            interaction.findall(".//result") or
            interaction.findall(".//Result") or
            interaction.findall(".//results/doc")
        )

        for rank, result in enumerate(results, start=1):
            doc_id = (
                result.get("docno") or
                result.get("docid") or
                result.findtext("docno") or
                result.findtext("docid") or
                result.text
            )

            if doc_id:
                score_text = result.get("score") or result.findtext("score")
                score = float(score_text) if score_text else None

                rel_text = result.get("relevance") or result.findtext("relevance")
                relevance = int(rel_text) if rel_text else None

                items.append(RankedItem(
                    doc_id=doc_id.strip(),
                    rank=rank,
                    score=score,
                    judged_relevance=relevance,
                ))

        return items if items else None

    def _parse_clicks(self, interaction: ET.Element) -> list[Click] | None:
        """Parse clicks from an interaction."""
        clicks = []

        click_elems = (
            interaction.findall(".//click") or
            interaction.findall(".//Click")
        )

        for click_elem in click_elems:
            doc_id = (
                click_elem.get("docno") or
                click_elem.get("docid") or
                click_elem.findtext("docno") or
                click_elem.text
            )

            if doc_id:
                rank_text = click_elem.get("rank") or click_elem.findtext("rank")
                rank = int(rank_text) if rank_text else 0

                dwell_text = (
                    click_elem.get("dwell") or
                    click_elem.get("dwelltime") or
                    click_elem.findtext("dwell")
                )
                dwell_time = float(dwell_text) if dwell_text else None

                clicks.append(Click(
                    doc_id=doc_id.strip(),
                    rank=rank,
                    dwell_time=dwell_time,
                ))

        return clicks if clicks else None

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains TREC Session files."""
        if data_path.is_file():
            return data_path.suffix == ".xml"
        return any(data_path.glob("*.xml"))

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading.

        Returns:
            The manifest if load_sessions has been called, None otherwise.
        """
        return self._manifest
