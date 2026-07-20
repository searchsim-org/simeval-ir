"""TREC CAsT (Conversational Assistance Track) adapter.

Converts TREC CAsT data to the SimEval-IR canonical format.
CAsT provides multi-turn conversational search topics with passage judgments.
"""

import json
from pathlib import Path
from typing import Iterator, Literal

from simeval_ir.core.session import (
    InteractionSession,
    Event,
    RankedItem,
)
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets.base import DatasetAdapter, adapter
from simeval_ir.datasets.adapters.trec_session import LossAccountingManifest


@adapter("trec-cast")
class TRECCaSTAdapter(DatasetAdapter):
    """Adapter for TREC CAsT (Conversational Assistance Track) data.

    Supports CAsT 2019-2022 formats with:
    - Conversational topics (multi-turn queries)
    - Manual and automatic rewrites
    - Passage-level relevance judgments
    """

    name = "trec-cast"
    dataset_type: set[Literal["T", "C", "A"]] = {"C"}
    domain = "conversational"
    description = "TREC Conversational Assistance Track (2019-2022)"
    url = "https://www.treccast.ai/"
    languages = ["en"]

    def __init__(self, use_manual_rewrites: bool = True):
        """Initialize CAsT adapter.

        Args:
            use_manual_rewrites: Whether to use manual query rewrites when available.
        """
        self.use_manual_rewrites = use_manual_rewrites
        self._manifest: LossAccountingManifest | None = None
        self._qrels: dict[str, dict[str, int]] = {}

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from TREC CAsT data.

        Args:
            data_path: Path to CAsT directory containing:
                      - topics/ or topics.json
                      - qrels/ (optional)
            split: "train", "dev", "test", "eval", or "all"

        Yields:
            InteractionSession objects (one per conversation topic).
        """
        self._manifest = LossAccountingManifest(
            source_format="trec-cast",
            fields_mapped=[
                "topic_id", "turn_id", "raw_utterance", "manual_rewrite",
                "automatic_rewrite", "canonical_result_id", "passage_id"
            ],
            fields_lost=[
                "system_response_text",  # Only IDs provided, not full text
                "user_satisfaction",  # Not collected
                "click_data",  # CAsT doesn't have clicks
            ],
            transformation_notes=[
                f"Using {'manual' if self.use_manual_rewrites else 'automatic'} rewrites",
                "Turns converted to USER_UTTERANCE/SYSTEM_UTTERANCE pairs",
                "Qrels attached to system responses when available",
            ],
        )

        # Load qrels if available
        qrels_path = data_path / "qrels"
        if qrels_path.exists():
            self._load_qrels(qrels_path)
        else:
            # Try single qrels file
            for qrels_file in data_path.glob("*qrels*.txt"):
                self._load_qrels_file(qrels_file)

        # Load topics
        topics_path = data_path / "topics"
        if topics_path.exists():
            topic_files = list(topics_path.glob("*.json"))
        else:
            topic_files = list(data_path.glob("*topics*.json"))

        for topic_file in topic_files:
            # Filter by split if specified
            if split != "all" and split not in topic_file.name.lower():
                continue

            yield from self._load_topics_file(topic_file)

    def _load_qrels(self, qrels_path: Path) -> None:
        """Load qrels from directory."""
        for qrels_file in qrels_path.glob("*.txt"):
            self._load_qrels_file(qrels_file)

    def _load_qrels_file(self, qrels_file: Path) -> None:
        """Load qrels from single file."""
        with open(qrels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id, _, doc_id, rel = parts[:4]
                    if query_id not in self._qrels:
                        self._qrels[query_id] = {}
                    self._qrels[query_id][doc_id] = int(rel)

    def _load_topics_file(self, topic_file: Path) -> Iterator[InteractionSession]:
        """Load topics from a JSON file."""
        with open(topic_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different CAsT formats
        if isinstance(data, list):
            topics = data
        elif isinstance(data, dict):
            topics = data.get("topics") or data.get("conversations") or [data]
        else:
            topics = []

        for topic in topics:
            session = self._parse_topic(topic)
            if session:
                self._manifest.sessions_processed += 1
                yield session

    def _parse_topic(self, topic: dict) -> InteractionSession | None:
        """Parse a CAsT topic into a session."""
        topic_id = topic.get("number") or topic.get("topic_id") or topic.get("id")
        if topic_id is None:
            return None

        topic_id = str(topic_id)
        title = topic.get("title") or topic.get("description") or ""

        # Get turns
        turns = topic.get("turn") or topic.get("turns") or topic.get("utterances") or []

        events = []

        for turn in turns:
            turn_id = turn.get("number") or turn.get("turn_id") or turn.get("id")
            if turn_id is None:
                continue

            turn_id = int(turn_id) - 1  # Convert to 0-indexed

            # User utterance
            raw_utterance = turn.get("raw_utterance") or turn.get("utterance") or turn.get("query")

            # Choose rewrite based on preference
            if self.use_manual_rewrites:
                rewrite = turn.get("manual_rewritten_utterance") or turn.get("manual_rewrite")
            else:
                rewrite = turn.get("automatic_rewritten_utterance") or turn.get("automatic_rewrite")

            utterance = rewrite or raw_utterance
            if not utterance:
                continue

            # Query ID for qrels lookup
            query_id = f"{topic_id}_{turn_id + 1}"

            # User utterance event
            events.append(Event(
                event_id=f"cast-{topic_id}-t{turn_id}-user",
                type=EventType.USER_UTTERANCE,
                timestamp=float(turn_id * 2),
                turn_id=turn_id,
                role=Role.USER,
                utterance=utterance,
                meta={
                    "raw_utterance": raw_utterance,
                    "rewrite": rewrite,
                },
            ))

            # System response with ranked results
            canonical_result = turn.get("canonical_result_id") or turn.get("response_id")
            passage_provenance = turn.get("passage_provenance") or turn.get("provenance") or []

            ranked_items = []
            if passage_provenance:
                for rank, prov in enumerate(passage_provenance, start=1):
                    if isinstance(prov, str):
                        doc_id = prov
                    else:
                        doc_id = prov.get("id") or prov.get("passage_id") or prov.get("docno")

                    if doc_id:
                        rel = None
                        if query_id in self._qrels:
                            rel = self._qrels[query_id].get(doc_id)

                        ranked_items.append(RankedItem(
                            doc_id=doc_id,
                            rank=rank,
                            judged_relevance=rel,
                        ))
            elif canonical_result:
                rel = None
                if query_id in self._qrels:
                    rel = self._qrels[query_id].get(canonical_result)

                ranked_items.append(RankedItem(
                    doc_id=canonical_result,
                    rank=1,
                    judged_relevance=rel,
                ))

            # System utterance event
            response_text = turn.get("response") or turn.get("system_response")
            events.append(Event(
                event_id=f"cast-{topic_id}-t{turn_id}-system",
                type=EventType.SYSTEM_UTTERANCE,
                timestamp=float(turn_id * 2 + 1),
                turn_id=turn_id,
                role=Role.SYSTEM,
                utterance=response_text or f"[Response for turn {turn_id + 1}]",
                ranked_items=ranked_items if ranked_items else None,
            ))

        if not events:
            return None

        return InteractionSession(
            session_id=f"cast-{topic_id}",
            dataset_id="trec-cast",
            session_type=SessionType.CONVERSATIONAL,
            events=events,
            topic_id=topic_id,
            domain="conversational",
            meta={"title": title},
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains CAsT files."""
        return (
            (data_path / "topics").exists() or
            any(data_path.glob("*topics*.json")) or
            any(data_path.glob("*.json"))
        )

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
