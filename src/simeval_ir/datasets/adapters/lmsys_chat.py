"""LMSYS-Chat-1M adapter.

Converts LMSYS-Chat-1M data to the SimEval-IR canonical format.
LMSYS-Chat-1M contains 1 million conversations with various LLMs from the Chatbot Arena.
"""

import json
from pathlib import Path
from typing import Iterator, Literal

from simeval_ir.core.session import (
    InteractionSession,
    Event,
)
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets.base import DatasetAdapter, adapter
from simeval_ir.datasets.adapters.trec_session import LossAccountingManifest


@adapter("lmsys-chat")
class LMSYSChatAdapter(DatasetAdapter):
    """Adapter for LMSYS-Chat-1M data.

    LMSYS-Chat-1M provides:
    - Multi-turn conversations with LLMs
    - Various model identifiers
    - Language labels
    - Conversation metadata

    Note: This is dialogue-only data without retrieval grounding.
    """

    name = "lmsys-chat"
    dataset_type: set[Literal["T", "C", "A"]] = {"C"}
    domain = "dialogue"
    description = "LMSYS-Chat-1M LLM Conversations"
    url = "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"
    languages = ["multi"]

    def __init__(self, language_filter: str | None = None):
        """Initialize LMSYS-Chat adapter.

        Args:
            language_filter: Optional language code to filter (e.g., "en", "zh").
        """
        self.language_filter = language_filter
        self._manifest: LossAccountingManifest | None = None

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from LMSYS-Chat-1M data.

        Args:
            data_path: Path to LMSYS data (Parquet or JSONL files).
            split: "train", "test", or "all"

        Yields:
            InteractionSession objects (one per conversation).
        """
        self._manifest = LossAccountingManifest(
            source_format="lmsys-chat-1m",
            fields_mapped=[
                "conversation_id", "model", "conversation", "turn",
                "language", "openai_moderation", "redacted"
            ],
            fields_lost=[
                "retrieval_grounding",  # No document retrieval
                "relevance_judgments",  # No qrels
                "user_preferences",  # Not in public release
            ],
            transformation_notes=[
                "Dialogue-only data (not retrieval-grounded)",
                f"Language filter: {self.language_filter or 'none (all languages)'}",
                "Model identifier stored in session metadata",
                "Redacted conversations preserved with [redacted] markers",
            ],
        )

        # Find data files
        if data_path.is_file():
            data_files = [data_path]
        else:
            data_files = list(data_path.glob("*.jsonl"))
            data_files.extend(data_path.glob("*.json"))
            data_files.extend(data_path.glob("*.parquet"))

        for data_file in data_files:
            if data_file.suffix == ".parquet":
                yield from self._load_parquet(data_file)
            else:
                yield from self._load_jsonl(data_file)

    def _load_parquet(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load from Parquet format (requires pyarrow)."""
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(file_path)

            for i in range(table.num_rows):
                row = {col: table[col][i].as_py() for col in table.column_names}
                session = self._parse_conversation(row, i)
                if session:
                    self._manifest.sessions_processed += 1
                    yield session

        except ImportError:
            self._manifest.transformation_notes.append(
                f"Skipped {file_path.name}: pyarrow not installed"
            )
        except Exception as e:
            self._manifest.transformation_notes.append(
                f"Error reading {file_path.name}: {str(e)}"
            )

    def _load_jsonl(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load from JSONL format."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    session = self._parse_conversation(data, line_num)
                    if session:
                        self._manifest.sessions_processed += 1
                        yield session
                except json.JSONDecodeError:
                    self._manifest.sessions_skipped += 1

    def _parse_conversation(self, data: dict, idx: int) -> InteractionSession | None:
        """Parse a conversation from dictionary data."""
        # Apply language filter
        language = data.get("language") or data.get("lang")
        if self.language_filter and language != self.language_filter:
            return None

        conversation_id = data.get("conversation_id") or data.get("id") or f"lmsys-{idx}"
        model = data.get("model") or data.get("model_name")

        # Get conversation turns
        conversation = data.get("conversation") or data.get("turns") or data.get("messages") or []

        if not conversation:
            return None

        events = []
        event_counter = 0

        for turn_id, turn in enumerate(conversation):
            if isinstance(turn, str):
                # Simple format - alternate human/assistant
                role = Role.USER if turn_id % 2 == 0 else Role.SYSTEM
                text = turn
            elif isinstance(turn, dict):
                # Dictionary format with role
                role_str = turn.get("role") or turn.get("from") or turn.get("speaker")
                text = turn.get("content") or turn.get("value") or turn.get("text")

                if role_str in ("user", "human"):
                    role = Role.USER
                elif role_str in ("assistant", "gpt", "bot", "model"):
                    role = Role.SYSTEM
                else:
                    role = Role.USER if turn_id % 2 == 0 else Role.SYSTEM
            else:
                continue

            if not text:
                continue

            event_type = EventType.USER_UTTERANCE if role == Role.USER else EventType.SYSTEM_UTTERANCE

            event_counter += 1
            events.append(Event(
                event_id=f"{conversation_id}-e{event_counter}",
                type=event_type,
                timestamp=float(turn_id),
                turn_id=turn_id // 2,
                role=role,
                utterance=text,
            ))

        if not events:
            return None

        # Build metadata
        meta = {}
        if model:
            meta["model"] = model
        if language:
            meta["language"] = language
        if data.get("redacted"):
            meta["redacted"] = True
        if data.get("openai_moderation"):
            meta["moderation"] = data["openai_moderation"]

        return InteractionSession(
            session_id=f"lmsys-{conversation_id}",
            dataset_id="lmsys-chat",
            session_type=SessionType.CONVERSATIONAL,
            events=events,
            domain="dialogue",
            meta=meta,
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains LMSYS files."""
        if data_path.is_file():
            return data_path.suffix in {".jsonl", ".json", ".parquet"}
        return (
            any(data_path.glob("*.jsonl")) or
            any(data_path.glob("*.json")) or
            any(data_path.glob("*.parquet"))
        )

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
