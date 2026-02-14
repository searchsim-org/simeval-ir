"""Persona-Chat adapter.

Converts Persona-Chat dialogue data to the SimEval-IR canonical format.
Persona-Chat is a multi-turn dialogue dataset where speakers have defined personas.
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


@adapter("persona-chat")
class PersonaChatAdapter(DatasetAdapter):
    """Adapter for Persona-Chat dialogue data.

    Persona-Chat provides:
    - Multi-turn dialogues
    - Speaker personas (personality descriptions)
    - Train/valid/test splits

    Note: This is dialogue-only data without retrieval grounding.
    """

    name = "persona-chat"
    dataset_type: set[Literal["T", "C", "A"]] = {"C"}
    domain = "dialogue"
    description = "Persona-Chat Multi-Turn Dialogues"
    url = "https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat"
    languages = ["en"]

    def __init__(self, include_persona: bool = True):
        """Initialize Persona-Chat adapter.

        Args:
            include_persona: Whether to include persona in metadata.
        """
        self.include_persona = include_persona
        self._manifest: LossAccountingManifest | None = None

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from Persona-Chat data.

        Args:
            data_path: Path to Persona-Chat files.
            split: "train", "valid", "test", or "all"

        Yields:
            InteractionSession objects (one per dialogue).
        """
        self._manifest = LossAccountingManifest(
            source_format="persona-chat",
            fields_mapped=[
                "dialogue_turns", "speaker_persona", "partner_persona"
            ],
            fields_lost=[
                "retrieval_candidates",  # Not grounded in documents
                "relevance_judgments",  # No qrels
                "click_data",  # Dialogue-only
            ],
            transformation_notes=[
                "Dialogue-only data (not retrieval-grounded)",
                "Persona included in session metadata" if self.include_persona else "Persona excluded",
                "Each dialogue converted to one session",
            ],
        )

        # Find data files
        if data_path.is_file():
            data_files = [data_path]
        else:
            data_files = []
            patterns = ["*.json", "*.jsonl", "*.txt"]

            for pattern in patterns:
                if split == "all":
                    data_files.extend(data_path.glob(pattern))
                else:
                    data_files.extend(data_path.glob(f"*{split}*{pattern[1:]}"))

        for data_file in data_files:
            if data_file.suffix == ".txt":
                yield from self._load_parlai_format(data_file)
            else:
                yield from self._load_json_format(data_file)

    def _load_parlai_format(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load from ParlAI text format."""
        current_dialogue_id = None
        current_events = []
        current_persona = []
        event_counter = 0
        turn_id = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Lines start with turn number
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue

                try:
                    line_num = int(parts[0])
                except ValueError:
                    continue

                content = parts[1]

                # New dialogue starts at line 1
                if line_num == 1:
                    if current_events:
                        session = self._build_session(
                            current_dialogue_id,
                            current_events,
                            current_persona,
                        )
                        if session:
                            self._manifest.sessions_processed += 1
                            yield session

                    current_dialogue_id = f"pc-{self._manifest.sessions_processed}"
                    current_events = []
                    current_persona = []
                    event_counter = 0
                    turn_id = 0

                # Parse persona lines (start with "your persona:")
                if content.startswith("your persona:"):
                    persona_text = content.replace("your persona:", "").strip()
                    current_persona.append(persona_text)
                    continue

                # Parse dialogue turns
                if "\t" in content:
                    user_text, system_text = content.split("\t", 1)

                    # User utterance
                    event_counter += 1
                    current_events.append(Event(
                        event_id=f"{current_dialogue_id}-e{event_counter}",
                        type=EventType.USER_UTTERANCE,
                        timestamp=float(turn_id * 2),
                        turn_id=turn_id,
                        role=Role.USER,
                        utterance=user_text.strip(),
                    ))

                    # System response
                    event_counter += 1
                    current_events.append(Event(
                        event_id=f"{current_dialogue_id}-e{event_counter}",
                        type=EventType.SYSTEM_UTTERANCE,
                        timestamp=float(turn_id * 2 + 1),
                        turn_id=turn_id,
                        role=Role.SYSTEM,
                        utterance=system_text.strip(),
                    ))

                    turn_id += 1

        # Emit final dialogue
        if current_events:
            session = self._build_session(
                current_dialogue_id,
                current_events,
                current_persona,
            )
            if session:
                self._manifest.sessions_processed += 1
                yield session

    def _load_json_format(self, file_path: Path) -> Iterator[InteractionSession]:
        """Load from JSON/JSONL format."""
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".jsonl":
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        session = self._parse_dialogue(data, line_num)
                        if session:
                            self._manifest.sessions_processed += 1
                            yield session
                    except json.JSONDecodeError:
                        self._manifest.sessions_skipped += 1
            else:
                data = json.load(f)
                dialogues = data if isinstance(data, list) else data.get("dialogues", [data])
                for idx, dialogue in enumerate(dialogues):
                    session = self._parse_dialogue(dialogue, idx)
                    if session:
                        self._manifest.sessions_processed += 1
                        yield session

    def _parse_dialogue(self, data: dict, idx: int) -> InteractionSession | None:
        """Parse a dialogue from dictionary data."""
        dialogue_id = data.get("dialogue_id") or data.get("id") or f"pc-{idx}"

        # Get persona
        persona = []
        if self.include_persona:
            persona = data.get("persona") or data.get("your_persona") or []
            if isinstance(persona, str):
                persona = [persona]

        # Get turns
        turns = data.get("turns") or data.get("utterances") or data.get("dialogue") or []

        if not turns:
            return None

        events = []
        event_counter = 0

        for turn_id, turn in enumerate(turns):
            if isinstance(turn, str):
                # Simple string format - alternate user/system
                role = Role.USER if turn_id % 2 == 0 else Role.SYSTEM
                event_type = EventType.USER_UTTERANCE if role == Role.USER else EventType.SYSTEM_UTTERANCE

                event_counter += 1
                events.append(Event(
                    event_id=f"{dialogue_id}-e{event_counter}",
                    type=event_type,
                    timestamp=float(turn_id),
                    turn_id=turn_id // 2,
                    role=role,
                    utterance=turn,
                ))
            elif isinstance(turn, dict):
                # Dictionary format
                text = turn.get("text") or turn.get("utterance") or turn.get("content")
                speaker = turn.get("speaker") or turn.get("role") or turn.get("agent")

                if speaker in ("user", "human", "0"):
                    role = Role.USER
                    event_type = EventType.USER_UTTERANCE
                else:
                    role = Role.SYSTEM
                    event_type = EventType.SYSTEM_UTTERANCE

                event_counter += 1
                events.append(Event(
                    event_id=f"{dialogue_id}-e{event_counter}",
                    type=event_type,
                    timestamp=float(turn_id),
                    turn_id=turn_id // 2,
                    role=role,
                    utterance=text,
                ))

        return self._build_session(dialogue_id, events, persona)

    def _build_session(
        self,
        dialogue_id: str,
        events: list[Event],
        persona: list[str],
    ) -> InteractionSession | None:
        """Build a session from events."""
        if not events:
            return None

        meta = {}
        if self.include_persona and persona:
            meta["persona"] = persona

        return InteractionSession(
            session_id=f"persona-chat-{dialogue_id}",
            dataset_id="persona-chat",
            session_type=SessionType.CONVERSATIONAL,
            events=events,
            domain="dialogue",
            meta=meta,
        )

    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains Persona-Chat files."""
        if data_path.is_file():
            return data_path.suffix in {".txt", ".json", ".jsonl"}
        return (
            any(data_path.glob("*.txt")) or
            any(data_path.glob("*.json")) or
            any(data_path.glob("*.jsonl"))
        )

    def get_manifest(self) -> LossAccountingManifest | None:
        """Get the loss accounting manifest after loading."""
        return self._manifest
