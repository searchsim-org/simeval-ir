"""Synthetic session generator for testing and demonstration.

Generates synthetic sessions that follow realistic patterns for testing
metrics without requiring real dataset access.
"""

import random
import string
from dataclasses import dataclass, field
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


@dataclass
class SessionGeneratorConfig:
    """Configuration for synthetic session generation.

    Attributes:
        n_sessions: Number of sessions to generate.
        session_type: Type of sessions to generate.
        min_queries: Minimum queries per search session.
        max_queries: Maximum queries per search session.
        min_turns: Minimum turns per conversational session.
        max_turns: Maximum turns per conversational session.
        min_results: Minimum results per SERP.
        max_results: Maximum results per SERP.
        click_probability: Probability of clicking on a result.
        max_clicks_per_query: Maximum clicks per query.
        add_dwell_time: Whether to add dwell time to clicks.
        seed: Random seed for reproducibility.
    """

    n_sessions: int = 100
    session_type: SessionType = SessionType.SEARCH
    min_queries: int = 1
    max_queries: int = 5
    min_turns: int = 2
    max_turns: int = 8
    min_results: int = 5
    max_results: int = 10
    click_probability: float = 0.3
    max_clicks_per_query: int = 3
    add_dwell_time: bool = True
    seed: int | None = 42


SAMPLE_QUERIES = [
    "python list comprehension",
    "machine learning tutorial",
    "best restaurants near me",
    "weather forecast tomorrow",
    "how to fix bug in code",
    "data science jobs",
    "neural network architecture",
    "information retrieval systems",
    "search engine optimization",
    "user behavior modeling",
]

SAMPLE_UTTERANCES = [
    "Can you help me find information about {}?",
    "I'm looking for {}.",
    "What can you tell me about {}?",
    "Show me results for {}.",
    "I need to know more about {}.",
]

SAMPLE_TOPICS = [
    "programming", "cooking", "travel", "health", "finance",
    "sports", "technology", "science", "education", "entertainment"
]


class SyntheticSessionGenerator:
    """Generator for synthetic session data.

    Creates realistic-looking sessions for testing and demonstration
    without requiring access to real datasets.
    """

    def __init__(self, config: SessionGeneratorConfig | None = None):
        """Initialize the generator.

        Args:
            config: Generation configuration. Uses defaults if None.
        """
        self.config = config or SessionGeneratorConfig()
        self._rng = random.Random(self.config.seed)

    def generate(self) -> list[InteractionSession]:
        """Generate synthetic sessions.

        Returns:
            List of generated InteractionSession objects.
        """
        return list(self.generate_iter())

    def generate_iter(self) -> Iterator[InteractionSession]:
        """Generate synthetic sessions as an iterator.

        Yields:
            InteractionSession objects.
        """
        for i in range(self.config.n_sessions):
            if self.config.session_type == SessionType.SEARCH:
                yield self._generate_search_session(i)
            elif self.config.session_type == SessionType.CONVERSATIONAL:
                yield self._generate_conversational_session(i)
            else:
                # Mixed: randomly choose
                if self._rng.random() < 0.5:
                    yield self._generate_search_session(i)
                else:
                    yield self._generate_conversational_session(i)

    def _generate_search_session(self, session_idx: int) -> InteractionSession:
        """Generate a synthetic search session."""
        n_queries = self._rng.randint(
            self.config.min_queries,
            self.config.max_queries,
        )

        events = []
        event_counter = 0
        timestamp = 0.0

        for query_idx in range(n_queries):
            # Query event
            event_counter += 1
            query = self._rng.choice(SAMPLE_QUERIES)
            if self._rng.random() < 0.3:
                query = f"{query} {self._random_suffix()}"

            events.append(Event(
                event_id=f"synth-{session_idx}-e{event_counter}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=query,
            ))
            timestamp += self._rng.uniform(0.5, 2.0)

            # SERP event
            event_counter += 1
            n_results = self._rng.randint(
                self.config.min_results,
                self.config.max_results,
            )
            ranked_items = [
                RankedItem(
                    doc_id=f"doc-{self._random_id()}",
                    rank=r + 1,
                    score=1.0 / (r + 1) + self._rng.uniform(-0.1, 0.1),
                    judged_relevance=self._rng.choice([0, 0, 0, 1, 1, 2]) if self._rng.random() < 0.3 else None,
                )
                for r in range(n_results)
            ]

            events.append(Event(
                event_id=f"synth-{session_idx}-e{event_counter}",
                type=EventType.SERP_VIEW,
                timestamp=timestamp,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            ))
            timestamp += self._rng.uniform(2.0, 10.0)

            # Click events (position-biased)
            clicks = []
            for item in ranked_items:
                # Higher rank = higher click probability
                position_bias = 1.0 / (item.rank + 0.5)
                click_prob = self.config.click_probability * position_bias
                if self._rng.random() < click_prob and len(clicks) < self.config.max_clicks_per_query:
                    dwell = None
                    if self.config.add_dwell_time:
                        # Longer dwell for relevant items
                        base_dwell = 30.0 if item.judged_relevance and item.judged_relevance > 0 else 10.0
                        dwell = self._rng.uniform(base_dwell * 0.5, base_dwell * 2.0)

                    clicks.append(Click(
                        doc_id=item.doc_id,
                        rank=item.rank,
                        dwell_time=dwell,
                    ))

            if clicks:
                event_counter += 1
                events.append(Event(
                    event_id=f"synth-{session_idx}-e{event_counter}",
                    type=EventType.CLICK,
                    timestamp=timestamp,
                    role=Role.USER,
                    clicked_items=clicks,
                ))
                timestamp += self._rng.uniform(10.0, 60.0)

        topic = self._rng.choice(SAMPLE_TOPICS)

        return InteractionSession(
            session_id=f"synth-search-{session_idx}",
            dataset_id="synthetic",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"user-{self._rng.randint(1, 100)}",
            topic_id=topic,
            domain="synthetic",
        )

    def _generate_conversational_session(self, session_idx: int) -> InteractionSession:
        """Generate a synthetic conversational session."""
        n_turns = self._rng.randint(
            self.config.min_turns,
            self.config.max_turns,
        )

        events = []
        event_counter = 0
        timestamp = 0.0
        topic = self._rng.choice(SAMPLE_TOPICS)

        for turn_idx in range(n_turns):
            # User utterance
            event_counter += 1
            template = self._rng.choice(SAMPLE_UTTERANCES)
            utterance = template.format(topic)

            events.append(Event(
                event_id=f"synth-{session_idx}-e{event_counter}",
                type=EventType.USER_UTTERANCE,
                timestamp=timestamp,
                turn_id=turn_idx,
                role=Role.USER,
                utterance=utterance,
            ))
            timestamp += self._rng.uniform(0.5, 3.0)

            # System response
            event_counter += 1
            response = f"Here is information about {topic}. {self._random_response()}"

            events.append(Event(
                event_id=f"synth-{session_idx}-e{event_counter}",
                type=EventType.SYSTEM_UTTERANCE,
                timestamp=timestamp,
                turn_id=turn_idx,
                role=Role.SYSTEM,
                utterance=response,
            ))
            timestamp += self._rng.uniform(2.0, 5.0)

        return InteractionSession(
            session_id=f"synth-conv-{session_idx}",
            dataset_id="synthetic",
            session_type=SessionType.CONVERSATIONAL,
            events=events,
            user_id=f"user-{self._rng.randint(1, 100)}",
            topic_id=topic,
            domain="synthetic",
        )

    def _random_id(self) -> str:
        """Generate a random document ID."""
        return "".join(self._rng.choices(string.ascii_lowercase + string.digits, k=8))

    def _random_suffix(self) -> str:
        """Generate a random query suffix."""
        suffixes = ["tutorial", "example", "guide", "2024", "best", "free"]
        return self._rng.choice(suffixes)

    def _random_response(self) -> str:
        """Generate a random response snippet."""
        snippets = [
            "This is a comprehensive overview.",
            "Based on recent research, here are the key points.",
            "Let me provide more details.",
            "Here are the most relevant findings.",
            "This should help answer your question.",
        ]
        return self._rng.choice(snippets)


@adapter("synthetic")
class SyntheticAdapter(DatasetAdapter):
    """Adapter for synthetic data generation.

    This adapter generates synthetic sessions on-the-fly for testing.
    It does not require a data path; the path parameter controls the
    number of sessions generated via a simple convention.
    """

    name = "synthetic"
    dataset_type: set[Literal["T", "C", "A"]] = {"T", "C"}
    domain = "synthetic"
    description = "Synthetic sessions for testing"
    languages = ["en"]

    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Generate synthetic sessions.

        Args:
            data_path: Ignored for synthetic data.
            split: Can be used to control generation:
                - "search": Generate only search sessions
                - "conversational": Generate only conversational sessions
                - "all" or other: Generate mixed sessions

        Yields:
            InteractionSession objects.
        """
        session_type = SessionType.MIXED
        if split == "search":
            session_type = SessionType.SEARCH
        elif split == "conversational":
            session_type = SessionType.CONVERSATIONAL

        config = SessionGeneratorConfig(
            n_sessions=100,
            session_type=session_type,
        )

        generator = SyntheticSessionGenerator(config)
        yield from generator.generate_iter()

    def validate_data_path(self, data_path: Path) -> bool:
        """Synthetic adapter always validates."""
        return True
