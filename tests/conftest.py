"""Test fixtures and configuration."""

import pytest

from simeval_ir.core.session import Click, Event, InteractionSession, RankedItem
from simeval_ir.core.types import EventType, Role, SessionType


@pytest.fixture
def sample_search_session() -> InteractionSession:
    """Create a sample search session for testing."""
    return InteractionSession(
        session_id="test_session_1",
        dataset_id="test",
        session_type=SessionType.SEARCH,
        events=[
            Event(
                event_id="e1",
                type=EventType.QUERY_ISSUED,
                timestamp=0.0,
                role=Role.USER,
                query="python tutorial",
            ),
            Event(
                event_id="e2",
                type=EventType.SERP_VIEW,
                timestamp=1.0,
                role=Role.SYSTEM,
                ranked_items=[
                    RankedItem(doc_id="d1", rank=1, score=0.9, judged_relevance=3),
                    RankedItem(doc_id="d2", rank=2, score=0.8, judged_relevance=2),
                    RankedItem(doc_id="d3", rank=3, score=0.7, judged_relevance=0),
                ],
            ),
            Event(
                event_id="e3",
                type=EventType.CLICK,
                timestamp=2.0,
                role=Role.USER,
                clicked_items=[
                    Click(doc_id="d1", rank=1, dwell_time=30.0),
                ],
            ),
            Event(
                event_id="e4",
                type=EventType.QUERY_ISSUED,
                timestamp=35.0,
                role=Role.USER,
                query="python tutorial beginner",
            ),
        ],
        user_id="user1",
        topic_id="topic1",
        domain="web",
    )


@pytest.fixture
def sample_conversational_session() -> InteractionSession:
    """Create a sample conversational session for testing."""
    return InteractionSession(
        session_id="conv_session_1",
        dataset_id="test",
        session_type=SessionType.CONVERSATIONAL,
        events=[
            Event(
                event_id="e1",
                type=EventType.USER_UTTERANCE,
                timestamp=0.0,
                turn_id=0,
                role=Role.USER,
                utterance="Hi, I'm looking for a good restaurant.",
            ),
            Event(
                event_id="e2",
                type=EventType.SYSTEM_UTTERANCE,
                timestamp=1.0,
                turn_id=0,
                role=Role.SYSTEM,
                utterance="What type of cuisine are you interested in?",
            ),
            Event(
                event_id="e3",
                type=EventType.USER_UTTERANCE,
                timestamp=5.0,
                turn_id=1,
                role=Role.USER,
                utterance="Italian food would be great.",
            ),
            Event(
                event_id="e4",
                type=EventType.SYSTEM_UTTERANCE,
                timestamp=6.0,
                turn_id=1,
                role=Role.SYSTEM,
                utterance="I recommend Bella Italia, it has great reviews.",
            ),
        ],
        user_id="user2",
        topic_id="restaurant_search",
        domain="local",
    )


@pytest.fixture
def sample_sessions(sample_search_session, sample_conversational_session):
    """Return both sample sessions."""
    return [sample_search_session, sample_conversational_session]


@pytest.fixture
def real_sessions() -> list[InteractionSession]:
    """Create a list of real sessions for testing."""
    sessions = []
    for i in range(5):
        sessions.append(InteractionSession(
            session_id=f"real_{i}",
            dataset_id="test",
            session_type=SessionType.SEARCH,
            events=[
                Event(
                    event_id=f"r{i}_e1",
                    type=EventType.QUERY_ISSUED,
                    timestamp=0.0,
                    role=Role.USER,
                    query=f"query {i}",
                ),
                Event(
                    event_id=f"r{i}_e2",
                    type=EventType.CLICK,
                    timestamp=1.0 + i * 0.5,
                    role=Role.USER,
                    clicked_items=[Click(doc_id=f"d{i}", rank=1)],
                ),
            ],
        ))
    return sessions


@pytest.fixture
def sim_sessions() -> list[InteractionSession]:
    """Create a list of simulated sessions for testing."""
    sessions = []
    for i in range(5):
        sessions.append(InteractionSession(
            session_id=f"sim_{i}",
            dataset_id="test",
            session_type=SessionType.SEARCH,
            events=[
                Event(
                    event_id=f"s{i}_e1",
                    type=EventType.QUERY_ISSUED,
                    timestamp=0.0,
                    role=Role.USER,
                    query=f"simulated query {i}",
                ),
                Event(
                    event_id=f"s{i}_e2",
                    type=EventType.SERP_VIEW,
                    timestamp=0.5,
                    role=Role.SYSTEM,
                ),
                Event(
                    event_id=f"s{i}_e3",
                    type=EventType.CLICK,
                    timestamp=2.0 + i * 0.3,
                    role=Role.USER,
                    clicked_items=[Click(doc_id=f"d{i}", rank=2)],
                ),
            ],
        ))
    return sessions
