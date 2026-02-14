"""Tests for core session data models."""

import pytest

from simeval_ir.core.session import Click, Event, InteractionSession, RankedItem
from simeval_ir.core.types import EventType, Role, SessionType


class TestInteractionSession:
    """Tests for InteractionSession."""

    def test_create_search_session(self, sample_search_session):
        """Test creating a search session."""
        session = sample_search_session
        assert session.session_id == "test_session_1"
        assert session.session_type == SessionType.SEARCH
        assert len(session.events) == 4

    def test_create_conversational_session(self, sample_conversational_session):
        """Test creating a conversational session."""
        session = sample_conversational_session
        assert session.session_id == "conv_session_1"
        assert session.session_type == SessionType.CONVERSATIONAL
        assert len(session.events) == 4

    def test_get_queries(self, sample_search_session):
        """Test extracting queries from a session."""
        queries = sample_search_session.get_queries()
        assert len(queries) == 2
        assert queries[0] == "python tutorial"
        assert queries[1] == "python tutorial beginner"

    def test_get_utterances(self, sample_conversational_session):
        """Test extracting utterances from a session."""
        all_utterances = sample_conversational_session.get_utterances()
        assert len(all_utterances) == 4

        user_utterances = sample_conversational_session.get_utterances(role=Role.USER)
        assert len(user_utterances) == 2

    def test_get_clicks(self, sample_search_session):
        """Test extracting clicks from a session."""
        clicks = sample_search_session.get_clicks()
        assert len(clicks) == 1
        assert clicks[0].doc_id == "d1"
        assert clicks[0].dwell_time == 30.0

    def test_get_turns(self, sample_conversational_session):
        """Test grouping events into turns."""
        turns = sample_conversational_session.get_turns()
        assert len(turns) == 2
        assert turns[0].turn_id == 0
        assert turns[0].user_event is not None
        assert turns[0].system_event is not None

    def test_num_queries(self, sample_search_session):
        """Test counting queries."""
        assert sample_search_session.num_queries() == 2

    def test_num_clicks(self, sample_search_session):
        """Test counting clicks."""
        assert sample_search_session.num_clicks() == 1

    def test_num_turns(self, sample_conversational_session):
        """Test counting turns."""
        assert sample_conversational_session.num_turns() == 2

    def test_duration(self, sample_search_session):
        """Test calculating session duration."""
        duration = sample_search_session.duration()
        assert duration == 35.0

    def test_to_dict(self, sample_search_session):
        """Test serialization to dict."""
        data = sample_search_session.to_dict()
        assert data["session_id"] == "test_session_1"
        assert data["session_type"] == "search"
        assert len(data["events"]) == 4

    def test_from_dict(self, sample_search_session):
        """Test deserialization from dict."""
        data = sample_search_session.to_dict()
        restored = InteractionSession.from_dict(data)
        assert restored.session_id == sample_search_session.session_id
        assert len(restored.events) == len(sample_search_session.events)


class TestEvent:
    """Tests for Event class."""

    def test_is_user_action(self):
        """Test identifying user actions."""
        query_event = Event(
            event_id="e1",
            type=EventType.QUERY_ISSUED,
            role=Role.USER,
        )
        assert query_event.is_user_action()

        system_event = Event(
            event_id="e2",
            type=EventType.SERP_VIEW,
            role=Role.SYSTEM,
        )
        assert not system_event.is_user_action()

    def test_has_text(self):
        """Test checking for text content."""
        query_event = Event(
            event_id="e1",
            type=EventType.QUERY_ISSUED,
            query="test query",
        )
        assert query_event.has_text()
        assert query_event.get_text() == "test query"

        empty_event = Event(
            event_id="e2",
            type=EventType.CLICK,
        )
        assert not empty_event.has_text()


class TestRankedItem:
    """Tests for RankedItem class."""

    def test_create_ranked_item(self):
        """Test creating a ranked item."""
        item = RankedItem(
            doc_id="doc1",
            rank=1,
            score=0.95,
            judged_relevance=3,
        )
        assert item.doc_id == "doc1"
        assert item.rank == 1
        assert item.score == 0.95
        assert item.judged_relevance == 3


class TestClick:
    """Tests for Click class."""

    def test_create_click(self):
        """Test creating a click."""
        click = Click(
            doc_id="doc1",
            rank=2,
            dwell_time=15.5,
            meta={"from_ad": True},
        )
        assert click.doc_id == "doc1"
        assert click.rank == 2
        assert click.dwell_time == 15.5
        assert click.meta["from_ad"] is True
