"""Base embedder interfaces."""

from abc import ABC, abstractmethod

import numpy as np

from simeval_ir.core.session import Event, InteractionSession


class SessionEmbedder(ABC):
    """Abstract base class for session embedders.

    Session embedders convert complete sessions into fixed-size
    vector representations.
    """

    embedding_dim: int

    @abstractmethod
    def embed_session(self, session: InteractionSession) -> np.ndarray:
        """Embed a single session.

        Args:
            session: The session to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        ...

    def embed_sessions(self, sessions: list[InteractionSession]) -> np.ndarray:
        """Embed multiple sessions.

        Args:
            sessions: List of sessions to embed.

        Returns:
            2D numpy array of shape (n_sessions, embedding_dim).
        """
        return np.array([self.embed_session(s) for s in sessions])

    def fit(self, sessions: list[InteractionSession]) -> "SessionEmbedder":
        """Fit the embedder to a collection of sessions.

        Default implementation does nothing (for pre-trained embedders).

        Args:
            sessions: Training sessions.

        Returns:
            Self.
        """
        return self


class TurnEmbedder(ABC):
    """Abstract base class for turn/event embedders.

    Turn embedders convert individual events or turns into
    vector representations.
    """

    embedding_dim: int

    @abstractmethod
    def embed_event(self, event: Event) -> np.ndarray:
        """Embed a single event.

        Args:
            event: The event to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        ...

    def embed_events(self, events: list[Event]) -> np.ndarray:
        """Embed multiple events.

        Args:
            events: List of events to embed.

        Returns:
            2D numpy array of shape (n_events, embedding_dim).
        """
        return np.array([self.embed_event(e) for e in events])

    def fit(self, events: list[Event]) -> "TurnEmbedder":
        """Fit the embedder to a collection of events.

        Args:
            events: Training events.

        Returns:
            Self.
        """
        return self
