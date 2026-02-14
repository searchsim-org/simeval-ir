"""Action sequence embedder."""

from collections import Counter

import numpy as np

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import EventType
from simeval_ir.embeddings.base import SessionEmbedder


class ActionSequenceEmbedder(SessionEmbedder):
    """Action sequence based session embedder.

    Creates embeddings based on action type counts and transitions.
    """

    # All possible event types
    EVENT_TYPES = [t.value for t in EventType]

    def __init__(self, include_transitions: bool = True):
        """Initialize action sequence embedder.

        Args:
            include_transitions: Whether to include transition counts.
        """
        self.include_transitions = include_transitions

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        n_types = len(self.EVENT_TYPES)
        dim = n_types  # Action counts
        if self.include_transitions:
            dim += n_types * n_types  # Transition matrix
        return dim

    def embed_session(self, session: InteractionSession) -> np.ndarray:
        """Embed a session based on action patterns.

        Args:
            session: The session to embed.

        Returns:
            Feature vector.
        """
        # Count action types
        type_counts = Counter(e.type.value for e in session.events)
        count_features = np.array([
            type_counts.get(t, 0) for t in self.EVENT_TYPES
        ], dtype=float)

        # Normalize
        if count_features.sum() > 0:
            count_features = count_features / count_features.sum()

        if not self.include_transitions:
            return count_features

        # Count transitions
        n_types = len(self.EVENT_TYPES)
        type_to_idx = {t: i for i, t in enumerate(self.EVENT_TYPES)}
        transition_matrix = np.zeros((n_types, n_types))

        for i in range(len(session.events) - 1):
            from_type = session.events[i].type.value
            to_type = session.events[i + 1].type.value
            from_idx = type_to_idx.get(from_type)
            to_idx = type_to_idx.get(to_type)
            if from_idx is not None and to_idx is not None:
                transition_matrix[from_idx, to_idx] += 1

        # Normalize transitions
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums

        # Flatten and concatenate
        transition_features = transition_matrix.flatten()
        return np.concatenate([count_features, transition_features])
