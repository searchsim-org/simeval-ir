"""TF-IDF based session embedder."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from simeval_ir.core.session import InteractionSession
from simeval_ir.embeddings.base import SessionEmbedder


class TfidfSessionEmbedder(SessionEmbedder):
    """TF-IDF based session embedder.

    Concatenates all text (queries and utterances) in a session
    and computes a TF-IDF vector.
    """

    def __init__(self, max_features: int = 1000):
        """Initialize TF-IDF embedder.

        Args:
            max_features: Maximum vocabulary size.
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self._fitted:
            return len(self.vectorizer.vocabulary_)
        return self.max_features

    def fit(self, sessions: list[InteractionSession]) -> "TfidfSessionEmbedder":
        """Fit the TF-IDF vectorizer.

        Args:
            sessions: Training sessions.

        Returns:
            Self.
        """
        texts = [self._session_to_text(s) for s in sessions]
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed_session(self, session: InteractionSession) -> np.ndarray:
        """Embed a session using TF-IDF.

        Args:
            session: The session to embed.

        Returns:
            TF-IDF vector.
        """
        if not self._fitted:
            raise RuntimeError("Embedder must be fitted before embedding.")

        text = self._session_to_text(session)
        vector = self.vectorizer.transform([text]).toarray()[0]
        return vector

    def _session_to_text(self, session: InteractionSession) -> str:
        """Convert session to text string."""
        parts = []
        for event in session.events:
            if event.query:
                parts.append(event.query)
            if event.utterance:
                parts.append(event.utterance)
        return " ".join(parts)
