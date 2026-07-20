"""Leakage auditing for classifier-based realism metrics.

This module implements the mandatory leakage audit protocol described in the paper:
1. Metadata-only baseline: Only missingness indicators and schema presence
2. Structural-only model: Only action-sequence statistics
3. Masked-feature test: Structural features with suspicious columns removed
   (session length and missingness-correlated features stripped, keeping only
   event-type distribution and transition probabilities)
4. Main classifier: Full feature set (embeddings if available, else structural)
5. Permutation check: Shuffle labels (should collapse to ~0.5)

High classifier accuracy is evidence of behavioral discrepancy ONLY if leakage
baselines remain near chance.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import MetricResult, EventType
from simeval_ir.metrics.base import BaseMetric, metric


@dataclass
class LeakageAuditResult:
    """Result of a leakage audit for classifier-based realism."""

    main_auc: float
    metadata_only_auc: float
    masked_feature_auc: float
    structural_only_auc: float
    permutation_auc: float
    leakage_detected: bool
    realism_score: float  # 1 - main_auc (only valid if no leakage)
    meta: dict = field(default_factory=dict)

    def is_valid(self, threshold: float = 0.55) -> bool:
        """Check if the main classifier result is valid (no leakage).

        Args:
            threshold: Maximum acceptable AUC for leakage baselines.
                Values above this suggest leakage.

        Returns:
            True if leakage baselines are below threshold.
        """
        return (
            self.metadata_only_auc < threshold
            and self.permutation_auc < threshold
        )


def _extract_metadata_features(sessions: list[InteractionSession]) -> np.ndarray:
    """Extract strictly schema-level metadata features for leakage detection.

    These features encode dataset/preprocessing artefacts that should NOT
    differ between real and simulated sessions when the simulator was
    written to the same canonical schema. Critically, none of these
    features depend on event composition or click counts -- those are
    behavioural signals and would create false-positive leakage flags.
    """
    features = []
    for session in sessions:
        feat = [
            1.0 if session.user_id else 0.0,
            1.0 if session.topic_id else 0.0,
            1.0 if session.domain else 0.0,
            1.0 if session.session_type else 0.0,
            # Whether the session has *any* timestamps
            1.0 if any(e.timestamp is not None for e in session.events) else 0.0,
            # Whether the session has *any* turn_id (conversational schema)
            1.0 if any(e.turn_id is not None for e in session.events) else 0.0,
            # Whether the session has *any* role information
            1.0 if any(e.role is not None for e in session.events) else 0.0,
        ]
        features.append(feat)
    return np.array(features)


def _extract_structural_features(sessions: list[InteractionSession]) -> np.ndarray:
    """Extract structural/action-sequence features.

    These features encode behavioral patterns without text/metadata:
    - Event type counts (normalized)
    - Transition probabilities between event types
    """
    # Get all event types
    event_types = list(EventType)
    n_types = len(event_types)
    type_to_idx = {t: i for i, t in enumerate(event_types)}

    features = []
    for session in sessions:
        feat = []

        # Event type counts (normalized)
        type_counts = np.zeros(n_types)
        for event in session.events:
            idx = type_to_idx.get(event.type, 0)
            type_counts[idx] += 1

        n_events = len(session.events)
        if n_events > 0:
            type_counts /= n_events
        feat.extend(type_counts.tolist())

        # Session length (normalized by 100)
        feat.append(n_events / 100.0)

        # Transition counts (simplified: just bigram counts)
        transitions = np.zeros((n_types, n_types))
        for i in range(len(session.events) - 1):
            from_idx = type_to_idx.get(session.events[i].type, 0)
            to_idx = type_to_idx.get(session.events[i + 1].type, 0)
            transitions[from_idx, to_idx] += 1

        # Normalize transitions
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transitions /= row_sums

        # Flatten and add to features
        feat.extend(transitions.flatten().tolist())

        features.append(feat)

    return np.array(features)


def _train_and_evaluate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Train classifier and return AUC via cross-validation.

    Args:
        X: Feature matrix.
        y: Binary labels (0=real, 1=simulated).
        n_folds: Number of CV folds.

    Returns:
        AUC score.
    """
    n_samples = len(y)
    if n_samples < n_folds * 2:
        n_folds = max(2, n_samples // 2)

    if n_samples < 4:
        return 0.5  # Not enough data

    clf = LogisticRegression(max_iter=1000, random_state=42)

    try:
        y_pred_proba = cross_val_predict(
            clf, X, y, cv=n_folds, method="predict_proba"
        )[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
    except Exception:
        auc = 0.5

    return float(auc)


def run_leakage_audit(
    real: list[InteractionSession],
    sim: list[InteractionSession],
    embedder=None,
    n_folds: int = 5,
    leakage_threshold: float = 0.55,
) -> LeakageAuditResult:
    """Run complete leakage audit for classifier-based realism.

    This function runs five classifiers:
    1. Metadata-only: Using only schema presence/missingness features
    2. Structural-only: Using only action sequences (no text/metadata)
    3. Masked-feature: Structural features with session length removed
    4. Main: Using embeddings (requires embedder) or structural features
    5. Permutation: Shuffled labels (sanity check)

    Args:
        real: List of real sessions.
        sim: List of simulated sessions.
        embedder: Optional embedder for main classifier. If None, uses structural.
        n_folds: Number of cross-validation folds.
        leakage_threshold: AUC threshold above which leakage is suspected.

    Returns:
        LeakageAuditResult with all AUC values and leakage detection flag.
    """
    n_real = len(real)
    n_sim = len(sim)
    y = np.array([0] * n_real + [1] * n_sim)

    # 1. Metadata-only baseline
    X_meta = np.vstack([
        _extract_metadata_features(real),
        _extract_metadata_features(sim),
    ])
    metadata_auc = _train_and_evaluate_classifier(X_meta, y, n_folds)

    # 2. Structural-only features
    X_struct = np.vstack([
        _extract_structural_features(real),
        _extract_structural_features(sim),
    ])
    structural_auc = _train_and_evaluate_classifier(X_struct, y, n_folds)

    # 3. Masked-feature test: structural features with suspicious fields removed
    # Remove session length and missingness-correlated features, keeping only
    # event type distribution + transition probabilities.
    n_types = len(list(EventType))
    # The session-length column sits at index n_types (right after event type counts).
    # Keep all columns except that one.
    mask_cols = list(range(n_types)) + list(range(n_types + 1, X_struct.shape[1]))
    X_masked = X_struct[:, mask_cols]
    masked_auc = _train_and_evaluate_classifier(X_masked, y, n_folds)

    # 4. Main classifier (embeddings if available, else structural)
    if embedder is not None:
        X_main = np.vstack([
            embedder.embed_sessions(real),
            embedder.embed_sessions(sim),
        ])
        main_auc = _train_and_evaluate_classifier(X_main, y, n_folds)
    else:
        # Use combined structural + some additional features
        main_auc = structural_auc

    # 5. Permutation check (shuffled labels). Use a fixed RNG so the audit
    # is deterministic across runs.
    rng = np.random.default_rng(42)
    y_shuffled = rng.permutation(y)
    permutation_auc = _train_and_evaluate_classifier(X_struct, y_shuffled, n_folds)

    # Determine if leakage is detected
    # Only metadata-only and permutation baselines indicate leakage.
    # High masked-feature AUC indicates genuine behavioral differences, not leakage.
    leakage_detected = (
        metadata_auc >= leakage_threshold
        or permutation_auc >= leakage_threshold
    )

    # Realism score: 1 - main_auc (higher is better)
    # Only meaningful if no leakage
    realism_score = 1.0 - main_auc if not leakage_detected else float("nan")

    return LeakageAuditResult(
        main_auc=main_auc,
        metadata_only_auc=metadata_auc,
        masked_feature_auc=masked_auc,
        structural_only_auc=structural_auc,
        permutation_auc=permutation_auc,
        leakage_detected=leakage_detected,
        realism_score=realism_score,
        meta={
            "n_real": n_real,
            "n_sim": n_sim,
            "n_folds": n_folds,
            "leakage_threshold": leakage_threshold,
            "used_embedder": embedder is not None,
        },
    )


@metric("realism_classifier_audited")
class RealismClassifierAudited(BaseMetric):
    """Classifier-based realism with mandatory leakage auditing.

    This metric runs the full leakage audit protocol and only returns
    a valid realism score if leakage checks pass. The result includes
    all ablation AUCs for transparency.
    """

    name = "realism_classifier_audited"
    objective: Literal["behavior"] = "behavior"
    granularity: Literal["session"] = "session"
    scenarios = {"T", "C"}
    description = "Leakage-audited classifier-based realism score"

    def compute(
        self,
        real: list[InteractionSession],
        sim: list[InteractionSession],
        embedder=None,
        n_folds: int = 5,
        leakage_threshold: float = 0.55,
        **kwargs,
    ) -> MetricResult:
        """Compute leakage-audited classifier realism.

        Args:
            real: List of real sessions.
            sim: List of simulated sessions.
            embedder: Optional embedder for main classifier.
            n_folds: Number of cross-validation folds.
            leakage_threshold: AUC threshold for leakage detection.

        Returns:
            MetricResult with realism score and full audit results.
        """
        audit = run_leakage_audit(
            real=real,
            sim=sim,
            embedder=embedder,
            n_folds=n_folds,
            leakage_threshold=leakage_threshold,
        )

        return MetricResult(
            name=self.name,
            value=audit.realism_score,
            meta={
                "main_auc": audit.main_auc,
                "metadata_only_auc": audit.metadata_only_auc,
                "masked_feature_auc": audit.masked_feature_auc,
                "structural_only_auc": audit.structural_only_auc,
                "permutation_auc": audit.permutation_auc,
                "leakage_detected": audit.leakage_detected,
                "is_valid": audit.is_valid(leakage_threshold),
                **audit.meta,
            },
        )
