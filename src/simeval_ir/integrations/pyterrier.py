"""PyTerrier integration for SimEval-IR.

This module provides utilities for converting between
PyTerrier data structures and SimEval-IR sessions.
"""

from typing import Any

import numpy as np

from simeval_ir.core.session import Event, InteractionSession, RankedItem
from simeval_ir.core.types import EventType, Role, SessionType


def from_pyterrier_experiment(
    topics,  # pd.DataFrame
    qrels,  # pd.DataFrame
    runs: dict[str, Any],  # dict[str, pd.DataFrame]
) -> dict[str, list[InteractionSession]]:
    """Convert PyTerrier experiment results to InteractionSession objects.

    Args:
        topics: PyTerrier topics DataFrame with qid, query columns.
        qrels: PyTerrier qrels DataFrame with qid, docno, label columns.
        runs: Dict mapping system names to run DataFrames (qid, docno, score, rank).

    Returns:
        Dict mapping system names to lists of InteractionSession objects.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "PyTerrier integration requires pandas. "
            "Install with: pip install simeval-ir[pyterrier]"
        )

    # Build qrels lookup
    qrels_lookup = {}
    for _, row in qrels.iterrows():
        qid = str(row["qid"])
        docno = str(row["docno"])
        label = int(row.get("label", row.get("relevance", 0)))
        if qid not in qrels_lookup:
            qrels_lookup[qid] = {}
        qrels_lookup[qid][docno] = label

    # Build query lookup
    query_lookup = {
        str(row["qid"]): row["query"]
        for _, row in topics.iterrows()
    }

    result = {}

    for system_name, run_df in runs.items():
        sessions = []

        # Group by query
        for qid, group in run_df.groupby("qid"):
            qid = str(qid)
            query_text = query_lookup.get(qid, "")

            # Create query event
            query_event = Event(
                event_id=f"{system_name}_{qid}_query",
                type=EventType.QUERY_ISSUED,
                role=Role.USER,
                query=query_text,
            )

            # Create SERP event with ranked items
            ranked_items = []
            group_sorted = group.sort_values("rank")
            for _, row in group_sorted.iterrows():
                docno = str(row["docno"])
                rank = int(row["rank"])
                score = float(row.get("score", 0))

                # Get relevance from qrels
                rel = qrels_lookup.get(qid, {}).get(docno)

                ranked_items.append(RankedItem(
                    doc_id=docno,
                    rank=rank,
                    score=score,
                    judged_relevance=rel,
                ))

            serp_event = Event(
                event_id=f"{system_name}_{qid}_serp",
                type=EventType.SERP_VIEW,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            )

            # Create session
            session = InteractionSession(
                session_id=f"{system_name}_{qid}",
                dataset_id="pyterrier",
                session_type=SessionType.SEARCH,
                events=[query_event, serp_event],
                topic_id=qid,
                meta={"system": system_name},
            )
            sessions.append(session)

        result[system_name] = sessions

    return result


def to_pyterrier_qrels(sessions: list[InteractionSession]):
    """Export sessions with relevance judgments to PyTerrier qrels format.

    Args:
        sessions: Sessions with relevance judgments in ranked_items.

    Returns:
        DataFrame with qid, docno, label columns.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "PyTerrier integration requires pandas. "
            "Install with: pip install simeval-ir[pyterrier]"
        )

    rows = []
    for session in sessions:
        qid = session.topic_id or session.session_id

        for event in session.events:
            if event.ranked_items:
                for item in event.ranked_items:
                    if item.judged_relevance is not None:
                        rows.append({
                            "qid": str(qid),
                            "docno": item.doc_id,
                            "label": item.judged_relevance,
                        })

    return pd.DataFrame(rows).drop_duplicates()


def to_pyterrier_run(sessions: list[InteractionSession], system_name: str = "simeval"):
    """Export sessions to PyTerrier run format.

    Args:
        sessions: Sessions with ranked items.
        system_name: Name to use for the system column.

    Returns:
        DataFrame with qid, docno, rank, score columns.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "PyTerrier integration requires pandas. "
            "Install with: pip install simeval-ir[pyterrier]"
        )

    rows = []
    for session in sessions:
        qid = session.topic_id or session.session_id

        for event in session.events:
            if event.ranked_items:
                for item in event.ranked_items:
                    rows.append({
                        "qid": str(qid),
                        "docno": item.doc_id,
                        "rank": item.rank,
                        "score": item.score or 0.0,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["name"] = system_name
    return df


class SimEvalEvaluator:
    """Evaluator that can be used in PyTerrier pipelines.

    Example:
        >>> evaluator = SimEvalEvaluator(metrics=["session_ndcg", "egu"])
        >>> results = evaluator.evaluate(run_df, qrels_df, topics_df)
    """

    def __init__(self, metrics: list[str] | None = None):
        """Initialize evaluator.

        Args:
            metrics: List of metric names to compute.
        """
        from simeval_ir.metrics import get_metric

        self.metric_names = metrics or ["session_ndcg"]
        self.metrics = [get_metric(name)() for name in self.metric_names]

    def evaluate(
        self,
        run,  # pd.DataFrame
        qrels,  # pd.DataFrame
        topics,  # pd.DataFrame
    ) -> dict[str, float]:
        """Evaluate a run using SimEval-IR metrics.

        Args:
            run: PyTerrier run DataFrame.
            qrels: PyTerrier qrels DataFrame.
            topics: PyTerrier topics DataFrame.

        Returns:
            Dict mapping metric names to values.
        """
        # Convert to sessions
        sessions_dict = from_pyterrier_experiment(
            topics, qrels, {"system": run}
        )
        sessions = sessions_dict["system"]

        # Compute metrics
        results = {}
        for metric in self.metrics:
            try:
                result = metric.compute(sessions=sessions)
                results[result.name] = result.value
            except Exception as e:
                results[metric.name] = float("nan")

        return results
