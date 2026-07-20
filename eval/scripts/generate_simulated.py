#!/usr/bin/env python
"""Generate simulated sessions for testing and demonstration.

This script creates synthetic simulated sessions with different characteristics
to test the SimEval-IR benchmarks without requiring actual simulator implementations.

Usage:
    python generate_simulated.py --output data/simulated/ --n-sessions 200
    python generate_simulated.py --output data/simulated/ --simulators heuristic,realistic,noisy
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simeval_ir.core.session import InteractionSession, Event, RankedItem, Click
from simeval_ir.core.types import EventType, Role, SessionType
from simeval_ir.datasets import save_sessions_to_jsonl

SAMPLE_TOPICS = [
    "programming", "cooking", "travel", "health", "finance",
    "sports", "technology", "science", "education", "entertainment",
]


def generate_heuristic_sessions(
    n_sessions: int,
    seed: int = 42,
) -> list[InteractionSession]:
    """Generate heuristic baseline sessions.

    Simple simulator that samples randomly with position bias.
    """
    rng = random.Random(seed)
    sessions = []

    queries = [
        "laptop review", "best phone 2024", "python tutorial",
        "machine learning", "travel deals", "recipe ideas",
    ]

    for i in range(n_sessions):
        n_queries = rng.randint(1, 3)
        events = []
        event_id = 0
        timestamp = 0.0

        for q_idx in range(n_queries):
            # Query
            event_id += 1
            events.append(Event(
                event_id=f"heur-{i}-e{event_id}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=rng.choice(queries),
            ))
            timestamp += rng.uniform(0.5, 2.0)

            # SERP
            event_id += 1
            n_results = rng.randint(5, 10)
            events.append(Event(
                event_id=f"heur-{i}-e{event_id}",
                type=EventType.SERP_VIEW,
                timestamp=timestamp,
                role=Role.SYSTEM,
                ranked_items=[
                    RankedItem(doc_id=f"doc-{rng.randint(1000, 9999)}", rank=r)
                    for r in range(1, n_results + 1)
                ],
            ))
            timestamp += rng.uniform(1.0, 5.0)

            # Clicks (simple position bias)
            clicks = []
            for r in range(1, n_results + 1):
                if rng.random() < 0.5 / r:  # Strong position bias
                    clicks.append(Click(
                        doc_id=f"doc-{rng.randint(1000, 9999)}",
                        rank=r,
                        dwell_time=rng.uniform(5, 30),
                    ))

            if clicks:
                event_id += 1
                events.append(Event(
                    event_id=f"heur-{i}-e{event_id}",
                    type=EventType.CLICK,
                    timestamp=timestamp,
                    role=Role.USER,
                    clicked_items=clicks,
                ))

            timestamp += rng.uniform(30, 120)

        sessions.append(InteractionSession(
            session_id=f"heuristic-{i}",
            dataset_id="simulated-heuristic",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"user-{rng.randint(1, 100)}",
            topic_id=rng.choice(SAMPLE_TOPICS),
            domain="synthetic",
        ))

    return sessions


def generate_realistic_sessions(
    n_sessions: int,
    seed: int = 123,
) -> list[InteractionSession]:
    """Generate more realistic simulated sessions.

    Simulates a click model with examination and relevance components.
    """
    rng = random.Random(seed)
    sessions = []

    queries = [
        "best laptop for programming", "how to learn python",
        "machine learning course", "travel insurance compare",
        "healthy dinner recipes", "smartphone camera review",
    ]

    for i in range(n_sessions):
        n_queries = rng.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]
        events = []
        event_id = 0
        timestamp = 0.0

        for q_idx in range(n_queries):
            # Query (with possible reformulation)
            event_id += 1
            query = rng.choice(queries)
            if q_idx > 0 and rng.random() < 0.4:
                query = query + " " + rng.choice(["review", "best", "compare", "2024"])

            events.append(Event(
                event_id=f"real-{i}-e{event_id}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=query,
            ))
            timestamp += rng.uniform(0.3, 1.5)

            # SERP
            event_id += 1
            n_results = 10
            ranked_items = []
            for r in range(1, n_results + 1):
                # Simulate relevance grades
                rel = rng.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
                ranked_items.append(RankedItem(
                    doc_id=f"doc-{i}-{q_idx}-{r}",
                    rank=r,
                    score=1.0 / r + rng.uniform(-0.1, 0.1),
                    judged_relevance=rel,
                ))

            events.append(Event(
                event_id=f"real-{i}-e{event_id}",
                type=EventType.SERP_VIEW,
                timestamp=timestamp,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            ))
            timestamp += rng.uniform(2.0, 8.0)

            # Clicks (cascade model: examine until satisfied or give up)
            clicks = []
            satisfied = False
            for item in ranked_items:
                if satisfied:
                    break

                # Examination probability decreases with rank
                exam_prob = 1.0 / (1 + 0.2 * (item.rank - 1))
                if rng.random() > exam_prob:
                    continue

                # Click probability depends on relevance
                rel = item.judged_relevance or 0
                click_prob = [0.1, 0.4, 0.8][rel]

                if rng.random() < click_prob:
                    dwell = rng.gauss(30 + rel * 20, 10)
                    dwell = max(5, dwell)
                    clicks.append(Click(
                        doc_id=item.doc_id,
                        rank=item.rank,
                        dwell_time=dwell,
                    ))

                    # Satisfaction probability
                    if rel >= 2 and rng.random() < 0.5:
                        satisfied = True

            if clicks:
                event_id += 1
                events.append(Event(
                    event_id=f"real-{i}-e{event_id}",
                    type=EventType.CLICK,
                    timestamp=timestamp,
                    role=Role.USER,
                    clicked_items=clicks,
                ))

            timestamp += rng.uniform(20, 90)

        sessions.append(InteractionSession(
            session_id=f"realistic-{i}",
            dataset_id="simulated-realistic",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"user-{rng.randint(1, 100)}",
            topic_id=rng.choice(SAMPLE_TOPICS),
            domain="synthetic",
        ))

    return sessions


def generate_noisy_sessions(
    n_sessions: int,
    seed: int = 456,
) -> list[InteractionSession]:
    """Generate noisy/random simulated sessions.

    Useful as a weak baseline to show metrics can detect poor simulators.
    """
    rng = random.Random(seed)
    sessions = []

    for i in range(n_sessions):
        # Random number of queries (often too many or too few)
        n_queries = rng.randint(0, 8)
        if n_queries == 0:
            n_queries = 1

        events = []
        event_id = 0
        timestamp = 0.0

        for q_idx in range(n_queries):
            # Random gibberish query
            event_id += 1
            query = " ".join(rng.choices("abcdefghij", k=rng.randint(1, 5)))

            events.append(Event(
                event_id=f"noisy-{i}-e{event_id}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=query,
            ))
            timestamp += rng.uniform(0.1, 10.0)

            # SERP with random size
            event_id += 1
            n_results = rng.randint(1, 20)
            events.append(Event(
                event_id=f"noisy-{i}-e{event_id}",
                type=EventType.SERP_VIEW,
                timestamp=timestamp,
                role=Role.SYSTEM,
                ranked_items=[
                    RankedItem(doc_id=f"doc-{rng.randint(1, 99999)}", rank=r)
                    for r in range(1, n_results + 1)
                ],
            ))
            timestamp += rng.uniform(0.1, 20.0)

            # Random clicks (no position bias)
            n_clicks = rng.randint(0, 5)
            if n_clicks > 0:
                event_id += 1
                events.append(Event(
                    event_id=f"noisy-{i}-e{event_id}",
                    type=EventType.CLICK,
                    timestamp=timestamp,
                    role=Role.USER,
                    clicked_items=[
                        Click(
                            doc_id=f"doc-{rng.randint(1, 99999)}",
                            rank=rng.randint(1, 20),
                            dwell_time=rng.uniform(0, 300),
                        )
                        for _ in range(n_clicks)
                    ],
                ))

            timestamp += rng.uniform(0, 200)

        sessions.append(InteractionSession(
            session_id=f"noisy-{i}",
            dataset_id="simulated-noisy",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"user-{rng.randint(1, 100)}",
            topic_id=rng.choice(SAMPLE_TOPICS),
            domain="synthetic",
        ))

    return sessions


def generate_simiir_pbm_sessions(
    n_sessions: int,
    seed: int = 789,
) -> list[InteractionSession]:
    """Generate SimIIR-style sessions using Position-Based Model (PBM).

    PBM assumes clicks depend only on position (examination) and relevance
    (attractiveness), with no cascade dependency between clicks.

    P(click|rank, rel) = P(exam|rank) * P(attract|rel)
    """
    rng = random.Random(seed)
    sessions = []

    # PBM parameters (typical values from literature)
    # Examination probability decreases with rank
    exam_probs = [0.95, 0.85, 0.70, 0.55, 0.40, 0.30, 0.22, 0.15, 0.10, 0.08]
    # Attractiveness probability by relevance grade (0, 1, 2)
    attract_probs = [0.05, 0.35, 0.75]

    queries = [
        "information retrieval", "search engine optimization",
        "web crawling techniques", "document ranking algorithms",
        "user click modeling", "query understanding",
    ]

    for i in range(n_sessions):
        n_queries = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        events = []
        event_id = 0
        timestamp = 0.0

        for q_idx in range(n_queries):
            # Query
            event_id += 1
            query = rng.choice(queries)
            if q_idx > 0 and rng.random() < 0.3:
                query = query + " " + rng.choice(["tutorial", "paper", "example"])

            events.append(Event(
                event_id=f"pbm-{i}-e{event_id}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=query,
            ))
            timestamp += rng.uniform(0.5, 2.0)

            # SERP
            event_id += 1
            n_results = 10
            ranked_items = []
            for r in range(1, n_results + 1):
                rel = rng.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0]
                ranked_items.append(RankedItem(
                    doc_id=f"doc-pbm-{i}-{q_idx}-{r}",
                    rank=r,
                    score=1.0 / r,
                    judged_relevance=rel,
                ))

            events.append(Event(
                event_id=f"pbm-{i}-e{event_id}",
                type=EventType.SERP_VIEW,
                timestamp=timestamp,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            ))
            timestamp += rng.uniform(1.0, 5.0)

            # PBM clicks: independent examination and attractiveness
            clicks = []
            for item in ranked_items:
                rank_idx = min(item.rank - 1, len(exam_probs) - 1)
                exam_prob = exam_probs[rank_idx]
                rel = item.judged_relevance or 0
                attract_prob = attract_probs[rel]

                # Click if examined AND attracted
                if rng.random() < exam_prob * attract_prob:
                    dwell = rng.gauss(25 + rel * 15, 8)
                    clicks.append(Click(
                        doc_id=item.doc_id,
                        rank=item.rank,
                        dwell_time=max(3, dwell),
                    ))

            if clicks:
                event_id += 1
                events.append(Event(
                    event_id=f"pbm-{i}-e{event_id}",
                    type=EventType.CLICK,
                    timestamp=timestamp,
                    role=Role.USER,
                    clicked_items=clicks,
                ))

            timestamp += rng.uniform(15, 60)

        sessions.append(InteractionSession(
            session_id=f"simiir-pbm-{i}",
            dataset_id="simulated-simiir-pbm",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"user-{rng.randint(1, 100)}",
            topic_id=rng.choice(SAMPLE_TOPICS),
            domain="synthetic",
            meta={"click_model": "PBM"},
        ))

    return sessions


def generate_simiir_dbn_sessions(
    n_sessions: int,
    seed: int = 999,
) -> list[InteractionSession]:
    """Generate SimIIR-style sessions using Dynamic Bayesian Network (DBN).

    DBN models cascade examination: users examine results sequentially
    and may stop after a satisfying click.

    P(click|rank, rel) = P(exam|rank, prev_clicks) * P(attract|rel)
    P(continue|clicked, satisfied) depends on satisfaction
    """
    rng = random.Random(seed)
    sessions = []

    # DBN parameters
    # Attractiveness by relevance
    attract_probs = [0.10, 0.45, 0.85]
    # Satisfaction probability given click (by relevance)
    satisfy_probs = [0.05, 0.30, 0.70]
    # Probability of continuing to examine after satisfaction
    gamma = 0.15

    queries = [
        "neural information retrieval", "bert for ranking",
        "learning to rank", "dense retrieval models",
        "click through rate prediction", "session search",
    ]

    for i in range(n_sessions):
        n_queries = rng.choices([1, 2, 3, 4], weights=[0.35, 0.40, 0.20, 0.05])[0]
        events = []
        event_id = 0
        timestamp = 0.0

        for q_idx in range(n_queries):
            # Query with reformulation
            event_id += 1
            query = rng.choice(queries)
            if q_idx > 0 and rng.random() < 0.45:
                # More sophisticated reformulation
                reformulations = ["2024", "survey", "comparison", "benchmark", "sota"]
                query = query + " " + rng.choice(reformulations)

            events.append(Event(
                event_id=f"dbn-{i}-e{event_id}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=query,
            ))
            timestamp += rng.uniform(0.3, 1.5)

            # SERP
            event_id += 1
            n_results = 10
            ranked_items = []
            for r in range(1, n_results + 1):
                # Higher relevance at top (more realistic ranking)
                if r <= 3:
                    rel = rng.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
                elif r <= 6:
                    rel = rng.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0]
                else:
                    rel = rng.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0]

                ranked_items.append(RankedItem(
                    doc_id=f"doc-dbn-{i}-{q_idx}-{r}",
                    rank=r,
                    score=1.0 / r + rng.uniform(-0.05, 0.05),
                    judged_relevance=rel,
                ))

            events.append(Event(
                event_id=f"dbn-{i}-e{event_id}",
                type=EventType.SERP_VIEW,
                timestamp=timestamp,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            ))
            timestamp += rng.uniform(2.0, 6.0)

            # DBN cascade clicks
            clicks = []
            examining = True
            satisfied = False

            for item in ranked_items:
                if not examining:
                    break

                rel = item.judged_relevance or 0
                attract_prob = attract_probs[rel]

                if rng.random() < attract_prob:
                    # Click!
                    dwell = rng.gauss(30 + rel * 20, 10)
                    clicks.append(Click(
                        doc_id=item.doc_id,
                        rank=item.rank,
                        dwell_time=max(5, dwell),
                    ))

                    # Check satisfaction
                    if rng.random() < satisfy_probs[rel]:
                        satisfied = True
                        # May still continue with low probability
                        if rng.random() > gamma:
                            examining = False
                else:
                    # Didn't click, continue examining (with decay)
                    if rng.random() > 0.85:  # Small chance to abandon
                        examining = False

            if clicks:
                event_id += 1
                events.append(Event(
                    event_id=f"dbn-{i}-e{event_id}",
                    type=EventType.CLICK,
                    timestamp=timestamp,
                    role=Role.USER,
                    clicked_items=clicks,
                ))

            # If satisfied, less likely to reformulate
            if satisfied and q_idx < n_queries - 1:
                if rng.random() < 0.6:
                    break  # End session early

            timestamp += rng.uniform(20, 80)

        sessions.append(InteractionSession(
            session_id=f"simiir-dbn-{i}",
            dataset_id="simulated-simiir-dbn",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"user-{rng.randint(1, 100)}",
            topic_id=rng.choice(SAMPLE_TOPICS),
            domain="synthetic",
            meta={"click_model": "DBN"},
        ))

    return sessions


def generate_llm_sim_sessions(
    n_sessions: int,
    seed: int = 1234,
) -> list[InteractionSession]:
    """Generate sessions mimicking LLM-based simulator behavior.

    LLM simulators tend to produce more natural language queries,
    diverse reformulations, and context-aware click patterns.
    """
    rng = random.Random(seed)
    sessions = []

    # More natural, varied queries that an LLM might generate
    query_templates = [
        "what is the best way to {topic}",
        "how do I {action} {topic}",
        "{topic} tutorial for beginners",
        "compare {topic} vs {topic2}",
        "{topic} best practices {year}",
        "why does {topic} matter",
    ]

    topics = ["machine learning", "python programming", "data science",
              "web development", "cloud computing", "cybersecurity"]
    actions = ["learn", "implement", "optimize", "debug", "deploy"]
    topics2 = ["deep learning", "java", "statistics", "mobile dev", "AWS"]

    for i in range(n_sessions):
        n_queries = rng.choices([1, 2, 3, 4, 5], weights=[0.2, 0.35, 0.30, 0.10, 0.05])[0]
        events = []
        event_id = 0
        timestamp = 0.0
        session_topic = rng.choice(topics)

        for q_idx in range(n_queries):
            # Generate natural query
            event_id += 1
            template = rng.choice(query_templates)
            query = template.format(
                topic=session_topic if rng.random() < 0.7 else rng.choice(topics),
                topic2=rng.choice(topics2),
                action=rng.choice(actions),
                year=rng.choice(["2024", "2025", ""]),
            ).strip()

            # LLM-style reformulation: more semantic variation
            if q_idx > 0 and rng.random() < 0.5:
                reformulations = [
                    f"more about {session_topic}",
                    f"{session_topic} examples",
                    f"alternatives to {session_topic}",
                    f"{session_topic} vs alternatives",
                ]
                query = rng.choice(reformulations)

            events.append(Event(
                event_id=f"llm-{i}-e{event_id}",
                type=EventType.QUERY_ISSUED,
                timestamp=timestamp,
                role=Role.USER,
                query=query,
            ))
            timestamp += rng.uniform(1.0, 4.0)  # LLM users may take longer to formulate

            # SERP
            event_id += 1
            n_results = 10
            ranked_items = []
            for r in range(1, n_results + 1):
                rel = rng.choices([0, 1, 2], weights=[0.45, 0.35, 0.20])[0]
                ranked_items.append(RankedItem(
                    doc_id=f"doc-llm-{i}-{q_idx}-{r}",
                    rank=r,
                    score=1.0 / r,
                    judged_relevance=rel,
                ))

            events.append(Event(
                event_id=f"llm-{i}-e{event_id}",
                type=EventType.SERP_VIEW,
                timestamp=timestamp,
                role=Role.SYSTEM,
                ranked_items=ranked_items,
            ))
            timestamp += rng.uniform(3.0, 10.0)  # More time examining

            # LLM-style clicks: more deliberate, often multiple
            clicks = []
            for item in ranked_items:
                rel = item.judged_relevance or 0
                # LLM users examine more positions
                exam_prob = 0.9 - (item.rank - 1) * 0.06
                click_prob = [0.08, 0.40, 0.80][rel]

                if rng.random() < exam_prob * click_prob:
                    # Longer dwell times (more reading)
                    dwell = rng.gauss(45 + rel * 25, 15)
                    clicks.append(Click(
                        doc_id=item.doc_id,
                        rank=item.rank,
                        dwell_time=max(10, dwell),
                    ))

            if clicks:
                event_id += 1
                events.append(Event(
                    event_id=f"llm-{i}-e{event_id}",
                    type=EventType.CLICK,
                    timestamp=timestamp,
                    role=Role.USER,
                    clicked_items=clicks,
                ))

            timestamp += rng.uniform(30, 120)

        sessions.append(InteractionSession(
            session_id=f"llm-sim-{i}",
            dataset_id="simulated-llm",
            session_type=SessionType.SEARCH,
            events=events,
            user_id=f"user-{rng.randint(1, 100)}",
            topic_id=rng.choice(SAMPLE_TOPICS),
            domain="synthetic",
            meta={"simulator_type": "LLM"},
        ))

    return sessions


SIMULATORS = {
    "heuristic": generate_heuristic_sessions,
    "realistic": generate_realistic_sessions,
    "noisy": generate_noisy_sessions,
    "simiir-pbm": generate_simiir_pbm_sessions,
    "simiir-dbn": generate_simiir_dbn_sessions,
    "llm-sim": generate_llm_sim_sessions,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated sessions for testing"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/simulated"),
        help="Output directory",
    )
    parser.add_argument(
        "--n-sessions", "-n",
        type=int,
        default=200,
        help="Number of sessions per simulator",
    )
    parser.add_argument(
        "--simulators", "-s",
        type=str,
        default="heuristic,simiir-pbm,simiir-dbn,llm-sim",
        help="Comma-separated list of simulators to generate. "
             "Available: heuristic, realistic, noisy, simiir-pbm, simiir-dbn, llm-sim",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    simulator_names = [s.strip() for s in args.simulators.split(",")]

    print("=" * 50)
    print("GENERATING SIMULATED SESSIONS")
    print("=" * 50)

    for i, sim_name in enumerate(simulator_names):
        if sim_name not in SIMULATORS:
            print(f"Unknown simulator: {sim_name}")
            continue

        print(f"\nGenerating {sim_name} sessions...")
        generator = SIMULATORS[sim_name]
        sessions = generator(args.n_sessions, seed=args.seed + i * 100)

        output_file = args.output / f"{sim_name}.jsonl"
        n_saved = save_sessions_to_jsonl(sessions, output_file)
        print(f"  Saved {n_saved} sessions to {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
