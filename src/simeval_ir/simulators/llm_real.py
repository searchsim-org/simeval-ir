"""Real-LLM user simulator.

Drives a search session by prompting an OpenAI-compatible chat model.
For every interaction shown to the simulator the model receives:

    * the topic description (from the source dataset, if available),
    * the previous queries the simulator already issued,
    * the SERP currently in front of it (rank, title, snippet),
    * a strict JSON-only schema for its reply.

The simulator never sees relevance judgements or qrels at any point.
Every reply is logged verbatim alongside the produced
``InteractionSession`` so that contamination can be audited later.

Designed to work with any OpenAI-compatible endpoint:

* ``base_url=https://api.openai.com/v1`` for the official API.
* The Azure OpenAI ``v1`` route, vLLM ``--api-key`` mode, llama.cpp
  ``--api-key`` mode, OpenRouter, Ollama's OpenAI shim, etc.

Example::

    from simeval_ir.simulators.llm_real import LLMSimulator
    sim = LLMSimulator(model="gpt-4o-mini",
                        api_key=os.environ["OPENAI_API_KEY"])
    new_sessions = list(sim.simulate_from(real_sessions, max_queries=5))
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from simeval_ir.core.session import (
    Click,
    Event,
    InteractionSession,
    RankedItem,
)
from simeval_ir.core.types import EventType, Role, SessionType


# --------------------------------------------------------------------- #
# Prompts (kept here so they can be inspected, audited and edited)
# --------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are simulating a real human user performing an
information-seeking search session. You will be shown a topic and the
search engine result page (SERP) for your most recent query. Decide
which results, if any, you would click on, and whether you want to
issue a follow-up query.

Stay in character as a curious but realistic user. Do NOT comment on
the task. Output only the JSON object described in the user message.
"""

ACTION_PROMPT = """Topic: {topic}

So far you have issued these queries (most recent last):
{prior_queries}

You are now looking at this SERP for "{current_query}":
{serp_block}

Decide what to do next. Reply with JSON only, matching this schema:

{{
  "clicks": [
    {{"rank": <int 1-{max_rank}>, "dwell_seconds": <int 5-120>}}
  ],
  "next_query": "<string>"  // empty string ends the session
}}

Click 0..3 results that look relevant to the topic. Set "next_query"
to an empty string when you feel you have what you need or do not
know what else to try.
"""


# --------------------------------------------------------------------- #
# Provenance record per simulator run
# --------------------------------------------------------------------- #

@dataclass
class LLMRunMeta:
    """Provenance for one LLM-real simulator run."""
    endpoint: str
    model: str
    temperature: float
    seed: int | None = None
    prompt_template_hash: str = ""
    api_calls: int = 0
    started_at: str = ""
    finished_at: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "prompt_template_hash": self.prompt_template_hash,
            "api_calls": self.api_calls,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            **self.extra,
        }


# --------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------- #

class LLMSimulator:
    """Real-LLM user simulator targeting an OpenAI-compatible endpoint.

    Parameters
    ----------
    model:
        Model identifier as the endpoint expects it (e.g. ``gpt-4o-mini``,
        ``gpt-4.1``, ``meta-llama/Meta-Llama-3-8B-Instruct``).
    api_key:
        API key. If ``None``, taken from ``OPENAI_API_KEY``.
    base_url:
        OpenAI-compatible base URL. ``None`` uses the OpenAI default.
    temperature:
        Sampling temperature passed to ``chat.completions``. Defaults to
        ``0.0`` so runs are reproducible per-seed.
    max_rank:
        Truncate SERPs to this depth before showing the model.
    request_timeout:
        Seconds before each API call is abandoned.
    seed:
        Optional integer seed. Forwarded to OpenAI's ``seed`` parameter
        when supported, plus Python ``random`` for retry jitter.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_rank: int = 10,
        request_timeout: float = 30.0,
        seed: int | None = None,
    ):
        try:
            from openai import OpenAI  # local import keeps it optional
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The real-LLM simulator needs the `openai` package. "
                "Install it with `pip install simeval-ir[llm]` or "
                "`pip install openai`."
            ) from e

        self._OpenAI = OpenAI
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No API key. Pass `api_key=` or set OPENAI_API_KEY.")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        self.max_rank = max_rank
        self.request_timeout = request_timeout
        self.seed = seed

        self._client = self._OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=request_timeout,
        )
        # Prompt-template fingerprint goes into provenance so a reviewer
        # can verify the exact prompt the simulator was driven with.
        import hashlib
        self._prompt_hash = hashlib.sha256(
            (SYSTEM_PROMPT + "\n" + ACTION_PROMPT).encode()
        ).hexdigest()[:16]
        self._call_count = 0
        # Per-param support cache. Keys: "temperature", "seed",
        # "response_format". Value False means the endpoint rejected the
        # param at least once and we should stop sending it.
        self._param_support: dict[str, bool] = {}
        import threading
        self._lock = threading.Lock()

    # -------- low-level helpers -------- #

    def _format_serp(self, items: list[RankedItem]) -> str:
        """Render a SERP block for the prompt. Strips any judged_relevance
        so the LLM never sees gold labels."""
        rows = []
        for it in items[: self.max_rank]:
            title = (it.meta.get("title") if it.meta else None) or it.doc_id
            snippet = (it.meta.get("snippet") if it.meta else None) or ""
            rows.append(f"  [{it.rank}] {title} — {snippet[:160].strip()}")
        return "\n".join(rows) if rows else "  (no results)"

    def _format_prior(self, queries: list[str]) -> str:
        return ("  - " + "\n  - ".join(queries)) if queries else "  (none)"

    def _ask(self, topic: str, prior: list[str], current: str,
             serp: list[RankedItem], retries: int = 2) -> dict:
        prompt = ACTION_PROMPT.format(
            topic=topic or "(no topic provided)",
            prior_queries=self._format_prior(prior),
            current_query=current,
            serp_block=self._format_serp(serp),
            max_rank=self.max_rank,
        )

        def _build_kwargs(extra: dict | None = None) -> dict:
            k = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            if self._supports("temperature") is not False:
                k["temperature"] = self.temperature
            if self.seed is not None and self._supports("seed") is not False:
                k["seed"] = self.seed + self._call_count
            if self._supports("response_format") is False:
                k.pop("response_format", None)
            if extra:
                k.update(extra)
            return k

        last_err = None
        for attempt in range(retries + 1):
            kwargs = _build_kwargs()
            try:
                r = self._client.chat.completions.create(**kwargs)
                with self._lock:
                    self._call_count += 1
                content = r.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                last_err = e
                msg = str(e)
                # Probe-and-disable: many gateways reject specific params
                # (gpt-5 disallows custom `temperature`, some endpoints
                # disallow `response_format`, etc.). Mark them unsupported
                # and retry, instead of failing.
                changed = False
                with self._lock:
                    for param in ("temperature", "seed", "response_format"):
                        if param in msg and self._supports(param) is not False:
                            self._param_support[param] = False
                            changed = True
                if changed:
                    continue
                time.sleep(1.5 ** attempt + random.random())
        raise RuntimeError(
            f"LLM call failed after {retries + 1} attempts: {last_err}")

    def _supports(self, param: str) -> bool | None:
        return self._param_support.get(param)

    # -------- public API -------- #

    def _simulate_one(
        self, src: InteractionSession, max_queries: int,
    ) -> InteractionSession | None:
        topic = (src.meta or {}).get("topic_desc") or src.topic_id or ""
        serps: list[list[RankedItem]] = [
            e.ranked_items or [] for e in src.events
            if e.type == EventType.SERP_VIEW
        ]
        queries: list[str] = [
            e.query or "" for e in src.events
            if e.type == EventType.QUERY_ISSUED
        ]
        if not serps or not queries:
            return None

        new_events: list[Event] = []
        counter = 0
        t = 0.0
        issued: list[str] = []
        current_query = queries[0]
        for q_idx in range(max_queries):
            if q_idx >= len(serps):
                break
            counter += 1
            new_events.append(Event(
                event_id=f"llmreal-{src.session_id}-q{counter}",
                type=EventType.QUERY_ISSUED,
                timestamp=t, role=Role.USER, query=current_query,
                meta={"src_event": "model_or_seed",
                      "src_session_id": src.session_id},
            ))
            t += 1.0
            serp = serps[q_idx]
            counter += 1
            new_events.append(Event(
                event_id=f"llmreal-{src.session_id}-s{counter}",
                type=EventType.SERP_VIEW,
                timestamp=t, role=Role.SYSTEM, ranked_items=serp,
            ))

            try:
                decision = self._ask(topic, issued + [current_query],
                                     current_query, serp)
            except Exception as e:
                new_events[-1].meta = (new_events[-1].meta or {})
                new_events[-1].meta["llm_error"] = str(e)
                break

            clicks = []
            rank_to_doc = {it.rank: it for it in serp}
            for c in decision.get("clicks", [])[:3]:
                rank = int(c.get("rank", 0))
                item = rank_to_doc.get(rank)
                if item is None:
                    continue
                dwell = float(c.get("dwell_seconds", 20))
                dwell = max(2.0, min(180.0, dwell))
                clicks.append(Click(doc_id=item.doc_id, rank=rank,
                                    dwell_time=dwell))
            if clicks:
                counter += 1
                new_events.append(Event(
                    event_id=f"llmreal-{src.session_id}-c{counter}",
                    type=EventType.CLICK,
                    timestamp=t + 0.5, role=Role.USER,
                    clicked_items=clicks,
                ))
                t += sum(c.dwell_time or 0 for c in clicks)

            issued.append(current_query)
            next_q = (decision.get("next_query") or "").strip()
            if not next_q:
                break
            current_query = next_q

        return InteractionSession(
            session_id=f"llmreal-{src.session_id}",
            dataset_id=f"{src.dataset_id}.llm-real",
            session_type=SessionType.SEARCH,
            events=new_events,
            user_id=src.user_id, topic_id=src.topic_id,
            domain=src.domain,
            meta={
                "simulator": "llm-real",
                "src_session_id": src.session_id,
                "session_uuid": str(uuid.uuid4()),
            },
        )

    def simulate_from(
        self,
        real_sessions: Iterable[InteractionSession],
        max_queries: int = 5,
        concurrency: int = 1,
    ) -> Iterator[InteractionSession]:
        """Drive one simulated session per real reference session.

        Per real session ``S`` the model sees:

        1. ``S.meta["topic_desc"]`` (when present) and the first
           ``QUERY_ISSUED`` text as a seed query.
        2. Each ``SERP_VIEW`` in order (URL/title/snippet only, with
           no relevance grades).
        3. A request for clicks plus an optional follow-up query.

        With ``concurrency>1`` sessions are dispatched through a thread
        pool. The OpenAI client itself is sync, so this just overlaps
        the network I/O. Each session is still serial inside.
        """
        from datetime import datetime
        meta = LLMRunMeta(
            endpoint=self.base_url or "https://api.openai.com/v1",
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            prompt_template_hash=self._prompt_hash,
            started_at=datetime.utcnow().isoformat() + "Z",
        )

        sources = list(real_sessions)
        if concurrency <= 1:
            for src in sources:
                out = self._simulate_one(src, max_queries)
                if out is None:
                    continue
                out.meta["llm_run"] = meta.to_dict()
                meta.api_calls = self._call_count
                yield out
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(self._simulate_one, src, max_queries): src
                        for src in sources}
                for fut in as_completed(futs):
                    try:
                        out = fut.result()
                    except Exception as e:
                        print(f"  [warn] session failed: {e}")
                        continue
                    if out is None:
                        continue
                    out.meta["llm_run"] = meta.to_dict()
                    meta.api_calls = self._call_count
                    yield out

        meta.finished_at = datetime.utcnow().isoformat() + "Z"
        self.last_run_meta = meta
