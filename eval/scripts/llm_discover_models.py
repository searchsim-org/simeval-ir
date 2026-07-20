#!/usr/bin/env python
"""List the models available on each OpenAI-compatible endpoint.

This is meant to be run by the user once, before kicking off any LLM
simulator runs, so that the model id passed to ``simeval generate-llm-sim``
is known to be valid on the chosen endpoint.

Usage:
    # Reads ~/.env or shell env: OPENAI_API_KEY, CUSTOM_LLM_ENDPOINT,
    # CUSTOM_LLM_API_KEY
    python eval/scripts/llm_discover_models.py

The script does not call any chat-completion endpoint. It only sends an
authenticated GET to ``$BASE_URL/v1/models``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def list_models(label: str, base_url: str, api_key: str) -> list[str]:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Run: pip install openai")
    print(f"\n=== {label} ===  base_url={base_url}")
    if not api_key:
        print("  (no API key set; skipping)")
        return []
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        models = client.models.list()
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")
        return []
    ids = sorted(m.id for m in models.data)
    print(f"  {len(ids)} model(s):")
    for m in ids:
        print(f"    - {m}")
    return ids


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _load_dotenv(repo_root.parent / ".env")  # SimEval-IR/.env (one level up from simeval-ir/)

    list_models(
        "OpenAI",
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    custom_base = os.environ.get("CUSTOM_LLM_ENDPOINT", "")
    if custom_base:
        if not custom_base.rstrip("/").endswith("/v1"):
            custom_base = custom_base.rstrip("/") + "/v1"
        list_models(
            "Custom endpoint",
            base_url=custom_base,
            api_key=os.environ.get("CUSTOM_LLM_API_KEY", ""),
        )


if __name__ == "__main__":
    main()
