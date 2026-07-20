"""Simulator implementations bundled with SimEval-IR.

Two families:

* Parametric click models (``simiir_pbm``, ``simiir_dbn``, ``heuristic``,
  ``llm_style``) — fast, deterministic, literature-parameterised.
  These are the four simulators used in the paper's Tables 2 and 3.

* Real-LLM simulator (``llm_real``) — sends prompts to an
  OpenAI-compatible endpoint (the official OpenAI API, an Azure
  deployment, vLLM, llama.cpp's OpenAI mode, OpenRouter, Ollama's
  OpenAI shim, etc.) and converts the model's responses into the
  canonical ``InteractionSession`` schema.
"""

__all__ = ["llm_real"]
