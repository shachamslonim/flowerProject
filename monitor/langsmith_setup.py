"""LangSmith tracing setup for the combined flower pipeline. Tracing is opt-in: if
LANGCHAIN_API_KEY isn't set, init_langsmith() just warns and no-ops rather than failing -
the API/pipeline must work identically with or without a LangSmith account.

InstructionsForTreatment.llm_renderer and flower_state_ai.agent both already use
langchain_ollama, so once LANGCHAIN_TRACING_V2=true is set, their calls are traced
automatically with no code changes in those packages - this module only needs to turn
tracing on and let monitor.tracing's @traceable wrapper (see tracing.py) group everything
under one pipeline-level trace.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

_initialized = False


def init_langsmith(project: str | None = None) -> bool:
    """Idempotent - safe to call once at API startup and again per-request if needed.
    Returns True if tracing is actually active, False if it no-op'd (no API key)."""
    global _initialized
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        if not _initialized:
            print("[monitor] LANGCHAIN_API_KEY not set - LangSmith tracing disabled "
                  "(set it in the repo-root .env to enable; see smith.langchain.com)")
        _initialized = True
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ.setdefault("LANGCHAIN_PROJECT", project or os.environ.get("LANGCHAIN_PROJECT", "flower-pipeline"))

    if not _initialized:
        print(f"[monitor] LangSmith tracing enabled -> project '{os.environ['LANGCHAIN_PROJECT']}'")
    _initialized = True
    return True
