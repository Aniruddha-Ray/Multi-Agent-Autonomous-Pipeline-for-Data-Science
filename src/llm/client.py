# src/llm/client.py
"""LLM transport layer.

Thin structured-output wrapper around either a deterministic mock backend or
an optional real Anthropic model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Type, TypeVar

from pydantic import BaseModel as _BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Config

_SchemaT = TypeVar("_SchemaT", bound=_BaseModel)


class LLMClient:
    """Thin structured-output wrapper around a mock or real model backend."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._real_llm_available = False
        self._chat_model = None
        if not cfg.use_mock_llm and os.environ.get("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq  # type: ignore

                self._chat_model = ChatGroq(
                    model=cfg.llm_model, temperature=0
                )
                self._real_llm_available = True
            except Exception as exc:  # noqa: BLE001
                print(f"[LLMClient] Falling back to mock backend ({exc})")

    def structured_call(
        self,
        schema: Type[_SchemaT],
        system_prompt: str,
        user_prompt: str,
        mock_fn: Callable[[], _SchemaT],
    ) -> _SchemaT:
        """Return a schema instance, either from the mock backend or a real LLM."""
        if not self._real_llm_available:
            return mock_fn()
        try:
            structured_model = self._chat_model.with_structured_output(schema)
            result = structured_model.invoke(
                [("system", system_prompt), ("human", user_prompt)]
            )
            return result if isinstance(result, schema) else schema(**dict(result))
        except Exception as exc:  # noqa: BLE001
            print(f"[LLMClient] Real LLM call failed, using mock instead ({exc})")
            return mock_fn()
