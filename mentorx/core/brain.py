"""
MentorX Core — Brain
====================
Provider-agnostic reasoning interface.

Every agent in MentorX calls Brain.think(...) — they don't know or care
which model provider is underneath. We swap providers via the
MENTORX_PROVIDER environment variable. Zero code changes.

Supported providers:
    - "gemini"    → Google Gemini (free tier, default during dev)
    - "anthropic" → Anthropic Claude (when sponsor credits land)
"""
from __future__ import annotations

import os
from typing import Optional


class Brain:
    """The single reasoning interface for all MentorX agents."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or os.getenv("MENTORX_PROVIDER", "gemini")).lower()
        self.model = model or self._default_model()
        self._client = self._init_client()

    def _default_model(self) -> str:
        if self.provider == "gemini":
            return "gemini-2.5-flash"
        if self.provider == "anthropic":
            return "claude-opus-4-5"
        raise ValueError(f"Unknown provider: {self.provider}")

    def _init_client(self):
        if self.provider == "gemini":
            from google import genai
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic()
        raise ValueError(f"Unknown provider: {self.provider}")

    def think(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> str:
        """Send a system + user message, get back the model's reply as a string.

        Args:
            system: The system instruction (role, persona, rules).
            user: The user message.
            max_tokens: Max output tokens (default 2048 — generous for free tier).
            thinking: If False (default), disable Gemini's internal thinking budget
                      so the full token budget goes to the visible reply.
        """
        if self.provider == "gemini":
            return self._think_gemini(system, user, max_tokens, thinking)
        if self.provider == "anthropic":
            return self._think_anthropic(system, user, max_tokens)
        raise ValueError(f"Unknown provider: {self.provider}")

    # ---- Provider-specific implementations ----

    def _think_gemini(self, system: str, user: str, max_tokens: int, thinking: bool) -> str:
        from google.genai import types

        config_kwargs = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        # Disable thinking mode by default so the full budget goes to the visible reply.
        # We can opt back in for hard reasoning tasks later.
        if not thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text or ""

    def _think_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# Convenience singleton
_default_brain: Optional[Brain] = None


def get_brain() -> Brain:
    global _default_brain
    if _default_brain is None:
        _default_brain = Brain()
    return _default_brain
