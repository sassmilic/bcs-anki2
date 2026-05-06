from __future__ import annotations

import logging
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .config import AppConfig
from .errors import EmptyLlmResponseError, MissingApiKeyError
from .prompts import (
    REVIEW_DEFINITION_SYSTEM,
    REVIEW_DEFINITION_USER,
    REVIEW_EXAMPLES_SYSTEM,
    REVIEW_EXAMPLES_USER,
)

logger = logging.getLogger(__name__)


_OK_SIGIL = "✓"

# Transient 5xx (typically 503 UNAVAILABLE — "spikes in demand are usually temporary")
# get retried with exponential backoff. 4xx errors (auth, quota, malformed input)
# are permanent for this request and bubble up immediately.
_MAX_ATTEMPTS = 3
_INITIAL_BACKOFF_SECONDS = 2.0


def _get_client(cfg: AppConfig) -> genai.Client:
    if not cfg.gemini_api_key:
        raise MissingApiKeyError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=cfg.gemini_api_key)


def _gemini_chat(cfg: AppConfig, system_prompt: str, user_prompt: str) -> str:
    delay = _INITIAL_BACKOFF_SECONDS
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            client = _get_client(cfg)
            response = client.models.generate_content(
                model=cfg.gemini_model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(system_instruction=system_prompt),
            )
            text = response.text
            if text is None:
                raise EmptyLlmResponseError("Gemini returned an empty response")
            return text
        except genai_errors.ServerError as exc:
            if attempt == _MAX_ATTEMPTS:
                raise
            logger.warning(
                "Gemini transient error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, _MAX_ATTEMPTS, delay, exc,
            )
            time.sleep(delay)
            delay *= 2
    # Unreachable: the loop either returns or raises.
    raise RuntimeError("Gemini retry loop exited without result")


def _apply_review(label: str, word: str, original: str, gemini_response: str) -> str:
    """Return the original if Gemini signaled OK, else log + return Gemini's correction."""
    stripped = gemini_response.strip()
    if stripped == _OK_SIGIL or stripped.startswith(_OK_SIGIL):
        return original
    logger.info("Gemini corrected %s for '%s':\n  before: %s\n  after:  %s", label, word, original, stripped)
    return stripped


def review_definition(cfg: AppConfig, word: str, definition_html: str) -> str:
    user_prompt = REVIEW_DEFINITION_USER.format(word=word, definition=definition_html)
    response = _gemini_chat(cfg, REVIEW_DEFINITION_SYSTEM, user_prompt)
    return _apply_review("definition", word, definition_html, response)


def review_examples(cfg: AppConfig, word: str, examples_html: str) -> str:
    user_prompt = REVIEW_EXAMPLES_USER.format(word=word, examples=examples_html)
    response = _gemini_chat(cfg, REVIEW_EXAMPLES_SYSTEM, user_prompt)
    return _apply_review("examples", word, examples_html, response)
