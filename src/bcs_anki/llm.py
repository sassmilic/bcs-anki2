from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from .config import AppConfig
from .images import ImageSource
import re

from .prompts import (
    CONTEXT_ADDENDUM,
    DEFINITION_SYSTEM,
    DEFINITION_USER,
    EXAMPLES_SYSTEM,
    EXAMPLES_USER,
    IMAGE_PROMPT_SYSTEM,
    IMAGE_PROMPT_USER,
    IMAGE_SEARCH_SYSTEM,
    IMAGE_SEARCH_USER,
    IMAGE_SOURCE_SYSTEM,
    IMAGE_SOURCE_USER,
    REFINE_DEFINITION_SYSTEM,
    REFINE_DEFINITION_USER,
    VALIDATE_DEFINITION_SYSTEM,
    VALIDATE_DEFINITION_USER,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedText:
    definition_html: str
    examples_html: str


def _get_client(cfg: AppConfig) -> OpenAI:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=cfg.openai_api_key)


def _chat(cfg: AppConfig, system_prompt: str, user_prompt: str) -> str:
    client = _get_client(cfg)
    response = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _with_context(prompt: str, context: str | None) -> str:
    if context is not None:
        return prompt + CONTEXT_ADDENDUM.format(context=context)
    return prompt


_CLOZE_RE = re.compile(r"\{\{c1::(.+?)\}\}")


def _mask_cloze(text: str) -> str:
    """Replace all {{c1::...}} with ___ for reverse-guess validation."""
    return _CLOZE_RE.sub("___", text)


def _parse_guesses(response: str) -> list[str]:
    """Parse 'word NN%' lines into a list of words, ordered by likelihood."""
    guesses = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip trailing percentage (e.g. "primirje 85%" -> "primirje")
        parts = line.rsplit(None, 1)
        if parts:
            word_part = parts[0].strip()
            guesses.append(word_part)
    return guesses


def _extract_lemma(definition_html: str) -> str | None:
    """Extract the first cloze-wrapped word as the lemma."""
    m = _CLOZE_RE.search(definition_html)
    return m.group(1) if m else None


def _validate_definition(cfg: AppConfig, word: str, definition_html: str) -> tuple[bool, list[str]]:
    """Reverse-guess validation: ask LLM to guess the word from the masked definition.

    Returns (passed, guesses) where passed is True if word matches the top guess.
    """
    masked = _mask_cloze(definition_html)
    user_prompt = VALIDATE_DEFINITION_USER.format(definition=masked)
    response = _chat(cfg, VALIDATE_DEFINITION_SYSTEM, user_prompt)
    guesses = _parse_guesses(response)
    logger.info("Definition validation for '%s': guesses=%s", word, guesses)

    lemma = _extract_lemma(definition_html) or word
    passed = bool(guesses) and guesses[0].lower() == lemma.lower()
    return passed, guesses


def _refine_definition(cfg: AppConfig, word: str, definition_html: str, wrong_guesses: list[str]) -> str:
    """Ask LLM to rewrite the definition to more clearly point to the target word."""
    plain_def = _mask_cloze(definition_html)
    user_prompt = REFINE_DEFINITION_USER.format(
        word=word,
        definition=plain_def,
        wrong_guesses=", ".join(wrong_guesses),
    )
    refined = _chat(cfg, REFINE_DEFINITION_SYSTEM, user_prompt).strip()

    # Fallback: if LLM didn't include cloze wrapping, re-wrap the lemma
    lemma = _extract_lemma(definition_html) or word
    if "{{c1::" not in refined and lemma.lower() in refined.lower():
        # Case-insensitive find and replace preserving original case
        import re as _re
        refined = _re.sub(
            _re.escape(lemma), "{{c1::" + lemma + "}}", refined, count=0, flags=_re.IGNORECASE,
        )

    return refined


def generate_definition_and_examples(cfg: AppConfig, word: str, context: str | None = None) -> GeneratedText:
    # Generate definition and examples with separate LLM calls
    def_prompt = _with_context(DEFINITION_USER.format(word=word), context)
    ex_prompt = _with_context(EXAMPLES_USER.format(word=word), context)

    definition_clean = _chat(cfg, DEFINITION_SYSTEM, def_prompt).strip()
    if definition_clean.startswith("DEFINICIJA:"):
        definition_clean = definition_clean[len("DEFINICIJA:"):].strip()

    # Validate definition via reverse-guess
    passed, guesses = _validate_definition(cfg, word, definition_clean)
    if passed:
        logger.info("Definition validated for '%s' — top guess matched", word)
    else:
        top_guess = guesses[0] if guesses else "???"
        logger.warning(
            "Definition for '%s' failed validation — top guess was '%s', refining...",
            word, top_guess,
        )
        definition_clean = _refine_definition(cfg, word, definition_clean, guesses)
        logger.info("Refined definition for '%s': %s", word, definition_clean)

    examples_clean = _chat(cfg, EXAMPLES_SYSTEM, ex_prompt).strip()

    return GeneratedText(definition_html=definition_clean, examples_html=examples_clean)


def decide_image_source(cfg: AppConfig, word: str, context: str | None = None) -> ImageSource:
    """Ask the LLM whether a word is best represented by a stock photo or AI image."""
    user_prompt = _with_context(IMAGE_SOURCE_USER.format(word=word), context)
    text = _chat(cfg, IMAGE_SOURCE_SYSTEM, user_prompt).strip().lower()
    if text in ("stock", "ai"):
        return text
    logger.warning("Unexpected image source response '%s' for '%s', defaulting to 'ai'", text, word)
    return "ai"


def generate_image_prompt(cfg: AppConfig, word: str, context: str | None = None) -> str:
    user_prompt = _with_context(IMAGE_PROMPT_USER.format(word=word), context)
    return _chat(cfg, IMAGE_PROMPT_SYSTEM, user_prompt)


def generate_image_search_term(cfg: AppConfig, word: str, context: str | None = None) -> str:
    """Translate a BHS word into short English keywords for stock image search."""
    user_prompt = _with_context(IMAGE_SEARCH_USER.format(word=word), context)
    text = _chat(cfg, IMAGE_SEARCH_SYSTEM, user_prompt)
    return text.strip().splitlines()[0].strip()
