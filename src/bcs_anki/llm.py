from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from .config import AppConfig
from .gemini import review_definition, review_examples
from .images import ImageSource

from .prompts import (
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
    LEMMA_SYSTEM,
    LEMMA_USER,
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


def resolve_lemma(cfg: AppConfig, word: str) -> str:
    """Return the canonical (lemma) form of the input. Corrects spelling and inflection."""
    user_prompt = LEMMA_USER.format(word=word)
    text = _chat(cfg, LEMMA_SYSTEM, user_prompt).strip()
    return text.splitlines()[0].strip().strip('"').strip("'")


def generate_definition_and_examples(cfg: AppConfig, word: str) -> GeneratedText:
    def_prompt = DEFINITION_USER.format(word=word)
    ex_prompt = EXAMPLES_USER.format(word=word)

    definition_clean = _chat(cfg, DEFINITION_SYSTEM, def_prompt).strip()
    examples_clean = _chat(cfg, EXAMPLES_SYSTEM, ex_prompt).strip()

    if cfg.gemini_api_key:
        definition_clean = review_definition(cfg, word, definition_clean)
        examples_clean = review_examples(cfg, word, examples_clean)
    else:
        logger.info("Gemini review disabled (GEMINI_API_KEY not set), using OpenAI output as-is")

    return GeneratedText(definition_html=definition_clean, examples_html=examples_clean)


def decide_image_source(cfg: AppConfig, word: str) -> ImageSource:
    """Ask the LLM whether a word is best represented by a stock photo or AI image."""
    user_prompt = IMAGE_SOURCE_USER.format(word=word)
    text = _chat(cfg, IMAGE_SOURCE_SYSTEM, user_prompt).strip().lower()
    if text in ("stock", "ai"):
        return text
    logger.warning("Unexpected image source response '%s' for '%s', defaulting to 'ai'", text, word)
    return "ai"


def generate_image_prompt(cfg: AppConfig, word: str) -> str:
    user_prompt = IMAGE_PROMPT_USER.format(word=word)
    return _chat(cfg, IMAGE_PROMPT_SYSTEM, user_prompt).strip()


def generate_image_search_term(cfg: AppConfig, word: str) -> str:
    """Translate a BHS word into short English keywords for stock image search."""
    user_prompt = IMAGE_SEARCH_USER.format(word=word)
    text = _chat(cfg, IMAGE_SEARCH_SYSTEM, user_prompt)
    return text.strip().splitlines()[0].strip()
