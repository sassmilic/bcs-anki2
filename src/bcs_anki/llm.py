from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from .config import AppConfig
from .prompts import (
    CONTEXT_ADDENDUM,
    DEFINITION_SYSTEM,
    DEFINITION_USER,
    IMAGE_PROMPT_SYSTEM,
    IMAGE_PROMPT_USER,
    IMAGE_SEARCH_SYSTEM,
    IMAGE_SEARCH_USER,
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


def generate_definition_and_examples(cfg: AppConfig, word: str, context: str | None = None) -> GeneratedText:
    user_prompt = _with_context(DEFINITION_USER.format(word=word), context)
    text = _chat(cfg, DEFINITION_SYSTEM, user_prompt)

    parts = text.split("PRIMJERI:")
    if len(parts) == 2:
        definition_part, examples_part = parts
    else:
        lines = text.splitlines()
        definition_part = "\n".join(lines[:1])
        examples_part = "\n".join(lines[1:])

    return GeneratedText(definition_html=definition_part.strip(), examples_html=examples_part.strip())


def generate_image_prompt(cfg: AppConfig, word: str, context: str | None = None) -> str:
    user_prompt = _with_context(IMAGE_PROMPT_USER.format(word=word), context)
    return _chat(cfg, IMAGE_PROMPT_SYSTEM, user_prompt)


def generate_image_search_term(cfg: AppConfig, word: str, context: str | None = None) -> str:
    """Translate a BHS word into short English keywords for stock image search."""
    user_prompt = _with_context(IMAGE_SEARCH_USER.format(word=word), context)
    text = _chat(cfg, IMAGE_SEARCH_SYSTEM, user_prompt)
    return text.strip().splitlines()[0].strip()
