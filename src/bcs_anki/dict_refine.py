"""Gemini-powered refinement of ocr-dict CSVs.

For each (english, serbian) row, asks Gemini to (1) ensure the Serbian/SC
term is in ijekavian form, and (2) replace the English gloss with a better
translation when it isn't ideal in the section's subject context. Sends all
rows in a single batched call. Output is a parallel CSV in the same format
as the input — the original is left untouched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from google.genai import types as genai_types

from .config import AppConfig
from .dictionary_csv import read_dict_csv, write_dict_rows
from .errors import EmptyLlmResponseError
from .gemini import _generate_with_retry
from .prompts import DICT_REFINE_SYSTEM, DICT_REFINE_USER

logger = logging.getLogger(__name__)


def _parse_response(text: str, expected_len: int) -> list[tuple[str, str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned non-JSON response: {text[:200]!r}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"Gemini response is not a JSON array: {payload!r}")
    if len(payload) != expected_len:
        raise ValueError(
            f"Gemini returned {len(payload)} rows but {expected_len} were sent; refusing to align."
        )

    out: list[tuple[str, str]] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict) or "eng" not in item or "sr" not in item:
            raise ValueError(f"Row {i} from Gemini is malformed: {item!r}")
        out.append((str(item["eng"]).strip(), str(item["sr"]).strip()))
    return out


def refine_rows(
    cfg: AppConfig,
    subject: str,
    rows: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Send rows to Gemini for ijekavian + gloss review; return same-length list."""
    if not rows:
        logger.info("refine_rows: no rows to refine; skipping Gemini call")
        return []

    rows_json = json.dumps(
        [{"eng": eng, "sr": sr} for eng, sr in rows],
        ensure_ascii=False,
    )
    user_prompt = DICT_REFINE_USER.format(subject=subject, rows_json=rows_json)
    config = genai_types.GenerateContentConfig(
        system_instruction=DICT_REFINE_SYSTEM,
        response_mime_type="application/json",
    )

    logger.info("Refining %d row(s) for subject %r via Gemini", len(rows), subject)
    response = _generate_with_retry(cfg, contents=user_prompt, config=config)
    text = response.text
    if text is None:
        raise EmptyLlmResponseError("Gemini returned an empty response for dict refine")

    return _parse_response(text, expected_len=len(rows))


def refine_csv(cfg: AppConfig, input_path: Path, output_path: Path) -> int:
    """Read input CSV, refine via Gemini, write to output. Returns row count written."""
    subject, rows = read_dict_csv(input_path)
    refined = refine_rows(cfg, subject, rows)
    write_dict_rows(subject, refined, output_path)
    return len(refined)
