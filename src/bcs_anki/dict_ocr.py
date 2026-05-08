"""Gemini-based OCR for Serbian-English thematic dictionary pages.

One invocation = one subject. Pass page image(s) of the same vocabulary
section in; get a parsed (subject, entries) result out, and write it as a
simple CSV with the subject on a `# Subject: ...` comment line.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from google.genai import types as genai_types

from .config import AppConfig
from .errors import EmptyLlmResponseError
from .gemini import _generate_with_retry
from .prompts import DICT_OCR_SYSTEM, DICT_OCR_USER

logger = logging.getLogger(__name__)


_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass
class DictEntry:
    number: int
    english: str
    serbian: str


@dataclass
class DictPage:
    subject: str
    entries: list[DictEntry]


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        return _MIME_BY_SUFFIX[suffix]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported image type {suffix!r} for {path}; "
            f"expected one of {sorted(_MIME_BY_SUFFIX)}"
        ) from exc


def _image_part(path: Path) -> genai_types.Part:
    return genai_types.Part.from_bytes(data=path.read_bytes(), mime_type=_mime_for(path))


def _parse_response(text: str) -> DictPage:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned non-JSON response: {text[:200]!r}") from exc

    if not isinstance(payload, dict) or "subject" not in payload or "entries" not in payload:
        raise ValueError(f"Gemini JSON missing required keys: {payload!r}")

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError(f"`entries` is not a list: {raw_entries!r}")

    entries: list[DictEntry] = []
    for item in raw_entries:
        try:
            entries.append(
                DictEntry(
                    number=int(item["n"]),
                    english=str(item["eng"]).strip(),
                    serbian=str(item["sr"]).strip(),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed entry from Gemini: {item!r}") from exc

    return DictPage(subject=str(payload["subject"]).strip(), entries=entries)


def extract_dict_pages(cfg: AppConfig, image_paths: list[Path]) -> DictPage:
    """Send dictionary page images to Gemini and return the parsed (subject, entries).

    All images must belong to the SAME subject (one Gemini call, one result).
    """
    if not image_paths:
        raise ValueError("extract_dict_pages requires at least one image path")

    contents: list = [_image_part(p) for p in image_paths]
    contents.append(DICT_OCR_USER)

    config = genai_types.GenerateContentConfig(
        system_instruction=DICT_OCR_SYSTEM,
        response_mime_type="application/json",
    )

    logger.info("OCR'ing %d dictionary page image(s) with Gemini", len(image_paths))
    response = _generate_with_retry(cfg, contents=contents, config=config)
    text = response.text
    if text is None:
        raise EmptyLlmResponseError("Gemini returned an empty response for dict OCR")

    page = _parse_response(text)
    logger.info("Parsed %d entries (subject: %r)", len(page.entries), page.subject)
    return page


def subject_slug(subject: str) -> str:
    """Filesystem-friendly slug for a subject heading.

    Lowercases, collapses runs of non-word characters into a single hyphen,
    strips edge hyphens. Preserves unicode letters (so "U šumi" → "u-šumi").
    """
    slug = re.sub(r"\W+", "-", subject, flags=re.UNICODE).strip("-").lower()
    return slug or "untitled"


def write_dict_csv(page: DictPage, output_path: Path) -> None:
    """Write the page to a CSV: `# Subject: ...` line, then `english,serbian` rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# Subject: {page.subject}\n")
        writer = csv.writer(fh)
        writer.writerow(["english", "serbian"])
        for entry in page.entries:
            writer.writerow([entry.english, entry.serbian])
