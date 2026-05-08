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
    ".heic": "image/heic",
    ".heif": "image/heif",
}


@dataclass
class DictEntry:
    # `number` is a string because some entries are category headers spanning a
    # range (e.g. "1-5") while individual rows are bare integers ("1", "2", ...).
    number: str
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
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict entry from Gemini: %r", item)
            continue
        n = item.get("n")
        eng = item.get("eng")
        sr = item.get("sr")
        # An entry is valid as long as both English and Serbian sides are
        # present. `n` can be a single number ("3") or a range ("1-5") for
        # category headers that span multiple subentries.
        if n in (None, "") or not eng or not sr:
            logger.warning("Skipping entry with missing n/eng/sr from Gemini: %r", item)
            continue
        entries.append(
            DictEntry(
                number=str(n).strip(),
                english=str(eng).strip(),
                serbian=str(sr).strip(),
            )
        )

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


def _write_csv_rows(subject: str, rows: list[tuple[str, str]], output_path: Path) -> None:
    """Low-level writer: `# Subject: ...` line, then header, then (english, serbian) rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# Subject: {subject}\n")
        writer = csv.writer(fh)
        writer.writerow(["english", "serbian"])
        for eng, sr in rows:
            writer.writerow([eng, sr])


def write_dict_csv(page: DictPage, output_path: Path) -> None:
    """Write the page to a CSV: `# Subject: ...` line, then `english,serbian` rows."""
    rows = [(e.english, e.serbian) for e in page.entries]
    _write_csv_rows(page.subject, rows, output_path)


def read_dict_csv(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Inverse of write_dict_csv: parse `# Subject: ...` + (english, serbian) rows.

    Returns (subject, rows). Raises ValueError on malformed input.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        first = fh.readline()
        if not first.startswith("# Subject:"):
            raise ValueError(f"{path}: missing `# Subject:` header on first line")
        subject = first[len("# Subject:"):].strip()

        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path}: missing CSV header row") from exc
        if header != ["english", "serbian"]:
            raise ValueError(f"{path}: unexpected CSV header {header!r}; expected ['english', 'serbian']")

        rows: list[tuple[str, str]] = []
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"{path}: row has {len(row)} columns, expected 2: {row!r}")
            rows.append((row[0], row[1]))

    return subject, rows
