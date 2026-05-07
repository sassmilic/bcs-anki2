from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

from .config import AppConfig
from .csv_writer import CsvRow, append_rows
from .errors import (
    EmptyLlmResponseError,
    ImageRejectedError,
    NoStockResultsError,
    UnsupportedStockProviderError,
)
from .gemini import _gemini_chat
from .images import build_image_filename, fetch_stock_image, generate_ai_image
from .progress import mark_completed, mark_failed
from .prompts import (
    IJEKAVIAN_NORMALIZE_SYSTEM,
    IJEKAVIAN_NORMALIZE_USER,
    PARSE_DICTIONARY_PAGE_SYSTEM,
    PARSE_DICTIONARY_PAGE_USER,
    SIMPLE_DRAW_PROMPT,
)

logger = logging.getLogger(__name__)


OCR_LANG = "eng+srp_latn+hrv+bos"
# --oem 1 = LSTM-only engine (more accurate than the legacy hybrid for printed text).
# --psm 3 = fully automatic page segmentation (default; works for multi-column layouts).
OCR_CONFIG = "--oem 1 --psm 3"


@dataclass(frozen=True)
class DictionaryEntry:
    n: int
    english: str
    serbian_raw: str


@dataclass(frozen=True)
class DictionaryPage:
    section: str
    entries: list[DictionaryEntry]


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Light, conservative preprocessing for photos of book pages.

    Grayscale + auto-contrast usually helps Tesseract on under-exposed phone
    photos without distorting clean scans. We deliberately avoid binarization
    or deskew here — those can hurt more than they help on already-decent
    images.
    """
    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray, cutoff=1)


def ocr_page(image_path: Path) -> str:
    """Run Tesseract on a page image and return the raw recognized text."""
    with Image.open(image_path) as image:
        prepared = _preprocess_for_ocr(image)
    text = pytesseract.image_to_string(prepared, lang=OCR_LANG, config=OCR_CONFIG)
    logger.info("OCR'd %s: %d chars of text (lang=%s)", image_path, len(text), OCR_LANG)
    return text


def _strip_json_fences(text: str) -> str:
    """Tolerate ```json ... ``` wrappers Gemini sometimes adds despite instructions."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def parse_page(cfg: AppConfig, raw_ocr_text: str) -> DictionaryPage:
    """Send OCR text to Gemini, parse the returned JSON into a DictionaryPage."""
    user_prompt = PARSE_DICTIONARY_PAGE_USER.format(ocr_text=raw_ocr_text)
    response = _gemini_chat(cfg, PARSE_DICTIONARY_PAGE_SYSTEM, user_prompt)
    payload = _strip_json_fences(response)
    data = json.loads(payload)

    section = str(data.get("section", "") or "")
    entries: list[DictionaryEntry] = []
    for raw in data.get("entries", []):
        try:
            n = int(raw["n"])
            english = str(raw["english"]).strip()
            serbian = str(raw["serbian_raw"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed entry from parse: %r (%s)", raw, exc)
            continue
        if not english or not serbian:
            continue
        entries.append(DictionaryEntry(n=n, english=english, serbian_raw=serbian))

    logger.info("Parsed %s: section=%r, %d entries", "dictionary page", section, len(entries))
    return DictionaryPage(section=section, entries=entries)


def extract_page(cfg: AppConfig, image_path: Path) -> DictionaryPage:
    return parse_page(cfg, ocr_page(image_path))


def canonicalize_ijekavian(cfg: AppConfig, words: list[str]) -> list[str]:
    """Batch Gemini call: input list of Serbian words, output ijekavian list (same order, same length)."""
    if not words:
        return []
    user_prompt = IJEKAVIAN_NORMALIZE_USER.format(words_json=json.dumps(words, ensure_ascii=False))
    response = _gemini_chat(cfg, IJEKAVIAN_NORMALIZE_SYSTEM, user_prompt)
    payload = _strip_json_fences(response)
    parsed = json.loads(payload)

    if not isinstance(parsed, list):
        raise EmptyLlmResponseError(f"Expected JSON list from ijekavian normalizer, got: {type(parsed).__name__}")
    if len(parsed) != len(words):
        raise EmptyLlmResponseError(
            f"Ijekavian normalizer returned {len(parsed)} items for {len(words)} inputs"
        )
    return [str(w).strip() for w in parsed]


def _slugify_tag(text: str) -> str:
    """Best-effort tag-safe slug: lowercase, ASCII-ish, underscores."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_").lower()
    return cleaned


def _entry_key(section_slug: str, n: int) -> str:
    """Stable identifier for progress-tracking dedup. `<section>:<n>`."""
    return f"{section_slug}:{n}"


@dataclass
class DictionaryRunContext:
    """Per-run state for a dictionary pipeline run.

    Mirrors `pipeline.RunContext` but keys progress on entry IDs (section:n)
    rather than lemma strings.
    """

    cfg: AppConfig
    state: object  # ProgressState (avoid circular type import)
    out_csv: Path
    progress_file: Path
    failed_csv: Path
    section_slug: str
    failed_lock: threading.Lock


def _fetch_image_for_entry(cfg: AppConfig, entry: DictionaryEntry, dest: Path) -> bool:
    """Try Unsplash with the English term, fall back to AI generation. Returns True on success."""
    if cfg.stock_image_api and cfg.stock_image_api_key:
        try:
            fetch_stock_image(cfg, entry.english, dest, count=1)
            logger.info("Stock image fetched for entry #%d (%s)", entry.n, entry.english)
            return True
        except (NoStockResultsError, UnsupportedStockProviderError) as exc:
            logger.info(
                "Stock miss for entry #%d (%s), falling back to AI: %s",
                entry.n, entry.english, exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Stock provider error for entry #%d (%s), falling back to AI: %s",
                entry.n, entry.english, exc,
            )
    else:
        logger.debug("No stock provider configured; using AI image for entry #%d", entry.n)

    prompt = SIMPLE_DRAW_PROMPT.format(word=entry.english)
    try:
        generate_ai_image(cfg, prompt, dest)
        logger.info("AI image generated for entry #%d (%s)", entry.n, entry.english)
        return True
    except ImageRejectedError as exc:
        logger.warning("AI image rejected for entry #%d (%s): %s", entry.n, entry.english, exc)
        return False


def process_dictionary_entry(
    entry: DictionaryEntry,
    ijekavian: str,
    ctx: DictionaryRunContext,
) -> bool:
    """Process a single (entry, ijekavian-form) pair into a Basic+reversed flashcard row."""
    cfg = ctx.cfg
    key = _entry_key(ctx.section_slug, entry.n)

    if key in ctx.state.completed_words or key in ctx.state.failed_words:
        logger.info("Skipping %s: already recorded", key)
        return True

    try:
        if not ijekavian:
            raise ValueError(f"empty ijekavian form for entry #{entry.n}")

        cfg.temp_image_folder.mkdir(parents=True, exist_ok=True)
        img_filename = build_image_filename(ijekavian)
        img_path = cfg.temp_image_folder / img_filename

        ok = _fetch_image_for_entry(cfg, entry, img_path)
        if not ok:
            from .pipeline import _append_failed, summarize_exception  # local to avoid cycle
            reason = f"image rejected for '{entry.english}'"
            with ctx.failed_lock:
                with ctx.failed_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{key}\t{reason}\n")
            mark_failed(ctx.progress_file, ctx.state, key)
            return False

        tags = " ".join(t for t in (cfg.tags, ctx.section_slug) if t)
        row = CsvRow(
            note_type="Basic (and reversed card)",
            field1=f'<img src="{img_filename}">',
            field2=ijekavian.lower(),
            tags=tags,
        )
        append_rows(ctx.out_csv, [row])
        mark_completed(ctx.progress_file, ctx.state, key)
        return True

    except Exception as exc:  # noqa: BLE001
        from .pipeline import summarize_exception  # local to avoid cycle
        logger.exception("Failed dictionary entry %s: %s", key, exc)
        with ctx.failed_lock:
            with ctx.failed_csv.open("a", encoding="utf-8") as f:
                f.write(f"{key}\t{summarize_exception(exc)}\n")
        mark_failed(ctx.progress_file, ctx.state, key)
        return False
