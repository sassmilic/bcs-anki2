from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click

from .config import AppConfig, load_config
from .costs import COST_TRACKER
from .csv_writer import ensure_header
from .dict_cards import run_generate_dict
from .dict_ocr import extract_dict_pages, subject_slug, write_dict_csv
from .dict_refine import refine_csv
from .health import check_apis
from .logging_utils import setup_logging
from .pipeline import RunContext, ensure_failed_header, process_word
from .progress import ProgressState, load_progress, progress_path_for

logger = logging.getLogger(__name__)


def _load_app_config(config_path: Optional[str], verbose: bool = False) -> AppConfig:
    path = Path(config_path).expanduser() if config_path else None
    cfg = load_config(path)
    setup_logging(cfg.log_file, verbose=verbose)
    return cfg


def _process_word(
    raw_word: str,
    cfg: AppConfig,
    state: ProgressState,
    out_csv: Path,
    progress_file: Path,
    failed_csv: Path,
) -> bool:
    """Process a single word. Resolves the lemma first, then uses it for all downstream work."""
    logger.info("Processing input: %s", raw_word)
    word = raw_word

    claimed = False
    try:
        word = resolve_lemma(cfg, raw_word)
        if word != raw_word:
            logger.info("Resolved lemma '%s' -> '%s'", raw_word, word)

        with _lemma_lock:
            if word in state.completed_words or word in _lemmas_in_progress:
                logger.info("Skipping '%s': lemma already completed or in progress", word)
                return True
            _lemmas_in_progress.add(word)
            claimed = True

        with ThreadPoolExecutor(max_workers=2) as inner_pool:
            def_future = inner_pool.submit(generate_definition_and_examples, cfg, word)
            img_future = inner_pool.submit(_fetch_image, cfg, word)

            gen = def_future.result()
            img_result = img_future.result()

        rows = []
        if "{{c1::" in gen.definition_html:
            rows.append(CsvRow(
                note_type="Cloze",
                field1=gen.definition_html,
                field2="",
                tags=cfg.tags,
            ))
        else:
            logger.warning("Skipping definition card for '%s': no cloze markers found", word)

        if "{{c1::" in gen.examples_html:
            rows.append(CsvRow(
                note_type="Cloze",
                field1=gen.examples_html,
                field2="",
                tags=cfg.tags,
            ))
        else:
            logger.warning("Skipping examples card for '%s': no cloze markers found", word)

        if img_result is not None:
            img_html = "".join(f'<img src="{fn}">' for fn, _path in img_result)
            rows.append(CsvRow(
                note_type="Basic (and reversed card)",
                field1=img_html,
                field2=word.lower(),
                tags=cfg.tags,
            ))

        append_rows(out_csv, rows)
        mark_completed(progress_file, state, word)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process word '%s': %s", word, exc)
        _append_failed(failed_csv, word, _summarize_exception(exc))
        mark_failed(progress_file, state, word)
        return False
    finally:
        if claimed:
            with _lemma_lock:
                _lemmas_in_progress.discard(word)


@click.group()
@click.version_option(package_name="bcs-anki")
def main() -> None:
    """BCS-to-Anki flashcard generator CLI."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", "output_csv", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--anki-media", type=click.Path(file_okay=False, path_type=Path))
@click.option("--resume", "-r", is_flag=True, help="Resume from checkpoint if available.")
@click.option("--fresh", is_flag=True, help="Ignore checkpoint and start fresh.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without making API calls.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--append", is_flag=True, help="Append to existing CSV instead of overwriting.")
def generate(
    input_file: Path,
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    resume: bool,
    fresh: bool,
    verbose: bool,
    dry_run: bool,
    workers: Optional[int],
    append: bool,
) -> None:
    """Generate Anki-ready CSV and images from a word list."""
    cfg = _load_app_config(str(config_path) if config_path else None, verbose=verbose)
    if anki_media:
        cfg.anki_media_folder = anki_media.expanduser()
    if workers is not None:
        cfg.max_workers = workers

    # Log effective (sanitized) configuration
    safe_cfg = {
        "openai_api_key": "set" if cfg.openai_api_key else "not set",
        "gemini_api_key": "set" if cfg.gemini_api_key else "not set",
        "image_generation_model": cfg.image_generation_model,
        "image_size": cfg.image_size,
        "stock_image_api": cfg.stock_image_api or "none",
        "stock_image_api_key": "set" if cfg.stock_image_api_key else "not set",
        "anki_media_folder": str(cfg.anki_media_folder),
        "output_folder": str(cfg.output_folder),
        "temp_image_folder": str(cfg.temp_image_folder),
        "log_file": str(cfg.log_file),
        "rate_limit_delay_seconds": cfg.rate_limit_delay_seconds,
        "tags": cfg.tags,
        "llm_model": cfg.llm_model,
        "gemini_model": cfg.gemini_model,
        "max_workers": cfg.max_workers,
    }
    logger.info("Loaded configuration: %s", safe_cfg)

    if not dry_run:
        check_apis(cfg)

    out_csv = output_csv or (cfg.output_folder / (input_file.stem + ".csv"))
    if not append and out_csv.exists():
        out_csv.unlink()
        logger.info("Removed existing output file: %s", out_csv)
    ensure_header(out_csv)

    failed_csv = cfg.output_folder / f"{input_file.stem}_failed.tsv"
    if not append and failed_csv.exists():
        failed_csv.unlink()
    ensure_failed_header(failed_csv)

    progress_file = progress_path_for(input_file, cfg.output_folder)
    words = [s for line in input_file.read_text(encoding="utf-8").splitlines() if (s := line.strip())]
    total_words = len(words)

    state: ProgressState
    if fresh or not resume or not progress_file.exists():
        state = ProgressState(
            input_file=str(input_file),
            total_words=total_words,
            completed_words=[],
            failed_words=[],
            last_updated="",
        )
    else:
        existing = load_progress(progress_file)
        if existing and existing.input_file == str(input_file):
            state = existing
        else:
            state = ProgressState(
                input_file=str(input_file),
                total_words=total_words,
                completed_words=[],
                failed_words=[],
                last_updated="",
            )

    already_done = len(state.completed_words) + len(state.failed_words)
    start_time = time.monotonic()
    processed_since_start = 0

    click.echo(f"Processing {total_words} words from {input_file} (already done: {already_done})...")

    # Cheap pre-filter on raw input. The lemma may differ from the input, so
    # _process_word does a second dedup check after resolving the lemma.
    pending = [w for w in words if w not in state.completed_words]

    if dry_run:
        for w in pending:
            click.echo(f"[DRY-RUN] Would process: {w}")
    else:
        effective_workers = min(cfg.max_workers, len(pending)) if pending else 1
        click.echo(f"Using {effective_workers} workers for {len(pending)} remaining words.")

        ctx = RunContext(
            cfg=cfg,
            state=state,
            out_csv=out_csv,
            progress_file=progress_file,
            failed_csv=failed_csv,
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_to_word = {
                pool.submit(process_word, w, ctx): w
                for w in pending
            }

            for future in as_completed(future_to_word):
                word = future_to_word[future]
                success = future.result()
                processed_since_start += 1

                # Periodic progress logging
                if processed_since_start % 10 == 0:
                    elapsed = time.monotonic() - start_time
                    avg_per_word = elapsed / processed_since_start
                    remaining = len(pending) - processed_since_start
                    if remaining < 0:
                        remaining = 0
                    eta_seconds = remaining * avg_per_word
                    minutes, seconds = divmod(int(eta_seconds), 60)
                    eta_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

                    done = already_done + processed_since_start
                    percent = (done / total_words * 100) if total_words else 100.0
                    logger.info(
                        "Progress: %d/%d words processed (%.1f%%). Remaining: %d. Approximate remaining time: %s.",
                        done,
                        total_words,
                        percent,
                        remaining,
                        eta_str,
                    )

    # Final progress/ETA log
    done_total = len(state.completed_words) + len(state.failed_words)
    remaining_total = max(total_words - done_total, 0)
    logger.info(
        "Finished processing. Total: %d, completed/failed: %d, remaining (unprocessed in file): %d.",
        total_words,
        done_total,
        remaining_total,
    )

    cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
    logger.info("Token/cost summary for this run: %s", cost_summary)

    click.echo("Done.")
    click.echo(f"Token/cost summary: {cost_summary}")
    if state.failed_words:
        click.echo(f"Failed words: {', '.join(state.failed_words)}")
        click.echo(f"See {failed_csv} for failure reasons.")


@main.command("copy-media")
@click.option("--from", "src", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Source folder (default: temp_image_folder from config).")
@click.option("--to", "dst", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Destination folder (default: anki_media_folder from config).")
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
def copy_media(src: Optional[Path], dst: Optional[Path], config_path: Optional[Path]) -> None:
    """Copy generated images to the Anki media folder."""
    cfg = _load_app_config(str(config_path) if config_path else None)
    src = src or cfg.temp_image_folder
    dst = dst or cfg.anki_media_folder

    if not src.exists():
        raise click.ClickException(f"Source folder does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for file in src.iterdir():
        if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            shutil.copy2(file, dst / file.name)
            count += 1

    click.echo(f"Copied {count} media files from {src} to {dst}.")


@main.command("ocr-dict")
@click.argument(
    "image_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
)
@click.option(
    "--output", "-o", "output_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output CSV path. Defaults to <output_folder>/dict/<subject-slug>.csv.",
)
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
def ocr_dict(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> None:
    """Extract a Serbian-English vocabulary section from dictionary page image(s).

    Run with no IMAGE_PATHS to open a file picker. Sends the page(s) to Gemini
    in one multimodal request and writes a CSV with the subject as a
    `# Subject: ...` comment header and one (english,serbian) row per numbered
    entry. All images must belong to the SAME subject.
    """
    cfg = _load_app_config(str(config_path) if config_path else None, verbose=verbose)

    if not image_paths:
        image_paths = _pick_image_files()
    if not image_paths:
        raise click.ClickException("No images selected.")

    click.echo(f"Selected {len(image_paths)} image(s):")
    for p in image_paths:
        click.echo(f"  - {p.resolve()}")

    click.echo(f"OCR'ing {len(image_paths)} dictionary page image(s) with Gemini...")
    page = extract_dict_pages(cfg, list(image_paths))

    if output_csv is None:
        output_csv = cfg.output_folder / "dict" / f"{subject_slug(page.subject)}.csv"

    write_dict_csv(page, output_csv)
    click.echo(
        f"Wrote {len(page.entries)} entries (subject: {page.subject!r}) to {output_csv}"
    )


def _pick_image_files() -> tuple[Path, ...]:
    """Open a native file dialog for selecting dictionary page images.

    Uses stdlib tkinter so no extra dependency is needed. Returns an empty
    tuple if the user cancels.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilenames(
            title="Select dictionary page image(s)",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp *.heic *.heif"),
                ("All files", "*"),
            ],
        )
    finally:
        root.destroy()
    return tuple(Path(p) for p in selected)


@main.command("refine-dict")
@click.argument(
    "csv_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
    required=True,
)
@click.option(
    "--output", "-o", "output_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path (single CSV). With multiple inputs, must be omitted.",
)
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
def refine_dict(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> None:
    """Refine ocr-dict CSVs: ekavian→ijekavian and improve English glosses.

    Default output: <output_folder>/dict/refined/<input-name>.csv (one file per input).
    Originals are left untouched. Multi-file: bcs-anki refine-dict output/dict/*.csv.
    """
    if output_csv is not None and len(csv_paths) > 1:
        raise click.UsageError("--output is only valid with a single input CSV.")

    cfg = _load_app_config(str(config_path) if config_path else None, verbose=verbose)

    for src in csv_paths:
        dst = output_csv or (cfg.output_folder / "dict" / "refined" / src.name)
        click.echo(f"Refining {src} → {dst}")
        n = refine_csv(cfg, src, dst)
        click.echo(f"  wrote {n} refined row(s)")


@main.command("generate-dict")
@click.argument(
    "csv_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
    required=True,
)
@click.option(
    "--output", "-o", "output_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output CSV path (single-input only). Defaults to <output_folder>/cards/<subject-slug>.csv.",
)
@click.option("--resume", "-r", is_flag=True, help="Resume from checkpoint if available.")
@click.option("--fresh", is_flag=True, help="Ignore checkpoint and start fresh.")
@click.option("--append", is_flag=True, help="Append to existing CSV instead of overwriting.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Skip API health check.")
def generate_dict_cmd(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    resume: bool,
    fresh: bool,
    append: bool,
    workers: Optional[int],
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
) -> None:
    """Generate Anki Basic+reversed flashcards (image ↔ Serbian) from refined dict CSV(s).

    Tries a stock photo for each English term first, falling back to AI image
    generation. Default output: <output_folder>/cards/<subject-slug>.csv per
    input. Multi-file: bcs-anki generate-dict output/dict/refined/*.csv.
    """
    if output_csv is not None and len(csv_paths) > 1:
        raise click.UsageError("--output is only valid with a single input CSV.")

    cfg = _load_app_config(str(config_path) if config_path else None, verbose=verbose)
    if workers is not None:
        cfg.max_workers = workers

    if not dry_run:
        check_apis(cfg)

    total_completed = 0
    total_failed = 0
    for src in csv_paths:
        click.echo(f"Generating cards from {src}")
        completed, failed = run_generate_dict(
            cfg, src, output_csv,
            resume=resume, fresh=fresh, append=append,
        )
        click.echo(f"  {completed} completed, {failed} failed")
        total_completed += completed
        total_failed += failed

    cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
    logger.info("Token/cost summary for this run: %s", cost_summary)
    click.echo(f"Total: {total_completed} completed, {total_failed} failed across {len(csv_paths)} file(s)")


if __name__ == "__main__":
    main()

