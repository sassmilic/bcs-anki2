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

logger = logging.getLogger(__name__)


def _load_app_config(config_path: Optional[Path | str], verbose: bool = False) -> AppConfig:
    path = Path(config_path).expanduser() if config_path else None
    cfg = load_config(path)
    setup_logging(cfg.log_file, verbose=verbose)
    return cfg


def _log_effective_config(cfg: AppConfig) -> None:
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


def _run_words_pipeline(
    input_file: Path,
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    verbose: bool,
    dry_run: bool,
    workers: Optional[int],
    append: bool,
) -> None:
    cfg = _load_app_config(config_path, verbose=verbose)
    if anki_media:
        cfg.anki_media_folder = anki_media.expanduser()
    if workers is not None:
        cfg.max_workers = workers

    _log_effective_config(cfg)

    if not dry_run:
        try:
            check_apis(cfg)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None

    out_csv = output_csv or (cfg.output_folder / (input_file.stem + ".csv"))
    if not append and out_csv.exists():
        out_csv.unlink()
        logger.info("Removed existing output file: %s", out_csv)
    ensure_header(out_csv)

    failed_csv = cfg.output_folder / f"{input_file.stem}_failed.tsv"
    if not append and failed_csv.exists():
        failed_csv.unlink()
    ensure_failed_header(failed_csv)

    words = [s for line in input_file.read_text(encoding="utf-8").splitlines() if (s := line.strip())]
    total_words = len(words)

    start_time = time.monotonic()
    processed_since_start = 0
    failed_words: list[str] = []

    click.echo(f"Processing {total_words} words from {input_file}...")

    if dry_run:
        for w in words:
            click.echo(f"[DRY-RUN] Would process: {w}")
    else:
        effective_workers = min(cfg.max_workers, len(words)) if words else 1
        click.echo(f"Using {effective_workers} workers for {len(words)} words.")

        ctx = RunContext(
            cfg=cfg,
            out_csv=out_csv,
            failed_csv=failed_csv,
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_to_word = {
                pool.submit(process_word, w, ctx): w
                for w in words
            }

            for future in as_completed(future_to_word):
                word = future_to_word[future]
                success = future.result()
                if not success:
                    failed_words.append(word)
                processed_since_start += 1

                # Periodic progress logging
                if processed_since_start % 10 == 0:
                    elapsed = time.monotonic() - start_time
                    avg_per_word = elapsed / processed_since_start
                    remaining = len(words) - processed_since_start
                    if remaining < 0:
                        remaining = 0
                    eta_seconds = remaining * avg_per_word
                    minutes, seconds = divmod(int(eta_seconds), 60)
                    eta_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

                    percent = (processed_since_start / total_words * 100) if total_words else 100.0
                    logger.info(
                        "Progress: %d/%d words processed (%.1f%%). Remaining: %d. Approximate remaining time: %s.",
                        processed_since_start,
                        total_words,
                        percent,
                        remaining,
                        eta_str,
                    )

    logger.info(
        "Finished processing. Total: %d, failed: %d.",
        total_words,
        len(failed_words),
    )

    cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
    logger.info("Token/cost summary for this run: %s", cost_summary)

    click.echo("Done.")
    click.echo(f"Token/cost summary: {cost_summary}")
    if failed_words:
        click.echo(f"Failed words: {', '.join(failed_words)}")
        click.echo(f"See {failed_csv} for failure reasons.")


def _run_dictionary_ocr(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> Path:
    cfg = _load_app_config(config_path, verbose=verbose)

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
    return output_csv


def _run_dictionary_refine(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> list[Path]:
    if output_csv is not None and len(csv_paths) > 1:
        raise click.UsageError("--output is only valid with a single input CSV.")

    cfg = _load_app_config(config_path, verbose=verbose)
    outputs: list[Path] = []

    for src in csv_paths:
        dst = output_csv or (cfg.output_folder / "dict" / "refined" / src.name)
        click.echo(f"Refining {src} -> {dst}")
        n = refine_csv(cfg, src, dst)
        click.echo(f"  wrote {n} refined row(s)")
        outputs.append(dst)

    return outputs


def _run_dictionary_cards(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    append: bool,
    workers: Optional[int],
    config_path: Optional[Path],
    verbose: bool,
    skip_health_check: bool,
) -> tuple[int, int]:
    if output_csv is not None and len(csv_paths) > 1:
        raise click.UsageError("--output is only valid with a single input CSV.")

    cfg = _load_app_config(config_path, verbose=verbose)
    if workers is not None:
        cfg.max_workers = workers

    if not skip_health_check:
        try:
            check_apis(cfg)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None

    total_completed = 0
    total_failed = 0
    for src in csv_paths:
        click.echo(f"Creating dictionary image cards from {src}")
        completed, failed = run_generate_dict(
            cfg, src, output_csv,
            append=append,
        )
        click.echo(f"  {completed} completed, {failed} failed")
        total_completed += completed
        total_failed += failed

    cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
    logger.info("Token/cost summary for this run: %s", cost_summary)
    click.echo(f"Total: {total_completed} completed, {total_failed} failed across {len(csv_paths)} file(s)")
    click.echo(f"Token/cost summary: {cost_summary}")
    return total_completed, total_failed


def _run_dictionary_pages(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    append: bool,
    workers: Optional[int],
    config_path: Optional[Path],
    verbose: bool,
    skip_health_check: bool,
) -> None:
    cfg = _load_app_config(config_path, verbose=verbose)
    if workers is not None:
        cfg.max_workers = workers

    if not skip_health_check:
        try:
            check_apis(cfg)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None

    if not image_paths:
        image_paths = _pick_image_files()
    if not image_paths:
        raise click.ClickException("No images selected.")

    click.echo(f"Selected {len(image_paths)} image(s):")
    for p in image_paths:
        click.echo(f"  - {p.resolve()}")

    click.echo(f"OCR'ing {len(image_paths)} dictionary page image(s) with Gemini...")
    page = extract_dict_pages(cfg, list(image_paths))

    slug = subject_slug(page.subject)
    raw_csv = cfg.output_folder / "dict" / f"{slug}.csv"
    refined_csv = cfg.output_folder / "dict" / "refined" / f"{slug}.csv"

    write_dict_csv(page, raw_csv)
    click.echo(f"Wrote raw dictionary CSV to {raw_csv}")

    click.echo(f"Refining {raw_csv} -> {refined_csv}")
    n = refine_csv(cfg, raw_csv, refined_csv)
    click.echo(f"  wrote {n} refined row(s)")

    click.echo(f"Creating dictionary image cards from {refined_csv}")
    completed, failed = run_generate_dict(
        cfg, refined_csv, output_csv,
        append=append,
    )
    click.echo(f"  {completed} completed, {failed} failed")

    cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
    logger.info("Token/cost summary for this run: %s", cost_summary)
    click.echo(f"Token/cost summary: {cost_summary}")


@click.group()
@click.version_option(package_name="bcs-anki")
def main() -> None:
    """BCS-to-Anki flashcard generator CLI."""


@main.command("words")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", "output_csv", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--anki-media", type=click.Path(file_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without making API calls.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--append", is_flag=True, help="Append to existing CSV instead of overwriting.")
def words_cmd(
    input_file: Path,
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    verbose: bool,
    dry_run: bool,
    workers: Optional[int],
    append: bool,
) -> None:
    """Create rich Anki cards from a personal word list."""
    _run_words_pipeline(
        input_file, output_csv, config_path, anki_media,
        verbose, dry_run, workers, append,
    )


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


@main.group("dictionary")
def dictionary_cmd() -> None:
    """Create cards from thematic dictionary sources."""


@dictionary_cmd.command("pages")
@click.argument(
    "image_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
)
@click.option(
    "--output", "-o", "output_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Final Anki CSV path. Defaults to <output_folder>/cards/<subject-slug>.csv.",
)
@click.option("--append", is_flag=True, help="Append to existing card CSV instead of overwriting.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
@click.option("--skip-health-check", is_flag=True, help="Skip startup API checks.")
def dictionary_pages_cmd(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    append: bool,
    workers: Optional[int],
    config_path: Optional[Path],
    verbose: bool,
    skip_health_check: bool,
) -> None:
    """Create Anki image cards from dictionary page image(s), end to end."""
    _run_dictionary_pages(
        image_paths, output_csv, append, workers, config_path, verbose,
        skip_health_check,
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


@dictionary_cmd.command("ocr")
@click.argument(
    "image_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
)
@click.option(
    "--output", "-o", "output_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output dictionary CSV path. Defaults to <output_folder>/dict/<subject-slug>.csv.",
)
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
def dictionary_ocr_cmd(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> None:
    """Extract a dictionary CSV from page image(s)."""
    _run_dictionary_ocr(image_paths, output_csv, config_path, verbose)


@dictionary_cmd.command("refine")
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
def dictionary_refine_cmd(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> None:
    """Refine dictionary CSVs: ekavian to ijekavian and improve English glosses.

    Default output: <output_folder>/dict/refined/<input-name>.csv (one file per input).
    Originals are left untouched.
    """
    _run_dictionary_refine(csv_paths, output_csv, config_path, verbose)


@dictionary_cmd.command("csv")
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
@click.option("--append", is_flag=True, help="Append to existing CSV instead of overwriting.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
@click.option("--skip-health-check", is_flag=True, help="Skip startup API checks.")
def dictionary_csv_cmd(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    append: bool,
    workers: Optional[int],
    config_path: Optional[Path],
    verbose: bool,
    skip_health_check: bool,
) -> None:
    """Create Anki image cards from prepared dictionary CSV(s).

    Tries a stock photo for each English term first, falling back to AI image
    generation. Default output: <output_folder>/cards/<subject-slug>.csv per
    input.
    """
    _run_dictionary_cards(
        csv_paths, output_csv, append, workers, config_path, verbose,
        skip_health_check,
    )


@main.command("generate", hidden=True)
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", "output_csv", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--anki-media", type=click.Path(file_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without making API calls.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--append", is_flag=True, help="Append to existing CSV instead of overwriting.")
def generate_alias_cmd(
    input_file: Path,
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    verbose: bool,
    dry_run: bool,
    workers: Optional[int],
    append: bool,
) -> None:
    """Deprecated alias for `words`."""
    _run_words_pipeline(
        input_file, output_csv, config_path, anki_media,
        verbose, dry_run, workers, append,
    )


@main.command("ocr-dict", hidden=True)
@click.argument(
    "image_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
)
@click.option(
    "--output", "-o", "output_csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output dictionary CSV path. Defaults to <output_folder>/dict/<subject-slug>.csv.",
)
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
def ocr_dict_alias_cmd(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> None:
    """Deprecated alias for `dictionary ocr`."""
    _run_dictionary_ocr(image_paths, output_csv, config_path, verbose)


@main.command("refine-dict", hidden=True)
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
def refine_dict_alias_cmd(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> None:
    """Deprecated alias for `dictionary refine`."""
    _run_dictionary_refine(csv_paths, output_csv, config_path, verbose)


@main.command("generate-dict", hidden=True)
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
@click.option("--append", is_flag=True, help="Append to existing CSV instead of overwriting.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Deprecated: skip API health check.")
def generate_dict_alias_cmd(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    append: bool,
    workers: Optional[int],
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
) -> None:
    """Deprecated alias for `dictionary csv`."""
    _run_dictionary_cards(
        csv_paths, output_csv, append, workers, config_path, verbose,
        skip_health_check=dry_run,
    )


if __name__ == "__main__":
    main()
