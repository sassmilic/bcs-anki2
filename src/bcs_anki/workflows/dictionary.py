from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import click

from ..costs import COST_TRACKER
from ..dict_cards import run_generate_dict
from ..dict_ocr import extract_dict_pages
from ..dict_refine import refine_csv
from ..dictionary_csv import subject_slug, write_dict_csv
from ..health import check_apis
from .common import load_app_config

logger = logging.getLogger(__name__)

ImagePicker = Callable[[], tuple[Path, ...]]


def run_dictionary_ocr(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
    pick_image_files: ImagePicker,
) -> Path:
    cfg = load_app_config(config_path, verbose=verbose)

    if not image_paths:
        image_paths = pick_image_files()
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


def run_dictionary_refine(
    csv_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    verbose: bool,
) -> list[Path]:
    if output_csv is not None and len(csv_paths) > 1:
        raise click.UsageError("--output is only valid with a single input CSV.")

    cfg = load_app_config(config_path, verbose=verbose)
    outputs: list[Path] = []

    for src in csv_paths:
        dst = output_csv or (cfg.output_folder / "dict" / "refined" / src.name)
        click.echo(f"Refining {src} -> {dst}")
        n = refine_csv(cfg, src, dst)
        click.echo(f"  wrote {n} refined row(s)")
        outputs.append(dst)

    return outputs


def run_dictionary_cards(
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

    cfg = load_app_config(config_path, verbose=verbose)
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


def run_dictionary_pages(
    image_paths: tuple[Path, ...],
    output_csv: Optional[Path],
    append: bool,
    workers: Optional[int],
    config_path: Optional[Path],
    verbose: bool,
    skip_health_check: bool,
    pick_image_files: ImagePicker,
) -> None:
    cfg = load_app_config(config_path, verbose=verbose)
    if workers is not None:
        cfg.max_workers = workers

    if not skip_health_check:
        try:
            check_apis(cfg)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None

    if not image_paths:
        image_paths = pick_image_files()
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
