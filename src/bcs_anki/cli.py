from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import click

from .workflows.common import load_app_config
from .workflows.dictionary import (
    run_dictionary_cards,
    run_dictionary_ocr,
    run_dictionary_pages,
    run_dictionary_refine,
)
from .workflows.words import run_words_pipeline


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
    run_words_pipeline(
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
    cfg = load_app_config(config_path)
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
    run_dictionary_pages(
        image_paths, output_csv, append, workers, config_path, verbose,
        skip_health_check, _pick_image_files,
    )


def _pick_image_files() -> tuple[Path, ...]:
    """Open a native file dialog for selecting dictionary page images."""
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
    run_dictionary_ocr(image_paths, output_csv, config_path, verbose, _pick_image_files)


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
    run_dictionary_refine(csv_paths, output_csv, config_path, verbose)


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
    run_dictionary_cards(
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
    run_words_pipeline(
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
    run_dictionary_ocr(image_paths, output_csv, config_path, verbose, _pick_image_files)


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
    run_dictionary_refine(csv_paths, output_csv, config_path, verbose)


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
    run_dictionary_cards(
        csv_paths, output_csv, append, workers, config_path, verbose,
        skip_health_check=dry_run,
    )


if __name__ == "__main__":
    main()
