from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


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


def subject_slug(subject: str) -> str:
    """Filesystem-friendly slug for a dictionary subject heading."""
    slug = re.sub(r"\W+", "-", subject, flags=re.UNICODE).strip("-").lower()
    return slug or "untitled"


def write_dict_rows(subject: str, rows: list[tuple[str, str]], output_path: Path) -> None:
    """Write `# Subject: ...` plus `(english, serbian)` rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# Subject: {subject}\n")
        writer = csv.writer(fh)
        writer.writerow(["english", "serbian"])
        for eng, sr in rows:
            writer.writerow([eng, sr])


def write_dict_csv(page: DictPage, output_path: Path) -> None:
    """Write a parsed dictionary page to CSV."""
    rows = [(e.english, e.serbian) for e in page.entries]
    write_dict_rows(page.subject, rows, output_path)


def read_dict_csv(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Parse `# Subject: ...` plus `(english, serbian)` rows."""
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
