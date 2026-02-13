from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


HEADER = """#separator:Tab
#html:true
#notetype column:1
#tags column:4
"""


@dataclass
class CsvRow:
    note_type: str
    field1: str
    field2: str
    tags: str


def _escape_field(value: str) -> str:
    # Anki tab-separated format; minimal escaping (we trust HTML).
    return value.replace("\n", "<br>")


def ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER, encoding="utf-8")


def append_rows(path: Path, rows: list[CsvRow]) -> None:
    ensure_header(path)
    with path.open("a", encoding="utf-8") as f:
        _write_rows(f, rows)


def _write_rows(f: TextIO, rows: list[CsvRow]) -> None:
    for row in rows:
        line = "\t".join(
            [
                row.note_type,
                _escape_field(row.field1),
                _escape_field(row.field2),
                row.tags,
            ]
        )
        f.write(line + "\n")

