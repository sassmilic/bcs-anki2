"""Tier 2: CSV writer tests with file I/O."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bcs_anki.csv_writer import HEADER, CsvRow, append_rows, ensure_header


class TestEnsureHeader:
    def test_creates_file_with_header(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        ensure_header(csv)
        assert csv.exists()
        assert csv.read_text(encoding="utf-8") == HEADER

    def test_idempotent(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        ensure_header(csv)
        ensure_header(csv)
        assert csv.read_text(encoding="utf-8") == HEADER

    def test_creates_parent_dirs(self, tmp_path: Path):
        csv = tmp_path / "sub" / "dir" / "test.csv"
        ensure_header(csv)
        assert csv.exists()


class TestAppendRows:
    def test_produces_valid_tsv(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        rows = [
            CsvRow("Cloze", "def text", "", "tag1"),
            CsvRow("Basic (and reversed card)", "<img>", "word", "tag1"),
        ]
        append_rows(csv, rows)
        text = csv.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        # 4 header lines + 2 data lines
        assert len(lines) == 6
        # Check tab columns
        data_line = lines[4]
        parts = data_line.split("\t")
        assert len(parts) == 4
        assert parts[0] == "Cloze"

    def test_correct_columns(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        row = CsvRow("Cloze", "field1", "field2", "tags")
        append_rows(csv, [row])
        lines = csv.read_text(encoding="utf-8").strip().splitlines()
        parts = lines[4].split("\t")
        assert parts == ["Cloze", "field1", "field2", "tags"]

    def test_thread_safety(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        ensure_header(csv)

        def write_one(i: int) -> None:
            row = CsvRow("Cloze", f"content_{i}", "", "tag")
            append_rows(csv, [row])

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(write_one, i) for i in range(10)]
            for f in as_completed(futures):
                f.result()

        text = csv.read_text(encoding="utf-8")
        lines = [l for l in text.strip().splitlines() if l and not l.startswith("#")]
        assert len(lines) == 10
        # No interleaved content
        for line in lines:
            parts = line.split("\t")
            assert len(parts) == 4
