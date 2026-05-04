"""Tier 3: Full word pipeline integration tests (all mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

from bcs_anki.cli import _process_word
from bcs_anki.csv_writer import ensure_header
from bcs_anki.llm import GeneratedText
from bcs_anki.progress import ProgressState, save_progress, load_progress


def _make_state(input_file: str = "words.txt") -> ProgressState:
    return ProgressState(
        input_file=input_file,
        total_words=5,
        completed_words=[],
        failed_words=[],
        last_updated="",
    )


class TestProcessWordSuccess:
    @patch("bcs_anki.cli.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.cli._fetch_image")
    @patch("bcs_anki.cli.generate_definition_and_examples")
    def test_returns_true_and_writes_csv(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        out_csv = tmp_path / "output.csv"
        progress_file = tmp_path / "progress.json"
        state = _make_state()
        save_progress(progress_file, state)

        mock_gen.return_value = GeneratedText(
            definition_html="{{c1::primirje}} — def",
            examples_html="1. Ex {{c1::one}}.<br>2. Ex {{c1::two}}.<br>3. Ex {{c1::three}}.",
        )
        mock_img.return_value = [("primirje_abc123.png", tmp_path / "primirje_abc123.png")]

        failed_csv = tmp_path / "failed.tsv"
        result = _process_word("primirje", mock_cfg, state, out_csv, progress_file, failed_csv)

        assert result is True
        assert "primirje" in state.completed_words

        # CSV should have header (4 lines) + 3 data rows
        lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 3

    @patch("bcs_anki.cli.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.cli._fetch_image")
    @patch("bcs_anki.cli.generate_definition_and_examples")
    def test_csv_has_correct_note_types(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        out_csv = tmp_path / "output.csv"
        progress_file = tmp_path / "progress.json"
        state = _make_state()
        save_progress(progress_file, state)

        mock_gen.return_value = GeneratedText(
            definition_html="{{c1::test}} — def", examples_html="Ex {{c1::test}}."
        )
        mock_img.return_value = [("img.png", tmp_path / "img.png")]

        failed_csv = tmp_path / "failed.tsv"
        _process_word("test", mock_cfg, state, out_csv, progress_file, failed_csv)

        lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
        data_lines = [l for l in lines if not l.startswith("#")]
        assert data_lines[0].startswith("Cloze")
        assert data_lines[1].startswith("Cloze")
        assert data_lines[2].startswith("Basic (and reversed card)")


class TestProcessWordFailure:
    @patch("bcs_anki.cli.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.cli._fetch_image")
    @patch("bcs_anki.cli.generate_definition_and_examples")
    def test_returns_false_on_error(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        out_csv = tmp_path / "output.csv"
        progress_file = tmp_path / "progress.json"
        state = _make_state()
        save_progress(progress_file, state)

        mock_gen.side_effect = RuntimeError("API error")
        mock_img.return_value = [("img.png", tmp_path / "img.png")]

        failed_csv = tmp_path / "failed.tsv"
        result = _process_word("fail_word", mock_cfg, state, out_csv, progress_file, failed_csv)

        assert result is False
        assert "fail_word" in state.failed_words
        assert failed_csv.exists()
        rows = failed_csv.read_text(encoding="utf-8").strip().splitlines()
        assert any(row.startswith("fail_word\t") and "API error" in row for row in rows)
