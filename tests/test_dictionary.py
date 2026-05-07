"""Tests for the dictionary-page pipeline."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bcs_anki.csv_writer import ensure_header
from bcs_anki.dictionary import (
    DictionaryEntry,
    DictionaryPage,
    DictionaryRunContext,
    canonicalize_ijekavian,
    ocr_page,
    parse_page,
    process_dictionary_entry,
    _slugify_tag,
)
from bcs_anki.errors import ImageRejectedError, NoStockResultsError
from bcs_anki.progress import ProgressState


class TestOcrPage:
    @patch("bcs_anki.dictionary._preprocess_for_ocr", side_effect=lambda im: im)
    @patch("bcs_anki.dictionary.pytesseract.image_to_string")
    @patch("bcs_anki.dictionary.Image.open")
    def test_calls_pytesseract_with_all_bcs_languages(self, mock_open, mock_ocr, _mock_pre, tmp_path):
        img_path = tmp_path / "page.jpg"
        img_path.write_bytes(b"fake")
        # `with Image.open(...) as image` requires context-manager support; MagicMock
        # provides __enter__/__exit__ automatically.
        mock_open.return_value = MagicMock(name="PIL.Image")
        mock_ocr.return_value = "1. river\n2. lake\n"

        text = ocr_page(img_path)

        assert text == "1. river\n2. lake\n"
        mock_open.assert_called_once_with(img_path)
        kwargs = mock_ocr.call_args.kwargs
        # All four language packs requested so Tesseract can confidence-check
        # words across English + the BCS Latin variants.
        assert kwargs.get("lang") == "eng+srp_latn+hrv+bos"
        assert "--oem 1" in kwargs.get("config", "")


class TestParsePage:
    def _gemini_returns(self, payload):
        resp = MagicMock()
        resp.text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return resp

    def test_returns_structured_entries(self, mock_cfg):
        payload = {
            "section": "Geografija II",
            "entries": [
                {"n": 1, "english": "river mouth a delta", "serbian_raw": "ušće reke, delta"},
                {"n": 2, "english": "lake", "serbian_raw": "jezero"},
            ],
        }
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = self._gemini_returns(payload)
            page = parse_page(mock_cfg, "raw OCR text")

        assert page.section == "Geografija II"
        assert len(page.entries) == 2
        assert page.entries[0] == DictionaryEntry(n=1, english="river mouth a delta", serbian_raw="ušće reke, delta")
        assert page.entries[1] == DictionaryEntry(n=2, english="lake", serbian_raw="jezero")

    def test_strips_markdown_fences(self, mock_cfg):
        payload = {"section": "X", "entries": [{"n": 1, "english": "a", "serbian_raw": "b"}]}
        wrapped = "```json\n" + json.dumps(payload) + "\n```"
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = self._gemini_returns(wrapped)
            page = parse_page(mock_cfg, "any")
        assert page.entries[0].english == "a"

    def test_skips_malformed_entries(self, mock_cfg):
        payload = {
            "section": "X",
            "entries": [
                {"n": 1, "english": "ok", "serbian_raw": "fine"},
                {"english": "missing-n", "serbian_raw": "x"},          # no n
                {"n": 3, "english": "", "serbian_raw": "blank-en"},    # empty english
                {"n": 4, "english": "good", "serbian_raw": "good"},
            ],
        }
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = self._gemini_returns(payload)
            page = parse_page(mock_cfg, "any")
        assert [e.n for e in page.entries] == [1, 4]


class TestCanonicalizeIjekavian:
    def _gemini_returns(self, payload):
        resp = MagicMock()
        resp.text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return resp

    def test_batch_preserves_order(self, mock_cfg):
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = self._gemini_returns(
                ["rijeka", "stijena", "vijenac"]
            )
            result = canonicalize_ijekavian(mock_cfg, ["reka", "stena", "venac"])
        assert result == ["rijeka", "stijena", "vijenac"]

    def test_empty_input_returns_empty(self, mock_cfg):
        # Should not call Gemini at all.
        with patch("bcs_anki.gemini._get_client") as mock_client:
            assert canonicalize_ijekavian(mock_cfg, []) == []
            mock_client.assert_not_called()

    def test_length_mismatch_raises(self, mock_cfg):
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = self._gemini_returns(["a"])
            with pytest.raises(Exception):
                canonicalize_ijekavian(mock_cfg, ["a", "b"])


def _make_ctx(mock_cfg, tmp_path: Path, *, completed=None, failed=None) -> DictionaryRunContext:
    out_csv = tmp_path / "out.csv"
    failed_csv = tmp_path / "failed.tsv"
    progress_file = tmp_path / ".progress.json"
    failed_csv.write_text("word\treason\n", encoding="utf-8")
    ensure_header(out_csv)

    state = ProgressState(
        input_file=str(tmp_path / "page.jpg"),
        total_words=10,
        completed_words=list(completed or []),
        failed_words=list(failed or []),
        last_updated="",
    )
    mock_cfg.temp_image_folder = tmp_path / "temp"
    mock_cfg.output_folder = tmp_path / "output"
    mock_cfg.output_folder.mkdir(exist_ok=True)
    return DictionaryRunContext(
        cfg=mock_cfg,
        state=state,
        out_csv=out_csv,
        progress_file=progress_file,
        failed_csv=failed_csv,
        section_slug="geografija_ii",
        failed_lock=threading.Lock(),
    )


class TestProcessDictionaryEntry:
    @patch("bcs_anki.dictionary.fetch_stock_image")
    def test_unsplash_path_writes_csv_row(self, mock_stock, mock_cfg, tmp_path):
        ctx = _make_ctx(mock_cfg, tmp_path)
        entry = DictionaryEntry(n=3, english="lake", serbian_raw="jezero")

        def _fake_fetch(cfg, query, dest, count=1):
            dest.write_bytes(b"PNG")
            return [dest]

        mock_stock.side_effect = _fake_fetch

        ok = process_dictionary_entry(entry, "jezero", ctx)
        assert ok is True

        csv_text = ctx.out_csv.read_text(encoding="utf-8")
        assert "Basic (and reversed card)" in csv_text
        assert "<img src=" in csv_text
        assert "jezero" in csv_text
        assert "geografija_ii" in csv_text  # tag from section slug
        assert "geografija_ii:3" in ctx.state.completed_words

    @patch("bcs_anki.dictionary.generate_ai_image")
    @patch("bcs_anki.dictionary.fetch_stock_image")
    def test_ai_fallback_when_stock_misses(self, mock_stock, mock_ai, mock_cfg, tmp_path):
        ctx = _make_ctx(mock_cfg, tmp_path)
        entry = DictionaryEntry(n=4, english="cliff face", serbian_raw="stijena")

        mock_stock.side_effect = NoStockResultsError("none")

        def _fake_ai(cfg, prompt, dest):
            dest.write_bytes(b"AI")

        mock_ai.side_effect = _fake_ai

        ok = process_dictionary_entry(entry, "stijena", ctx)
        assert ok is True
        # Verify AI was called with literal "draw {english}" prompt
        called_prompt = mock_ai.call_args.args[1]
        assert called_prompt == "draw cliff face"

    @patch("bcs_anki.dictionary.generate_ai_image")
    @patch("bcs_anki.dictionary.fetch_stock_image")
    def test_ai_rejection_marks_failed(self, mock_stock, mock_ai, mock_cfg, tmp_path):
        ctx = _make_ctx(mock_cfg, tmp_path)
        entry = DictionaryEntry(n=5, english="something", serbian_raw="nešto")

        mock_stock.side_effect = NoStockResultsError("none")
        mock_ai.side_effect = ImageRejectedError("safety filter")

        ok = process_dictionary_entry(entry, "nešto", ctx)
        assert ok is False
        assert "geografija_ii:5" in ctx.state.failed_words
        # No CSV row written (only header lines remain)
        csv_text = ctx.out_csv.read_text(encoding="utf-8")
        assert "Basic (and reversed card)" not in csv_text

    @patch("bcs_anki.dictionary.fetch_stock_image")
    def test_already_completed_is_skipped(self, mock_stock, mock_cfg, tmp_path):
        ctx = _make_ctx(mock_cfg, tmp_path, completed=["geografija_ii:7"])
        entry = DictionaryEntry(n=7, english="lake", serbian_raw="jezero")

        ok = process_dictionary_entry(entry, "jezero", ctx)
        assert ok is True
        mock_stock.assert_not_called()
        # CSV still has only header
        csv_text = ctx.out_csv.read_text(encoding="utf-8")
        assert "Basic (and reversed card)" not in csv_text


class TestSlugify:
    def test_basic(self):
        assert _slugify_tag("Geografija II") == "geografija_ii"

    def test_strips_punctuation(self):
        assert _slugify_tag("  Page #1038!  ") == "page_1038"

    def test_empty(self):
        assert _slugify_tag("") == ""


class TestEditOcrHook:
    """The --edit-ocr step writes OCR to a file, opens $EDITOR, then re-reads it."""

    def test_writes_ocr_file_and_uses_edited_text(self, mock_cfg, tmp_path, monkeypatch):
        from bcs_anki.pipeline import _edit_ocr_text

        mock_cfg.output_folder = tmp_path / "out"
        image_path = tmp_path / "page.jpg"
        raw = "1. river\n2. lake\n"

        edited_payload = "1. river\n2. lake (better!)\n"

        def _fake_edit(filename: str) -> None:
            # User "edits" the file in place.
            Path(filename).write_text(edited_payload, encoding="utf-8")

        monkeypatch.setattr("bcs_anki.pipeline.click.edit", _fake_edit)

        result = _edit_ocr_text(mock_cfg, image_path, raw)

        assert result == edited_payload
        ocr_file = mock_cfg.output_folder / "page.ocr.txt"
        assert ocr_file.exists()
        assert ocr_file.read_text(encoding="utf-8") == edited_payload

    def test_falls_back_when_editor_unavailable(self, mock_cfg, tmp_path, monkeypatch):
        import click as click_mod
        from bcs_anki.pipeline import _edit_ocr_text

        mock_cfg.output_folder = tmp_path / "out"
        image_path = tmp_path / "page.jpg"
        raw = "raw OCR text"

        def _fake_edit(filename: str) -> None:
            raise click_mod.UsageError("no editor available")

        monkeypatch.setattr("bcs_anki.pipeline.click.edit", _fake_edit)

        result = _edit_ocr_text(mock_cfg, image_path, raw)
        assert result == raw


class TestCliDictionaryFlag:
    """Smoke test that --dictionary-image dispatches to run_dictionary_pipeline."""

    def test_dispatches_to_dictionary_pipeline(self, mock_cfg, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from bcs_anki import cli

        img = tmp_path / "page.jpg"
        img.write_bytes(b"fake-image-bytes")

        called = {}

        def _fake_run(cfg, image_paths, *, output_csv=None, resume=False, fresh=False, append=False, edit_ocr=True):
            called["image_paths"] = list(image_paths)
            called["output_csv"] = output_csv
            called["edit_ocr"] = edit_ocr
            return (0, 0)

        monkeypatch.setattr(cli, "run_dictionary_pipeline", _fake_run)
        monkeypatch.setattr(cli, "_load_app_config", lambda *_a, **_kw: mock_cfg)
        monkeypatch.setattr(cli, "check_apis", lambda _cfg: None)

        runner = CliRunner()
        result = runner.invoke(cli.main, ["generate", "--dictionary-image", str(img), "--no-edit-ocr"])

        assert result.exit_code == 0, result.output
        assert called["image_paths"] == [img]
        assert called["edit_ocr"] is False
