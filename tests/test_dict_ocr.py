"""Tests for the Gemini-based dictionary-page OCR module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from bcs_anki.dict_ocr import extract_dict_pages
from bcs_anki.dictionary_csv import (
    DictEntry,
    DictPage,
    subject_slug,
    write_dict_csv,
)


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def _server_error(code: int = 503):
    return genai_errors.ServerError(
        code,
        {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}},
        None,
    )


def _make_image(tmp_path: Path, name: str = "page.jpg") -> Path:
    p = tmp_path / name
    # The bytes don't need to be a real image — the genai client is mocked.
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return p


_SAMPLE_JSON = json.dumps(
    {
        "subject": "Astronomija",
        "entries": [
            {"n": "1", "eng": "star", "sr": "zvijezda"},
            {"n": "2", "eng": "planet", "sr": "planeta"},
            {"n": "3", "eng": "moon", "sr": "mjesec"},
        ],
    }
)


class TestExtractDictPages:
    def test_single_image_returns_parsed_page(self, mock_cfg, tmp_path):
        img = _make_image(tmp_path)
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(_SAMPLE_JSON)
            page = extract_dict_pages(mock_cfg, [img])

        assert page.subject == "Astronomija"
        assert len(page.entries) == 3
        assert page.entries[0] == DictEntry(number="1", english="star", serbian="zvijezda")
        assert page.entries[2] == DictEntry(number="3", english="moon", serbian="mjesec")

    def test_multiple_images_passed_as_separate_parts(self, mock_cfg, tmp_path):
        img1 = _make_image(tmp_path, "p1.jpg")
        img2 = _make_image(tmp_path, "p2.png")
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(_SAMPLE_JSON)
            extract_dict_pages(mock_cfg, [img1, img2])

        call_kwargs = mock_client.return_value.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        # Two image parts + one text prompt = 3 items.
        assert len(contents) == 3
        # Last item is the user prompt text.
        assert isinstance(contents[-1], str)
        # First two are Part objects with image MIME types.
        assert contents[0].inline_data.mime_type == "image/jpeg"
        assert contents[1].inline_data.mime_type == "image/png"

    def test_heic_image_uses_heic_mime(self, mock_cfg, tmp_path):
        img = _make_image(tmp_path, "iphone.heic")
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(_SAMPLE_JSON)
            extract_dict_pages(mock_cfg, [img])

        contents = mock_client.return_value.models.generate_content.call_args.kwargs["contents"]
        assert contents[0].inline_data.mime_type == "image/heic"

    def test_retries_on_transient_server_error(self, mock_cfg, tmp_path):
        img = _make_image(tmp_path)
        with patch("bcs_anki.gemini.time.sleep"), \
             patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = [
                _server_error(503),
                _mock_response(_SAMPLE_JSON),
            ]
            page = extract_dict_pages(mock_cfg, [img])
        assert page.subject == "Astronomija"
        assert mock_client.return_value.models.generate_content.call_count == 2

    def test_malformed_json_raises_value_error(self, mock_cfg, tmp_path):
        img = _make_image(tmp_path)
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response("not json at all")
            with pytest.raises(ValueError, match="non-JSON"):
                extract_dict_pages(mock_cfg, [img])

    def test_missing_required_keys_raises(self, mock_cfg, tmp_path):
        img = _make_image(tmp_path)
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(
                json.dumps({"subject": "X"})
            )
            with pytest.raises(ValueError, match="missing required keys"):
                extract_dict_pages(mock_cfg, [img])

    def test_includes_range_category_entries(self, mock_cfg, tmp_path):
        """Range-spanning category headers (n='1-5') with paired eng+sr come through."""
        img = _make_image(tmp_path)
        payload = json.dumps({
            "subject": "Geografija I",
            "entries": [
                {"n": "1-5", "eng": "layered structure of the earth", "sr": "slojevita gradja zemlje"},
                {"n": "1", "eng": "earth's crust", "sr": "zemljina kora"},
                {"n": "2", "eng": "hydrosphere", "sr": "hidrosfera"},
                {"n": "6-12", "eng": "hypsographic curve", "sr": "hipsografska krivulja"},
                {"n": "6", "eng": "peak", "sr": "vrh"},
            ],
        })
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(payload)
            page = extract_dict_pages(mock_cfg, [img])

        assert [e.number for e in page.entries] == ["1-5", "1", "2", "6-12", "6"]
        assert page.entries[0].english == "layered structure of the earth"
        assert page.entries[3].serbian == "hipsografska krivulja"

    def test_skips_unpaired_entries(self, mock_cfg, tmp_path):
        """Entries missing eng or sr (whether single or range) are dropped, not raised."""
        img = _make_image(tmp_path)
        payload = json.dumps({
            "subject": "Astronomija",
            "entries": [
                {"n": "1", "eng": "star", "sr": "zvijezda"},
                {"n": "22-28", "eng": "rotary motions", "sr": None},   # no sr pair
                {"n": "2", "eng": "planet", "sr": "planeta"},
                {"n": "3", "eng": "moon", "sr": ""},                    # empty sr
                {"n": None, "eng": "x", "sr": "y"},                     # missing n
            ],
        })
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(payload)
            page = extract_dict_pages(mock_cfg, [img])

        assert [e.number for e in page.entries] == ["1", "2"]

    def test_empty_image_list_raises(self, mock_cfg):
        with pytest.raises(ValueError, match="at least one image"):
            extract_dict_pages(mock_cfg, [])

    def test_unsupported_image_extension_raises(self, mock_cfg, tmp_path):
        bad = tmp_path / "page.bmp"
        bad.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported image type"):
            extract_dict_pages(mock_cfg, [bad])


class TestWriteDictCsv:
    def test_writes_subject_header_and_rows(self, tmp_path):
        page = DictPage(
            subject="Astronomija",
            entries=[
                DictEntry(number=1, english="star", serbian="zvijezda"),
                DictEntry(number=2, english="planet", serbian="planeta"),
            ],
        )
        out = tmp_path / "out.csv"
        write_dict_csv(page, out)

        text = out.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "# Subject: Astronomija"
        assert text.splitlines()[1] == "english,serbian"
        assert "star,zvijezda" in text
        assert "planet,planeta" in text

    def test_quotes_fields_with_commas(self, tmp_path):
        page = DictPage(
            subject="Geografija",
            entries=[DictEntry(number=1, english="rock", serbian="stijena, hrid")],
        )
        out = tmp_path / "out.csv"
        write_dict_csv(page, out)

        text = out.read_text(encoding="utf-8")
        # csv module wraps the comma-bearing field in double quotes.
        assert '"stijena, hrid"' in text

    def test_creates_parent_directory(self, tmp_path):
        page = DictPage(subject="X", entries=[])
        out = tmp_path / "nested" / "deeper" / "out.csv"
        write_dict_csv(page, out)
        assert out.exists()


class TestSubjectSlug:
    def test_simple_word(self):
        assert subject_slug("Astronomija") == "astronomija"

    def test_multi_word_with_roman_numeral(self):
        assert subject_slug("Geografija II") == "geografija-ii"

    def test_preserves_unicode_letters(self):
        # š/ž/ć are word characters under re.UNICODE — keep them.
        assert subject_slug("U šumi") == "u-šumi"

    def test_collapses_runs_of_separators(self):
        assert subject_slug("Foo --- Bar / Baz") == "foo-bar-baz"

    def test_strips_edge_separators(self):
        assert subject_slug("  Astronomija!  ") == "astronomija"

    def test_empty_subject_falls_back(self):
        assert subject_slug("") == "untitled"
        assert subject_slug("---") == "untitled"
