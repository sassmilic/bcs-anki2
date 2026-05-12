"""Tests for the Gemini-based dict-CSV refinement module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bcs_anki.dictionary_csv import read_dict_csv
from bcs_anki.dict_refine import refine_csv, refine_rows


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def _write_csv(path: Path, subject: str, rows: list[tuple[str, str]]) -> None:
    lines = [f"# Subject: {subject}", "english,serbian"]
    lines.extend(f"{eng},{sr}" for eng, sr in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestReadDictCsv:
    def test_round_trip_simple(self, tmp_path):
        csv = tmp_path / "geo.csv"
        _write_csv(csv, "Geografija I", [("river", "reka"), ("rock", "stena")])
        subject, rows = read_dict_csv(csv)
        assert subject == "Geografija I"
        assert rows == [("river", "reka"), ("rock", "stena")]

    def test_quoted_field_with_comma(self, tmp_path):
        csv = tmp_path / "x.csv"
        csv.write_text(
            '# Subject: Geografija II\nenglish,serbian\nrock,"stena, hrid"\n',
            encoding="utf-8",
        )
        subject, rows = read_dict_csv(csv)
        assert rows == [("rock", "stena, hrid")]

    def test_missing_subject_header_raises(self, tmp_path):
        csv = tmp_path / "x.csv"
        csv.write_text("english,serbian\nfoo,bar\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing `# Subject:` header"):
            read_dict_csv(csv)

    def test_wrong_csv_header_raises(self, tmp_path):
        csv = tmp_path / "x.csv"
        csv.write_text("# Subject: X\neng,sr\nfoo,bar\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected CSV header"):
            read_dict_csv(csv)


class TestRefineRows:
    def test_returns_corrected_pairs(self, mock_cfg):
        rows = [("river", "reka"), ("rock", "stena")]
        gemini_payload = json.dumps([
            {"eng": "river", "sr": "rijeka"},
            {"eng": "cliff", "sr": "stijena"},
        ])
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(gemini_payload)
            refined = refine_rows(mock_cfg, "Geografija I", rows)

        assert refined == [("river", "rijeka"), ("cliff", "stijena")]

    def test_passes_subject_into_prompt(self, mock_cfg):
        rows = [("river", "rijeka")]
        gemini_payload = json.dumps([{"eng": "river", "sr": "rijeka"}])
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(gemini_payload)
            refine_rows(mock_cfg, "Astronomija", rows)

        sent_user_prompt = mock_client.return_value.models.generate_content.call_args.kwargs["contents"]
        assert 'Astronomija' in sent_user_prompt
        # Input rows are JSON-serialized into the prompt body.
        assert '"sr": "rijeka"' in sent_user_prompt

    def test_length_mismatch_raises(self, mock_cfg):
        rows = [("a", "b"), ("c", "d"), ("e", "f")]
        gemini_payload = json.dumps([
            {"eng": "a", "sr": "b"},
            {"eng": "c", "sr": "d"},  # only 2, expected 3
        ])
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(gemini_payload)
            with pytest.raises(ValueError, match="returned 2 rows but 3 were sent"):
                refine_rows(mock_cfg, "X", rows)

    def test_malformed_row_raises(self, mock_cfg):
        rows = [("river", "reka")]
        gemini_payload = json.dumps([{"eng": "river"}])  # missing sr
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(gemini_payload)
            with pytest.raises(ValueError, match="malformed"):
                refine_rows(mock_cfg, "X", rows)

    def test_empty_rows_skips_api_call(self, mock_cfg):
        with patch("bcs_anki.gemini._get_client") as mock_client:
            result = refine_rows(mock_cfg, "X", [])
        assert result == []
        assert mock_client.call_count == 0


class TestRefineCsv:
    def test_round_trip(self, mock_cfg, tmp_path):
        src = tmp_path / "geo.csv"
        _write_csv(src, "Geografija I", [("river", "reka"), ("rock", "stena")])
        dst = tmp_path / "refined" / "geo.csv"

        gemini_payload = json.dumps([
            {"eng": "river", "sr": "rijeka"},
            {"eng": "cliff", "sr": "stijena"},
        ])
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(gemini_payload)
            n = refine_csv(mock_cfg, src, dst)

        assert n == 2
        assert dst.exists()
        # Original is untouched.
        assert read_dict_csv(src)[1] == [("river", "reka"), ("rock", "stena")]
        # Refined matches Gemini's output.
        subject, rows = read_dict_csv(dst)
        assert subject == "Geografija I"
        assert rows == [("river", "rijeka"), ("cliff", "stijena")]
