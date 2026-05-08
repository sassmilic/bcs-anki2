"""Tier 3: Image pipeline tests (mocked APIs)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import BadRequestError

from bcs_anki.errors import (
    ImageRejectedError,
    NoStockResultsError,
    UnsupportedStockProviderError,
)
from bcs_anki.images import fetch_stock_image, generate_ai_image


class TestFetchStockImage:
    def _mock_response(self, json_data: dict, image_bytes: bytes = b"PNG_DATA") -> tuple:
        """Return (search_response, image_response) mocks."""
        search_resp = MagicMock()
        search_resp.json.return_value = json_data
        search_resp.status_code = 200

        img_resp = MagicMock()
        img_resp.content = image_bytes
        img_resp.status_code = 200

        return search_resp, img_resp

    @patch("bcs_anki.images.request_with_retries")
    def test_unsplash(self, mock_req, mock_cfg, tmp_path):
        mock_cfg.stock_image_api = "unsplash"
        dest = tmp_path / "img.png"

        search_resp, img_resp = self._mock_response(
            {"results": [{"urls": {"regular": "https://unsplash.com/photo.jpg"}}]}
        )
        mock_req.side_effect = [search_resp, img_resp]

        paths = fetch_stock_image(mock_cfg, "ceasefire", dest)
        assert dest.read_bytes() == b"PNG_DATA"
        assert len(paths) == 1
        # First call should be to unsplash API
        assert "unsplash" in mock_req.call_args_list[0].args[1]

    @patch("bcs_anki.images.request_with_retries")
    def test_pexels(self, mock_req, mock_cfg, tmp_path):
        mock_cfg.stock_image_api = "pexels"
        dest = tmp_path / "img.png"

        search_resp, img_resp = self._mock_response(
            {"photos": [{"src": {"medium": "https://pexels.com/photo.jpg"}}]}
        )
        mock_req.side_effect = [search_resp, img_resp]

        paths = fetch_stock_image(mock_cfg, "apple", dest)
        assert dest.read_bytes() == b"PNG_DATA"
        assert len(paths) == 1
        assert "pexels" in mock_req.call_args_list[0].args[1]

    @patch("bcs_anki.images.request_with_retries")
    def test_pixabay(self, mock_req, mock_cfg, tmp_path):
        mock_cfg.stock_image_api = "pixabay"
        dest = tmp_path / "img.png"

        search_resp, img_resp = self._mock_response(
            {"hits": [{"webformatURL": "https://pixabay.com/photo.jpg"}]}
        )
        mock_req.side_effect = [search_resp, img_resp]

        paths = fetch_stock_image(mock_cfg, "cat", dest)
        assert dest.read_bytes() == b"PNG_DATA"
        assert len(paths) == 1
        assert "pixabay" in mock_req.call_args_list[0].args[1]

    @patch("bcs_anki.images.request_with_retries")
    def test_no_results_raises(self, mock_req, mock_cfg, tmp_path):
        mock_cfg.stock_image_api = "unsplash"
        search_resp = MagicMock()
        search_resp.json.return_value = {"results": []}
        mock_req.return_value = search_resp

        with pytest.raises(NoStockResultsError, match="No Unsplash results"):
            fetch_stock_image(mock_cfg, "xyz", tmp_path / "img.png")

    def test_unknown_provider_raises_typed_error(self, mock_cfg, tmp_path):
        mock_cfg.stock_image_api = "bogus"
        with pytest.raises(UnsupportedStockProviderError):
            fetch_stock_image(mock_cfg, "x", tmp_path / "img.png")


class TestGenerateAiImage:
    @patch("bcs_anki.images.OpenAI")
    def test_generates_and_downloads(self, mock_openai_cls, mock_cfg, mock_openai_image, tmp_path):
        dest = tmp_path / "ai_img.png"
        mock_openai_cls.return_value.images.generate.return_value = mock_openai_image(b"AI_IMAGE_DATA")

        generate_ai_image(mock_cfg, "a peaceful scene", dest)
        assert dest.read_bytes() == b"AI_IMAGE_DATA"
        mock_openai_cls.return_value.images.generate.assert_called_once()

    @patch("bcs_anki.images.OpenAI")
    def test_safety_rejection_raises_image_rejected(self, mock_openai_cls, mock_cfg, tmp_path):
        # OpenAI's BadRequestError requires a response/body shape; build a minimal one.
        resp = MagicMock()
        resp.request = MagicMock()
        mock_openai_cls.return_value.images.generate.side_effect = BadRequestError(
            message="content rejected", response=resp, body={"error": {"code": "safety"}}
        )
        with pytest.raises(ImageRejectedError):
            generate_ai_image(mock_cfg, "anything", tmp_path / "img.png")

    @patch("bcs_anki.images.OpenAI")
    def test_records_legacy_per_image_cost_when_no_usage_field(
        self, mock_openai_cls, mock_cfg, mock_openai_image, tmp_path,
    ):
        """Legacy models (dall-e-3) return no `usage` → tracked as per-image counts."""
        from bcs_anki.costs import COST_TRACKER

        COST_TRACKER.images.clear()
        COST_TRACKER.image_tokens.clear()
        COST_TRACKER.image_token_counts.clear()

        mock_cfg.image_generation_model = "dall-e-3"
        mock_cfg.image_size = "1024x1024"
        mock_cfg.image_quality = "standard"
        mock_openai_cls.return_value.images.generate.return_value = mock_openai_image(b"X")
        generate_ai_image(mock_cfg, "p", tmp_path / "a.png")
        generate_ai_image(mock_cfg, "p", tmp_path / "b.png")

        assert COST_TRACKER.images[("dall-e-3", "1024x1024", "standard")] == 2
        assert COST_TRACKER.image_tokens == {}

    @patch("bcs_anki.images.OpenAI")
    def test_records_token_cost_when_usage_present(
        self, mock_openai_cls, mock_cfg, mock_openai_image_with_usage, tmp_path,
    ):
        """Token-priced models (gpt-image-2) return `usage` → tracked as token totals."""
        from bcs_anki.costs import COST_TRACKER

        COST_TRACKER.images.clear()
        COST_TRACKER.image_tokens.clear()
        COST_TRACKER.image_token_counts.clear()

        mock_cfg.image_generation_model = "gpt-image-2"
        mock_openai_cls.return_value.images.generate.return_value = mock_openai_image_with_usage(
            input_tokens=100, output_tokens=4096,
        )
        generate_ai_image(mock_cfg, "p", tmp_path / "a.png")
        generate_ai_image(mock_cfg, "p", tmp_path / "b.png")

        assert COST_TRACKER.image_token_counts["gpt-image-2"] == 2
        usage = COST_TRACKER.image_tokens["gpt-image-2"]
        assert usage.text_input_tokens == 200
        assert usage.image_output_tokens == 8192
        # No legacy per-image rows for this model.
        assert COST_TRACKER.images == {}

    @patch("bcs_anki.images.OpenAI")
    def test_does_not_record_cost_on_rejection(self, mock_openai_cls, mock_cfg, tmp_path):
        from bcs_anki.costs import COST_TRACKER

        COST_TRACKER.images.clear()
        COST_TRACKER.image_tokens.clear()
        COST_TRACKER.image_token_counts.clear()

        resp = MagicMock(); resp.request = MagicMock()
        mock_openai_cls.return_value.images.generate.side_effect = BadRequestError(
            message="rejected", response=resp, body={"error": {"code": "safety"}}
        )
        with pytest.raises(ImageRejectedError):
            generate_ai_image(mock_cfg, "x", tmp_path / "img.png")
        assert COST_TRACKER.images == {}
        assert COST_TRACKER.image_tokens == {}
