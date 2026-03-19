"""Tier 3: Image pipeline tests (mocked APIs)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

        fetch_stock_image(mock_cfg, "ceasefire", dest)
        assert dest.read_bytes() == b"PNG_DATA"
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

        fetch_stock_image(mock_cfg, "apple", dest)
        assert dest.read_bytes() == b"PNG_DATA"
        assert "pexels" in mock_req.call_args_list[0].args[1]

    @patch("bcs_anki.images.request_with_retries")
    def test_pixabay(self, mock_req, mock_cfg, tmp_path):
        mock_cfg.stock_image_api = "pixabay"
        dest = tmp_path / "img.png"

        search_resp, img_resp = self._mock_response(
            {"hits": [{"webformatURL": "https://pixabay.com/photo.jpg"}]}
        )
        mock_req.side_effect = [search_resp, img_resp]

        fetch_stock_image(mock_cfg, "cat", dest)
        assert dest.read_bytes() == b"PNG_DATA"
        assert "pixabay" in mock_req.call_args_list[0].args[1]

    @patch("bcs_anki.images.request_with_retries")
    def test_no_results_raises(self, mock_req, mock_cfg, tmp_path):
        mock_cfg.stock_image_api = "unsplash"
        search_resp = MagicMock()
        search_resp.json.return_value = {"results": []}
        mock_req.return_value = search_resp

        with pytest.raises(RuntimeError, match="No Unsplash results"):
            fetch_stock_image(mock_cfg, "xyz", tmp_path / "img.png")


class TestGenerateAiImage:
    @patch("bcs_anki.images.request_with_retries")
    @patch("bcs_anki.images.OpenAI")
    def test_generates_and_downloads(self, mock_openai_cls, mock_req, mock_cfg, mock_openai_image, tmp_path):
        dest = tmp_path / "ai_img.png"
        mock_openai_cls.return_value.images.generate.return_value = mock_openai_image()

        img_resp = MagicMock()
        img_resp.content = b"AI_IMAGE_DATA"
        mock_req.return_value = img_resp

        generate_ai_image(mock_cfg, "a peaceful scene", dest)
        assert dest.read_bytes() == b"AI_IMAGE_DATA"
        mock_openai_cls.return_value.images.generate.assert_called_once()
