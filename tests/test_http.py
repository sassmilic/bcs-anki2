"""Tier 3: HTTP retry logic tests (mocked requests)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bcs_anki.http import request_with_retries


class TestRequestWithRetries:
    @patch("bcs_anki.http.requests.request")
    def test_success_on_first_try(self, mock_request):
        resp = MagicMock()
        resp.status_code = 200
        mock_request.return_value = resp

        result = request_with_retries("GET", "https://example.com", delay_seconds=0)
        assert result is resp
        assert mock_request.call_count == 1

    @patch("bcs_anki.http.time.sleep")
    @patch("bcs_anki.http.requests.request")
    def test_retries_on_500(self, mock_request, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        success_resp = MagicMock()
        success_resp.status_code = 200

        mock_request.side_effect = [fail_resp, success_resp]

        result = request_with_retries("GET", "https://example.com", max_retries=3, delay_seconds=0.01)
        assert result is success_resp
        assert mock_request.call_count == 2

    @patch("bcs_anki.http.time.sleep")
    @patch("bcs_anki.http.requests.request")
    def test_exhausts_retries(self, mock_request, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        mock_request.return_value = fail_resp

        with pytest.raises(RuntimeError, match="HTTP 500"):
            request_with_retries("GET", "https://example.com", max_retries=2, delay_seconds=0.01)
        assert mock_request.call_count == 2

    @patch("bcs_anki.http.requests.request")
    def test_non_retryable_passes_through(self, mock_request):
        """Non-5xx responses (like 404) are returned immediately, not retried."""
        resp = MagicMock()
        resp.status_code = 404
        mock_request.return_value = resp

        result = request_with_retries("GET", "https://example.com", delay_seconds=0)
        assert result.status_code == 404
        assert mock_request.call_count == 1
