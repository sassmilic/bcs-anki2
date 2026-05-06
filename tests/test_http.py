"""Tier 3: HTTP retry logic tests (mocked requests)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from bcs_anki.errors import HttpTransientError
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

        with pytest.raises(HttpTransientError, match="HTTP 500"):
            request_with_retries("GET", "https://example.com", max_retries=2, delay_seconds=0.01)
        assert mock_request.call_count == 2

    @patch("bcs_anki.http.requests.request")
    def test_non_retryable_4xx_raises_immediately(self, mock_request):
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = HTTPError("404 Client Error")
        mock_request.return_value = resp

        with pytest.raises(HTTPError, match="404 Client Error"):
            request_with_retries("GET", "https://example.com", delay_seconds=0)
        assert mock_request.call_count == 1

    @patch("bcs_anki.http.time.sleep")
    @patch("bcs_anki.http.requests.request")
    def test_retries_on_429_and_error_includes_excerpt(self, mock_request, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 429
        fail_resp.text = "rate limit exceeded"

        success_resp = MagicMock()
        success_resp.status_code = 200

        mock_request.side_effect = [fail_resp, success_resp]

        result = request_with_retries("GET", "https://example.com", max_retries=3, delay_seconds=0.01)
        assert result is success_resp
        assert mock_request.call_count == 2

    @patch("bcs_anki.http.time.sleep")
    @patch("bcs_anki.http.requests.request")
    def test_5xx_error_message_contains_status_and_excerpt(self, mock_request, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 502
        fail_resp.text = "upstream timed out"
        mock_request.return_value = fail_resp

        with pytest.raises(HttpTransientError, match="HTTP 502: upstream timed out"):
            request_with_retries("GET", "https://example.com", max_retries=1, delay_seconds=0)
