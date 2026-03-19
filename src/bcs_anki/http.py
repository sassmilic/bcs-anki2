from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)


def request_with_retries(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    json_body: dict | None = None,
    max_retries: int = 3,
    delay_seconds: float = 2.0,
) -> requests.Response:
    """HTTP request with exponential-backoff retries on transient (5xx) errors."""
    delay = delay_seconds
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=60,
            )
            if resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}")
            return resp
        except Exception as exc:  # noqa: BLE001
            logger.error("HTTP request failed (attempt %s): %s", attempt, exc)
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
