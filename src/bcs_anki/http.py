from __future__ import annotations

import logging
import time

import requests
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, Timeout

from .errors import HttpTransientError

logger = logging.getLogger(__name__)


def _status_error(resp: Response) -> RuntimeError:
    excerpt = (resp.text or "")[:200].replace("\n", " ").strip()
    message = f"HTTP {resp.status_code}"
    if excerpt:
        message = f"{message}: {excerpt}"
    return RuntimeError(message)


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
    """HTTP request with exponential-backoff retries on transient failures."""
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
            if resp.status_code == 429 or resp.status_code >= 500:
                raise HttpTransientError(resp.status_code)

            return resp
        except (Timeout, ConnectionError, RuntimeError) as exc:
            logger.error("HTTP request failed (attempt %s): %s", attempt, exc)
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
        except HTTPError:
            raise
