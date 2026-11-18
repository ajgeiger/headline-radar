"""HTTP client with retries, backoff, and an optional minimum interval."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from utils.HttpError import HttpError

logger = logging.getLogger("headline_radar")


class HttpClient:
    """Small wrapper around requests.Session.

    ``min_interval_seconds`` enforces a pause between calls (needed for the
    GDELT DOC API, which asks for at least five seconds between requests).
    On 429 or 5xx responses the client waits and retries.
    """

    def __init__(
        self,
        user_agent: str,
        timeout_seconds: int = 30,
        max_retries: int = 5,
        min_interval_seconds: float = 0.0,
        retry_after_seconds: float = 30.0,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "*/*",
            }
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.min_interval_seconds = min_interval_seconds
        self.retry_after_seconds = retry_after_seconds
        self._last_request_at: float | None = None

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        """GET a URL and return the response, retrying transient failures."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_interval()
            try:
                response = self.session.get(
                    url, params=params, timeout=self.timeout_seconds
                )
                self._last_request_at = time.monotonic()
                if response.status_code in {429, 500, 502, 503, 504}:
                    wait = self._retry_wait(response, attempt)
                    logger.warning(
                        "HTTP %s from %s (attempt %s/%s); waiting %.1fs",
                        response.status_code,
                        url,
                        attempt,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    last_error = HttpError(
                        f"HTTP {response.status_code} for {url}: {response.text[:200]}",
                        status_code=response.status_code,
                    )
                    continue
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise HttpError(
                        f"HTTP {response.status_code} for {url}: {response.text[:200]}",
                        status_code=response.status_code,
                    ) from exc
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                wait = min(2**attempt, 60)
                logger.warning(
                    "Request error for %s (attempt %s/%s): %s; waiting %.1fs",
                    url,
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_error if last_error is not None else HttpError(f"GET failed for {url}")

    def _wait_for_interval(self) -> None:
        if self.min_interval_seconds <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _retry_wait(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        if response.status_code == 429:
            return self.retry_after_seconds
        return min(2**attempt, 60)
