"""Thin OpenAI-compatible client for Fireworks AI with multi-key round-robin
and failover. Never logs key material.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time

import requests

logger = logging.getLogger("agent.fireworks")


class FireworksError(Exception):
    """Raised when a chat completion could not be obtained from any key."""


class _KeyRotator:
    """Thread-safe round-robin over a fixed list of API keys."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("no API keys provided")
        self._keys = list(keys)
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(range(len(self._keys)))

    def ordered_keys(self) -> list[str]:
        """Return all keys starting from the next round-robin position, so
        concurrent callers spread their *first* attempt across keys while
        each call still has every key available as a failover fallback.
        """
        with self._lock:
            start = next(self._cycle)
        return self._keys[start:] + self._keys[:start]


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


class FireworksClient:
    def __init__(self, config):
        self.config = config
        self._rotator = _KeyRotator(config.api_keys)
        self._session = requests.Session()

    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 700,
        temperature: float = 0.4,
        timeout: int | None = None,
    ) -> str:
        """Returns the assistant message content as a string. Raises
        FireworksError if every key/retry combination fails.
        """
        timeout = timeout or self.config.request_timeout
        url = f"{self.config.fireworks_base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        reasoning_effort = getattr(self.config, "reasoning_effort", "")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        keys = self._rotator.ordered_keys()
        last_error: str | None = None

        for key in keys:
            backoff = 1.0
            for attempt in range(self.config.request_retries_per_key):
                try:
                    resp = self._session.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=timeout,
                    )
                except (requests.Timeout, requests.ConnectionError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "fireworks request error key=%s attempt=%d err=%s",
                        _mask(key), attempt, last_error,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        return data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError) as exc:
                        last_error = f"unexpected response shape: {exc}"
                        logger.warning("fireworks malformed success body: %s", last_error)
                        break  # try next key, retrying same shape won't help

                if resp.status_code in (429, 402):
                    last_error = f"HTTP {resp.status_code} on key={_mask(key)}"
                    # Back off and retry the SAME key first (respecting
                    # Retry-After if the server sent one) -- this matters a
                    # lot for single-key deployments, which would otherwise
                    # give up after exactly one rate-limit hit since there's
                    # no next key to fall back to. Only once local retries
                    # on this key are exhausted do we move to the next key.
                    if attempt < self.config.request_retries_per_key - 1:
                        wait = backoff
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = max(wait, float(retry_after))
                            except ValueError:
                                pass
                        logger.warning(
                            "fireworks rate-limited/exhausted key=%s attempt=%d, backing off %.1fs: %s",
                            _mask(key), attempt, wait, last_error,
                        )
                        time.sleep(wait)
                        backoff *= 2
                        continue
                    logger.warning("fireworks key exhausted/limited after retries: %s", last_error)
                    break

                if resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning(
                        "fireworks server error key=%s attempt=%d err=%s",
                        _mask(key), attempt, last_error,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                # Other 4xx: retrying won't help (bad request / auth / etc).
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error("fireworks request failed key=%s err=%s", _mask(key), last_error)
                break

        raise FireworksError(f"all Fireworks API keys exhausted; last error: {last_error}")
