"""Video download with retries and exponential backoff."""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger("agent.download")


class DownloadError(Exception):
    pass


def download_video(url: str, dest_path: str, timeout: int = 60, retries: int = 3) -> None:
    """Stream-downloads url to dest_path. Raises DownloadError if every
    attempt fails.
    """
    last_error: str | None = None
    backoff = 1.5

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("download attempt %d/%d failed: %s", attempt, retries, last_error)
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2

    raise DownloadError(f"failed to download {url} after {retries} attempts: {last_error}")
