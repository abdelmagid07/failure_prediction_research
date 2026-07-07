"""Sample seed paragraphs from English Wikipedia (50–200 words).

Fetches are batched (one request returns up to ~20 random extracts) and use
exponential backoff on HTTP 429/5xx, so generating hundreds of conversations does
not melt down against Wikipedia's rate limits.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

USER_AGENT = "ValueAxisStage1/0.1 (research; https://github.com/abdelmagid07/latent_failiure_prediction)"

# Wikipedia caps random+extracts batches; 20 is the practical max per request.
_MAX_BATCH = 20


def _get(url: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _trim_to_word_range(text: str, min_words: int = 50, max_words: int = 200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        text = " ".join(words)
    if len(words) < min_words and len(text) > 0:
        return text
    if len(words) < 20:
        return ""
    return text


def fetch_random_paragraphs(count: int, timeout: float = 20.0) -> list[str]:
    """Return up to ``count`` Wikipedia extracts (one API request).

    Uses ``exlimit=max`` so a single request returns extracts for every random
    page, rather than just the first.
    """
    count = max(1, min(count, _MAX_BATCH))
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "exsection": "0",
            "exlimit": "max",
            "generator": "random",
            "grnnamespace": "0",
            "grnlimit": str(count),
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    data = _get(url, timeout=timeout)
    pages = data.get("query", {}).get("pages", {})
    out: list[str] = []
    for page in pages.values():
        paragraph = _trim_to_word_range(page.get("extract", ""))
        if paragraph:
            out.append(paragraph)
    return out


def fetch_random_paragraph(timeout: float = 20.0) -> str:
    """Return one Wikipedia extract trimmed to roughly 50–200 words."""
    batch = fetch_random_paragraphs(1, timeout=timeout)
    return batch[0] if batch else ""


def fetch_paragraphs(
    n: int,
    rng,
    *,
    max_attempts: int = 25,
    on_retry: Callable[[str], None] | None = None,
) -> list[str]:
    """Fetch n distinct non-empty Wikipedia paragraphs, batching requests.

    Backs off exponentially on HTTP errors (429/5xx) and politely spaces
    successful requests so large runs stay under rate limits.
    """
    seen: set[str] = set()
    out: list[str] = []

    for attempt in range(max_attempts):
        if len(out) >= n:
            break
        need = n - len(out)
        try:
            batch = fetch_random_paragraphs(min(_MAX_BATCH, max(need, 5)))
        except urllib.error.HTTPError as exc:
            wait = min(30.0, 2.0 * (2 ** min(attempt, 4)))  # 2,4,8,16,30...
            if on_retry:
                on_retry(f"Wikipedia HTTP {exc.code}; backing off {wait:.0f}s...")
            time.sleep(wait)
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            wait = min(10.0, 1.0 * (attempt + 1))
            if on_retry:
                on_retry(f"Wikipedia fetch failed ({exc}); retrying in {wait:.0f}s...")
            time.sleep(wait)
            continue

        added = 0
        for paragraph in batch:
            if paragraph and paragraph not in seen:
                seen.add(paragraph)
                out.append(paragraph)
                added += 1
                if len(out) >= n:
                    break
        # Politeness delay between successful requests.
        if len(out) < n:
            time.sleep(0.5)
        if added == 0 and on_retry:
            on_retry("Wikipedia batch returned no usable paragraphs; retrying...")

    if len(out) < n:
        raise RuntimeError(
            f"Could not fetch {n} Wikipedia paragraphs (got {len(out)}) after {max_attempts} attempts"
        )
    return out
