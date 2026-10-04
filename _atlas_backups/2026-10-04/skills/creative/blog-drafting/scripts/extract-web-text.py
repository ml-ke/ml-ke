#!/usr/bin/env python3
"""Fetch a URL with a browser UA and print tag-stripped text around a key phrase.

Use when web_extract is unavailable (search-only backend) and you must verify
exact figures quoted in a blog post against the live article. Search snippets
are NOT enough for quoted numbers — this fetches the real body.

Usage:
    python3 scripts/extract-web-text.py URL [KEYPHRASE] [WINDOW_CHARS]

Exit codes:
    0  OK (text printed; with KEYPHRASE, the window around it)
    1  fetch failed
    2  body suspiciously small (<2000 chars) — page is likely JS-gated
       ("Enable JavaScript and cookies to continue"); try a secondary source
       that quotes the primary (TechCrunch, CNBC, SCMP, Simon Willison, ...)
    3  KEYPHRASE not found in the cleaned text

Examples:
    python3 scripts/extract-web-text.py https://techcrunch.com/... "under 13 hours"
"""
import re
import sys
import html
import urllib.request

url = sys.argv[1]
phrase = sys.argv[2] if len(sys.argv) > 2 else None
window = int(sys.argv[3]) if len(sys.argv) > 3 else 2500

req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
)
try:
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
except Exception as e:  # noqa: BLE001 — report and exit; caller picks another source
    print(f"FETCH FAILED: {e}", file=sys.stderr)
    sys.exit(1)

text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.S)
text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S)
text = re.sub(r"<[^>]+>", " ", text)
text = html.unescape(text)
text = re.sub(r"\s+", " ", text).strip()

if len(text) < 2000:
    print(
        f"WARNING: only {len(text)} chars — page is likely JS-gated. "
        "Try a curl-friendly secondary source that quotes the primary.",
        file=sys.stderr,
    )
    sys.exit(2)

if phrase:
    idx = text.find(phrase)
    if idx == -1:
        print(f"KEYPHRASE NOT FOUND: {phrase!r}", file=sys.stderr)
        sys.exit(3)
    print(text[max(0, idx - window // 2): idx + window])
else:
    print(text[:window])
