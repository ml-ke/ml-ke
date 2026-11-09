# Social / X (Twitter) Tweet Content Retrieval — No Auth Required

Verified Aug 2026. Use when you need the *text* of a specific tweet for an
evidence chain (incident verification, quote capture) and `xurl` is not
installed/authenticated, or X blocks direct access (login wall,
`net::ERR_HTTP_RESPONSE_CODE_FAILURE`, nitter "Verifying your browser…").

## Fastest path: vxtwitter / fxtwitter (no auth, works from curl)

```bash
curl -s "https://api.vxtwitter.com/<handle>/status/<tweet_id>"   # JSON, includes full text + media
curl -s "https://api.fxtwitter.com/<handle>/status/<tweet_id>"   # same, nested under .tweet
```

- Returns: full `text` (untruncated), `date`, `likes`/`retweets`/`replies`,
  `mediaURLs`, `lang`, `qrt` (quoted tweet).
- Works for any public tweet without OAuth. This is the single most reliable
  fallback when X itself is unreachable from the box.
- vxtwitter response is flat; fxtwitter wraps everything under `.tweet`.

## Decode the tweet timestamp from the snowflake ID (stdlib)

```python
tid = 2091064574074040629
ts_ms = (tid >> 22) + 1288834974657   # epoch ms → 2010-11-04 epoch
import datetime; print(datetime.datetime.utcfromtimestamp(ts_ms/1000).isoformat(), "UTC")
```

Useful to date a tweet before/without fetching it, and to sanity-check that a
shared tweet is recent enough to be the incident the user means.

## Fallback ladder (when vxtwitter/fxtwitter fail)

1. `https://publish.twitter.com/oembed?url=<full_url>` — returns HTML embed
   with the text; frequently empty for new/high-traffic tweets.
2. Direct `https://x.com/<handle>/status/<id>` via browser tool — often
   blocked (login wall / HTTP 400-500).
3. Nitter instances (`nitter.net`, `xcancel.com`, `nitter.poast.org`) —
   mostly Cloudflare-verification-walled; rarely usable from scripts.
4. Web search the tweet ID + handle (`<handle> tweet <id>`) — search
   snippets sometimes carry the text; also surfaces news coverage of the
   incident, which is what you actually cite in a report.

## Verification discipline

- A tweet is a PRIMARY claim, not a fact. Cross-check any incident figures
  (amounts, dates, names) against 2+ independent news sources before putting
  them in a report/blog post (this caught one unverifiable Sidian 2022 claim
  that was replaced with a verified Oct 2025 incident).
- Capture the tweet's `date` field and use it as the incident's publication
  date anchor.
- When writing about the incident, cite the news sources, not the tweet, as
  the References — tweets rot behind login walls.
