# MCP Fetch Robots.txt Gate — SSRF Nuance

## The Bug

The MCP Fetch server's `check_may_autonomously_fetch_url()` (line 66-108 in `server.py`) fetches `/robots.txt` before proceeding with the actual URL fetch. This is intended as a safety check — if `robots.txt` says `Disallow: /`, the fetch should be blocked.

## Why It Fails as SSRF Protection

```python
try:
    resp = await client.get(robots_url, follow_redirects=True)
    resp.raise_for_status()
    ...
except httpx.HTTPError:  # ← Only catches HTTPError
    return True  # Can't check, allow fetch
```

The except clause only catches `httpx.HTTPError`, **NOT** `httpx.ConnectError`:

| Scenario | Exception Type | Caught? | Result |
|----------|---------------|---------|--------|
| Target port closed (ECONNREFUSED) | `ConnectError` | ❌ No | Uncaught exception → fetch fails with internal error |
| Target responds | `HTTPError` (4xx/5xx) | ✅ Yes | `return True` → main fetch **proceeds** |
| Target responds 200 | None | N/A | robots.txt parsed normally |

## Practical Impact

- **Checking a closed port** → fetch dies with unhandled exception (blocks SSRF)
- **Checking any listening service** → robots.txt fetch returns 404 or connection proceeds → main fetch proceeds normally

## Exploitation Angle

Target services that are likely listening on standard ports:
- Cloud metadata: `169.254.169.254:80` (HTTP)
- Elasticsearch: `10.0.0.1:9200` (HTTP API)
- Internal Kibana: `127.0.0.1:5601` (HTTP)
- Redis: `127.0.0.1:6379` (raw TCP — may not respond to HTTP GET cleanly)

## Lesson for Other Programs

When auditing any SSRF protection layer that includes a "preflight" request:
1. **Check the exception hierarchy**: Does the catch-all cover both transport errors (`ConnectError`, `TimeoutException`) AND HTTP errors (`HTTPStatusError`)?
2. **Test closed vs open ports**: A protection that only blocks closed ports is not really protection.
3. **Look at `follow_redirects`**: If the preflight follows redirects, it could be tricked into hitting a whitelisted domain while the main fetch hits the internal target.
