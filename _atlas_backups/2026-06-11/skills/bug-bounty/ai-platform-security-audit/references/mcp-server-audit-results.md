# MCP Fetch Server SSRF Analysis

**Date**: 2026-05-31
**Source**: github.com/modelcontextprotocol/servers (src/fetch/)
**Language**: Python

## Finding: No SSRF Protection

The `fetch_url` function in `src/fetch/src/mcp_server_fetch/server.py` makes HTTP requests with NO URL validation whatsoever:

```python
async def fetch_url(
    url: str, user_agent: str, force_raw: bool = False, proxy_url: str | None = None
) -> Tuple[str, str]:
    from httpx import AsyncClient, HTTPError
    async with AsyncClient(proxy=proxy_url) as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
                timeout=30,
            )
```

### Missing Protections
- No IP address filtering (private ranges, loopback, metadata)
- No hostname blocklist
- No redirect target validation (`follow_redirects=True`)
- No protocol restriction
- No URL length limit
- No content size limit (beyond available memory)

### Attack Surface
- `http://127.0.0.1:9200/` — Local Elasticsearch
- `http://localhost:5601/` — Local Kibana
- `http://169.254.169.254/latest/meta-data/` — Cloud metadata (if on AWS)
- `http://10.0.0.1:9200/` — Internal network services

### Contrast with Claude Code WebFetchTool (TypeScript)

| Protection | MCP Fetch Server | Claude Code WebFetchTool |
|------------|-----------------|------------------------|
| IP filtering | None | Server-side blocklist via api.anthropic.com |
| Hostname validation | None | Blocks <2 part hostnames |
| Redirect validation | None (`follow_redirects=True`) | `isPermittedRedirect` — same-host only |
| HTTP→HTTPS upgrade | None | Yes, forced |
| URL length limit | None | 2000 chars |
| Content size limit | None | 10MB |
| Timeout | 30s | 60s |
| Auth/user check | None | Preapproval + permission system |

## MCP Filesystem Server Path Validation

File: `src/filesystem/path-validation.ts`

```typescript
export function isPathWithinAllowedDirectories(absolutePath, allowedDirectories) {
  normalizedPath = path.resolve(path.normalize(absolutePath));
  return allowedDirectories.some(dir => {
    normalizedDir = path.resolve(path.normalize(dir));
    return normalizedPath === normalizedDir || 
           normalizedPath.startsWith(normalizedDir + path.sep);
  });
}
```

### Protections Present
- `path.resolve(path.normalize(...))` — handles `..` traversal
- Null byte rejection
- Absolute path enforcement

### Gaps
- Symlink bypass: allowed directory contains symlink → `path.resolve` follows it → escapes
- TOCTOU: validation and file access are separate operations
- No `fs.realpath` on individual file paths (only on allowed directories at startup)
