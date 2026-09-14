# GitLab Redirect SSRF Analysis — Live Test Results (May 2026)

## Webhook Redirect Chain Test

### Creation-Time Results
Redirect URLs pointing to internal IPs were **accepted** at webhook creation:

| URL | Creation Result | Hook ID |
|-----|----------------|---------|
| `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/test` | ✅ Created | 79898624 |
| `https://httpbin.org/redirect-to?url=http://169.254.169.254/` | ✅ Created | 79898626 |
| `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/ssrf-test` | ✅ Created | 79898654 |
| `https://httpbin.org/redirect-to?url=http://169.254.169.254/latest/meta-data/` | ✅ Created | 79898655 |
| `https://httpbin.org/redirect-to?url=http://httpbin.org/redirect-to%3Furl%3Dhttp%3A%2F%2Flocalhost%3A9200%2Fssrf-chain` | ✅ Created (double redirect) | 79898656 |

Creation-time validation only checks the initial URL (httpbin.org). Redirect targets are NOT validated at creation.

### Execution-Time Results
After triggering via push event:

| Hook URL | Response Status | Interpretation |
|----------|----------------|---------------|
| Normal webhook (webhook.site) | `200` | Delivered successfully |
| nip.io bypass domains | `403` | Connection succeeded but server rejected |
| **httpbin.org → 127.0.0.1 redirect** | **`internal error`** | **UrlBlocker blocked redirect target** |
| **httpbin.org → 169.254.169.254 redirect** | **`internal error`** | **UrlBlocker blocked redirect target** |
| **httpbin.org → httpbin.org → localhost** | **`internal error`** | **UrlBlocker blocked redirect target** |

### HTTParty Redirect Flow

1. GitLab's `WebHookService` calls `Gitlab::HTTP.post(url, ...)` via `make_request` (line 155-163 of `web_hook_service.rb`)
2. HTTParty creates a connection via `NewConnectionAdapter#connection` which validates the URL through `UrlBlocker.validate_url_with_proxy!()`
3. If the server responds with a redirect, HTTParty follows it by creating a NEW request
4. The new request goes through `NewConnectionAdapter` AGAIN → redirect target is independently validated
5. If UrlBlocker blocks the redirect target, `BlockedUrlError` is raised (caught by `rescue *HTTP_ERRORS` at line 107)

### Webhook Error Interpretation

| Status | Meaning | Source |
|--------|---------|--------|
| `200` | Delivered | Normal HTTP response |
| `403` | Server responded with rejection | External server returned 403 |
| `internal error` | Exception raised before/during connection | `InternalErrorResponse` created in `web_hook_service.rb:6-28`, caught by `rescue *HTTP_ERRORS` at line 107 |
| `internal error` with redirect | **UrlBlocker blocked redirect target** | `NewConnectionAdapter` raises `BlockedUrlError` → included in `HTTP_ERRORS` (`exceptions.rb:21-24`) |

### Key Source Files

| File | Path |
|------|------|
| WebHookService | `app/services/web_hook_service.rb` (lines 107-123: error handling, 155-163: HTTP request) |
| NewConnectionAdapter | `gems/gitlab-http/lib/gitlab/http_v2/new_connection_adapter.rb` (line 38-58: connection validation) |
| HTTP_ERRORS | `gems/gitlab-http/lib/gitlab/http_v2/exceptions.rb` (line 21-24: includes BlockedUrlError) |
| HTTParty override | `gems/gitlab-http/lib/gitlab/http_v2/client.rb` (line 49-71: perform_request override) |
