# Webhook SSRF Test Results (May 31, 2026)

## Environment
- Target: gitlab.com (19.1.0-pre, Enterprise)
- User: h0d4r1-bugbounty
- Project: ssrf-test-project (ID 82711314)
- Auth: PAT at ~/.gitlab-token
- Version from API: `{'version': '19.1.0-pre', 'enterprise': True}`

## Test 1: Redirect SSRF — Creation

**Method**: POST /api/v4/projects/82711314/hooks

**Results** — All redirect-chain URLs accepted at creation:

| URL | Hook ID | Created |
|-----|---------|---------|
| `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/ssrf-test` | 79898654 | ✅ |
| `https://httpbin.org/redirect-to?url=http://169.254.169.254/latest/meta-data/` | 79898655 | ✅ |
| `https://httpbin.org/redirect-to?url=http://httpbin.org/redirect-to%3Furl%3Dhttp%3A%2F%2Flocalhost%3A9200%2Fssrf-chain` | 79898656 | ✅ (double redirect) |
| `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/test` | 79898624 | ✅ (from prior session) |
| `https://httpbin.org/redirect-to?url=http://169.254.169.254/` | 79898626 | ✅ (from prior session) |

**Direct SSRF URLs** — ALL blocked at creation:
- `http://127.0.0.1:9200/` → ❌ 422
- `http://169.254.169.254/` → ❌ 422
- `http://10.0.0.1:9200/` → ❌ 422

## Test 2: Webhook Execution via Push Event

**Method**: Created file on main branch via POST /api/v4/projects/82711314/repository/files

**Results** — Status from webhook logs after push event:

| Hook ID | URL | Status | Interpretation |
|---------|-----|--------|----------------|
| 79786008 | `http://1.0.0.127.nip.io:8080/webhook` | 403 | Passed UrlBlocker, server responded |
| 79786028 | `http://1.0.0.127.nip.io:8080/test` | 403 | Same — nip.io server responds |
| 79786030 | `http://1.1.1.1.nip.io:8080/test` | 403 | Same |
| 79786031 | `http://0x7f000001.nip.io:8080/test` | internal error | UrlBlocker blocked (hex IP resolved to loopback) |
| 79786032 | `http://2130706433.nip.io:8080/test` | internal error | UrlBlocker blocked (decimal IP resolved to loopback) |
| 79786033 | `http://65535.nip.io:8080/test` | internal error | UrlBlocker blocked |
| 79786035 | `http://0.nip.io:8080/test` | internal error | UrlBlocker blocked (0.0.0.0) |
| 79786088 | `https://webhook.site/...` | **200** | ✅ Delivered successfully |
| 79898624 | `httpbin.org/redirect-to?url=http://127.0.0.1:9200/test` | internal error | Redirect followed, target blocked by UrlBlocker |
| 79898626 | `httpbin.org/redirect-to?url=http://169.254.169.254/` | internal error | Redirect followed, metadata blocked |
| 79898654 | `httpbin.org/redirect-to?url=http://127.0.0.1:9200/ssrf-test` | internal error | Redirect followed, target blocked |
| 79898655 | `httpbin.org/redirect-to?url=http://169.254.169.254/latest/meta-data/` | internal error | Redirect followed, metadata blocked |
| 79898656 | `httpbin.org/redirect-to?url=http://httpbin.org/redirect-to%3Furl%3Dhttp%3A%2F%2Flocalhost%3A9200%2Fssrf-chain` | internal error | Double redirect, each hop validated |

## Interpretation

**`internal error`** = `BlockedUrlError` raised by `NewConnectionAdapter#connection` when UrlBlocker rejects the redirect target. Caught by `rescue *HTTP_ERRORS` at `web_hook_service.rb:107`.

**`403`** = Connection succeeded through UrlBlocker, external server responded with 403.

## Cleanup

All test hooks deleted after testing. Trigger file (`trigger-81190.txt`) deleted. Project (82711314) preserved.
