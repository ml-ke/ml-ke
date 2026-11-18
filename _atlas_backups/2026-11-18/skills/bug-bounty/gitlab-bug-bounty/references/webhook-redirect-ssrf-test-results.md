# Webhook Redirect SSRF — Test Results

**Date**: 2026-05-31
**Target**: GitLab.com (v19.1.0-pre)
**Project**: h0d4r1-bugbounty/ssrf-test-1 (ID: 82711314)
**Method**: Python scripts via REST API with PAT

## Summary

Confirmed that GitLab's creation-time URL validation does NOT follow redirect chains. URLs pointing to internal IPs via redirect are accepted at creation time. However, execution-time validation via `NewConnectionAdapter` → `UrlBlocker` catches the redirect target and blocks it.

## Creation-Time Results (All Accepted)

| URL | Hook ID | Status |
|-----|---------|--------|
| `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/test` | 79898624 | ✅ Created |
| `https://httpbin.org/redirect-to?url=http://169.254.169.254/` | 79898626 | ✅ Created |
| `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/ssrf-test` | 79898654 | ✅ Created |
| `https://httpbin.org/redirect-to?url=http://169.254.169.254/latest/meta-data/` | 79898655 | ✅ Created |
| `https://httpbin.org/redirect-to?url=http://httpbin.org/redirect-to%3Furl%3Dhttp%3A%2F%2Flocalhost%3A9200%2Fssrf-chain` | 79898656 | ✅ Created |

## Execution-Time Results (Push Event Trigger)

| URL Pattern | Response Status | Interpretation |
|-------------|----------------|----------------|
| Normal webhook (`webhook.site`) | `200` | Delivered successfully |
| nip.io bypass domains | `403` | Server responded with rejection |
| Redirect chain to `127.0.0.1:9200` | `internal error` | UrlBlocker blocked redirect target |
| Redirect chain to `169.254.169.254` | `internal error` | UrlBlocker blocked redirect target |
| Double redirect chain | `internal error` | Caught on first blocked target |

## Key Code Paths

### NewConnectionAdapter (validates ALL connections including redirect targets)
`gems/gitlab-http/lib/gitlab/http_v2/new_connection_adapter.rb:38-58`
```ruby
def connection
  result = validate_url_with_proxy!(uri)
  @uri = result.uri
  hostname = result.hostname
  http = super
  http.hostname_override = hostname if hostname
  ...
end
```

### UrlBlocker is in HTTP_ERRORS (caught as internal error)
`gems/gitlab-http/lib/gitlab/http_v2/exceptions.rb:21-24`
```ruby
HTTP_ERRORS = HTTP_TIMEOUT_ERRORS + [
  ...
  Gitlab::HTTP_V2::BlockedUrlError, Gitlab::HTTP_V2::RedirectionTooDeep,
]
```

### WebHookService catches exceptions as internal error
`app/services/web_hook_service.rb:107-123`
```ruby
rescue *Gitlab::HTTP::HTTP_ERRORS, Zlib::DataError, ... => e
  ...
  log_execution(response: InternalErrorResponse.new, ...)
```

## Where Redirect SSRF Would Work

On self-hosted instances where:
1. `dns_rebinding_protection_enabled?` is `false` (admin setting)
2. `allow_local_requests_from_web_hooks_and_services?` is `true`

## Cleanup

All test hooks were deleted after testing. Trigger file `trigger-81190.txt` was deleted.
