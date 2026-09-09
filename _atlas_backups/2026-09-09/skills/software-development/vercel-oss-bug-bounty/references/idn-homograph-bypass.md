# IDN Homograph SSRF Bypass in validateDownloadUrl

## Root Cause

The `validateDownloadUrl()` function checks hostnames using exact ASCII string matching:

```javascript
if (hostname === 'localhost' || hostname.endsWith('.local') || hostname.endsWith('.localhost'))
```

But `new URL()` automatically converts Unicode characters in hostnames to Punycode (IDN encoding). This means the ASCII string comparison fails for visually-identical Unicode homoglyphs.

## How It Works

| Input URL | `new URL().hostname` | Hostname Check | Result |
|-----------|---------------------|----------------|--------|
| `http://localhost:8080/` | `localhost` | `=== 'localhost'` → ✅ blocked | Correct |
| `http://lоcalhоst:8080/` (Cyrillic 'o') | `xn--lcalhst-9ige` | `=== 'localhost'` → 🚨 fails | BYPASS |
| `http://test.lоcal/` (Cyrillic 'o') | `test.xn--lcal-55d` | `.endsWith('.local')` → 🚨 fails | BYPASS |
| `http://test.lоcalhost/` (Cyrillic 'o') | `test.xn--lcalhost-nbh` | `.endsWith('.localhost')` → 🚨 fails | BYPASS |

## Unicode Homoglyphs That Work

Common Cyrillic characters that look identical to Latin letters:

| Latin | Cyrillic | Unicode | Looks like |
|-------|----------|---------|------------|
| `o` | `о` | U+043E | identical |
| `a` | `а` | U+0430 | identical |
| `e` | `е` | U+0435 | identical |
| `c` | `с` | U+0441 | identical |
| `p` | `р` | U+0440 | identical |
| `x` | `х` | U+0445 | identical |

## Verification

```javascript
const url = 'http://lоcalhоst:8080/test';  // Cyrillic 'o' (U+043E)
const parsed = new URL(url);
console.log(parsed.hostname);
// → "xn--lcalhst-9ige" (Punycode - NOT "localhost")

// validateDownloadUrl checks:
console.log(parsed.hostname === 'localhost');
// → false — BYPASS!

console.log(parsed.hostname.endsWith('.local'));
// → false — BYPASS!
```

## Exploitation Path

1. Register a domain with Unicode homoglyphs
   - Example: `lоcalhоst.com` (Cyrillic 'o') → Punycode: `xn--lcalhst-9ige.com`
2. Point the domain to a private IP (127.0.0.1, 169.254.169.254, etc.)
3. Supply a URL to the AI SDK like `http://xn--lcalhst-9ige.com:8080/internal`
4. `validateDownloadUrl` checks `xn--lcalhst-9ige.com` → not localhost, not an IP → PASSES
5. `fetch()` resolves the domain → connects to private IP → SSRF achieved

## Impact

**Medium** — Requires domain registration. However, the visual similarity makes this effective for phishing-style attacks and CI/CD pipeline injection where automated scanners check for "localhost" in URLs.

## Fix

Decode Punycode before comparing against ASCII blocklist:

```javascript
function isBlockedHostname(hostname) {
  try {
    hostname = new URL(`http://${hostname}`).hostname;
    // or use punycode.toUnicode(hostname)
  } catch {}
  
  return (
    hostname === 'localhost' ||
    hostname.endsWith('.local') ||
    hostname.endsWith('.localhost')
  );
}
```
