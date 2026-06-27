# SSRF Bypass: CGNAT & Reserved IP Range Validation Gap

## Overview

The `isPrivateIPv4()` function in `@ai-sdk/provider-utils` (v4.0.19–4.0.27) only blocks 6 well-known private IP ranges but misses 7+ additional reserved/special-purpose ranges. This allows direct IP-based SSRF bypass — no DNS trickery needed.

## Blocked vs Missed Ranges

### Correctly Blocked (6 ranges)
```
0.0.0.0/8       → Current network
10.0.0.0/8      → Private (RFC 1918)
127.0.0.0/8     → Loopback
169.254.0.0/16  → Link-local
172.16.0.0/12   → Private (RFC 1918)
192.168.0.0/16  → Private (RFC 1918)
```

### NOT Blocked (7 ranges)
```
100.64.0.0/10   → CGNAT (Carrier-Grade NAT) → 🔥 MOST IMPACTFUL
198.18.0.0/15   → Benchmarking (RFC 2544)
192.0.2.0/24    → TEST-NET-1 (RFC 5737)
198.51.100.0/24 → TEST-NET-2 (RFC 5737)
203.0.113.0/24  → TEST-NET-3 (RFC 5737)
240.0.0.0/4     → Reserved
```

## Verification

```javascript
import { validateDownloadUrl } from '@ai-sdk/provider-utils';

// These pass validation — NOT blocked:
validateDownloadUrl('http://100.64.0.1:8080/config');      // CGNAT
validateDownloadUrl('http://100.100.100.100:8080/');       // CGNAT
validateDownloadUrl('http://198.18.0.1:8080/test');        // Benchmarking
validateDownloadUrl('http://192.0.2.1:8080/');             // TEST-NET-1
validateDownloadUrl('http://240.0.0.1:8080/');             // Reserved

// These correctly throw DownloadError:
validateDownloadUrl('http://127.0.0.1:8080/secrets');      // blocked
validateDownloadUrl('http://10.0.0.1:8080/internal');      // blocked
validateDownloadUrl('http://192.168.1.1:8080/config');     // blocked
```

## Impact

CGNAT (100.64.0.0/10, RFC 6598) is used by:
- **Google Cloud** — internal VPC networking, GKE
- **Azure** — internal infrastructure
- **AWS** — EKS, some managed services
- **ISPs worldwide** — shared addressing

## Comparison: GitLab gets it right

GitLab's `UrlBlocker.validate_shared_address()` (url_blocker.rb:315):
```ruby
def validate_shared_address(addrs_info)
  netmask = IPAddr.new('100.64.0.0/10')
  return unless addrs_info.any? { |addr| netmask.include?(addr.ip_address) }
  raise BlockedUrlError, "Requests to the shared address space are not allowed"
end
```

Next.js image optimizer also gets it right using the `ipaddr` library:
```javascript
const range = addr.range();
return range !== 'unicast';  // blocks ALL non-public ranges
```

## Fix

Replace the manual `isPrivateIPv4` with the `ipaddr` library's `range()` method, or add the missing ranges:

```javascript
if (a === 100 && b >= 64 && b <= 127) return true; // CGNAT
if (a === 198 && (b === 18 || b === 19)) return true; // Benchmarking
// ... etc.
```

## PoC Script

See `scripts/test-cgnat-bypass.mjs` in this skill.
