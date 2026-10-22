# bodyHash Report Fact-Check — Worked Example

This is a record of the fact-checking process applied to a real cryptographic finding.
Use as a template for reviewing crypto reports before submission.

## Original Claims (v1 of report)

| Claim | Status After Verification |
|-------|--------------------------|
| Bug exists in `@fireblocks/ts-sdk@19.1.0` | **Correct** but incomplete — ALL 64 versions (v8.0.1→v20.0.0) |
| SDK crashes on Node 22 (TypeError) | **Confirmed** — `crypto.update(body)` throws `ERR_INVALID_ARG_TYPE` |
| All POST bodies produce identical hash on Node 18 | **Confirmed** — all produce SHA256("[object Object]") |
| **JWT can be replayed with different body** | **FALSE** — server tracks nonces, returns code -13 on replay |
| Severity P2 (CVSS 7.5) | **Incorrect** — P2 was justified by false replay claim. Correct baseline: P3 |
| VRT: "Cryptographic Vulnerability" | **Wrong format** — corrected to `Cryptographic Weakness > Broken Cryptography > Use of Broken Cryptographic Primitive` |

## Verification Process

### Step 1: Test every impact claim against the live server

```javascript
// Claim: "JWT can be replayed"
// Test: send same JWT (same nonce) twice
// Result: First=200, Second=401 "nonce already used" (code -13)
// → Claim is FALSE. Remove from report.
```

### Step 2: Check scope across ALL versions

```bash
npm pack @fireblocks/ts-sdk@8.0.1
npm pack @fireblocks/ts-sdk@10.0.0
npm pack @fireblocks/ts-sdk@14.0.0
npm pack @fireblocks/ts-sdk@19.1.0
npm pack @fireblocks/ts-sdk@20.0.0
# All five have the same bug at network/bearerTokenProvider.ts:47
# → 64 versions total, ALL affected. Update scope claim.
```

### Step 3: Check compiled output too

```bash
grep "bodyHash" package/dist/network/bearerTokenProvider.js
# Bug is also in the compiled JS that npm actually serves
```

### Step 4: Check other SDKs for comparison

```bash
grep "bodyHash" fireblocks-sdk-js/src/api-token-provider.ts
# → Uses JSON.stringify(bodyJson || "") — correct
# Only the auto-generated @fireblocks/ts-sdk is broken
```

## Narrative Reframe

The lesson from this review: **lead with the broken security mechanism, not the symptom.**

| Before | After |
|--------|-------|
| "SDK crashes on POST" (availability framing) | "bodyHash cryptographic integrity mechanism is nullified — the hash never represents the actual body" (crypto weakness framing) |
| "Write operations fail" (DoS framing) | "Server-enforced integrity check cannot be satisfied — every write request returns code -9" |
| PoC shows crash | PoC shows the crypto failure: invariant hash proves the primitive is broken |
| Focus on symptom | Focus on mechanism: bodyHash guarantees request integrity, but the broken computation makes this guarantee impossible for TypeScript users |

## Key Takeaways

1. **Test every impact claim** against the live server before writing it in the report
2. **Check all versions** — not just the one you first found. The bug might be systemic.
3. **Check compiled output** — npm serves JS, not TS. The bug must be in the JS.
4. **Compare across SDKs** — if other SDKs do it correctly, that strengthens the case.
5. **Frame around the security mechanism** — not the symptom. The mechanism is the vulnerability; the crash/401 is just how it manifests.
