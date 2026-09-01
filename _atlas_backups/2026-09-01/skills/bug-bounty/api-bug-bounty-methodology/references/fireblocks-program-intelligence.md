# Fireblocks Web Bug Bounty — Program Intelligence

## Program Overview

- **Bugcrowd URL**: `https://bugcrowd.com/engagements/fireblocks-mbb-og` (Web)
- **Second program**: `https://bugcrowd.com/engagements/fireblocks-mbb-og2` (MPC — cryptographic protocol attacks)
- **Payouts**: P1 $7K-$12K, P2 $1K-$9K, P3 $300-$1.5K, P4 $20-$300
- **Scope rating**: 1/4 (limited targets — sandbox API only)
- **Bugs rewarded**: 2 (as of Jun 2026 — low competition)
- **Validation time**: avg 19 days
- **Status**: Ongoing since Sep 09, 2025
- **Signup**: @bugcrowdninja.com email via https://www.fireblocks.com/developer-sandbox-sign-up

## In-Scope Targets (Web Program)

| Target | Status | Notes |
|--------|--------|-------|
| `sandbox-api.fireblocks.io` | ✅ **Accessible** | Returns 401 (not CF!), Express + Cloudflare |
| `sb-console-api.fireblocks.io` | ⚠️ Returns Express 404 | Different Express app, no endpoints found |
| `sb-mobile-api.fireblocks.io` | ⚠️ Returns Express 404 | Same as console — different routing |

Key finding: `sandbox-api.fireblocks.io` returns `401: JWT is missing` — NOT a Cloudflare JS challenge. API is directly accessible with valid credentials.

## Auth Mechanism

Fireblocks uses per-request RS256-signed JWTs. **Critical: Python PyJWT produces incorrect RS256 signatures for Fireblocks. Always use Node.js `jsonwebtoken`.**

```
Headers: X-API-Key + Authorization: Bearer <JWT>
JWT payload: { uri, nonce, iat, exp, sub, bodyHash }
Algorithm: RS256
Validity: 55 seconds
```

For GET: bodyHash = SHA256 of empty string.
For POST/PUT/PATCH: bodyHash = SHA256 of JSON.stringify(body).
uri = pathname + search from the full URL (must use `new URL(url)` parsing).

See `bug-bounty/fireblocks-api-toolkit` skill for the Node.js signing function and error code reference.

## POST 401 Solution (code -7)

The code -7 "Error getting User certificate" had three causes, all solved:

1. **Missing Content-Length header** — Node.js `https.request()` doesn't auto-set it. Fix: `'Content-Length': Buffer.byteLength(JSON.stringify(body))`
2. **Incorrect bodyHash** — Must be SHA256 of `JSON.stringify(body)`, NOT the raw JS object
3. **Rate limiting** — Rapid requests (10+ in 30s) lock the key. Wait 10-30s.

After fixes: **POST vault account creation, transaction creation, and internal wallet creation all work.**

## Live Testing Results (sandbox-api.fireblocks.io/v1)

### Working Endpoints
- `GET /vault/accounts_paged` — Vault list (5 vaults found: IDs 0-4)
- `GET /vault/accounts/{id}` — Individual vault (enumeration: sequential IDs)
- **`POST /vault/accounts` — ✅ Created vaults 1, 2, 3, 4 successfully**
- **`POST /v1/transactions` — ✅ Created transaction `6684a082` with status `SUBMITTED`**
- **`POST /v1/internal_wallets` — ✅ Wallet `80a6c38c` created**
- `GET /transactions` — Transaction list
- `GET /supported_assets` — All 1000+ supported coins
- `GET /internal_wallets` — Internal wallets (Gas Station Wallet pre-loaded)
- `GET /gas_station` — Gas station config
- `GET /staking/chains` — Supported staking chains
- `GET /cosigners` — Cosigners (Fireblocks Communal Test Co-signer)

**POST operations work** — fix was adding Content-Length header and using correct bodyHash (SHA256 of JSON.stringify, not raw object).

### Protected Endpoints
- `GET /users` — 403 "Insufficient permissions"
- **`POST /v1/webhooks` — 403 "Forbidden resource"** — URL validation blocks ALL URLs (internal + external). No SSRF bypass found.
- Most advanced endpoints (audit_logs, nfts, policy, keys, smart_transfers) — 404 "Endpoint not defined"

### Key Observations
- **Vault IDs are sequential integers** (0, 1, 2, 3, 4...) — confirmed by enumerating IDs
- **POST operations work when** Content-Length header is set AND bodyHash uses JSON.stringify
- **Rate limiting is aggressive** — ~10 rapid POSTs lock the key for 10-30s (code -7 on ALL endpoints until it recovers)
- **Webhook SSRF blocked** — all URL formats (IP, domain, DNS rebinding) return 403
- **GitHub org**: 70 repos at github.com/orgs/fireblocks — ts-sdk (has bodyHash bug), py-sdk, java-sdk, developers-hub

## Attack Chain

**bodyHash bug (TS SDK) + idempotency key bypass:**

1. The published `@fireblocks/ts-sdk@19.1.0` has `crypto.createHash("sha256").update(bodyJson || "")` at bearerTokenProvider.ts:47 — raw object passed to crypto.update()
2. Node v22: throws TypeError (SDK crashes). Node ≤18: hashes `[object Object]` (ALL POSTs have same bodyHash)
3. Combined with `Idempotency-Key` not being in the JWT (added after signing at axiosManager.ts:78-83):
4. Captured JWT can be replayed with attacker-controlled body + different idempotency key within 55s

**Every other SDK does it correctly:** Python json.dumps(), Java getBytes(), Go json.Marshal(). Only new TS SDK is broken.

## Top Findings Summary

| # | Finding | CVSS | As an attacker I would... |
|---|---------|------|--------------------------|
| 1 | TS SDK bodyHash bug | 7.5 | capture one JWT, replay with any POST body — bodyHash always SHA256("[object Object]") |
| 2 | Idempotency key not signed | 5.4 | replay same JWT with 20 different idempotency keys — 20 duplicate transactions |
| 3 | Sequential vault IDs | 5.3 | enumerate vaults 0..N to map entire workspace before selecting a target |
