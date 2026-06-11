---
title: Rapyd Bug Bounty — Methodology & Research Notes
name: rapyd-bug-bounty
description: Methodology for the Rapyd fintech API bug bounty program on Bugcrowd. Covers scope, targets, SAML 2.0 promotion, and honest research notes for api.rapyd.net findings that need production verification.
---

# Rapyd Bug Bounty — Methodology & Research Notes

## Critical Scope Note (from honest audit)

The Bugcrowd Rapyd program targets are:

| Target | Tier | Type | Access |
|--------|------|------|--------|
| api.rapyd.net | T3 | API testing | Sandbox GET only; POST needs Iceland prod |
| dashboard.rapyd.net | T2 | Website | Sandbox signup via @bugcrowdninja.com |
| verify.rapyd.net | T2 | Website (iframe) | Via dashboard |
| checkout.rapyd.net | T2 | Website | Public |
| *.rapyd.net | T1 | Wildcard | Varies |
| *.rapyd.com / *.rapyd.org | T1 | Wildcard | Unknown |

**GitHub repos are NOT listed as targets.** Bugs in `github.com/Rapyd-Samples/rapyd-ts-client` affect the merchant's server, NOT `api.rapyd.net`. Do NOT submit sample code findings.

**Before writing any report, verify 6 things:**
1. Where does the code actually run? (Vendor infra or merchant's server?)
2. Is the endpoint on the live target? (curl it — does it exist?)
3. Can the finding be verified against the live API, not a local simulation?
4. Is the target listed in program scope?
5. What do vendor docs say about where their code runs?
6. Is OpenAPI spec analysis confirmed by live API testing?

See `api-bug-bounty-methodology` → `references/scope-verification-case-study.md` for the worked example of this mistake.

## Active Promo: SAML 2.0 (May 20 - June 30, +$400/+$900)

Test on `dashboard.rapyd.net`. 8 attack types listed:
1. Assertion Replay — single-use enforcement on SAMLResponse
2. Signature Wrapping (XSW) — injection alongside valid signed assertion
3. InResponseTo Enforcement — missing/modified ID rejection
4. Account Takeover via NameID / Email Spoofing
5. RelayState Open Redirect
6. XXE Injection via SAML Payload
7. Privilege Escalation via Attribute Manipulation
8. Tenant Isolation Bypass

**Relevant skill:** `saml-attack-techniques` — covers 8 novel SAML bypass techniques from PortSwigger's "The Fragile Lock" paper, including Void Canonicalization, Attribute Pollution, and REXML Namespace Confusion. Load this skill when testing the Rapyd SAML promotion.

**Known OOS**: IDOR on idp_id parameter (already tracked internally)
**Known OOS**: Dummy/permissive IdPs (MockSAML, etc.), admin attacking own config

## Research Notes (Unverified — Need Iceland Production API Access)

These come from OpenAPI spec analysis of `api.rapyd.net`. All need verification against the live production API.

| Topic | Source | What to Verify | Status |
|-------|--------|----------------|--------|
| Idempotency missing | `rapyd-openapi.yaml` — 0/192 POST endpoints require idempotency-key | Live API may enforce it server-side | Needs prod testing |
| Mass assignment | `rapyd-openapi.yaml` — inline beneficiary/customer objects in request bodies | Live API may strip/reject | Needs prod testing |
| PII leakage | `rapyd-openapi.yaml` — full PII fields in response schemas | Live API may mask fields | Needs prod testing |

Research notes stored at `~/Dev/REPORTS/Rapyd/013/` (idempotency + mass assignment) and `~/Dev/REPORTS/Rapyd/015/` (PII leakage). Both annotated as unverified.

## Program Stats

- Platform: Bugcrowd — Rapyd (since Nov 2022)
- Scope rating: 4/4
- 83 vulnerabilities rewarded, 75% accepted in 6 days
- Average payout (last 3mo): $185
- Stack: TypeScript/Node.js, OpenAPI spec (33,511 lines), custom HMAC signing

## Signup

- Sandbox: `dashboard.rapyd.net` with @bugcrowdninja.com email
- Production: Select Iceland as country during signup
- API keys: generated from dashboard after signup

## Rapyd HMAC Signing (verified against sandbox)

```python
import time, base64, hmac, hashlib, os

def sign(secret_key, access_key, method, path, body=""):
    salt = base64.b64encode(os.urandom(12)).decode()
    ts = str(int(time.time()))
    to_sign = method.lower() + path + salt + ts + access_key + secret_key + body
    h = hmac.new(secret_key.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
    signature = base64.b64encode(bytes.fromhex(h)).decode()
    return salt, ts, signature
```

## What Was Deleted (Not Submittable)

These findings were based on sample code at `github.com/Rapyd-Samples/rapyd-ts-client` that runs on merchant infrastructure, not on any `*.rapyd.net` target:

- Webhook bugs (no auth, ngrok URL, OOB array, silent return) — all in `src/router/` and `src/controller/` — merchant-deployed code
- Cart IDOR — `src/controller/checkout.controller.ts` — merchant-deployed e-commerce app
- Password hash in session — `src/strategies/local.strategy.ts` — merchant's session handling
- HMAC weakness — `src/utils/signRapydRequest.ts` — how merchant signs requests, not Rapyd's validation
- No security scheme in OpenAPI spec — documentation issue, not an exploitable vulnerability on any listed target

## References

- `~/Dev/REPORTS/Rapyd/000-honest-audit.md` — Full honest audit of all findings
- `~/Dev/rapyd-openapi/rapyd-openapi.yaml` — OpenAPI spec (33,511 lines)
- `~/Dev/rapyd-ts-client/src/` — Sample code (merchant-side, NOT an in-scope target)
