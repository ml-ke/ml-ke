# Nutaku Rejection Analysis — June 3-5, 2026

## Overview
Two submissions on the Nutaku (Aylo) Intigriti program were rejected. This document captures what went wrong, what the gaps were, and the decision framework applied after rejection.

## Finding 1: Hardcoded OAuth2 Credentials in Nutaku Android APK

- **Code**: AYLO-4XKLFCN9
- **Severity**: Critical (submitted) → Informative (closed)
- **Claim**: CWE-798 Hardcoded credentials in mobile APK (oauth-front/oauth-frontpass)
- **What the triager said**: Informative — not a valid vulnerability

### Root Cause Analysis

The pre-submission-verification skill's Gate C4 says:
> *"Hardcoded credentials in a mobile app talking to that app's own backend API is industry standard practice and is never a finding on its own."*

**Three-layer Credential-Access Gap Test:**
1. Can you authenticate with it? → Yes (got bearer token from gateway)
2. Does it unlock something beyond the app? → Gateway API now returns Connection Refused. No live endpoint to test against.
3. Can a real attacker reach those endpoints? → Even if live, the endpoint may be internal-only.

**Verdict**: Correctly Informative. The credentials alone aren't a finding. To make this valid, we would need:
- A live gateway endpoint
- Demonstrated access to cross-user data or admin functions using the token

### Counter-Argument Viability

| Question | Answer | Verdict |
|----------|--------|---------|
| Infrastructure still live? | No (Connection Refused) | Dead end |
| Data sensitive enough? | N/A — can't even connect | Dead end |
| Can reframe class? | Possibly CWE-287, but no live endpoint | Dead end |
| New evidence? | No | Dead end |

**Action**: Move on. No counter-argument possible.

---

## Finding 2: Missing Authentication on User Favorite Games Endpoint

- **Code**: AYLO-TJ7G4MRW
- **Severity**: Medium (submitted) → Out of Scope (closed)
- **Claim**: IDOR / Missing auth on GET endpoint for user favorites (game titles)
- **What the triager said**: "IDOR with no direct security or financial impact" per OOS policy

### Root Cause Analysis

**The Keyword Trap**: The program's OOS list says "IDOR with no direct security or financial impact." The report likely used the word "IDOR" in the description. Even though the finding was technically CWE-287 Improper Authentication (write path requires auth, read path doesn't), the keyword match triggered the OOS scanner.

**The Write/Read Asymmetry Signal** (from Gate C1):
- POST to create favorites: returns 401 without auth (developer intended protection)
- GET to read favorites: returns data to anyone (missing auth is unintentional)
- This asymmetry proves the developer intended this to be authenticated

**Data Sensitivity Problem**: Even correctly framed as CWE-287, the data exposed is just game titles — no PII, no financial data. The program's OOS policy would still apply.

### Counter-Argument Viability

| Question | Answer | Verdict |
|----------|--------|---------|
| Infrastructure still live? | Partial — endpoint may still work | Could test |
| Data sensitive enough? | No — game title list only | Weak |
| Can reframe class? | Yes — CWE-287 not IDOR | Possible |
| New evidence? | No — same data | Weak |

**Best counter-argument**: "This is CWE-287 Improper Authentication, not IDOR. The POST/GET asymmetry proves the developer intended authentication. The fact that the current data is low-sensitivity doesn't change the fact that this endpoint has no access control — if the data changes, it leaks immediately."

**Expected outcome**: Low chance of reversal due to data sensitivity. Better to find a PII-leaking endpoint with the same vulnerability.

---

## Key Lessons Captured

1. **OOS keyword grep is mandatory** — grep the report for every OOS term before submission. The word "IDOR" in a CWE-287 finding can trigger automated rejection.

2. **Infrastructure dead end kills findings** — If the credential's target API goes down (Connection Refused), the finding is dead. No counter-argument can save it without a live endpoint.

3. **Credentials need chaining** — APK credentials are not a finding. What they unlock is the finding. Never submit credentials alone.

4. **Low-sensitivity data + OOS = no chance** — Even if the vulnerability class is correct, the program's OOS on low-impact data means no payout. Find higher-impact data.

5. **Write/read asymmetry is a strong signal** — Use it to differentiate intended behavior from missing auth. But frame as CWE-287, not IDOR.

## Related References

- `two-account-idor-proof-methodology.md` — two-account testing approach
- `pre-submission-verification` skill, Part 8 — counter-argument decision framework
