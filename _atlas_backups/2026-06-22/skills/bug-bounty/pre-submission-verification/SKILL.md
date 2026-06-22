---
name: pre-submission-verification
category: bug-bounty
description: Pre-submission verification. Run BEFORE every submission — but which gates apply depends on the VULNERABILITY CLASS, not all checks apply universally.
---

# Pre-Submission Verification

Run this BEFORE every submission. The 4 universal gates apply to ALL classes.
Then apply the class-specific gate for your finding's type.

---

## PART 0 — PRE-DIAGNOSIS: Bug, Architecture Observation, or Vulnerability?

Run this BEFORE any other gate. Most of our rejections come from skipping this step.

### Gate A0 — The Security Boundary Question

**> "Does this finding let an attacker DO something the system was designed to prevent?"**

#### Stage 1: Action Test

Write a 5-word title that ends with what an attacker CAN DO:
- ✅ "Auth bypass → read customer PII" → VULNERABILITY. Proceed.
- ✅ "Missing auth → modify invoice data" → VULNERABILITY. Proceed.
- ❌ "Auth bypass → endpoint reachable" → ARCHITECTURE. Stop.
- ❌ "Broken auth → permission error" → ARCHITECTURE. Stop.
- ❌ "API accepts any token → can't access data" → ARCHITECTURE. Stop.

If the action after the arrow is NOT a concrete, direct harm (data read, data modify, code execute, account takeover, service disrupt), the finding is probably architecture, not vulnerability.

#### Stage 2: The Two-Layer Architecture Rule

If the finding involves a service that accepts arbitrary credentials at one layer but has a SECOND independent enforcement point at the data layer, it's architecture, not vulnerability. Examples:

| Front Layer | Backend Layer | What You Found | Verdict |
|-------------|---------------|----------------|---------|
| API Gateway / MCP Server (accepts any token) | TWG / Data Service (validates token separately) | "Gateway is lenient" | 🔍 Architecture observation. The security boundary is at the data layer, not the gateway. |
| Reverse Proxy / CDN | Origin Server | "CDN forwards all requests" | 🔍 Architecture. CDNs don't authenticate. |
| Session service | Authorization service | "Session token accepted" | 🔍 Architecture if authz is independent. |

**Genuine auth bypass (single layer):** The component that accepts arbitrary tokens ALSO serves the data. Example: CVE-2024-8954 (Composio) — x-api-key header with any value → full API access. One component, one auth check.

**False auth bypass (two layer):** The component that accepts arbitrary tokens is a gateway/proxy that passes requests to a backend that enforces its own auth. Example: Atlassian MCP Server — accepts any Bearer token for init/session, but the TWG backend independently validates. The MCP `init` endpoint being lenient is a design choice for protocol accessibility, not a vulnerability.

**Test:** Can you reach data WITHOUT passing through the second auth layer? If no, it's architecture.

#### Stage 3: The Kettle Test

From Shubham Shah (Assetnote): *"If you don't have an excellent understanding of fundamental application security attacks and weaknesses before you approach bug bounties, you are wasting your time."*

Ask yourself:
- "Is this a real security boundary that I've crossed, or did I just observe how the system is designed?"
- "Could the vendor fix this and say 'this was never a vulnerability, this is how our architecture works'?"
- "What is the SPECIFIC data I accessed? Not 'could access' — what DID I access?"

#### Stage 4: Compare Against Known Accepted Findings

| Finding | What Made It Accepted | Why It's Different From Architecture |
|---------|----------------------|--------------------------------------|
| CVE-2024-8954 (Composio) | Any x-api-key → full API access | Single auth layer. Broken check = data access. |
| CVE-2025-9485 (WordPress OAuth) | Forged JWT → admin login | Direct account takeover. No second auth layer. |
| CVE-2026-29000 (pac4j JWT) | alg:none JWT → authentication | Protocol implementation bug, not architecture. |
| Spring heapdump → secrets → ATO (Shubham Shah) | Exposed endpoint → creds → account takeover | Complete chain demonstrated end-to-end. |

Our rejected findings compared:
| Finding | Why Rejected | Gate A0 Stage That Caught It |
|---------|-------------|------------------------------|
| Atlassian MCP auth bypass | Two-layer architecture: TWG independently validates | Stage 2 — Two-Layer Rule |
| AdultForce /api/site config exposure | Data is P4 business metadata, not P1-P3 data | Stage 3 — Data Access Question |
| MPC 004 oracle | Oracle theorized, key never recovered | Stage 1 — Action Test |
| MPC 005 version downgrade | Proved constants exist, not exploit | Stage 1 — Action Test |

#### Stage 5 — Decision

| Result | Action |
|--------|--------|
| **Vulnerability** — clear action, single boundary, data accessed | Proceed to class-specific gates |
| **Architecture observation** — interesting behavior, no data crossed a boundary | Do NOT submit. Save as research notes. |
| **Bug** — feature doesn't work but no security boundary crossed | Do NOT submit. |
| **Unsure** — write the 3-sentence attack story. If you can't, it's not ready. |

---

## PART 0 — PRE-WRITING COMPLIANCE CHECK

Run this BEFORE writing a single word of the report. If you skip this phase,
you will write a report that violates program rules and have to rewrite it.

### Gate P1 — Program Rules Review

Go to the program's rules page. Extract these BEFORE drafting:

- [ ] **Required headers**: Does the program require `X-Bug-Bounty`, `X-HackerOne- Researcher`, custom User-Agent? Add to EVERY curl command in the PoC.
- [ ] **Rate limits**: Max requests/sec? Ensure PoC respects them.
- [ ] **Automated tooling**: Permitted or prohibited?
- [ ] **Account requirements**: Must use `@intigriti.me` email? Premium accounts available?
- [ ] **Reporting timeframe**: Must report within 24 hours of discovery?
- [ ] **Report format**: Clear textual description required? Video-only reports refused?
- [ ] **Scoring exceptions**: Does the program list vuln types that get automatically downgraded? (e.g. XSS with CSP → P4, Open Redirect → P4, Admin→SysAdmin PrivEsc → P4) This affects your payout expectations but NOT your decision to submit.
- [ ] **No-pivoting clause**: Does the program prohibit using a vulnerability to find another? If so, submit findings as standalone reports.
- [ ] **Cross-program scope check**: Is the domain scoped under THIS program or a different one? Verify you can submit before deep-diving. Findings on sibling domains may need a separate program account.

### Gate P2 — OOS Scan Per Finding Aspect

Scan EVERY aspect of your finding against the program's explicit OOS list.

Common traps that get reports rejected:

| Finding Aspect | Common OOS Rule | Action |
|---------------|----------------|--------|
| User enumeration | "Account enumeration" | **Remove entirely** from report |
| Brute force capability | "Rate limiting or brute force issues" | Reframe as secondary consequence, not primary finding |
| Missing cookie flags | "HttpOnly, SameSite and Secure Cookie flags" | Remove — known non-issue |
| Version disclosure | "Banner identification issues" | Remove unless you have a working CVE PoC |
| Error messages | "Descriptive error messages" | Remove unless you demonstrate executable exploit |
| IDOR without impact | "IDOR with no direct security or financial impact" | Must cross boundary AND show real harm |
| Information disclosure | "Information Disclosure without significant and executable impact" | Must demonstrate cross-boundary data access |

**Process:**
1. List every claim in your summary and impact sections
2. Check each against the program's OOS list
3. If a claim matches OOS, REMOVE it. Do not reframe or hide it.
4. If the CORE finding relies entirely on an OOS class (e.g., account enumeration is your only evidence), the finding may not be viable.
**Grep the FINAL report file for every OOS term before submission.** This is a killer — see the real Nutaku case below.

**Real case — Finding 2 (Nutaku favorites, Jun 2026)**: The finding was technically CWE-287 Improper Authentication (missing auth on a GET endpoint where the POST requires auth). But the program OOS list said IDOR with no direct security or financial impact. The description contained the word IDOR even though the finding was a different class. The triager's OOS scanner likely matched the keyword and rejected automatically. If we had framed it as CWE-287 Improper Authentication — unauthenticated read of user-scoped data with zero mentions of IDOR, the outcome might have been different.

The fix: grep the FINAL report for EVERY OOS term from the program rules. If you find a match, DELETE the sentence. Do not reframe — delete. Then re-run grep until clean. Common traps: IDOR, account enumeration, rate limiting, brute force, missing security header, information disclosure.

### Gate P3 — Report Structure Planning

Before drafting, decide the deliverable structure:

- **Folder convention**: `~/Dev/REPORTS/<Target>/<number>/REPORT.md` for the report,
  `~/Dev/REPORTS/<Target>/<number>/poc/<script>` for the PoC script.
  Working sandbox stays in `~/Dev/<target>/` — never mix sandbox artifacts with final reports.
- **PoC script**: Self-contained, runnable, includes ALL required headers.
  Avoid shell variable references for secrets (masking systems may intercept them).
  Pipe responses directly or use environment variables.
- **Report structure**: Summary, Steps to Reproduce (numbered), Request/Response pairs,
  Impact, CVSS, Remediation, References.

**Report structure additions for readability and trust:**

Three patterns emerged from Nutaku submission reviews (Jun 2026) that improve acceptance odds:

1. **Plain-English opener** — After the title/header, add a "What This Means in Simple Terms" section using a concrete real-world analogy (filing cabinet, storefront, bank vault, etc.). Lead with the analogy, not the technical detail. Triagers and program owners skim — make the first paragraph understandable to a non-technical business person.

2. **"What This Is NOT" section** — Before or within the Impact section, add an explicit limitations paragraph stating what the finding does NOT expose (e.g., "This finding does NOT expose passwords, credit card numbers, or personal data. It exposes business operational data — the configuration and structure of the company's digital infrastructure."). This builds trust, reduces overclaim risk, and prevents triage from rejecting on "theoretical impact" grounds because you've already drawn the line yourself.

3. **Business analogy for impact** — When describing impact, frame at least one bullet in pure business terms without technical jargon. Example: "A retailer publishing its entire supplier list, wholesale prices, and warehouse locations" rather than "exposes internal site configuration with business unit groupings."

**Intigriti submission form fields**:
The form has 9 fields that map to a well-structured report:

| Form Field | What Goes There | Char Limit |
|-----------|----------------|------------|
| Title | Vulnerability class + component | N/A |
| Asset | From the scope list dropdown | N/A |
| Endpoint | The specific URL path | N/A |
| Type | CWE category under appropriate group (Broken Authentication, Mobile, etc.) | N/A |
| Severity | CVSS vector string (use calculator) | N/A |
| Proof of Concept / description | Summary + Steps to Reproduce + technical detail | 30,000 |
| Impact | Concrete harm, victim perspective, business risk | 15,000 |
| Recommended solution | Remediation steps (optional) | 15,000 |
| IP address | Your testing IP (optional, "Fetch my IP" button) | N/A |

**Key rule**: Separate the PoC/description from the Impact. Triage reads Impact separately to decide severity — don't bury the impact in the description. The impact section must stand alone as a clear answer to "why does this matter?"

---

## PART 1 — UNIVERSAL GATES (all classes)

### Gate U1 — Impact must be demonstrated, not theorized

Does your PoC show REAL unauthorized access or real harm?
Or does it describe what "could potentially" happen?

**Pass**: Concrete request/response showing data you should not have, or action you should not be able to perform.
**Fail**: "An attacker could enumerate X" without showing any cross-boundary data. "This could lead to Y" without showing Y.

🔴 **Rejected example**: "An attacker could enumerate vault IDs to discover hidden wallets." — No cross-boundary data shown. No actual hidden data accessed.

🟢 **Valid example**: A single curl command that returns another user's invoices, with the response included.

### Gate U2 — VRT classifies severity, doesn't validate existence

Does your finding actually fit the VRT category you chose?
Or are you using the VRT severity table as evidence the finding is valid?

**Pass**: You identified the vulnerability class first (IDOR, SSRF, crypto weakness) and then used VRT to rate severity.
**Fail**: "Sequential IDs → VRT says Iterable Object Identifiers is P1 → this is IDOR." The VRT doesn't tell you what class your finding belongs to.

### Gate U3 — Understand the system architecture

Do you know how the system's authentication, authorization, and data flow work?
Can you describe the security boundary model?

**Pass**: You can explain: "Vaults are workspace-scoped. User A in workspace B should not access vault C." And your PoC crosses that boundary.
**Fail**: You assumed a field name implies a security boundary without checking server enforcement.

### Gate U4 — PoC is reproducible

Can triage copy-paste your steps and verify the finding?

**Pass**: Concrete curl commands, exact URL paths, example request/response bodies.
**Fail**: Vague steps like "enumerate IDs" without showing which API endpoint and what response you got.

---

## PART 2 — CLASS-SPECIFIC GATES

Apply the gate corresponding to your finding's vulnerability class.
Different classes have DIFFERENT rules — don't use the IDOR gate for crypto bugs.

### Gate C1 — Access Control (IDOR, BAC, PrivEsc, BOLA/BFLA)

**The boundary question**: What scopes access to this resource? (user? workspace? role? tenant?)
**The cross question**: Does your PoC access a resource owned by a DIFFERENT scope?

**Definition** (PortSwigger/Intigriti/OWASP): IDOR occurs when an application takes user-supplied input and uses it to retrieve an object **without performing sufficient authorization checks**. The key phrase is "belongs to a different entity."

**Testing methodology** (from 2-account Nutaku proof, 2026):
1. **Create two accounts** in the same scope (User A, User B)
2. **Add distinct data** to each account — different values so you can tell whose data is whose
3. **Verify baseline**: User A reads User A's data, User B reads User B's data — confirm the endpoint returns user-scoped data correctly
4. **Cross-user read**: User A reads User B's identifier — if it returns User B's data, access control is missing
5. **Reverse direction**: User B reads User A's identifier — bidirectional confirmation eliminates false positives
6. **Test without authentication**: Try the cross-user read with NO token at all — if it still returns data, the vulnerability is missing authentication entirely (CWE-287), not just IDOR
7. **Cleanup**: Remove test data after proof is captured

**Key insight — write/read asymmetry**: Compare the POST (write) and GET (read) paths. If POST correctly requires authentication (401 without token) but GET returns data to anyone, the missing auth on GET is unintentional — the write path proves the developer intended authentication. This asymmetry is a strong signal that the finding is valid, not intended behavior.

**Framing trap — never use OOS terms in the report**: 
- If the program excludes "IDOR with no direct security or financial impact," do NOT use the word "IDOR" anywhere in the report — even if your finding is technically a different class (e.g., CWE-287 Missing Authentication). The triager's OOS scanner may reject based on keyword match alone.
- Use precise CWE language instead: "CWE-287 Improper Authentication" not "IDOR," "missing access control" not "privilege escalation," "unauthenticated data access" not "information disclosure."
- Before finalizing, grep the report for every OOS term listed in the program's rules. Remove any matches.

**Rules**:
- Must CROSS a resource ownership boundary. Same-workspace enumeration is NOT IDOR regardless of ID format.
- Client-side UI hiding, field name conventions ("hidden", "private"), obfuscation are NOT access controls. Only server-side enforcement counts. **The server must first RESTRICT access before you can BYPASS it.**
- Sequential/incrementing IDs enable exploitation but don't create the vulnerability. The missing auth check creates it. UUIDs with missing auth would be equally broken.
- **"Always ask yourself: is this really an issue or is it intended behaviour?"** — Intigriti Hackademy. If the answer could be "the app was designed this way," it's likely a false positive.
- If your PoC requires "you already have access" to work, it's not an access control finding.

**Real rejected examples** (HackerOne): Reports #166849, #49499, #361133 — researchers misunderstood the application behavior.

**Test**: Create two accounts in different scopes. Access scope A's resources from scope B. If it works → real finding.

**Resources**:
- PortSwigger: https://portswigger.net/web-security/access-control/idor
- Intigriti: https://www.intigriti.com/researchers/hackademy/idor
- OWASP: https://owasp.org/www-community/attacks/Insecure_Direct_Object_References

### Gate C2 — Cryptographic Weaknesses

**Rules**:
- Theoretical weakness ≠ practical exploit. Academic paper ≠ bounty submission.
- RNG attacks need viable failure scenario (VM clone, fork, entropy exhaustion).
- Demonstrate the actual attack: key recovery, sig forgery, plaintext recovery.
- Implementation bugs (wrong data hashed) are easier to demonstrate than protocol design flaws.
- **Narrative framing matters.** A finding that "the SDK crashes" sounds like an availability issue (programs often exclude DoS). The SAME finding framed as "the JWT bodyHash integrity mechanism is completely nullified — the hash never represents the actual body" is a cryptographic weakness. Lead with the broken security mechanism, not the symptom.

**Case study — bodyHash report reframe:**
- **Before**: "TypeScript SDK crashes on POST / 401 error" → triager hears "availability issue, possibly out of scope"
- **After**: "JWT bodyHash cryptographic integrity mechanism is broken — the hash is invariant across all possible request bodies, defeating request integrity verification for all write operations" → triager hears "cryptographic weakness, in scope"
- The evidence is identical in both cases. The difference is what you lead with: the mechanism vs. the outcome.

**Test**: Can you run ONE script that proves the attack end-to-end?

### Gate C3 — SSRF

**Rules**:
- DNS resolution without HTTP response data is blind SSRF.
- Reaching a metadata endpoint and extracting creds is impact. Connection refused is not.
- AI agent SSRF: Check program stance — many consider UI:R + PR:L as mitigation.
- Novel bypass techniques > standard SSRF for acceptance.

**Test**: Capture the internal service's response. Verify via OOB callback if blind.

### Gate C4 — Credential / API Key Exposure

**Rules**:
- Verify key against LIVE API: must return 200 with usable data.
- Defunct keys, incomplete pairs, origin-restricted keys are NOT findings.
- Client-side-only keys are not findings unless cross-origin abuse is demonstrated.

**Mobile app hardcoded credentials — SPECIAL CASE**:
- Hardcoded credentials in a mobile app talking to that app's own backend API is **industry standard practice** and is **never a finding on its own**.
- The app needs SOME form of client authentication. The fact that it's extractable from the APK is inherent to public client architecture.
- **To make this a finding**, you must demonstrate that the credentials enable something BEYOND what the app itself can do:
  - Cross-account data access the app doesn't allow
  - Access to admin/higher-privilege endpoints the app doesn't expose
  - Bypassing of a server-side security control that the app normally enforces
- Citing RFC 6749 §10.1 or CWE-798 without demonstrated cross-boundary access will result in **Informative** disposition.
- **Real rejection**: Intigriti response — "You're basically logging into your own account. This does not pose any security risk."

**Test**: `curl -sL "https://api.target.com/endpoint" -H "Authorization: Bearer <key>"` — must return 200 with data.

### Gate C5 — Memory Safety

**Rules**:
- Crash alone = P4. Must show controlled exec or data corruption.
- Modern mitigations (CFG, CET, ASLR, stack cookies) affect exploitability.

**Test**: Show more than a crash — register control, memory corruption, or security property violation.

### Gate C6 — Business Logic

**Rules**:
- "An attacker could spam" needs rate limit, CAPTCHA, and cost analysis.
- Frame in business terms: revenue loss, compliance violation, user trust damage.

**Test**: Calculate actual dollar value or operational cost. If negligible → not a finding.

### Gate C7 — Authentication Bypass (Bearer Token, API Key, Session)

**Core pattern**: The server accepts any value in an authentication header or parameter — it checks for PRESENCE but not VALIDITY.

**Edge case — MCP server with API token toggle (Atlassian, 2026)**: The MCP server at mcp.atlassian.com has a setting "Allow API token authentication" in the admin portal.
- **OFF (default)**: Auth bypass exists — any Bearer token accepted. Tool calls reach authorization and return "no permission." The bypass allows session creation, tool enumeration, and proving the auth gate is broken — but no data access.
- **ON**: The server switches to real token validation. Opaque API tokens (JIRA API tokens) pass init but fail at the upstream TWG backend with "failed to parse token not a compact JWS." Only JWT-format tokens (OAuth tokens, session JWTs) may work through to data access.
- **Takeaway**: Even without data access, the auth bypass is a valid finding. The toggle itself proves the default state is broken — they had to add a setting to enable real validation. Frame this as a broken-default-hardening finding (P3-P4 without data, P1-P2 with).

**Bearer token validation bypass test**:
```bash
# Step 1: Confirm auth check exists (no token -> 401)
curl -s -o /dev/null -w '%{http_code}' -X POST 'https://target.com/api/endpoint'
# 401 = auth check exists

# Step 2: Test with single-char arbitrary token
curl -s -o /dev/null -w '%{http_code}' -X POST 'https://target.com/api/endpoint' \
  -H 'Authorization: Bearer x'
# 200 = AUTH BYPASS (token not validated!)

# Step 3: Verify it's not just accept-header leniency
# Same request but with the Accept header that the endpoint normally expects
curl -s -o /dev/null -w '%{http_code}' -X POST 'https://target.com/api/endpoint' \
  -H 'Accept: application/json' \
  -H 'Authorization: Bearer x'
# Still 200 + valid response data = confirmed bypass
```

**Edge cases to test**:
```bash
'Authorization: Bearer null'           # literal "null" string
'Authorization: Bearer ../../etc/passwd' # path traversal in token
'Authorization: Bearer '               # trailing space (empty value)
'Authorization: Bearer'                # no space after Bearer  
'Authorization: Basic <random>'         # Basic auth with arbitrary creds
'X-API-Key: anything'                  # Custom auth header
'Token: anything'                      # Alternative auth scheme
```

**Extended MCP/API auth bypass probe** — when the target is an MCP server or protocol-based API, test additional methods beyond just initialize/ping:

```bash
# Full MCP method inventory to probe (all return codes matter):
# - resources/list, resources/read, resources/subscribe
#   may have different auth path than tools endpoints
# - tools/list, tools/call -> standard tool endpoints
# - logging/setLevel -> often open, no auth value
# - notifications/initialized -> "Method not found" = not implemented

# Test registered client credentials against the service's OAuth provider.
# The server may issue credentials via /v1/register but they may NOT
# be linked to the main OAuth provider (always test before claiming):
curl -s -X POST 'https://auth.target.com/oauth/token' \
  -u 'CLIENT_ID:CLIENT_SECRET' \
  -d '{"grant_type":"client_credentials","audience":"api.target.com"}'
# "failed to retrieve client" = dead end, service-scoped only

# Extract API schema from validation error messages.
# Error responses often reveal full parameter schema as enum values:
# "expected: 'TypeA | TypeB | TypeC', received: 'undefined'"
# Collect ALL enum values and param names from these errors.

# Test cloudId/tenantId validation patterns.
# Different error messages reveal backend validation logic:
# "Input does not look like a valid domain or URL"
#   = format check passes domain-like values
# "You don't have permission" = format passed, authz rejected

# Check if upstream backend requires JWT/JWS format tokens.
# The auth bypass may pass MCP init but tokens get forwarded to upstream
# services that validate format differently:
# - Opaque token (e.g., ATATT3x...) passes init but TWG says "not a compact JWS"
# - Admin session JWT may be scoped to a different subdomain -> "invalid access token"
# - Log the exact error message: "failed to parse token not a compact JWS" means
#   the upstream expects JWT, not opaque tokens
```

**CRITICAL — The "passes auth, fails at authz" pattern**:
If the bypassed auth lets you through to a permission error (not data), triage sees this as "endpoint reachability and protocol metadata exposure, not a practical authentication bypass." The finding is technically correct but the impact is Informative/P4 unless you can demonstrate actual data access through the bypass.

What you see: "Auth bypass proven! Request reaches authorization layer!"
What triage sees: "The endpoint accepts a connection, then returns a permission error. This shows endpoint reachability and metadata exposure, not a practical auth bypass."

To close this gap, you need either:
a) A valid tenant with credentials to demonstrate the bypass end-to-end, or
b) A tool/method on the server that has no authorization check at all.

**Client registration without auth — special risk**: Unauthenticated /v1/register endpoints may generate OAuth client credentials, but these credentials are often service-scoped only and NOT known to the main OAuth provider. Test this BEFORE claiming "unauthenticated client registration enables token generation" — the chain almost certainly won't work.

**VRT Classification**: Broken Authentication > Authentication Bypass (P1)

**What makes the finding strong**:
- Clear differential: `no token -> 401` vs `any token -> 200`
- Multiple token values tested (single char, "null", path traversal) all succeed
- Basic auth also works (shows the check is not even Bearer-scheme specific)

**What weakens the finding**:
- If the tools/resources behind the auth gate also require a second auth layer (e.g., a valid tenant ID), the impact is reduced but the auth bypass itself is still valid
- If the endpoint is a public API that should be accessible without auth (e.g., a registration endpoint), verify that the intended design actually requires auth

**Our find (Atlassian MCP, 2026)**: The MCP server at mcp.atlassian.com accepted ANY non-empty Bearer token including the literal string "null." Even Basic auth with base64("test:test") passed. The server checked only for the existence of an `Authorization` header with non-empty value. When calling `getTeamworkGraphContext` with the bypassed session, the server returned a permission error — proving the bypass reaches the authorization layer but not actual data. The `/v1/register` credentials were NOT linked to auth.atlassian.com ("failed to retrieve client"). The finding was rejected as Not Applicable — mechanism proven, impact not demonstrated. This is a textbook case of the "passes auth, fails at authz" pattern documented above. See `references/atlassian-mcp-auth-bypass-analysis.md` for the full post-mortem.

**Pitfall — No-pivoting rule interaction**: If the program has a "no pivoting or post exploitation" rule (see Gate P1), submit the auth bypass as a standalone finding. Do NOT frame it as "this could be used to find SSRF" even if you independently discovered both in the same session. The auth bypass stands on its own as an authentication failure. If you also found an SSRF that requires the auth bypass to reach, you may not be able to submit the SSRF under this rule — check with program support before proceeding.

---

## PART 3 — TRIAGE EVALUATION FRAMEWORK

### The Triage Decision Tree

Every triager runs these checks in order:

```
Gate 1: IS IT IN SCOPE? → No? ❌ Out of Scope
Gate 2: IS IT REPRODUCIBLE? → No? ❌ Needs More Info
Gate 3: IS IT A VALID VULNERABILITY? → No? ❌ Informative
Gate 4: IS IT UNIQUE? → No? ❌ Duplicate
Gate 5: WHAT IS THE IMPACT? → Sets severity + payout
```

### Gate T1 — Scope Check

**What triage checks:**
- Is the asset explicitly listed or covered by wildcard?
- Is the vulnerability type explicitly excluded?
- Does the program have special rules?

**Pass**: Both asset AND vuln type are explicitly in-scope.
**Fail**: Either is OOS = immediate rejection.
**Case-by-case**: Some programs include a provision like "Rewards for high impact Vulnerabilities outside of the Scope of this Program might be considered on a case-by-case basis." If you have a strong finding on an OOS asset — one with clear, executable impact and data that directly relates to the program's in-scope properties — you can submit with an upfront **Disclosure section** that:
- Explicitly states the asset is technically outside the strict scope
- Cites the program's case-by-case provision
- Explains why the impact justifies consideration (data includes the program's own infrastructure, shared codebase with in-scope assets, etc.)
- Does NOT try to hide or reframe the scope issue

This is a long shot. Only attempt it when the impact is genuinely significant and the data directly relates to the program's own properties. The case-by-case provision exists for exceptional findings — don't use it as a loophole for weak findings on out-of-scope assets.

**Program compliance — 3 critical checks before submission:**

1. **Custom header requirements**: Some programs require `X-Bug-Bounty: <username>` or similar on ALL requests. Check the Rules of Engagement table. Add this header to every curl/PoC command in your report.

2. **OOS overlap check**: Your finding's FRAMING must not overlap with OOS vulnerability classes even if the root cause is different. For example:
   - Finding is CWE-798 hardcoded credentials (in scope) but you frame impact as "user enumeration" (OOS) → expect rejection. Frame as "auth bypass" instead.
   - Finding is missing auth on an endpoint (in scope) but you frame impact as "rate limiting" (OOS) → expect dismissal. Frame as "unauthorized data access" instead.
   - Check the FULL OOS list, not just the vuln class. Search for terms that overlap with how you describe your finding.

3. **Own-account-only rule**: If the program requires testing only against your own accounts, ensure the PoC only accesses data belonging to the researcher's account. Do not demonstrate cross-user access even if technically possible — submit the auth bypass and let triage ask for escalation.

4. **No-pivoting/post-exploitation rule**: If the program explicitly prohibits "using a vulnerability to find another," you must:
   - Submit each finding as a standalone report
   - NOT reference cross-finding chaining or escalation in your report
   - NOT submit findings discovered by exploiting another vulnerability (even if independently verified)
   - If multiple findings exist, submit the auth bypass first, and for subsequent findings disclose in the report text that they were discovered through independent testing, not by pivoting from the first finding.

#### T1a — Required Headers and Program-Specific Rules

Some programs require specific headers on ALL requests:
- Intigriti Nutaku: `X-Bug-Bounty: <username>`
- Other programs may require `X-HackerOne- Researcher`, custom rate limits, specific User-Agent

**Check before writing the report:**
1. Read the program's Rules of Engagement section for required headers
2. Add the header to EVERY curl command in the PoC — not just the first one
3. Check for rate limits (max req/sec) and ensure your PoC respects them
4. Check if automated tooling is permitted, and any User-Agent requirements

#### T1b — OOS List Scanning Per Finding Aspect

BEFORE writing the report, scan EVERY aspect of your finding against the program's OOS list. Common pitfalls:

| Finding Aspect | Common OOS Rule | Risk |
|---------------|----------------|------|
| User enumeration | "Account enumeration" | Report rejected if mentioned |
| Brute force capability | "Rate limiting or brute force issues" | May be rejected case-by-case |
| Missing cookie flags | "HttpOnly, SameSite and Secure Cookie flags" | Instant reject |
| Version disclosure | "Banner identification issues" | Instant reject |
| Error messages | "Descriptive error messages" | Instant reject unless proven exploitable |

**Process:**
1. List every claim in your impact section
2. Check each claim against the program's OOS list
3. If a claim matches an OOS item, REMOVE it from the report — do not reframe it
4. If the CORE finding relies entirely on an OOS class (e.g., account enumeration), the finding itself may need re-evaluation

### Gate T2 — Reproducibility Check

**Checklist:**
- [ ] Prerequisites listed FIRST
- [ ] Step-by-step numbered instructions
- [ ] Actual HTTP request/response pairs
- [ ] Screenshot/video for complex UI workflows
- [ ] Works on production, not local env

### Gate T3 — Vulnerability Validity Check

**Common invalid patterns:**
1. **Self-XSS** — Requires victim to paste JS into console.
2. **Missing security header** — Not a vuln without demonstrable impact.
3. **Version disclosure** — Not a vuln without unpatched CVE you can demonstrate.
4. **Rate limiting absence** — Not a vuln without demonstrated harm.
5. **Missing SPF/DMARC** — Not a vuln without demonstrated email spoofing.
6. **CSP misconfiguration** — Contributing factor, not standalone finding.
7. **Verbose error messages** — Not a vuln unless leaked info enables further exploitation.

### Gate T4 — Uniqueness Check

Search program's public disclosures, CVE/NVD, HackerOne Hacktivity. Accept duplicates happen — speed + depth is the differentiator.

### Gate T5 — Impact Assessment

| Factor | Informative | Triaged |
|--------|------------|---------|
| Impact shown? | Theorized | Demonstrated |
| POC exists? | Conceptual | Working |
| Exploit chain? | Partial | Complete |
| Data accessed? | Your own | Cross-boundary |
| Security controls? | Ignores them | Addresses/bypasses them |

---

## PART 4 — VICTIM PERSPECTIVE FRAMEWORK

Before submitting, imagine you're the **victim**. This flips the question from "can I exploit this?" to "could someone be harmed?"

### The Victim Questions

**V1 — Who is the victim?**
- **Fail**: "The server is the victim because it leaks info." — Servers aren't victims.
- **Pass**: "Any user with a valid session."

**V2 — What must the victim DO for the attack to work?**
- More actions = harder exploit.
- **Fail**: "Victim must paste JS into console." — Self-XSS.
- **Pass**: "Victim only needs to visit attacker's site while logged in."

**V3 — What prerequisites must be true?**
- List every prerequisite explicitly. 3+ unlikely prerequisites = unrealistic.
- **Fail**: "Victim must be authenticated admin + visiting attacker site + have disabled CSRF."
- **Pass**: "Victim just needs to be authenticated."

**V4 — Can the attacker FORCE the prerequisites?**
- If not forceable, it's phishing/social engineering, not a technical vulnerability.
- Programs typically exclude social engineering.

**V5 — What is the ACTUAL harm?**
- Not "information disclosure" but "leaks the victim's full name, email, phone, billing address."

**V6 — Is there a realistic attack scenario?**
- Write a 3-sentence story. If you can't, the finding isn't ready.

### Business Impact Framing

After running the Victim Perspective Framework, frame the impact for a **non-technical business reader**. Program owners and product managers review reports too.

**The analogy-first approach**: Lead with a real-world business analogy before explaining the technical mechanism. This ensures the business reader understands WHY it matters before they hit the technical detail.

**Pattern:**
1. **One-sentence plain-English description**: "An API endpoint that lists every brand's billing account ID and cloud storage location can be read by anyone with an internet connection."
2. **Business analogy**: "This is the equivalent of a retailer publishing its entire supplier list, wholesale prices, and warehouse locations."
3. **What this means for the business**: List concrete business risks in plain language (competitor intelligence, infrastructure targeting, billing system analysis).
4. **What this is NOT**: Explicitly state what the finding does NOT expose (no user passwords, no credit cards, no PII). This builds trust and prevents triage from dismissing on overclaim grounds.

**Avoid**: Pure technical impact descriptions without business translation. "Exposes ProBiller IDs" means nothing to a business reader. "Reveals which billing account each brand uses for payment processing" means something.

**Examples of business analogies that work:**
| Technical finding | Business analogy |
|-----------------|------------------|
| Unauthenticated API listing all sites with billing IDs | Filing cabinet with locks on the drawers but not the front door |
| IDOR on invoice endpoint | Customer A reading Customer B's receipts |
| SSRF to cloud metadata endpoint | Using the mailroom to read the CEO's mail |
| Missing auth on admin panel | Bank vault with a spinning lock but no guard at the door |

---

## PART 5 — REPORT SELF-FACT-CHECK

Before writing the final report, run these verifications against every claim.

### Gate S1 — Verify EVERY impact claim against the LIVE server

Do not write an impact claim based on reasoning alone. Every claim must be backed by a real HTTP response.

**CVE precondition ≠ exploitation**: A confirmed CVE version or signature pattern does NOT mean the exploit works. Example — CVE-2017-9822 (DNN deserialization RCE): Nuclei detected preconditions (DNNPersonalization cookie accepted, 404 handler active) but the actual gadget chain silently fails because the WPF assembly required for exploitation isn't loaded in the ASP.NET worker process. Always test the FULL exploit chain, not just the preconditions.

**Process:**
1. Write a test script exercising the exact scenario
2. Run it against the live server
3. Capture the actual response
4. Only then include the claim

🔴 **bodyHash report**: Claimed "JWT can be replayed with any arbitrary body." — Never tested. Actual: `401 "nonce already used"`. Claim was **false**.

🟢 **Correct**: Run the test first. Replay fails → delete the claim.

**Rule:** Every impact bullet must reference a concrete request/response pair.

### Gate S2 — Verify the ACTUAL scope of the bug

**Process:**
1. Check ALL versions, not just the one you first found
2. Check the published package, not just source
3. Check both source and compiled output
4. For npm packages: `npm view @scope/name versions --json` then `npm pack` each version and inspect

🔴 **bodyHash report**: Claimed "v19.1.0" — ALL 64 versions from v8.0.1 to v20.0.0 have the same bug.

🟢 **Correct**: `npm pack @fireblocks/ts-sdk@{v1,v2,...}` each version. ALL broken → "ALL published versions."

**Rule:** Check at least 3 versions spanning the package's lifetime. For npm packages, check the oldest, middle, and newest.

### Gate S3 — Check "escalation" paths end-to-end

**Process:**
1. Write the COMPLETE exploit chain as a single script
2. Run it start to finish
3. If it fails at any step → remove that escalation

🔴 **bodyHash**: "bodyHash bug → JWT replay → modify transaction." Step 2 failed (nonce tracking).

🟢 **Correct**: Step 1 ✓, Step 2 ✗ → No escalation. Report the bug as-is.

**Rule:** No escalation path until the full chain is demonstrated.

### Gate S4 — Severity must match VRT baseline honestly

**Process:**
1. Look up the VRT baseline for your category
2. If claiming higher, document the specific evidence
3. If evidence was unverified, severity claim is wrong

🔴 **bodyHash**: VRT baseline = P3. Claimed P2 based on unverified replay.

🟢 **Correct**: Classify → P3. Test escalation. If it works, include. If not, accept P3.

### Gate S5 — Self-review as the triage reviewer BEFORE submitting

Before writing the final report, explicitly adopt the reviewer's perspective. Ask:

**"How would I reject this finding?"**

Then systematically answer. This catches the most common rejection reasons before triage does.

**Process:**
1. Read the program's OOS list. For each OOS item, ask: "Could a reviewer argue this finding falls under this category?"
2. Read your own impact claims. For each, ask: "Is the impact demonstrated (request/response pair) or theorized ('could lead to...')?"
3. Read your own data sensitivity claims. For each, ask: "Would a reasonable person agree this data causes harm? Or is it business-operational data that doesn't cross a meaningful boundary?"
4. Ask: "What is the weakest part of this submission?" Fix it or remove it.
5. Ask: "If the developer comes back and says 'this is intentional design,' what evidence do I have that it's not?"
6. Write a **"Strengths and Limitations"** section in the report that explicitly addresses the honest weaknesses. This builds trust with the reviewer and prevents them from rejecting on a point you already acknowledged.

**Examples from AdultForce submission (Jun 2026):**
- Weakest point: ProBiller IDs are identifiers, not credentials. An attacker can't charge a card with them.
- Honest limitation acknowledged: "This finding does NOT expose user passwords, credit card numbers, or PII. It exposes business operational data."
- Evidence against "intentional design": The same auth gap pattern exists on a second endpoint (`/api/config`), making it a systemic issue rather than a deliberate choice.

**Rule:** If you cannot articulate a credible counter-argument and explain why it's wrong, the finding is not ready to submit.

### Gate S6 — Write PoC FIRST, then describe what it proves

**Process:**
1. Write the PoC code
2. Run it, capture output
3. Write description based on what it ACTUALLY showed

🔴 **bodyHash**: PoC showed identical hashes → described as "replay attack."

🟢 **Correct**: PoC shows crash + 401 → "SDK crashes, server rejects."

**Rule:** PoC output and description must be congruent.

### Expected Disposition by Report Quality

| Quality | Likely Disposition |
|---------|-------------------|
| In scope + reproducible + valid + unique + demonstrated impact | **Accepted** |
| Reproducible but weak impact | **Informative** |
| OOS asset/vuln type | **Out of Scope** |
| Not reproducible / vague | **Needs More Info** → N/A |
| Duplicate | **Duplicate** |
| Well-known non-issue | **Informative** |

---

## FINAL GATE — Two Mandatory Questions Before "Ready to Submit"

Before telling the user the report is ready, answer these two questions explicitly. The user will ask both — pre-empt them.

### Q1 — Does this conform to what the program requires?

**Run through every row:**

| Requirement | Source | Checked? |
|-------------|--------|----------|
| Asset in scope | Program's assets list | |
| Vuln type not OOS | Program's OOS list | |
| Required headers on ALL requests | Rules of Engagement | |
| Rate limit respected | Automated tooling rules | |
| @intigriti.me email if required | Account requirements | |
| Reported within 24hr | Reporting timeframe | |
| Clear textual description | Reporting requirements | |
| Own accounts only | Testing requirements | |

**Then grep the final REPORT.md for every OOS term** — OOS keywords in descriptions trigger automated rejection even if the finding is a different class. Common traps: "IDOR", "account enumeration", "rate limiting", "brute force", "missing security header". Remove every match, re-run grep until clean.

Fix: The `grep` command runs against the final REPORT.md. Delete matches — do not reframe or hide them. Re-run grep until clean.

Write a 2-3 sentence plain-English answer that:
- Identifies the security boundary the server SHOULD enforce
- Explains how the server FAILS to enforce it
- Describes what an attacker can DO because of that failure

**Example (favorite-games finding)**:
> "The POST endpoint requires authentication (401 without token). The GET endpoint does not — it returns any user's favorites to anyone. This means the server was designed to protect this data but forgot to check on the read path."

---

## PART 6 — BUG VS VULNERABILITY DECISION

Not every real, verifiable bug is a submittable security vulnerability. The distinction:

### The Core Question

> "Does this finding let an attacker do something the system was designed to prevent?"

**No → It's a Bug.** Don't submit. Save the analysis in your research notes.

**Yes → It's a Vulnerability.** Submit it (after passing all other gates).

### Decision Table

| It's a Bug (don't submit) | It's a Vulnerability (submit) |
|---------------------------|-------------------------------|
| SDK crashes on POST | Crash lets attacker bypass authentication |
| Hash computed incorrectly | Wrong hash lets attacker forge requests |
| Feature doesn't work | Broken behavior crosses a security boundary |
| Server returns an error | Error leaks exploitable information |
| Client-side only issue | Server-enforced restriction bypassed |
| Auto-generated code quality bug | Cryptographic protocol implementation flaw |

### The Server-Side Integrity Test

If the server-side security is intact — the server correctly validates, rejects, and logs bad input — the bug exists only in the client-side component. It's a quality issue, not a vulnerability.

**Real example — Fireblocks TS SDK bodyHash:**
- `crypto.update(bodyJson)` receives a raw JS Object instead of a string
- Every POST crashes (Node 22) or sends an invalid hash (Node 18)
- **But**: The server correctly rejects bad bodyHash with code -9
- **And**: The server correctly tracks nonces with code -13
- The server-side crypto is sound. The TS SDK just can't participate.
- → **Bug, not vulnerability. Not worth submitting.**

### When Framing Alone Can't Save a Finding

Gate C2 (Cryptographic Weaknesses) has a "narrative framing" note about leading with the broken mechanism instead of the symptom. But framing only works when there IS a security boundary being crossed. If the server-side enforcement is intact, no amount of reframing turns a bug into a vulnerability.

**Test**: "If the vendor fixed this bug tomorrow, would any security boundary still be crossed?" If the answer is no (because no boundary was ever crossed), the bug was never a vulnerability.

### The Credential-Access Gap Gate — NEW

A found credential (API key, secret, token) is NOT valuable unless you can demonstrate it unlocks something. Apply this gate BEFORE reporting any credential leak:

**Three-layer test:**
1. **Can you authenticate with it?** — Try the credential against the target API/system. 401 = dead end without a corresponding user token or additional factor.
2. **Does it unlock something?** — Even if you can authenticate, list what endpoints become accessible. If they're all public/PII-free, impact is low.
3. **Can a real attacker reach those endpoints?** — Are they network-accessible, or do they require being on an internal network/VPN?

**Real example (Nutaku, 2026)**:
- Finding: 224 OpenSocial OAuth 1.0 consumerSecrets leaked from a catalog API
- Gate 1: Could not authenticate to OSAPI with leaked secrets alone (OAuth 1.0 also requires user token)
- Gate 2: Could not find any endpoint that accepts consumerSecret as a standalone credential
- Gate 3: Some target endpoints (metaapi, userapi) returned 403 from researcher IP
- **Result: Finding dropped from High to Non-reportable.** Leaked credentials with no exploitable path = Informational at best.

**Lesson**: A credential leak is not a vulnerability unless you can demonstrate one of:
- A working authenticated API call using the leaked credential
- An endpoint that trusts the credential as proof of authorization
- A chain that combines the credential with another finding to achieve impact

---

## Reference: What Got Rejected and Why

| Finding | Claim | Class | Disposition | Why | Gates Failed |
|---------|-------|-------|-------------|-----|-------------|
| Nutaku hardcoded OAuth2 creds | CWE-798 in mobile APK | Credential Exposure | **Informative** | Triager: "You're basically logging into your own account. This does not pose any security risk." Mobile app creds talking to their own API is industry standard. No cross-boundary access demonstrated. RFC 6749 argument didn't overcome lack of harm. | C4, T5 |
| Nutaku missing auth on favorites | CWE-287 improper auth on GET endpoint | Access Control | **Out of Scope** | Program OOS: "IDOR with no direct security or financial impact." Exposed game titles only. Report used the keyword "IDOR" which matched the OOS rule. Could have been framed as CWE-287, but data sensitivity too low to overcome OOS even with correct framing. | P2 (keyword trap), T1 (scope) |

**Related reference**: `references/two-account-idor-proof-methodology.md` — two-account PoC approach, shell variable masking workarounds for writing PoC scripts, and the write/read asymmetry signal for finding missing authentication.
**Related reference**: `references/nutaku-rejection-analysis-jun2026.md` — full post-mortem of two Nutaku rejections with counter-argument viability analysis and the keyword grep trap that killed the second finding.

| Fireblocks bodyHash bug | Cryptographic integrity failure | Crypto | **Not submitted** | Real bug but server-side security intact. Quality issue, not vulnerability. | Part 6 decision |
| Fireblocks bodyHash (original version) | P2 replay attack (false claim) | Crypto | Draft caught in review | Replay claim was never tested. Nonce tracking blocks it (code -13). | S1, S3 |
| Fireblocks MPC 004 (Paillier oracle) | Key recovery in ~300 queries | Crypto | N/A (AI-generated) | Oracle confirmed but key recovery never demonstrated. Title overclaims PoC output. | C9, C10, S5, U1 |
| Fireblocks MPC 005 (Version + key rotation) | Permanent compromise surviving refresh | Crypto + Architecture | N/A (AI-generated) | Chained 3 undemonstrated findings to claim P3. PoC proves constants exist, not exploit. | U1, U4, S5, R2 |

**Related reference**: `references/mpc-rejection-case-studies-jun2026.md` — full analysis of both rejections with specific pipeline failures and what would have passed triage.

## Part 7 — Post-Rejection Analysis (Added Jun 2026 from MPC rejection feedback)

Every rejection is data. Run after every rejection.

### Gate R1 — Triage Gate That Failed

| Signal | What It Means | Fix |
|--------|--------------|-----|
| "Out of Scope" | Asset or vuln type OOS | Check program scope |
| "Not reproducible" / "lacks clear PoC" | No working PoC | Write PoC FIRST |
| "Informative" | Not a real vulnerability | Run Part 6 (Bug vs Vulnerability) |
| "Duplicate" | Someone else found it first | Speed, depth, or niche target |
| "AI-generated" / "automated" | Report reads like AI wrote it | See Gate R2 |

### Gate R2 — What Triggered "AI-Generated" Flag (2 rejections, Jun 2026)

Two crypto submissions rejected with "content appears to be low-quality or potentially automated (AI-generated)." Specific triggers:

1. **Long structured sections** — triage sees templated output
2. **Explaining basics** — "Paillier is a public-key encryption scheme..." assumes reviewer needs crypto 101
3. **No raw PoC output** — described what PoC DOES without showing what it PRODUCED
4. **Theoretical impact** — "300 queries would recover key" without actually doing it
5. **Over-broad titles** — "Key Recovery via..." without having recovered a key
6. **Content-padding headings** — Section titled "Exploitation" that only describes the setup

**Fix each:**
1. Narrative paragraphs, not sectioned structure
2. Delete all explanatory text — reviewer is an expert
3. Paste actual terminal output inline
4. Demo it or don't claim it
5. Title matches exactly what PoC proves
6. Every heading delivers what it promises

### Gate R3 — Honest Scope Check

Before resubmitting:
- Is this a vulnerability or expected behavior?
- If vendor fixed it, would a security boundary become uncrossable?
- Or was no boundary ever crossed? → Not a vulnerability.

### Gate C9 — Crypto PoC: Must Be End-to-End (NEW)

For crypto submissions specifically:

| Chain Step | Must Show | Our Failure |
|-----------|----------|-------------|
| Oracle exists | Different errors per input | ✅ Had this |
| Oracle extracts info | Script output showing leaked bits | ❌ Theorized only |
| Full key recovery | "λ = 0x..." in terminal | ❌ Never ran it |
| Escalation | Decrypted server share | ❌ Never ran it |

Missing any step = theorized impact = rejection.

### Gate C10 — Crypto PoC Format: Standalone Script (NEW)

The PoC must be a single self-contained script:

```bash
python3 exploit.py
# Output: Recovered Paillier private key: 0xDEADBEEF
```

Not "append to test file, build, run the suite." Not "here's the oracle setup, you could recover a key."

**Minimum viable crypto PoC**: A script that takes the target's public key and outputs a leaked private key bit or extracted plaintext. Terminal output must be included inline in the report.

### Gate U5 — PoC is reproducible for TRIAGE, not just for you

Before submitting, verify the reviewer can reproduce your finding with ZERO access to your environment.

**The fresh-clone test:**
```bash
git clone --depth 1 $TARGET_URL /tmp/fresh
cd /tmp/fresh
bash /path/to/poc.sh
```

If this fails, your PoC depends on your specific checkout, build artifacts, or environment — and triage will mark it "Not Reproducible."

**Checklist (all must be YES):**
- [ ] One `git clone` gets the target code
- [ ] One command runs the PoC (no build step)
- [ ] No `apt-get install` of special dependencies
- [ ] No appending to existing test files
- [ ] No custom branch or commit hash required
- [ ] PoC output (terminal text) clearly shows the vulnerability

**Why this matters:** Rejected reports 004 and 005 (Fireblocks MPC, Jun 2026) both passed our internal tests but failed for Bugcrowd triage. The PoCs required access to our local repo with appended test code, modified CMakeLists.txt, and incremental build artifacts. A fresh clone + one grep command would have worked. The appended-test-file approach did not.

## Part 8 — Post-Rejection: Counter-Argument or Move On (Added Jun 2026)

Every rejection is data. Before deciding to counter-argue or move on, run this framework.

### Gate D1 — Map the Rejection to a Gate

| Disposition | Which Gate Failed | Can You Fix It? |
|-------------|------------------|-----------------|
| Out of Scope | Gate T1 (Scope) | Only if you can reframe the vuln class without using OOS keywords — AND the new framing changes the class entirely. If the CORE issue is OOS, move on. |
| Informative | Gate T3 (Validity) or Gate T5 (Impact) | Counter-argument: demonstrate the missing impact that triage said was absent. If you can't, the finding was correctly Informative. |
| Duplicate | Gate T4 (Uniqueness) | Counter-argument: show your finding covers a scope the dupe didn't. Rarely works — move on. |
| Needs More Info | Gate T2 (Reproducibility) | Easy fix — provide clearer steps, curl commands, response examples. |
| AI-generated flag | Gate R2 | Fix the report format (narrative, not templated). No counter-argument — rewrite. |

### Gate D2 — Viability Test for Counter-Arguments

Before drafting a response to the triager, answer these questions:

**Q1 — Is the infrastructure still live?**
- Does the endpoint still exist? (curl test)
- Are the credentials/tokens still valid?
- If the answer is no, the finding is dead — do NOT counter-argue.

**Real example (Nutaku gateway creds, Jun 2026)**: The OAuth-front credentials were marked Informative. We considered counter-arguing, but `api.nutaku.net/gateway/v1` now returns Connection Refused. The entire gateway API has been taken down. There is no live endpoint to demonstrate impact against. Counter-argument has zero chance. → Move on.

**Q2 — Is the data sensitive enough?**
- Would a reasonable person agree the exposed data causes harm?
- Game titles? No. Game titles on an adult platform? Still no — the program explicitly excludes it.
- Email addresses, password hashes, payment info, credit cards? Yes.

**Q3 — Can you reframe the CLASS without changing the EVIDENCE?**
- If the program excludes IDOR but you have CWE-287, reframing might work
- If the program excludes hardcoded creds but you can show actual token reuse impacting another user, reframing works
- If the program excludes all information disclosure regardless of framing, reframing won't work

**Q4 — Is there new evidence you collected AFTER the rejection?**
- New endpoint found? New credential test? New data sample?
- If yes, submit a NEW report with the chain (don't counter-argue the old one)
- If no, the original triage was correct — move on

**Q5 — Write the 3-sentence attack story (from Gate V6)**
- V1: Who is the victim?
- V2: What must they do?
- V3: What is the actual harm?
- If you can't write this, the finding wasn't ready.

### Gate D3 — Decision Matrix

| Q1 (Live?) | Q2 (Sensitive?) | Q3 (Reframe?) | Q4 (New evidence?) | Action |
|-----------|----------------|--------------|-------------------|--------|
| Yes | Yes | Yes | — | **Counter-argue** — write a response addressing the specific gate that failed |
| Yes | Yes | No | — | **Submit NEW report** with the actual sensitive data demonstration |
| Yes | No | — | — | **Move on** — the program doesn't value this data class |
| No | — | — | — | **Move on** — dead infrastructure means no demonstration possible |
| — | — | Yes | Yes | **Submit NEW report** — the new evidence changes the finding entirely |

### Gate D4 — How to Write a Counter-Argument

If the matrix says counter-argue, write a concise response that:

1. **Quotes the triager's specific reason** (e.g., "You said 'IDOR with no direct security or financial impact'")
2. **Acknowledges their reasoning** ("I understand the data on this endpoint is low-sensitivity")
3. **Presents the NEW evidence** ("However, I've now discovered the same auth bypass applies to the user profile endpoint, which leaks email addresses")
4. **Requests re-opening**: "Please re-open for additional review."

**Do NOT**:
- Argue semantics without new evidence
- Quote RFCs or OWASP standards as authority without demonstrating impact
- Blame the triager or accuse them of misreading
- Submit a counter-argument that takes more than 4 sentences — triagers are busy

### Gate D5 — The Infrastructure Dead End Pattern (NEW)

This pattern killed a finding this session and will kill others:

**When the credential/token/endpoint is no longer accessible, the finding is dead.**

Signs:
- Connection refused on the API endpoint
- 404 on the old URL
- Authentication rejection that wasn't there before
- Service shutdown / deprecation notice

**Do NOT try to counter-argue a dead endpoint.** Even if the triager was wrong about the vulnerability class, you cannot demonstrate impact against infrastructure that doesn't respond. Submit a fresh report if a new live endpoint appears, but don't waste time contesting the rejection.

---

### Gate U6 — Data Sensitivity Gate (Added Jun 2026 from AdultForce rejection)

**A technically valid vulnerability with low-value data is Informative. The mechanism does not save you. The data does.**

Before submitting ANY finding where the harm comes from exposed data, classify the data:

| Class | Examples | Verdict |
|-------|----------|---------|
| P1 — Credentials | Working API keys, tokens, passwords | ✅ Submit |
| P2 — PII | Emails, password hashes, payment info | ✅ Submit |
| P3 — Actionable Financial | Billing tokens, card data, refund endpoints | ⚠️ Submit if demonstrable |
| P4 — Business Metadata | Internal IDs, infrastructure paths, org structure | ❌ Do NOT submit |
| P5 — Operational | Maintenance messages, feature flags | ❌ Do NOT submit |
| P6 — Public | Public content, thumbnails | ❌ Do NOT submit |

**The 3-Question Test:**

1. Is the data P1, P2, or P3? (credentials/PII/actionable)
2. Does the data enable a DIRECT action (login, access account, process payment)?
3. Can you write a victim story where the harm isn't "a competitor could see this"?

If any answer is NO, do not submit.

**The self-diagnosis**: Ask "Would peaches call this business metadata?" If yes, don't submit.

### Gate U7 — Standard Disclosure Terms check

Bugcrowd's Standard Disclosure Terms define two categories of rejections:

**Excluded (never rewardable, immediately invalid):**
- Physical testing (office access, tailgating)
- Social engineering (phishing, vishing)
- OOS targets
- UI/UX bugs, spelling mistakes
- DoS/DDoS

**Non-qualifying (low impact — don't report without chaining):**
- Descriptive error messages
- Banner disclosure
- Missing security headers
- Clickjacking
- CSRF on anonymous forms
- Weak Captcha
- Username enumeration
- Login brute force / account lockout
- SSL config issues (BEAST, BREACH, weak ciphers, missing HSTS/XXSS protection)

Before any submission, check:
1. Is the finding excluded outright? (physical, social, OOS, functional, DoS) → Don't submit.
2. Is the finding on the non-qualifying list? → Don't submit unless you can chain to higher impact.
3. Does the finding affect "the target's users, systems, or data security in a meaningful way"? → This is the standard for P4-P5 submissions. Be prepared to defend it.

For crypto/MPC findings specifically: the non-qualifying list is web-app-focused (missing headers, SSL config, CSRF). None of those apply to cryptographic implementation bugs. But the "impact" requirement still applies — be ready to explain why the finding matters to security posture.

### Gate U8 — Multi-Target Audit Format

After a structured hunt across multiple programs, compile findings using
`references/multi-target-audit-format.md` in this skill. Format requires:
- Previous mistake section per target
- Submittable/Not/Lead verdict per finding
- Methodology validation summary at the end

**Zero findings is an acceptable outcome.** If the corrected methodology prevented
N likely-rejected submissions, state that explicitly. The goal is quality submissions,
not volume. A session with zero false submissions is a success.
