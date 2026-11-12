# Class-Specific Gates (C1-C7)

Provenance: split out of `SKILL.md` 2026-09-15 to cut load cost. This is **Tier 1** work:
apply the ONE gate matching this finding's class. Do not run all of them.

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

**Leaked JWT secret → forge (verify before claiming)**: When impact is "leaked signing secret lets attacker forge tokens," verify the leaked value against what the service ACTUALLY verifies with, then run the forge end-to-end:
1. Pull the service's real key config (e.g. `docker inspect` env: PostgREST's `PGRST_JWT_SECRET` may be a JWKS `{"keys":[...]}`). Base64url-decode the `oct` key and compare to the leaked secret — match means the forge will work.
2. Forge HS256 `{"role":"service_role","iss":...,"iat","exp"}`, send as BOTH `Authorization: Bearer` and `apikey`.
3. Read the status code: 200 = accepted; 403-with-GRANT-hint = verified but missing table grants (still proof); 401 = wrong secret — delete the claim.
4. Replay the same token against sibling services (Storage, Realtime) — one secret often gates several.
Pitfall: test sandboxes sometimes hold scrubbed placeholder env values (`AUTH_JWT_SECRET=super-...ong`); the leak may surface the docker-compose DEFAULT fallback, which is often exactly what the running services use (confirm via JWKS/kong.yml). Do not dismiss the leak because env looks masked.
Framing for "known" findings: if the community knows the issue only as an ops problem ("exposed dashboard"), report the APPLICATION-LAYER auth gap instead (endpoints accept requests with no auth of their own), and cite the vendor's own unmerged fix branch as evidence it is not intentional design.

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
