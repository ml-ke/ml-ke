# Zendesk Bug Bounty — Program Intelligence

## Program Overview
- **Bugcrowd**: Managed Bug Bounty Engagement
- **Started**: Dec 11, 2025 (relatively new — less competition)
- **Scope rating**: 2/4 (focused, manageable)
- **Signup**: Self-service at https://www.zendesk.com/register with @bugcrowdninja.com
- **Company name format**: `bb-<username>` (e.g., bb-atlas-01)
- **Instance domain**: `{company}.zendesk.com`

## Scope Targets

### 1. Zendesk AI (P1 $5K–$20K)
- Tags: API Testing, Large Language Model, Penetration Testing
- AI Agent: prompt injection, RAG poisoning, data leakage, model extraction
- Action execution abuse, privilege escalation, cross-tenant leakage

### 2. Zendesk Front End (P1 $5K–$20K)
- Scope: Anonymous user, end-user, customer context only (NOT agents/admins)
- Web widget and web widget SDK
- Zendesk Mobile SDK (iOS/Android)
- Social channel integrations
- Voice & Contact Center front end
- Knowledge front end: *.zendesk.com/hc/ (form submission, "My activities", "My profile")
- Authentication front end: *.zendesk.com/auth/
- OOS: HTTP header misconfigs (HSTS, CSP, XFO), Self-XSS, HTML injection via phishing links

### 3. Zendesk Suite (P1 $5K–$10K)
- Agent workspace: *.zendesk.com/agent
- Knowledge: *.zendesk.com/knowledge/
- Contact center, Analytics: *.zendesk.com/explore/
- WFM: *.zendesk.com/wfm/
- QA: *.zendesk.com/qa
- Admin center: *.zendesk.com/admin/
- Same OOS as Front End

## Bounty Payouts
| Severity | AI / Front End | Suite |
|----------|---------------|-------|
| P1 | $5,000–$20,000 | $5,000–$10,000 |
| P2 | $2,000–$3,000 | $2,000 |
| P3 | $750–$1,500 | $500 |
| P4 | $250 | $100 |

## Bugcrowd Submission Format (Exact)

Each report must follow this field sequence in the Bugcrowd form:

```
## Summary
Brief 2-3 sentence overview. "Help us get an idea of what this vulnerability is about."

## Submission title
Descriptive title. Starting with the finding class.

## Target
Dropdown from program scope. For Zendesk: "Zendesk Front End" or "Zendesk Suite" or "Zendesk AI".

## Technical severity
VRT v1.18 baseline (P1-P5). Impact context may adjust.

## VRT Category
Dropdown from VRT. Examples: "Sensitive Data Exposure", "Broken Auth → Authentication Bypass".

## URL / Location of vulnerability
Optional. Exact URL path where the vulnerability was tested.

## Description
Max 25K chars. Must include:
- Vulnerability description and impact
- Proof of concept with replication steps
- Actual terminal output from verified PoC (not "expected output")
- Comparison with properly-secured endpoints
- Remediation recommendations

## Attachments
Max 20 files, <400MB each. PoC scripts, screenshots, output logs.
```

## Instance Deep Recon (mlke.zendesk.com — Professional Trial)

### Account Plan
- **Plan**: Professional (trial, expires June 15, 2026)
- **Trial days left**: 13 (as of June 1, 2026)
- **Agent Workspace**: Enabled (`hasAgentWorkspace: true`)
- **Multiproduct**: true
- **Sandbox**: false (production instance)
- **Help desk size**: 1-9, **Region**: EMEA, **Country**: KE
- **Subscriptions**: trial, self-service, Large plan, 5 max agents

### AI Settings (from account/settings.json)
```json
{
  "ai": {
    "macro_content_suggestions_title_gen_enabled": false,
    "admin_copilot_conversational_enabled": true
  },
  "knowledge": {
    "generative_answers": true,
    "search_articles": true
  }
}
```
- `admin_copilot_conversational_enabled: true` — AI Copilot is active
- `generative_answers: true` — AI-powered KB answers are enabled
- Despite these settings, all AI API endpoints (`/api/v2/ai/*`) return 404. The AI features require additional plan upgrades to access via API.

### Session Cookie Authentication

When the user is logged into the Zendesk dashboard, they can share their `_zendesk_shared_session` cookie. This cookie:
- Is HTTPOnly and Secure (cannot be set via `document.cookie` — must be used from curl/Python)
- Works for API access: `curl -H "Cookie: _zendesk_shared_session=VALUE" https://mlke.zendesk.com/api/v2/users/me.json`
- CSRF token: extracted from `https://mlke.zendesk.com/agent/admin` HTML with `grep -oP 'csrf-token" content="\\K[^"]+'`
- CSRF token is per-page-load — must be freshly extracted before each PUT/POST/DELETE
- State-changing requests (PUT/POST/DELETE) require X-CSRF-Token header even with cookie auth
- GET requests work with cookie alone (no CSRF needed)

### Additional Dead Ends (Systematic Testing)
| Vector | Method | Result |
|--------|--------|--------|
| Placeholder injection `{{ticket.requester.*}}` in subject/comment | POST /tickets | Not rendered — `{{` and `}}` stripped, raw text stays |
| SSRF via remote_photo_url on user create | POST /users | HTTP 201 but URL not fetched (photo stays null) |
| SSRF via webhook to internal IPs | POST /webhooks | All HTTP 400 — robust URL validation |
| SSRF via webhook DNS rebinding | POST /webhooks | All HTTP 400 — resolves DNS before checking |
| SSRF via webhook to our public IP | POST /webhooks | HTTP 400 — even public IPs blocked for non-TLS |
| HTML file upload (.html) | POST /uploads | HTTP 422 — extension explicitly blocked |
| SVG file upload with XSS | POST /uploads | HTTP 201 — file stored but attachments require auth |
| Path traversal `../../../etc/passwd` | GET | HTTP 400 — blocked by Rack middleware |
| IDOR via Requests API | GET /api/v2/requests | HTTP 401 — properly locked down |
| OAuth implicit grant | GET /oauth/authorizations | HTTP 401 — requires auth |
| GraphQL schema introspection | POST /explore/graphql | Returns "undefinedField" — disabled |
| Macro suggestions API | GET /api/v2/macros/suggestions | HTTP 400 — exists but needs proper params |
| AI Copilot API | GET/POST /api/v2/ai/* | HTTP 404 — plan-restricted |
| Sunshine Conversations API | GET /sc/v2/* | HTTP 401 — wrong auth type (needs Sunshine API keys) |

### Help Center Articles (Public Access)
- `/api/v2/help_center/en-us/articles`: Returns **1** published article without auth
- The 11 draft-only articles are properly hidden from public view
- `/api/v2/help_center/en-us/categories`: 1 category ("General") — public
- `/api/v2/help_center/en-us/sections`: 1 section — public
- The `/hc/en-us` web page is still behind Cloudflare (403), but the API endpoints work without auth
- **Behavior is correct** — published articles are intentionally public; drafts are properly hidden

### Instance Metadata
- **Account name**: MLKE
- **Country code**: KE (Kenya)
- **Timezone**: Africa/Nairobi
- **Multiproduct**: true (all Zendesk products available)
- **Created**: 2026-06-01T17:56:35Z
- **Sandbox**: false (production instance)
- **Help desk size**: 1-9
- **Region**: EMEA
- **Owner ID**: 27859861923996
- **Brand ID**: 27859877743004

### API Access
API token auth works with `{email}/token:{token}` as Basic auth credentials. Token: generated in Admin → Apps & Integrations → Zendesk API.

Endpoint base: `https://{subdomain}.zendesk.com/api/v2`

Users:
- **Admin**: ID 27859861923996, name "Atlas Ninja", email h0d4r1@bugcrowdninja.com
- **Customer**: ID 27859854141852, name "Customer", email example@zendesk.com
- No agents other than the admin

### Endpoint Status (unauthenticated)
| Path | HTTP Code | Notes |
|------|-----------|-------|
| / | 301 | Redirects to /auth/v3/signin |
| /agent | 302 | Agent workspace → login |
| /admin | 302 | Admin center → login |
| /knowledge | 302 | Knowledge → login |
| /hc/en-us | 403/CF | Help center → Cloudflare challenge |
| /auth/v2/login | 302 | Login endpoint |
| /auth/v3/signin | 200 | Sign-in page (brand_id + role in URL) |
| /api/v2/* | 401 | All API endpoints require auth |
| /api/v2/help_center | 302 | HC API → login (may be public if HC enabled) |
| static.zdassets.com/embeds.js | 403 | Web widget CDN, blocked from datacenter |

### Auth Endpoint
- Login: `https://{subdomain}.zendesk.com/auth/v3/signin?brand_id={id}&role={end_user|agent}`
- Brand ID visible in URL (27859877743004 on mlke)
- Role parameter determines login form display

### Help Center (Not Public — Needs Admin Setup)
12 sample articles about fictional furniture company "Homebuy". All drafts except 1 promoted article.
Topics: gift cards, payment plans, returns, shipping, assembly. All draft status means the HC isn't published.

Go to Admin → Help Center → Enable to make the KB public.

### AI Detected Tags on Sample Ticket
The sample ticket "SAMPLE generated by AI: Issue accessing my Bugcrowd Ninja email alias" came with:
- `zendesk_accelerated_setup` tag (generated during instance creation)
- AI intent classification: `intent__account__activation__waiting` (confidence: medium)
- Language: `language__en` (confidence: high)
- Sentiment: `sentiment__neutral` (confidence: high)
- 16 custom fields including all AI classifications
- Channel: email (simulated)

### Trigger/Automation/Macro Inventory
- **8 triggers**: All default notification triggers (welcome, assignment, comment updates)
- **3 automations**: Close ticket 4 days after solved (active), 2 pending notifications (inactive)
- **2 macros**: Customer not responding, Downgrade and inform
- **10 views**: Unsolved, unassigned, recently updated, pending, overdue, solved
- **0 OAuth clients**: No end-user OAuth flow configured
- **0 user fields**: No custom user fields
- **0 organization fields**: No custom org fields
- **0 custom roles** beyond default
- **Apps installed**: Zendesk Chat (app_id=30460)

### Rate Limiting
- Measured at ~1.3 req/s for rapid API calls against `/api/v2/tickets`
- This is aggressive — account for it in test scripts with delays or backoff

### Voice Settings
- Maximum queue size: 5
- Maximum wait: 1 minute
- Voice AI: disabled (`voice_ai_enabled: false`)
- Voice transcriptions PII redaction: disabled
- Voice Zendesk QA: disabled

### OAuth / Auth Security
- OAuth authorize endpoint: HTTP 401 (requires authentication to access)
- OAuth authorizations management: HTTP 403
- No OAuth clients configured (0)
- No OAuth tokens issued (0)
- Anonymous ticket creation via API: HTTP 401 (requires auth)
- API token scoping: Admin-level access to all endpoints

### Web Widget
- `static.zdassets.com/web_widget/latest/embeds.js` — HTTP 403 from datacenter IP
- Widget needs to be configured and embedded in a page to test
- Widget SDK is TypeScript/JavaScript — potential reverse engineering target

## AI Agent Attack Vectors (from Bugcrowd scope)

The program explicitly lists these AI testing areas — target these first when AI Agent is enabled:
1. Prompt injection — system prompt bypass, hidden instructions
2. RAG (retrieval) poisoning — malicious content via connectors
3. Retrieval/provenance bypass — low-trust sources without provenance flags
4. Response data leakage — PII, secrets, credentials in AI output
5. Attachment exfiltration — hidden metadata/footers in documents
6. Model extraction — surrogate model from API queries
7. Membership inference — proving training data presence
8. Data poisoning/backdoors — crafted examples during ingestion
9. Action execution abuse — generating text that runs macros/webhooks
10. Privilege escalation — broken access control, token replay
11. Cross-tenant leakage — other tenants' tickets/content

**Note**: AI Agent features require the Zendesk AI add-on plan. On a fresh trial instance, AI endpoints returned 404 (not available on the base plan).

## Setup Required to Maximize Attack Surface
1. Sign up at https://www.zendesk.com/register (done)
2. Enable Help Center: Admin → Settings → Help Center → Enable
3. Publish at least 1 article so the KB is publicly accessible
4. Configure Web Widget: Admin → Channels → Web Widget
5. Create additional end-user accounts for IDOR testing
6. If AI Agent is available: configure a test agent with sample knowledge base content

## Confirmed Findings

### Finding 1: Unauthenticated Ticket Forms Disclosure (P5 — Informational, No Reward)
**Endpoint**: `GET /api/v2/ticket_forms` — returns partial form config without auth.
**Evidence**: HTTP 200 with valid Accept header (`application/json`, `*/*`, `application/vnd.zendesk+json`), no session/API token needed.
**Data leaked**: form IDs, field IDs (`ticket_field_ids` — only standard Subject+Description fields, custom fields are properly filtered), brand restrictions (`restricted_brand_ids`), visibility settings (`end_user_visible`), conditional logic (`end_user_conditions`), timestamps, form type.
**Comparison**: All other API v2 list endpoints return 401 without auth (ticket_fields, brands, tags, custom_roles, sla_policies, satisfaction_reasons).
**Escalation**: Individual form access (`/ticket_forms/:id`) returns 401. Ticket creation returns 401. CSRF token from `/users/me` anonymous response is single-use. Write operations (POST/PUT/DELETE) all return 401/403.
**Note**: 18 out of 19 response fields are identical between auth and unauth. The only filtered fields are `agent_conditions` (properly hidden) and 16 custom `ticket_field_ids` (only Subject+Description exposed). This suggests Zendesk partially designed this for public access — the help center widget needs form data to function.
**Severity note**: Rated P5 (Informational) by Bugcrowd triage. Feedback: *"this issue is considered to be a P5 (Informational) finding as per Bugcrowd's VRT. This is typically the case when an issue lacks a demonstrated risk and is considered security best practice."* The form data (form names, 2 standard field IDs, brand info) doesn't enable an attacker to DO anything meaningful. Lesson: "as an attacker I could..." must have demonstrated impact, not just "information disclosure."
**Status**: Report at ~/Dev/REPORTS/Zendesk/001/REPORT.md. PoC at .../001/poc/poc_ticket_forms_unauth.py.

### Finding 2: GraphQL Endpoint Without Auth (P4 — Research)
**Endpoint**: `GET/POST /explore/graphql` — accessible without authentication.
**Evidence**: Returns `{"data":{"__typename":"Query"}}` with zero auth headers. Needs `apollographql-client-name` and `apollographql-client-version` headers.
**Limitations**:
- Introspection is disabled: `__schema` returns "Field '__schema' doesn't exist on type 'Query'"
- Only `node(id: ...)` query found — requires encoded Relay global IDs (base64 "TypeName:integer")
- Inline fragment probing shows no matching types (`Ticket`, `User` fragments get "No such type" errors)
- This is the Explore analytics GraphQL, not the main Support API
- Mutation endpoints (`/api/v2/explore/execute-query`) require proper auth (401)
**Note**: Low-severity. The endpoint exists without auth but returns no useful data. Could become useful if schema introspection is re-enabled.

### Finding 3: Stored XSS in Ticket Subject (Unverified — Needs UI Access)
**Evidence**: Ticket subject `<script>alert("XSS")</script>` stored as-is without sanitization.
**Problem**: Cannot verify rendering — the agent dashboard (where subjects are displayed) is behind Cloudflare WAF from datacenter IPs. Would need user to check in their browser.
**Note**: If the subject renders unescaped in the agent workspace, this is stored XSS. Otherwise, subjects are properly escaped and this is harmless.

### Finding 4: SVG Upload Works but Behind Auth
**Evidence**: SVG files (including `<script>` tags) upload successfully via `POST /api/v2/uploads?filename=xss.svg`.
**Problem**: Attachment URLs redirect to login page (302 → `/access/unauthenticated`). Require authentication to access. No public XSS vector.

## Testing Dead Ends

| Vector | Result | Why |
|--------|--------|-----|
| Placeholder injection `{{ticket.requester.email}}` | ❌ Not rendered | `{{` and `}}` stripped, raw text remains |
| SSRF via `remote_photo_url` | ❌ Not fetched | Zendesk doesn't auto-fetch avatar URLs on user creation |
| SSRF via webhook internal IPs | ❌ Blocked (400) | Zendesk validates webhook endpoint URLs |
| HTML file upload | ❌ Blocked | `.html` extension explicitly blocked |
| Path traversal (`../../../etc/passwd`) | ❌ Blocked | 400 Bad Request |
| IDOR via Requests API (`/api/v2/requests`) | ❌ 401 without auth | Properly locked down |
| CSRF via anonymous `authenticity_token` | ❌ Single-use per request | Tokens change on every page load |
| OAuth implicit grant | ❌ Not checkable | Would need OAuth client configured |
| AI Agent endpoints | ❌ Plan-restricted | Return InvalidEndpoint even with auth |
| Sunshine Conversations API | ❌ Not enabled | `sc/v2/apps` returns 401 with wrong auth type |
| Help Center KB unauth | ❌ Cloudflare blocked | All `/hc/` paths hit Cloudflare JS challenge |

## Systematic Unauthenticated Endpoint Scan Pattern

When testing a new target for unauth access, use this systematic pattern rather than spot-checking:

1. **Create endpoint list** — Compile ALL known API v2 paths from the vendor's API docs
2. **Send each WITHOUT auth** — Just `Accept: application/json` header. Record HTTP status codes.
3. **Compare against similar endpoints** — If `/ticket_forms` returns 200 but `/ticket_fields` returns 401, that's the finding. The comparison proves the auth gap is intentional.
4. **Individual access test** — The list endpoint may be open but individual resource access should be 401. Test `/resource/:id` to confirm.
5. **Escalation test** — Try creating/modifying without auth (POST/PUT/DELETE). Usually these are properly locked down even if GET lists are open.
6. **Cross-reference with docs** — Check if the endpoint is documented as public. If docs say "requires auth" but doesn't, that's additional evidence.

**Key insight from Zendesk testing**: `/api/v2/ticket_forms` (list) returned 200 without auth while `/api/v2/ticket_forms/:id` (individual) returned 401. The list endpoint was the only one missing auth. Always test both list and individual endpoints — they may have different auth configurations.

**How findings survive fact-checking** (the Zendesk pattern):
- ✅ Endpoint returns real data (not redirect/error) without auth
- ✅ Same data is NOT accessible via any other unauth path
- ✅ Similar endpoints ARE properly locked down (comparison evidence)
- ✅ Multiple Accept headers tested (application/json, */*, custom)
- ✅ No custom headers (cookies, auth tokens) sent — only Accept
- ✅ Individual resource access returns 401 (proves auth exists but list was missed)
- ✅ POST/PUT/DELETE operations return 401 (proves write operations are locked)
- ✅ CSRF token from anonymous endpoint cannot be reused (tokens are single-use)

## Escalation Testing Checklist (Post-Discovery)

After finding a vulnerability, systematically test ALL escalation paths before claiming higher severity:

| Escalation | Test | Expected Result |
|------------|------|-----------------|
| Individual resource access | `GET /resource/:id` (no auth) | 401 |
| Write operations | `POST/PUT/DELETE /resource` (no auth) | 401 |
| CSRF bypass | Reuse `authenticity_token` from anonymous response | 401/403 |
| Origin/Referer spoofing | Add `Origin: evil.com` or `Referer: target.com` | 401 |
| Cookie-based auth | Send request with cookie jar but no session | 401/415 |
| Path traversal | `GET /../etc/passwd` | 404/400 |
| Accept header variation | Try `*/*`, `application/xml`, `text/html` | 415 or 200 for JSON |

Document ALL failed attempts in the report. Showing what doesn't work is as important as showing what does — it proves you tried to escalate and couldn't.

## JWT SSO Sample Code Audit (github.com/zendesk/zendesk_jwt_sso_examples)

All 9 language implementations of Zendesk JWT SSO were audited. Key findings:

### CRITICAL: Hardcoded Shared Key in C# Handler
- File: `jwt_generation/c_sharp_handler.cs`, line 14
- `private static readonly string SHARED_KEY = "{my Zendesk token}";`
- String literal placeholder — if a developer copies this without replacing the value, the shared key is literally the string `{my Zendesk token}` — trivially guessable.

### HIGH: Weak/Predictable `jti` in PHP
- File: `jwt_generation/php_jwt.php`, line 13
- `"jti" => md5($now . rand())` — uses `rand()` (LCG, predictable), not `random_bytes()`
- All other implementations use SecureRandom, UUIDv4, or Node's uuid.

### MEDIUM: Python uuid1() Leaks MAC Address
- `python_django_jwt.py` (line 15), `python_flask_jwt.py` (line 14)
- `str(uuid.uuid1())` embeds server MAC address + timestamp.
- Should use `uuid.uuid4()` like every other implementation.

### MEDIUM: Node.js Uses Deprecated `jwt-simple`
- File: `jwt_generation/node_jwt.js`, lines 1, 5
- `jwt-simple` is unmaintained; its decode() doesn't validate algorithm header against whitelist.
- Risk of algorithm confusion when this library is used for verification.
- Example also doesn't specify algorithm — relies on default (HS256), but without explicit param.

### MEDIUM: No `exp` Claim in Any Implementation
- All 9 files generate JWT with `iat` but never `exp`.
- No self-contained expiration. Token validity depends entirely on Zendesk's session timeout.
- Per RFC 7519, `exp` is the standard mechanism for limiting token lifetime.

### MEDIUM: jQuery Form Submission Appends Raw Query String
- File: `form_submission/jquery_xhr_jwt.js`, line 11
- `attr("action", data['url'] + window.location.search)` — blindly appends query string.
- No parameter whitelist (contrast with Rails implementation which uses `params.slice`).
- Attacker controlling URL params can inject `return_to` or other parameters.

### Git History Findings (deleted/patched)
- Old Classic ASP file had debug mode that leaked JWT tokens and LDAP results
- Old Classic ASP file had LDAP injection (unvalidated `LOGON_USER` in LDAP query)
- Old C# and ASP redirect implementations had open redirect via `return_to` param

## Cross-Storage Library Audit (github.com/zendesk/cross-storage v1.0.0)

12 security issues found. Key ones:

### CRITICAL: `"null"` Origin → `"file://"` Mapping (Sandboxed Iframe Bypass)
- `lib/hub.js` line 72, `lib/client.js` line 272
- `origin = (message.origin === 'null') ? 'file://' : message.origin;`
- Browser sets `message.origin` to `"null"` for sandboxed iframes. This transparently remaps to `file://` which may have broad permissions.
- **Impact**: A sandboxed iframe on any arbitrary page can match `file://` permission.

### HIGH: Hub Broadcasts to `*` on Init
- `lib/hub.js` lines 34, 42 — `postMessage('cross-storage:ready', '*')`
- Any listening window can detect hub initialization.

### HIGH: Poll Handler Origin Inconsistency
- `lib/hub.js` lines 75-77 — poll response uses raw `message.origin` (not `'file://'`)
- Browser-dependent behavior: Firefox delivers to `*` when origin is `"null"`.

### MEDIUM: Prototype Property Access via `response.id`
- `lib/client.js` lines 312-313 — `client._requests[response.id]` 
- `"__proto__"` maps to `Object.prototype` — not callable, hangs promise permanently.
- `"constructor"` maps to `Object` — callable but returns empty object.

### MEDIUM: Iframe Hijacking via `frameId`
- `lib/client.js` lines 55-57 — `document.getElementById(opts.frameId)`
- If attacker injects element with matching ID before client initializes, attacker's iframe receives all storage requests.

### MEDIUM: No Key Namespace Isolation
- All authorized origins share flat `localStorage` namespace. No key prefixing.
- Origin A can overwrite / read keys set by Origin B.

## Common Investigation Paths

### API-Level Testing (auth token available)
- IDOR on tickets, users, organizations (two-account testing)
- Custom role privilege escalation
- Trigger/automation abuse for data leakage
- Webhook endpoint security testing
- Suspended/deleted ticket data retention

### Browser-Level Testing (Help Center public)
- XSS in article content/rendering
- CSRF in ticket submission forms
- Rate limiting on form submission
- File upload abuse in attachment endpoints
