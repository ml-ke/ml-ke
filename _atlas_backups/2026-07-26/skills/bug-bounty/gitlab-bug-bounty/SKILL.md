---
title: GitLab Bug Bounty Methodology
name: gitlab-bug-bounty
description: Systematic approach to finding vulnerabilities in GitLab open source code and gitlab.com for the HackerOne bug bounty program. Combines static analysis on the cloned Ruby on Rails codebase with live API testing.
---

# GitLab Bug Bounty Methodology

Target: GitLab (hackerone.com/gitlab)
Bounties: $100–$35,000
  - CVSS-based calculator at https://gitlab-com.gitlab.io/gl-security/product-security/appsec/cvss-calculator/
  - Current bounty ranges (post-Nov 2021):
    - Critical (CVSS 9.0-10.0): $20,000 – $35,000
    - High (CVSS 7.0-8.9): $5,000 – $15,000
    - Medium (CVSS 4.0-6.9): $1,000 – $2,500
    - Low (CVSS 0.1-3.9): $100 – $750
  - Timeline: $1M+ paid in 2024 across 275 valid reports from 457 researchers
  - DoS now out of scope (policy update Jan 20, 2026)
  - 90-day themed challenges for bonus payouts (historical)
Scope: gitlab.com, self-hosted instances, source code (gitlab-org/*), registry.gitlab.com
Code: Cloned to ~/Dev/gitlab/

## API Hacking Methodology (from Hacking APIs book)

### Reconnaissance Phase
1. **API documentation audit**: Read all endpoint docs, auth requirements, rate limits
2. **Endpoint discovery**: Use Kiterunner or directory brute-force to find endpoints
3. **Authentication mapping**: Register accounts at all privilege levels (user, admin if possible)
4. **Tech stack fingerprinting**: Identify frameworks, databases, WAFs from response headers
5. **Parameter discovery**: Use Arjun or fuzzing to find hidden parameters

### Authentication & Authorization Testing
1. **JWT attacks**:
   - Check `alg: none` bypass
   - Weak signing keys (brute-force with hashcat)
   - Token expiration bypass (set exp to far future)
   - Role escalation (modify claims like `admin:false` → `admin:true`)
2. **OAuth 2.0 attacks**:
   - CSRF on authorization endpoint (no state parameter)
   - Redirect URI manipulation (open redirect in OAuth flow)
   - Token leakage via referrer headers
3. **Mass assignment**:
   - Add `admin: true`, `role: admin`, `is_admin: 1` to registration requests
   - Fuzz for hidden parameters with Arjun (`--include='{$arjun$}'`)
   - Try blind mass assignment (send many variable names at once)

### Injection Testing (from Ch12)
1. **NoSQL injection** (MongoDB operators):
   - `{"$gt":""}` — greater than (bypass auth)
   - `{"$ne":""}` — not equal (bypass auth)
   - `{"$nin":[1]}` — not in (data extraction)
   - `{"$where": "sleep(1000)"}` — time-based detection
2. **OS Command injection**: Try separators `|`, `||`, `&`, `&&`, `;`, `` ` `` after URL params, request params, headers
3. **SQL injection**: Use sqlmap with `--os-pwn` option for maximum exploitation

### Evasive Techniques & Rate Limit Bypass (from Ch13)
1. **WAF bypass via encoding**: Send payloads through multiple encoders (base64 → URL → unicode)
2. **IP rotation**: Use AWS API Gateway + Burp Suite IP Rotate extension (rotates IP per request)
3. **Rate limit bypass methods**:
   - Alter URL path slightly (add meaningless params like `?test=001`)
   - Add/rotate arbitrary request parameters
   - Origin header spoofing: `X-Forwarded-For`, `X-Originating-IP`, `X-Client-IP`, `X-Remote-Addr`
   - Rotate User-Agent headers using SecLists wordlists
   - Use burner accounts (create multiple accounts if one gets rate-limited)
4. **WAF fingerprinting**: Identify WAF from 403/406 responses, then tailor payloads to known bypasses
5. **Slow/Fast fuzzing**: Use Arjun `--stable` for slow scans when rate-limited

### GraphQL Attack Methodology (from Ch14)
1. **Discovery**:
   - Directory brute-force for `/graphql`, `/graphiql`, `/playground`, `/v1/graphql`, `/api/graphql`
   - Look for common GraphQL cookies/headers (e.g., `graphql_token`)
2. **Introspection** (when enabled — GitLab has it ON):
   ```graphql
   { __schema { types { name fields { name } } } }
   ```
   Extract all query/mutation/subscription names, field types, and argument schemas
3. **Field suggestion attacks** (when enabled):
   - Type field names by typing partial names and reading error suggestions
   - Extract hidden/deprecated fields that have inadequate authorization
4. **Query depth attacks**: Craft deeply nested queries to cause DoS
   ```graphql
   query { project { issues { project { issues { project { ... } } } } } }
   ```
5. **Alias batching** (confirmed ON in GitLab):
   ```graphql
   query {
     a: project(fullPath: "gitlab-org/gitlab") { id visibility }
     b: project(fullPath: "gitlab-org/gitlab-runner") { id visibility }
   }
   ```
   Bypasses per-query rate limits. Up to 10+ aliases per request confirmed.
6. **Authorization testing**: Check if nested resolvers check permissions independently
   - Try accessing a private project's data through its parent group
   - Try querying mutation fields without proper authentication
7. **Mutation fuzzing**: Try unexpected input types on mutation arguments
8. **GraphQL query depth + aliasing brute force**:
   ```python
   # Generate aliased queries to bypass rate limits
   batch = "query { "
   for i, path in enumerate(projects):
       batch += f'p{i}: project(fullPath: "{path}") {{ id fullPath visibility }} '
   batch += " }"
   ```

### Report Structure (for GitLab bounty calculator)
When submitting to GitLab, use their CVSS calculator for suggested bounty:
1. Compute CVSS vector via https://gitlab-com.gitlab.io/gl-security/product-security/appsec/cvss-calculator/
2. Note bounty range suggested by calculator
3. Report goes through peer review for final amount
4. GitLab has 6-hour initial response time target

## Key Attack Vectors

### 1. SSRF — Package Registry (HIGH)

**Main finding**: `lib/packages/ssrf_protection.rb` shows that SSRF protection for the package registry is:
- Feature-gated: `generic_package_registry_ssrf_protection` defaults to `false`
- For non-generic packages (npm, maven, pypi): `ssrf_filter: false` — NO SSRF protection
- Workhorse `sendurl` only applies SSRF filter when `SSRFFilter=true` is sent from Rails

**Attack**: Upload a package with a remote URL pointing to `http://169.254.169.254/` or `http://localtest.me:8080/` to reach cloud metadata or internal services.

### 2. SSRF — DNS Rebinding Bypass

**Main finding**: `UrlBlocker.validate!()` (in `gems/gitlab-http/lib/gitlab/http_v2/url_blocker.rb`) resolves DNS and checks IPs, but `dns_rebind_protection` defaults to `false` in most callers (see `AddressableUrlValidator` line 64).

**Bypass**: Set up a domain that:
1. First resolves to a public IP (validation passes)
2. Then resolves to 127.0.0.1 when the actual HTTP request is made
This bypasses the URL validation but reaches localhost on execution.

### 3. SSRF — Webhook URL Variables

**Main finding**: Webhook URLs can use `url_variables` (interpolated after validation). The template URL is validated, but the interpolated URL (after variable substitution) uses `public_url` validator which blocks localhost by default — unless `allow_local_requests_from_web_hooks_and_services?` is enabled.

**Attack**: Inject localhost IP via URL variable substitution.

### 4. SSRF — Workhorse SendURL

**Main finding**: Workhorse (`workhorse/internal/sendurl/sendurl.go`) fetches external URLs when instructed by Rails. The `SSRFFilter` parameter controls whether SSRF protection is used. Many code paths send `ssrf_filter: false`.

**Attack**: Find Rails endpoints that use Workhorse SendURL with `ssrf_filter: false` and provide user-controlled URLs.

### 5. Push Mirror URL Validation Gap (CONFIRMED)

**Status**: ✅ Confirmed via live API testing on gitlab.com

The push mirror endpoint accepts URLs with DNS bypass domains:
- `http://localhost:8080/mirror.git` → ❌ Blocked
- `http://1.0.0.127.nip.io:8080/mirror.git` → ✅ **Accepted** (mirror #4053688, enabled: true)

Root cause: The `nip.io` domain `1.0.0.127.nip.io` resolves to `1.0.0.127` which is a public IP (APNIC range), not a loopback address. The URL validation checks the resolved IP, and since it's public, validation passes.

**Impact on self-hosted**: If `allow_local_requests_from_web_hooks_and_services?` is enabled, private IPs could be targeted.

### 6. Webhook SSRF — 7 Bypass Domains Confirmed (CONFIRMED)

All 7 nip.io-format webhooks remain **executable** (not disabled) on gitlab.com:
- `1.0.0.127.nip.io`, `0x7f000001.nip.io`, `2130706433.nip.io`, `65535.nip.io`, `0.nip.io`, `1.1.1.1.nip.io`

GitLab's backend (v19.1.0-pre, IP `34.74.226.28`) attempts outbound connections. While execution-time `UrlBlocker` protection blocks loopback IPs, the URLs all passed creation-time validation.

### 7. Import System Makes Live HTTP Connections (CONFIRMED)

When importing from `http://1.0.0.127.nip.io:8080/test.git`, GitLab responded with "Unable to access repository" — confirming it made a live HTTP connection attempt to the user-provided URL.

### 8. GraphQL Alias Batching (CONFIRMED)

GitLab's GraphQL API supports alias-based query batching — 10+ independent queries can be bundled in a single HTTP request:
```graphql
query {
  p1: project(fullPath: "gitlab-org/gitlab") { id visibility }
  p2: project(fullPath: "gitlab-org/gitlab-runner") { id visibility }
  # ... up to at least 10+
}
```

This can bypass per-query rate limits for data collection.

### 9. Project Member Enumeration via GraphQL (CONFIRMED)

Project and group members are accessible via GraphQL on public projects:
```graphql
{
  project(fullPath: "gitlab-org/gitlab") {
    projectMembers(first: 3) {
      nodes {
        user { username name }
        accessLevel { integerValue stringValue }
      }
    }
  }
}
```
Returns usernames, full names, and access levels. Expected behavior for public projects but useful for reconnaissance.

### 10. Internal Project Discovery (CONFIRMED)

GitLab's project search API reveals "internal" visibility projects to any authenticated user. Found examples:
- `daljeet.singh/internal`
- `mdiazg/internal`
- `yashsodhamaintainer/internal`

These are accessible (readable) by any authenticated gitlab.com user — this is by design for "internal" visibility, but worth noting for reconnaissance.

### 5. IDOR — GraphQL API

GitLab uses extensive GraphQL endpoints:
- Check for field suggestions enabled (introspection)
- Batch attack to bypass rate limits
- Check authorization on nested queries
- Look for missing `authorize` directives on new/resolver fields

### 6. IDOR — REST API

- Check project/group ID enumeration for access control bypass
- Look for parameter tampering in `projects/:id` endpoints
- Check merge request approval bypass
- Check CI/CD variable visibility

### 7. CI/CD Pipeline Injection

- Check if downstream pipeline variables can be poisoned
- Check artifact path traversal
- Check runner token leakage via job logs

### Report Directory Structure

```
~/Dev/REPORTS/GitLab/<SubmissionNumber>/<finding-name>/
├── REPORT.md           ← Standalone writeup (NOT in zip). Root of finding folder.
└── poc/
    └── poc-<name>.py   ← Working exploit/PoC script
```

Rules:
- REPORT.md sits at the finding folder root, never inside the zip
- Source repos stay at ~/Dev/<vendor>/ (e.g., ~/Dev/gitlab/)
- PoC/test code written to test the program stays at ~/Dev/ root (not in REPORT/)
- Use published packages (not repo clones) — triager should not build from source

### Pre-Submission Fact-Check

Before submitting, verify every claim against actual source code:

1. **Check migration files** — `t.string` = varchar(255), `t.text` = unbounded. Don't claim full payload exposure for truncated columns.
2. **Trace the auth chain** — Read the guardian/policy file, not just the controller. Does it verify OWNERSHIP or just VISIBILITY?
3. **Test on live instance** — Verify the route exists, auth works as expected, and error codes match your claims.
4. **Check escalation claims** — Can you really enumerate IDs? Is the data really accessible cross-user? Or do rate limits / auth gates prevent it?
5. **Test redirects** — Does the endpoint follow redirects? Are redirect targets re-validated?
6. **Use the CVSS calculator** — https://gitlab-com.gitlab.io/gl-security/product-security/appsec/cvss-calculator/
7. **Be honest about severity** — A Medium finding with accurate analysis is worth more than a High with inflated claims. The user prioritizes signal rating over inflated severity.

## AI Feature Recon Results

GitLab has extensive AI features exposed via GraphQL (Enterprise Edition, `ee/` directory). Key findings from live testing on gitlab.com. Full trial transcript in `references/ai-trial-testing-results.md`.

### Confirmed GraphQL Surface
- `aiAction` mutation — main entry point for ALL AI features
- `aiConversationThreads` query — lists user's conversation threads (properly scoped)
- `aiMessages` query — lists user's AI messages (fields: id, content, role, requestId, timestamp, errors, extras)
- `aiCatalog*` mutations — AI workflow catalog management
- `duoChatAvailable`, `duoStatus`, `duoSettings` — status queries
- Field suggestions are ENABLED on all AI types

### aiAction Authorization Model

The mutation (`ee/app/graphql/mutations/ai/action.rb`) chains these checks:
1. Rate limit via `ai_action` throttle (per user)
2. Feature flag check via `FlagChecker.flag_enabled_for_feature?`
3. Resource authorization via `Authorizer.resource(resource, user).allowed?` (checks READ access)
4. User licensing via `user.allowed_to_use?(ai_action)`
5. `user_can_send_to_ai?` via `ChatAuthorizer` for chat features

AI methods are defined in `ee/lib/gitlab/llm/utils/ai_features_catalogue.rb`. External methods (accessible via `aiAction`): chat, explain_vulnerability, resolve_vulnerability, summarize_review, generate_description, generate_commit_message, description_composer, summarize_new_merge_request, agentic_chat. Internal methods (not exposed via aiAction): categorize_question, review_merge_request, classify_code_review_mention_intent, duo_workflow, code_suggestions, etc.

### The Duo Chat Access Wall

Duo Chat is gated behind a PAID GitLab subscription. Free accounts get `"AI features are not enabled or resource is not permitted to be sent."` Non-chat methods (`explain_vulnerability`, `generate_description`, etc.) return empty error arrays but don't queue actual work — they silently skip on auth/licensing checks.

### Conversation Thread Security (Verified)

- ThreadFinder scopes to `current_user.ai_conversation_threads` — proper user isolation
- ThreadEnsurer uses `user.ai_conversation_threads.in_organization(organization)` — scoped
- aiMessages returns only current user's messages
- DeleteConversationThread uses `authorized_find!` with policy check
- Source proves proper scoping at the ActiveRecord level

### Duo Workflows

- `ee/app/graphql/mutations/ai/duo_workflows/create.rb` — creates workflows
- `ee/app/graphql/mutations/ai/duo_workflows/delete_workflow.rb` — deletes workflows
- `ee/app/graphql/mutations/ai/duo_workflows/update_tool_call_approvals.rb` — tool approval
- Resolvers in `ee/app/graphql/resolvers/ai/duo_workflows/`
- Tool execution authorization needs live testing with a licensed account

### Testing Constraints

- Free gitlab.com accounts CANNOT test Duo Chat or most AI features
- The aiAction mutation is in the schema but returns auth errors
- Non-chat methods exist but don't execute without proper licensing/context
- To properly test AI features, need a GitLab Ultimate trial or paid account
- Source code analysis is comprehensive but live testing is limited without a license

## CI_JOB_TOKEN Scope Analysis

**Key finding**: `ci_job_token_scope_enabled = False` by default on projects. When scope is disabled, CI_JOB_TOKEN has unrestricted project access within the user's permission scope. Only projects that explicitly enable scoping via settings or UI get the benefit of cross-project access limits.

- Confirm scope status: `GET /projects/:id` → check `ci_job_token_scope_enabled` field
- If disabled, any pipeline's CI_JOB_TOKEN can access any project on the instance
- The JWT-based job tokens (`Ci::JobToken::Jwt.encode`) use RSA signing and are time-limited — check `jwt_ci_cd_job_token_enabled?` on namespace settings
- `ci_push_repository_for_job_token_allowed = False` by default — CI_JOB_TOKEN cannot push to repos
- Source: `lib/ci/job_token/jwt.rb`, `app/models/ci/build.rb`, `app/finders/ci/auth_job_finder.rb`

### 9. Snippet Content Extraction

Public snippet content is accessible via GraphQL and REST:

**GraphQL method**:
```graphql
{
  snippets(explore: true, first: 3) {
    nodes {
      id title visibilityLevel
      blobs { nodes { plainData } }
    }
  }
}
```

**REST method**:
```
GET /snippets/public?per_page=3
GET /-/snippets/{id}/raw
```

Note: `snippets(explore: true)` only returns personal public snippets (by design). The `snippets(type: project)` query may time out on gitlab.com due to volume. Use REST API for reliable access.

### 10. GraphQL Field Suggestion Probing

GitLab has field suggestions ENABLED. When you query a non-existent field, the error message often includes a suggestion:
```json
{
  "message": "Field 'ciConfigPath' doesn't exist on type 'Project' (Did you mean `ciConfigPathOrDefault`?)"
}
```

Use this to discover hidden/undocumented fields without full introspection. However, `__type(name: "X")` introspection returns the **full schema** regardless of the type name on GitLab — not useful for targeted field probing. Use field suggestion via error messages instead.

### 11. User Enumeration via GraphQL

GraphQL `users()` query returns 100 users per query (with auth):
```graphql
{ users(search: "a") { nodes { username name id } } }
```

REST `/api/v4/users` returns 403 without auth (properly blocked). GraphQL user enumeration is an authenticated operation and returns usernames, names, IDs, but NOT emails (`publicEmail` and `commitEmail` are null).

### 12. Runner Token Exposure Check

- Project API response: `runners_token` field is **NOT present** (properly hidden)
- Group API response: `runners_token` key **IS present** but value is `None` (intentionally hidden)
- GitLab.com has `allow_runner_registration_token = false` — tokens are not exposed
- Runner registration tokens can only be accessed via the reset endpoint (`POST /runners/reset_registration_token`) with proper permissions
- Source: `app/services/ci/runners/reset_registration_token_service.rb`

### 13. Echo Resolver (Debug Endpoint)

GitLab exposes a debug/test GraphQL endpoint in production:
```graphql
{ echo(text: "hello") }
```
Response: `"h0d4r1-bugbounty" says: hello`. Includes the current username. No injection possible (text is stringified, not executed). Worth noting as an unnecessary production attack surface.

### 14. Permissions Confirmed Blocked (Good Security)

The following operations are all properly blocked for non-authorized users:
- CI variables on gitlab-org/gitlab → 403 Forbidden
- Runner tokens on gitlab-org/gitlab → 403 Forbidden
- Deploy keys on gitlab-org/gitlab → 403 Forbidden
- Group variables on gitlab-org → 403 Forbidden
- Project export download (unauthenticated) → 404
- Sudo endpoint → blocked without admin token
- All mutation authorization tests → properly blocked
- Unauthenticated user search on REST API → 403

This means new findings on gitlab.com will come from newer/less-audited features (AI, Duo Chat, bulk import) or from creative escalation chains, not from basic auth bypasses.

## Source Code Navigation

### SSRF Protection Code
- `gems/gitlab-http/lib/gitlab/http_v2/url_blocker.rb` — Main URL validation
- `gems/gitlab-http/lib/gitlab/http_v2/url_allowlist.rb` — Allow list checking
- `app/validators/public_url_validator.rb` — Public URL validator (blocks localhost)
- `app/validators/addressable_url_validator.rb` — Base URL validator
- `config/initializers/ssrf_filter_patch.rb` — SSRF Filter gem patch
- `lib/packages/ssrf_protection.rb` — Package registry SSRF protection

### HTTP Clients
- `gems/gitlab-http/lib/gitlab/http_v2/` — HTTP client classes
- `gems/gitlab-http/lib/net_http/connect_patch.rb` — Net::HTTP connect patch

### Workhorse (Go)
- `workhorse/internal/sendurl/sendurl.go` — URL sending
- `workhorse/internal/transport/transport.go` — Transport with IP restrictions

### Webhook/Integration Code
- `app/models/concerns/web_hooks/hook.rb` — WebHook URL validation
- `app/models/concerns/integrations/base/` — Integration HTTP clients

### Import Code
- `app/services/projects/import_service.rb` — Project import
- `app/services/import/validate_remote_git_endpoint_service.rb` — Git endpoint validation
- `lib/import/framework/url_blocker_params.rb` — Import URL blocker params

## Token & API Auth

**Token format**: `glpat-...` — GitLab Personal Access Token.

Store securely and construct auth header via string concatenation (never inline):
```python
token = open('/home/pro-g/.gitlab-token').read().strip()
headers = f"Authorization: Bearer {token}"
```

### Why Python scripts > bash for API testing

Bash quoting becomes error-prone with complex JSON payloads and Bearer token interpolation. The `$(cat token)` pattern fails on multi-line output, and quoted strings with nested JSON break terminal escaping.

**Preferred pattern**: Write a standalone `.py` script to disk and execute it:
```python
#!/usr/bin/env python3
import json
import urllib.request

token = open('/home/pro-g/.gitlab-token').read().strip()
API = "https://gitlab.com/api/v4"

def api(path, method="GET", data=None):
    url = f"{API}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-HackerOne-Research", "h0d4r1")
    req.add_header("Content-Type", "application/json")
    if data:
        req.data = json.dumps(data).encode()
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        try:
            return {"error": e.code, "body": json.loads(body)}
        except:
            return {"error": e.code, "body": body}
    except Exception as e:
        return {"error": str(e)}

def gql(query, timeout=20):
    req = urllib.request.Request("https://gitlab.com/api/graphql", method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-HackerOne-Research", "h0d4r1")
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps({"query": query}).encode()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except Exception as e:
        return {"_error": str(e)}
```

Save to `/tmp/` and run: `python3 /tmp/gitlab-test.py`

This avoids all bash quoting issues, handles JSON natively, and makes it easy to add conditional branching, parallel probes, and GQL queries.

### Common API Patterns

**Create a project**:
```python
api("/projects", "POST", {"name": "test-project", "visibility": "private"})
```

**List project webhooks**:
```python
api(f"/projects/{PID}/hooks")
```

**Create webhook with bypass URL**:
```python
api(f"/projects/{PID}/hooks", "POST", {
    "url": "http://1.0.0.127.nip.io:8080/test",
    "push_events": True,
    "enable_ssl_verification": False
})
```

**Create push mirror**:
```python
api(f"/projects/{PID}/remote_mirrors", "POST", {
    "url": "http://1.0.0.127.nip.io:8080/mirror.git",
    "enabled": True
})
```

**Upload generic package**:
```python
# Uses PUT with raw bytes
import http.client
conn = http.client.HTTPSConnection("gitlab.com")
url = f"/api/v4/projects/{PID}/packages/generic/{name}/{version}/{filename}"
conn.request("PUT", url, b"file content", {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/octet-stream"
})
```

**GraphQL introspection** (extract mutation names):
```python
r = gql("{ __schema { mutationType { fields { name } } queryType { fields { name } } } }")
mutations = [f["name"] for f in r["data"]["__schema"]["mutationType"]["fields"]]
queries = [f["name"] for f in r["data"]["__schema"]["queryType"]["fields"]]
```

**GraphQL batch via aliases** (bypass per-query rate limits):
```python
batch = "query { "
for i, path in enumerate(projects):
    batch += f'p{i}: project(fullPath: "{path}") {{ id fullPath visibility }} '
batch += " }"
r = gql(batch)
```

## Live Testing Results

Confirmed on gitlab.com with PAT (user: h0d4r1-bugbounty, project ID 82711314):

### Push Mirror — Creation-time URL Bypass
- `http://localhost:8080/mirror.git` → ❌ 400: "is blocked: Requests to localhost"
- `http://1.0.0.127.nip.io:8080/mirror.git` → ✅ **Created** (ID 4053688, `enabled: true`)

**Why this passed**: The URL was validated by `PublicUrlValidator` → `AddressableUrlValidator` → `UrlBlocker.validate!()`. DNS resolution of `1.0.0.127.nip.io` returned `1.0.0.127` — a public IP (APNIC), not loopback. The validator only blocks actual private/loopback IPs.

**Caveat**: The resolved IP `1.0.0.127` is a real public IP, so the request goes there, not to an internal service. The `1.0.0.127.nip.io` format is an INVERTED nip.io address — unlike `127.0.0.1.nip.io` (which correctly resolves to `127.0.0.1`), the inverted `1.0.0.127.nip.io` resolves to `1.0.0.127`.

**True SSRF risk on self-hosted**: If the instance has `allow_local_requests_from_web_hooks_and_services?` enabled, or if DNS rebinding protection is disabled via `dns_rebinding_protection_enabled?`, then domains that DO resolve to loopback (like `127.0.0.1.nip.io`, `localtest.me`, `lvh.me`) would work.

### Import System — Live HTTP Connection Attempts
- Import URL `http://1.0.0.127.nip.io:8080/test.git` → Returned: "Unable to access repository with the URL and credentials provided"
- This means GitLab's backend made a live HTTP connection to the provided URL. If the URL pointed to a valid Git server, the import would proceed.

**Gitaly URL flow** — The import system validates URLs in two stages:
1. **Ruby validation** (`app/services/import/validate_remote_git_endpoint_service.rb:55-63`) via `UrlBlocker.validate!()` with `Import::Framework::UrlBlockerParams`
2. **Gitaly execution** (`lib/gitlab/gitaly_client/remote_service.rb:11-22`) — URL passed as string to Go service over gRPC, Gitaly makes actual Git connection

**Parser differential risk**: Ruby's Addressable::URI vs Go's URL parser may handle URLs differently. A URL that passes Ruby validation could reach an unintended host when parsed by Gitaly's Go code. Attack scenario: domain that resolves to public IP during Ruby validation but switches to private IP when Gitaly connects (DNS rebinding) — especially if `dns_rebind_protection?` returns `false` (line 66-70: returns false when proxy is configured).

### Webhook Error Interpretation

After triggering webhooks via push events, the `response_status` field in webhook logs reveals:

| Status | Meaning | Typical Cause |
|--------|---------|--------------|
| `200` | Delivered | Normal webhook success (e.g., webhook.site) |
| `403` | Server responded | External server rejected request (e.g., nip.io server responded with 403) |
| `internal error` | Exception raised | `BlockedUrlError` caught by `rescue *Gitlab::HTTP::HTTP_ERRORS` (lines 107-123 of `web_hook_service.rb`) |
| `404`, `405` etc | Server responded | External server returned error |

For SSRF testing specifically: `internal error` after a redirect chain means UrlBlocker blocked the redirect target. The redirect IS followed by HTTParty, but the new connection goes through `NewConnectionAdapter#connection` → `UrlBlocker.validate_url_with_proxy!()` which blocks private IPs. The `BlockedUrlError` is included in `HTTP_ERRORS` (defined at `gems/gitlab-http/lib/gitlab/http_v2/exceptions.rb:21-24`) and caught by WebHookService.

The 403 on nip.io domains means the connection succeeded (passed UrlBlocker) but the external server responded with a rejection — different from `internal error` which means an exception was raised before/during connection.

### Generic Package Registry — PUT Works
- Generic package upload via `PUT /projects/:id/packages/generic/:name/:version/:file` → 201 Created
- Download URL is served by Workhorse (no redirect URL in response)
- Package registry SSRF protection (from `lib/packages/ssrf_protection.rb`) controls the download path, not the upload path

### Webhook Creation — URL Bypass (Confirmed)
- `http://127.0.0.1.nip.io:8080/webhook` → ❌ 422 (blocked by loopback check)
- `http://1.0.0.127.nip.io:8080/webhook` → ✅ **Created** (ID 79786008, `alert_status: "executable"`)
- `http://0x7f000001.nip.io:8080/test` → ✅ **Created** (hex IP format)
- `http://2130706433.nip.io:8080/test` → ✅ **Created** (decimal IP format)
- `http://0.nip.io:8080/test` → ✅ **Created** (0.0.0.0)

**Execution-time behavior**: The webhook was triggered by a push event. The external monitoring webhook (to webhook.site) successfully fired and delivered the payload. The SSRF-pattern webhooks (`1.0.0.127.nip.io:8080`) would be blocked at execution time because `UrlBlocker` resolves DNS and blocks loopback.

### GraphQL Introspection — Full Schema Access
- 3,873 types, 625 Mutation fields, 163 Query fields, 34 Subscription fields
- Field suggestions are ON — can enumerate every query/mutation/field

### GraphQL Introspection — Full Schema Access

GitLab uses HTTParty for HTTP requests, which follows redirects by default. The `NewConnectionAdapter#connection` (`gems/gitlab-http/lib/gitlab/http_v2/new_connection_adapter.rb:38-58`) validates the URL via `UrlBlocker.validate_url_with_proxy!()` at connection-creation time. When HTTParty follows a redirect, it creates a NEW HTTP connection to the redirect target, which goes through `NewConnectionAdapter` AGAIN. This means redirect targets ARE validated at execution time.

**Creation-time bypass confirmed** — Redirect URLs pointing to internal IPs are accepted at creation time:
- `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/test` → ✅ **Created** (ID 79898624)
- `https://httpbin.org/redirect-to?url=http://169.254.169.254/` → ✅ **Created** (ID 79898626)
- `https://httpbin.org/redirect-to?url=http://127.0.0.1:9200/ssrf-test` → ✅ **Created** (ID 79898654)
- `https://httpbin.org/redirect-to?url=http://169.254.169.254/latest/meta-data/` → ✅ **Created** (ID 79898655)

GitLab only validates the INITIAL URL at creation time. The redirect target (127.0.0.1, 169.254.169.254) is NOT checked — only the httpbin.org domain is validated.

**Execution-time behavior**: When the webhook fires, HTTParty connects to httpbin.org (allowed), gets the 302 redirect, then creates a new connection to the redirect target. The NEW connection goes through `NewConnectionAdapter#connection` → `UrlBlocker` which resolves DNS and blocks private IPs. So the redirect to `127.0.0.1` or `169.254.169.254` would be blocked at execution time on gitlab.com.

**Where this works** — If `dns_rebinding_protection_enabled?` is `false` on the instance (admin setting), the execution-time UrlBlocker uses `dns_rebind_protection: false` → hostname is NOT replaced with resolved IP → the check passes for loopback IPs if the hostname doesn't look like an IP.

**Attack potential**: On self-hosted instances where:
1. Admin has disabled DNS rebinding protection, OR
2. Admin has enabled `allow_local_requests_from_web_hooks_and_services?`

### Two-Phase Validation (Critical Architectural Insight)

GitLab validates URLs in TWO phases, with different protection levels:

| Phase | When | `dns_rebind_protection` | Code path |
|-------|------|------------------------|-----------|
| **Creation** | Validation on POST/PUT of webhooks/mirrors/imports | `false` (AddressableUrlValidator default) | `app/validators/addressable_url_validator.rb` → `UrlBlocker.validate!()` |
| **Execution** | Actual HTTP request when webhook fires / mirror syncs | `true` (UrlBlocker default) | `gems/gitlab-http/lib/gitlab/http_v2/new_connection_adapter.rb:38-58` → `UrlBlocker.validate_url_with_proxy!()` |

The `NewConnectionAdapter` (`gems/gitlab-http/lib/gitlab/http_v2/new_connection_adapter.rb`) validates URLs at connection time. Key code path:
```ruby
def connection
  result = validate_url_with_proxy!(uri)  # ← validates via UrlBlocker
  @uri = result.uri
  hostname = result.hostname
  http = super
  http.hostname_override = hostname if hostname
  ...
end
```

The `perform_request` override (`client.rb:49-71`) calls `httparty_perform_request` (the original HTTParty method). When HTTParty follows redirects, it creates new HTTP connections that go through `NewConnectionAdapter` again. Each redirect hop is independently validated.

**Attack surface**: If `dns_rebinding_protection_enabled?` is `false` on the instance, execution phase uses `dns_rebind_protection: false` → hostname is NOT replaced with resolved IP → the check passes even for loopback IPs. This creates a working SSRF if:
1. Admin has disabled DNS rebinding protection (unlikely on GitLab.com but common on self-hosted), OR
2. Admin has enabled `allow_local_requests_from_web_hooks_and_services?`

The `NewConnectionAdapter` at line 35 reads `dns_rebinding_protection_enabled` from options and passes it to `UrlBlocker.validate_url_with_proxy!()`. If nil/absent, UrlBlocker defaults to `true` — but the instance setting can override this.

### GraphQL CSRF Protection Analysis

**File**: `app/controllers/graphql_controller.rb`

```ruby
# Line 40-45
skip_before_action :verify_authenticity_token, if: -> {
  current_user.nil? || sessionless_user? || !any_mutating_query?
}
```

CSRF verification is SKIPPED when:
1. No user is logged in (`current_user.nil?`)
2. User authenticated via token/PAT (`sessionless_user?`)
3. Query has no mutations (`!any_mutating_query?`)

CSRF check ONLY applies when:
1. User logged in via session cookie
2. NOT using token auth
3. Request contains a mutation

**`sessionless_user?`** (`app/controllers/concerns/sessionless_authentication.rb:18-20`):
```ruby
def sessionless_user?
  current_user && @sessionless_sign_in
end
```
Set by `authenticate_sessionless_user!` which runs as a `prepend_before_action`. If the request has a valid PAT/API token, `@sessionless_sign_in = true` and CSRF is skipped — correct by design since tokens aren't vulnerable to CSRF.

**`any_mutating_query?`** (lines 226-232):
```ruby
def any_mutating_query?
  if multiplex?
    multiplex_param.any? { |q| mutation?(q[:query], q[:operationName]) }
  else
    mutation?(query)
  end
end
```

**`mutation?`** parsers the GQL and checks operation types:
```ruby
rescue GraphQL::ParseError
  true  # ← Malformed/unparseable queries are treated AS mutations (safe default)
end
```

**Relevance**: CVE-2026-4922 (fixed in 18.11.1) was a CSRF in the GraphQL API allowing unauthenticated users to execute mutations on behalf of session-authenticated users. The current code adds `!any_mutating_query?` to the skip condition. Test for bypasses:
- Form-encoded POST (bypasses JSON Content-Type CORS preflight)
- Multipart form data submission
- SameSite cookie bypass (SameSite=None without Secure)
- GET request → blocked by `disallow_mutations_for_get` (lines 204-209)
- Check if the `sessionless_user?` check can be bypassed with mixed auth types

### Generic Package Registry — PUT Uploads Work
- Generic package upload via `PUT /projects/:id/packages/generic/:name/:version/:file` → 201 Created
- Download URL is served by Workhorse internally (no user-facing redirect URL)
- SSRF protection from `lib/packages/ssrf_protection.rb` controls the **download** path (Workhorse `sendurl`), not the upload path
- Feature flag `generic_package_registry_ssrf_protection` (default: `false` on self-hosted) controls whether SSRF filtering applies to generic package downloads

### Cloudflare WAF — Testing Constraints
GitLab.com and api.syfe.com both run behind Cloudflare WAF which blocks CLI-based API calls with challenges. Workarounds:
- Use the browser tool to access pages (browser handles CF challenges via Browserbase)
- Use the SPA's own credentials (localStorage token) through the browser
- Python scripts with `urllib.request` + Bearer token sometimes work better than curl due to different TLS fingerprinting
- The AU region API (`api-au.syfe.com`) has weaker WAF rules than the main SG API

## Testing Approach

1. Create free gitlab.com account with @wearehackerone.com email
2. Create a personal access token with `api` scope (`glpat-...`)
3. Write Python scripts to `/tmp/` for complex API testing
4. Test each attack vector:
   - Package registry SSRF: Upload package with remote URL
   - Webhook SSRF: Create project webhook with DNS bypass URLs
   - Push mirror SSRF: Create mirror with bypass URLs
   - Import SSRF: Try importing from external Git URL
   - Redirect SSRF: Use public → private redirect chain
   - GraphQL: Run introspection queries, batch attacks
   - REST API: Test IDOR with different project IDs

## Report Template

Follow the standard structure:
1. Descriptive title (e.g., "SSRF in Package Registry Allowing Access to Internal Services")
2. Summary of vulnerability
3. Severity assessment (CVSS)
4. Steps to reproduce
5. PoC (code/screenshots)
6. Impact analysis
7. Recommended fix

### Report Directory Structure

```
~/Dev/REPORTS/GitLab/<SubmissionNumber>/<finding-name>/
├── REPORT.md           ← Standalone writeup (NOT in zip)
└── poc/
    ├── poc-pocname.py  ← Working exploit/PoC script
    └── submission.zip  ← PoC archive for HackerOne upload
```

### Pre-Submission Fact-Check

Before submitting, verify every claim:
1. **Check migration files** — `t.string` = varchar(255), `t.text` = unbounded. Don't claim full payload exposure for truncated columns.
2. **Trace the auth chain** — Read the guardian/policy file, not just the controller. Does it verify OWNERSHIP or just VISIBILITY?
3. **Test on live instance** — Verify the route exists, auth works as expected, and error codes match your claims.
4. **Check escalation claims** — Can you really enumerate IDs? Is the data really accessible cross-user? Or do rate limits / auth gates prevent it?
5. **Test redirects** — Does the endpoint follow redirects? Are redirect targets re-validated?
6. **Use the CVSS calculator** — https://gitlab-com.gitlab.io/gl-security/product-security/appsec/cvss-calculator/
7. **Be honest about severity** — A Medium finding with accurate analysis is worth more than a High with inflated claims.

### Report Directory Structure
```
~/Dev/REPORTS/<Program>/<SubmissionNumber>/<finding-name>/
├── REPORT.md           ← Standalone writeup, NOT in zip. Read this first.
└── poc/
    └── submission.zip  ← PoC code for reproduction
```

### Report File Structure

All reports follow this layout:
```
~/Dev/REPORTS/<Program>/<number>/<finding-name>/
├── REPORT.md           ← Standalone writeup, NOT in zip. Root of finding folder.
└── poc/
    └── submission.zip  ← PoC code + src/ + README. Triager: unzip && npm/node/python poc
```

Rules:
- REPORT.md sits at the finding folder root, never inside the zip
- The zip contains everything needed to reproduce (npm install + run, or python3 poc.py)
- Use published packages (not repo clones) — triager should not build from source
- Source repos stay at ~/Dev/<vendor>/ (e.g., ~/Dev/gitlab/, ~/Dev/discourse/)
- PoC/test code written to test the program stays at ~/Dev/ root (not in REPORT/)

## Reference Files

- `references/live-testing-results.md` — Full transcript of live API testing on gitlab.com (webhook/import/mirror/package registry results, DNS resolution data, GraphQL introspection output)
- `references/graphql-authz-sweep.md` — Comprehensive authorization sweep results (May 2026): mutation testing, REST API auth checks, source code analysis, runner token exposure check
- `references/ai-features-catalogue.md` — Complete AI features catalogue with execute methods, auth flow, source file locations, and constraint notes
