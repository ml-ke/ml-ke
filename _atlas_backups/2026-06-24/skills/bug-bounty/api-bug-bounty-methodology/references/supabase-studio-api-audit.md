# Supabase Studio API — Source Code Audit

Date: 2026-06-01
Source: github.com/supabase/supabase (apps/studio/)
Branch: main
Files analyzed: 3,679 TypeScript files in apps/studio/

## Architecture

### Routing
- Primary: Next.js Pages Router (`apps/studio/pages/`)
- Secondary: App Router (`apps/studio/app/`) — only incident-status and incident-banner
- Proxy layer: `apps/studio/proxy.ts` — blocks all `/api/*` routes EXCEPT allowlisted ones on hosted platform

### Auth Model
- `lib/api/apiWrapper.ts` wraps route handlers with opt-in auth
- `{ withAuth: true }` enables JWT validation via `apiAuthenticate()`
- Without `withAuth`, the handler runs with NO auth check
- `IS_PLATFORM` constant gates platform-specific behavior (self-hosted vs hosted)

### Proxy Endpoint Filtering (proxy.ts)

Only these endpoints are accessible on hosted supabase.com:
```
/ai/sql/generate-v4       /ai/sql/policy          /ai/feedback/rate
/ai/code/complete         /ai/sql/cron-v2         /ai/sql/title-v2
/ai/sql/filter-v1         /ai/onboarding/design   /ai/feedback/classify
/ai/docs                  /ai/sql/parse-client-code
/get-ip-address           /get-utc-time           /get-deployment-commit
/check-cname              /edge-functions/test    /edge-functions/body
/generate-attachment-url  /incident-status        /incident-banner
/status-override          /api/integrations/stripe-sync
/content/graphql          /parse-query
```

Everything under `/api/platform/*` (org management, project CRUD, auth admin, pg-meta, etc.) returns 404 on hosted.

## Findings

### 1. check-cname — DNS SSRF Probe

**File**: `pages/api/check-cname.ts` (22 lines)
**Auth**: None — no apiWrapper call at all
**Proxy**: Allowed on hosted (`/check-cname` in allowlist)

```typescript
const { domain } = req.query
const result = await fetch(
  `https://cloudflare-dns.com/dns-query?name=${domain}&type=CNAME`,
  { headers: { Accept: 'application/dns-json' } }
).then((res) => res.json())
```

**Vulnerabilities**:
- `domain` param concatenated directly into URL — **no URL encoding**
- No validation on what `domain` can contain
- Could inject `#` or `&` to add arbitrary query params
- Response returned directly to caller (data exfiltration channel)

**SSRF potential**:
- DNS query via Cloudflare DNS resolver (`cloudflare-dns.com`)
- Limited SSRF since it's a DNS-only endpoint, but:
  - Can probe internal DNS namespaces
  - Can detect which internal services exist via DNS resolution
  - Could leak internal network structure

**Exploitation**:
```
GET /api/check-cname?domain=metadata.google.internal
GET /api/check-cname?domain=internal.service.consul
GET /api/check-cname?domain=169.254.169.254.nip.io
```

### 2. edge-functions/test — HTTP SSRF

**File**: `pages/api/edge-functions/test.ts` (128 lines)
**Auth**: None
**Proxy**: Allowed on hosted (`/edge-functions/test` in allowlist)

```typescript
const { url: requestUrl, method, body: requestBody, headers: customHeaders } = req.body
const validEdgeFnUrl = isValidEdgeFunctionURL(url, IS_PLATFORM)
if (!validEdgeFnUrl) { return 400 }
const response = await fetch(url, { method, headers: requestHeaders, body: finalBody })
```

**URL validation** (edgeFunctions.ts):

| Mode | Regex | Example matching URLs |
|------|-------|---------------------|
| Hosted | `^https://[a-z]{20}\.supabase\.(red\|co)/functions/v[0-9]{1}/.*$` | `https://abc...20chars.supabase.co/functions/v1/test` |
| Self-hosted (NIMBUS_PROD_PROJECTS_URL set) | `^https://[a-z]*\.<apex>/functions/v[0-9]{1}/.*$` | Configurable apex domain |
| Self-hosted (no env) | `^https?://[^\s/?#]+/functions/v[0-9]{1}/.*$` | `http://any-host/functions/v1/anything` |

**Self-hosted SSRF**: The local regex accepts ANY URL with `/functions/v[0-9]/` in the path. Can reach:
- Internal Kubernetes services: `http://kong:8000/functions/v1/test`
- Cloud metadata: `http://169.254.169.254/computeMetadata/v1/` — BUT must have `/functions/v1/` in path
- Internal DB: `http://postgres:5432/` — BUT must have `/functions/v1/` in path

**Bypass attempt**: URL parser quirks where `fetch()` normalizes differently than regex match.
- URL-encoded `#` as `%23`: regex sees `%23` (passes `[^\s/?#]+`), fetch decodes to fragment
  - `http://evil%23.supabase.co/functions/v1/test` → regex PASSES → fetch URL parses as `http://evil/` with fragment `.supabase.co/functions/v1/test`
  - But DNS resolution of `evil%23.supabase.co` fails in most DNS implementations

**On hosted**: Effectively restricted to the user's own Supabase edge functions (20-char project ref).

### 3. Auth User Endpoint — Missing Auth (Self-Hosted)

**File**: `pages/api/platform/auth/[ref]/users/[id]/index.ts` (39 lines)
**Auth**: None (`apiWrapper` without `withAuth: true`)
**Proxy**: BLOCKED on hosted (returns 404)

```typescript
const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_KEY!  // <-- SERVICE_ROLE KEY
)

// PATCH: Ban user
supabase.auth.admin.updateUserById(id as string, { ban_duration })

// DELETE: Delete user
supabase.auth.admin.deleteUser(id as string)
```

**Impact on self-hosted**:
- Anyone reaching this endpoint can delete or ban arbitrary users
- Full admin access via the `SUPABASE_SERVICE_KEY`
- No JWT validation, no role check

**Platform mitigation**: Proxy.ts blocks `/api/platform/*` on hosted Supabase.

### 4. API REST Endpoint — Service Key Exposure

**File**: `pages/api/platform/projects/[ref]/api/rest.ts` (39 lines)
**Auth**: None
**Proxy**: BLOCKED on hosted

```typescript
const response = await fetch(`${process.env.SUPABASE_URL}/rest/v1/`, {
  headers: { apikey: process.env.SUPABASE_SERVICE_KEY! },
})
```

Transmits the service role key in headers. Read-only (GET/HEAD only) but service key exposure is still a concern on self-hosted.

### 5. generate-attachment-url — Proper Auth

**File**: `pages/api/generate-attachment-url.ts` (88 lines)
**Auth**: ✅ `apiWrapper(req, res, handler, { withAuth: true })`
**Proxy**: Allowed on hosted
**Security**:
- ✅ Validates JWT claims first
- ✅ User ID check on file prefix (prevents accessing other users' files)
- ✅ Bucket enum validation (restricted to allowed buckets)
- ✅ Proper `createSignedUrls` with long expiry (10 years)

### 6. AI SQL generate-v4 — connectionString Injection

**File**: `pages/api/ai/sql/generate-v4.ts` (255 lines)
**Auth**: ✅ `apiWrapper(req, res, handler, { withAuth: true })`
**Proxy**: Allowed on hosted (`/ai/sql/generate-v4`)

**Interesting vector**: The `connectionString` parameter from the request body is passed through to `getTools()`, `executeSql()`, and `generateAssistantResponse()`. If a user can supply a `connectionString` pointing to an attacker-controlled database:
- The AI assistant would connect to the attacker's DB
- Schema introspection would send schema info to the attacker's DB
- Tool execution would write/read from the wrong database

**Mitigation**: Standard Supabase API auth requires the `Authorization` token, and the connection string is typically scoped to the user's project.

## API Endpoint Map

All discovered endpoints under `pages/api/`:

```
├── ai/
│   ├── code/complete.ts           ✅ Auth | ✅ Hosted | Code completion
│   ├── docs.ts                    ✅ Auth | ✅ Hosted | AI docs
│   ├── feedback/
│   │   ├── classify.ts            ✅ Auth | ✅ Hosted
│   │   └── rate.ts                ✅ Auth | ✅ Hosted
│   ├── onboarding/design.ts       ✅ Auth | ✅ Hosted
│   └── sql/
│       ├── check-api-key.ts       ✅ Auth | ⛔ Hosted
│       ├── cron-v2.ts             ✅ Auth | ✅ Hosted
│       ├── filter-v1.ts           ✅ Auth | ✅ Hosted
│       ├── generate-v4.ts         ✅ Auth | ✅ Hosted
│       ├── parse-client-code.ts   ✅ Auth | ✅ Hosted
│       ├── policy.ts              ✅ Auth | ✅ Hosted
│       └── title-v2.ts            ✅ Auth | ✅ Hosted
├── check-cname.ts                 ⛔ Auth | ✅ Hosted | 🔴 DNS SSRF
├── cli-release-version.ts         -- | ⛔ Hosted
├── connect/index.ts               -- | ⛔ Hosted
├── content/graphql.ts             -- | ✅ Hosted
├── edge-functions/test.ts         ⛔ Auth | ✅ Hosted | 🔴 HTTP SSRF (self-hosted)
├── enabled-features-overrides.ts  -- | ⛔ Hosted
├── generate-attachment-url.ts     ✅ Auth | ✅ Hosted
├── get-deployment-commit.ts       -- | ✅ Hosted
├── get-ip-address.ts              -- | ✅ Hosted
├── get-utc-time.ts                -- | ✅ Hosted
├── integrations/stripe-sync.ts    -- | ✅ Hosted
├── mcp/index.ts                   -- | ⛔ Hosted | MCP server endpoint
├── parse-query.ts                 -- | ✅ Hosted
├── platform/                      ⛔ Auth (varies) | ⛔ Hosted (blocked by proxy)
│   ├── auth/[ref]/users/          🔴 Missing auth (self-hosted)
│   ├── database/[ref]/            varies
│   ├── organizations/             varies
│   ├── pg-meta/[ref]/             ✅ Auth (query: POST)
│   ├── profile/                   ✅ Auth
│   └── projects/[ref]/            varies
```

## Priorities for Next Session

1. Test `check-cname` on live supabase.com to verify SSRF
2. Create account to test authenticated endpoint IDOR
3. Test edge-functions/test with crafted URLs for SSRF
4. Audit AI SQL endpoint for connectionString injection
