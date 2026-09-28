# Supabase Reconnaissance

Date: 2026-05-31 to 2026-06-01
Program: HackerOne VDP (recognition-only, paid bounties planned 2026)
Scope: supabase.com, api.supabase.com, github.com/supabase, mcp.supabase.com/mcp

## Key Findings

### 1. Missing Auth on ~40 Admin API Endpoints (Self-Hosted)
**Root cause**: `apiWrapper(req, res, handler)` called without `{ withAuth: true }` on dozens of endpoints using `SUPABASE_SERVICE_KEY`.

**Affected paths**: 
- `/api/platform/auth/[ref]/users` — create, delete, ban users
- `/api/platform/auth/[ref]/users/[id]/factors` — MFA management
- `/api/platform/auth/[ref]/invite`, `/magiclink`, `/otp`, `/recover`
- `/api/platform/storage/[ref]/buckets/**` — full storage CRUD
- `/api/platform/projects/[ref]/api/rest`, `/api/graphql` — API proxies
- `/api/platform/projects/[ref]/api-keys/temporary`
- `/api/platform/organizations/**`, `/api/platform/profile/**`

**Protected by**: Vercel Edge proxy (only allowlisted URLs pass through on hosted). Completely open on self-hosted.

### 2. SSRF via /api/check-cname (Platform + Self-Hosted)
- No auth, domain param injected raw into fetch URL
- On hosted: in proxy allowlist, confirmed 200 at `/dashboard/api/check-cname?domain=x`
- Returns Cloudflare DNS CNAME lookup results

### 3. SSRF via /api/edge-functions/test (Self-Hosted, Restricted on Hosted)
- No auth, accepts URL from POST body, calls `fetch(url, ...)`
- Self-hosted regex: permissive (any URL with `/functions/v1/xxx`)
- Hosted regex: restricted to `*.supabase.co`

### 4. Supabase MCP Server
Source: github.com/supabase-community/supabase-mcp
- Well-secured: all API URLs hardcoded at construction
- Prompt injection protection via `<untrusted-data>` wrappers
- SQL execution by design (rejected as OOS)

## API Surface Map

### Management API (api.supabase.com/v1/)
- Needs Personal Access Token (sbp_...)
- 403 from Cloudflare (error 1010) from external IPs

### Project Internal APIs ({ref}.supabase.co)
- Auth Admin: `/auth/v1/admin/users` — works with service_role key
- Storage: `/storage/v1/bucket` — needs anon key
- PostgREST: `/rest/v1/` — needs anon key
- PG-Meta: `/pg-meta/v1/` — not directly accessible

### Dashboard API (supabase.com/dashboard/api/)
- Proxy blocks all non-allowlisted endpoints with 404
- Allowlisted: check-cname, edge-functions/test, parse-query, get-ip-address, content/graphql, ai/*, etc.

## Credentials Used
- Service Role Key: `eyJ...` JWT (role: service_role) — Auth Admin API
- PAT: `***redacted***` — Management API (blocked)
- Project: `vwzgvxpffyzuookzeidz`
