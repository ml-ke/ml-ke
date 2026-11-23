# Supabase self-hosted Studio auth bypass — worked case (2026-08-01)

Verified live against the official docker-compose self-hosted stack (studio
`2026.07.27-sha-cbb076d`, PostgREST 14.15, PG 17.6, Kong 2.8.1, gotrue v2.194.0).
Full repro + raw output: `~/Dev/REPORTS/Supabase/002-live-verification/`.

## The core pattern: env-flag-gated auth wrapper

`apps/studio/lib/api/apiWrapper.ts`:

```ts
if (IS_PLATFORM && withAuth) { const response = await apiAuthenticate(req, res); ... }
```

On self-hosted builds `IS_PLATFORM=false`, so the gate never runs — **no studio
API endpoint authenticates**, including handlers that pass `{ withAuth: true }`.
To enumerate the blast radius, grep for `apiWrapper(req, res, handler)` calls
that omit the options object (those are bare even on hosted), then note that on
self-hosted even `withAuth: true` routes are open.

Fix-status check that works: the vendor had an unmerged branch
(`middleware-studio`) adding an **opt-in** `KONG_MIDDLEWARE_KEY` →
`x-middleware-auth` header check — `if(middlewareAuth && ...)` skips auth when
the env var is absent (the default). Prove unfixed with
`git grep KONG_MIDDLEWARE_KEY origin/master` (empty) + confirm the branch still
exists unmerged.

## Endpoint map (all unauthenticated on self-hosted, port = studio)

- `GET /api/v1/projects/default/api-keys` — keys; values masked in Jul-2026+
  builds, **full on current master** (legacy anon/service JWTs returned unmasked
  by default; `?reveal=true` returns the `secret` type too).
- `GET /api/platform/projects/default/settings` — plaintext `jwt_secret`
  (unmasked on master: `jwt_secret: AUTH_JWT_SECRET`).
- `POST /api/platform/auth/default/users` — create email-confirmed user
  (admin-only op); `DELETE /api/platform/auth/default/users/<id>` also works.
- `POST /api/platform/pg-meta/default/query` — arbitrary SQL.
- `GET /api/platform/storage/default/buckets`, `GET /api/platform/props/project/default/api`,
  `GET /api/platform/projects/default` — config + key material.
- `POST /api/edge-functions/test` — SSRF fetch proxy, **no apiWrapper at all**.

## SQL-proxy privilege check BEFORE claiming RCE

The pg-meta query endpoint runs as `postgres` but NOT superuser on modern
images. Run this first:

```sql
select current_user, current_setting('is_superuser'), rolsuper, rolcreaterole, rolcreatedb
from pg_roles where rolname=current_user;
```

This build: `rolsuper=false, rolcreaterole=true, rolcreatedb=true` →
`pg_read_file`, `pg_ls_dir`, `COPY ... TO PROGRAM` all denied (42501). Still a
full data read/write + role/db creation primitive. Older images were superuser
→ RCE. Verify per-target; never claim RCE off `current_user=postgres` alone.

## JWT secret leak → forge: verify against the service's ACTUAL key

PostgREST's `PGRST_JWT_SECRET` may be a JWKS (`{"keys":[...]}`), not a bare
string. Extract via `docker inspect <rest-container> --format '{{range .Config.Env}}{{println .}}{{end}}'`,
parse the JSON, base64url-decode the `oct` key, compare to the leaked secret.
Match → forge works:

```python
# HS256 forge, role=service_role, iss=supabase, iat/exp now+3600
# send as BOTH Authorization: Bearer <tok> and apikey: <tok>
```

Status-code read: **200** = accepted (openapi/bucket list); **403 with
GRANT-hint** = token verified but role lacks table grants (still proof);
**401** = wrong secret — delete the claim. Test the same token against sibling
services (Storage `/storage/v1/bucket` accepted it too).

## Test-env pitfall: masked placeholder secrets

AI-provisioned sandboxes may hold scrubbed env values (`AUTH_JWT_SECRET=super-...ong`).
The settings endpoint then surfaces the docker-compose **default fallback**
(`super-secret-jwt-token-with-at-least-32-characters-long`) — which is
typically exactly what the running services use (confirmed via the JWKS oct
key). Do not dismiss the leak because container env looks masked; compare with
what services actually verify against.

## SSRF fetch-proxy pattern

Handler does `fetch(url)` and returns the upstream body. URL validation is a
host regex: on self-hosted it accepts **any host** with `/functions/v1/` in the
path. Test tiers (all returned bodies here): internet (`http://example.com/functions/v1/x`),
internal docker names (`http://supabase_pg_meta_draiva:8080/functions/v1/health`),
host via bridge gateway (`http://172.20.0.1:54341/functions/v1/rest/v1/` — the
`via: kong/2.8.1` header proves host-network reach). These proxies often skip
the apiWrapper entirely, so they are unauthenticated even where sibling routes
are gated.

## Deployment recon via docker

- `docker inspect <container> --format '{{range .Config.Env}}{{println .}}{{end}}'`
  → secrets, JWKS, URLs, feature flags.
- Kong declarative config (`kong.yml`, mounted): `request-transformer` plugins
  map new-style API keys (`sb_secret_...`) to legacy JWTs — full keys live in
  the host file; routes may carry `# TODO: validate apikey` comments. Verify
  whether those routes are data APIs or PostgREST admin-server health routes
  (`/live`, `/ready` — not data) before reporting.
- Mask any secrets in report output (display-only transformation).

## Duplicate-risk framing

"Studio has no auth" is public knowledge as an **ops** issue (GitHub discussion
#43852, Reddit, pentest-tools entry; docs tell operators to reverse-proxy the
dashboard). No hacktivity found on the API-layer flaw. Frame the report as the
application-layer auth gap (endpoints accept requests with no auth of their
own) and cite the vendor's own unmerged fix branch as evidence it is not
intentional design.
