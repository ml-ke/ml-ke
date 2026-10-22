# Supabase Local-Dev Stack — Exposure Map (verified 2026-06 on draiva)

Supabase local dev stacks are **admin-powerful and default-insecure on 0.0.0.0**. When reviewing a host with `supabase start` containers, check these:

## Ports (bound 0.0.0.0 by default)
- **54341 → Kong gateway** (the API front door — everything routes through here)
- 54342 → Postgres direct
- 54343 → Supabase Studio (admin UI)
- 54344 → Mailpit (email preview)
- 54347 → Logflare analytics

## Kong routes to probe (no auth needed on many)
- `POST /pg/query` with `{"query":"select current_user, current_database()"}` → **arbitrary SQL as `postgres` superuser**. Verified: returned `postgres`. This is total DB compromise: read/write all tables, `pg_read_file()` host files, `COPY ... FROM PROGRAM` = RCE in container.
- `GET /pg/roles` → role list incl. password hash column (postgres superuser can read it)
- `GET /rest/v1/<table>?select=...` → PostgREST CRUD. **Use the REAL anon key** (see below).
- `GET /pg/` → 200 (pg-meta reachable)

## Supabase Studio — NO LOGIN in local dev
- `GET http://<host>:54343/` → 307 redirect to `/project/default` = full admin dashboard (SQL editor, user manager). Anyone on the network.

## Keys — where they live, how they behave
- Kong config: `<project>/supabase/.temp/start-secrets/supabase_kong_draiva/secret-0`
- It literally contains: `headers.apikey == 'sb_secret_...' and 'Bearer <JWT>'` plus `sb_publishable_...` mapping.
- **service_role / secret key**: bypasses RLS = full read/write on ALL tables. Expected behavior — the finding is plaintext storage + 0.0.0.0 binding.
- **anon / publishable key**: RLS-restricted. Verified `42501` (permission denied) on reads, `401` on writes. If anon key CAN read/write user data → RLS misconfig = CRITICAL.

## Test order (fastest proof)
1. Extract both keys from kong secret-0 (regex: `headers\.apikey == '([^']+)' and 'Bearer ([^']+)'`)
2. anon: `curl -s http://127.0.0.1:54341/rest/v1/users?select=email,phone` → expect 42501
3. service: same call → expect data (proves exposure of real user data incl. emails, M-PESA payment rows)
4. `POST /pg/query` → superuser SQL proof
5. `curl -L http://127.0.0.1:54343/` → studio dashboard without login

## Fixes
- Publish ports on 127.0.0.1 only (compose `ports: "127.0.0.1:54341:8000"`)
- `docker compose down` when not actively developing
- Enable ufw: `sudo ufw default deny incoming; sudo ufw enable`
- Never commit `.temp/start-secrets/`; rotate service_role key before any deploy
- Remove dev user from docker group (`sudo gpasswd -d <user> docker`)
