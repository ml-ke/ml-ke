# Container Crash-Loop Diagnosis — Worked Example

## Symptom

`docker ps` shows a container in `Restarting (1) 34 seconds ago` state — stuck in a crash-restart cycle, often every ~60s. In `docker stats --no-stream` it appears as `0B / 0B` memory because it never stays up long enough to be measured.

## Diagnosis Sequence

```bash
# 1. Confirm the restart cadence
docker ps -a --filter name=<name>

# 2. The repeated fatal line IS the root cause
docker logs <name> --tail 15

# 3. Find the compose project so you can read its config
docker inspect <name> --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}'

# 4. Check env vars for placeholder credentials
docker inspect <name> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -iE 'DB|DATABASE|PASSWORD|URL'
```

## Case: yucan-auth crash loop (GoTrue/Supabase auth)

Repeated fatal every 60s:

```
running db migrations: ... failed to connect to `host=db user=supabase_auth_admin database=postgres`:
failed SASL auth (FATAL: password authentication failed for user "supabase_auth_admin" (SQLSTATE 28P01))
```

Three contributing factors:

1. **Compose placeholder password**: `GOTRUE_DB_DATABASE_URL: "postgres://supabase_auth_admin:***@db:5432/postgres"` — the `***` was never replaced. `docker inspect` shows it verbatim (it is NOT redacted; that IS the value).
2. **Role has no password**: `psql -U postgres -t -c "SELECT rolname, rolpassword IS NULL FROM pg_authid WHERE rolname='supabase_auth_admin';"` → `t` (NULL). The Supabase image created the role but the password never got set.
3. **pg_hba.conf split**: `local all all trust` + `host all all 127.0.0.1/32 trust` but `host all all all scram-sha-256` for everything else. The auth container connects over the Docker network (`db:5432` → container IP, NOT 127.0.0.1) so it hits the scram path → auth fails.

## THE GOTCHA: localhost psql test is NOT a network test

```bash
# This SUCCEEDS even with a wrong password, because 127.0.0.1 is trust:
docker exec yucan-db psql "postgres://supabase_auth_admin:***@localhost:5432/postgres" -c "SELECT 1;"
```

A successful `psql` from inside the DB container via `localhost`/`127.0.0.1` proves NOTHING about the app's connection path when pg_hba has a trust rule for loopback. The app connects via the Docker network (service name → container IP) which hits the `scram-sha-256` rule.

Correct verification:

```bash
# Check pg_hba first
docker exec <db> grep -v '^#' /var/lib/postgresql/data/pg_hba.conf | grep -v '^$'

# Test from the APP's network namespace / with the app's exact URL
docker run --rm --network <project>_default postgres:15 psql "postgres://user:pw@<db-service>:5432/db" -c "SELECT 1;"
```

## Fixes

- **Stack unused?** `docker compose down` — stops the crash-loop churn (restart churn generates log/DNS/connection noise even when the container is tiny).
- **Stack needed?** Set a real password and wire it through:
  ```bash
  docker exec <db> psql -U postgres -c "ALTER ROLE <role> WITH PASSWORD '<real>';"
  ```
  Then replace `***` in the compose file and `docker compose up -d`.

## Pattern to Remember

A `Restarting (N) seconds ago` state + a repeated fatal in `docker logs --tail` is always diagnosable from the log line itself. The crash-loop cadence (~60s) is usually the app's startup timeout or restart policy backoff. Tiny containers in crash loops are individually negligible but generate constant churn — worth stopping if the stack is unused.
