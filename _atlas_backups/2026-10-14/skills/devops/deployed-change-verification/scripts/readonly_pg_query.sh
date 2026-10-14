#!/usr/bin/env bash
# Read-only SELECT against a Railway-hosted Postgres branch.
#
# Usage: readonly_pg_query.sh "<SELECT ...>" [production|testing]
# Env overrides: PGQ_PROJECT_ID, PGQ_SERVICE, PGQ_ENVIRONMENT
#
# Defaults target the YucanPay project (prod service `api`,
# staging service `api-staging`). Refuses anything whose first token is not
# SELECT/WITH, and redacts the password in the banner it prints to stderr.
set -euo pipefail

SQL="${1:?usage: readonly_pg_query.sh \"<SELECT ...>\" [production|testing]}"
ENV_NAME="${PGQ_ENVIRONMENT:-production}"
PROJECT_ID="${PGQ_PROJECT_ID:-bf850671-ac0a-4a93-80c0-67d0992c8f6e}"

if [ -n "${PGQ_SERVICE:-}" ]; then
  SERVICE="$PGQ_SERVICE"
else
  case "$ENV_NAME" in
    production) SERVICE="api" ;;
    testing)    SERVICE="api-staging" ;;
    *) echo "env must be 'production' or 'testing' (or set PGQ_SERVICE)" >&2; exit 2 ;;
  esac
fi

# first token must be SELECT or WITH (read-only)
FIRST="$(printf '%s' "$SQL" | awk '{print toupper($1)}')"
case "$FIRST" in
  SELECT|WITH) ;;
  *) echo "refusing: read-only helper, first token was '$FIRST'" >&2; exit 3 ;;
esac

URL="$(railway variables --project "$PROJECT_ID" --environment "$ENV_NAME" \
        --service "$SERVICE" --kv | grep -E '^DATABASE_URL=' | cut -d= -f2- || true)"
if [ -z "$URL" ]; then
  echo "could not read DATABASE_URL from Railway ($ENV_NAME / $SERVICE)" >&2; exit 4
fi

echo "# target: $ENV_NAME ($SERVICE) $(printf '%s' "$URL" | sed -E 's#(://[^:]+:)[^@]+@#\1***@#')" >&2

SQL="$SQL" DBURL="$URL" uv run --quiet --with "psycopg[binary]" python - <<'PY'
import os, psycopg
with psycopg.connect(os.environ["DBURL"], connect_timeout=25) as c:
    cur = c.cursor()
    cur.execute(os.environ["SQL"])
    if cur.description:
        print("\t".join(d.name for d in cur.description))
        for row in cur.fetchall():
            print("\t".join("" if v is None else str(v) for v in row))
    else:
        print("(no rows)")
PY
