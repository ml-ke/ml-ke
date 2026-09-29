# Per-layer verification recipes

Commands that produce the evidence, layer by layer. Substitute the platform's
own CLI where noted.

## Git

```bash
git fetch --all --prune
git rev-parse origin/main                     # the truth for "what is merged"
git log --oneline -15 --date=short --pretty='%h %ad %s'
git branch -r --merged origin/main            # content that reached main
git branch -r --no-merged origin/main         # inspect each — squash merges look unmerged
git log --all -S"<symbol>" --oneline          # which commit introduced a string
```

## CI runs (GitHub Actions)

```bash
gh run list --limit 15 --json databaseId,name,headBranch,status,conclusion,createdAt,displayTitle
gh run list --workflow=<deploy.yml>  --limit 3 --json headSha,conclusion,createdAt
gh run list --workflow=<promote.yml> --limit 3 --json headSha,conclusion,createdAt
gh run view <databaseId>            # job list — a "-" job is SKIPPED, not failed
gh run view <databaseId> --json headSha,conclusion,createdAt,event
```

Read the job list of the promotion run before claiming a layer deployed: a
step shown as skipped did not run.

## API deployment (Railway example)

```bash
railway whoami
railway deployment list --project <project-id> --environment production --service api
railway variables --project <project-id> --environment production --service api --kv
railway logs      --project <project-id> --environment production --service api
```

The newest `SUCCESS` deployment's metadata carries the commit it was built from
(`cliMessage: "production deploy: <sha>"`) — that is the API's real version,
regardless of what the promotion workflow reported. `--kv` prints raw
`KEY=value` lines; the JSON form truncates long values.

## Web bundle

```bash
curl -s -o /dev/null -w "%{http_code}\n" <origin>/
curl -s <origin>/main.dart.js | grep -o "<new user-facing string>" | sort -u
curl -s <origin>/main.dart.js | wc -c          # size sanity vs the built bundle
```

A hit proves the deployed web build contains the change. No hit for a string you
expect means the worker/CDN or the client's service worker is serving an older
build.

## Mobile artifact

```bash
gh release list --limit 8
gh release view <tag> --json tagName,targetCommitish,createdAt,publishedAt,isPrerelease
gh run view <release-run-id> --json headSha,conclusion,createdAt
curl -sI <artifact-url> | grep -iE "HTTP/|content-length|last-modified|content-disposition|etag"
```

Three checks together: the release targets the main branch, the release
workflow built the expected `headSha`, and the download host's object
`last-modified` moved after that build.

## Database schema and rows

There is usually **no migrations-tracking table** — verify a migration by
probing for something it created:

```sql
select table_name from information_schema.tables where table_schema='public' order by 1;
select column_name from information_schema.columns where table_name='<table>';
select indexname  from pg_indexes where tablename='<table>';
```

Read-only query recipe (no `psql`, no venv) for a platform-hosted Postgres:

```bash
# 1. pull the branch's connection string from the platform CLI
railway variables --project <project-id> --environment production --service api --kv \
  | grep '^DATABASE_URL=' | cut -d= -f2- > /tmp/db_url
# 2. one-shot query
uv run --quiet --with "psycopg[binary]" python -c '
import sys, psycopg
with psycopg.connect(open("/tmp/db_url").read().strip(), connect_timeout=25) as c:
    cur = c.cursor(); cur.execute(sys.argv[1])
    print("\t".join(d.name for d in cur.description))
    for row in cur.fetchall():
        print("\t".join("" if v is None else str(v) for v in row))' "select 1"
```

Or `scripts/readonly_pg_query.sh "<SELECT ...>" [production|testing]`.

Two traps: (1) an environment's variables are the only source of truth for what
that environment runs — the worktree `.env` is usually the local/staging target;
(2) `ilike`-based lookups beat guessing exact spellings when hunting a person or
entity by name.

Type-cast your joins: id columns stored as text/varchar (`entity_id`-style audit
columns) need `::uuid` to join a uuid primary key, otherwise Postgres raises
`operator does not exist: uuid = character varying`.

## Reading a live database without disturbing it

When you are reading a store the app is actively writing, open it read-only by
URI mode instead of the default read-write connection:

```python
sqlite3.connect("file:/path/state.db?mode=ro", uri=True)   # SQLite: avoids WAL/recovery writes
```

Same principle for Postgres: SELECT-only, small `LIMIT`s, never `SET`/DDL/DML.

## Session-history fallback (Hermes)

`session_search` can return zero results. The transcripts are readable in
`~/.hermes/state.db`:

```python
import os, sqlite3
con = sqlite3.connect("file:" + os.path.expanduser("~/.hermes/state.db") + "?mode=ro", uri=True)
con.execute("PRAGMA table_info(messages)").fetchall()   # always introspect first
```

Useful columns (they do NOT match the obvious names): `sessions(id, source,
title, started_at, last_activity_at, message_count)`, `messages(id, session_id,
role, content, tool_name, timestamp)`. Search with `content LIKE '%term%'` and
`ORDER BY timestamp`; group hits by session to find the conversation that did
the work. Prefer this over grepping log files, which hold only recent turns.
