# Postgres evidence recipes

Read-only queries for identifying a target and for proving/disproving that a
write caused a reported loss. Postgres 13+; snippets assume a throwaway script
in the project's own driver so placeholders and type decoding match the app.

## 1. Did any trigger cascade?

```sql
SELECT c.relname AS tbl, t.tgname, pg_get_triggerdef(t.oid) AS def
FROM pg_trigger t
JOIN pg_class c ON c.oid = t.tgrelid
WHERE NOT t.tgisinternal
  AND c.relname IN ('users', 'refresh_tokens', 'wallets', 'individual_accounts')
ORDER BY 1;
```

Zero rows means an `UPDATE` on the parent table could not have propagated into
linkage tables — the strongest single fact when answering "your write unlinked
my account".

## 2. What does this account actually own?

Enumerate the FK columns that exist in *this* schema, then count per table —
do not assume table names.

```sql
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND column_name IN ('user_id', 'owner_id', 'account_id', 'choice_user_id')
ORDER BY 1;
```

Then, per table, `SELECT count(*) FROM "<t>" WHERE user_id = $1`. Wrap each in
try/catch — a table may exist with a differently typed FK.

Read the result as a shape, not a number: a provisioned merchant shows rows in
the account, wallet and sync tables; an unprovisioned signup shows none. Print
the suspect **and** a known-good account from the same environment side by side;
the contrast is the argument.

## 3. Near-miss identifiers

Transposed digits are the classic wrong-target cause. With `fuzzystrmatch`:

```sql
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;  -- needs superuser on some hosts
SELECT id, phone, full_name,
       levenshtein(right(regexp_replace(phone, '\D', '', 'g'), 9), $1) AS dist
FROM users
WHERE levenshtein(right(regexp_replace(phone, '\D', '', 'g'), 9), $1) <= 2
ORDER BY dist;
```

Without the extension (common on managed Postgres), normalise both sides in the
script and diff in code — list every row's 9-digit normalised value and compute
the distance there. Either way, present **both** candidates to the requester
rather than choosing.

## 4. Before-image and narrow write

```sql
-- before
SELECT id, phone, full_name, updated_at, password_hash, pin_hash
FROM users WHERE id = $1;

-- write, keyed on the primary key only
UPDATE users
   SET password_hash = $2, pin_hash = $3, pin_updated_at = NOW(),
       pin_failed_attempts = 0, pin_locked_until = NULL, updated_at = NOW()
 WHERE id = $1;

-- dependent side-effect the app would also do
UPDATE refresh_tokens SET revoked_at = NOW()
 WHERE user_id = $1 AND revoked_at IS NULL;
```

Keep the full `password_hash`/`pin_hash` strings in your notes until the task is
finished. A truncated prefix cannot restore anything.

Hash with the **same library and cost the app uses** (e.g. `package:bcrypt`
`gensalt(logRounds: 10)` producing `$2a$10$…`) and self-check with
`checkpw` before writing — a dialect mismatch silently locks the account out.
Leaving a nullable hash column `NULL` often means "provider default still in
effect", which is a different state from "explicitly set to that value"; pick
deliberately and check what the read path does with `NULL`.

## 5. Undo: point-in-time branch (Neon example)

```bash
neon branches create --project-id <project> --name pitr-<ts> \
  --parent <branch-or-timestamp-or-lsn>
neon connection-string pitr-<ts> --project-id <project> \
  --role-name <role> --database-name <db>
# read the pre-image values on the new branch, restore them on the real one,
# then:
neon branches delete pitr-<ts> --project-id <project>
```

`--parent` accepts a branch name, a timestamp or an LSN (`neon branches create
--help` is the ground truth for the exact form). Restore windows are plan- and
branch-dependent, and managed providers refuse snapshots on non-root branches —
verify the window **before** writing, because after the window closes the
pre-image is gone and only an app-level recovery flow remains.

## 6. Verify through the deployed service

```bash
# 1. the service reads the DB you just wrote → proves they are the same store
curl -s -X POST https://<api>/auth/login -H 'Content-Type: application/json' \
  -d '{"phone":"<phone>","password":"<new>"}'

# 2. the new PIN/authoriser value works
curl -s -X POST https://<api>/auth/pin/verify \
  -H "Authorization: Bearer <token>" -H 'Content-Type: application/json' \
  -d '{"pin":"<new-pin>"}'

# 3. revoke the session you created
curl -s -X POST https://<api>/auth/logout -H 'Content-Type: application/json' \
  -d '{"refreshToken":"<refresh>"}'
```

Assert on specific response fields (`ok: true`, `mustChange: false`), not on a
200 alone. Login and PIN verification are rate-limited per identity and per
user — one deliberate attempt; a probe loop locks the account you are trying to
fix.

## 7. Comparing environments

Parameterise the same script by connection URL and run it against prod and
staging before concluding anything. An identifier present in one branch and
absent in the other explains most "but it works in the app" reports without any
code change.

## 8. Before a delete/merge: which references block, which cascade

Never delete a parent row on the assumption that the FK will refuse. Read the
behaviour first:

```sql
SELECT c.conname, c.confdeltype,
       child.relname AS child_table, a.attname AS child_col
FROM pg_constraint c
JOIN pg_class child ON child.oid = c.conrelid
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
WHERE c.contype = 'f' AND c.confrelid = '<parent>'::regclass
ORDER BY child.relname;
```

`confdeltype`: `c` cascade · `a` no action · `r` restrict · `n` set null.
Map it to a coarse type first (`confdeltype::text`) — a driver that does not
register the `"char"` type returns it as raw bytes.

Zero rows with `'a'`/`'r'` means **nothing will block the delete**, so nothing
will warn you either: the row disappears together with every cascaded child.
Re-point what must survive before deleting.

## 9. Dependent census before a merge or delete

One query, one round-trip, so a slow cross-region pooler does not turn recon
into minutes:

```sql
SELECT 'refresh_tokens' AS t,
       count(*) FILTER (WHERE user_id::text = $1) AS keep,
       count(*) FILTER (WHERE user_id::text = $2) AS drop
FROM refresh_tokens
UNION ALL SELECT 'wallets', count(*) FILTER (WHERE user_id::text = $1),
                            count(*) FILTER (WHERE user_id::text = $2) FROM wallets
UNION ALL SELECT 'support_tickets', count(*) FILTER (WHERE user_id::text = $1),
                                    count(*) FILTER (WHERE user_id::text = $2) FROM support_tickets;
-- repeat for every table from §8
```

Cast both sides to `text`: mixing a `uuid` column with a text parameter raises
`operator does not exist: text = uuid` and kills the whole union. Run it before
the write and quote the numbers in the report, so the user sees exactly what the
delete would take with it.

## 10. Merge two rows into one (transaction skeleton)

Order matters — an alias is UNIQUE, so it must be released before it can be
reclaimed, and credentials must land on the row that survives.

```sql
BEGIN;
-- 1. dependents that must survive (each column separately!)
UPDATE support_tickets SET user_id = $keep, status = 'resolved', resolved_at = NOW()
 WHERE user_id = $drop;
UPDATE ticket_messages SET sender_id = $keep WHERE sender_id = $drop;

-- 2. free the unique alias on the losing row
UPDATE users SET email = NULL, updated_at = NOW() WHERE id = $drop;

-- 3. alias + agreed credentials on the winner
UPDATE users
   SET email = $email, password_hash = $pw, pin_hash = $pin,
       pin_updated_at = NOW(), pin_failed_attempts = 0, pin_locked_until = NULL,
       updated_at = NOW()
 WHERE id = $keep;

-- 4. the side-effect the app would perform on a password change
UPDATE refresh_tokens SET revoked_at = NOW()
 WHERE user_id = $keep AND revoked_at IS NULL;

-- 5. audit row: actor = an admin user id, before/after JSON, plus the reason
INSERT INTO audit_logs (user_id, action, entity_type, entity_id, before, after)
VALUES ($admin, 'ops_consolidate_user_accounts', 'users', $keep, $before, $after);

-- 6. retire the duplicate (cascades what you deliberately left behind)
DELETE FROM users WHERE id = $drop;
COMMIT;
```

Driver notes for the throwaway script: `pool.runTx(fn, settings:
TransactionSettings())` (that settings object is **not** `const`), and stringify
`DateTime` values before `jsonEncode` — a throw inside the post-commit print is
indistinguishable from a failed transaction until you re-read the rows.

## 11. Post-merge verification matrix

| Check | Expected |
|---|---|
| login with each alias (phone, email) | same surviving `user.id` |
| login with the retired identifier | `401 E_INVALID_CREDENTIALS` |
| PIN verify with the new PIN | `ok: true`, `mustChange: false` |
| `GET /auth/me` (or equivalent) | onboarding/complete flags true |
| primary read endpoint (wallet/account) | the linked provider account id, live balance source |
| reassigned records (ticket, messages) | still readable under the survivor |
| row count for either identifier | exactly one |

Then revoke the sessions the verification minted, delete the scripts, and
confirm the working tree is clean.
