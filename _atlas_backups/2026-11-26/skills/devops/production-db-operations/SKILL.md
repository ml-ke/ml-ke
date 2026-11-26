---
name: production-db-operations
description: Use when writing to a live production database or reconciling a customer row against a provider's record. Target identification, before-images, narrow writes, undo windows, vendor existence checks, mirror-vs-ledger diffs, and "your write broke X" forensics.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [production, database, ops, irreversible-writes, forensics, verification]
    related_skills: [systematic-debugging, postgresql-table-design]
---

# Production Database Operations

Irreversible writes against a live DB — credential resets, account-linkage
repairs, backfills, hard deletes — are an ops task, not a debugging task. Two
failures dominate the class: **writing to the wrong row**, and **repairing a
premise that was never true**. Both are cheap to prevent and expensive to
unwind.

## When to Use

- A request to change one identified account's credentials, PIN, status,
  balance, or account linkage in a live environment.
- Any direct `UPDATE`/`DELETE`/`INSERT` against production that no application
  route performs — including "just fix this one row" asks.
- Someone reports that an earlier write removed or broke data, and you must
  establish what actually happened before touching anything.
- Backfills, reconciliations, or manual data moves between environments.

Not for: schema changes (that is a migration), or diagnosing a bug that has a
reproduction — use `systematic-debugging` for those.

## Always-on rules

1. **Investigation is autonomous; the write is not.** Recon, read-only queries
   and evidence gathering you do freely. Any irreversible prod write gets an
   outlined plan first — the rows, the columns, the undo — and an explicit
   go-ahead. When the user asks for an outline, stop at the outline.
2. **Identify the row by more than the identifier you were handed.** A phone
   number, account number or ID comes from a human and is the likeliest thing
   to be wrong. Corroborate with a second independent field (linkage rows, KYC
   name, email) before writing.
3. **Map every login alias, not just the identifier.** Accounts are reachable
   by more than one credential (phone *and* email/username), and the alias
   columns are UNIQUE. When one human has two rows — the usual cause being a
   signup with a typo'd identifier — the alias often sits on the *wrong* row:
   credentials written to the correct-looking row still sign them into the
   empty account, and the UNIQUE constraint blocks simply copying it back. List
   which row owns each alias, decide which row is canonical, then free and
   re-point the alias deliberately.
4. **Capture the before-image VALUES.** Not a prefix, not "was set" — the full
   value, held until the task completes. Logging `left(password_hash,7)` is not
   an undo; it is a receipt for an irreversible change.
5. **Know the undo before you start, and its expiry.** Snapshot, point-in-time
   branch, or nightly dump — confirm it exists and how long it lives, and state
   the deadline in your report because it changes the user's decision.
6. **Never repair a linkage on a single-field match.** Creating or re-pointing
   an account link is a money-movement change: a wrong link shows someone
   else's balance and routes someone else's collection account. Require
   verified identity before linking anything financial.
7. **Verify at the deployed layer, not the row.** Write → the live service reads
   it back → the user-facing path works. Revoke any session you minted for the
   verification.

## Procedure

### 1. Establish the target
- Resolve the identifier to **exactly one** row. More than one match means ask,
  not pick.
- Normalise stored formats before comparing (leading zero, country code,
  spacing, case): the same identifier is often stored in two shapes across
  environments.
- Check **linkage existence** — does this row actually own the related rows the
  request implies (account/wallet/profile/transactions)? A target with zero
  linkage cannot be "the customer whose account we just broke". Say that out
  loud instead of writing.
- Search for **near-miss identifiers** (edit distance 1) and show both
  candidates. A transposed digit is the standard cause of a wrong-target write,
  and the neighbour is usually the real, fully-provisioned account.
- Build the **alias map** for the candidate row(s): which row owns each unique
  login field (email, username, alternate phone). A row created minutes ago
  holding the customer's real email while the older row holds a near-miss phone
  is the signature of a duplicate signup, and it decides which row is canonical.

### 2. Capture the before-image
- `SELECT` the exact columns you will overwrite, full values, before any write.
- Record `updated_at`/version so you can later prove only your write touched it.

### 3. Write narrowly
- One statement, only the intended columns, keyed on the **primary key** — never
  on the identifier the requester typed.
- Perform the side-effects the application would perform (e.g. revoke refresh
  tokens when a password changes) instead of leaving a state the app itself can
  never produce.
- If the system has an audit table and your write bypasses the code that
  normally writes one, insert the audit row yourself.

### 4. Verify end-to-end
- Exercise the real entry point (login, wallet read, list page) and assert the
  specific fields changed.
- Credential changes: authenticate with the new value, exercise the new
  PIN/password, then revoke the session you created. Login and PIN endpoints are
  rate-limited — one deliberate attempt, never a probe loop.
- After a merge, assert the alias in both directions: each login identifier
  (phone, email) resolves to the same surviving id, and the retired identifier
  is now rejected. That pair is the only proof the duplicate is really gone.
- Read the row back and diff against step 2. Anything you did not intend to
  change is a bug in your statement.

### 5. Clean up
- Delete throwaway scripts and any temp file holding a connection URL, token or
  hash; drop snapshot branches you created; confirm the working tree is clean.

## Reconciling a mirror row against the system of record

A local row that mirrors an external provider (ledger, payment rail, identity
vendor) can be incomplete without anything having been lost. Verify against the
provider, not against expectations:

- **Ask the provider whether the identity exists at all.** Most expose a lookup
  keyed by the same identifier. A hard "not found" is the strongest possible
  answer to "did we unlink them?" and it stops an unnecessary repair. Drive the
  call through the repo's own client/SDK so auth and signing are correct by
  construction — a hand-rolled request signature wastes a session and proves
  nothing when it fails.
- **Diff field by field, and expect the fields nothing refreshes to be stale.**
  Mirrors are written by two paths: an event/webhook handler, and a poll. A
  field only the handler writes stays at its column default when that specific
  event never arrived — even on an account that is otherwise fully provisioned.
  Read a default as "unknown", not as "failed/never verified", and confirm
  against the provider's live value plus the event log for that identifier.
- **Backfill deliberately**: one statement, only the derived columns,
  before/after captured, an audit row if the app's audit table would otherwise
  miss it, then re-read and re-run the diff until it is empty.
- **Never invent linkage rows to make a screen look right.** If the provider
  has no account for that identity, the correct output is "nothing to link"
  plus the evidence — not rows pointing at somebody else's account id.
- **Check what a legitimate account of that type looks like** before calling a
  missing table a gap: personal and business flows often use different linkage
  tables, so one of them legitimately has no profile row.
- **Duplicate rows are a resolution decision, not a write.** Moving a unique
  alias, reassigning dependent records (tickets, notifications, consent rows),
  and deleting the duplicate needs sign-off — deletion usually cascades, so
  reassign first and say what will disappear.

### Executing a consolidation (once it is approved)

- **Merge INTO the row that owns the provider linkage**, never into the newer
  empty one. External ids (`choice_user_id`, provider account ids) are what
  callbacks and balance reads resolve against and cannot be re-invented.
- **One transaction, in this order**: re-point the dependents you must keep →
  free the UNIQUE alias on the losing row → set the alias *and* the agreed
  credentials on the winner → revoke the winner's live sessions → write the
  audit row → delete the loser. Credentials written before the merge land on a
  row that is about to disappear.
- **Enumerate every referencing column, not just `user_id`.** Merge-relevant
  FKs also hide in `sender_id`, `actor_user_id`, `assigned_to`,
  `assigned_lead_id`. One child table can point at the parent through two
  different columns at once (a support message has both `ticket_id` and
  `sender_id`), so re-point each one explicitly.
- **Do not expect the delete to warn you.** When every FK is `ON DELETE
  CASCADE` / `SET NULL` the delete always succeeds and silently takes the dead
  row's history with it. Print a per-table dependent census first, re-point
  what must survive, and state in the report what you accepted losing.
- **Verify the merge from outside**: both aliases authenticate to the same
  surviving id, the retired identifier now fails with invalid credentials, the
  reassigned records still read back, and the app's own status endpoints
  (`me` / account / wallet) report the merged account as complete.

Worked field map and the reconcile loop:
`references/mirror-vs-ledger-reconciliation.md`.

## When someone says "your write broke X"

Do not accept the premise, and do not start repairing — the repair is often a
bigger write than the original. Keep this order:

1. State exactly what the statement touched: tables and columns.
2. Rule out cascade: list triggers on the affected and related tables.
3. Enumerate every table carrying a user/owner FK and count rows for the
   target; compare with a **known-good account**. Zero rows in linkage tables is
   evidence that nothing was removed — not evidence that you removed it.
4. Check the near-miss identifier from step 1. The reported "lost" account is
   frequently a different person with a similar number.
5. Only then propose the repair, with the evidence and whatever identity checks
   are still missing, and wait for approval.

Two things that make this fast: one recon script parameterised by DB URL so
prod and staging can be compared, and printing the suspect account beside a
known-linked one.

Concrete Postgres queries (trigger listing, FK sweep, near-miss search,
pre-delete cascade check, dependent census, the merge transaction skeleton, the
post-merge verification matrix, point-in-time restore, credential verification)
live in `references/postgres-evidence-recipes.md`.

## Reporting

- Lead with what the data shows, then what you propose. No apology paragraph and
  no restating the request.
- Name the exact row(s) touched and paste the verification output, not a claim
  that it worked.
- Give hard deadlines their own line (undo window expiry, rate-limit windows) —
  they change the decision more than anything else in the report.
- Separate **verified** from **proposed/unverified** explicitly; never present a
  forensic conclusion as certain when it rests on one match.

## Pitfalls

- **A single matching row proves the identifier exists, not that it is the
  intended account.** Check the linkage tables too; an account created minutes
  ago with no wallets, profiles or transactions is a new signup, not a customer
  who lost anything.
- **Storing only a hash prefix destroys the undo.** A bcrypt plaintext cannot be
  recovered from the new hash, so the pre-image is the only rollback.
- **A crash in post-commit verification is not a failed write.** Printing a row
  that holds a `DateTime` (or any non-JSON-serialisable type) throws *after*
  `COMMIT`; the data is already applied. Re-read the state before retrying, or
  you apply a merge or credential change twice.
- **Migrations are schema, not data.** One-off row fixes belong in a direct
  write against the target branch; a migration file re-runs on every environment
  and every promotion.
- **A revoked session does not delete client-side state**, and a fresh login
  re-reads the DB. If a user reports losing data after a credential change, the
  data question is answered in the DB, not in the app cache.
- **A connection-string quirk belongs to a driver, not to the database** (e.g.
  `channel_binding=require` breaks Node `pg` but is fine for other drivers). Do
  not "fix" a URL that already works for the runtime in use.
