---
name: deployed-change-verification
description: Use when asked whether a change is live in production.
metadata:
  last_modified: Sat, 12 Sep 2026 00:00:00 GMT
---

# Verifying a deployed change is live

Use when asked whether a change is live in production: triggers on "is this
live", "did that ship", "are these changes deployed", "is the fix in prod",
and whenever the user references work you cannot find in code or chat history.
Verifies per layer — git SHA chain → CI runs → API deployment → web bundle →
mobile artifact → DB schema and rows — instead of trusting pipeline status, and
reports the raw evidence behind each claim.

Answers "is this live?" — a question that recurs for every shipped fix — with
per-layer evidence, not with pipeline status.

## Always-on rules

- **A green pipeline is not evidence a specific layer changed.** Verify each
  layer separately and say which layer each claim is about.
- **Every claim names its source:** a commit SHA, run id, timestamp, HTTP code,
  or DB row. No adjectives in place of an artifact.
- **Skipped ≠ failed.** Workflows that detect what changed legitimately skip
  steps; a skipped backend deploy says nothing about the API either way — go
  read the API's own deployment record.
- **Verification passes are read-only against production.** SELECTs only, no
  writes, no ad-hoc fixes: the deliverable is a verdict, not a repair.
- **Check the head of every unmerged branch before concluding** — "merged to
  main" and "promoted to prod" are different claims, and squash-merged branch
  refs still show as unmerged.
- **Close with what is NOT live:** pending merges, users whose experience
  predates the fix, clients that still need the new build (native APK / cached
  service worker).
- **Report shape the user expects:** per-layer verdict with the command behind
  each line, then the same write-up saved to `~/Dev/REPORTS/<project>/` and
  attached to the chat reply (`MEDIA:/abs/path`). Chat body = summary; the file
  carries the detail and the reproduce section.

## Procedure

1. **Pin what "the work" is.** Identify the commit(s) or PR(s). If the user
   references work you cannot find, run the discovery step at the bottom before
   answering anything.
2. **SHA chain.** `git rev-parse origin/main` (or the branch's remote head) →
   newest staging run's `headSha` → newest production-promotion run's
   `headSha`. All equal ⇒ nothing merged is sitting undeployed; a mismatch
   between the last two is the answer all by itself.
3. **Per-layer proof** (recipes in `references/live-verification-recipes.md`):

   | Layer | Proof |
   |---|---|
   | Git | `git log` / `git branch -r --merged`, unmerged branch refs |
   | CI | `gh run list/view --json headSha,conclusion,createdAt` |
   | API | the service's own deployment list; the deploy message carries the commit |
   | Web | HTTP status **plus** a grep of the deployed bundle for a string the change introduced |
   | Mobile | release tag targets the main branch, release workflow `headSha`, artifact host `last-modified` |
   | DB | probe for the objects the migration creates (no migration-log table) |
   | Runtime | health endpoint, then the changed flow if you can reach it |

4. **Prove app-side fixes from the artifact, not the UI.** Web bundles ship the
   whole UI: `curl -s <origin>/main.dart.js | grep -c "<new user-facing
   string>"` returns proof without a login or a browser session. The same trick
   tells you a cached client is stale when the string is missing.
5. **Prove user-facing impact from the live record.** For "is it fixed for
   <person>", read their rows (account/wallet state, audit trail,
   notifications, transactions) on the production branch and put their
   timestamps next to the deploy timestamp — a record created minutes after a
   deploy is the argument.
6. **Write the report** (disk + attachment), including a reproduce block with
   the exact commands.

## Discovery: when you cannot find "the work"

- **Absence from code and git history is not absence of the thing.** Support
  actions, resets, user rows and money movement live in the database, not in
  commits — `git log -S<name>` only finds names that were ever in code. Search
  the production DB before saying you have no record.
- **Exclude build artifacts when grepping the repo for a name or string.**
  Flutter `.dill` caches (`build/`, `.dart_tool/`, `test_cache`) embed the whole
  font catalogue and dependency strings, so a name hit there is noise.
- **Session history:** `session_search` can return zero results while the
  conversation exists. Fall back to a read-only query of `~/.hermes/state.db`
  (`file:...?mode=ro` URI) — see the recipes file for the table/column quirks.
- Say plainly which source you could not confirm and what you assumed, then ask
  the one question that actually changes the answer.

## Pitfalls

- **App-only commits skip backend deploy steps** in a change-detecting
  promotion workflow: the API keeps running its previous commit, which is
  correct and expected. Read the API deployment record and report it as "API:
  <sha> (unchanged — this commit touched only the app)" rather than implying a
  redeploy happened.
- **Health-check `ok` + HTTP 200 prove liveness, not content.** Only the bundle
  grep / release checks prove the new code is being served.
- **Cross-region serverless Postgres costs ~1s per query** (TLS + pooler) —
  batch related reads into one statement instead of looping queries.
- **Do not read the worktree `.env` to learn what production runs.** It points
  at the local/staging target; pull the environment's variables from the
  platform's CLI.

## Package

- `references/live-verification-recipes.md` — concrete commands per layer, the
  read-only Postgres recipe, and the session-history fallback.
- `references/yucanpay-env-and-queries.md` — the YucanPay/hodaripay environment
  map, table/column gotchas, and account-timeline queries.
- `scripts/readonly_pg_query.sh` — run a SELECT against a Railway-hosted Neon
  branch (guards the first token, redacts the password in its banner).
