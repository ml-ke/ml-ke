---
name: hodaripay-docs-workflow
description: >-
  Documentation workflow for the YucanPay repo (BongweKE/hodaripay). Use when
  shipping any feature, route, schema change, admin capability, or when the
  project asks for "developer documentation", "update the docs", "add to
  AGENTS.md", or a new-feature checklist. Encodes the repo's Documentation rule:
  every feature ships with the right docs updated (api.md, database.md,
  admin-platform.md, developer-guide.md, gotchas, future-roadmap.md) before it
  is considered done.
metadata:
  last_modified: Sat, 15 Aug 2026 00:00:00 GMT
---

# YucanPay Documentation Workflow

The repo's rule (see `AGENTS.md` → "Documentation rule"): **a feature is NOT
done until its docs are updated.** This skill is the playbook for doing that
quickly and completely.

## Plan first: diagrams before implementation (IMPORTANT)

Before writing a stateful/complex feature (onboarding/KYC, money movement,
collections, bulk batches, verification, account lifecycle), commit the design
to `docs/` while it is still a plan:

1. A **mermaid state machine** (states, transitions, terminal states, error/
   retry paths) in `docs/brs/NNNN-*.md` (or a design doc).
2. A **user journey** (flowchart/timeline) covering the happy path AND the edge
   cases — multi-wallet display, sandbox-unsupported writes, already-assigned
   resources, offline/retry.
3. **Validate the diagrams** against the vendor docs + the current code
   (endpoints, error codes, columns, existing UI) BEFORE coding; note what was
   verified live/empirically.
4. Only then start routes/widgets. Reference model:
   `docs/brs/0002-shortcode-collection.md` (state machine + journey +
   multi-wallet display rules) and `docs/brs/0001-personal-wallet-kyc-otp.md`.

This mirrors `AGENTS.md` → "Planning discipline". An ADR (see below) is still
required separately when the decision is architecturally significant.

## The docs map (`docs/`)

| Doc | Update it when… |
|-----|-----------------|
| `developer-guide.md` | setup, local gates, conventions, or the feature checklist change |
| `admin-platform.md` | a new admin role, capability, lifecycle action, or admin page ships |
| `architecture.md` | auth/roles, shell, data flow, or the doc index change |
| `api.md` | a route is added/changed/removed (update the API table) |
| `database.md` | a table/column/index is added or changed (update the schema section) |
| `backend-gotchas.md` | you hit a non-obvious backend pitfall (camelCase, `$N` placeholders, TLS, swallowed errors…) |
| `flutter-app-gotchas.md` | you hit a non-obvious Flutter pitfall (shell constraints, initState, widget-test tricks) |
| `brs/` | a stateful process is being planned — commit the state machine + journey BEFORE coding |
| `future-roadmap.md` | you research a future feature (never leave it only in chat) |
| `ci-cd.md` | the pipeline, environments, secrets, or promotion steps change |
| `decisions/` | you make an architecturally significant decision (new ADR + README row) |

## The workflow — when a feature ships

1. Make the code change + tests.
2. Grep the docs for the touched surfaces:
   - routes → `docs/api.md` table
   - schema → `docs/database.md`
   - admin → `docs/admin-platform.md` (roles table + capability note + feature)
   - a gotcha that cost you time → the matching gotchas doc + one line in AGENTS.md
   - future idea → `docs/future-roadmap.md`
   - architecturally significant decision → `docs/decisions/NNNN-title.md` + README row
3. If a brand-new developer-facing behavior shipped, also check
   `docs/developer-guide.md` (checklist items, conventions).
4. If the pipeline/deploy process changed (promote runbook, environment IDs,
   the no-git-source-on-prod rule), update `docs/ci-cd.md` — see ADR-0022.
4. If you wrote an ADR, run `bash scripts/check-adrs.sh` to validate.
4. Add/refresh the "Adding a new feature — the checklist" doc if a step changed.

## Key facts to keep accurate

- Admin routes are gated by **capabilities** (`lib/auth/roles.dart`); the
  `/admin` middleware only checks "is staff". Any new admin route MUST call
  `requireCapability(context, Capability.x)` — say so in docs if it's new.
- The API wire contract is camelCase (`jsonResponse`/`dataResponse` in
  `lib/http.dart`). Never document a snake_case response.
- Roles: `merchant`, `admin`, `support`, `content_editor`, `finance`.
- Migrations are idempotent and re-run on every deploy — document schema
  changes as additive (`IF NOT EXISTS`).
- Staging test staff accounts are seeded via `tool/seed.dart` (email+password).

## Gotcha: skills vs repo docs

`AGENTS.md` and `docs/` are versioned in the repo. Global opencode skills (this
one, `hodaripay-backend-debugging`, `hodaripay-choicebank`) live in
`~/.config/opencode/skills/` and are NOT in git — if you update a repo doc with
a lesson, also mirror the one-liner into AGENTS.md so it survives. Keep repo
docs as the source of truth; keep skills as fast-loading pointers to them.

## Versioning & release documentation

Every releaseable improvement bumps the app version
(`apps/hodaripay/pubspec.yaml`, e.g. `1.0.1+2 → 1.0.2+3`) and must update
`docs/changes.md`:

1. Add a **newest-first** versioned entry: what shipped, the commit/PR, and the
   **GitHub release link** (the tagged, signed APK/AAB). Keep the top
   "Releases" table current (version, tag, link, summary).
2. Release builds point at the staging API and are signed with the upload key
   (package `co.ke.yucanpay.app`) — installing a newer release APK updates an
   older one with the same signature.
3. Mirror the *what/why* into the relevant skill (e.g. `hodaripay-testing`,
   `hodaripay-backend-debugging`) and a one-liner into `AGENTS.md`.
4. Re-run the full gates before tagging a release.
