---
name: hodaripay-fintech-cicd
description: >-
  CI/CD compliance playbook for fintech applications in the YucanPay repo.
  Use when setting up, reviewing, or modifying deployment pipelines, CI
  gates, environment configs, or release processes for any fintech app.
  Covers the controls a fintech pipeline needs that a generic pipeline
  doesn't: segregation of duties, signed artifacts, audit evidence, secret
  scanning, dependency audit, rollback, and kill-switches. Triggers on
  "fintech CI/CD", "compliance pipeline", "deployment controls", "release
  gate", "SOC2", "audit trail", "sox compliance", "change management".
metadata:
  last_modified: Sat, 26 Aug 2026 00:00:00 GMT
---

# YucanPay Fintech CI/CD Compliance Playbook

A payments platform touching real money needs CI/CD controls beyond
what a generic web app requires. This playbook covers the fintech-specific
gates and controls that a generic pipeline misses.

## Current pipeline (already implemented)

| Control | Status | Where |
|---------|--------|-------|
| PR gates (analyze + test) | ✅ | `pr.yml` |
| Migration lint (idempotent, no destructive) | ✅ | `pr.yml` |
| Staging-first: `main` → staging auto | ✅ | `deploy-staging.yml` |
| Gated production: manual dispatch + env approval | ✅ | `promote.yml` |
| Branch-isolated DBs (Neon) | ✅ | `testing` / `production` |
| Signed release APKs (CI-built) | ✅ | `release-apk.yml` |
| Version alignment guard (tag ↔ pubspec) | ✅ | `release-apk.yml` |

## Controls to add (fintech-specific)

### 1. Segregation of duties (SoD)

The principle: **committer ≠ approver ≠ deployer**.

- **Committer**: writes code on a feature branch.
- **Approver**: reviews and approves the PR (GitHub CODEOWNERS or manual).
- **Deployer**: triggers the promotion to production (GitHub environment
  approval — different person from the committer).

**Current gap**: `promote.yml` requires environment approval, but the
committer could also approve if they have the right GitHub role.
**Mitigation**: enforce at least 1 approval on `main` (already done) and
require a second GitHub user to approve production promotions.

### 2. Signed artifacts + SBOM

Every production artifact should be:
- Built in CI (not locally) — already done for APK.
- Signed with a known key — APK is signed with upload key.
- Accompanied by an SBOM (Software Bill of Materials) for dependency
  audit.

**Current gap**: no SBOM generation, no dependency audit in CI.
**Add**: `dart pub deps` → SBOM in the release, `dart pub audit` or
`dart pub outdated` in `pr.yml`.

### 3. Secret scanning

Secrets must never reach git. CI should catch accidental commits.

**Current gap**: no secret-scanning gate in `pr.yml`.
**Add**: `gitleaks` or `trufflehog` step in `pr.yml`.

### 4. Audit evidence for deployments

Auditors need: who deployed, when, what changed, what was approved.

**Current gap**: GitHub Actions logs are ephemeral. No deployment log
retention beyond GitHub's default.
**Add**: a `deploy-log.md` (or database table) that records each
deployment: commit SHA, environment, deployer, timestamp, approval
reference.

### 5. Rollback procedure

What happens if a production deploy breaks something?

**Current gap**: no documented rollback procedure.
**Add**: `docs/ci-cd.md` section on rollback:
1. Revert the commit on `main`.
2. Merge to `main` → staging auto-redeploys.
3. Verify staging.
4. Promote the revert to production.

### 6. Kill-switch / feature flags

Can a feature be disabled without a deploy?

**Current gap**: `app_settings` table exists but is unused.
**Add**: wire `app_settings` as a feature-flag store. Admin can set
`payouts_enabled=false` to halt all outbound payments without a redeploy.

### 7. Dependency audit

Outdated or vulnerable dependencies are a supply-chain risk.

**Current gap**: no `dart pub audit` in CI.
**Add**: `dart pub outdated --json` step in `pr.yml`, fail on known
vulnerabilities.

### 8. Database migration safety

Migrations must be zero-downtime and reversible.

**Current state**: migrations are idempotent (`IF NOT EXISTS`), no
destructive statements. `tool/migrate.dart` re-runs every file. This is
already good. **Enhance**: add a `rollback` column to the migration
manifest for explicit rollback instructions.

## Fintech-specific CI gates (recommended additions)

```yaml
# In pr.yml, add these steps:
- name: Dependency audit
  run: dart pub deps --style=compact

- name: Secret scan
  uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

- name: ADR validation
  run: bash scripts/check-adrs.sh
```

## What auditors look for (SOC2 / CBK alignment)

| Control | What auditors check | Current coverage |
|---------|-------------------|-----------------|
| Change management | PR reviewed before merge | ✅ |
| Segregation of duties | Committer ≠ deployer | ⚠️ Partial |
| Deployment approval | Production gated | ✅ |
| Audit trail | Who deployed, when, what | ⚠️ GitHub logs only |
| Secret management | No secrets in git | ✅ (env vars only) |
| Dependency audit | Known vulnerabilities tracked | ❌ Not automated |
| Rollback capability | Can deploy be reversed? | ❌ Not documented |
| Feature kill-switch | Can feature be disabled? | ❌ Not wired |
| SBOM | Dependency inventory | ❌ Not generated |

## Production promotion discipline (ADR-0022, incident 2026-09-03)

The Railway **production `api` service must have NO git-source**. A git-source
(`BongweKE/hodaripay` @ `main`) auto-deploys the API on every merge, outrunning
the Neon production migrate + Cloudflare web deploy that `promote.yml`
performs. On 2026-09-03 that split production: API ran #117 code while the
production DB was still on migration `0022` and the web worker was 13 commits
behind.

How to promote (full runbook in `docs/ci-cd.md`):
1. `deploy-staging.yml` green on the `main` commit + staging manually verified.
2. `gh workflow run "Promote to Production" --ref main` (never a feature
   branch).
3. Approve the GitHub `production` environment gate (required reviewer).
4. Watch: test → migrate Neon production → deploy Railway API → deploy web →
   smoke.
5. Verify prod `/health` + the changed flow; if a migration shipped, confirm it
   applied (the API must never outrun its schema — run `promote.yml` to catch
   it up if it ever does).

`prod-api` / `staging-api` git tags are force-moved **only** by their
workflows and record the last API commit each environment runs; the promote
"backend changed?" check diffs against `prod-api`.

## Key facts for this repo

- Railway handles the runtime: build → deploy → health check.
- Neon handles the DB: branch-per-env, migrations via `tool/migrate.dart`.
- Cloudflare handles static assets: `_headers` for security headers.
- GitHub Actions handles the gates: `pr.yml`, `deploy-staging.yml`,
  `promote.yml`, `release-apk.yml`.
- Secrets live in Railway env vars (not GitHub) — never in git.
