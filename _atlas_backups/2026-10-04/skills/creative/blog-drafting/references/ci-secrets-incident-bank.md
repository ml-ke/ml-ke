# CI/CD Secrets Incident Bank (verified anchors for Cybersecurity-theme posts)

Recurring Friday Cybersecurity theme. All anchors below verified against 2+
sources on Sep 1, 2026 for the Aug 28 backfill post
`secrets-in-ci-credential-leaks` (permalink `/posts/secrets-in-ci-credential-leaks/`).
Reuse freely; re-check figures before reuse per the blog's rules of the road.

## CircleCI breach (Dec 2022 – Jan 2023) — CI provider as secrets vault
- Dec 16, 2022: malware on a CircleCI engineer's laptop (undetected by AV);
  session-cookie theft let attacker impersonate the employee **despite 2FA**,
  escalate to production, and (Dec 22) exfiltrate customer environment
  variables, tokens, and keys — pulling encryption keys from a running process
  to decrypt data at rest.
- Dec 29–30: a customer alerted suspicious GitHub OAuth activity; Dec 31:
  CircleCI **proactively rotated ALL GitHub OAuth tokens on behalf of every
  customer**.
- Jan 4, 2023: public incident report + advisory to "rotate any secrets stored
  in CircleCI".
- Lesson: a CI provider is a secrets concentration point — provider compromise
  ⇒ rotation must be broad, not surgical.
- Sources: circleci.com/blog/jan-4-2023-incident-report/,
  circleci.com/blog/january-4-2023-security-alert/, malwarebytes.com (Jan 2023).

## Mercedes-Benz GitHub token leak (found Sep 29, 2023; disclosed Jan 2024)
- RedHunt Labs found a GitHub PAT in an employee's **public repo** giving
  "unrestricted" and "unmonitored" access to the company's internal GitHub
  Enterprise Server: database connection strings, cloud access keys, blueprints,
  SSO passwords, API keys.
- Informed via RedHunt + TechCrunch on Jan 22, 2024; token revoked Jan 24.
  BleepingComputer coverage Jan 30, 2024.
- Lesson: one over-scoped token + one public repo = whole org exposed. Abuse
  evidence only exists if **audit logs are enabled**.
- Sources: bleepingcomputer.com (Jan 30, 2024), redhuntlabs.com blog, cybernews.com.

## Toyota (Oct 2022) — long-exposure variant
- Exposed GitHub access key left customer information publicly accessible for
  roughly **five years**.
- Source: BleepingComputer "Toyota confirms leak of 296,000 customer records".

## CISA "Private-CISA" repo (created Nov 13, 2025; exposed May 2026)
- GitGuardian researcher Guillaume Valadon found a public repo named
  `Private-CISA` with internal CISA/DHS credentials: **AWS GovCloud keys,
  GitHub PATs, plaintext passwords, JFrog Artifactory tokens, Azure registry
  keys, Kubernetes manifests, Terraform code, Entra ID SAML certificates** —
  plus an explicit how-to guide for *disabling GitHub's secret scanning*.
- Repo taken down ~26h after discovery (May 15, 2026), after escalation via
  journalist Brian Krebs.
- Lesson: even the top cyber-defense agency ran the same unsafe playbook —
  the problem is process, not people.
- Sources: krebsonsecurity.com (May 2026), darkreading.com,
  theregister.com (May 19, 2026).

## GitHub secret scanning scale stats
- **39M+ secrets leaked across GitHub in 2024**; push protection blocks
  "several secrets" every minute.
- Sources: github.blog/security/application-security/next-evolution-github-advanced-security/,
  securityweek.com, bleepingcomputer.com.

## Firsthand: this repo's GH013 push-protection block
- GitHub push protection rejected a push because a Supabase PAT sat in an OLD
  commit; fix = interactive rebase → redact token → force-push (see
  blog-drafting Pitfall #7).
- Lesson: secret scanning catches tokens **in history**, not just new commits —
  a committed secret never dies until scrubbed AND rotated.

## Overlap map (what NOT to re-cover)
- tj-actions/changed-files CVE-2025-30066 + reviewdog + SolarWinds 2020 →
  already the anchors of `2026-08-21-cicd-security-control-fintech.md`. Link
  to that post instead of re-narrating.
- Vault/API-key hygiene for ML pipelines → `2026-06-12-ml-secrets-management.md`.

## Controls table (used in the Aug 28 post)
| Control | Stops | Case that proves it |
|---------|-------|---------------------|
| Secret scanning + push protection ON | tokens entering repo at all | CISA (scanning disabled on purpose) |
| Rotate broadly after any provider breach | stolen secrets staying valid | CircleCI (all OAuth tokens rotated) |
| Short-lived creds / OIDC / scoped tokens | a leaked token being worth anything | Mercedes (unrestricted token, months) |
| Vault injection at runtime + log masking | secrets in code and build output | CircleCI (env vars exfiltrated) |
| Pin actions to commit SHAs | supply-chain tampering in CI | link to Aug 21 CI/CD post |
| Enable audit logs | silently reusing a leaked token | Mercedes (no evidence without logs) |
| Scan git history (gitleaks/trufflehog) | old commits re-exposing secrets | own GH013 push block |
