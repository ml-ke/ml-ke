---
title: "Secrets in CI: How Credential Leaks Actually Happen (and the Push Protection Fix)"
date: 2026-08-28 00:00:00 +0300
categories: [Cybersecurity, ML Ops]
tags: [secrets-management, CI/CD, GitHub Actions, supply-chain-security, devsecops, credential-leaks]
image:
  path: /assets/img/cover-secrets-in-ci-credential-leaks.webp
  alt: A golden key slips through a crack in a CI pipeline stage while a green shield with scan lines blocks further exposure
---

## The 39 Million Secret Problem

GitHub's secret scanning service found **more than 39 million secrets leaked across the platform in 2024 alone** — and push protection still blocks several secrets *every minute* ([GitHub Blog](https://github.blog/security/application-security/next-evolution-github-advanced-security/), [SecurityWeek](https://www.securityweek.com/39-million-secrets-leaked-on-github-in-2024/), [BleepingComputer](https://www.bleepingcomputer.com/news/security/github-expands-security-tools-after-39-million-secrets-leaked-in-2024/)). These are API keys, personal access tokens, cloud credentials, database passwords, and signing keys — the exact credentials that let CI pipelines build, deploy, and move money.

> **The key concept**
> Credential leaks are not exotic nation-state operations. They are committed files, unrotated tokens, and provider compromises — and for fintechs, a leaked CI secret is a direct path to production systems and customer data.
{: .prompt-info }

## Anatomy: Three Ways Secrets Escape CI

### 1. The CI Provider Is a Vault — and Vaults Get Robbed

CircleCI is the canonical case. On **December 16, 2022**, malware landed on a CircleCI engineer's laptop — and their antivirus did not detect it. The malware executed **session-cookie theft**, letting the attacker impersonate the employee despite 2FA, then escalate into production systems. Because the employee's role included generating production access tokens, the attacker exfiltrated **customer environment variables, tokens, and keys** on December 22, pulling encryption keys out of a running process to decrypt data at rest ([CircleCI incident report](https://circleci.com/blog/jan-4-2023-incident-report/), [Malwarebytes](https://www.malwarebytes.com/blog/news/2023/01/circleci-malware-stole-github-oauth-keys-bypassing-2fa/)).

CircleCI's response is the lesson: on December 31 it **proactively rotated all GitHub OAuth tokens on behalf of every customer**, and on January 4, 2023 it told the world to **rotate any secrets stored in CircleCI** ([security alert](https://circleci.com/blog/january-4-2023-security-alert/)). A CI provider is a secrets concentration point — when it is compromised, *every* project's secrets are suspect. Rotation must be broad, not surgical.

### 2. The Over-Scoped Token in a Public Repo

On **September 29, 2023**, researchers at RedHunt Labs found a GitHub token in a public repository belonging to a Mercedes-Benz employee. The token gave **"unrestricted" and "unmonitored" access to the company's entire internal GitHub Enterprise Server** — database connection strings, cloud access keys, blueprints, SSO passwords, and API keys ([BleepingComputer](https://www.bleepingcomputer.com/news/security/a-mishandled-github-token-exposed-mercedes-benz-source-code/), [RedHunt Labs](https://redhuntlabs.com/blog/mercedes-benz-github-token-leak/)). The token sat in the open for months; the company was informed on January 22, 2024 and revoked it two days later.

Toyota's 2022 disclosure shows the same failure at larger scale: an exposed GitHub access key left **customer information publicly accessible for roughly five years** ([BleepingComputer](https://www.bleepingcomputer.com/news/security/toyota-confirms-leak-of-296000-customer-records/)). One over-scoped token plus one public repo equals the whole organization exposed — and as the Mercedes report notes, you only generate evidence of abuse if **audit logs are enabled**.

### 3. Even the Defenders Leak

In **May 2026**, GitGuardian researcher Guillaume Valadon found a public GitHub repository named `Private-CISA` — created November 13, 2025 — stuffed with internal CISA/DHS credentials: **AWS GovCloud keys, GitHub personal access tokens, plaintext passwords, JFrog Artifactory tokens, Azure registry keys, Kubernetes manifests, Terraform code, and Entra ID SAML certificates**. It even contained an explicit how-to guide for *disabling GitHub's secret scanning* ([Krebs on Security](https://krebsonsecurity.com/2026/05/cisa-admin-leaked-aws-govcloud-keys-on-github/), [Dark Reading](https://www.darkreading.com/cybersecurity-operations/cisa-exposes-secrets-credentials-private-repo), [The Register](https://www.theregister.com/security/2026/05/19/americas-top-cyber-defense-agency-left-a-github-repo-open-with-passwords-keys-tokens-and-incredibly-obvious-filenames/5242915)). The repo was taken down roughly 26 hours after discovery, following escalation through journalist Brian Krebs. The agency tasked with defending US networks ran the same unsafe playbook as everyone else — the problem is process, not people.

### 4. The Firsthand Case: This Blog's Own Push

This repository hit the wall GitHub built. A Supabase personal access token sat in an old commit in our history; when a later push went up, **GitHub push protection rejected it with a GH013 rule violation**, pointing at the exact commit and file ([GitHub docs](https://docs.github.com/en/code-security/concepts/secret-security/secret-leakage-risks)). The fix was an interactive rebase to redact the token, an amended commit, and a force-push. The uncomfortable truth: secret scanning catches tokens **in history**, not just in new commits — a secret committed once never dies until it is scrubbed and rotated.

## How We Can Do Better

| Control | What it stops | Case that proves it |
|---------|--------------|---------------------|
| Secret scanning + push protection on | tokens entering the repo at all | CISA repo (scanning disabled on purpose) |
| Rotate broadly after any provider breach | stolen secrets staying valid | CircleCI (all OAuth tokens rotated) |
| Short-lived credentials (OIDC, scoped tokens) | a leaked token being worth anything | Mercedes (unrestricted token, 4 months) |
| Inject secrets at runtime from a vault; mask logs | secrets in code and build output | CircleCI (env vars exfiltrated) |
| Pin actions to commit SHAs | supply-chain tampering in CI | [CI/CD as a Security Control](/posts/cicd-security-control-fintech/) |
| Enable audit logs | silently reusing a leaked token | Mercedes (no evidence without logs) |
| Scan git history (gitleaks/trufflehog) | old commits re-exposing secrets | our own GH013 push block |

For fintechs running payments through CI, treat the pipeline as critical infrastructure: a leaked CI secret is a *fraud* risk, not just a *security* risk — the same blast radius as the [NCBA contractor incident](/posts/insider-threat-privileged-access-fintech/) but reachable remotely.

## Conclusion

The 39 million figure is not a headline — it is the baseline. CI secrets leak through provider compromise (CircleCI), over-scoped tokens in public repos (Mercedes-Benz, Toyota), and plain process failure (CISA). The controls are cheap and mostly free: turn on secret scanning and push protection, rotate broadly after any breach, prefer short-lived credentials, and treat old git history as contaminated until proven otherwise.

## References

- [Next evolution of GitHub Advanced Security — GitHub Blog](https://github.blog/security/application-security/next-evolution-github-advanced-security/)
- [39 Million Secrets Leaked on GitHub in 2024 — SecurityWeek](https://www.securityweek.com/39-million-secrets-leaked-on-github-in-2024/)
- [GitHub expands security tools after 39 million secrets leaked in 2024 — BleepingComputer](https://www.bleepingcomputer.com/news/security/github-expands-security-tools-after-39-million-secrets-leaked-in-2024/)
- [CircleCI incident report for January 4, 2023 security incident](https://circleci.com/blog/jan-4-2023-incident-report/)
- [CircleCI security alert: Rotate any secrets stored in CircleCI](https://circleci.com/blog/january-4-2023-security-alert/)
- [A mishandled GitHub token exposed Mercedes-Benz source code — BleepingComputer](https://www.bleepingcomputer.com/news/security/a-mishandled-github-token-exposed-mercedes-benz-source-code/)
- [Mercedes-Benz GitHub token leak — RedHunt Labs](https://redhuntlabs.com/blog/mercedes-benz-github-token-leak/)
- [CISA Admin Leaked AWS GovCloud Keys on GitHub — Krebs on Security](https://krebsonsecurity.com/2026/05/cisa-admin-leaked-aws-govcloud-keys-on-github/)
- [CISA Exposes Secrets, Credentials in 'Private' Repo — Dark Reading](https://www.darkreading.com/cybersecurity-operations/cisa-exposes-secrets-credentials-private-repo)
- [America's top cyber-defense agency left a GitHub repo open — The Register](https://www.theregister.com/security/2026/05/19/americas-top-cyber-defense-agency-left-a-github-repo-open-with-passwords-keys-tokens-and-incredibly-obvious-filenames/5242915)
- [Secret leakage risks — GitHub Docs](https://docs.github.com/en/code-security/concepts/secret-security/secret-leakage-risks)

**Related:** [CI/CD as a Security Control](/posts/cicd-security-control-fintech/) · [ML Pipeline Secrets Management](/posts/ml-secrets-management/) · [When the Contractor Has the Keys](/posts/vendor-risk-fintech-contractors/)
