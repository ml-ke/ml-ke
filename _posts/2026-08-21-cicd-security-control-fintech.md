---
title: "CI/CD as a Security Control: How Automated Gates Stop Rogue Deploys"
date: 2026-08-21 00:00:00 +0300
categories: [ML Ops, AI Security]
tags: [CI/CD, DevSecOps, Supply Chain Security, Fintech, Auditability]
image:
  path: /assets/img/cover-cicd-security-control-fintech.webp
  alt: Illustration of a CI/CD pipeline acting as a security gate in front of a bank's production systems
---

## The 5:33 AM Change That Cost USD 446,000

On June 6, 2025, during a scheduled maintenance window at NCBA Bank Rwanda, the bank activated a vendor contract that gave Evans Nandwa — a contractor from Nairobi-based Ronford Digital Limited — access to live backend systems. What happened next, per court filings by Kenya's DCI Banking Fraud Investigation Unit, was remarkably simple: at 5:33 AM, about three minutes after access was granted, Nandwa altered the mobile money integration logic. No pull request, no peer review, no approval. The change went straight into production ([254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/), [Kenya Insights](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)).

The altered logic enabled integration services that allowed unauthorized withdrawals from the Rwandan system. By the time the anomaly surfaced on June 14, about 70 accounts had been used for 260 transactions totalling roughly USD 446,000 (Ksh 57.5 million) ([Court Helicopter](https://www.courthelicopter.ke/court-detains-contractor-over-ksh-57-5-million-fraud-at-ncba-bank/), [Nairobi Times](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)).

> **The core lesson**
> The vulnerability was not a missing firewall or a weak password. It was an unreviewed, unattended change to production code. The deployment mechanism itself was the attack surface.
{: .prompt-danger }

## The Counterfactual: What CI/CD Would Have Blocked

Nothing in the NCBA timeline is exotic. It is precisely the sequence a well-configured CI/CD pipeline makes structurally impossible.

**No direct-to-production edits.** The production codebase is write-protected. A 5:33 AM change must begin as a branch and a pull request — never a direct edit to a live backend. Branch protection enforces this at the repository level: required status checks, required reviewers, signed commits.

**Automated checks run before anything ships.** The moment a PR opens, CI runs the build and test suite, then layers on SAST, dependency scanning, and a secrets scan. A logic alteration to money-movement integration code would have to survive the test suite, and planted credentials or backdoor patterns would trip the scanners. At 5:33 AM, with no human watching, these checks never sleep.

**Human review is a control, not a ceremony.** Requiring at least one approver who understands the payments domain means changes to withdrawal logic attract eyes. The NCBA change had no reviewer because the process had no review step.

**Protected environments and promotion gates.** Production is a protected environment reachable only through explicit promotion — typically with a manual approval step. Feature branches flow freely to dev and staging, but a human with the right role must approve promotion to prod. The rogue 5:33 AM change dies at the gate.

**Immutable, signed artifacts.** Build once, sign, and promote the same artifact through every environment. Deployment verifies the artifact hash against the signature; any mismatch halts the deploy and raises an alert. This closes the "hotfix directly on the server" loophole contractors exploit.

**Environment separation and canaries.** No production deploys from feature branches; ship canaries to a small traffic slice with automatic rollback on error-rate or settlement anomalies. The NCBA fraud ran unnoticed for eight days partly because nothing monitored for the behavioral signature of the altered integration.

**Auditability answers "who changed what at 5:33 AM?"** Every production change carries a PR link, approver identity, artifact hash, and timestamp. The question that took investigators days becomes a lookup instead of an investigation.

## Pipelines Are Targets Too: SolarWinds and tj-actions

CI/CD is a control, but it is also an asset worth attacking — so the control itself must be defended. SolarWinds (2020) is the worst case: adversaries compromised the Orion build pipeline and shipped trojanized updates to roughly 18,000 organizations, including US government agencies, before CISA issued alert AA20-352A ([CISA](https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-352a), [SecurityWeek](https://www.securityweek.com/solarwinds-says-18000-customers-may-have-used-compromised-product/)).

More recently, in March 2025, attackers retroactively modified version tags of the popular `tj-actions/changed-files` GitHub Action to point at a malicious commit, exposing CI/CD secrets in workflow logs — CVE-2025-30066, flagged by CISA and patched in v46.0.1 ([GitHub Advisory](https://github.com/advisories/ghsa-mrrh-fwg8-r2c3), [CISA](https://www.cisa.gov/news-events/alerts/2025/03/18/supply-chain-compromise-third-party-tj-actionschanged-files-cve-2025-30066-and-reviewdogaction), [Wiz](https://www.wiz.io/blog/github-action-tj-actions-changed-files-supply-chain-attack-cve-2025-30066)).

The defenses mirror the ones above: pin third-party actions and dependencies to immutable commit hashes, never store secrets in code or logs (inject them at runtime from a vault), sign everything, and treat build infrastructure with production-grade access control. The same discipline applies to ML artifacts — models and datasets are software too, and should ride the same signed, gated pipeline.

> **Remember**
> A pipeline is only a control if it is protected. Unpinned actions, secrets in logs, and write access to build runners recreate the NCBA hole at a much larger blast radius.
{: .prompt-warning }

## Conclusion

The NCBA incident was not sophisticated. It was a three-minute, unreviewed change to live code. CI/CD transforms fintech deployments from "anyone with access can change anything" into a gated, signed, auditable chain. For fintechs in Kenya and across the region, treating the pipeline as a security control — not a delivery convenience — is the difference between catching a 5:33 AM anomaly in seconds and discovering it eight days and USD 446,000 later.

## References

- [How a three-minute code change triggered Sh57.5 million NCBA fraud — 254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)
- [How NCBA Software Engineer Opened Floodgates For Mobile Banking System Fraud — Kenya Insights](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
- [Banking Fraud Unit Allowed to Detain Software Developer — Nairobi Times](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)
- [Court Detains Contractor Over Ksh 57.5 Million Fraud at NCBA Bank — Court Helicopter News](https://www.courthelicopter.ke/court-detains-contractor-over-ksh-57-5-million-fraud-at-ncba-bank/)
- [How bank customers lost Sh517m in IT system fraud — Business Daily Africa](https://www.businessdailyafrica.com/bd/corporate/companies/how-bank-customers-lost-sh517m-in-it-system-fraud-5112506)
- [CISA AA20-352A: APT Compromise of Government Agencies, Critical Infrastructure, and Private Sector Organizations](https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-352a)
- [SolarWinds Says 18,000 Customers May Have Used Compromised Orion Product — SecurityWeek](https://www.securityweek.com/solarwinds-says-18000-customers-may-have-used-compromised-product/)
- [CVE-2025-30066 — GitHub Advisory GHSA-mrrh-fwg8-r2c3](https://github.com/advisories/ghsa-mrrh-fwg8-r2c3)
- [CISA: Supply Chain Compromise of Third-Party tj-actions/changed-files (CVE-2025-30066)](https://www.cisa.gov/news-events/alerts/2025/03/18/supply-chain-compromise-third-party-tj-actionschanged-files-cve-2025-30066-and-reviewdogaction)
- [GitHub Action tj-actions/changed-files Supply Chain Attack — Wiz](https://www.wiz.io/blog/github-action-tj-actions-changed-files-supply-chain-attack-cve-2025-30066)
