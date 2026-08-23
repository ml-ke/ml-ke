---
title: "When the Contractor Has the Keys: Vendor Risk in Financial Systems"
date: 2026-08-17 00:00:00 +0300
categories: [Fintech, AI Security]
tags: [vendor-risk, third-party-risk, access-control, supply-chain, fintech-security]
image:
  path: /assets/img/cover-vendor-risk-fintech-contractors.webp
  alt: A contractor's hand reaching toward a bank's core system while a clock ticks
---

## Your risk register has a blind spot

Every fintech runs on vendors: core banking platforms, mobile money integrations, cloud infrastructure, and the consultants who maintain them. That dependency is also the attack surface your board least understands. Three incidents show why. In June 2025, a contractor hired by NCBA Bank to maintain its Rwanda subsidiary's mobile banking platform abused live backend privileges on the very day his contract activated. Evans Nandwa of Ronford Digital Limited altered the integration logic connecting NCBA Rwanda to the MTN mobile money network so that withdrawals from 70 ghost accounts skipped balance and existence checks entirely. Between June 6 and June 14, 260 transactions drained Ksh 57.5 million (about USD 446,000), detected only when end-of-week reconciliation against MTN's ledger exposed the gap. Meanwhile, the SolarWinds Orion compromise (2020) showed nation-state adversaries injecting backdoors into a trusted vendor's build pipeline, pushing trojanized updates to roughly 18,000 organizations, and 3CX's 2023 breach showed a VoIP vendor shipping trojanized software to paying customers. None of these were classic insider attacks — they were failures of vendor risk management.

## The vendor lifecycle is a control surface

The NCBA timeline is a masterclass in what not to do, and every failure maps to a control fintechs can implement today.

### Before activation: due diligence, not paperwork

Nandwa was contracted on June 6, 2025, and granted live backend privileges the same day — contract activation and privileged access happened simultaneously. Vendor due diligence should be a completed milestone before any access is issued: verify the vendor's security posture, review their employee screening practices, and require background checks on any individual who will touch production systems. A single developer with three minutes of tenure should never hold the keys to core payment logic.

### Scope-limited access from day one

The principle is simple: the vendor gets only what the task needs. In a properly scoped engagement, a vendor performs maintenance in a sandbox or staging environment that mirrors production but cannot move money; changes are packaged and deployed by bank staff through a change-management process with explicit sign-off. Granting live core privileges on activation day converts a maintenance window into a fraud window.

### Separation of duties and change approval

The three-minute code change that triggered the fraud was made without any apparent review. Separation of duties means the person who writes a change cannot be the person who approves and deploys it. Every production change should require bank-side approval, a documented business justification, and an automated audit trail. "The vendor said it was a maintenance window" is a schedule, not a control.

### Privileged access management and activity logging

Vendor accounts should sit behind a privileged access management (PAM) layer: time-boxed credentials, session recording, and just-in-time elevation that expires when the task ends. Activity logging should capture who, what, when, and from where — and feed the same SIEM that monitors bank staff. NCBA's own systems looked normal for eight days; the rogue logic was invisible to availability-focused monitoring. Integrity-focused logging of vendor sessions is what makes detection possible.

### Ongoing review and offboarding

Access governance is a lifecycle, not a one-time grant. Re-certify vendor access quarterly, revoke privileges the moment a contract ends or an employee changes role, and audit entitlements against the current statement of work. Nandwa's abuse ran for eight days after a single activation; in many breaches, dormant vendor accounts are exploited months after the engagement ends.

## Contracts are controls too

Technical controls fail; contracts should catch the residual risk. Vendor agreements should include explicit liability for fraud or loss caused by vendor personnel, audit rights over vendor environments and access logs, and incident-notification SLAs measured in hours, not weeks. NCBA's recovery now depends on the courts; a well-drafted contract would have triggered liability and insurance clauses much earlier.

> **The uncomfortable truth:** SolarWinds and 3CX prove that even reputable vendors can be compromised upstream. Assume your vendors will be breached and architect for it — least privilege, network segmentation, and continuous verification of what vendor software actually does.
{: .prompt-warning }

## Conclusion

The NCBA, SolarWinds, and 3CX incidents are the same story at different scales: trust placed in third parties without commensurate controls. For fintechs, the stakes are customers' money and regulatory standing. Vendor risk management is not a procurement checkbox — it is a security discipline: due diligence before activation, scope-limited access, separation of duties, privileged access management, lifecycle governance, and contracts that make liability real. The next contractor who asks for the keys deserves a question in return: "What do you need them for — and who is watching while you use them?"

### References

- [Kenya Insights — How NCBA Software Engineer Opened Floodgates For Mobile Banking System Fraud](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
- [254 News — How a three-minute code change triggered Sh57.5 million NCBA fraud](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)
- [Nairobi Times — Banking Fraud Unit allowed to detain software developer suspected of defrauding bank of Sh57 million](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)
- [Business Daily — Software developer held on Sh58m NCBA fraud claim](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)
- [CISA — Advanced Persistent Threat Compromise of Government Agencies, Critical Infrastructure, and Private Sector Organizations (AA20-352A)](https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-352a)
- [CISA — Active Exploitation of SolarWinds Software](https://www.cisa.gov/news-events/alerts/2020/12/13/active-exploitation-solarwinds-software)
- [Mandiant — 3CX Software Supply Chain Compromise](https://cloud.google.com/blog/topics/threat-intelligence/3cx-software-supply-chain-compromise)
- [CISA — Supply Chain Attack Against 3CXDesktopApp](https://www.cisa.gov/news-events/alerts/2023/03/30/supply-chain-attack-against-3cxdesktopapp)
