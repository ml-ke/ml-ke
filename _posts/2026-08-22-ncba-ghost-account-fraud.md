---
title: "The NCBA Sh57.5M Ghost Account Fraud: Anatomy of a Three-Minute Code Change"
date: 2026-08-22 00:00:00 +0300
categories: [AI Security, Fintech]
tags: [ncba, mobile-money, insider-threat, fraud, rwanda, fintech-security, ghost-accounts]
image:
  path: /assets/img/cover-ncba-ghost-account-fraud.webp
  alt: A clock face at 5:33 with a red injection needle piercing a code bracket, while faint ghost figures fade into a dark corridor
---

## Three minutes was all it took

At **5:30 AM on June 6, 2025**, NCBA Bank activated a vendor contract and granted a consultancy employee — Evans Nandwa of Nairobi-based **Ronford Digital Limited** — live backend privileges to run maintenance and upgrades for its **NCBA Rwanda** subsidiary. Bank maintenance windows are deliberately scheduled for the dead of night, when customer traffic is at its lowest.

At **5:33 AM**, three minutes after receiving that access, Nandwa altered the core application codebase.

According to DCI Banking Fraud Investigation Unit (BFIU) court filings, he manipulated the mobile integration logic that connects NCBA Rwanda to the **MTN mobile money network**. Under his modified logic, a withdrawal request hitting the network would skip validation entirely and receive an automatic, fake "Success" status. The change was pre-programmed with **70 ghost account profiles** and a hardcoded filter that applied only to those exact numbers: if a withdrawal came from any of the 70 accounts, the system skipped the balance check, never verified whether the account was real, and instantly signalled "Success" to MTN Rwanda.

> **The containment trick:** by restricting the loophole to exactly 70 accounts, the fraudsters made sure only their own scripted wallets could siphon money — random members of the public couldn't stumble on the glitch and trigger chaos or alerts.
{: .prompt-warning }

## Eight days of invisibility

From June 6 to June 14, 2025, everything looked normal. Core systems reported standard operational metrics while the exploit quietly ran in the background. The rogue logic was hidden deep in the database queries, invisible to daily operational dashboards.

The bank only caught it when **physical cash balances were tallied during the standard end-of-week settlement** on June 14. NCBA's technical risk team ran a routine reconciliation and found a massive cash deficit: **Ksh 57.5 million (≈ USD 446,000)** attributable to 70 ghost accounts and **260 transactions** pushed through the Rwandan MTN network.

How does an end-of-week reconciliation catch what daily monitoring misses? The bank's systems automatically cross-reference two datasets:

1. What the bank's internal database says customers withdrew
2. What the telecom partner (MTN Rwanda) actually paid out in cash

Normally these match to the cent. On June 14, the automated script flagged an irreconcilable gap: MTN Rwanda's ledger showed **Ksh 57.5M paid out** to mobile money users, but NCBA's internal deposit accounts showed **no corresponding debit entries, fees, or even valid account holders**. The money had vanished into the mobile ecosystem through "ghost" approvals. Nandwa was detained by the BFIU, which obtained court orders to hold him for 21 days as investigations continued ([Nairobi Times](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/), [Kenya Insights](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/), [254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)).

## Why this case matters for every fintech

The NCBA heist is a perfect storm of the failure modes that plague payment systems across Africa:

- **Privileged access granted faster than trust is established** — live backend access to a core system, three minutes before abuse
- **No change-control gates on production code** — a single developer altering core integration logic with no review, approval, or rollback
- **Validation logic that could be switched off for selected accounts** — a hardcoded allowlist of ghosts
- **Monitoring tuned to availability, not integrity** — dashboards showed "normal" because the exploit was designed to be invisible to them
- **Reconciliation on a weekly cycle** — eight days between the first fraudulent withdrawal and detection

Each of these is a design decision, not fate. In the companion post we break down exactly how **AI, automation, AI-enabled security, notifications, and CI/CD** would have stopped this attack at every stage — and the concrete controls any fintech can adopt today.

**Next in this series:** [How AI, Automation and AI-Enabled Security Could Have Stopped the NCBA Heist](/posts/ai-automation-fintech-security-playbook/)

### References

- [Kenya Insights — How NCBA Software Engineer Opened Floodgates For Mobile Banking System Fraud](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
- [254 News — How a three-minute code change triggered Sh57.5 million NCBA fraud](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)
- [Nairobi Times — Banking Fraud Unit allowed to detain software developer suspected of defrauding bank of Sh57 million](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)
- [Court Helicopter News — Court detains contractor over Ksh 57.5 million fraud at NCBA Bank](https://www.courthelicopter.ke/court-detains-contractor-over-ksh-57-5-million-fraud-at-ncba-bank/)
- [KICTANet — How NCBA Software Engineer Opened Floodgates For Mobile Banking System Fraud](https://posts.kictanet.or.ke/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
