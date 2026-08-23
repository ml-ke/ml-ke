---
title: "The Three-Minute Insider: Privileged Access and the Insider Threat in Fintech"
date: 2026-08-16 00:00:00 +0300
categories: [AI Security, Fintech]
tags: [insider-threat, privileged-access, pam, fintech-security, ueba, fraud]
image:
  path: /assets/img/cover-insider-threat-privileged-access-fintech.webp
  alt: A vendor badge granting live backend access at 5:30 AM, with a three-minute countdown to a code change
---

## The three-minute insider

At **5:30 AM on June 6, 2025**, NCBA Bank activated a vendor contract that handed Evans Nandwa — a software developer with consultancy firm **Ronford Digital Limited** — live backend privileges on its **NCBA Rwanda** subsidiary for what was supposed to be a routine system maintenance window on the bank's MTN-powered mobile money platform. At **5:33 AM**, three minutes later, Nandwa altered the core integration code. By June 14, when an end-of-week settlement reconciliation against MTN Rwanda's records exposed the gap, roughly **Ksh 57.5 million (about USD 446,000)** had been drained through **260 transactions across 70 accounts** ([254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/), [Nairobi Times](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)).

Not every bank heist needs a badge. In October 2025, a 20-year-old university student was charged with stealing **Sh7.8 million** from Sidian Bank after allegedly bypassing multiple security layers ([Kenya Insights](https://kenyainsights.com/exposed-how-a-20-year-old-university-student-breached-sidian-banks-security-fortress-and-walked-away-with-ksh-7-8-million/), [Tuko](https://www.tuko.co.ke/kenya/counties/607644-mku-student-charged-hacking-sidian-bank-stealing-ksh78-million/)). External attacks get the headlines, but the NCBA case is the harder lesson: **the fastest money moves are executed by people who are handed the keys.** Trusted access, not malware, was the attack.

## Privilege granted faster than trust is established

The NCBA timeline is a study in inverted security. The contractor's first day on the job was also his first minute of abuse: no break-in period, no supervised shadowing, no staged rollout. Court filings from the DCI Banking Fraud Investigation Unit describe Nandwa accessing the system at 0533 hours and implementing an amendment through a logic change that caused the integration service to return a "Success" status for every MTN withdrawal request — **even when funds were insufficient or the account did not exist** ([Nairobi Times](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)).

Three problems compound here:

- **Standing access:** the vendor held live backend privileges on a production money system, rather than scoped, time-boxed permissions for the maintenance task.
- **No separation of duties:** one person could modify and deploy integration logic with no second approver.
- **No behavioural baseline:** nothing flagged a first-day contractor touching core validation logic at 5:33 AM.

> **The uncomfortable arithmetic:** access was activated at 5:30, abused at 5:33. The more standing privilege a vendor holds, the shorter the distance between opportunity and execution.
{: .prompt-warning }

## Eight invisible days

The amended logic did not fail loudly — it failed correctly. Withdrawals returned "Success" without valid balances, and per court records even without valid accounts, so daily dashboards showed a healthy system while the money flowed for eight days. Detection came only when NCBA compared its internal transaction records with MTN Rwanda's payout figures during the end-of-week settlement reconciliation and found an irreconcilable gap ([254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/), [Business Daily](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)). The Banking Fraud Investigation Unit subsequently obtained a court order to detain Nandwa for 21 days pending investigation ([Nairobi Times](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/), [Court Helicopter](https://www.courthelicopter.ke/court-detains-contractor-over-ksh-57-5-million-fraud-at-ncba-bank/)).

The lesson: **monitoring tuned to availability, not integrity, cannot see an integrity attack.** A "Success" status without a matching debit is an anomaly only if you are looking for it.

## Mapping controls to the NCBA timeline

Each control below targets a specific minute of the attack:

- **Least privilege + just-in-time (JIT) access.** Grant privileged access only for the maintenance window, with an approval workflow and a hard expiry. Nandwa should have received scoped entitlements to the mobile-money integration module at 5:30 — and lost them at 6:00. Standing vendor access turns every future employee into a standing risk.
- **Two-person control (dual authorization).** A second authorized engineer should approve any production change. The 5:33 amendment — a logic change on the money path — is precisely the change that must never ship on one person's say-so.
- **Session recording and replay.** Terminal and screen sessions recorded and streamed to a SIEM make the 5:33 session reconstructible: who, what commands, what diff. Replay capability compresses forensics from weeks to minutes.
- **Break-glass accounts with instant alerting.** Emergency access should be the only path into live production, and every use should page security in real time with the account, the timestamp, and the commands executed. Unexplained 5:30 AM vendor logins then become alerts, not footnotes.
- **UEBA (user and entity behavior analytics).** ML models learn each engineer's normal behaviour — systems touched, hours worked, command patterns. A first-day contractor altering core integration logic at 5:33 AM is a massive deviation from every baseline, and should score as an anomaly in real time, not eight days later.
- **Quarterly access reviews and vendor offboarding.** Recertify every entitlement quarterly and revoke access the day a contract ends. The attack surface here was not a vulnerability in code — it was a vendor relationship with unlimited, persistent reach into production.

> **The control that matters most:** make privileged access *ephemeral, approved, recorded, and reviewed*. Each property independently breaks this attack; together they make it impossible to run silently.
{: .prompt-info }

## Conclusion

The insider threat in fintech is not a people problem — it is a privilege problem. The NCBA fraud required no vulnerability scan, no phishing email, and no malware; it required a contractor, a maintenance window, and three minutes of unsupervised access to a live money system. Insider threat cannot be patched away, but it can be engineered away: least privilege, JIT access, dual authorization, session recording, break-glass alerting, UEBA baselines, and ruthless access reviews. In a sector where a three-minute window moves half a billion shillings, the price of standing trust is measured in exactly the minutes you did not control. For the step-by-step prevention playbook, see [How AI, Automation and AI-Enabled Security Could Have Stopped the NCBA Heist](/posts/ai-automation-fintech-security-playbook/).

## References

- [254 News — How a three-minute code change triggered Sh57.5 million NCBA fraud](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)
- [Nairobi Times — Banking Fraud Unit allowed to detain software developer suspected of defrauding bank of Sh57 million](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)
- [Kenya Insights — How NCBA Software Engineer Opened Floodgates For Mobile Banking System Fraud](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
- [Court Helicopter — Court detains contractor over Ksh 57.5 million fraud at NCBA Bank](https://www.courthelicopter.ke/court-detains-contractor-over-ksh-57-5-million-fraud-at-ncba-bank/)
- [Business Daily — Software developer held on Sh58m NCBA fraud claim](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)
- [Kenya Insights — How a 20-year-old university student breached Sidian Bank's security fortress and walked away with KSh 7.8 million](https://kenyainsights.com/exposed-how-a-20-year-old-university-student-breached-sidian-banks-security-fortress-and-walked-away-with-ksh-7-8-million/)
- [Tuko — MKU student charged with hacking Sidian Bank, stealing KSh 7.8 million](https://www.tuko.co.ke/kenya/counties/607644-mku-student-charged-hacking-sidian-bank-stealing-ksh78-million/)
