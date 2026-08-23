---
title: "Reconciliation Analytics: How Data Catches What Dashboards Miss"
date: 2026-08-15 00:00:00 +0300
categories: [Data Science, Fintech]
tags: [reconciliation, fraud detection, fintech, data analytics, ledgers, anomaly detection]
image:
  path: /assets/img/cover-reconciliation-analytics-fintech.webp
  alt: Cover image for reconciliation analytics in fintech
---

## The dashboard said everything was fine

On the morning of June 6, 2025, a contractor logged into NCBA Bank Rwanda's mobile banking platform and made a small logic change. Evans Nandwa, a software developer contracted through Ronford Digital Limited, altered the MTN mobile money integration so that every withdrawal request returned a success status — even when the account did not exist or had insufficient funds. Over the next eight days, 70 accounts initiated 260 transactions that drained roughly USD 446,000 (about Ksh 57.5 million) from the bank before the fraud was stopped on June 14, 2025.

Here is the part that should worry every fintech team: throughout that window, the bank's production dashboards almost certainly showed green. Uptime was fine. APIs were responsive. Transactions were "succeeding" — the system was doing exactly what the altered logic told it to do. What finally caught the fraud was not a real-time alert but an end-of-week settlement reconciliation, when NCBA compared its transaction records against the figures from MTN Rwanda and found a major difference.

**The system saying "normal" is not the same as the money matching.** That gap is what reconciliation analytics exists to close.

## Why availability dashboards miss integrity failures

Most monitoring stacks are built around availability: latency, error rates, throughput, uptime. These metrics answer *"is the service up?"* — they cannot answer *"is the money right?"* The NCBA attack is a textbook case: the injected logic made the integration *always* return success, which means the fraud actively suppressed the very error signals that dashboards watch. A system can be perfectly healthy and perfectly wrong at the same time.

Reconciliation analytics works on a different axis. Instead of measuring service health, it cross-references the internal ledger against an independent source of truth — the settlement partner's ledger (in NCBA's case, MTN Rwanda's records), the switch, the card scheme, or the bank's own general ledger. The core primitive is simple: every transaction in our books should have a matching counterparty record, and aggregate control totals (transaction counts, gross and net sums, hash totals) should agree to the cent. Any break — an unmatched item, a count mismatch, a total that differs by one transaction — is a signal that something happened that the dashboards cannot see.

## The Flutterwave case: value concentration and the ledger trail

The lesson is not unique to Kenya. In early February 2023, attackers moved over ₦2.9 billion out of Flutterwave's accounts in Nigeria. According to documents reviewed by TechCrunch, the funds crossed 28 accounts in just 63 transactions, reportedly via API calls, before Flutterwave reported the matter to police on February 19 and obtained court orders freezing beneficiary accounts. Hundreds of accounts were subsequently frozen as investigators traced where the money went.

Two analytics observations stand out. First, the event was *small in count and enormous in value* — 63 transactions against ₦2.9 billion. Availability dashboards would see a handful of API calls; reconciliation analytics sees control totals that stop matching the settlement ledger within hours. Second, recovery depended on the ledger trail: the transactions were traceable precisely because every movement was recorded, which is what made freezing and clawback possible. When the books are the source of truth, fraud leaves a paper trail by construction.

## Practical reconciliation analytics techniques

If you build one analytics capability this year, make it these four:

- **Ledger-to-settlement matching.** Match every internal transaction to the counterparty record on transaction reference, amount, and timestamp; escalate unmatched or partially matched items as breaks, not exceptions to be cleaned up later.
- **Control totals and break flags.** Recompute counts, sums, and hash totals per settlement cycle. Flag any break immediately — even a Ksh 1 difference can be the visible tip of a logic-level failure, as NCBA's case shows.
- **Cumulative thresholds.** Track rolling exposure per wallet, account, agent, or API credential: cumulative withdrawal velocity, count-to-value ratios, and day-over-day deltas. Flutterwave's 63 transactions for ₦2.9 billion is exactly the signature such thresholds catch.
- **Automate the job.** A reconciliation that runs weekly leaves six days of exposure. Run the job nightly — or intraday — and route breaks to an owner with a service-level agreement. Also monitor *healthy* patterns: in NCBA's case the integration suddenly stopped producing any failures at all. A "zero failures, ever" signal is itself an anomaly worth investigating.

> **A practical rule**
> Reconcile against the partner's ledger at least daily, and treat the *absence* of breaks as suspicious when it coincides with code or logic changes. Most fraud is caught by the boring weekly job — make the boring job run every night.
{: .prompt-info }

## Conclusion

Availability dashboards tell you the service is running; reconciliation analytics tells you the money is real. NCBA lost Ksh 57.5 million because the fraud was designed to look like success, and only a ledger-vs-settlement comparison exposed it. Flutterwave's recovery depended on a traceable ledger trail and fast action on control breaks. For ML teams in fintech, the message is clear: invest in the unglamorous pipeline that matches your books against the counterparty's, automates the cycle, and treats every break as a first-class alert. The dashboard will tell you the system is up. Reconciliation will tell you whether it is telling the truth.

## References

1. [How a three-minute code change triggered Sh57.5 million NCBA fraud – 254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)
2. [How NCBA Software Engineer Opened Floodgates For Mobile Banking System Fraud – Kenya Insights](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
3. [Banking Fraud Unit allowed to detain software developer suspected of defrauding bank of Sh57 million – Nairobi Times](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)
4. [Software developer held on Sh58m NCBA fraud claim – Business Daily](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)
5. [Hundreds of accounts frozen in relation to alleged Flutterwave hack – TechCabal](https://techcabal.com/2023/03/10/hundreds-of-accounts-frozen-during-flutterwave-hack-allegation/)
6. [Alleged security breach leaves millions of dollars missing from Flutterwave accounts – TechCrunch](https://techcrunch.com/2023/03/05/alleged-security-breach-leaves-millions-of-dollars-missing-from-flutterwave-accounts/)
7. [Hackers steal ₦2.9 billion from Flutterwave accounts – Techpoint Africa](https://techpoint.africa/2023/03/05/hackers-have-stolen-2-9-billion-from-flutterwave/)
