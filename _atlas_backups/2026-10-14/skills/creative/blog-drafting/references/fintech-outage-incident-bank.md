# Fintech Outage / Operational-Resilience Incident Bank

Verified anchors for outage, degraded-mode, replatforming and DR posts (Sat Fintech
slot — first used 2026-09-12, slug `fintech-outage-post-mortem`). Every figure below
was checked against the linked source body, not a snippet. Re-verify dates/fines
before reuse: regulators update pages (the FCA TSB release carries a 2026 update
stamp even though it is a 2022 decision).

## Card-network / processor

**Visa Europe — 1 June 2018** (letter to the UK Treasury Committee, 11 pages)
- Failure began **14:35 Fri 1 Jun**; fully restored only **00:45 the next day** (~10h).
- **5.2M transactions failed to process correctly**: **2.4M UK**, **2.8M rest of Europe**.
- **1.7M UK Visa cards** affected = **10.4%** of the cards used during the outage.
- Root cause: a switch in the **primary** data centre suffered "a very rare, partial
  failure" which **prevented the backup switch from activating**. The malfunctioning
  primary kept trying to synchronise messages with the secondary, creating a
  **backlog at the secondary** that slowed it further. Time to **isolate** the primary:
  nearly 5 hours.
- Sources: theguardian.com/money/2018/jun/19/visa-admits-5m-payments-failed-over-a-broken-switch ·
  fstech.co.uk/fst/Visa_Data_Centre_Malfunction.php · finextra.com/newsarticle/32277
- Reuse for: correlated-failure/redundancy design, time-to-isolate vs time-to-failover.

## Bank replatforming / change risk

**TSB — April 2018 migration, fined 20 Dec 2022**
- All branches and **a significant proportion of its 5.2M customers** affected from the
  first day; **business-as-usual only in December 2018**.
- **£32.7M** paid in customer redress. Fine **£48.65M total = FCA £29.75M + PRA £18.9M**
  (30% settlement discount). Findings: failed to organise/control the migration and to
  manage outsourcing risk with its critical third-party supplier.
- Source: fca.org.uk/news/press-releases/tsb-fined-48m-operational-resilience-failings
- ⚠️ Common error: swapping the FCA/PRA split (FCA is the **larger** at £29.75M).
- Reuse for: data migrated fine but the platform failed around it; incident duration
  measured in months, not minutes; outsourced-platform governance.

## Mobile money / partner-system

**M-Shwari (Safaricom + NCBA) — Nov 2025 and Feb 2026**
- Nov 2025: outage from ~Wed ran **more than three days**; balances showed **zero**
  despite savings. Safaricom said access was restored Sunday "but the reconciliation
  of account balances was still ongoing" (TechCabal, 3 Nov 2025,
  techcabal.com/2025/11/03/ncba-works-to-restore-m-shwari-outage/).
- Feb 2026: **~36-hour** outage; Safaricom attributed it to a "technical issue"
  affecting **their partner system** (NCBA runs the banking rails). Announced "fully
  restored" while users still reported **hanging transactions and wrong balances at
  10 p.m.** the same day (tech-ish 9 Feb 2026 — tech-ish.com/2026/02/09/safaricom-m-shwari-outage/;
  K24 — k24.digital/411/m-shwari-system-outage-persists-despite-safaricom-assurance).
- Reuse for: "restored ≠ reconciled"; the contract boundary as the limit of the
  post-mortem; reconciliation debt visible as customer complaints.

## Cloud / shared dependency

**AWS us-east-1 — 19–20 October 2025**
- Root cause per AWS post-event summary: **latent race condition in the DynamoDB DNS
  management system** → an incorrect **empty DNS record** for
  `dynamodb.us-east-1.amazonaws.com` that the automation **failed to repair**.
- AWS **disabled the DynamoDB DNS Planner and DNS Enactor worldwide** as the
  mitigation; adding velocity controls for NLB AZ failover and EC2 data propagation.
- Some customers saw issues for **up to 15 hours** (reports of ~2:40 AM → 2:20 PM PDT).
- Sources: infoq.com/news/2025/11/aws-dynamodb-outage-postmortem/ (quotes AWS) ·
  forbes.com/sites/kateoflahertyuk/2025/10/23/... · thousandeyes.com blog timeline.
- Reuse for: "your MTTR cannot beat your provider's"; automation as the failure; the
  global-blast-radius shape of a single malformed record.

## Industry statistics

**Uptime Institute, Annual Outage Analysis 2026** (from the 2025 Annual Survey)
- **57%** of respondents: most recent major outage cost **more than $100,000**;
  **1 in 5** exceeded **$1 million** (second consecutive year).
- **~1 in 10** report serious or severe impacts from their last outage.
- Third-party IT/data-centre providers ≈ **two-thirds** of publicly reported outages
  over the nine years tracked.
- Failure to follow established procedures is the leading human-error driver.
- Sources: uptimeinstitute.com/about-ui/press-releases/uptime-announces-annual-outage-analysis-report-2026 ·
  businesswire.com/news/home/20260513289344/en/...

## Reporting clocks (post-mortem must respect these)

- **DORA** (EU, in force Jan 2025): initial notification **within 4 hours of
  classifying an incident as major, and no later than 24 hours from becoming aware**;
  intermediate report at **72h**; final within **1 month**. Source: EBA joint RTS —
  eba.europa.eu/activities/single-rulebook/regulatory-activities/operational-resilience/
  joint-technical-standards-major-incident-reporting (also FMA Austria summary).
- **CBK (Kenya)**: Guidance Note on Cybersecurity (Aug 2017) requires notification of
  incidents with a **significant and adverse impact within 24 hours**, plus quarterly
  reporting. Source: centralbank.go.ke/wp-content/uploads/2017/09/GUIDANCE-NOTE-ON-
  CYBERSECURITY-FOR-THE-BANKING-SECTOR.pdf (practitioner summaries: Inside Privacy,
  National Law Review). Note there is also a separate 2019 Guideline on Cybersecurity
  for Payment Service Providers.
- **FCA PS21/3 (UK)**: identify important business services, set **impact tolerances**,
  and perform mapping/testing so the firm can stay within them **no later than
  31 March 2025**. Source: fca.org.uk/publications/policy-statements/ps21-3-building-operational-resilience.
- Point to make in prose: every clock starts at **awareness/classification**, so
  detection latency is a compliance metric, not a dashboard aesthetic.

## Runnable demo recipe (verified 2026-09-12)

`verify-post-code.py`-clean, **stdlib only**, deterministic; published output matched
seed 7 exactly. Shape: 96 x 15-min bins, diurnal `base_attempts()`, baseline approval
rate learned as the **median of healthy bins**, residual σ from the same bins.
- Injected incident: 8 brownout bins at **0.93** approval (above a static 0.90 alarm),
  2 hard-fail bins at **0.02**, retries at **1.35x attempts for 6 bins** after restore.
- Seed-7 results: base rate **0.9824**, σ **0.00426**, attempts 138,025, stranded
  **4,659 approvals = 3.6%** of the day; **18%** of the deficit outside the hard-stop
  bins; static threshold first fires **90 min** late vs **CUSUM (k=0.5, h=20) 15 min**;
  retry surplus retires **85%**; **713 (15%)** never re-attempted; customer-visible
  window **240 min** vs **30 min** of hard-down.
- Robustness: seeds 1–7 keep the CUSUM alarm at 09:45 and the recovered share at
  84–87% — quote the range when arguing the method does not depend on one seed.
- Honesty caveat worth reusing: the deficit integral **understates** impact because
  attempts themselves fall during a degradation.

## Alerting / on-call / money-SLI anchors (verified 2026-09-13, slug `real-time-reconciliation-alerting`)

For posts about *paging* a reconciliation or integrity queue (as opposed to detecting
breaks). All figures below were read from the source body.

**RBI harmonised TAT — the regulatory clock that makes detection mandatory**
- Circular **RBI/2019-20/67**, dated **20 Sep 2019**, in force **15 Oct 2019**, issued
  under s.10(2) read with s.18 of the Payment and Settlement Systems Act 2007.
- Annex table (read verbatim): UPI fund transfer / IMPS / NACH / APBS — account debited
  but beneficiary not credited → auto-reversal by the beneficiary bank **latest on T+1**;
  **INR 100 per day** of delay beyond that, per transaction. UPI **merchant** payments
  and PoS/ATM → **T+5**; same INR 100/day.
- "the prescribed TAT is the outer limit"; compensation must be credited **suo moto,
  without waiting for a complaint or claim from the customer** (para 5).
- Source: rbi.org.in/commonman/English/scripts/Notification.aspx?Id=3074 (curl-friendly).
- Reuse for: "a break is a liability with a deadline"; statutory margins in SLIs; why
  detection cannot depend on a customer raising a ticket. No Kenyan equivalent found —
  CBK's clock is the 24-hour incident-notification duty, not per-transaction
  compensation (state only what the CBK note supports).

**Google SRE Workbook, "Alerting on SLOs" (Table 5-6)** — the burn-rate ladder
- 2% of budget in 1 h = burn rate **14.4** → page; 5% in 6 h = **6** → page; 10% in 3 d
  = **1** → ticket.
- Table 5-4 (99.9% SLO): burn 1 → 30 days to exhaustion; 2 → 15 days; 10 → 3 days;
  1,000 → 43 minutes.
- Duration-based alerts fail: 100% error spikes of 5 min every 10 min consumed **35%**
  of the budget and **never** alerted; a 100% outage alerts after 1 h — same detection
  time as a 0.2% outage.
- With three windows configured, one bad minute satisfies all three → **suppression is
  required**; the confirmation short window should be **1/12** the long window.
- Source: sre.google/workbook/alerting-on-slos/

**Google SRE Book, "Being On-Call"** — the page budget
- One incident (root cause + remediation + follow-ups like the postmortem) averages
  about **6 hours** of work ⇒ at most **2 incidents per 12-hour shift**, with a likely
  **median of 0**; sustained breach means operational overload.
- Source: sre.google/sre-book/being-on-call/

**Visa Smarter Stand-in Processing (STIP)** — the network decides for you
- Announced **26 Aug 2020**; deep learning at cardholder level to approve/decline
  **on the issuer's behalf** when issuer systems are offline; Visa claims up to **50%**
  fewer declines in some cases. Classic STIP used static, portfolio-wide parameters
  (PYMNTS).
- Reuse for: stand-in authorisations are your ledger's blind spot until clearing; every
  stand-in decision is a reconciliation item.
- Sources: usa.visa.com/about-visa/newsroom/press-releases.releaseId.17301.html ·
  pymnts.com/news/payments-innovation/2020/how-ai-helps-visa-step-in-smartly-when-issuers-go-offline/

**EPC SCT Inst rulebook 2025 (v1.1, effective to 21 Nov 2027)** — 10-second hard timeline
- Per the EPC page: the EU **Instant Payments Regulation** (Reg. (EU) 2024/886) shortens
  the hard timeline for instant euro credit transfers to **10 seconds**; attribute
  **AT-T056** (timestamp of the SCT Inst transaction) must be unambiguous and include
  **milliseconds** partly for that reason. Scheme markets funds available in under ten
  seconds. Rulebook entry-into-force time is 03:30 CET.
- Source: europeanpaymentscouncil.eu → SCT Inst rulebook page.
- Reuse for: "at 10 seconds a timeout is a queue item, not an edge case"; millisecond
  timestamps as a reconciliation requirement, not a nicety.

**Daraja silent-callback failure mode (Kenya colour for alerting posts)**
- Daraja rejects HTTP callback URLs (HTTPS required, e.g. Let's Encrypt); callbacks POST
  `ResultCode` / `CheckoutRequestID` / `MpesaReceiptNumber`. A misconfigured endpoint
  therefore stops delivering *without erroring* — the only symptom is an empty queue.
- Source: kenzobe.com/blog/mpesa-daraja-api-errors (plus kodaschool.com for the OAuth and
  STK mechanics already banked in the Daraja bank).

**Reusable demo recipe (stdlib, deterministic) — the page-count comparison**
- 720 one-minute bins; KES 1bn/day settled, SLI = unmatched value ≤ 0.1% of settled over
  30 days ⇒ burn 1.0 = **KES 694/min**. Regimes: drift minutes 120–300 (ramp to
  20,000/min), failover burst 480–500 (60,000/min).
- Policies and results (seed 13; seeds 7/42 move queue 0.1%, first page ±1 min):
  page-per-break **1,785/shift (one every 24.2 s)**; static 25,000/min **20 pages, first
  minute 480**; burn-rate 14.4x (1h + 5m windows) **80 pages, first minute 238**;
  burn-rate + group-by-reason + 60-min silence **5 pages, minute 238**. Queue at shift
  end **KES 3,262,711 / 1,785 lines**; burn at drift end **24.6x**, at burst **30x**;
  INR **178,500/day** statutory exposure if left past T+1.
- Full code lives in `_posts/2026-09-13-real-time-reconciliation-alerting.md`; run it
  from the post with `verify-post-code.py` before reuse (it is quote-verified).
