# BaaS / Aggregator / Third-Party API Security — Verified Anchor Bank

Class: third-party payment-API and Banking-as-a-Service integration security
(the Sun "Fintech Security" prevention-playbook slot, and the layer ABOVE the
Sep 4 direct-Daraja rail). First used: Sep 6 2026 post
`aggregator-baas-security-playbook` (published
`_posts/2026-09-06-aggregator-baas-security-playbook.md`). Directly reusable
for any future "middleman layer" post (BaaS diligence, partner-bank breach,
payment-processor credential hygiene, middleware solvency/reconciliation).

## Anchors (all body-verified with the sources listed)

### 1. Evolve Bank & Trust — LockBit ransomware breach (2024)
- Partner bank whose rails sit under fintechs incl. Shopify, Stripe, Mercury;
  breach touched fintech-partner customers, not just its own. Affirm, Wise and
  Bilt each independently confirmed their customers were impacted.
- Numbers: 7,640,112 people impacted (Maine AG filing); Evolve notified
  ~7.6M. Entry: employee clicked a malicious link → LockBit operator gained
  access to databases + file shares. Detection May 29, 2024 ("systems not
  working properly" — first read as hardware failure); notification showed
  initial access Feb 9, 2024 → ~4 months dwell. Customer funds remained safe
  (loss was data, not balances). LockBit first published the dump mislabeled
  as a US Federal Reserve breach.
- Lesson: your customers' KYC/data physically sit in the partner's perimeter;
  vendor diligence must cover real detection time + notification SLA.
- Sources: BleepingComputer (Bill Toulas, Jul 9 2024),
  https://www.bleepingcomputer.com/news/security/evolve-bank-says-data-breach-impacts-76-million-americans/
  ; TechCrunch (Jul 9 2024),
  https://techcrunch.com/2024/07/09/evolve-bank-says-ransomware-gang-stole-personal-data-on-millions-of-customers/

### 2. Juspay — unrecycled AWS access key (2020)
- Bengaluru payments processor for merchants incl. Amazon, Swiggy,
  MakeMyTrip, Yatra, Freecharge, BookMyShow, Snapdeal; ~650k tx/day.
- Detected unauthorized activity Aug 18, 2020 in a data store; cause: an OLD,
  UNRECYCLED AWS ACCESS KEY (long-lived credential never deleted). Breach went
  public Jan 2021 when researcher Rajshekhar Rajaharia spotted ~3.5 crore
  (35M) records for sale on the dark web; CPO Magazine put ~100M records
  circulating (matches Juspay's 10 crore user-metadata figure). Juspay said
  exposed records were masked card data + card fingerprints, but a portion of
  the 10-crore user metadata incl. plain-text email addresses was also
  exposed.
- Remediation (the playbook in miniature): refreshed merchant API keys and
  invalidated old ones, enforced 2FA on all internal tools, moved off
  long-lived AWS access keys to IAM-based access.
- Lesson: access-key lifecycle is a SHARED control — the key the processor
  forgot is the class of key your integration holds with every vendor.
- Sources: Business Today (Jan 5 2021),
  https://www.businesstoday.in/technology/news/story/amazon-swiggy-payments-partner-juspay-suffers-data-breach-35-crore-records-compromised-283598-2021-01-05
  ; CPO Magazine,
  https://www.cpomagazine.com/cyber-security/amazon-swiggy-payment-processor-juspay-downplays-data-breach-that-led-to-100-million-records-circulating-on-the-dark-web/

### 3. Synapse — middleware collapse / $85M shortfall (2024)
- Software layer bridging fintech apps (incl. Mercury) to partner banks:
  Evolve, American Bank, AMG National Trust, Lineage. Filed Ch.11 Apr 22 2024.
- Trustee (former FDIC chair Jelena McWilliams, named May 24 2024) report:
  customers of Synapse-linked fintechs held $265M in balances while partner
  banks held only $180M → $85M shortfall; source unknown at the time. 100,000+
  customers locked out of savings for weeks; ledgers commingled across banks.
- By Mar 2025 (Fortune): frozen money approached $200M; court removed
  Synapse's management over "gross" mismanagement; fintechs incl. Mercury
  suing; TabaPay rescue collapsed.
- Lesson: reconciliation failure (two ledgers disagreeing, no daily
  two-sided match) is a security failure of the same class as intrusion — no
  pentest finds an $85M hole, only a two-ledger diff does.
- Sources: CNBC (Hugh Son, Jun 7 2024),
  https://www.cnbc.com/2024/06/07/synapse-bankruptcy-trustee-85-million-of-customer-savings-is-missing.html
  ; CFPB enforcement page (Ch.11 date),
  https://www.consumerfinance.gov/enforcement/actions/synapse-financial-technologies-inc/
  ; Fortune (Mar 7 2025),
  https://fortune.com/2025/03/07/synapse-evolve-mercury-bankruptcy-lawsuits/

### 4. CBK — Third-Party Agents Guideline (2016), clause 5.1.6
- The licensed institution "shall also be responsible for assessing the
  adequacy of controls of outsourced activities by taking appropriate direct
  or third party audits of the same as mandated under relevant outsourcing
  agreements." (verbatim from the PDF)
- Use: local regulatory grounding — outsourcing transfers operations, never
  accountability. Scope is bank/DTM third-party agents; apply by analogy to
  BaaS/aggregator reliance.
- Source: https://www.centralbank.go.ke/wp-content/uploads/2016/08/GUIDELINE-ON-THE-APPOINTMENT-AND-OPERATIONS-OF-THIRD-PARTY-AGENTS-BY.pdf

### 5. Kenya aggregator precedent (reuse via cross-link, do not re-narrate)
- Safaricom v. East African Data Handlers KES 20.3M aggregation suit — full
  treatment lives in the Sep 4 post `mpesa-daraja-api-pitfalls` and in
  daraja-api-incident-bank.md anchor #1. Cite Tech-ish + cross-link.

## Reusable demo recipe (verified, deterministic, no network)
1. HMAC webhook verification (Gate: authenticate every vendor message) —
   recompute sha256-HMAC over the raw body with the vault-held secret;
   forged signature → False, genuine → True.
2. Two-ledger reconciliation (Gate: reconcile across hops) — set-diff of
   (event_id, amount) pairs your app recorded vs the vendor statement;
   print missing credits + action (freeze payout, dispute within SLA).
Full code + verbatim output in
`_posts/2026-09-06-aggregator-baas-security-playbook.md`. Cover metaphor:
"opaque middleman" — YOUR APP / BANK RAIL on either side of a dashed red
AGGREGATOR / BaaS black box with breach arrows, rail vanishing inside;
VERIFY THE EDGES + RECONCILE THE MIDDLE chips.

## Cross-links that pair well
mpesa-daraja-api-pitfalls (direct-rail joints), vendor-risk-fintech-
contractors (contractor/access lifecycle), secrets-in-ci-credential-leaks
(leaked keys, push protection), flutterwave-fraud-anatomy (aggregator
internal-fraud class), insider-threat-privileged-access-fintech.
