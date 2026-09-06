---
title: "The Middleman Problem: A Security Playbook for Aggregator and BaaS Integrations"
date: 2026-09-06 00:00:00 +0300
categories: [Fintech, Cybersecurity]
tags: [baas, api-security, payment-aggregators, third-party-risk, webhook-security, reconciliation, fintech-security]
image:
  path: /assets/img/cover-aggregator-baas-security-playbook.webp
  alt: Three-layer payment stack where the middle aggregator or BaaS layer is an opaque black box under red breach arrows, and the app above it can only verify the edges with HMAC webhook checks and daily ledger reconciliation
---

## Your payment stack has a middleman

Payment integrations come in two shapes. **Direct**: your server talks to the rail itself — Daraja, a card scheme, a bank API — and you hold every credential and callback. **Intermediated**: an **aggregator** fronts several rails behind one API, or a **Banking-as-a-Service (BaaS)** platform issues accounts, cards and ledgers under a partner bank's licence. Middlemen compress months of connectivity and compliance work into weeks — and quietly move the security boundary.

> **The key concept**
> On a direct rail you secure the joints you hold — the four joints in [Mobile Money API Security](/posts/mpesa-daraja-api-pitfalls/). With a middleman, your customers' money and data sit inside a perimeter you cannot see. You cannot secure the pipe; you can only **verify the edges you control and reconcile the middle you cannot**.

Three real failures show where the middleman leaks: a partner bank's network breached for months, a processor's forgotten key, and a middleware company's books that stopped adding up.

## When the partner bank is breached: Evolve and LockBit

In June 2024 LockBit published a stolen data dump under the false banner of a US Federal Reserve breach. The data belonged to **Evolve Bank & Trust**, a partner bank whose rails sit under fintechs including Shopify, Stripe and Mercury — so the breach touched *their* customers too ([BleepingComputer](https://www.bleepingcomputer.com/news/security/evolve-bank-says-data-breach-impacts-76-million-americans/)). Evolve notified **7.6 million people** (7,640,112 per a Maine attorney general filing), and **Affirm, Wise and Bilt each confirmed their customers were affected**. Funds stayed safe; the loss was data.

The post-mortem is what to internalize ([TechCrunch](https://techcrunch.com/2024/07/09/evolve-bank-says-ransomware-gang-stole-personal-data-on-millions-of-customers/)): an employee clicked a malicious link and a LockBit operator reached Evolve's databases and file shares. Evolve noticed on **May 29, 2024** only because "some systems were not working properly" — first read as hardware failure; the intrusion had begun on **February 9, 2024**. Nearly four months of undetected dwell time in a regulated bank, with your customers' onboarding records inside the window.

**Gate:** diligence must interrogate detection, not marketing: real mean time to detect, a breach-notification SLA in hours, audit evidence that detection is staffed.

## When the processor forgets a key: Juspay

Juspay is the Bengaluru payments processor behind merchants including **Amazon and Swiggy**, processing roughly 650,000 transactions a day ([Business Today](https://www.businesstoday.in/technology/news/story/amazon-swiggy-payments-partner-juspay-suffers-data-breach-35-crore-records-compromised-283598-2021-01-05)). On **August 18, 2020** it detected unauthorized activity in a data store. The cause, per the company: **an old, unrecycled AWS access key** — a long-lived credential nobody had deleted ([CPO Magazine](https://www.cpomagazine.com/cyber-security/amazon-swiggy-payment-processor-juspay-downplays-data-breach-that-led-to-100-million-records-circulating-on-the-dark-web/)). The breach became public in January 2021 after researcher Rajshekhar Rajaharia spotted over **3.5 crore (35 million) records** for sale on the dark web. Juspay said the exposed records were masked card data and card fingerprints, but acknowledged a portion of its **10 crore (100 million) user records — including plain-text email addresses** — was also exposed (Business Today; [CPO Magazine](https://www.cpomagazine.com/cyber-security/amazon-swiggy-payment-processor-juspay-downplays-data-breach-that-led-to-100-million-records-circulating-on-the-dark-web/) reported 100 million records circulating).

The remediation list is the playbook in miniature: Juspay worked with merchants to **refresh API keys and invalidate the old ones**, enforced **two-factor authentication across internal tools**, and **moved off long-lived AWS access keys to IAM-based access** ([Business Today](https://www.businesstoday.in/technology/news/story/amazon-swiggy-payments-partner-juspay-suffers-data-breach-35-crore-records-compromised-283598-2021-01-05)). Rotate shared keys on incident, kill standing access, make credentials short-lived — the key Juspay forgot is the class of key you hold with every vendor.

## When the ledger stops adding up: Synapse

The most expensive middleman failure was not a hack at all. **Synapse** built the software layer linking fintech apps to partner banks — Evolve, American Bank, AMG National Trust and Lineage — and filed for Chapter 11 on **April 22, 2024** ([CFPB](https://www.consumerfinance.gov/enforcement/actions/synapse-financial-technologies-inc/)). Court-appointed trustee Jelena McWilliams, a former FDIC chair, then delivered the finding that froze the sector: **customers of Synapse-linked fintechs held $265 million in balances while the partner banks held only $180 million** — an **$85 million shortfall** whose source remained unknown ([CNBC](https://www.cnbc.com/2024/06/07/synapse-bankruptcy-trustee-85-million-of-customer-savings-is-missing.html)). More than **100,000 customers were locked out of their savings** for weeks while banks reconciled commingled ledgers. By March 2025 frozen money approached **$200 million**, the court had removed Synapse's management over "gross" mismanagement, and fintechs including Mercury were suing ([Fortune](https://fortune.com/2025/03/07/synapse-evolve-mercury-bankruptcy-lawsuits/)).

The failure was **reconciliation**: Synapse's ledgers disagreed with the banks', funds were commingled, and no control forced the two books to match each day. No penetration test finds an $85 million hole. Only a two-sided reconciliation does.

## The same failure modes, closer to home

Kenya has its own aggregator precedent. Safaricom sued aggregation partner **East African Data Handlers over KES 20.3 million** in losses from 2016 across the Lipa na M-PESA and Buy Goods platforms; the partner countersued, alleging Safaricom staff held back-end access to the aggregation system ([Tech-ish](https://tech-ish.com/2020/02/06/safaricom-east-african-data-handlers-kes-20-million-m-pesa-loss/), [full treatment](/posts/mpesa-daraja-api-pitfalls/)).

The regulatory floor points the same way. The Central Bank of Kenya's guideline on third-party agents (2016) states that the licensed institution "shall also be responsible for **assessing the adequacy of controls of outsourced activities** by taking appropriate direct or third party audits of the same as mandated under relevant outsourcing agreements" (clause 5.1.6, [CBK](https://www.centralbank.go.ke/wp-content/uploads/2016/08/GUIDELINE-ON-THE-APPOINTMENT-AND-OPERATIONS-OF-THIRD-PARTY-AGENTS-BY.pdf)). A BaaS or aggregator contract transfers *operations* — never accountability.

## The playbook: eight gates

| Gate | The question your integration must answer | Failure it exists to catch |
|------|------------------------------------------|---------------------------|
| 1. Layer map | Where does each credential, byte of customer data and shilling actually sit — and who else can reach it? | Evolve (data lived at the partner) |
| 2. Diligence | Breach history, real detection times, audit reports, sub-processors | Juspay (a researcher found the dump first) |
| 3. Contract | Notification SLA in hours, liability, audit rights, data handling on termination | Evolve (customers learned via filings) |
| 4. Credential lifecycle | No long-lived keys anywhere; vendor-key rotation runbook; secrets in a vault | Juspay (unrecycled AWS key) |
| 5. Authenticate every message | HMAC-verify webhooks and statements; replay windows; never credit an unsigned event | Forged "credited" messages |
| 6. Reconcile across hops | Daily two-ledger diff: your records vs the vendor's statement, by ID and amount | Synapse ($85M shortfall) |
| 7. Kill switch | Freeze flows, drain balances and export data within the SLA, without vendor goodwill | Synapse (100,000 locked out) |
| 8. Test your edges | Red-team the webhooks and status APIs *you* trust, not the vendor's internal apps | Blind trust in callbacks |

## Thirty lines that verify the edges

Two of the gates' controls, small enough to paste anywhere. First, **authenticate the vendor's messages**: the aggregator or BaaS signs each webhook (a deposit credit, a settlement) with an HMAC — recompute and compare before your ledger moves:

{% raw %}
```python
import hashlib, hmac, json

VENDOR_SECRET = "whsec_demo_2026"  # vault-held, never in git

def verify_webhook(body: bytes, signature: str) -> bool:      # Gate 5
    expect = hmac.new(VENDOR_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expect, signature)

genuine = {"type": "deposit.credited", "account": "ACC-7712",
           "amount_kes": 5000, "id": "evt_20260906_0001"}
raw = json.dumps(genuine, sort_keys=True, separators=(",", ":")).encode()
sig = hmac.new(VENDOR_SECRET.encode(), raw, hashlib.sha256).hexdigest()
print("genuine webhook accepted?", verify_webhook(raw, sig))
print("forged webhook accepted? ", verify_webhook(raw, "0" * 64))
```
{% endraw %}

Second, the **Synapse control**: reconcile daily. Your ledger records what you credited when each push arrived; the vendor's statement is the second source of truth:

{% raw %}
```python
ours   = {("evt_20260906_0001", 5000), ("evt_20260906_0002", 1250),
          ("evt_20260905_0999", 800)}        # Gate 6: what YOU recorded
theirs = {("evt_20260906_0001", 5000), ("evt_20260906_0002", 1250)}   # vendor statement
missing = sorted(ours - theirs)
print("credits vendor's books do not show:", missing)
print("action: freeze payout for that account, open a dispute within the SLA")
```
{% endraw %}

```text
genuine webhook accepted? True
forged webhook accepted?  False
credits vendor's books do not show: [('evt_20260905_0999', 800)]
action: freeze payout for that account, open a dispute within the SLA
```

The checks catch different lies. The HMAC gate rejects a **forged** message — an attacker who knows your webhook URL but not the vendor's secret. The ledger diff catches the harder case: every message genuine, yet the vendor's books still disagree with yours by one credit — the quiet condition that became Synapse's $85 million hole. Run both daily and the middleman's failure is your earliest warning, not your last surprise.

## The bottom line

A direct rail asks you to secure the joints you hold. A middleman asks you to trust a box you cannot see — and the failures above are that box leaking through a phishing click, a forgotten key and an unreconciled ledger. None would have been caught by hardening your own app; all were found — or could have been — by verifying the edges and reconciling the middle. The CBK's view is on the record: outsourcing transfers operations, never accountability. Treat every aggregator and BaaS partner as an extension of your perimeter — gates, signatures, daily two-sided reconciliation — and the box stops being a leap of faith.

## References

- BleepingComputer — [Evolve Bank says data breach impacts 7.6 million Americans](https://www.bleepingcomputer.com/news/security/evolve-bank-says-data-breach-impacts-76-million-americans/)
- TechCrunch — [Evolve Bank says ransomware gang stole personal data on millions of customers](https://techcrunch.com/2024/07/09/evolve-bank-says-ransomware-gang-stole-personal-data-on-millions-of-customers/)
- CNBC — [Synapse bankruptcy trustee says $85 million of customer savings is missing](https://www.cnbc.com/2024/06/07/synapse-bankruptcy-trustee-85-million-of-customer-savings-is-missing.html)
- CFPB — [Synapse Financial Technologies, Inc. enforcement action](https://www.consumerfinance.gov/enforcement/actions/synapse-financial-technologies-inc/)
- Fortune — [The spectacular Synapse collapse](https://fortune.com/2025/03/07/synapse-evolve-mercury-bankruptcy-lawsuits/)
- Business Today — [Amazon, Swiggy payments partner Juspay suffers data breach; 3.5 crore records compromised](https://www.businesstoday.in/technology/news/story/amazon-swiggy-payments-partner-juspay-suffers-data-breach-35-crore-records-compromised-283598-2021-01-05)
- CPO Magazine — [Juspay data breach: 100 million records circulating](https://www.cpomagazine.com/cyber-security/amazon-swiggy-payment-processor-juspay-downplays-data-breach-that-led-to-100-million-records-circulating-on-the-dark-web/)
- Tech-ish Kenya — [Safaricom sues East African Data Handlers over KES 20M M-PESA loss](https://tech-ish.com/2020/02/06/safaricom-east-african-data-handlers-kes-20-million-m-pesa-loss/)
- Central Bank of Kenya — [Guideline on the Appointment and Operations of Third-Party Agents (2016)](https://www.centralbank.go.ke/wp-content/uploads/2016/08/GUIDELINE-ON-THE-APPOINTMENT-AND-OPERATIONS-OF-THIRD-PARTY-AGENTS-BY.pdf)

## Related posts

- [Mobile Money API Security: Daraja Integration Pitfalls](/posts/mpesa-daraja-api-pitfalls/)
- [When the Contractor Has the Keys: Vendor Risk in Financial Systems](/posts/vendor-risk-fintech-contractors/)
- [Secrets in CI: How Credential Leaks Actually Happen](/posts/secrets-in-ci-credential-leaks/)
- [Anatomy of Flutterwave's Unauthorized Transfers](/posts/flutterwave-fraud-anatomy/)
- [The Three-Minute Insider: Privileged Access in Fintech](/posts/insider-threat-privileged-access-fintech/)
