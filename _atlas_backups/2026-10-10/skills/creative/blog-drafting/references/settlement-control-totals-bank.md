# Settlement control totals & break aging — verified anchor bank

Class: **batch/settlement input integrity** (pre-posting control totals) and **break aging**
(post-posting liability management). Distinct from the reconciliation family:
- `reconciliation-analytics-fintech` (Aug 15) = what to compare
- `anomaly-detection-reconciliation` (Aug 31) = what to look for
- `real-time-reconciliation-alerting` (Sep 13) = how to page it
- `fintech-outage-post-mortem` (Sep 12) = what to count after an incident
- **`settlement-control-totals-break-aging` (Sep 14) = the gate before posting + the aging ladder**

## Control-total mechanics (scheme documentation, verbatim-grade)

- **ACH (Nacha) file structure** — Batch Control Record (Type 8): Entry/Addenda Count,
  **Entry Hash** = hash total of the routing numbers in the batch (right-justified to 10
  digits if the sum exceeds 10 digits), Total Debit Entry Dollar Amount, Total Credit Entry
  Dollar Amount; File Control Record (Type 9) aggregates the batch records.
  Source (readable, no paywall): https://www.timetrex.com/glossary/what-is-an-ach-file
  (Nacha's own Operating Rules are paywalled — cite the public format reference.)
- **Bacs** — Submission Report shows file name, submission date, **number of transactions and
  total value**; Input Report gives the per-transaction detail. The payroll-team check
  ("expected number of payments and total value match their payroll summary") is the
  real-world use of a control total.
  Sources: https://www.paygate.uk/blog/bacs-payment-reports-guide/ ,
  https://www.gocardless.com/direct-debit/receiving-messages
- **Kenya KEPSS (CBK) Revised Rules and Procedures** — §11.2 payment instructions are "final
  and irrevocable once the Forwarding bank's account is debited and the Executing bank
  account is credited"; §11.3(b)(ii) explicitly lists "the payment was made in error by the
  Forwarding bank" as a recall reason, with the Executing bank seeking an **indemnity**;
  §11.3(f) a bank receiving a multiple third-party payment with one bad payment **must NOT
  reject the whole message** — return only the problem payment quoting main and related
  references, apply the rest. §15.4 liquidity before Final Cut-off.
  PDF: https://www.centralbank.go.ke/wp-content/uploads/2023/08/Revised-KEPSS-Rules-and-Procedures.pdf
  (Fetch with `curl -sL -o k.pdf URL && pdftotext -layout k.pdf k.txt` — the extract-web-text.py
  script mangles PDFs, printing "KEYPHRASE NOT FOUND".)
  KEPSS operating hours extended to **07:00–19:00** from 1 July 2025 (was 08:30–16:30):
  https://www.capitalfm.co.ke/business/2025/06/cbk-extends-kepsss-bulk-payment-settlement-time/

## Input-control incident anchors (2+ sources each, bodies read)

| Incident | Key verified figures | Sources |
|---|---|---|
| **Citi $81T near miss** (Apr 2024; disclosed Feb 2025) | account number pasted into amount field; **$280 → $81 trillion** between two internal ledger accounts; missed by two staff; third found it ~90 min after processing; reversed hours later; reported to Fed + OCC as "near miss"; Citi: "our **detective** controls promptly identified the inputting error"; 10 near misses ≥$1B in 2024 vs 13 in 2023 | CBS News https://www.cbsnews.com/news/citi-mistakenly-credited-81-trillion-to-customer-account/ ; NYT https://www.nytimes.com/2025/02/28/nyregion/citigroup-81-trillion-error.html |
| **Citi/Revlon erroneous wire** (Aug 2020) | intended **$7.8M** interest; wired **just under $900M** (Reuters: $893M) structured as a payoff; some lenders refused to return → c. **$500M** loss; CEO "massive, unforced error"; manual adjustment by operator | Reuters https://www.reuters.com/article/us-citigroup-revlon-lawsuit/citigroup-cannot-recoup-revlon-payouts-after-nearly-900-million-gaffe-u-s-judge-idUSKBN2AG1TJ/ ; Maryland Smith https://www.rhsmith.umd.edu/research/lessons-citis-revlon-error |
| **Citi "fat finger"** (2 May 2022) | intended **$58M** sell; `58m` keyed into *quantity* not *notional* → 349-stock basket, **$444bn** notional; systems blocked $255bn, remaining **$189bn** hit the algo, **$1.4bn** sold; cancelled 15 min later; **$48M** loss; FCA **£27.77M** + PRA **£33.88M** = **£61.6M**; trader could override a pop-up without scrolling through the alerts in it | Guardian https://www.theguardian.com/business/article/2024/may/22/citigroup-fined-over-fat-finger-error-mistaken-orders ; Reuters https://www.reuters.com/business/finance/citi-fined-79-mln-by-uk-regulators-over-trading-control-failures-2024-05-22/ |
| **Deutsche Bank €28bn** (16 Mar 2018) | €28bn (≈$34bn) transferred to its own account at Eurex in routine derivatives dealing; **more than its €24bn market cap**; "meant to involve a far smaller sum, which the bank has not revealed"; corrected same day; ECB asked for clarification | AFP https://phys.org/news/2018-04-oopsdeutsche-bank-28bn-euro-error.amp ; Reuters https://www.reuters.com/article/us-ecb-deutsche-bank/ecb-asks-deutsche-bank-to-clarify-mistaken-34-billion-transfer-report-idUSKBN1HR2SV/ |

Reuters.com returns **HTTP 401 to curl** (2026) — use the Guardian/CBS/AFP/Maryland variants above;
Reuters URLs still work as citations (they open in a browser) but cannot be body-verified by curl.

## Aged-break / unclaimed end-state anchors (Kenya)

- **UFAA baseline survey** (Kenya News Agency): unclaimed financial assets estimated at
  **KES 241,105,748,942**; ~477,112 estimated holders; **financial services sector ≈ 62%** of
  the total; projected KES 156bn over 5 years; assets declared unclaimed after **2 years** of
  no active involvement; UFAA had collected KES 20.3bn cash + KES 1.2bn shares.
  https://www.kenyanews.go.ke/sh241-billion-unclaimed-financial-assets-ufaa-releases/
  (fetch via `https://r.jina.ai/<url>` — the site serves a self-signed cert that breaks
  extract-web-text.py: "SSL: CERTIFICATE_VERIFY_FAILED".)
- **The Star (6 May 2026)**: unclaimed financial assets hit a record **KES 5.182bn** in 2025,
  claimants down 32.7%.
  https://www.the-star.co.ke/news/2026-05-06-unclaimed-financial-assets-hit-sh518bn-claimant-numbers-drop

## Demo recipe (verified, seed-robust)

`_posts/2026-09-14-*.md` block 1 — stdlib only, ~60 lines: 6 batches × 180–260 entries, each
batch force-balanced with a netting entry; five gates (G1 trailer present, G2 sequence 1..n,
G3 recompute vs trailer, G4 file control record, G5 DR == CR); five injected faults
(truncate, duplicate/renumbered replay, in-flight alteration, renumbered sequence, and the
**honest-trailer duplicate** that must PASS every gate). Verified output (seed 14): clean day
1,170 entries, DR = CR = 5,759,092,508, hash 4,862,209,760; 4/5 faults REFUSED; `inside` POSTED
with 1,172 entries. Gate verdicts identical for seeds 7/14/42 (only values move).

Key design point: the "inside" fault must compute the trailer **after** the fault, otherwise
G3 trivially catches it and the post's thesis (control totals can't see a well-formed duplicate)
collapses. Watch for this when reusing the recipe.
