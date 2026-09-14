---
title: "The Four Numbers Before the Ledger: Control Totals and Break Aging in Settlement"
date: 2026-09-14 00:00:00 +0300
categories: [Data Science, Fintech]
tags: [settlement, control totals, batch processing, reconciliation breaks, ach, bacs, kepss, payments, financial controls, ml ops]
image:
  path: /assets/img/cover-control-totals-break-aging.webp
  alt: A settlement batch of four detail blocks feeding a hexagonal control gate that checks count, hash, debit and credit totals, passing to POST on a green rail or falling into a DO NOT POST bin on a red rail, beside an aging ladder showing breaks at under 1 day, 1 to 7 days, 8 to 30 days and over 30 days sliding into a suspense account
---

## Two questions that look alike

> **Completeness is not correctness.**
> A file can arrive short, replayed, altered or internally balanced — and only one of those is a *reconciliation* problem. Control totals answer the pre-posting question (*is this what the sender sent?*); breaks answer the post-posting one (*does my ledger agree with the other side?*).
{: .prompt-info }

[Reconciliation analytics](/posts/reconciliation-analytics-fintech/) owns *what to compare*, [anomaly detection](/posts/anomaly-detection-reconciliation/) *what to look for*, [real-time alerting](/posts/real-time-reconciliation-alerting/) the paging path, [outage post-mortems](/posts/fintech-outage-post-mortem/) what an incident leaves behind. This post is the gate in front of them.

## The four numbers in a trailer record

Every mature clearing scheme carries its own arithmetic witness. An ACH batch ends with a **Batch Control Record** — Entry/Addenda Count, Entry Hash (a hash total of routing numbers, right-justified to ten digits) and Total Debit and Credit Entry Dollar Amounts — and the file ends with a **File Control Record** aggregating them ([ACH format](https://www.timetrex.com/glossary/what-is-an-ach-file)). Bacs returns a **Submission Report** with transaction count and total value, then an **Input Report** with the detail — the report a payroll team checks to confirm "the expected number of payments and total value match their payroll summary" ([paygate](https://www.paygate.uk/blog/bacs-payment-reports-guide/), [GoCardless](https://www.gocardless.com/direct-debit/receiving-messages)).

| Control total | Question answered | Blind spot |
|---------------|-------------------|------------|
| **Count** | Did every record arrive? | Which one is missing |
| **Hash total** (sum of account numbers) | Did the *set* change? | An amount altered in-set |
| **Debit total** | Does one side equal the sender's figure? | Extra records the sender counted too |
| **Credit total** | Is the file two-sided, `DR == CR`? | A valid payment sent twice |

Hash totals are the underrated number: summing account numbers yields a meaningless value, which is the point — it fingerprints the *identity and count* of records, not their value. Drop a record and count and hash move; alter an amount and the debit total moves; replay and the sequence number moves.

## The five gates, and what each one catches

Each gate must be a **refusal**: a total an operator can override is documentation, not a control.

| Gate | Question | Failure caught | Action |
|------|----------|----------------|--------|
| **G1** trailer present | Did every batch bring its trailer? | Truncated file, lost tail record | Refuse; request retransmission |
| **G2** batch sequence | Is this the batch we expect next? | Replay, retransmission | Refuse; a replay generates duplicates |
| **G3** batch recompute | Does the trailer describe what arrived? | In-flight alteration, dropped record | Refuse; the sender's arithmetic is the witness |
| **G4** file control record | Do the batches sum to the file header? | Partial file, mis-chunked upload | Refuse the file |
| **G5** two-sided | Is the file balanced? | One-sided entries, corrupted direction codes | Refuse; an unbalanced file is not postable |

Kenya's KEPSS rules show the item-level half of the same discipline: a bank receiving a multiple third-party payment with a problem in one payment **must not return the whole message** — only the problem payment, quoting main and related references, the rest applied as instructed ([KEPSS §11.3(f)](https://www.centralbank.go.ke/wp-content/uploads/2023/08/Revised-KEPSS-Rules-and-Procedures.pdf)). Refuse the item, settle the batch, track the difference.

## A control-total failure is not a break

| | Control-total failure | Break |
|---|---|---|
| **When** | Before posting | After posting |
| **Unit** | File or batch | An item, or an aggregate cell |
| **Signal** | Binary — refused | Graded — a value that should be zero |
| **Action** | Do not post; nothing moves | Post; carry the difference with an owner and an age |
| **Clock** | The sender's retransmission window | The customer's — RBI's T+1 reversal, or ₹100/day |
| **Cost** | A delayed batch | An aged break, or someone else's asset |

## The incidents were not exotic

**Citi, April 2024 — $81 trillion.** An employee pasted an account number into the amount field, turning a **$280** transfer into **$81 trillion** between two internal ledger accounts. A second employee checking it missed it too; a third caught it about 90 minutes after processing, and the entry was reversed hours later. No funds left the bank; it went to the Federal Reserve and the OCC as a "near miss" ([CBS](https://www.cbsnews.com/news/citi-mistakenly-credited-81-trillion-to-customer-account/), [NYT](https://www.nytimes.com/2025/02/28/nyregion/citigroup-81-trillion-error.html)). Citi's wording is the tell — "our **detective** controls promptly identified the inputting error". Detection after posting is downstream of a total that would have caught it.

**Citi, August 2020 — Revlon.** As agent for a 2016 Revlon loan the bank intended a **$7.8 million** interest payment and wired **just under $900 million** as a payoff; lenders refused to return funds, producing a c. **$500 million** loss — a "massive, unforced error" in its CEO's words ([Reuters](https://www.reuters.com/article/us-citigroup-revlon-lawsuit/citigroup-cannot-recoup-revlon-payouts-after-nearly-900-million-gaffe-u-s-judge-idUSKBN2AG1TJ/), [Maryland Smith](https://www.rhsmith.umd.edu/research/lessons-citis-revlon-error)). A total checked against scheduled interest plus fees would have failed by two orders of magnitude.

**Citi, 2 May 2022 — the fat finger.** A trader meant to sell **$58 million** of equities, keyed `58m` into *quantity* instead of *notional*, and built a 349-stock basket with a **$444 billion** notional. Systems blocked $255bn; the remaining **$189bn** reached the algorithm, which sold **$1.4bn** before the trader cancelled 15 minutes later — a **$48 million** loss. Regulators fined Citigroup Global Markets **£27.77M** (FCA) and **£33.88M** (PRA) — **£61.6m** — noting the trader could override a pop-up alert *without scrolling through the alerts inside it* ([Guardian](https://www.theguardian.com/business/article/2024/may/22/citigroup-fined-over-fat-finger-error-mistaken-orders), [Reuters](https://www.reuters.com/business/finance/citi-fined-79-mln-by-uk-regulators-over-trading-control-failures-2024-05-22/)). The limit existed; the enforcement did not.

**Deutsche Bank, 16 March 2018 — €28 billion.** In routine derivatives dealings the bank transferred **€28bn (≈$34bn)** to its own account at Eurex — more than its entire **€24bn** market capitalisation — an operation "meant to involve a far smaller sum, which the bank has not revealed". It was corrected the same day, and the ECB asked for clarification ([AFP](https://phys.org/news/2018-04-oopsdeutsche-bank-28bn-euro-error.amp), [Reuters](https://www.reuters.com/article/us-ecb-deutsche-bank/ecb-asks-deutsche-bank-to-clarify-mistaken-34-billion-transfer-report-idUSKBN1HR2SV/)).

Four times, one shape: a plausible instruction, a pre-computed number nobody forced to agree, and a system happy to move the money.

## The demo: five faults, five gates, one that walks through

Six two-sided batches, each with an honest trailer; five injected faults, distinguishable only by the control totals.

{% raw %}
```python
import random
from collections import namedtuple

ENTRY = namedtuple("ENTRY", "account amount direction")   # D = debit, C = credit
NETTING = 99_999_999          # internal netting account, keeps every batch two-sided


def make_day(seed=14, n_batches=6):
    rng, batches, seq = random.Random(seed), [], 1
    for _ in range(n_batches):
        e = []
        for _ in range(rng.randint(180, 260)):
            e.append(ENTRY(rng.randint(10_000_000, 99_999_999),
                           rng.choice([rng.randint(500, 90_000),
                                       rng.randint(100_000, 4_800_000),
                                       rng.randint(5_000_000, 40_000_000)]),
                           "D" if rng.random() < 0.55 else "C"))
        net = sum(x.amount if x.direction == "D" else -x.amount for x in e)
        if net > 0:
            e.append(ENTRY(NETTING, net, "C"))
        elif net < 0:
            e.append(ENTRY(NETTING, -net, "D"))
        batches.append({"seq": seq, "entries": e})
        seq += 1
    return batches


def totals(entries):
    return {"count": len(entries),
            "hash": sum(x.account for x in entries) % 10 ** 10,
            "dr": sum(x.amount for x in entries if x.direction == "D"),
            "cr": sum(x.amount for x in entries if x.direction == "C")}


def trailer(batch):
    return dict(totals(batch["entries"]), seq=batch["seq"])


def file_record(batches):
    t = {"count": 0, "hash": 0, "dr": 0, "cr": 0, "batches": len(batches)}
    for b in batches:
        for k in ("count", "hash", "dr", "cr"):
            t[k] += trailer(b)[k]
    t["hash"] %= 10 ** 10
    return t


def gates(after, declared_file, declared_trailers):
    """Five pre-posting questions. Each one is answerable from the file alone."""
    fired = []
    missing = [b["seq"] for b in after if b["seq"] not in declared_trailers]
    if missing:                                            # G1 every batch has a trailer
        fired.append(("G1", f"batch {missing[0]} has no trailer"))
    seqs = [b["seq"] for b in after]
    if seqs != list(range(1, len(seqs) + 1)):              # G2 batch sequence is 1..n
        fired.append(("G2", f"sequence {seqs} is not 1..{len(seqs)}"))
    for b in after:                                        # G3 recompute vs trailer
        d = declared_trailers.get(b["seq"])
        if not d:
            continue
        r = trailer(b)
        for k in ("count", "hash", "dr", "cr"):
            if r[k] != d[k]:
                fired.append(("G3", f"batch {b['seq']} {k}: recomputed {r[k]:,} "
                                    f"!= trailer {d[k]:,} (delta {abs(r[k]-d[k]):,})"))
    r = file_record(after)
    if not missing:                                        # G4 file control record
        for k in ("count", "hash", "dr", "cr", "batches"):
            if r[k] != declared_file[k]:
                fired.append(("G4", f"file {k}: recomputed {r[k]:,} "
                                    f"!= control record {declared_file[k]:,}"))
    if r["dr"] != r["cr"]:                                 # G5 the file is two-sided
        fired.append(("G5", f"DR {r['dr']:,} vs CR {r['cr']:,}"))
    return fired, r


def f_truncate(b):
    b[-1]["entries"] = [b[-1]["entries"][0]] + b[-1]["entries"][2:]
    return "last batch arrives with one detail record missing"


def f_duplicate(b):
    d = dict(b[2]); d["seq"] = 9; b.append(d)
    return "batch 3 retransmitted, renumbered 9"


def f_alter(b):
    e = b[2]["entries"][7]
    b[2]["entries"][7] = e._replace(amount=e.amount + 360_000)
    return "one amount altered in transit (+360,000)"


def f_sequence(b):
    b[4]["seq"] = 15
    return "batch 5 renumbered 15 by a broken integration"


def f_inside(b):
    e = b[5]["entries"][3]
    b[5]["entries"].append(ENTRY(e.account, e.amount, "C"))
    b[5]["entries"].append(ENTRY(e.account, e.amount, "D"))
    return "a duplicate payment with both legs, netted and honestly counted"


FAULTS = [("truncate", f_truncate, False), ("duplicate", f_duplicate, False),
          ("alter", f_alter, False), ("sequence", f_sequence, False),
          ("inside", f_inside, True)]   # True = the sending system's own totals include it


def run(fault, seed=14):
    day = make_day(seed)
    declared_file, declared_trailers = file_record(day), {b["seq"]: trailer(b) for b in day}
    label = fault[1](day)
    if fault[2]:                       # honest trailer: totals computed after the fault
        declared_file = file_record(day)
        declared_trailers = {b["seq"]: trailer(b) for b in day}
    fired, r = gates(day, declared_file, declared_trailers)
    verdict = "REFUSED PRE-POSTING" if fired else "POSTED"
    first = f"{fired[0][0]} {fired[0][1]}" if fired else f"{r['count']:,} entries, DR == CR"
    return label, verdict, first


clean = file_record(make_day(14))
print(f"clean day: {clean['count']:,} entries, DR {clean['dr']:,} = CR {clean['cr']:,}, "
      f"hash {clean['hash']:,}\n")
print(f"{'fault':<10}{'verdict':<21}first gate to fire")
for name, fn, honest in FAULTS:
    label, verdict, first = run((name, fn, honest))
    print(f"{name:<10}{verdict:<21}{first}")
    print(f"{'':<10}{label}")

BUCKETS = [("< 1 day", 41, 18_400_000, 0.25, 0.00), ("1-7 days", 27, 9_650_000, 0.75, 0.00),
           ("8-30 days", 14, 4_100_000, 3.00, 0.05), ("> 30 days", 6, 2_780_000, 8.00, 0.60)]
total = sum(b[2] for b in BUCKETS)
print(f"\n{'breaks by age':<14}{'n':>4}{'value (KES)':>15}{'age-share':>11}{'effort h':>10}")
for name, n, val, hours, _ in BUCKETS:
    print(f"{name:<14}{n:>4}{val:>15,}{100*val/total:>10.1f}%{n*hours:>10.1f}")
aged = [b for b in BUCKETS if b[0] in ("8-30 days", "> 30 days")]
print(f"aged > 7 days: {sum(b[1] for b in aged)} of {sum(b[1] for b in BUCKETS)} breaks, "
      f"KES {sum(b[2] for b in aged):,} ({100*sum(b[2] for b in aged)/total:.1f}% of value); "
      f"write-off risk KES {sum(int(b[2]*b[4]) for b in BUCKETS):,}")
```
{% endraw %}

```
clean day: 1,170 entries, DR 5,759,092,508 = CR 5,759,092,508, hash 4,862,209,760

fault     verdict              first gate to fire
truncate  REFUSED PRE-POSTING  G3 batch 6 count: recomputed 190 != trailer 191 (delta 1)
          last batch arrives with one detail record missing
duplicate REFUSED PRE-POSTING  G1 batch 9 has no trailer
          batch 3 retransmitted, renumbered 9
alter     REFUSED PRE-POSTING  G3 batch 3 cr: recomputed 875,767,475 != trailer 875,407,475 (delta 360,000)
          one amount altered in transit (+360,000)
sequence  REFUSED PRE-POSTING  G1 batch 15 has no trailer
          batch 5 renumbered 15 by a broken integration
inside    POSTED               1,172 entries, DR == CR
          a duplicate payment with both legs, netted and honestly counted

breaks by age    n    value (KES)  age-share  effort h
< 1 day         41     18,400,000      52.7%      10.2
1-7 days        27      9,650,000      27.6%      20.2
8-30 days       14      4,100,000      11.7%      42.0
> 30 days        6      2,780,000       8.0%      48.0
aged > 7 days: 20 of 88 breaks, KES 6,880,000 (19.7% of value); write-off risk KES 1,873,000
```

Four of the five faults never reach the ledger, each named by the gate that refused it. The fifth — a duplicate created *inside* the batch, both legs present, honestly counted, netting to zero — **passes every gate**, because each gate asks whether the file is faithful to itself, not whether the batch should have been sent. Control totals bound *mechanical* failure to zero; only the other side of the book — the beneficiary's statement, the counterparty's file — sees a well-formed duplicate.

## Break aging: the cost that survives the gates

A break is not an anomaly with a resolution time; it is a **liability with an owner and an age**.

| Bucket | Meaning | Handling |
|--------|---------|----------|
| **< 1 day** | Timing — the other side has not posted | Auto-retry next cycle |
| **1–7 days** | Real difference — reference, fee, FX, wrong account | Named owner, statement pulled, cause coded |
| **8–30 days** | Disputed, or the counterparty unreachable | Escalation, formal claim, provisional posting |
| **> 30 days** | Nobody owns it | Suspense or write-off |

The demo's ladder is small — 88 breaks, KES 34.9M — and its shape is the durable part: **the oldest bucket holds the smallest share of items and the largest share of the risk**. Twenty breaks aged past seven days carry KES 6.88M — under a fifth of the value, but 90 hours of investigation and nearly all of the KES 1.87M expected write-off.

Unclosed items end somewhere. In Kenya an item with no active involvement for **two years** is declared unclaimed, and an estimated **KES 241.1 billion** of unclaimed financial assets sits across the economy — about **62% in financial services** — with dormancy-driven items at a record **KES 5.182 billion** in 2025 while claimants fell 32.7% ([Kenya News Agency](https://www.kenyanews.go.ke/sh241-billion-unclaimed-financial-assets-ufaa-releases/), [The Star](https://www.the-star.co.ke/news/2026-05-06-unclaimed-financial-assets-hit-sh518bn-claimant-numbers-drop)). An aged break changes owner, and becomes a statutory problem.

## How we can do better

| Control | Why it holds |
|---------|--------------|
| **Gate, don't warn** — a failed total refuses the file | Citi's overridable pop-up cost £61.6m |
| **Recompute, never trust** — recalculate from the details received | A trailer is the sender's claim, not evidence |
| **Two-sided by construction** — add an explicit netting entry | An unbalanced file should be unpostable |
| **Hash totals on identity** — sum account numbers, not only values | Catches an altered set that still balances |
| **Sequence discipline** — batch numbers strictly `1..n` | Turns a replay into a refusal, not a duplicate |
| **Age on arrival** — stamp the settlement date it should have had | A break found late must not look young |
| **Owner and root cause per break** | Feeds the histogram that fixes integrations |

## Key takeaways

| Takeaway | Why it matters |
|----------|----------------|
| Control totals are a *pre-posting* completeness proof | A count, a sum and a hash catch truncation, replay and alteration for free |
| Internal consistency is not intent | A well-formed duplicate passes every gate; only the counterparty sees it |
| Breaks age into liabilities | Two years of dormancy makes an unclosed item someone else's asset |
| The incidents were ordinary input errors | Citi's four, at $81T, $900M, $444bn and €28bn, were all stoppable before posting |

## References

- ACH file format: control records and entry hash — [timetrex.com](https://www.timetrex.com/glossary/what-is-an-ach-file)
- Bacs payment reports — [paygate.uk](https://www.paygate.uk/blog/bacs-payment-reports-guide/), [GoCardless](https://www.gocardless.com/direct-debit/receiving-messages)
- Central Bank of Kenya, *Revised KEPSS Rules and Procedures*, §11.3(f) — [centralbank.go.ke](https://www.centralbank.go.ke/wp-content/uploads/2023/08/Revised-KEPSS-Rules-and-Procedures.pdf)
- CBS News, *Citi mistakenly credits $81 trillion*, 28 Feb 2025 — [cbsnews.com](https://www.cbsnews.com/news/citi-mistakenly-credited-81-trillion-to-customer-account/)
- Reuters, *Citigroup cannot recoup Revlon payouts* — [reuters.com](https://www.reuters.com/article/us-citigroup-revlon-lawsuit/citigroup-cannot-recoup-revlon-payouts-after-nearly-900-million-gaffe-u-s-judge-idUSKBN2AG1TJ/)
- Maryland Smith, *Lessons from Citi's Revlon Error*, 22 Mar 2021 — [rhsmith.umd.edu](https://www.rhsmith.umd.edu/research/lessons-citis-revlon-error)
- Guardian, *Citigroup fined over 'fat finger' error*, 22 May 2024 — [theguardian.com](https://www.theguardian.com/business/article/2024/may/22/citigroup-fined-over-fat-finger-error-mistaken-orders)
- AFP, *Deutsche Bank makes 28bn euro transfer in error*, 20 Apr 2018 — [phys.org](https://phys.org/news/2018-04-oopsdeutsche-bank-28bn-euro-error.amp)
- Kenya News Agency, *Sh241 billion unclaimed financial assets, UFAA releases* — [kenyanews.go.ke](https://www.kenyanews.go.ke/sh241-billion-unclaimed-financial-assets-ufaa-releases/)
- The Star, *Unclaimed financial assets hit Sh5.18bn*, 6 May 2026 — [the-star.co.ke](https://www.the-star.co.ke/news/2026-05-06-unclaimed-financial-assets-hit-sh518bn-claimant-numbers-drop)

## Related posts

- [Paging the Ledger: Real-Time Reconciliation Alerting That Actually Wakes Someone](/posts/real-time-reconciliation-alerting/)
- [Anomaly Detection for Reconciliation at Scale](/posts/anomaly-detection-reconciliation/)
- [Reconciliation Analytics: How Data Catches What Dashboards Miss](/posts/reconciliation-analytics-fintech/)
- [The Money Stopped Moving: What Fintech Outage Post-Mortems Actually Measure](/posts/fintech-outage-post-mortem/)
- [Building an Aggregator/BaaS Security Playbook](/posts/aggregator-baas-security-playbook/)
