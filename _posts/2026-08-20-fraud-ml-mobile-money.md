---
title: "Fraud ML in Mobile Money: Building Models That Catch the 70-Account Loophole"
date: 2026-08-20 00:00:00 +0300
categories: [Machine Learning, Fintech]
tags: [fraud detection, anomaly detection, mobile money, feature engineering, model monitoring]
image:
  path: /assets/img/cover-fraud-ml-mobile-money.webp
  alt: Machine learning for fraud detection in mobile money
---

## The fraud that slipped through for eight days

In June 2025, NCBA Bank Rwanda lost approximately Ksh 57.5 million (USD 446,000) in one of the region's most instructive mobile money frauds. A contractor engaged for system maintenance modified the bank's MTN Mobile Money (MoMo) integration so that balance validation was bypassed — for exactly 70 accounts. Between June 6 and 14, those "ghost" accounts executed 260 withdrawals of money that did not exist, and the fraud was discovered only during an end-of-week reconciliation when NCBA compared its records against MTN Rwanda's ([254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/), [Kenya Insights](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/), [Business Daily](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)).

The uncomfortable question for ML teams: why did no system flag this within hours? Because the attack was engineered to look normal at the transaction level. This post walks through the model architecture that would have caught it.

## Why supervised models miss novel fraud

The NCBA pattern is a textbook case of what fraud researchers call the **zero-day problem**. Each withdrawal was individually unremarkable: a plausible amount, a valid account, a successful switch response. What made it fraudulent was a property no single transaction exposes — that 70 accounts were withdrawing against balances that were never debited.

Supervised classifiers trained on labelled historical fraud learn to recognise *known* attack shapes: unusual amounts, odd hours, rapid carding. Novel attacks occupy regions of feature space the model never saw, and worse, they are optimised by attackers to sit inside the "normal" region. The NCBA withdrawals were deliberately routed through the legitimate integration, so per-transaction scores stayed low.

The answer is a layered defence: **unsupervised anomaly detection** (isolation forests, autoencoders) that flags anything far from the historical distribution, plus **graph features** that capture relationships between accounts — because fraud is rarely one account, it is a cluster.

## Features that matter for mobile money fraud

> **Rule of thumb**
> If a feature can only be computed after the fraud is known, it is useless for detection. Everything below is computable in real time from data you already log.
{: .prompt-tip }

**Velocity features.** Count and sum of withdrawals per account per hour/day, compared against a rolling 30-day baseline:

```python
vel = (txn.groupby(["account_id", txn["ts"].dt.date])
          .agg(wd_count=("amount", "count"), wd_sum=("amount", "sum")))

baseline = txn[txn["ts"] < cutoff].groupby("account_id")["amount"] \
    .agg(["mean", "std"]).rename(columns={"mean": "avg_daily", "std": "sd_daily"})

flag = vel.join(baseline, on="account_id")
flag["z"] = (flag["wd_sum"] - flag["avg_daily"]) / flag["sd_daily"].clip(lower=1)
```

**Consistency features.** The killer feature for this incident: does the ledger say *Success* while the settlement/switch record shows *no debit*? Derive this label from dual-side logs:

```python
mismatch = (ledger["status"] == "SUCCESS") & (~settlement["debited"])
# money left the bank with no corresponding source of funds
```

For the 70 NCBA accounts this mismatch rate was ~100%; for legitimate traffic it should be near zero. That single derived feature separates this fraud class from everything else.

**Graph features.** Shared device IDs, wallet numbers, counterparties, and IP clusters. Seventy accounts created around the same time, transacting with the same few counterparties, form a dense community that a community-detection pass or simple degree features expose quickly.

**KYC-state features.** Account age, verification status, channel of onboarding. Ghost accounts tend to be new, unverified, and opened in bursts.

## Imbalance, thresholds, and cost-sensitive evaluation

Fraud is rare — often well under 0.1% of transactions. Accuracy is meaningless; you need **precision/recall** and a threshold tuned on business cost, not F1. A missed fraud of Ksh 800,000 outweighs thousands of false positives that an analyst dismisses in seconds. Evaluate models on **monetary loss avoided per Ksh of review cost**: rank alerts by expected value `P(fraud) × amount`, and measure cumulative loss captured down the queue. AUC hides this; a cost curve does not.

## Monitoring: catching the "invisible" attack

The NCBA fraud changed the *distribution* of features — the mismatch flag went from ~0% to 100% on one channel — even though no single transaction screamed. That is exactly what **drift monitoring** is for. Track:

- **PSI (Population Stability Index)** per feature per channel; PSI > 0.25 on the consistency features means something structural changed.
- **Prediction distribution shifts**: if the share of transactions scoring above 0.8 on a channel doubles, alert — regardless of labels.
- A fixed **retraining cadence** (weekly/monthly) plus on-alert retraining with newly confirmed cases.

Channel-level alerting would have fired on the MoMo integration within hours of the first mismatched withdrawal.

## Human-in-the-loop

Models queue alerts; people close the loop. Investigators review the alert queue (ranked by expected loss), confirm or dismiss, and confirmed cases flow back into training data with a feedback label. This is what makes the unsupervised layer improve over time — and it is also your control against the ML itself being gamed.

## The NCBA counterfactual

Which layers catch 70 accounts withdrawing against un-debited balances within hours?

1. **Consistency flag**: 100% success-without-debit → instant alert.
2. **Velocity**: 260 withdrawals in 8 days across 70 accounts means several accounts hit multi-sigma daily counts.
3. **Graph + KYC**: a dense cluster of new, unverified accounts sharing counterparties.
4. **Monitoring**: PSI spike on the MoMo channel even if thresholds were lax.

None of these require knowing the attack in advance. That is the point: in mobile money, the next fraud will not look like the last one — build for novelty.

## Conclusion

The NCBA incident cost Ksh 57.5 million because detection relied on reconciliation after the fact. An ML defence built on consistency features, velocity, graph structure, drift monitoring, and cost-aware thresholds would have surfaced the 70 accounts in hours — not days. For fintech ML teams in East Africa, the lesson is direct: your models must be built to catch what you have never seen.

## References

- [How a three-minute code change triggered Sh57.5 million NCBA fraud — 254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)
- [How NCBA Software Engineer Opened Floodgates For Mobile Banking System Fraud — Kenya Insights](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
- [Software developer held on Sh58m NCBA fraud claim — Business Daily](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)
- [How bank customers lost Sh517m in IT system fraud — Business Daily](https://www.businessdailyafrica.com/bd/corporate/companies/how-bank-customers-lost-sh517m-in-it-system-fraud-5112506)
