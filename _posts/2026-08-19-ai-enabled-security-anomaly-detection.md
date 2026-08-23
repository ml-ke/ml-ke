---
title: "AI-Enabled Security: Seeing the Anomaly Humans and Static Rules Miss"
date: 2026-08-19 00:00:00 +0300
categories: [AI Security]
tags: [anomaly detection, fraud detection, UEBA, fintech security, machine learning, transaction monitoring]
image:
  path: /assets/img/cover-ai-enabled-security-anomaly-detection.webp
  alt: AI-enabled security and anomaly detection in fintech
---

## The Fraud That Looked Like Normal Traffic

In June 2025, a contractor hired to maintain NCBA Bank Rwanda's mobile banking platform allegedly altered its integration logic with MTN's mobile money rails. Court filings reported by [254 News](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/) describe access activated at 5:30 a.m. on his first day and a "three-minute code change" letting 70 customer accounts initiate 260 withdrawals totalling USD 446,000 (~Ksh 57.5 million) — paid out against accounts that were empty or did not exist, per [Court Helicopter](https://www.courthelicopter.ke/court-detains-contractor-over-ksh-57-5-million-fraud-at-ncba-bank/) and [Nairobi Timez](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/). Evans Nandwa of Ronford Digital Limited was contracted on June 6, 2025 to upgrade the subsidiary's MTN-based platform ([Kenya Insights](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/), [Business Daily](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)).

The scheme was not stopped by monitoring, rules, or alerts. It surfaced only during an end-of-week reconciliation against MTN Rwanda's records. That gap is this post's subject: the class of fraud that static, rule-based detection is structurally blind to — and how AI/ML sees it.

## The Class of Anomaly Rules Miss

Rules are the workhorses of fintech security: velocity limits, country blocks, device blacklists, amount thresholds. They encode what we already know. The NCBA scheme had every property that defeats a rule: **novel** (no prior example to pattern-match), **low-frequency** (260 withdrawals spread across 70 accounts is not a per-account velocity spike), and **designed to look normal** (payments flowing through a legitimate integration, from accounts that existed in the system). Viewed account-by-account and transaction-by-transaction, every individual event was boring.

This is the standard failure mode: rule sets are precision machines — every alert is meaningful — but their recall is bounded by what someone thought to encode. Novel fraud sits in the recall gap.

## What AI Sees Instead

**User and Entity Behavior Analytics (UEBA)** scores activity against baselines built from the user's own history and their peer group. A contractor on his first day, at 5:30 a.m., exercising privileged access to a core integration service is an extreme outlier on every axis: time-of-day, recency-of-onboarding, resource touched, and deviation from peer behaviour.

**Unsupervised anomaly detection** — isolation forests, autoencoders, or one-class SVMs over transaction and session features — flags events that are rare in feature space without labelled fraud. It catches the *shape* of the NCBA pattern: withdrawals from a freshly modified service, with inter-arrival times and amount distributions unlike anything in history.

**Transaction graph analytics** attacks account-level blindness directly. Individually innocuous, the 70 accounts cluster in the graph of shared wallets, devices, IPs, and counterparties — a dense subgraph of accounts with no other reason to be connected. Graph neural networks and community-detection surface such collusion rings routinely.

**Drift monitoring** closes the loop: a sudden shift in the distribution of channel-level predictions is itself an alert, signalling that the underlying process changed — exactly what happens when someone flips a logic switch in an integration.

## Proof It Works at Scale

This isn't theoretical. **PayPal** runs an adaptive ML fraud engine over its two-sided network, scoring transactions on hundreds of signals in real time; its risk platform is the company's largest production AI/ML environment ([PayPal](https://www.paypal.com/us/brc/article/payment-fraud-detection-machine-learning), [PayPal Tech Blog](https://medium.com/paypal-tech/machine-learning-model-ci-cd-and-shadow-platform-8c4f44998c78), [Emerj](https://emerj.com/artificial-intelligence-at-paypal/)).

**Stripe Radar** trains on network-wide data — 70 trillion data points and a 32% average fraud reduction, per Stripe — and scores every payment with an adaptive risk model in real time ([Stripe Radar](https://stripe.com/radar), [Stripe docs](https://docs.stripe.com/radar/risk-evaluation), [Stripe's ML primer](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection)).

**Mastercard's Decision Intelligence** scores transactions during authorization in about 50 milliseconds ([Business Insider](https://www.businessinsider.com/mastercard-ai-credit-card-fraud-detection-protects-consumers-2025-5), [Mastercard](https://www.mastercard.com/global/en/business/cybersecurity-fraud-prevention/risk-decisioning/decision-intelligence.html)).

The most instructive documented case is **Danske Bank**, whose rules-based engine caught only ~40% of fraud while drowning investigators in up to 1,200 false positives per day — 99.5% of investigated cases were innocent. After deploying ML models with Teradata, false positives fell by ~50% while detection rates rose ~60% ([PR Newswire](https://www.prnewswire.com/news-releases/danske-bank-and-teradata-implement-artificial-intelligence-ai-engine-that-monitors-fraud-in-real-time-300540944.html), [Teradata case study](https://assets.teradata.com/resourceCenter/downloads/CaseStudies/CaseStudy_EB9821_Danske_Bank_Saves_Millions_Fighting_Fraud_With_Deep_Learning_and_AI.pdf), [O'Reilly](https://www.oreilly.com/library/view/achieving-real-business/9781492038214/ch06.html)).

## The Deployment Pattern That Works

The mature pattern is layered, not adversarial: **rules catch known fraud, AI catches unknown fraud, humans review the queue.** Rules stay in the hot path: fast, explainable, auditable. ML models operate alongside, ranking the long tail rules never encoded. Investigators work a prioritized queue, not a firehose — which is why false-positive rate, not just recall, determines whether a deployment survives contact with a SOC.

Fraud is rare, so precision/recall trade-offs are brutal. Fintechs tune for high recall in the top of the score distribution, accept false positives as a cheap review cost, and monitor per-channel thresholds and model drift continuously. Explainability matters too: alerts should carry their top contributing features, or investigators stop trusting the model.

> **Rules are necessary, not sufficient**
> No serious fraud team deletes its rules — but rules alone cannot see what nobody has encoded yet, and that is where the next NCBA-style scheme will live.
{: .prompt-info }

The NCBA loss was discovered by reconciliation — after the money was gone. The lesson is not to replace rules with ML, but to add the layer that watches for the anomaly no one thought to define.

## References

1. [254 News — How a three-minute code change triggered Sh57.5 million NCBA fraud](https://254news.co.ke/how-a-three-minute-code-change-triggered-sh57-5-million-ncba-fraud/)
2. [Kenya Insights — How NCBA software engineer opened floodgates for mobile banking fraud](https://kenyainsights.com/how-ncba-software-engineer-opened-floodgates-for-mobile-banking-system-fraud/)
3. [Nairobi Timez — Banking Fraud Unit allowed to detain software developer](https://nairobitimez.co.ke/2025/06/18/banking-fraud-unit-allowed-to-detain-software-developer-suspected-of-defrauding-bank-of-sh57-million/)
4. [Court Helicopter — Court detains contractor over Ksh 57.5M fraud at NCBA](https://www.courthelicopter.ke/court-detains-contractor-over-ksh-57-5-million-fraud-at-ncba-bank/)
5. [Business Daily — Software developer held on Sh58m NCBA fraud claim](https://www.businessdailyafrica.com/bd/corporate/companies/software-developer-held-on-sh58m-ncba-fraud-claim-5086392)
6. [PayPal — Machine Learning Fraud Detection Technologies](https://www.paypal.com/us/brc/article/payment-fraud-detection-machine-learning)
7. [PayPal Technology Blog — Deploying Large-scale Fraud Detection ML Models](https://medium.com/paypal-tech/machine-learning-model-ci-cd-and-shadow-platform-8c4f44998c78)
8. [Stripe — Radar: AI-powered Fraud Detection](https://stripe.com/radar)
9. [Stripe Docs — Radar risk evaluation](https://docs.stripe.com/radar/risk-evaluation)
10. [Business Insider — Mastercard uses AI to detect credit card fraud](https://www.businessinsider.com/mastercard-ai-credit-card-fraud-detection-protects-consumers-2025-5)
11. [Mastercard — Decision Intelligence for Fraud and Risk Management](https://www.mastercard.com/global/en/business/cybersecurity-fraud-prevention/risk-decisioning/decision-intelligence.html)
12. [PR Newswire — Danske Bank and Teradata implement AI engine that monitors fraud in real time](https://www.prnewswire.com/news-releases/danske-bank-and-teradata-implement-artificial-intelligence-ai-engine-that-monitors-fraud-in-real-time-300540944.html)
13. [Teradata — Danske Bank fights fraud with deep learning and AI (case study)](https://assets.teradata.com/resourceCenter/downloads/CaseStudies/CaseStudy_EB9821_Danske_Bank_Saves_Millions_Fighting_Fraud_With_Deep_Learning_and_AI.pdf)
14. [O'Reilly — Danske Bank Case Study Details](https://www.oreilly.com/library/view/achieving-real-business/9781492038214/ch06.html)
