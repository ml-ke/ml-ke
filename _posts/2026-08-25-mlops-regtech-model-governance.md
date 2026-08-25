---
title: "MLOps for RegTech: Model Governance Under CBK, ODPC and EU AI Act Rules"
date: 2026-08-25 00:00:00 +0300
categories: [ML Ops, Fintech]
tags: [model governance, MLOps, regulatory compliance, credit scoring, explainability, data protection]
image:
  path: /assets/img/cover-mlops-regtech-model-governance.webp
  alt: MLOps model governance pipeline for regulated fintech
---

## Introduction

Kenya's digital lending boom produced hundreds of app-based lenders — and a regulatory reckoning. In March 2022, the Central Bank of Kenya (CBK) gazetted the Digital Credit Providers (DCP) Regulations, bringing previously unregulated lenders under licensing and oversight ([CBK](https://www.centralbank.go.ke/2022/03/21/central-bank-of-kenya-digital-credit-providers-regulations-2022/)). The Office of the Data Protection Commissioner (ODPC) is issuing real fines, and the EU AI Act now classifies credit scoring as high-risk. If your ML models decide who gets credit, model governance is no longer a data-science slide — it is what regulators will audit. This post maps all three regimes to the MLOps controls that keep you compliant.

## The regulatory map: CBK, ODPC and the EU AI Act

**Kenya — CBK DCP Regulations 2022.** Digital lenders must obtain a CBK licence, and the Regulations impose conduct duties that touch ML systems directly: a customer complaints redress mechanism, reliable and secure systems protecting the confidentiality and integrity of customer information, and a ban on unrestricted sharing of customer data without consent ([MMAN Advocates](https://mman.co.ke/content/central-bank-kenya-digital-credit-providers-regulations-2022), [Business Daily](https://www.businessdailyafrica.com/bd/opinion-analysis/columnists/cbk-tigthens-noose-in-new-rules-targeting-digital-lenders-5159518)). Licensed DCPs must register as data controllers with the ODPC ([Techpression](https://techpression.com/kenya-tightens-digital-lending-regulation-licenses-41-new-providers/)).

**Kenya — ODPC enforcement is real.** Under the Data Protection Act, the ODPC can fine up to KSh 5 million or 1% of annual turnover ([Digital Banking News](https://digitalbankingnews.co.ke/data-protection-rules-for-loan-apps-in-kenya/)). In 2023 it fined Mulla Pride Ltd (KeCredit, FairKash) about KSh 2.9 million for processing personal data without consent and using third-party data to harass defaulters ([TechCabal](https://techcabal.com/2023/09/26/digital-lenders-fined-in-kenya/)); the High Court later dismissed the lender's challenge ([Kenyan Wall Street](https://kenyanwallstreet.com/court-dismisses-digital-lenders-suit-over-ksh-2-98mn-data-protection-fine/)). Mulla Pride was unlicensed at the time — non-compliance compounds ([Mukamba Law](https://mukambalaw.com/digital-lenders/)).

**EU — the AI Act makes credit scoring high-risk.** Regulation (EU) 2024/1689 classifies AI systems that evaluate natural persons' creditworthiness or establish credit scores as high-risk via Article 6(2) and Annex III, point 5(b) ([EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), [Annex III](https://artificialintelligenceact.eu/annex/3/)). It triggers the Articles 8–17 duties: risk management, data governance, technical documentation, logging and human oversight ([RegulatoryAI](https://www.regulatoryai.eu/ai-creditworthiness/)). If you serve EU consumers, your Nairobi-built scorecard is in scope.

**Why it matters: Apple Card, 2019.** After complaints that the Apple Card algorithm gave women lower credit limits, New York's financial regulator opened an investigation ([NYT](https://www.nytimes.com/2019/11/10/business/Apple-credit-card-investigation.html), [BBC](https://www.bbc.com/news/business-50365609)). No verdict was needed: an unexplained, ungoverned model is a liability ([The Guardian](https://www.theguardian.com/technology/2019/nov/10/apple-card-issuer-investigated-after-claims-of-sexist-credit-checks)).

## The MLOps governance stack

All three regimes converge on the same operational demands. Here is the stack.

**Model inventory and risk tiering.** Every model — scorecard, collection-ranking — registered in a central inventory with owner, data dependencies and a risk tier. Tiering mirrors the AI Act's Annex III logic and tells a CBK examiner you know what you run.

**Documentation: model cards plus technical documentation.** The AI Act (Article 11) requires technical documentation; regulators expect the practical version — model cards covering intended use, training data, performance by segment and known limitations. Write them at build time, not audit time.

**Validation before deployment.** Backtest on out-of-sample windows, run challenger models, and compute fairness metrics (disparate impact, equalized odds) across gender, age and region before sign-off. The Apple Card episode is the canonical case of skipping this step; the AI Act's Article 10 data governance duty demands the same examination.

**Monitoring, drift and audit trails.** Track population and feature drift in production, and log every prediction with model version and input snapshot. The AI Act's logging (Article 12) and post-market monitoring (Article 72) are, practically, "a feature store plus a decent audit table"; ODPC's accountability principle wants the same evidence.

**Human oversight and override.** Article 14 requires that a human can interpret outputs, override decisions and suspend the system. For credit: an override path when a model declines someone a human disagrees with, and a record of every override.

**Explainability for adverse-action notices.** When you decline a loan, the customer can ask why — CBK's complaints mechanism and ODPC's fairness requirements make that unavoidable. Serve reason codes (SHAP-based or monotonic scorecard attributes) from the same pipeline that produced the decision. If a model can't explain itself, it shouldn't be in production.

**Versioned data and reproducible pipelines.** DVC, feature stores and locked dataset snapshots make every experiment reproducible and give regulators clean answers to "what data, what version, what code." Data minimization for ODPC and data governance for the AI Act both start here.

**Approval workflows for model promotion.** Treat promotion like CI/CD for ML: automated tests, then human approval gates with named approvers before staging and production. A signed-off promotion trail is the first artifact any examiner asks for.

**Continuous compliance.** Automate reporting — inventory changes, incident logs, drift alerts — into scheduled summaries for risk committees and regulators, instead of scrambling at inspection time.

> **Governance is evidence, not process**
> Regulators rarely ask whether your model is fair; they ask what you measured, when, and who signed off. Every control above exists to produce that evidence.
{: .prompt-info }

## Conclusion

CBK licensing, ODPC fines and the EU AI Act are three rulebooks saying the same thing: ML credit decisions must be documented, validated, monitored and explainable. The good news: the compliance stack is just disciplined MLOps — inventory, documentation, validation, monitoring, human oversight, reproducibility and approval gates. Build it in now, and the next regulatory inquiry becomes a demo instead of a fire drill.

## References

- CBK — [Central Bank of Kenya (Digital Credit Providers) Regulations 2022](https://www.centralbank.go.ke/2022/03/21/central-bank-of-kenya-digital-credit-providers-regulations-2022/)
- [MMAN Advocates — DCP Regulations 2022 summary](https://mman.co.ke/content/central-bank-kenya-digital-credit-providers-regulations-2022)
- [Business Daily — CBK data-sharing rules](https://www.businessdailyafrica.com/bd/opinion-analysis/columnists/cbk-tigthens-noose-in-new-rules-targeting-digital-lenders-5159518)
- [Techpression — DCPs must register as data controllers](https://techpression.com/kenya-tightens-digital-lending-regulation-licenses-41-new-providers/)
- [Digital Banking News — ODPC fines (KSh 5M or 1% turnover)](https://digitalbankingnews.co.ke/data-protection-rules-for-loan-apps-in-kenya/)
- [TechCabal — Mulla Pride fined KES 2.9M](https://techcabal.com/2023/09/26/digital-lenders-fined-in-kenya/)
- [Kenyan Wall Street — court dismisses suit over KSh 2.98M fine](https://kenyanwallstreet.com/court-dismisses-digital-lenders-suit-over-ksh-2-98mn-data-protection-fine/)
- [Mukamba Law — Mulla Pride unlicensed when fined](https://mukambalaw.com/digital-lenders/)
- [EUR-Lex — Regulation (EU) 2024/1689 (AI Act)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- [EU AI Act Annex III — credit scoring high-risk](https://artificialintelligenceact.eu/annex/3/)
- [RegulatoryAI — creditworthiness under the AI Act](https://www.regulatoryai.eu/ai-creditworthiness/)
- [NYT — Apple Card gender discrimination investigation](https://www.nytimes.com/2019/11/10/business/Apple-credit-card-investigation.html)
- [BBC — Apple's 'sexist' credit card investigated](https://www.bbc.com/news/business-50365609)
- [The Guardian — Apple Card issuer investigated](https://www.theguardian.com/technology/2019/nov/10/apple-card-issuer-investigated-after-claims-of-sexist-credit-checks)
