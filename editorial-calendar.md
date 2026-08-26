# ML Kenya Blog — Editorial Calendar

**Purpose:** daily posts on **ml.co.ke** rotating through five themes — **Fintech, AI, Cybersecurity, ML, Analytics** — anchored on **reports of real incidents in actual systems**, each ending with "how we can do better." This converts news into durable engineering lessons (the blog's "news-to-ML conversion" pattern).

**Publishing mechanics:**
- Daily cron `blog-poster` (14:05 EAT) publishes one file per day from `.scheduled/YYYY-MM-DD-slug.md` (filename date = publish date).
- **Tuesdays are reserved for the Global AI Roundup** — generated automatically by the `tuesday-ai-update` cron (14:00 EAT, resumed Aug 2026). Never stage a post for a Tuesday.
- Gap-fill rule: if a day is missed, write directly to `_posts/` with the actual past date (see blog-drafting skill, Pitfall #11).
- Every post: front matter `image.path` → `/assets/img/cover-<slug>.webp` (SVG source in `assets/blog/`), no `{% post_url %}` tags, plain markdown links, verified incidents with 2+ sources.

**Theme rotation (weekly):**

| Day | Theme | Default anchor |
|-----|-------|----------------|
| Mon | Analytics / Data Science | reconciliation & fraud analytics |
| Tue | AI Update (cron) | global roundup, 7 regions |
| Wed | AI / AI Security | LLM security, AI-enabled defense |
| Thu | ML | fraud ML, model governance, MLOps |
| Fri | Cybersecurity | CI/CD security, insider threat, vendor risk |
| Sat | Fintech | incident deep-dive (like NCBA) |
| Sun | Fintech Security | prevention playbook / best practices |

---

## Week 1 — Catch-up series (published Aug 2026) ✅

| Date | Slug | Theme | Anchor |
|------|------|-------|--------|
| Aug 15 | reconciliation-analytics-fintech | Analytics | NCBA reconciliation + Flutterwave 2023 |
| Aug 16 | insider-threat-privileged-access-fintech | Cybersecurity | NCBA 3-min insider |
| Aug 17 | vendor-risk-fintech-contractors | Fintech | NCBA/Ronford + SolarWinds + 3CX |
| Aug 18 | tuesday-ai-update | AI Update | Global roundup |
| Aug 19 | ai-enabled-security-anomaly-detection | AI Security | NCBA counterfactual + PayPal/Stripe/Mastercard |
| Aug 20 | fraud-ml-mobile-money | ML | NCBA 70-account loophole |
| Aug 21 | cicd-security-control-fintech | Cybersecurity | NCBA + SolarWinds + tj-actions |
| Aug 22 | ncba-ghost-account-fraud | Fintech | NCBA incident anatomy |
| Aug 23 | ai-automation-fintech-security-playbook | Fintech Security | Prevention playbook |

## Week 2 — Staged queue (`.scheduled/`)

| Date | Slug | Theme | Anchor / angle |
|------|------|-------|----------------|
| Aug 24 (Mon) | kyc-aml-analytics-african-fintech | Analytics | Digital-lending data scandals (Kenya 2023-24) → KYC/AML analytics pipelines |
| Aug 25 (Tue) | *(tuesday-ai-update cron)* | AI Update | — |
| Aug 26 (Wed) | llm-security-financial-chatbots | AI Security | OWASP LLM Top 10 + financial chatbot incidents → prompt-injection defenses |
| Aug 27 (Thu) | mlops-regtech-model-governance | ML | CBK/ODPC governance + MLOps model governance for fintech |
| Aug 28 (Fri) | *(empty — fill next session)* | Cybersecurity | proposed: secrets in CI / credential leaks (tj-actions, GitHub push-protection) |

**Publishing note (Aug 26):** the staged Week-2 queue was consumed one day early by the
catch-up cron (date-fixed publishes: `kyc-aml-analytics-african-fintech` → Aug 23,
`llm-security-financial-chatbots` → Aug 24, `mlops-regtech-model-governance` → Aug 25).
`.scheduled/` is now empty. Gap-fill on Aug 26:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Aug 26 (Wed) | deepfake-fraud-financial-services | AI Security | Arup HK$200M deepfake CFO + UK voice-clone $243K + Group-IB 8,065 KYC injections + Sumsub Kenya 10% | ✅ published |
| Aug 29 (Sat) | *(empty)* | Fintech | proposed: Flutterwave 2023 deep-dive |
| Aug 30 (Sun) | *(empty)* | Fintech Security | proposed: playbook — KYC/AML automation |
| Aug 31 (Mon) | *(empty)* | Analytics | proposed: anomaly detection for reconciliation at scale |

## Week 3 — Proposed (Sep 1–7)

| Date | Theme | Proposed topic |
|------|-------|----------------|
| Sep 1 (Mon) | Analytics | CBK bank-fraud statistics → analytics of fraud trends |
| Sep 2 (Tue) | AI Update | Global AI Roundup |
| Sep 3 (Wed) | AI Security | AI red-teaming for financial LLM apps (OWASP LLM Top 10 walkthrough) |
| Sep 4 (Thu) | ML | Model drift & monitoring for fraud models (PSI, data quality) |
| Sep 5 (Fri) | Cybersecurity | Mobile money API security: M-PESA/Daraja integration pitfalls |
| Sep 6 (Sat) | Fintech | Sidian Bank 2025 incident (verified reporting) deep-dive |
| Sep 7 (Sun) | Fintech Security | Playbook: third-party API & BaaS integration security |

## Week 4 — Proposed (Sep 8–14)

| Date | Theme | Proposed topic |
|------|-------|----------------|
| Sep 8 (Mon) | Analytics | Control totals & break detection in settlement systems |
| Sep 9 (Tue) | AI Update | Global AI Roundup |
| Sep 10 (Wed) | AI Security | LLM data-exfiltration via indirect prompt injection |
| Sep 11 (Thu) | ML | Graph ML for fraud rings (transaction graph clustering) |
| Sep 12 (Fri) | Cybersecurity | Privileged access management: JIT/PAM for fintech |
| Sep 13 (Sat) | Fintech | Global fintech outage post-mortems (e.g. major card outages) |
| Sep 14 (Sun) | Fintech Security | Playbook: real-time reconciliation + alerting |

---

## Event library (verified anchors for future posts)

- **NCBA Bank Rwanda (Jun 2025)** — contractor abuse of live backend access; 70 ghost accounts, 260 txs, Ksh 57.5M / USD 446k; caught by EOW reconciliation. Sources: kenyainsights.com, 254news.co.ke, nairobitimez.co.ke, courthelicopter.ke, businessdailyafrica.com.
- **Flutterwave (Feb-Mar 2023)** — ~₦2.9B moved via 63 txs / 28 accounts; police report + court freeze orders. Sources: TechCrunch, TechCabal, Techpoint Africa.
- **SolarWinds (2020)** — Orion build pipeline compromise; ~18,000 orgs. Source: CISA AA20-352A.
- **3CX (2023)** — trojanized Desktop App updates. Sources: CISA alert (Mar 30, 2023), Mandiant.
- **tj-actions/changed-files (2025)** — CVE-2025-30066; retroactive tag rewrite exposed CI/CD secrets. Sources: GitHub Advisory GHSA-mrrh-fwg8-r2c3, CISA (Mar 18, 2025), Wiz.
- **Sidian Bank (Oct 2025)** — MKU student charged with Sh7.8M theft. Sources: kenyainsights.com, tuko.co.ke.
- **Danske Bank + Teradata fraud ML** — rules ~40% detection/1,200 FPs per day → ML cut FPs ~50%, raised detection ~60%. Sources: Teradata case study, Fintech Futures.
- **PayPal / Stripe Radar / Mastercard Decision Intelligence** — production AI fraud engines (verify current figures before reuse).

## Rules of the road

1. Every post must cite 2+ verifiable sources per factual claim; drop unverifiable claims.
2. Every post ends with actionable "how we can do better" content (controls, code, checklists).
3. Covers are unique per post (see blog-drafting skill cover-metaphor-library); never reuse a metaphor.
4. Fill empty calendar slots by batch-writing to `.scheduled/` (max 3 subagents parallel; verify post_url/cover/webp after).
5. Re-check the event library before reuse — reporting may have evolved (e.g. court outcomes).
