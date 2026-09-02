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
| Aug 29 (Sat) | flutterwave-fraud-anatomy | Fintech | Flutterwave 4 incidents / 14 months (Feb 2023 ₦2.9B → Apr 2024 ₦11B) | ✅ published |
| Aug 30 (Sun) | *(empty)* | Fintech Security | proposed: playbook — KYC/AML automation |
| Aug 31 (Mon) | anomaly-detection-reconciliation | Analytics | anomaly detection for reconciliation at scale (NCBA EOW catch + Flutterwave ₦11B threshold evasion + Danske/Teradata FPs) | ✅ published |

**Publishing note (Aug 29):** `.scheduled/` was still empty at the Aug 29 cron run
(14:05 EAT). Per gap-fill rule, `flutterwave-fraud-anatomy` was written directly to
`_posts/` with date `2026-08-29 00:00:00 +0300`, cover
`assets/img/cover-flutterwave-fraud-anatomy.webp` (SVG source in `assets/blog/`).
Facts verified against TechCrunch (Mar 5 2023), TechCabal (Mar 10 2023, May 16 2024),
Techpoint Africa, Techloy — full anchor expanded in the event library below.
Note: Aug 27 (Thu) and Aug 28 (Fri) remain UNPUBLISHED (no files were staged for
those days); the next gap-fill session should backfill them if daily continuity
matters. Future sessions: stage `.scheduled/` files again to keep the daily cron
fed — an empty queue silently stops publishing.

**Publishing note (Aug 30):** `.scheduled/` still empty at the Aug 30 cron run.
Per gap-fill rule, today's Sun (Fintech Security) slot was written directly to
`_posts/` with date `2026-08-30 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Aug 30 (Sun) | kyc-aml-automation-playbook | Fintech Security | Playbook: 5-checkpoint KYC/AML automation (Flutterwave trigger-limit bypass ₦11B, NCBA ghost accounts Ksh 57.5M, Group-IB 8,065 KYC injections, Danske/Teradata ML FP reduction) | ✅ published |

**Publishing note (Aug 31):** `.scheduled/` was still empty at the Aug 31 cron run
(14:05 EAT). Per gap-fill rule, today's Mon (Analytics) slot was written directly
to `_posts/` with date `2026-08-31 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Aug 31 (Mon) | anomaly-detection-reconciliation | Analytics | Anomaly detection at scale: NCBA caught only by EOW reconciliation (70 ghost accts, 260 txs, Ksh 57.5M, Jun 6-14 2025) + Flutterwave Apr 2024 ₦11B kept below trigger limits + Danske/Teradata 1,200 FPs/day → 50% cut. Verified Python demo (seed 42): z-score catches spike (z≈44.7) but 0/12 distributed days; Isolation Forest on entity-day aggregates catches 12/12 + spike (115/11,462 flags, ~0.9% FP) | ✅ published |

Cover: `assets/img/cover-anomaly-detection-reconciliation.webp` (SVG source
`assets/blog/cover-anomaly-detection-reconciliation.svg`, radar/sonar lock-on
metaphor — distinct from the Aug 15 "two ledgers" reconciliation cover). All
code blocks executed and outputs verified (scikit-learn, seed 42). Facts
verified: NCBA event library, TechCabal (May 16 2024) for Flutterwave,
Teradata case study + Fintech Futures for Danske (1,200 FPs/day, 99.5% not
fraud, 50% FP reduction).

**Still UNPUBLISHED: Aug 27 (Thu, ML) and Aug 28 (Fri, Cybersecurity)** — no
files were ever staged for those days. `.scheduled/` remains empty: the daily
cron will silently report "Nothing to do" tomorrow (Sep 1) unless files are
staged. **Next session actions:** (1) backfill Aug 27 + Aug 28 directly to
`_posts/` if daily continuity matters (proposed anchors: Aug 27 = MLOps/regtech
model governance exists → use ML model monitoring/drift instead; Aug 28 =
secrets in CI / credential leaks — tj-actions, GitHub push protection); (2)
stage `.scheduled/2026-09-01-*.md` … Sep 7 queue per Week 3 table (Mon Sep 1 =
CBK bank-fraud statistics → fraud-trend analytics; **never stage Tue Sep 2** —
Tuesday AI Update cron owns it); (3) confirm `tuesday-ai-update` cron remains
active for Sep 2.

Cover: `assets/img/cover-kyc-aml-automation-playbook.webp` (SVG source
`assets/blog/cover-kyc-aml-automation-playbook.svg`, "checkpoint conveyor"
metaphor). Facts verified against the event library + incident bank (2+ sources
each). **Still UNPUBLISHED: Aug 27 (Thu), Aug 28 (Fri), Aug 31 (Mon)** — Aug 31
(Mon, Analytics: "anomaly detection for reconciliation at scale") has no staged
file either. `.scheduled/` remains empty: the daily cron will silently report
"Nothing to do" tomorrow unless files are staged. Next session: backfill Aug 27/28
and stage `.scheduled/2026-08-31-*.md` + Sep 1–7 queue.

**Publishing note (Sep 1):** `.scheduled/` still empty at the Sep 1 cron run
(14:05 EAT). Sep 1 is a **Tuesday** — the `tuesday-ai-update` cron (14:00 EAT)
published `2026-09-01-tuesday-ai-update.md` today, so today is covered. Per the
Aug 31 note's action list, **Aug 28 (Fri, Cybersecurity) was backfilled** directly
to `_posts/` with date `2026-08-28 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Aug 28 (Fri) | secrets-in-ci-credential-leaks | Cybersecurity | Secrets in CI: CircleCI Dec 2022/Jan 2023 (malware → session-cookie theft → customer env vars/tokens/keys exfiltrated; "rotate all secrets"), Mercedes-Benz PAT in public repo (Sep 29 2023, unrestricted access to GitHub Enterprise; DB strings/cloud keys/SSO passwords), Toyota 2022 (5-yr exposure), CISA "Private-CISA" repo May 2026 (GitHub GovCloud keys, PATs, plaintext passwords; secret-scanning-disable guide), GitHub 39M secrets leaked 2024, push protection GH013 block on THIS repo (rebase+redact+force-push) | ✅ published |

Cover: `assets/img/cover-secrets-in-ci-credential-leaks.webp` (SVG source
`assets/blog/cover-secrets-in-ci-credential-leaks.svg`, "leaking pipeline"
metaphor: golden key falls through a crack in the BUILD stage into an EXPOSED
pool, green PUSH PROTECTION shield on the right — distinct from the Aug 21
pipeline-gate and Jun 12 vault-door covers). Facts verified: circleci.com
incident report + Malwarebytes; BleepingComputer + RedHunt Labs (Mercedes);
Krebs on Security + Dark Reading + The Register (CISA); GitHub Blog +
SecurityWeek + BleepingComputer (39M stat). 1,012 body words.

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only remaining gap. The calendar's
original Aug 27 topic (mlops-regtech-model-governance) was consumed by the
date-fixed publish on Aug 25, so the backfill needs a fresh ML topic (e.g.
model drift/monitoring for fraud models — PSI, data quality gates). `.scheduled/`
remains empty — the daily cron will silently report "Nothing to do" tomorrow
(Sep 2) unless files are staged.

**⚠️ Week 3 weekday correction:** the table below was written assuming Sep 1 =
Monday, but **Sep 1, 2026 is a Tuesday**. Correct rotation for the staging queue:
Wed Sep 2 = AI Security, Thu Sep 3 = ML, Fri Sep 4 = Cybersecurity, Sat Sep 5 =
Fintech, Sun Sep 6 = Fintech Security, Mon Sep 7 = Analytics. Never stage
Tuesdays (Sep 8/15/22/29) — the Tuesday AI Update cron owns them.

**Next session actions:** (1) backfill Aug 27 (Thu, ML — fresh topic, NOT
mlops-regtech-model-governance which is published Aug 25); (2) stage
`.scheduled/` files for Sep 2 (Wed, AI Security) through Sep 7 (Mon, Analytics)
per the corrected rotation above — an empty queue silently stops the daily
cron; (3) keep `tuesday-ai-update` cron active (it owns all Tuesdays).

**Publishing note (Sep 2):** `.scheduled/` was still empty at the Sep 2 cron run
(14:05 EAT). Per gap-fill rule, today's Wed (AI Security) slot was written
directly to `_posts/` with date `2026-09-02 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 2 (Wed) | ai-red-teaming-financial-llm-apps | AI Security | Red-teaming financial LLM apps: Blue41/Bunq €0.02 SEPA-memo indirect prompt injection (Apr 2026; DD identified Bunq, 20M+ customers), Unit 42 in-the-wild IDPI incl. unauthorized-transaction intents (Mar 2026), Morris II RAG email-assistant worm (2024), WithSecure refund-bot refusal bypass; OWASP LLM Top 10 2025 checklist + runnable naive-vs-tagged demo | ✅ published |

Cover: `assets/img/cover-ai-red-teaming-financial-llm-apps.webp` (SVG source
`assets/blog/cover-ai-red-teaming-financial-llm-apps.svg`, "poisoned SEPA memo
→ LLM → human-approval gate" metaphor: paper transfer slip with red payload
pill + red-team probe reticle, money path diverging to a blocked `refund_tx`
tool call — distinct from the Jun 8 shield+4-arrows cover and the Aug 24
chat-UI cover). Facts verified: blue41.com case study + Developers Digest
(Bunq attribution, attack chain); unit42.paloaltonetworks.com (in-the-wild
intents, 22 techniques); arXiv:2403.02817 + IBM Think (Morris II); WithSecure
Labs publications page. Code block executed — output verified
(naive EXECUTED vs tagged BLOCKED; base64/wordmix evade marker filters).
1,360 prose words (1,580 with code).

**⚠️ Week 4 weekday correction:** the Week 4 table below lists "Sep 8 (Mon)" —
Sep 8, 2026 is actually a **TUESDAY**, owned by the AI Update cron. Shift the
Week 4 rotation by one day: Tue Sep 8 = AI Update, Wed Sep 9 = AI Security
(LLM data-exfiltration via indirect prompt injection), Thu Sep 10 = ML,
Fri Sep 11 = Cybersecurity, Sat Sep 12 = Fintech, Sun Sep 13 = Fintech
Security, Mon Sep 14 = Analytics.

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only calendar gap. `.scheduled/`
remains empty: the daily cron will silently report "Nothing to do" tomorrow
(Sep 3) unless files are staged. **Next session actions:** (1) backfill Aug 27
(Thu, ML — fresh topic, NOT mlops-regtech-model-governance, published Aug 25);
(2) stage `.scheduled/` files for Sep 3 (Thu, ML: model drift/monitoring PSI),
Sep 4 (Fri, Cybersecurity: M-PESA/Daraja API security), Sep 5 (Sat, Fintech:
Sidian Bank deep-dive), Sep 6 (Sun, Fintech Security: third-party API/BaaS
playbook), Sep 7 (Mon, Analytics: CBK fraud statistics) per the Week 3 table;
(3) never stage Tue Sep 8 (AI Update cron owns it) and apply the Week 4
correction above when staging Sep 9+; (4) keep `tuesday-ai-update` cron active.

## Week 3 — Proposed (Sep 2–7, corrected weekdays)

| Date | Theme | Proposed topic |
|------|-------|----------------|
| Sep 2 (Wed) | AI Security | AI red-teaming for financial LLM apps — ✅ published as `ai-red-teaming-financial-llm-apps` (see note above) |
| Sep 3 (Thu) | ML | Model drift & monitoring for fraud models (PSI, data quality) |
| Sep 4 (Fri) | Cybersecurity | Mobile money API security: M-PESA/Daraja integration pitfalls |
| Sep 5 (Sat) | Fintech | Sidian Bank 2025 incident (verified reporting) deep-dive |
| Sep 6 (Sun) | Fintech Security | Playbook: third-party API & BaaS integration security |
| Sep 7 (Mon) | Analytics | CBK bank-fraud statistics → analytics of fraud trends |

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
- **Flutterwave (Feb 2023 – Apr 2024)** — 4 unauthorized-transfer incidents in 14 months. Feb 2023: ₦2.9B (~$4.2M) moved in 63 txs across 28 accounts (TechCrunch), spread to 107 accounts in 27 banks (court petition dated Feb 20, 2023, TechCabal); hundreds of accounts frozen; merchant-key/social-engineering theory; Flutterwave denied hack. Mar 2023: ₦550M to ~107 accounts in 27 banks (court docs). Oct 2023: ₦19B (~$24M) via unauthorized POS-merchant txs, ~6,000 holders across 35 banks; court order ~Mar 2024 to recover $24M. Apr 2024: ₦11B ($7M; insider says ≥₦20B/$13.5M) to 5 institutions over 4 days, undetected because deposits kept below fraud-check trigger limits; Mareva injunction Feb 2024; no customer funds lost per Flutterwave. Kenya: Jul 2022 ARA froze KSh 6.2B (~$52.5M) in 62 accounts (money-laundering allegations); allegations withdrawn Feb 2023. Sources: TechCrunch (Mar 5 2023), TechCabal (Mar 10 2023; May 16 2024), Techpoint Africa (Mar 5 2023; Feb 6 2023), TechCabal (Aug 30 2022), Techloy.
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
