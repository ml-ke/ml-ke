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

**Publishing note (Sep 3):** `.scheduled/` was still empty at the Sep 3 cron run
(14:05 EAT). Per gap-fill rule, today's Thu (ML) slot was written directly to
`_posts/` with date `2026-09-03 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 3 (Thu) | fraud-model-drift-monitoring | ML | Fraud models rot quietly: PSI on the score distribution (OK <0.10 / WATCH 0.10–0.25 / RETRAIN >0.25, credit-scorecard convention), monthly score-creep demo (seed 42: Jan PSI 0.000 → Apr 0.121 WATCH → May 0.344 RETRAIN → Jun 1.223; 50–70 band ~9% → ~44% of live population), feature fill-rate gates (device_id 99.4% → 93.1% FAIL while score stays in-band) — anchored to Flutterwave Apr 2024 ₦11B sub-trigger adaptation and NCBA ghost accounts as *adversarial population drift*, NOT per-line anomalies (differentiated from Aug 31 anomaly post and Jul 11 ml-monitoring post) | ✅ published |

Cover: `assets/img/cover-fraud-model-drift-monitoring.webp` (SVG source
`assets/blog/cover-fraud-model-drift-monitoring.svg`, "population creep"
metaphor: dashed cyan EXPECTED (TRAIN) curve + red ghost curves creeping right
month-over-month into a LIVE (JUN) curve, PSI zone ruler OK/WATCH/RETRAIN with
needle at PSI 1.22, FILL-RATE GATE panel with device_id 93% ✗ GATE: FAIL —
distinct from #30 ml-monitoring dashboard, #33 anomaly radar, #29 CI/CD
pipeline). Facts verified: TechCabal + Techpoint + Business Insider Africa
(Flutterwave ₦11B, 5 institutions, 4 days, sub-trigger amounts); NCBA event
library (70 accounts, 260 txs, Ksh 57.5M, 8 days); Teradata/Fintech Futures
(Danske 1,200 FPs/day, 99.5% not fraud); PSI threshold convention via Fiddler
AI, Coralogix, and Yildirim/ResearchGate. Code executed — output quoted
verbatim (PSI zone table + Jun band contributions + fill-rate gate). 1,414
prose words (excluding code).

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only remaining calendar gap.
`.scheduled/` remains empty: the daily cron will silently report "Nothing to
do" tomorrow (Sep 4) unless files are staged. **Next session actions:**
(1) backfill Aug 27 (Thu, ML — fresh topic, NOT mlops-regtech-model-governance,
published Aug 25); (2) stage `.scheduled/` files for Sep 4 (Fri, Cybersecurity:
M-PESA/Daraja API security), Sep 5 (Sat, Fintech: Sidian Bank deep-dive),
Sep 6 (Sun, Fintech Security: third-party API/BaaS playbook), Sep 7 (Mon,
Analytics: CBK fraud statistics) per the Week 3 table — or gap-fill each day
directly as this session did; (3) never stage Tue Sep 8 (AI Update cron owns
it) and apply the Week 4 correction when staging Sep 9+; (4) keep
`tuesday-ai-update` cron active.

**Publishing note (Sep 4):** `.scheduled/` was still empty at the Sep 4 cron run
(14:05 EAT). Per gap-fill rule, today's Fri (Cybersecurity) slot was written
directly to `_posts/` with date `2026-09-04 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 4 (Fri) | mpesa-daraja-api-pitfalls | Cybersecurity | Mobile money API security: 4 "open joints" of Daraja integration — (1) leaked consumer key/secret (Quest: leaked creds "can be exploited to initiate unauthorized transactions"; GitHub 39M secrets 2024; r/nairobitechies rotation-didn't-help case), (2) trusted/insider access (Safaricom v. EADH KES 20.3M 2016-era aggregation suit; Constitutional Petition E095/2026 — High Court KES 9.9M ruling, rogue-employee defense rejected), (3) spoofed STK pushes + fake confirmations (unsolicited BETGR8_CS push; The Star scheme explainer, verify via 456), (4) callback/timestamp/sandbox-prod hygiene — verified Python demo (STK password prefix + 1-min replay window; forged ResultCode 0 callback REJECTED vs genuine FULFIL) | ✅ published |

Cover: `assets/img/cover-mpesa-daraja-api-pitfalls.webp` (SVG source
`assets/blog/cover-mpesa-daraja-api-pitfalls.svg`, "payment rail with numbered
open joints" metaphor: merchant server ↔ Daraja gateway on a dashed cyan rail,
red attack arrows at joints 1/2/3 = LEAKED SECRET / FAKE CALLBACK / SPOOFED
PUSH, green shield on the DARAJA side — distinct from the Aug 28 CI-pipeline
leak cover and the Jun ml-secrets vault cover). Facts verified: tech-ish.com
(Feb 6 2020, reporting Business Daily) for EADH; Techweez (May 18 2026) +
Nairobi Wire (Apr 22 2026) for E095/2026; Quest Web guide; The Star (Mar 27
2025); Reddit threads cited only at snippet level (bot-walled); Koda School +
KenZobe for Daraja mechanics. Code block executed — output quoted verbatim.
1,456 prose words (excluding code).

**Still UNPUBLISHED: Aug 27 (Thu, ML) and Sep 5 (Sat, Fintech: Sidian Bank
2025 incident deep-dive)** — Sep 6 was covered by today's gap-fill (note
below); `.scheduled/` remains empty, so the daily cron will silently report
"Nothing to do" tomorrow (Sep 7) unless files are staged. **Next session
actions:** (1) backfill Sep 5 (Sidian Bank — verified anchors already in the
event library: MKU student charged with Sh7.8M theft, kenyainsights + tuko)
and Aug 27 (fresh ML topic, NOT mlops-regtech-model-governance, published Aug
25) directly to `_posts/` if daily continuity matters; (2) stage or gap-fill
Sep 7 (Mon, Analytics: CBK bank-fraud statistics); (3) never stage Tue Sep 8
(AI Update cron owns it) and apply the Week 4 weekday correction when staging
Sep 9+; (4) keep `tuesday-ai-update` cron active.

**Publishing note (Sep 6):** `.scheduled/` was still empty at the Sep 6 cron run
(14:05 EAT). Per gap-fill rule, today's Sun (Fintech Security) slot was written
directly to `_posts/` with date `2026-09-06 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 6 (Sun) | aggregator-baas-security-playbook | Fintech Security | Playbook: third-party API & BaaS integration security — "the middleman problem." Differentiated from Sep 4 (direct Daraja rail joints): the layer ABOVE the rail. Cases: Evolve Bank & Trust LockBit breach (Jul 2024: 7,640,112 notified per Maine AG filing; phishing-click entry Feb 9 2024, detected May 29 2024 — ~4 months dwell; Affirm/Wise/Bilt customers impacted; BleepingComputer + TechCrunch), Juspay processor breach (Aug 18 2020, old unrecycled AWS access key; 3.5 crore records + portion of 10-crore user metadata w/ plaintext emails; Business Today + CPO Magazine), Synapse middleware collapse (Ch.11 Apr 22 2024; trustee McWilliams: $265M balances vs $180M held = $85M shortfall; 100k+ customers locked out; CNBC + CFPB + Fortune), Kenya echo = EADH KES 20.3M aggregator suit (cross-link) + CBK Third-Party Agents Guideline 2016 clause 5.1.6 (verbatim: institution "responsible for assessing the adequacy of controls of outsourced activities"). Eight-gate playbook table (layer map → diligence → contract → credential lifecycle → HMAC webhook auth → daily two-ledger reconciliation → kill switch → edge red-teaming) + verified Python demo (HMAC forged-vs-genuine webhook; ledger diff) with verbatim output | ✅ published |

Cover: `assets/img/cover-aggregator-baas-security-playbook.webp` (SVG source
`assets/blog/cover-aggregator-baas-security-playbook.svg`, "opaque middleman"
metaphor: YOUR APP (green, the only audited layer) and BANK/RAIL (faint,
out of reach) on either side of a dashed red AGGREGATOR / BaaS black box
with a red ? and breach arrows labeled phishing click / unrecycled key;
money rail disappears into the box and reappears at the bank; green
VERIFY THE EDGES chip (HMAC) + cyan RECONCILE THE MIDDLE chip (ledger diff)
— distinct from the Sep 4 numbered-rail-joints cover and the Aug 30
checkpoint-conveyor cover). Facts verified at body level: BleepingComputer
(7,640,112, Maine AG filing; phishing-click entry; Feb 9 2024 initial access,
detected May 29 2024; Affirm/Wise/Bilt), TechCrunch (LockBit ransomware),
CNBC (trustee report: $265M vs $180M = $85M shortfall, 100k+ locked out),
CFPB (Ch.11 Apr 22 2024), Fortune (management removed, Mercury suits),
Business Today (Aug 18 2020 detection; 3.5 crore records; masked card data +
plaintext-email user metadata; remediation: key refresh, 2FA, IAM), CPO
Magazine (old unrecycled AWS access key; ~100M records circulating), CBK
guideline PDF clause 5.1.6 (verbatim). Code executed — output quoted
verbatim. 1,471 full-body words (minus code fences).

**Publishing note (Sep 7):** `.scheduled/` was still empty at the Sep 7 cron run
(14:05 EAT). Per gap-fill rule, today's Mon (Analytics) slot was written directly
to `_posts/` with date `2026-09-07 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 7 (Mon) | cbk-fraud-trend-analytics | Analytics | CBK Financial Sector Stability Report 2024 (pub. Sep 2025) Table 14 read as an analytics dataset: reported cyber-fraud cases 173 → 353; exposed KSh 680.9m → 1,963.2m (×2.9); actual loss KSh 412.5m → 1,594.4m (×3.9); recovery only ×1.4 (268.4m → 368.8m). Mix shift: mobile banking = 50.8% of 2024 loss (KSh 810.7m, +344% from 182.4m), card ×16.9 (15.6m → 263.3m on 24 cases), identity ×6.1 (32.6m → 199.1m), online-banking cases 19 → 106 with flat loss. Key derived ratio: loss/exposure conversion 60.6% → 81.2%; identity conversion flipped 14.6% → 97.9% (attempts fell, losses sextupled — control-failure signal) while card fell 96.6% → 59.9% (~40% clawed back = reversal machinery). Severity/case 2.38m → 4.52m (×1.9). Late-night Fri/Sat pattern (BD). Insider half: TechCabal Utawala/Ruiru shadow call-centres + BFIU, KCB fired 34 (25 Kenya), Equity 1,200+ show-cause, Absa blocked 306m/lost 169m, CBK notes AI employee monitoring, NCBA ghost-account cross-link | ✅ published |

Cover: `assets/img/cover-cbk-fraud-trend-analytics.webp` (SVG source
`assets/blog/cover-cbk-fraud-trend-analytics.svg`, "midnight heist clock"
metaphor: clock face near 23:55 with a red arc over the Fri/Sat 23:00–03:00
attack window, next to 2023-vs-2024 loss bars (412 → 1,594) and channel chips
MOBILE 50.8% / CARD ×16.9 / IDENTITY ×6.1 / conversion 61% → 81% — no clock or
dial metaphor exists in the cover library). **Primary-source fact check:** all
Table 14 figures verified directly against the CBK FSSR PDF (the report's prose
paragraph says "153 in 2023" but Table 14 totals 173 → 353 — outlets split on
which to quote; the table's 173 is authoritative for the channel breakdown and
is what the post uses). Coverage corroborated by Business Daily ×2, TechCabal,
TechTrends KE, Money254. Code executed — output quoted verbatim. 1,499
full-body words (minus code fences).

**Still UNPUBLISHED: Aug 27 (Thu, ML) and Sep 5 (Sat, Fintech: Sidian Bank
2025 incident deep-dive)** — `.scheduled/` remains empty, so the daily cron
will silently report "Nothing to do" tomorrow (Sep 8) unless files are staged.
**⚠️ Sep 8 is a TUESDAY — never stage it (AI Update cron owns it).** **Next
session actions:** (1) backfill Sep 5 (Sidian Bank — verified anchors already
in the event library: MKU student charged with Sh7.8M theft, kenyainsights +
tuko) and Aug 27 (fresh ML topic, NOT mlops-regtech-model-governance,
published Aug 25) directly to `_posts/` if daily continuity matters; (2) stage
or gap-fill Sep 9 (Wed, AI Security: LLM data-exfiltration via indirect prompt
injection) onwards per the Week 4 rotation below; (3) keep `tuesday-ai-update`
cron active.

**Publishing note (Sep 8):** Sep 8 is a **Tuesday** — the `tuesday-ai-update` cron (14:00 EAT) published `2026-09-08-tuesday-ai-update.md` today, so today is covered and nothing was staged for the blog-poster. Per the Sep 7 note's action list, **Sep 5 (Sat, Fintech: Sidian Bank) was backfilled** directly to `_posts/` with date `2026-09-05 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 5 (Sat) | sidian-bank-mule-heist | Fintech | Sidian Bank Sh7,882,845 heist (Jan 11 2025) mule anatomy: two charge waves — trio (Nangole/Odidi JKUAT/Ochieng Thika TTI, Aug 25 2025, SPM Onsarigo) + Collins Mutuma (MKU, Oct 27 2025, CM Onyina) — same incident date/amount "jointly with others not before court"; documented dispersal: DTB 471,302 (Karoki victim leg), I&M 458,313, National Bank 451,346 (Kericho Tractor Centre victim), M-Pesa 113,220 + 169,900 onward; named legs = Sh1,494,181 (~19% of pot); vector NOT publicly established (stated honestly); entry vector unproven → focus on first-hop/fan-out/pass-through detection; runnable fan-out heuristic demo (verified output verbatim) + new-beneficiary gap analysis; 1,481 full-body words | ✅ published |

Cover: `assets/img/cover-sidian-bank-mule-heist.webp` (SVG source `assets/blog/cover-sidian-bank-mule-heist.svg`, "one bank, many buckets" metaphor: red siphon leaves SIDIAN BANK building into a FIRST HOP SPLIT box under a green targeting reticle, four colored pipes fanning into DTB/I&M/NATIONAL BANK/M-PESA wallet chips with the charge-sheet amounts, dashed gray ~Sh6.4M "not tied to named accounts" branch — no manifold/split metaphor exists in the cover library; distinct from Aug 22 ghost-account, Aug 29 hub-frozen, Sep 4 numbered-joints, Sep 7 clock covers). Facts verified at body level: tuko.co.ke + Bizna Kenya (charge-sheet quote, counts s.317/s.268(1)+275/POCAMLA 4(a), bail), Kenya Insights (bypassed-multiple-layers, laundering framing), Kahawatungu (count 2: Karoki Sh471,302 to DTB), Nairobi Wire + Education News Kenya (trio legs, bail 300k/1M, Sep 3 mention), Wikipedia (K-Rep→Sidian 2016). Code executed — output quoted verbatim (4 flags; plain bank-to-bank first hops NOT flagged → new-beneficiary rule gap analysis).

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only remaining calendar gap. `.scheduled/` remains empty: the daily cron will silently report "Nothing to do" unless files are staged. **Next session actions:** (1) backfill Aug 27 (Thu, ML — fresh topic, NOT mlops-regtech-model-governance, published Aug 25) directly to `_posts/` if daily continuity matters; (2) stage or gap-fill Sep 9 (Wed, AI Security: LLM data-exfiltration via indirect prompt injection) onwards per the Week 4 rotation below; (3) keep `tuesday-ai-update` cron active (it owns all Tuesdays, incl. Sep 15/22/29).

**Publishing note (Sep 9):** `.scheduled/` was still empty at the Sep 9 cron run
(14:05 EAT). Per gap-fill rule, today's Wed (AI Security) slot was written
directly to `_posts/` with date `2026-09-09 00:00:00 +0300`:

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 9 (Wed) | llm-data-exfiltration-prompt-injection | AI Security | LLM data exfiltration via indirect prompt injection — differentiated from Jun 1 prompt-injection post (Slack AI / CVE-2024-5184 / Copilot cross-repo case studies) and Sep 2 red-teaming: THIS post owns the four exfiltration CHANNELS. (1) LINK a human clicks: Rehberger (wunderwuzzi) M365 Copilot 2024 — email body encoded as Unicode Tag chars (U+E0000–E007F, "ASCII smuggling") inside a benign-looking clickable URL; disclosed HITCON CMT 2024. (2) FETCH that auto-runs: Varonis CoSnitch (CVE-2026-24301) Copilot Personal — undocumented `autorun=1` + `q` executes attacker prompt on page load; queries already-authorized connected apps, exfils via built-in URL fetch to webhook; found via meta-hacking; reported Dec 2025, patched Aug 18 2026; preceded by Reprompt (CVE-2026-24307); parallels Rehberger CVE-2026-24299 (memory writes/deletions). (3) DIAGRAM with hyperlink: Adam Logue M365 Copilot Mermaid "login button" carrying hex-encoded tenant data (blog Oct 21 2025; reported Aug 2025; patched by removing interactive hyperlinks from rendered Mermaid; Register Oct 24 2025; Cursor IDE sibling Aug 2025). (4) MEMORY persistence: CoSnitch finding 3 + CVE-2026-24299 — retrieved pages write standing instructions into memory store. Controls table (render-no-network, two-way tool gating, egress anomaly detection, least privilege, memory write policy, channel red-teaming) + runnable Unicode-tag/beacon triage demo, output quoted verbatim | ✅ published |

Cover: `assets/img/cover-llm-data-exfiltration-prompt-injection.webp` (SVG source `assets/blog/cover-llm-data-exfiltration-prompt-injection.svg`, "four exfil pipes" metaphor: poisoned document with red dashed invisible-payload line → LLM chip → LINK (red) / FETCH (cyan) / DIAGRAM (yellow) pipes carrying byte cubes into an attacker dish, MEMORY (purple) self-loop labelled "persists" — no pipe/beacon metaphor exists in the cover library; distinct from Sep 2 memo-gate and Jun 1 covers). Facts verified at body level: The Hacker News (Aug 18 2026) + Varonis blog + Dark Reading + Cybersecurity News (CoSnitch CVE-2026-24301/24299/24307, autorun=1 + q, meta-hacking, patch date, no in-wild exploitation); embracethered.com primary + Infosecurity Magazine (Rehberger ASCII smuggling); adamlogue.com (Oct 21 2025) + The Register (Oct 24 2025) + CSO Online (Logue Mermaid fix). Code executed via verify-post-code.py — output quoted verbatim. 1,499 full-body words (minus code). New verified anchors mirrored into `llm-agent-red-team-incident-bank.md` (§5–7).

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only remaining calendar gap. `.scheduled/` remains empty: the daily cron will silently report "Nothing to do" unless files are staged. **Next session actions:** (1) backfill Aug 27 (Thu, ML — fresh topic, NOT mlops-regtech-model-governance, published Aug 25) directly to `_posts/` if daily continuity matters; (2) stage or gap-fill the corrected Week 4 rotation for Sep 10 (Thu, ML: graph ML for fraud rings), Sep 11 (Fri, Cybersecurity: PAM/JIT for fintech), Sep 12 (Sat, Fintech: outage post-mortems), Sep 13 (Sun, Fintech Security: real-time reconciliation playbook), Sep 14 (Mon, Analytics: control totals & break detection — the table's "Sep 8 (Mon)" row shifts here); (3) never stage Tuesdays (Sep 15/22/29 — AI Update cron owns them); (4) keep `tuesday-ai-update` cron active.

**Publishing note (Sep 10 & Sep 11 — backfilled 2026-09-11 late evening):** the
`blog-poster` cron (137b7dcf653c) **FAILED on both days with no post written** — the
morning runs died on a provider error (`HTTP 401`, invalid API key, for the ATLAS sync
job on Sep 10; `HTTP 402: Insufficient Balance` for all four LLM-consuming jobs on Sep
10–11). Both missing days were gap-filled manually in one session, per the gap-fill rule
(no `.scheduled/` staging — the queue was empty):

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 10 (Thu) | graph-fraud-ring-detection | ML | "Fraud Rings Are a Graph Problem" — the transaction graph as the detection unit. Anchors: Europol EMMA 9 (Dec 5 2023 — 1,013 arrests in 26 countries, 10,759 mules + 474 recruiters, 2,800+ banks, >€100M exposed, €32M prevented); Operation Jackal IV (Nov 2025–Jun 2026 sweep, 58 arrested / 263 suspects / 22 countries, BBC Aug 25 2026; South Africa 39 arrests, $2.67m seized, 257 accounts blocked); Cifas Fraudscape 2026 (444,000+ NFD cases in 2025, >1,200/day, £2.4bn prevented, SIM swaps +38%). Method: hard links (phone/card/ID → union-find components) vs soft links (device/cookie/IP → clustering) per arXiv:2512.19061 (25M → 7.7M nodes, coverage doubled); Louvain for "fraud islands"; directed cycle detection for layering; GNN layer (arXiv:2411.05815, Elliptic 203,769 nodes / 234,355 edges / 4,545 illicit ≈2%); precision caution (Aite-Novarica ~90% of declines legitimate). Runnable networkx demo: two rings surface through two different doors, shared-device-only cluster lands at REVIEW not FLAG | ✅ published |
| Sep 11 (Fri) | sim-swap-otp-interception-mobile-banking | Cybersecurity | "The Number Is the Password" — the identity layer above the rail (distinct from Sep 4's Daraja joints and Sep 6's aggregator layer). Kenyan anchor: **DTB v Safaricom**, High Court at Machakos, Justice Asenath Ongeri, **June 18 2026** — KES 4,418,601 taken after a Feb 6 2022 SIM swap via an M-PESA agent, **60:40 → Safaricom KES 2,630,000 / DTB KES 1,788,601**, upholding the Mavoko Chief Magistrate ruling (Hon. R.W. Gitau, Mar 2024); quote "a bank cannot hide behind a customer's PIN…". Carrier numbers: Safaricom CCSO Nick Mulila (The Star, Nov 6 2024) — "about 40 fraudulent swaps out of about 750K swaps", ~28,000 swaps/day; INTERPOL African Cyberthreat Assessment 2026 (SIM-swap +327% in 2025, 123,000+ fraudulent SIMs, ~US$3.8m). Global: SEC @SECGov X hack (Jan 9 2024, BleepingComputer); Scattered Spider AA23-320A + MGM ~$100m + Urban 10 years/§13m restitution (Krebs); 0ktapus (169 domains, 5,441 MFA-code records); WindRelay/SpyNote NFC relay (Aug 2026, Malwarebytes). Standards: NIST SP 800-63B PSTN restriction, CISA FIDO/number-matching, FCC rules in force Jul 8 2024. Runnable 30-line step-up gate: PIN-only rule releases all KES 4,418,601; signal-count gate (re-bind age + new beneficiary + crowding the daily limit) holds every transaction | ✅ published |

Covers: `assets/img/cover-graph-fraud-ring-detection.webp` (SVG source in `assets/blog/`,
"ring inside the mesh" metaphor: dim cyan transaction mesh, one 4-account red ring closed
around a shared DEV node with a money loop, a greyed weak same-IP link marked FP, and a
right-hand panel contrasting PER-ACCOUNT SCORE = FLAGGED: NONE with RING/GRAPH SCORE =
4 accounts — distinct from the KG/GNN covers) and
`assets/img/cover-sim-swap-otp-interception-mobile-banking.webp` ("SIM leaves the phone"
metaphor: handset with un-delivered OTPs, SIM card detached and re-bound at a carrier
desk, code rerouted into an attacker handset, bank auth panel showing PIN ✓ / OTP ✓ /
number re-bound 2 days ago / MFA PASSED — money released, plus the 60:40 liability chip
— first SIM/identity cover in the library).

**⚠️ Topic deviation (Sep 11):** the Sep 9 note proposed "Cybersecurity: PAM/JIT for
fintech". That lane is already owned by the Aug 16 `insider-threat-privileged-access-fintech`
post (tags: pam, ueba, privileged-access), so a PAM/JIT sibling would have duplicated it.
Substituted the mobile-identity lane (SIM swap / OTP interception / account takeover),
which no existing post covers.

**Root cause + fix (the two-day outage):** deepseek key `****efc0` was rejected
(401) on Sep 10, then the account hit `HTTP 402: Insufficient Balance` on Sep 10–11.
Re-verified 2026-09-11 23:2x EAT: current key (`****4ac8`) returns
`is_available: true`, balance $9.74, and a live chat completion — the cron LLM path is
healthy again. **⚠️ Fragility:** only one provider key is configured for the whole agent
(`fallback_providers: []`), so a single balance/auth failure stops EVERY LLM-consuming
job at once (blog-poster, tuesday-ai-update, ATLAS sync, AI Crime Watch, weekly sweep).
Consider a second provider key + a fallback chain.

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only calendar gap older than this backfill.
`.scheduled/` remains empty. **Next session actions:** (1) stage or gap-fill Sep 12 (Sat,
Fintech: outage post-mortems), Sep 13 (Sun, Fintech Security: real-time reconciliation
playbook), Sep 14 (Mon, Analytics: control totals & break detection — the table's "Sep 8
(Mon)" row shifts here); (2) never stage Tuesdays (Sep 15/22/29 — the AI Update cron owns
them); (3) keep `tuesday-ai-update` active; (4) consider the Aug 27 backfill (fresh ML
topic — NOT `mlops-regtech-model-governance`, published Aug 25) if daily continuity
matters.

## Week 3 — Proposed (Sep 2–7, corrected weekdays)

| Date | Theme | Proposed topic |
|------|-------|----------------|
| Sep 2 (Wed) | AI Security | AI red-teaming for financial LLM apps — ✅ published as `ai-red-teaming-financial-llm-apps` (see note above) |
| Sep 3 (Thu) | ML | Model drift & monitoring for fraud models (PSI, data quality) — ✅ published as `fraud-model-drift-monitoring` (see note above) |
| Sep 4 (Fri) | Cybersecurity | Mobile money API security: M-PESA/Daraja integration pitfalls — ✅ published as `mpesa-daraja-api-pitfalls` (see note above) |
| Sep 5 (Sat) | Fintech | Sidian Bank 2025 incident (verified reporting) deep-dive — ✅ published as `sidian-bank-mule-heist` (Sep 8 backfill; see note above) |
| Sep 6 (Sun) | Fintech Security | Playbook: third-party API & BaaS integration security — ✅ published as `aggregator-baas-security-playbook` (see note above) |
| Sep 7 (Mon) | Analytics | CBK bank-fraud statistics → analytics of fraud trends — ✅ published as `cbk-fraud-trend-analytics` (see note above) |

## Week 4 — Proposed (Sep 8–14)

| Date | Theme | Proposed topic |
|------|-------|----------------|
| Sep 8 (Mon*) | Analytics | Control totals & break detection in settlement systems (→ shift to Mon Sep 14 per weekday correction) |
| Sep 9 (Tue*) | AI Update | Global AI Roundup — actual Tue Sep 8, published by AI Update cron as `2026-09-08-tuesday-ai-update` |
| ~~Sep 10 (Wed*)~~ → **Sep 9 (Wed)** | AI Security | LLM data-exfiltration via indirect prompt injection — ✅ published as `llm-data-exfiltration-prompt-injection` (see Sep 9 note below) |
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
- **Sidian Bank (Jan 2025 heist; charges Aug/Oct 2025)** — Sh7,882,845 siphoned from customer accounts Jan 11 2025; two charge waves, same incident: trio Nelson Christiano Nangole, John Oboni Odidi (JKUAT), Phostine Hesbon Ochieng (Thika TTI) charged Aug 25 2025 (SPM Geoffrey Onsarigo; legs: I&M 458,313 / National Bank + M-Pesa 451,346 from Kericho Tractor Centre / retained 113,220 M-Pesa; bail 300k cash or 1M bond); Collins Mutuma (MKU, B.Ed Science, 20) charged Oct 27 2025 (CM Lucas Onyina; counts s.317, s.268(1)/275, POCAMLA 4(a); Karoki Sh471,302 → his DTB, onward 300,000 to Dominic Gichiri + ~169,900 to Samuel Mukola Matheka M-Pesa). Vector never publicly established. Sources: kenyainsights.com, tuko.co.ke, biznakenya.com, kahawatungu.com, nairobiwire.com, educationnews.co.ke. Blogged: 2026-09-05-sidian-bank-mule-heist.
- **CBK Financial Sector Stability Report 2024 (pub. Sep 2025)** — Table 14 "Fraud cases and Exposure": cases 173→353; exposed KSh 680.9m→1,963.2m; lost KSh 412.5m→1,594.4m; recovered 368.8m. Channels (lost 23→24): mobile 182.4m→810.7m (+344%), card 15.6m→263.3m (×16.9, 24 cases), identity 32.6m→199.1m (×6.1), computer 74.8m→203.4m, online 106.2m→111.8m (cases 19→106), internet scam 0.8m→6.1m. Conversion (lost/exposed) 60.6%→81.2%; identity 14.6%→97.9%; card 96.6%→59.9%. PDF: centralbank.go.ke/uploads/financial_sector_stability/1556846189_FSR%202024%20Sept.%20Final%202025.pdf (NOTE: report prose says "153 in 2023" but Table 14 totals 173 — table is authoritative). Blogged: 2026-09-07-cbk-fraud-trend-analytics.
- **Danske Bank + Teradata fraud ML** — rules ~40% detection/1,200 FPs per day → ML cut FPs ~50%, raised detection ~60%. Sources: Teradata case study, Fintech Futures.
- **PayPal / Stripe Radar / Mastercard Decision Intelligence** — production AI fraud engines (verify current figures before reuse).

## Rules of the road

1. Every post must cite 2+ verifiable sources per factual claim; drop unverifiable claims.
2. Every post ends with actionable "how we can do better" content (controls, code, checklists).
3. Covers are unique per post (see blog-drafting skill cover-metaphor-library); never reuse a metaphor.
4. Fill empty calendar slots by batch-writing to `.scheduled/` (max 3 subagents parallel; verify post_url/cover/webp after).
5. Re-check the event library before reuse — reporting may have evolved (e.g. court outcomes).

---

# AI Crime Watch — bi-weekly series (starts Sep 2026)

**What:** wholesome, solutions-first collation on crime & AI (how criminals
use AI; how law enforcement curbs it, how it changes investigations; public
vigilance; which AI/ML research helps enforcement or gives criminals an
edge). Home: ml.co.ke, category `AI Security`. Slug scheme:
`ai-crime-watch-issue-<N>`. Series runbook + state machine:
skill `ai-crime-watch-series`; state in `_ai-crime-watch/STATE.md`
(commit changes together with each post).

**Mechanics (do not disturb):**
- Engine cron `AI Crime Watch engine (bi-weekly)`: fires every 14 days
  (~Wed, created 2026-09-09, first fire ~Sep 23) — ALWAYS researches,
  publishes only if the ready-gate (concrete + unique) passes, else opens
  the finisher window (`due = fire day + 3`).
- Finisher cron `AI Crime Watch finisher`: daily 16:30 EAT, monitor-gated
  by `~/.hermes/scripts/ai_crime_watch_signal.py` — sleeps while idle,
  wakes only while an issue is pending. Do not stage posts for the
  finisher; it only writes drafts already in `_ai-crime-watch/cycles/`.
- The series post is an EXTRA post on its day — the daily rotation and
  Tuesday AI Update are unaffected. `.scheduled/` is never touched by the
  series.

| Planned engine fire | Expected | Status |
|--------------------|----------|--------|
| 2026-09-09 (kickoff session) | issue 1 | **PUBLISHED 2026-09-09** — `ai-crime-watch-issue-1`: Taiwan voice-clone romance-scam indictment (Sep 2; 57 indicted, NT$900M, 20,000+ victims, bespoke voice AI). Finisher (first wake) verified the anchor the kickoff missed and closed the cycle READY. Cover: `cover-ai-crime-watch-issue-1.webp`. Anchor banked in `_ai-crime-watch/ANCHORS.md`. |
| ~2026-09-23 | issue 2 cycle | — |
| ~2026-10-07 | issue 3 cycle | — |
| ~2026-10-21 | issue 4 cycle | — |

**Publishing note (2026-09-09):** issue 1 published as an extra post on a
Wednesday (daily post 2026-09-09-llm-data-exfiltration ran as usual).
Slug `ai-crime-watch-issue-1` now exists — the next issue must be
`ai-crime-watch-issue-2`. STATE.md: status published, issue 2, cycle 1
closed. Next action for the ~Sep 23 engine fire: open cycle 2, research a
fresh window (~Sep 9–23), ready-gate, publish or open finisher window.
Do not reuse: Taiwan voice-clone romance ring (issue 1 centerpiece),
INTERPOL 55% African report (Aug 11 spotlight), Operation Jackal / $442B /
Sumsub Kenya (Aug 26 post), EU AI Act activation (Aug 11/18 updates).


---

# Weekly rotation — publishing notes (September 2026)

**Publishing note (Sep 12):** `.scheduled/` was empty at the Sep 12 cron run, so
today's post was gap-filled **directly to `_posts/`** (backdating/gap-fill rule —
the daily cron only ever matches today's date; it never reaches backfill).

| Date | Slug | Theme | Anchor / angle | Status |
|------|------|-------|----------------|--------|
| Sep 12 (Sat) | `fintech-outage-post-mortem` | Data Science, Fintech | **"The Money Stopped Moving: What Fintech Outage Post-Mortems Actually Measure"** — the unit of a payments post-mortem is stranded value + reconciliation debt, not downtime minutes. Anchors (all verified): **Visa Europe 1 Jun 2018** (5.2M transactions failed; 2.4M UK / 2.8M rest of Europe; 1.7M UK cards = 10.4% of cards in use; 14:35 → 00:45; the primary switch's "very rare, partial failure" **blocked** failover and the primary's sync attempts congested the secondary — Guardian + FStech on the 11-page Treasury Committee letter); **TSB Apr–Dec 2018** (all branches + a significant proportion of 5.2M customers; BAU only in Dec 2018; **£32.7M** redress; **£48.65M** fine = FCA £29.75M + PRA £18.9M — FCA press release); **M-Shwari (Safaricom + NCBA) Nov 2025** (>3-day outage, zero balances, "restored" Sunday with reconciliation ongoing — TechCabal 3 Nov 2025) and **Feb 2026** (~36h repeat, "technical issue" at the partner, users still reporting hanging transactions at 10 p.m. on the restoration day — tech-ish 9 Feb 2026, K24); **AWS us-east-1 19–20 Oct 2025** (latent race condition in DynamoDB DNS automation → empty DNS record; automation disabled worldwide; up to 15h for some customers — InfoQ + Forbes); **Uptime Institute 2026** (third parties ≈ two-thirds of publicly reported outages over nine years; 57% of last-major-outage costs > $100k; 1 in 5 > $1M); **DORA** 4h/24h/72h/1-month clocks (EBA RTS); **CBK** 24-hour incident notification (Guidance Note on Cybersecurity, 2017); **FCA PS21/3** impact-tolerance mapping/testing by 31 Mar 2025. Runnable stdlib demo (96 bins, diurnal demand): learned baseline 0.9824, σ 0.00426; deficit integral 4,659 stranded approvals (3.6% of the day), 18% of it outside the hard-stop bins; static 90% threshold fires 90 min late while CUSUM fires 15 min in; retry surge retires 85% and 713 approvals never return; status page "30 min down" vs 240-min customer-visible window. Seed-robust across seeds 1–7 | ✅ published |

Cover: `assets/img/cover-fintech-outage-post-mortem.webp` (SVG source
`assets/blog/cover-fintech-outage-post-mortem.svg`, generated from the same seed-7
simulation). Metaphor: **the outage as a balance sheet** — per-15-min *stranded
approvals* above the zero line (a thin two-hour brownout shelf then two tall
hard-stop bars) and *retries recovered* below it, with a right-hand panel carrying
4,659 / "30 min down" vs "4 h 00 m degraded" / CUSUM 09:45 vs threshold 11:00 /
713 never returned. First deficit-vs-recovery chart in the library; distinct from
`fraud-model-drift-monitoring` (PSI bands) and the two reconciliation covers.

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only calendar gap older than this
week. `.scheduled/` remains EMPTY: the daily cron will keep reporting "All posts
published! Nothing to do." until files are staged, so **every day needs a manual
gap-fill post until the queue is restaged**.

**Next session actions:**
1. **Sep 13 (Sun, Fintech Security — real-time reconciliation + alerting).**
   ⚠️ Topic-overlap risk: `reconciliation-analytics-fintech` (Aug 15) and
   `anomaly-detection-reconciliation` (Aug 31) already own reconciliation
   *analytics*. Differentiate hard: make it the **real-time alerting/runbook**
   piece (streaming SLIs, alert routing and on-call policy, dedup/burn-rate,
   alert-fatigue math, and the post-restart forced sweep) rather than another
   detector toolkit — and state the differentiation in the intro.
2. **Sep 14 (Mon, Analytics — control totals & break detection in settlement)**
   — the Week-4 table row that keeps shifting; still unblogged.
3. Never stage Tuesdays (Sep 15 / 22 / 29 — the AI Update cron owns them);
   keep `tuesday-ai-update` active.
4. Consider the **Aug 27 (Thu, ML)** backfill if daily continuity matters — fresh
   topic, NOT `mlops-regtech-model-governance` (published Aug 25).
5. Verified outage anchors are now banked agent-side in
   `~/.hermes/skills/creative/blog-drafting/references/fintech-outage-incident-bank.md`
   — reuse those instead of re-researching for any incident/resilience/DR post.

---

## Publishing note (Sep 13 — Sunday, Fintech Security)

`.scheduled/` was empty at the Sep 13 cron run (as it has been since Aug 25), so
today's post was gap-filled **directly to `_posts/`** with the calendar date at
`00:00:00 +0300` (past-date rule: the daily cron only ever matches today's date and
never reaches backfill).

| Date | Slug | Theme | Status |
|------|------|-------|--------|
| Sep 13 (Sun) | `real-time-reconciliation-alerting` | Data Science, Fintech | ✅ published |

**"Paging the Ledger: Real-Time Reconciliation Alerting That Actually Wakes Someone"** —
the alerting/on-call piece the Sep 12 note asked for, differentiated from the two
reconciliation siblings by construction: Aug 15 owns *what to compare*, Aug 31 owns
*what to look for* (z-score/IQR/CUSUM, Isolation Forest), Sep 12 owns *what to count
afterwards* (stranded value). This post owns the **paging path** — money SLIs, burn-rate
triggering, dedup/grouping, the on-call page budget, the runbook table, and the
forced sweep after a restart — and says so in the intro.

Verified anchors used (2+ sources each, bodies read, not snippets):
- **RBI harmonised TAT**, circular RBI/2019-20/67 dated 20 Sep 2019, in force 15 Oct
  2019 — UPI/IMPS/NACH transfer debited but not credited: auto-reversal by **T+1**
  (T+5 merchant payments) or **INR 100/day per transaction**, credited **"suo moto,
  without waiting for a complaint or claim"**; TAT is an "outer limit". Annex table read
  verbatim from rbi.org.in (Notification Id=3074).
- **Google SRE Workbook, Table 5-6** — 2% budget/1h = burn 14.4 (page), 5%/6h = 6
  (page), 10%/3d = 1 (ticket); plus the "one bad minute satisfies all three windows, so
  you need suppression" warning and the short-window 1/12 rule.
- **Google SRE Book, Being On-Call** — about 6 hours of work per incident, so a maximum
  of **2 incidents per 12-hour shift**, median 0.
- **Visa Smarter STIP** (26 Aug 2020) — approves/declines on the issuer's behalf during
  outages; deep learning; up to 50% fewer declines claimed.
- **EPC SCT Inst rulebook 2025 v1.1** — the EU Instant Payments Regulation shortens the
  hard timeline for instant euro credit transfers to **10 seconds**, hence millisecond
  timestamps (AT-T056).
- **KenZobe** Daraja callback URL requirements (HTTPS only) — the silent-callback
  failure mode; M-Shwari/TechCabal "restored, not reconciled" reused for the sweep rule.

Runnable stdlib demo (seed 13, robust across seeds 7/13/42 — queue varies 0.1%, first
page by 1 min): 720 bins, KES 1bn/day settled, 0.1%/30-day SLI, so burn 1.0 = KES
694/min. Page-per-break **1,785/shift (one every 24.2 s)**; static 25,000/min **20
pages, first at minute 480** (blind to the whole 360-min drift); raw burn-rate 14.4x
**80 pages, first at minute 238**; burn-rate + group + 60-min silence **5 pages, minute
238**; queue at shift end **KES 3,262,711** across **1,785 lines**, i.e. **INR
178,500/day** of RBI statutory exposure if left past T+1. Output quoted verbatim from
the run and re-verified in-post with `verify-post-code.py`.

Cover: `assets/img/cover-real-time-reconciliation-alerting.webp` (SVG source
`assets/blog/cover-real-time-reconciliation-alerting.svg`). Metaphor: **the paging
path** — a dense red break-line stream funnelled through a purple DEDUP hexagon into a
single green "5 PAGES" pager, over an area chart where the drift stays under the yellow
25,000/min alarm and only the failover burst crosses it, with the burn-rate ladder
(14.4x/6x/1x) and the T+1 clock in the side panel. First pager/dedup/meter cover in the
library — distinct from `ml-monitoring` (dashboard plus alerts panel), the PSI-band
drift cover, the Aug 15 ledger-vs-settlement cover, the Aug 31 radar cover and the
Sep 12 deficit chart.

Word count: **1,718** full-count (body minus code fences, headings/tables/references
included) against the 1,117–1,675 sibling band, i.e. 2.6% over — the same tolerance the
Sep 4 and Sep 12 posts shipped under (quote-, table- and citation-dense).

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only old calendar gap. `.scheduled/`
remains EMPTY, so every day still needs a manual gap-fill until the queue is restaged.

**Next session actions:**
1. **Sep 14 (Mon, Analytics — control totals & break detection in settlement)** — the
   Week-4 row that keeps shifting; still unblogged. Gap-fill directly to `_posts/`.
2. Never stage Tuesdays (Sep 15 / 22 / 29 — the AI Update cron owns them); keep
   `tuesday-ai-update` active.
3. Consider restaging `.scheduled/` for Sep 16–22 (non-Tuesdays) so the daily cron stops
   reporting "Nothing to do"; otherwise keep gap-filling by hand each day.
4. Optional: backfill **Aug 27 (Thu, ML)** with a fresh topic (not
   `mlops-regtech-model-governance`, published Aug 25).
5. AI Crime Watch engine fires around **Sep 23** (issue 2; issue 1 published Sep 9).

---

## Publishing note (Sep 14 — Monday, Analytics)

`.scheduled/` was empty at the Sep 14 cron run (as it has been since Aug 25), so today's post was
gap-filled **directly to `_posts/`** with the calendar date at `00:00:00 +0300` (past-date rule:
the daily cron only ever matches today's date and never reaches backfill).

| Date | Slug | Theme | Status |
|------|------|-------|--------|
| Sep 14 (Mon) | `settlement-control-totals-break-aging` | Data Science, Fintech | ✅ published |

**"The Four Numbers Before the Ledger: Control Totals and Break Aging in Settlement"** — the Week-4
Analytics row that had shifted three times. Differentiated by construction from the reconciliation
family: Aug 15 owns *what to compare*, Aug 31 *what to look for*, Sep 13 the *paging path*, Sep 12
*what to count after*. This post owns the **pre-posting gate** (trailer records: count / hash / debit
total / credit total, batch and file level) and the **aging ladder** for breaks that survive it — and
says so in the intro.

Verified anchors (2+ sources each, bodies read — not snippets):
- **ACH file structure** — Batch Control Record (Type 8): Entry/Addenda Count, Entry Hash (hash total
  of routing numbers, right-justified to 10 digits), Total Debit and Credit Entry Dollar Amounts; File
  Control Record (Type 9) aggregates. Public spec: timetrex.com glossary (Nacha rules are paywalled).
- **Bacs reports** — Submission Report carries transaction count + total value; Input Report the detail;
  the payroll-team check is the control total in the wild (paygate.uk, GoCardless).
- **CBK KEPSS Revised Rules** — §11.2 finality ("final and irrevocable once the Forwarding bank's
  account is debited and the Executing bank account is credited"), §11.3(b)(ii) "the payment was made in
  error by the Forwarding bank" + indemnity route, §11.3(f) a multiple third-party payment with one bad
  payment must NOT be rejected wholesale. Read from the PDF via `curl` + `pdftotext`. KEPSS hours now
  07:00–19:00 from 1 Jul 2025 (Capital FM).
- **Citi $81T near miss** (Apr 2024, disclosed Feb 2025) — $280 → $81 trillion, two staff missed it, third
  caught it ~90 min after processing, reported to Fed/OCC, Citi's "detective controls" wording (CBS, NYT).
- **Citi/Revlon** (Aug 2020) — $7.8M intended, just under $900M wired as a payoff, c. $500M not returned.
- **Citi fat finger** (2 May 2022) — $58M intended, `58m` into *quantity* → $444bn basket, $255bn blocked,
  $189bn to the algo, $1.4bn sold, $48M loss, FCA £27.77M + PRA £33.88M = £61.6M, pop-up overridable.
- **Deutsche Bank €28bn** (16 Mar 2018) — to its own Eurex account, more than its €24bn market cap.
- **UFAA / Kenya unclaimed** — baseline survey KES 241.1bn unclaimed (62% financial services), 2-year
  dormancy rule; record KES 5.182bn in 2025 with claimants down 32.7% (KNA, The Star).

Runnable stdlib demo (seed 14, gate verdicts identical across seeds 7/14/42): 1,170-entry day,
DR = CR = 5,759,092,508, hash 4,862,209,760; **four of five faults REFUSED pre-posting** (truncate →
G3 count 190≠191; renumber/replay → G1/G2; in-flight +360,000 → G3 cr 875,767,475≠875,407,475 then
G4/G5) and the fifth — a duplicate created *inside* the batch with both legs and an honest trailer —
**passes all five gates** (1,172 entries, DR == CR). Break aging: 88 breaks, KES 34.9M; >7 days = 20
breaks / KES 6.88M / 90 h / nearly all of the KES 1.87M write-off risk. Output quoted verbatim and
re-verified in-post with `verify-post-code.py`.

Cover: `assets/img/cover-control-totals-break-aging.webp` (SVG source
`assets/blog/cover-control-totals-break-aging.svg`). Metaphor: **the gate** — a batch stack and its
yellow TRAILER RECORD entering a green CONTROL GATE with four ✓ checks, exiting on a green rail to
POST or a red rail to DO NOT POST, beside a four-rung aging ladder whose bars shrink as the write-off
column grows, feeding a red SUSPENSE → UNCLAIMED box. First gate/rail metaphor in the library —
distinct from the two reconciliation covers, the Sep 12 deficit chart and the Sep 13 pager.

Word count: **1,759** full-count (body minus code fences, headings/tables/references included) against
the 1,117–1,675 sibling band — 5% over, the same table- and citation-density tolerance Sep 4 / Sep 12 /
Sep 13 shipped under (two code blocks, five tables, ten references).

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only old calendar gap. `.scheduled/` remains EMPTY, so
every day still needs a manual gap-fill until the queue is restaged.

**Next session actions:**
1. **Sep 15 is a Tuesday** — the `tuesday-ai-update` cron owns it; do not gap-fill it.
2. **Sep 16 (Wed, AI Security)** and **Sep 17 (Thu, ML)** are the next unblogged non-Tuesday slots —
   gap-fill directly to `_posts/`. For Thu, reuse `references/psi-drift-monitoring-bank.md`; for Wed,
   check the LLM/agent red-team bank before picking an anchor.
3. Consider restaging `.scheduled/` for Sep 16–22 (non-Tuesdays) so the daily cron stops reporting
   "Nothing to do"; otherwise keep gap-filling by hand each day.
4. Optional: backfill **Aug 27 (Thu, ML)** with a fresh topic (not `mlops-regtech-model-governance`,
   published Aug 25).
5. AI Crime Watch engine fires around **Sep 23** (issue 2; issue 1 published Sep 9).
6. Verified control-total + break-aging anchors (and the demo's honest-trailer gotcha) are banked
   agent-side in `~/.hermes/skills/creative/blog-drafting/references/settlement-control-totals-bank.md`
   — reuse before re-researching this class.

## Publishing note — Sep 15, 2026 (Tue, `tuesday-ai-update` cron)

**Published:** `_posts/2026-09-15-tuesday-ai-update.md` → `/posts/tuesday-ai-update/` (slug is shared
by design across the Tuesday series — the newest date wins the permalink; the Sep 8 post remains in
`_posts/` and is what the "Related" link resolves to). Cover: reused
`/assets/img/cover-global-ai-roundup-july-2026.webp` (generic roundup cover, as instructed).

**Title:** "Tuesday AI Update: Sep 15, 2026 — Washington Names Six Chinese Labs in Model-Distillation
Crackdown". Body: **894** full-count words (body minus front matter, headings/links/references
included) against the Tuesday 600–900 budget and the Sep 8 sibling's 893 — in band after four trim
passes (1,681 → 1,176 → 978 → 894). Method note: Tuesday posts are link- and bullet-dense, so the
naive `\S+` count on a first pass badly overshoots; calibrate against the previous Tuesday post
before drafting, not after.

**Anchors used (all 2+ sourced):** CISA/NSA/FBI advisory **AA26-251A** (Sep 8) + Anthropic's
distillation report (~190M Claude exchanges, 151M Alibaba); DeepSeek STAR Market IPO at ~¥500bn/$75bn
+ 160,000 Huawei Ascend 950DT chips for a 1GW Inner Mongolia site (inference only); Supreme People's
Court **Fa Fa [2026] No. 10**; Tencent Hy4 preview as OpenRouter's most-used model; Google's €13bn
Finland investment + 22-year Fortum Loviisa nuclear PPA; Sep 2–10 model wave (Muse Spark 1.3, Fable
5.1/Mythos 5.1, GPT-6 Astra, Gemini 3.8 Flash + Flash Cyber); Mistral's €3bn at >€21bn; LEAP 2026
(AMD/Cisco/HUMAIN MI355X cluster, HUMAIN HGX B300 at >90% utilisation, G42 fundraise); **Egypt's
200MW/$1bn Nvidia data centre** (20MW/$200m first phase; Vodafone Business, Elsewedy Electric,
Cassava Technologies); **Kenya Konza–AWS** Outpost/certification/startup-centre agreement; Tether AI
TranslatePsy-AfriSLM (800M params, 18 African languages, offline); Africa H1-2026 funding ($1.36bn,
only 190 rounds ≥$100k); Brazil's R$2.3bn plan with ~R$1.3bn to Chinese vendors; Sber open-sourcing
GigaChat Ultra Preview + speech models.

**Spotlight differentiation:** framed distillation as a *security-control → compliance* reclassification
(detection signals + the degrade-transcript-fidelity defence), not a model-launch story — the Sep 8
post covered the model wave, so this one leads on provenance and API abuse.

**Build verification (this session's pattern to reuse):** the post's own Actions run came back
`cancelled` because the **ATLAS backup cron pushed to this same repo 49 seconds later** (Sep 15
14:04:35 → 14:05:24), and GitHub cancels the in-progress run for the superseded SHA. The backup
commit's run then completed `success`, and since it is a descendant of the post commit, the Pages
deployment includes the post. **Do not read `cancelled` as a failed publish here** — confirm the
successor SHA's run is `success` and that the post blob is in `origin/main`
(`git ls-tree origin/main _posts/<file>`), then verify the live permalink. Same pattern occurred
Sep 14 (`05e77fd5` cancelled → `94fed0b9` success).

**Verified live:** `/posts/tuesday-ai-update/` returned HTTP 200 with the Sep 15 title and all key
figures rendering; homepage lists the slug; `ml.co.ke` DNS resolved this run.

**Next session actions:** unchanged from the Sep 14 note — **Sep 16 (Wed, AI Security)** and **Sep 17
(Thu, ML)** are the next unblogged non-Tuesday slots. `.scheduled/` is still EMPTY. Aug 27 (Thu, ML)
remains the only old gap. Do not stage Tuesdays (Sep 22 / 29 are `tuesday-ai-update`).
