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

---

## Publishing note (Sep 15 — staging session; no publish today)

Today is **Tuesday**: the `tuesday-ai-update` cron owns it and already published
`2026-09-15-tuesday-ai-update.md` (894 words). The daily blog cron therefore did **not** gap-fill
today — a second post would have collided with the Tuesday slot. It used the run to restage the
empty `.scheduled/` queue, which had been empty since Aug 25 and was silently reporting
"Nothing to do" on every run.

**Staged (both dated `00:00:00 +0300` — safe, since the cron only fires at 14:05 EAT):**

| Date | Slug | Theme | Words (full body, code fences excluded) |
|------|------|-------|------------------------------------------|
| Sep 16 (Wed) | `mcp-payments-attack-surface` | AI Security, LLM | 1,910 |
| Sep 17 (Thu) | `temporal-validation-fraud-models` | Machine Learning, Data Science | 1,443 |

Calibration: the same measure script on live siblings returns 1,408 (Sep 3), 1,499 (Sep 9), 1,718
(Sep 13) and 1,753 (Sep 14). Sep 17 sits mid-band; Sep 16 is ~9% above the densest sibling and was
kept there deliberately — six incidents, three tables, one code block and 13 references, the same
citation-density tolerance Sep 12 / Sep 13 / Sep 14 shipped under. Both demos were re-run from the
files with `verify-post-code.py` and the quoted stdout matches byte-for-byte.

### Sep 16 — "The Backdoor You Approved: MCP Servers as a Payments Attack Surface"

Anchor choice: the LLM/agent red-team bank was checked first, and its two flagship anchors (Blue41/
Bunq €0.02 memo, Varonis CoSnitch) are already consumed by the Sep 2 and Sep 9 posts. This post
takes the layer underneath both — the MCP tool server itself — and says so in the intro callout
(Sep 9 = exfiltration channels, Sep 2 = red-teaming, this = tool layer, supply chain and scope).

Verified anchors (bodies read, not snippets): nginx-ui **CVE-2026-33032** (CVSS 9.8, missing auth on
the MCP endpoint, fixed in 2.3.4 on 15 Mar 2026, ~2,689 Shodan instances, in Recorded Future's list
of 31 actively-exploited March 2026 flaws); **postmark-mcp** (first real-world malicious MCP server;
npm upload 15 Sep 2025, malicious at 1.0.16 on 17 Sep, BCC to `phan@giftshop[.]club`, 1,643
downloads, Koi Security); Invariant Labs **tool poisoning / rug pull / tool shadowing**; **Supabase
`service_role`** RLS bypass (General Analysis) with **Supabase's rebuttal cited as the counterpoint**;
**CVE-2025-6514** mcp-remote (9.6, 0.0.5–0.1.15 → 0.1.16); **CVE-2025-49596** MCP Inspector (9.4 →
0.14.1); **CVE-2026-27825/27826** mcp-atlassian ("MCPwnfluence", 9.1/8.2); **Asana MCP** cross-tenant
(released 1 May 2025, found 4 Jun, ~1,000 customers, notifications from 16 Jun). The controls section
is the **NSA CSI "MCP: Security Design Considerations for AI-Driven Automation"** (May 2026,
U/OO/6030316-26). Scanner statistics are quoted with their counterweight (~78% YARA-scanner
false-positive rate, methodology caveats).

Runnable demo: a stdlib rug-pull and blast-radius check — fingerprint each tool's name + description
+ JSON schema at approval, re-check at connect, flag instruction-shaped text, score credential scope
→ three verdicts on three servers (ALLOW / BLOCK / NEEDS APPROVAL).

Cover: `assets/img/cover-mcp-payments-attack-surface.webp` (SVG in `assets/blog/`). Metaphor: **the
approved rack** — an agent connector into four approved MCP tool sockets, the CRM socket's card
swapped for a red "same name, new card" that reads the secret store, beside a `service_role` scope
panel and a CVSS ladder. New to the library; `cover-agent-tool-calling` is the function-interface
cover, not this.

### Sep 17 — "Split Before You Believe: Why Offline Fraud Models Score Better Than They Perform"

ML slot. The psi-drift bank is already consumed by the Sep 3 post, so a non-overlapping ML angle was
used instead: **validation before deployment** (temporal ordering, label maturity, metric choice),
stated in the intro against the five drift / detection / graph siblings.

Verified anchors: **Visa dispute rules** verbatim — "no later than 120 calendar days from the last
date the cardholder expected to receive the merchandise or services, not to exceed 540 calendar days
from the transaction processing date" (read from the Visa PDF via `pdftotext`); **CBK** complaints
guidance verbatim — acknowledge "within 48 hours", resolve "within 7 days" (PDF); **Kapoor &
Narayanan** arXiv:2207.07048 (17 fields, 329 papers; every civil-war paper failed to reproduce);
**Kaufman et al.** TKDD leakage definition; **Google Rules of ML #33 and #29** quoted from the guide;
**Breck et al.** ML Test Score (28 tests, "none of these tests was implemented by more than 80% of
teams"); **Sculley et al.** (pipeline jungles, CACE); **Saito & Rehmsmeier** PLOS ONE (both quotes
read off the article body).

Demo: stdlib logistic regression, the same rows scored under a shuffled split with a full-history
feature versus a time-ordered split with a point-in-time feature — ROC-AUC 0.968 → 0.789 but average
precision 0.279 → 0.047, plus a 120-day label-maturity count (60% of rows unresolved). Seed
20260917.

Cover: `assets/img/cover-temporal-validation-fraud-models.webp`. Metaphor: **the shuffle** — TEST and
TRAIN piles with a "FUTURE ROW" card dealt into training, a PR plate showing the leaky curve high and
the honest curve flat on the base rate, and a label-maturity panel with the 120-day dispute window.

**Still UNPUBLISHED: Aug 27 (Thu, ML)** — the only old calendar gap.

**Next session actions:**
1. `.scheduled/` now holds Sep 16 + Sep 17 — the daily cron should publish both automatically at
   14:05 EAT. Confirm each landed in `_posts/` via a `Publish:` or `Publish (date-fixed):` commit.
2. **Stage Sep 18 (Fri, Cybersecurity — payments API/aggregator class, see
   `daraja-api-incident-bank.md`) and Sep 19 (Sat, Fintech outage/operational-resilience, see
   `fintech-outage-incident-bank.md`)** — the queue runs dry again after Sep 17.
3. Never stage Tuesdays (Sep 22 / 29 — the `tuesday-ai-update` cron owns them).
4. Optional: backfill **Aug 27 (Thu, ML)** with a fresh topic (not `mlops-regtech-model-governance`).
5. AI Crime Watch fires ~**Sep 23** (issue 2; issue 1 published Sep 9).
6. New verified-anchor bank for this class:
   `~/.hermes/skills/creative/blog-drafting/references/mcp-security-incident-bank.md` — reuse before
   re-researching MCP / tool-layer material.

## Publishing note (Sep 16 — Wednesday, AI Security)

Published today's staged post on schedule: `.scheduled/2026-09-16-mcp-payments-attack-surface.md` →
`_posts/`, commit **`14b79c6`**, pushed to `origin/main` (the `.scheduled/` deletion was staged in the
same commit via `git add -A`).

**Verification (all green):** Actions run for `14b79c6` = `completed success`; live permalink
`https://ml.co.ke/posts/mcp-payments-attack-surface/` = **200**; homepage listing returns the slug.
Pre-publish static checks: no `post_url` tags, no `cover:` key, `image.path` ends `.webp`
(`assets/img/cover-mcp-payments-attack-surface.webp`, 1200×630 VP8 WebP, 32 KB), all four `/posts/`
cross-links resolve (`llm-data-exfiltration-prompt-injection`, `ai-red-teaming-financial-llm-apps`,
`agent-tool-calling`, `ml-secrets-management`), and `verify-post-code.py` re-ran the demo and
reproduced the quoted stdout byte-for-byte, including the fingerprints
(`541e034e7e77 → d0a00f24b39e`, `fb5e2e711c87`).

Word count re-measured this run: **2,422 full body / 1,916 code-excluded**, matching the 1,910 in the
Sep 15 note to rounding. The density decision stands and the band should be read off live siblings
rather than the older 1,074–1,421 figures in the skill: Sep 12 = 2,328/1,675, Sep 13 = 2,475/1,718,
Sep 14 = 2,583/1,759 (full / code-excluded). This post sits mid-band on full count.

**Queue state after this publish:** `.scheduled/` holds only
`2026-09-17-temporal-validation-fraud-models.md`. **It is empty after tomorrow's run.**

**Next session actions:**
1. Sep 17 cron publishes the last staged file — confirm via a `Publish:` commit in `git log`.
2. **Stage Sep 18 (Fri, Cybersecurity) and Sep 19 (Sat, Fintech) before the Sep 18 run.** With an
   empty queue the cron reports "Nothing to do" and stays silent — no alert anywhere. That is exactly
   how the Aug–Sep gap began, so treat restaging as the default action for any run that finds one file
   or fewer.
3. Never stage Tuesdays (Sep 22 / 29 — the `tuesday-ai-update` cron owns them).
4. Still UNPUBLISHED: **Aug 27 (Thu, ML)** — the only remaining old calendar gap.
5. AI Crime Watch issue 2 fires ~**Sep 23** (issue 1 published Sep 9).

## Publishing note (Sep 16 — CONTENT PIVOT + first positive-AI post)

**PIVOT (user decision, Sep 16 2026):** the NCBA / bank-hack **incident rotation is
RETIRED**. The Week 3 and Week 4 "proposed" tables, and any pending incident slots
(the Sep 18 Cybersecurity and Sep 19 Fintech staging actions listed in earlier notes)
are **CANCELLED — do not stage incident, breach, fraud, hack or outage posts any more.**

From Sep 16 the daily `blog-poster` cron (137b7dcf653c) has a new self-contained mandate:
every non-Tuesday run writes ONE post that is **positive or useful** —
**Lane A**: one verified positive AI story (<=21 days, 2+ body-level sources,
non-Western coverage preferred), or **Lane B**: one AI/ML/ML-Engineering tutorial with
code the agent actually runs and quotes verbatim. Lanes alternate; Tuesdays are still
owned by `tuesday-ai-update`; a legacy `.scheduled/` file dated today still publishes
first so nothing is orphaned (this also removes the old "queue ran dry -> silent gap"
failure mode). **AI Crime Watch engine (d75da864fce0) and finisher (8ea5a3a5de4d) are
PAUSED** — the series does not fire unless the user asks for it back.

### Sep 16 — "The Lab Partner That Never Sleeps: How AI Now Designs Physics Experiments"

First post in the positive-AI lane (Lane A). Slug `ai-designed-physics-experiments`,
commit **`b1f8b2c`**, cover `assets/img/cover-ai-designed-physics-experiments.webp`
(metaphor: an optical table with the conventional cyan beam path, a dashed purple
AI-proposed path through four phase plates, and a search-space panel scoring
candidate layouts — new metaphor class, no sibling uses optics/experiment design).

**Anchor (verified, 2+ body-level sources):** Klimesch, Arlt, Ruiz-Gonzalez et al.,
*Designing physics experiments with artificial intelligence*, **Nature 657, 47–58
(2026)**, DOI 10.1038/s41586-026-10898-6, published **2 Sep 2026** — AI-proposed
experimental layouts "often challenge established design conventions while matching or
even exceeding the performance of human-designed set-ups". Sources read at body level:
Nature article page (title/authors/volume/abstract), University of Vienna Faculty of
Physics release (3 Sep 2026), Phys.org (3 Sep 2026, Krenn + Haslinger quotes), Tübingen
AI Center news (citation + framing), Mario Krenn's blog (7 Sep 2026 announcement).
Background anchors: Krenn, Malik, Fickler, Lapkiewicz & Zeilinger, *Automated Search for
new Quantum Experiments*, **Phys. Rev. Lett. 116, 090405 (2016)** / arXiv:1509.02749;
Ruiz-Gonzalez et al., *Digital Discovery of 100 diverse Quantum Experiments with
PyTheus*, **Quantum 7, 1204 (2023)** / arXiv:2210.09980, open source at
github.com/artificial-scientist-lab/PyTheus.

**Runnable demo (re-verified via `scripts/verify-post-code.py`, stdout reproduced
verbatim):** 5 component types x 6 slots = **15,625** designs enumerated; the
conventional layout scores fidelity **0.6830**, the best six-slot design
(`PS90 PS90 PS90 PS90 PS45 BS22`) reaches **0.982963**; **27 designs tie** at the best
score; **115** clear 0.95; median **0.3536** — used to make the interpretability and
non-uniqueness points concrete.

**Verification:** Actions run for `b1f8b2c` triggered (in progress at write time;
superseding ATLAS-backup `success` runs are the normal pattern here), live permalink
`https://ml.co.ke/posts/ai-designed-physics-experiments/` = **200** with the correct
title, cover WebP = **200**. Static checks: no `post_url`, no `cover:` key, `image.path`
ends `.webp`, all four `/posts/` cross-links resolve. Word count **2,384 full / 1,986
code-excluded**, inside the live sibling band (Sep 12 2,328/1,675, Sep 13 2,475/1,718,
Sep 16 MCP 2,422/1,916).

**Queue state:** `.scheduled/` holds only `2026-09-17-temporal-validation-fraud-models.md`
(the last legacy staged post — an ML tutorial, so it fits the new mandate). From Sep 18
the cron researches, drafts and publishes on its own; nothing needs staging.

**Next session actions:**
1. Confirm the Sep 17 staged post published (a `Publish:` commit in `git log`).
2. From Sep 18 onward, do NOT stage posts — the daily cron's Lane A / Lane B mandate is
   self-contained; only step in if a run reports it found nothing verifiable.
3. Never stage Tuesdays (Sep 22 / 29 are `tuesday-ai-update`).
4. Keep the positive/constructive framing: no incident, breach, fraud or outage posts.
5. AI Crime Watch stays paused unless the user explicitly asks for the series back.

### Sep 17 — "Split Before You Believe: Why Offline Fraud Models Score Better Than They Perform"

**The last legacy staged post** (written and staged Sep 15 before the mandate change;
published by the Sep 17 cron via the legacy flow). Slug `temporal-validation-fraud-models`,
commit **`f411b51`**, cover `assets/img/cover-temporal-validation-fraud-models.webp`
(metaphor: one shuffle fanning into two piles with a "future row" spilling into TRAIN,
paired precision-recall panels scoring different splits of the same rows, and a label-maturity
bar showing 60% of rows unresolved — no sibling cover uses split hygiene / PR-vs-ROC).

**Lane:** B (ML-engineering tutorial) — data-leakage validation for offline fraud models.
Positive/useful framing: it teaches you to earn the number on your model card rather than
republish an inflated one; no incident, breach or fraud-case content.

**Technique + verbatim output (re-verified via `scripts/verify-post-code.py`, stdout quoted
exactly):** stdlib-only 24,000-row synthetic payments set, seed 20260917, prevalence 0.45%;
random split + full-history feature gives **ROC-AUC 0.968 / avg-precision 0.279**, the same
model under a time-ordered split with point-in-time features gives **0.789 / 0.047** — a
**0.179 ROC-AUC** gap but a **0.232 AP** gap, and 14,400/24,000 = **60%** of rows unresolved
under a 120-day confirmation window. Point: ROC hides the leak, PR exposes it at low prevalence.

**Sources (body-level where a figure or quote is used):** Visa dispute-rule language (120-day /
540-day clocks), CBK customer-complaints guidance (48h / 7d), Kapoor and Narayanan
arXiv:2207.07048 (17 fields, 329 papers), Kaufman et al. ACM TKDD 6(4) (leakage definition),
Saito and Rehmsmeier PLOS ONE 2015 (P/(P+N) PR baseline), Google Rules of ML, ML Test Score,
Hidden Technical Debt.

**Verification:** Actions run for `f411b51` = **completed success** (a second run on the same
SHA was superseded/`cancelled` — the known ATLAS-backup race, not a failure); live permalink
`https://ml.co.ke/posts/temporal-validation-fraud-models/` = **200** with the correct title;
cover WebP = **200**. Static checks: no `post_url`, no `cover:` key, `image.path` ends `.webp`,
all six `/posts/` cross-links resolve, no slug conflict. Word count **2,171 full / 1,449
code-excluded** (siblings: Sep 12 2,328/1,675, Sep 16 physics 2,383/1,985, Sep 16 MCP
2,422/1,916 — in band, modestly lean, appropriate for a tutorial).

**Queue state:** `.scheduled/` is now **EMPTY** — and that is the expected, healthy state from
here on: the cron's Lane A / Lane B mandate is self-contained, so an empty queue no longer
means a silent gap.

**Next session actions:**
1. Sep 18+ runs research and publish on their own — do NOT stage posts (the old queue mechanic
   is retired; staging is only useful if the user wants a specific pre-reviewed piece).
2. Never stage or publish a Tuesday: Sep 22 and Sep 29 are `tuesday-ai-update` days.
3. Keep alternating lanes: the last Lane A was Sep 16 (`ai-designed-physics-experiments`),
   the last Lane B is Sep 17, so **Sep 18 should be Lane A** — a verified positive AI story
   (<=21 days, 2+ body-level sources, non-Western coverage preferred).
4. Positive/constructive framing only: no incident, breach, fraud, hack or outage posts.
5. AI Crime Watch (d75da864fce0 / 8ea5a3a5de4d) stays paused unless the user asks for it back.

### Sep 18 — "Certified in Europe, Judged at the Health Post: What a Class IIb CE Mark Really Buys Primary Care"

**Lane:** A (positive AI story, non-Western) — Qure.ai (India) announced **Class IIb CE
certification under EU MDR for Aira**, an LLM-based primary-care clinical decision support
system, on **17 Sep 2026**. Slug `ai-primary-care-class-iib-ce`, commit **`a6108ef`**, cover
`assets/img/cover-ai-primary-care-class-iib-ce.webp` (metaphor: a four-step EU MDR class ladder
with a CE seal hovering over the IIb step + a Se 90/Sp 70 operating-point gauge whose output
splits into a thin "180 cases found" bar and a thick "2,940 confirmatory tests" bar — no
sibling cover uses a certification ladder, seal medallion or gauge; the closest siblings use
gates and precision-recall panels).

**Why positive:** a regulatory milestone that makes frontline AI *trustable* plus real reported
operational wins in Kenya/Nigeria/Mozambique (Qure.ai-reported: -32% documentation time at 98%
completion in Kenya; +67% clinic-to-admin ratio in Nigeria). No incident, breach or fraud content.

**Primary anchors used (all body-level verified):** Qure.ai release 17 Sep 2026 (10 pilot sites
across Nigeria, Kenya, Mozambique, Solomon Islands, Bangladesh; Kenya "deployed with local
implementation partners and Ministry of Health departments"); USA Today/EZ Newswire + CXOtoday
(syndicated release — post states this honestly); MobiHealthNews on AIRA (28 May 2025, independent,
"40% of CHW time on manual data collection"); **MDCG 2025-6** (EU Commission FAQ — MDAI is high-risk
under AI Act Art. 6(1) if it is a device/safety component AND subject to notified-body conformity
assessment); **AI Act Art. 113(c)** (Art. 6(1)/Annex I from 2 Aug 2028, Annex III from 2 Dec 2027);
EU MDR Rule 11 class ladder (TrustedTraceMed); WHO TB screening TPP (90% sensitivity / 70%
specificity minimum, tbksp.who.int); **Sci Rep 2025 PMC12215708** (12 CAD vs 11 radiologists, 774
chest X-rays, South African National TB Prevalence Survey — source of every sensitivity/specificity
pair in the demo; Lunit AUC 0.902, qXR overlapped radiologists); KEMSA + The Star (80 AI X-ray units
to 43 counties, 13 Oct 2025); The Standard (Amref KES 154.4m / $1.2m CAD programme, Global Fund);
PPB MDSW guideline + healthbusiness.co.ke (Dr Ahmed Mohamed, risk-based framework, IEC 62304 /
ISO 14971, post-market surveillance).

**Verification:** code block re-run via `scripts/verify-post-code.py` — output quoted in the post
**matches stdout verbatim** (2% prevalence row: WHO floor 180/2940/20, 5.8%, 56, 16.3;
India-youngest 150/725/50, 17.1%, 67, 4.8). Static checks clean: no `post_url`, no `cover:` key,
`image.path` ends `.webp`, all four `/posts/` cross-links resolve, slug unique. Actions run for
`a6108ef` = **completed success** (no supersession this time). Live permalink
`https://ml.co.ke/posts/ai-primary-care-class-iib-ce/` = **200** with correct title (took ~60s of
CDN propagation); cover WebP = **200**. Word count **2,548 full / 2,330 code-excluded** (siblings
Sep 16 physics 2,383/1,985; Sep 16 MCP 2,422/1,916; Sep 13 2,475/1,718; Sep 17 2,171/1,449) —
above the code-excluded band only because the post carries a single short code block; full-body
count sits inside the sibling range.

**Next session actions:**
1. **Sep 19 should be Lane B** (alternating) — a hands-on AI/ML tutorial with code you run and
   quote verbatim. Last Lane B was Sep 17 (`temporal-validation-fraud-models`).
2. Never publish on a Tuesday: **Sep 22** and **Sep 29** are `tuesday-ai-update` days.
3. `.scheduled/` is EMPTY and that is the healthy state — the Lane A/B mandate is self-contained;
   do NOT stage posts. Only step in if a run reports it found nothing verifiable.
4. Grep `_posts/` for the subject before writing: the AI-in-Africa series (Jul) already covers
   broad healthcare/agriculture/education/fintech surveys — take a narrow, specific angle.
5. Positive/constructive framing only; AI Crime Watch (d75da864fce0 / 8ea5a3a5de4d) stays paused.

### Sep 19 — "The Denominator Is the Metric: Auditing a RAG Retriever Before You Blame the Model"

**Lane:** B (AI/ML tutorial) — last Lane B was Sep 17, last Lane A was Sep 18, so the alternation
held. Slug `rag-recall-at-k-denominator`, commit **`740310b`**, cover
`assets/img/cover-rag-recall-at-k-denominator.webp` (metaphor: three denominator bars of shrinking
length — 12 truly relevant, 8 judged, 5 capped at k — feeding three different "recall@5" values
0.417 / 0.500 / 0.800, beside a ranked top-5 list with four green hits; no sibling cover uses
a fraction panel, denominator bars or a cutoff arithmetic, so it is a new visual identity).

**Why useful (positive/useful mandate):** a teachable measurement technique, not an incident. The
post gives readers four IR metrics in standard-library Python, three ways Recall@k inflates without
anyone lying, and a copy-paste audit function that refuses to report a recall figure without |Rel|,
k, label coverage and a dedup count. No crime, breach, fraud, outage or hack material.

**Differentiation (checked before writing):** no sibling covers RAG *evaluation*. `rag-low-resource-african-languages`
(Aug 2) builds retrieval; `kg-llm-rag` (Jun 1) compares RAG architectures; `evaluating-llms-african-use-cases`
(Aug 7) is model-level benchmarks via lm-eval-harness; `rag-security-attacks` (Jun 1) is attacks.
The measurement layer was a genuine gap.

**Primary anchors (all body-level verified):** RAGAS arXiv 2309.15217 and ARES arXiv 2311.09476
(abstracts — reference-free metrics; lightweight fine-tuned judges + prediction-powered inference
from "a few hundred" annotations across eight KILT/SuperGLUE/AIS tasks, robust to domain shift);
**RAGBench arXiv 2407.11005** ("LLM-based RAG evaluation methods struggle to compete with a
finetuned RoBERTa model on the RAG evaluation task", 100k examples, five industry domains, TRACe);
**BEIR arXiv 2104.08663** (18 datasets, 10 systems, BM25 "a robust baseline", re-ranking best
zero-shot "at high computational costs"); **Buckley/Dimmick/Soboroff/Voorhees, *Information
Retrieval* 10:491–508 (2007)** via the Springer abstract **and** the NIST PDF (pooling: unjudged
assumed nonrelevant; a constant-size pool "represents an increasingly small" sample; judgment sets
"can be biased in that they favor relevant documents that contain topic title words"); **123ofAI
recall@k guide** (Precision@K divides by K, Recall@K divides by |Rel|; partial labels ⇒ overestimate);
**Evidently AI** (P/R at K cannot see ordering; NDCG = DCG/IDCG, 1.0 = ideal); **CIRAL** — HF dataset
card (English queries + Hausa/Somali/Swahili/Yoruba qrels in TREC format) and SIGIR '24 DOI
10.1145/3626772.3657884 pp. 293–302; **Judging the Judges**, IJCNLP 2025 (position consistency via
swapped prompts, primacy/recency-preferring judges — read from the PDF body); **Future AGI** RAG
metrics guide (0.7+ narrow / 0.5+ broad faithfulness, 0.8+ recall at k=20, "Target kappa is 0.6 or
higher"); **Pinecone** rerankers/two-stage retrieval; **The Neural Base** (stage-1 recall@k must be
>95% or reranking cannot compensate). ACM's DL page is JS-gated — CIRAL was verified via the HF
dataset card + citation block instead.

**Verification:** both code blocks re-run via `scripts/verify-post-code.py`; a separate tokenizer
script confirmed each block's stdout equals a quoted output block **verbatim** (block 1 = 32 lines
/ 1122 chars, block 2 = 11 lines / 887 chars). Numbers in the prose and in the cover SVG match
stdout exactly (0.500 vs 0.800 for `q_broad`; mean 0.625 vs 0.700; judged 0.500 vs true 0.417;
R@10 = 1.000 vs R@5 0.667/0.500; reorder: MRR 1.000 → 0.333, nDCG@5 0.811 → 0.481; dupe: 3
positional hits vs 2 distinct). Static checks clean: no `post_url`, no `cover:` key, no `.png`
image path, no `{{` Liquid hazard, slug unique, **6/6 `/posts/` cross-links resolve**. Actions run
for `740310b` = **completed success** (no supersession this time); live permalink
`https://ml.co.ke/posts/rag-recall-at-k-denominator/` = **200 at ~70s** with the correct title and
the code output rendered; cover WebP = **200**; listed on the homepage. Word count **3,647 full /
2,511 code-excluded** (siblings: Sep 18 2,548/2,330; Sep 17 2,171/1,449; Sep 16 2,383/1,985). The
full count is high only because this post carries ~1,140 words of code across two blocks; the
code-excluded count sits ~8% over the sibling top, after four trim passes from an initial 2,914.

**Next session actions:**
1. **Sep 20 = Lane A** (alternating; last Lane A was Sep 18 `ai-primary-care-class-iib-ce`) — one
   verified positive AI story, ≤21 days old, 2+ body-level sources, non-Western coverage preferred.
2. Never publish on a Tuesday: **Sep 22** and **Sep 29** are `tuesday-ai-update` days.
3. `.scheduled/` is EMPTY and that is the healthy state — do NOT stage posts.
4. Grep `_posts/` slugs **and headings** for the subject before writing; RAG *evaluation* is now
   consumed, so a future RAG post must take a different layer (chunking strategy, embedding
   selection, or reranker training).
5. Blog-drafting skill patched this session: code blocks in a post are executed **in isolation** by
   `scripts/verify-post-code.py` (no shared namespace) — keep each block self-contained — and
   `{% raw %}` must sit on its own line *before* the fence, never on the fence line.

---

## Publishing note — 2026-09-20 (Lane A, positive AI story)

**Post:** `_posts/2026-09-20-translatepsy-afrislm-offline-translation.md` — *Nineteen Languages, One
Download: Reading the TranslatePsy-AfriSLM Release Past the Headline*. Cover
`assets/img/cover-translatepsy-afrislm-offline-translation.webp` (metaphor: a handset whose screen is
a grid of 19 language-code chips, a dashed cloud behind it struck through in red and labelled "NO
UPLINK NEEDED", and a size ladder comparing 21-35 MB per-pair Nano with the 641 MB Q4 SLM and its
1.05 GB peak RSS; no sibling cover uses a handset chip-grid, a crossed-out cloud or a size ladder).

**Lane:** A — one verified positive development, ≤21 days old, non-Western focus. Chosen because the
last two posts were Lane B tutorials (Sep 17 `temporal-validation-fraud-models`, Sep 19
`rag-recall-at-k-denominator`); last Lane A was Sep 18.

**Story:** Tether AI Research (QVAC) released the open TranslatePsy-AfriSLM family on 2 Sep 2026 —
19 Sub-Saharan African languages, Apache-2.0 weights on Hugging Face, EMNLP 2026 main-conference
paper (arXiv:2608.18655), plus TranslatePsy-AfriNano (8 languages, 17M-43M params, 21-35 MB/pair)
and TranslatePsy-EuroNano (9 languages, 90 directions). Positive/useful framing: offline on-device
translation for users the cloud never reached.

**Why useful (positive/useful mandate):** no crime, breach, fraud, outage or hack material. The post
gives a verified release inventory + licence split (weights Apache-2.0, synthetic data CC BY-NC 4.0,
raw open-source mix unreleased), the GSMA 2026 connectivity case for offline, a reproducible local
benchmark, and a reusable metric-aware way to read a "beats models 100x larger" claim.

**Differentiation (checked before writing):** no sibling covers machine translation. `swahili-nlp`
(Jun 23) is Swahili tooling, `low-resource-nlp` (Jul 9) is the data problem, `rag-low-resource-african-languages`
(Aug 2) builds retrieval, `fine-tuning-african-language-llms` (Aug 4) fine-tunes, `evaluating-llms-african-use-cases`
(Aug 7) benchmarks LLMs, `edge-ai-mobile-african-markets` (Aug 5) is the device class. Translation
*release + deployment sizing* was an uncovered layer; the intro callout states it.

**Primary anchors (all body-level verified):** arXiv 2608.18655 v2 (abstract, Table 3 SSA-COMET,
Table 25 significance deltas and p-values, Figure 9 prompt template, §5.2 96% filtering at 0.530 vs
0.528 SSA-COMET / 1.76B vs 44.93B tokens, Limitations: no human evaluation / dialect
under-representation / synthetic provenance); tether.io release page (19 language names, 0.8B beats
Qwen3.5-122B-A10B + TranslateGemma-27B + NLLB-3.3B on FLORES-200/BOUQuET/SMOL); Hugging Face model
cards (`TranslatePsy-AfriSLM-0.8B` Apache-2.0 full-parameter SFT of Qwen/Qwen3.5-0.8B, `AfriNano`
96.24% of NLLB-200 accuracy, 56.7x smaller, 3.53x lower peak RAM, Marian/Bergamot, `AfriNano` =
8 languages) plus the HF API listing (25 qvac repos, real file sizes); Crypto Briefing 2 Sep 2026
(21-35 MB per pair); iAfrica 8 Sep 2026; GSMA *State of Mobile Internet Connectivity 2026* via
Capital Ethiopia 20 Sep 2026 (25% SSA using mobile internet, 66%/820M usage gap, 9%/110M coverage
gap, handset 76% of poorest-quintile income, sub-$100 shipments -36%) and Nairametrics 17 Sep 2026
(3.1bn global usage gap, Nigeria ~140M); benchmark provenance FLORES-200 arXiv:2207.04672, BOUQuET
arXiv:2502.04314, SMOL arXiv:2502.12301.

**Original verification this session (not copied from the release):** downloaded the Q4_K_M GGUF
(672,329,792 bytes) from `qvac/TranslatePsy-AfriSLM-0.8B-Q4-GGUF`, fetched `llama.cpp` build `b11062`,
and ran five directions on CPU only (4 threads, no GPU, greedy). Verbatim excerpt — en→sw:
`Mvua zimeanza mapema mwaka huu, na wakulima wanahitaji kujua ni mbegu gani wanapaswa kupanda.`
(`[ Prompt: 100.1 t/s | Generation: 28.7 t/s ]`; repeat runs 91-103 / 25.9-28.8). Peak RSS 1.05 GB
(`/usr/bin/time -v`). Honest edge cases recorded in the post: `haina haja` agreement slip in the
en→sw sentence with `Juni 2027` preserved, and the sw→ha zero-shot greeting staying Swahili
(`Karibu` rather than `Barka da zuwa`), consistent with the paper's OOD caveats.

**Verification:** code block re-run via `scripts/verify-post-code.py` — stdout equals the quoted
output byte-for-byte; the published bash command was re-executed with `-st` and exits cleanly
(without `-st` llama-cli hangs in conversation mode — fixed before publishing). Static checks clean:
no `post_url`, no `cover:` key, no `.png` image path, no Liquid `{{`, slug unique (self-match
confirmed), **6/6 `/posts/` cross-links resolve**, cover WebP = 29.8 KB VP8 1200x630. Word count
**2,810 full / 2,464 code-excluded** (siblings: Sep 19 3,647/2,511; Sep 18 2,548/2,330; Sep 17
2,171/1,449) — inside the live band after one trim pass from 2,878/2,532.

**Next session actions:**
1. **Sep 21 = Lane B** (alternate): one hands-on AI/ML tutorial, code executed and stdout quoted
   verbatim, one code block per concept kept self-contained.
2. Never publish on a Tuesday: **Sep 22** and **Sep 29** are `tuesday-ai-update` days; the cron
   must report and stop.
3. `.scheduled/` is EMPTY and that is the healthy state — do NOT stage posts.
4. Consumed and closed: Tether TranslatePsy-AfriSLM release, offline/on-device translation sizing,
   GSMA SOMIC 2026 connectivity figures. A future African-NLP post must take another layer
   (speech/ASR, dialect coverage, or a human-evaluation harness for MT).
5. New reusable asset: `assets/blog/cover-translatepsy-afrislm-offline-translation.svg` shows the
   handset-chip-grid + crossed-cloud + size-ladder pattern (chips generated programmatically) —
   reuse the generator approach for future "what fits on a device" posts, not the metaphor.

---

## Publishing note — 2026-09-20 (EXTRA post, user-requested: repetition-collapse tutorial)

The user supplied a forensic log from their own coding-agent session (Gemini-family CLI,
non-reasoning "Medium" configuration): at 21:11:36 UTC on 2026-09-19 the model emitted the
token `" shame"` **2,537 consecutive times / 15,222 bytes**, zero tool calls, `thinking: null`,
ending only at the output-token cap; the user then respawned and switched to a reasoning
configuration, which diagnosed an Ahem-font `RenderFlex` overflow and completed the task. They
asked for the science behind the failure plus a blog post, with a **foolproof reproduction that
actually works**. Published as an EXTRA post on top of the cron's own Lane A post
(`2026-09-20-translatepsy-afrislm-offline-translation`) — precedent: two posts on one date
(Sep 9, Sep 16).

| Date | Slug | Theme | Status |
|------|------|-------|--------|
| Sep 20 (Sun) | `inconsistent-decoding-repetition-collapse` | AI Engineering / ML (Lane B-style tutorial) | ✅ published |

**Title:** "Inconsistent by Construction: Reproducing and Catching a Repetition Collapse".
Commit **`1799a49`**. Cover `assets/img/cover-inconsistent-decoding-repetition-collapse.webp`
(SVG in `assets/blog/`; metaphor: **stuck record** — a needle locked in one groove, beside the
6-token lane `' k' 'azi' ' ya' ' k' 'uf' 'anya' × 41`, a blocked `{"tool"...}` chip, and a
sampler-belief panel with p(top-1) 0.291→0.994 and H 3.13→0.05 bits — no sibling cover uses a
record/groove, a token-chip lane or a probability/entropy meter pair).

**All numbers are locally reproduced, CPU-only (4 threads, no GPU), on pinned artifacts:**

| Run (greedy, temp 0, top-k 1) | Tokens | Distinct ids | Longest periodic run | EOS | p(top-1) | H(top-20) |
|---|---|---|---|---|---|---|
| `qvac/TranslatePsy-AfriSLM-0.8B-Q4-GGUF` translation prompt, cap 256 | 256 | 0.035 | 246 = 6 × 41 | no | 0.291 → 0.994 | 3.13 → 0.05 bits |
| same model, `shame` prompt, cap 256 | 256 | 0.023 | 247 = 2 × 123 | no | 0.192 → 0.968 | 3.18 → 0.22 bits |
| repeat-penalty 1.10 / 1.30 | 93 / 37 | 0.796 / 1.000 | none | yes | 0.291 → 0.119 / 0.166 | 3.13 → 3.79 / 3.16 |
| DRY 0.8 / base 1.75 / len 2 | 54 | 0.537 | 5 | yes | 0.291 → 0.244 | 3.13 → 3.18 |
| `bartowski/Llama-3.2-1B-Instruct-GGUF` JSON tool-call task, cap 384 | 379 | 0.066 | 13 tool calls, **1 unique** (335/367 dup 12-grams) | after array closed | — | — |
| same, repeat-penalty 1.30 | 70 | 0.729 | 3 objects, 3 unique, 1 unparseable (`/lib/.../LandingPage.dart`) | yes | — | — |
| same, DRY 0.8 (then 2.0 / len 1) | 379 (then 384 cap) | 0.066 (then 0.151) | 1 unique (then 13 unique, 99/372 dup 12-grams) | same as greedy (then cap) | — | — |

Greedy runs are **deterministic**: byte-identical sha256 on rerun (`fe022903217efd0b` for the
translation collapse, `df95d9fcc088bde4` for the `shame` trace at cap 48). Artifact shas:
afrislm `4af8ee1d…e560afc` (672,329,792 bytes), Llama-3.2-1B `6f85a640…5611df83` (807,694,464 bytes).
The published guard `LoopWatch` (periodicity, exact match) fires at token **20 of 2,537**
(0.8%, 2,517 recoverable), at token 30 of the 246-token local collapse, and produced **0 false
alarms over 26,478 word-tokens** of blog prose at `min_reps=5 / min_tokens=20` (3, 2, 2 alarms at
3/12, 4/16, 4/24 — all on genuinely periodic quoted program output).

**Science anchored body-level (not snippets):** Welleck et al., *Consistency of a Recurrent
Language Model With Respect to Incomplete Decoding* (EMNLP 2020, arXiv:2002.02492 — definition of
inconsistency + Theorem 3.4, read from the PDF; note the term is **not** "sink state", and the
word "sink" does not appear in that PDF at all); Holtzman et al. ICLR 2020 (arXiv:1904.09751);
Olsson et al. (arXiv:2209.11895); Su et al. NeurIPS 2022 (arXiv:2202.06417); **Gu et al. ICLR 2025
(arXiv:2410.10781)** — attention sinks are first-token key biases caused by softmax
normalisation, which **corrects** the popular "the repeated tokens become an attention sink"
explanation the incident report offered; Keskar et al. CTRL §4 (arXiv:1909.05858 — the penalty
formula and "θ ≈ 1.2 … θ = 1 is equivalent to" no penalty, read from the PDF); Weidmann et al.,
DRY (arXiv:2608.22761, 24 Aug 2026 — 47% suffix-extension reduction, placebo control, adopted by
llama.cpp/ExLlamaV2/text-generation-webui); Li et al. ICLR 2024 (arXiv:2402.12875 — CoT as serial
computation, the reason a reasoning respawn behaves differently).

**Verification (all green):** both published code blocks re-run from the file via
`scripts/verify-post-code.py` and their stdout matches the quoted output byte-for-byte; the bash
download/serve recipes and both `curl` recipes were executed verbatim; static checks clean (no
`post_url`, no `cover:` key, no `.png` path, 4/4 `/posts/` cross-links resolve, slug unique);
Actions run for `1799a49` = **completed success**; live permalink
`https://ml.co.ke/posts/inconsistent-decoding-repetition-collapse/` = **200** with the correct
title, the 2,537 figure and the code output rendering; cover WebP = **200** (40,630 bytes).
Word count **3,654 full / 2,822 code-excluded / 2,206 prose-only** (siblings: Sep 19 3,647/2,511,
Sep 20 Lane A 2,810/2,464, Sep 18 2,548/2,330) — full count is in band; the code-excluded figure
runs ~12% over the densest sibling because the post carries **six tables** (626 table tokens vs
394 for the Sep 19 RAG post). Trimmed from 4,395/3,547 in three passes without dropping a fact.

**New reusable asset:** `~/.hermes/skills/creative/blog-drafting/references/decoding-loop-incident-bank.md`
— papers, exact commands, artifact hashes, sampler settings and the measured loop/legality numbers,
so a future decoding/agent-failure post does not re-derive them.

**Next session actions:**
1. **Sep 21 = Lane B** (hands-on AI/ML tutorial) for the daily cron — the alternation is unchanged
   by today's extra post; the cron's last Lane A was Sep 20 (`translatepsy-afrislm-offline-translation`).
2. Never publish on a Tuesday: **Sep 22** and **Sep 29** are `tuesday-ai-update` days.
3. `.scheduled/` stays EMPTY (healthy) — the Lane A/B mandate is self-contained; do not stage.
4. Consumed and closed: greedy/beam inconsistency, induction-head copying, representation
   anisotropy, attention-sink corrections, CTRL repeat penalty, DRY, and the locally reproduced
   collapses on both models. A future post in this area must take a new layer (e.g. constrained
   decoding/grammar-based tool-call generation, KV-cache compression effects, or speculative
   decoding's interaction with loops).
5. AI Crime Watch (`d75da864fce0` / `8ea5a3a5de4d`) stays paused.

---

## Publishing note — Sep 21, 2026 (Lane B tutorial)

**Published:** `_posts/2026-09-21-constrained-decoding-token-mask.md` — "The Mask Is the Contract: What
Grammar-Constrained Decoding Actually Guarantees" (categories AI Engineering / Machine Learning, 8 tags).
Commit **`ba510fc`**; Actions run for that SHA = **completed success**; live permalink
`https://ml.co.ke/posts/constrained-decoding-token-mask/` = **200** on the second retry (~60 s after push)
with the title, the quoted program output and the cover rendering; cover WebP
`assets/img/cover-constrained-decoding-token-mask.webp` = **200** (37,282 bytes, 1200x630 VP8), SVG source
at `assets/blog/cover-constrained-decoding-token-mask.svg` (mask-plate / stencil metaphor: die-cut windows
over a token stream, -inf marks on the blocked chips, state graph, output panel).

**Lane:** B (hands-on AI/ML tutorial). The calendar's own next-session list assigned Sep 21 = Lane B; the
last Lane A was Sep 20 (`translatepsy-afrislm-offline-translation`), so the alternation is preserved. This
is also the "new layer" the Sep 20 note asked for: the post is about the *mask* (grammar-constrained
decoding), not about repetition collapse, and it is differentiated in the intro from
`agent-tool-calling` (what to declare) and `inconsistent-decoding-repetition-collapse` (what
unconstrained decoding drifts into).

**Bodies of evidence, all read at body level:** JSONSchemaBench arXiv:2501.10868v3 (declared vs empirical
coverage definitions; LM-only empirical coverage 0.90 GlaiveAI → 0.38 GitHub Medium → 0.13 GitHub Hard →
0.21 JSONSchemaStore; Guidance GitHub Hard 0.60/0.41, XGrammar 0.69/0.28; failure taxonomy Outlines
42/16/8, Llamacpp 37/18/7, XGrammar 3/5/38, Guidance 25/7/1; Table 2 TPOT medians with the LlamaCpp
backend — LM only 15.40–16.68 ms, Guidance 6.37–9.47 ms, Llamacpp 27.22–29.98 ms, Outlines 30.33–46.57 ms;
GCT Outlines 3.48–8.05 s; Table 3 HF backend Guidance 35.88–44.21 ms vs XGrammar 65.20–66.78 ms);
llama.cpp `grammars/README.md` (GBNF syntax, token matching `<[token-id]>` / `!<token>`, the `x{0,N}` vs
`x? x? ...` slowness warning, the `item-age` range alternation for minimum 0 / maximum 150, and the
schema-is-NOT-injected-into-the-prompt note); `grammars/json.gbnf`; llama.cpp issue **#19051** (opened
2026-01-23, closed 2026-03-09 as *stale*, labels bug-unconfirmed — fail-open on grammar parse failure, 200
OK with unconstrained text); XGrammar arXiv:2411.15100 (MLSys 2025, context-independent prechecking,
persistent stack, up to 100x); llguidance README (~50 µs CPU/token at 128k vocab, integrations: vLLM
0.8.2, SGLang 0.4.4, llama.cpp b4613, Chromium, OpenAI JSON Schema, v1.0.0 Jun 2025); trie automata
arXiv:2608.12574 (0.65 µs vs 5.8 µs per step, 219 vs 7.5 req/s at batch 256, sub-100 ms compile to
K=10,000); zeroentropy constrained-decoding explainer for the -inf masking description.

**Code actually executed (three self-contained stdlib blocks, each run in isolation):** a token mask for
`{"name": str<=12, "age": int 0..150}` against an interpolated token trigram — masked lane **1000/1000
parseable and closed**, 520/1000 with a corpus name, mask overruled the model's top token on **9,202/25,450
steps (36%)**, cache **64 distinct states / 99.75% hits**; the range enumeration (151/1110 = 13.6%,
51/1000 fixed-width, `151` unreachable, `15`→`{0}`, `150`→none); and the unsatisfiable-mask walk (strict
`151` → Stalled, substitute → `150`, `200` → dead end both ways). `verify-post-code.py` = all blocks ran;
a pairing script confirmed each block's stdout equals the quoted block **byte-for-byte**. The two `bash`
blocks are quoted from the project's GBNF guide and marked as documented usage — no llama.cpp binary or
GGUF exists on this host, so they were not executed and no output is attributed to them.

**Word count:** 4,672 full / **2,714 code-excluded**. Siblings measured with the same script: Sep 17
2,171/1,449, Sep 18 2,548/2,330, Sep 19 3,647/2,511, Sep 20 Lane A 2,810/2,464, Sep 20 decoding 3,636/2,793.
Code-excluded is inside that range; the full count runs above the siblings because this post carries three
fenced program listings plus their outputs (~1,960 tokens), which is the tutorial's substance. Five trim
passes took it from 4,935/2,957 without dropping a fact.

**New reusable asset:** `~/.hermes/skills/creative/blog-drafting/references/constrained-decoding-bank.md`
— the verified coverage/efficiency/failure figures, the GBNF snippets, the fail-open issue, engine
integration facts and the demo recipe, so a future structured-output post does not re-derive them.

**Next session actions:**
1. **Sep 22 is a Tuesday** — `tuesday-ai-update` owns the day; skip the daily lane, do not stage a post.
2. **Sep 23 = Lane A** (positive AI story, non-Western preferred). Last Lane A was Sep 20, so the
   alternation resumes there; grep `_posts/` slugs and headings before writing.
3. `.scheduled/` stays EMPTY (healthy) — the lane mandate is self-contained; nothing to stage.
4. Consumed by this post: constrained decoding / token masks / GBNF / JSON-schema-to-grammar coverage and
   efficiency, the fail-open class, range-compilation cost. Adjacent layers still open for a future Lane B:
   KV-cache compression effects, speculative decoding's interaction with loops, jump-forward decoding,
   tool-call grammar generation from framework signatures.
5. AI Crime Watch (`d75da864fce0` / `8ea5a3a5de4d`) stays paused.

---

## Publishing note — Tuesday AI Update, Sep 22 2026

**Published:** `_posts/2026-09-22-tuesday-ai-update.md` (slug `tuesday-ai-update`, live at
`/posts/tuesday-ai-update/`, HTTP 200 verified). Actions run for `4b7a0e7` = **success**.
Cover: `/assets/img/cover-global-ai-roundup-july-2026.webp` (generic roundup cover, WebP confirmed).
**Word count:** 918 full / 868 code-excluded (no code blocks). Siblings: Sep 15 894, Sep 8 893 — ~2% over
band top on the citation-density precedent (17 external links, 6 regions).

**Title:** "Tuesday AI Update: Sep 22, 2026 — Claude Now Leads 26% of Anthropic's Own R&D".
**Week covered:** Sep 15–21, 2026. Anchors used (all body-verified via `extract-web-text.py` or two+
independent snippets): Anthropic R&D Automation Index (Sep 17, 26% / 30k agents / 1-in-47,000 blocked);
Z.ai GLM-5.3 Infra Agent on 100,000+ Chinese accelerators (Sep 17, dense-feedback method, ox-alpha
confirmed); Alibaba DAMO RADAR open-sourced Apache 2.0 (Sep 18, 146 findings, AUC 0.913, *Science* paper);
OpenAI Sponsored Agents (Sep 16); Grok 4.7 (Sep 21); AI Energy Management Alliance (Sep 16, Google 1GW
reducible demand — verified); EU transparency Code of Practice signatories; US clears 70,000 chips for
G42/HUMAIN + Humain IPO prep/$2.5bn fund; Russia's first AI law (Sep 1); Africa (Egypt–Intel 1M/yr,
22 On Sloane KUMii + R1bn, Askya, Janguru $25m, Synapse $13m, Aeon $1m, DFC $155m WIOCC, Ethiopia
hydropower, Cape Town Equinix protests/2.2GW); IDB LatAm figures (Sep 21, 5.1% GDP / −20.9% wages,
verified against 4 outlets).

**Not used / deliberately skipped:** South America had no other week-fresh item; LatAm-GPT and Brazil's
supercomputer plan are older (Aug). Reddit and JS-gated primaries (globenews wire, investing.com) were
snippet-level only — figures cross-confirmed across CNA/WHBL/IndexBox before use.

**Next session actions:**
1. **Sep 23 = Lane A** (positive AI story, non-Western preferred) — Sep 22 was Tuesday-owned, so the
   daily-lane alternation resumes at Lane A. Grep `_posts/` slugs and headings before writing.
2. `tuesday-ai-update` resumes **Sep 29**; do not stage a Tuesday file.
3. `.scheduled/` stays EMPTY (healthy) — the self-contained lane needs nothing staged.
4. External-link note: `datacenterdynamics.com`, `technology.org`, `investing.com`, `japantimes.co.jp`
   return 403/405 to curl (bot walls) — cite them, but body-verify figures via a curl-friendly mirror.

---

## Publishing note — Daily lane (Lane A), Wed Sep 23 2026

**Published:** `_posts/2026-09-23-goalkeepers-2026-ai-equity-pledge.md` (slug `goalkeepers-2026-ai-equity-pledge`,
live at `/posts/goalkeepers-2026-ai-equity-pledge/`). Cover: `/assets/img/cover-goalkeepers-2026-ai-equity-pledge.webp`
(new metaphor: 40/40/10/10 allocation bar with a magnifier over the 10% data slice + four outcome chips; no sibling
reuse). **Word count:** 2,541 full / 2,402 code-excluded — inside the live sibling band (Sep 16 2,383/1,985,
Sep 18 2,548/2,330, Sep 20 2,810/2,464, Sep 14 2,583/1,759).

**Lane:** A (positive AI story). Selected because the last two daily posts were Lane B tutorials (Sep 21
constrained-decoding, Sep 20 inconsistent-decoding) and Lane A is the alternating choice. Not a Tuesday.

**Story:** Gates Foundation commits at least US$1 billion over two years to widen AI access, announced
**September 14, 2026** with the 10th annual Goalkeepers Report *Make This Matter: AI, Equity, and the Choice We
Can't Delay*. Split ~40% education / 40% health / 10% agriculture / 10% digital foundation.

**Sources (body-level verified):** gatesfoundation.org press release (primary, dated Sep 14); the
2026 Goalkeepers Report **PDF** (primary — fetched with curl + `pdftotext -layout`, all figures grepped:
p.22 language gap <6% English vs >60% Yoruba, >90% English training data, 1B/7B "illustrative" endnote, p.17
one doctor per ~2,000 people in SSA vs <200 high-income / >2M extra doctors, p.28 the four tools); Benton
Institute (40/40/10/10 split + endnote caveats); Ghana Business News/GNA (split, second source); CIO Africa
(Sep 15); Innovation Village Vol 23. Independent evidence for the Kenya number: University of Birmingham news
(26 Jun 2026) + *Nature Medicine* cluster-randomised trial (9,600+ patients, 16 clinics, no significant change
in 14-day treatment failure, 2.2% vs 2.0%) and *Nature Health* safety paper (10 Mar 2026: 3.4% hallucinations,
7.8% actively harmful recommendations, 62% documentation unmodified).

**Code:** one stdlib power-calculation block (`statistics.NormalDist`), extracted from the file and run with
`verify-post-code.py`; quoted stdout diffed programmatically byte-for-byte → MATCH. Key numbers: at a 2.0%
control event rate, detecting a 30/20/10/5% relative drop needs 14,566 / 34,676 / 146,287 / 600,267 patients
(two-arm, before cluster design effect). Post states these are the author's own two-proportion calculation.

**Verification:** no `post_url` tags, no `cover:` key, no `.png` image paths; all four `/posts/` cross-links
resolve (ai-primary-care-class-iib-ce, translatepsy-afrislm-offline-translation, rag-recall-at-k-denominator,
fine-tuning-african-language-llms). Cover SVG has no bare `&` (only the valid `&lt;` entity). WebP confirmed
1200×630 VP8.

**Deliberately avoided:** Tether TranslatePsy-AfriSLM (consumed Sep 20), African Next Voices / tokenizer-tax
(themes already covered Jun 23 `swahili-nlp` and Aug 4 `fine-tuning-african-language-llms`), CommonLingua
(April 2026 — outside the 21-day window), Nairobi AI Forum (Feb 2026 — too old), and Penda Health's regulatory
class IIb angle (consumed Sep 18). Differentiation stated in the intro: this post reads the **evidence level**
of the four headline numbers, not the funding announcement itself.

**Next session actions:**
1. **Sep 24 = Lane B** (tutorial) — the daily-lane alternation resumes at Lane B after today's Lane A. Grep
   `_posts/` slugs and headings before writing; open Lane B space noted by earlier notes: KV-cache compression,
   speculative decoding, jump-forward decoding, tool-call grammar generation.
2. `tuesday-ai-update` resumes **Sep 29**; do not stage a Tuesday file.
3. `.scheduled/` stays EMPTY (healthy) — the self-contained lane needs nothing staged.
4. Bot-wall note for future sessions: `undp.org` press releases return 403 to `extract-web-text.py`, and
   `devex.com` returns 403; `gatesfoundation.org` press releases work fine; primary PDFs (Goalkeepers report)
   fetch cleanly with `curl -sL` + `pdftotext -layout`.

---

## Publishing note — Daily lane (Lane B tutorial), Thu Sep 24 2026

**Published:** `_posts/2026-09-24-embedding-compression-audit.md` (slug `embedding-compression-audit`,
live at `/posts/embedding-compression-audit/`). Cover: `/assets/img/cover-embedding-compression-audit.webp`
(new metaphor: ONE 384-dim vector rendered three times at falling precision — float32 cell grid, int8
coarse blocks, a 1-bit packed row — beside a measured recall panel; no sibling reuse). **Word count:**
4,035 full / 2,780 code-excluded, inside the live sibling band (Sep 23 2,541/2,402, Sep 21 4,672/2,714,
Sep 19 3,647/2,511) at 2.4% over the band top on the Sep 4/Sep 12 density precedent.

**Lane:** B (tutorial). Sep 23 was Lane A (Goalkeepers), so the alternation resumed here. Not a Tuesday.
Chosen from the open Lane B space the previous note listed (KV-cache compression, speculative decoding,
jump-forward decoding, tool-call grammar generation) — deliberately NOT another decoding post, since
Sep 20/21 covered repetition collapse and grammar-constrained decoding.

**Technique:** quantizing embeddings (int8 scalar, 1-bit sign, dimension truncation) and measuring what
each costs — recall@10 against exact float32 ground truth, plus the shortlist oversampling factor needed
to recover it. Five runnable blocks: memory arithmetic (stdlib), the recall measurement (numpy),
bytes-moved-per-query (stdlib), a speed benchmark (numpy), and a reusable `audit(embeddings)` function.

**Measured anchors (all from local runs on this host; reproducible, seeded):** 384 dims = 1536 B float32 /
388 B int8+scale / 52 B 1-bit+norm (29.5x once the norm is stored, vs the 32x codebook figure); a 16 GB
node holds 10.4M / 41.2M / 307.7M vectors respectively. Single-stage recall@10 on a 10k clustered corpus:
float32 1.000, int8 0.965, 1-bit 0.516, first-96-dims 0.758, first-192-dims 0.826, ground-truth
stability 0.988. Shortlist recall by oversampling: 10k docs 1x 0.516 / 2x 0.872 / 5x 1.000; 100k docs
1x 0.119 / 2x 0.207 / 10x 0.667 / 25x 0.995 / 50x 1.000. I/O at 100M vectors: 153.6 GB scanned per
query for float32 vs 5.2 GB for the bits (0.260 s at 20 GB/s); 25x shortlist reads 375 KB of float32
rows or 95 KB of int8.

**⚠️ Reusable finding — the speed claim inverts in numpy.** Packed Hamming search in numpy was 5x-20x
SLOWER than the plain BLAS float32 matmul across six comparisons (`np.bitwise_count` path) and 37x-78x
slower with `np.unpackbits`. The published "2 CPU cycles" / "7x faster than angular" figures come from
engines with hardware popcount in C++/SIMD, not from byte-wise numpy. `np.bitwise_count` (NumPy 2.0+)
is 4x-8x faster than `unpackbits` with byte-identical distances. Bank this for any future "is X faster"
claim: measure the implementation, not the format.

**Sources (body-level verified):** huggingface.co/blog/embedding-quantization (threshold at 0, 32x, "2 CPU
cycles", Yamada et al. rescore, ~92.5% without / ~96% with rescoring, MRL 93.1% @12x / 95.8% @3x);
qdrant.tech quantisation guide (SQ 4x, BQ up to 32x "centered vector distributions", PQ 64x, TurboQuant
bit depths, `rescore: true` + `oversampling: 2.0` in the documented request example); Vespa
"Embedding Tradeoffs, Quantified" (~1B hamming/s, ~7x, 32x storage, rescore modes); Vespa
"Matryoshka and Binary vectors" (post-rescore retention 95-96%); arXiv 2608.19388 (VecDB @ VLDB 2026,
+8% PQ / +18% SQ from non-uniform bit allocation); numpy.org bitwise_count 2.0 manual page. The sbert.net
quantization page is cited as the library recipe but was unreachable from this host (network flake) — no
figure is attributed to it.

**Verification:** no `post_url` tags, no `cover:` key, no `.png` image paths; all 7 `/posts/` cross-links
resolve (rag-recall-at-k-denominator, vllm-llm-serving, model-serving-101, agent-memory-systems,
self-hosting-open-weight-llms, edge-ai-mobile-african-markets, mlops-constrained-environments); slug
unique; `verify-post-code.py` -> "OK: all blocks ran"; a claim-checker re-ran every block and asserted
each number quoted in the prose appears in fresh stdout (all 5 passed); cover SVG parses as XML with no
bare `&`, WebP confirmed 1200x630 VP8 (35 KB).

**Next session actions:**
1. **Sep 25 = Lane A** (positive AI story, non-Western preferred) — alternation resumes after today's
   Lane B. Grep `_posts/` slugs and headings before writing.
2. `tuesday-ai-update` resumes **Sep 29**; do not stage a Tuesday file.
3. `.scheduled/` stays EMPTY (healthy) — the self-contained lane needs nothing staged.
4. Open Lane B space still unclaimed: KV-cache compression, speculative decoding, learned sparse
   retrieval (SPLADE), cross-encoder reranker latency budgets, agent tool-call grammar generation.
5. Reusable measurement recipes now banked in this note: seeded clustered-corpus recall harness, the
   ground-truth stability gate (nudge queries 1% before trusting any recall number), the oversampling
   sweep, and the numpy popcount-vs-BLAS benchmark.
