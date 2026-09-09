# Cycle 1 — research sources & uniqueness map (issue 1)

Window: series launch, 2026-09-09. Researched 2026-09-09 (kickoff session).
Search backend: Brave (search-only — extract via curl script when body-level
verification is needed: `python3 ~/.hermes/skills/creative/blog-drafting/scripts/extract-web-text.py URL KEYPHRASE`).

## Uniqueness map — what the blog ALREADY took (do NOT re-centerpiece)

- INTERPOL African Cyberthreat Assessment 2026 (pub ~Aug 4 2026; "AI in 55% of
  reported African cybercrime") → FULL Spotlight in `2026-08-11-tuesday-ai-update`
  + re-cited Aug 25 & Sep 1 AI Updates. TAKEN.
- INTERPOL Global Financial Fraud Threat Assessment 2026 ($442B lost 2025,
  AI-fraud 4.5x more profitable, notices +54%, $1.1B recovered) → cited in
  `2026-08-26-deepfake-fraud-financial-services` + Mar 2026 The Register.
  TAKEN as a stat; may only be *referenced* via cross-link, never re-narrated.
- Operation Jackal (Aug 2026, 58 arrests / 263 suspects / 22 countries) →
  covered in `2026-08-26-deepfake-fraud-financial-services`. TAKEN.
- Sumsub 2026: deepfakes ~10% of fraud attempts in Kenya; SA +269% YoY →
  `2026-08-26` post. TAKEN.
- Arup HK$200M / UK $243K voice / general deepfake-fraud canon →
  `2026-08-26` post (deep-dive). TAKEN.
- EU AI Act "enforcement phase" → name-dropped in Aug 11/18 AI Updates and
  `2026-08-25-mlops-regtech-model-governance` (financial governance angle —
  NOT the crime/deepfake-labeling angle). PARTIALLY AVAILABLE for a
  crime-response framing (Article 50 labeling as public-vigilance + LE tool).
- INTERPOL 651 arrests Africa op = Feb 2026 (Operation Red Card 2.0 era) —
  old AND tangentially used; skip.

## Candidates found (2026-09-09 sweep) — ranked for issue 1

### A. Pillar 5 / research-edge led (STRONGEST differentiation — blog rarely covers)
1. **"The Deepfakes We Missed: We Built Detectors for a Threat That Didn't
   Arrive"** — arXiv 2605.12075 (May 2026). Systematic 2017→2026 corpus review:
   where deepfake harm actually materialized vs where detection research went.
   Angle: calibrating detection/provenance research to real crime data.
   Status: SNIPPET-ONLY — verify abstract/authors on arxiv.org/abs/2605.12075.
   Blog coverage: NONE (grep).
2. **ViKing synthetic-voice study** — voice bot extracted sensitive info from
   52% of participants; even warned participants still leaked. Cited via
   Pindrop page. Status: SNIPPET-ONLY — find the underlying academic paper.
3. **Pindrop +1,300% deepfake-fraud attempts 2024** (over 1.2B calls; +475%
   insurance, +149% banks; +173% synthetic calls Q1→Q4 2024) — PRNewswire +
   Pindrop. Older (2025 data) but pillar-5/industry-signal usable as context.
   SNIPPET-ONLY.
4. C2PA / watermarking + EU Art 50 operational link (below) — the *labeling*
   research/policy stack as LE evidence layer. Needs a research anchor
   (e.g. C2PA adoption stats 2026, detection-eval work) — research target.

### B. Enforcement-infrastructure led
5. **EU AI Act Article 50 transparency rules live 2026-08-02** (EC news
   2026-08-02: "Safer and more transparent AI"; deepfake/content labeling
   obligations for gen-AI; Aug 2 2026 effective date). Fresh, concrete,
   dated. Blog used EU AI Act only passingly → crime×AI reading is open:
   labels as the "nutrition label" for synthetic media → helps platforms,
   police (evidence marking), and the public. Status: SNIPPET-ONLY — verify
   EC page + BiometricUpdate/BrightDefense body-level.
6. **ASIC warning 2026-08-17** — Australian regulator warns consumers about
   investment scams using gen-AI deepfakes/fabricated content. Fresh,
   unused on blog (grep asic = false-positive substring matches only).
   SNIPPET-ONLY — verify ASIC primary release.
7. **NY State Division of Consumer Protection warning Aug 2026** citing FTC
   data (ScamWatchHQ). Verify — SNIPPET-ONLY, thin; optional.

### C. Background / context (verified enough to reuse as backdrop only)
- Europol IOCTA 2026 (pub 2026-04-28): AI across fraud/identity/social
  engineering; "velocity gap". Europol EU OCTA May 2026: fraud = fastest-
  growing organised-crime area, EU losses $64.1B in 2025; criminal
  "fraud-as-a-service" kits incl. voice cloners/document forgers.
  (TechTimes Jun 16 2026). NOT blog-covered as such (check before use).
- Bank of England deepfake scam warning (Guardian Jun 9 2026 — Farage/Bailey
  fakes) — backdrop for public-figure deepfakes → vigilance section.
- Bitdefender Reddit ads impersonating BBC/FT/Guardian for fake AI
  investment schemes (Jun 2026) — tactic detail for the public lens.
  SNIPPET-ONLY.
- 29 US states (Apr 2026) criminal NCII statutes; TAKE IT DOWN Act signed
  May 2025, FTC enforcement from May 2026, first federal arrests late May
  2026 (TechTimes) — enforcement-progress pillar. SNIPPET-ONLY.

## Queries already run (avoid blind re-runs; go deeper instead)

AI fraud arrests Sep 2026; INTERPOL AI ops; EU AI Act LE Aug 2026; voice
clone prosecutions; deepfake CEO 2026; DOJ AI fraud Aug 2026; Interpol
financial fraud 2026; Kenya DCI AI fraud; Europol IOCTA 2026; "August 2026"
busts; INTERPOL 651; regulators Aug 2026; Guardian/Reuters/BBC Sep 2026;
Hiya/Pindrop 2026. → In-window (Aug 19–Sep 9) crime×AI items are THIN in
this index; Sep 1-9 specifically: nothing surfaced.

## Next research targets for the finisher (Sep 10–12)

1. Fresh-in-window crime events: interpol.int newsroom, europol.europa.eu
   newsroom, justice.gov press, fbi.gov, thehackernews.com AI-crime,
   theregister.com — anything Sep 1–12 2026.
2. Pillar-5 deepening: verify arXiv 2605.12075 (read abstract, authors);
   find the ViKing paper; arXiv cs.CR daily feed (search_query cat:cs.CR
   submittedDate desc — the raw API query used 2026-09-09 returned 0
   entries: retry with abs:deepfake AND cat:cs.CV separately);
   C2PA/watermark adoption + EU Art 50 implementing guidance.
3. Africa: TechCabal / Techpoint / Disrupt Africa / The Star KE for Sep 2026
   AI-scam or police-tech items; ODPC/DCI releases.
4. If by Sep 12 nothing clears the gate → mark cycle skipped (valid
   outcome). The engine (next fire ~Sep 23) opens cycle 2 regardless.
