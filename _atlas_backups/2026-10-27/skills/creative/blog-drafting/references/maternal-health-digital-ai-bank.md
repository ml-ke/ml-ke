# Maternal & Digital-Health AI — Verified Anchor Bank

Body-level verified anchors for posts in the African / low-resource **health-AI** class
(maternal health, frontline-worker decision support, digital health deployments, health-system
readiness). Every figure below was read in the source body, not from a search snippet. Reuse
these instead of re-fetching; re-verify only if a post quotes them as current.

## The Nigeria Maternal Health Multiplier (consumed 2026-10-10)

Announced **22 Sep 2026** at Semafor's *Next 3 Billion* event, sidelines of the 81st UNGA, New York.
Gates Foundation + MTN Group Foundation. Post slug: `nigeria-maternal-health-multiplier`.

- Structure: ~**US$25M** anchor, four years (2026-2030), direct + in-kind; explicitly an anchor meant
to catalyse further funding, not full programme cost. First market Nigeria; stated ambition to extend
across Africa (Mupita).
- Targets for 2030: **500,000 women**, **5,000 frontline health workers**, **500 primary health care
  facilities**.
- Three components: AI-enabled phone-based decision support (workers + mothers); affordable
  smartphones and mobile data; improved connectivity at facilities.
- Quotes: Tijani (Minister of Communications, Innovation and Digital Economy) — "the biggest barrier
  ... has not been a lack of innovations, but a lack of focus on the foundational systems necessary to
  drive adoption"; Pate (Coordinating Minister of Health and Social Welfare) — "Partnerships that work
  with our existing health systems, rather than around them"; Suzman (Gates CEO) — "Left to the market
  alone, potential tends to benefit those who can already afford it".
- Body sources: gatesfoundation.org release (22 Sep 2026); mtn.com release (23 Sep dateline); Fortune
  22 Sep 2026 (~$25M, four years, three parts); healthwise.punchng.com 23 Sep; businessamlive.com
  (UNGA sidelines).
- Derived arithmetic (computed, not from the release): 100 women/worker, 1,000 women/facility,
  0.10 facilities/worker, $50.00 per woman, $12.50 per woman-year; $62.50 / $83.33 / $125.00 per woman
  at 80% / 60% / 40% realised reach (67.6x to 168.9x the PROMPTS 74c per-participant cost).

**Trap:** the release's "risk roughly **250 times** higher than Western Europe" is a **lifetime-risk**
comparison, not a ratio of MMRs. Attribute it to the announcement and pair it with Nigeria's MMR
(>1,000 per 100,000 live births, WHO Africa "extremely high" band; SDG target is 70) when a rate is
needed.

## Nigeria maternal burden

- **29% of all maternal deaths worldwide** in 2023, per the most recent UN estimates: ~**75,000**
  women a year, rendered as **one death every seven minutes**; lifetime risk about **1 in 100**.
  Source: BBC, https://www.bbc.com/news/articles/c5yk8ek86kdo. The release's own wording is "nearly
  30%" — citing both gives an independent second pass.
- Only **59% of Nigerian women complete four or more antenatal care visits** (initiative figure,
  repeated by Fortune). WHO's 2016 ANC guideline recommends a **minimum of eight contacts** — the 4+
  indicator is the older, lower threshold, which is the nuance to state.
- Capacity context (BBC): 121,000 midwives for 218 million people (2021); under half of births attended
  by a skilled health worker; health spending at **5% of the federal budget** against the **15% Abuja
  Declaration** target.
- **MAMII** (Maternal and Neonatal Mortality Reduction Innovation Initiative): launched **November
  2024**, Gates-funded, midwifery-led, now 33 states / 172 LGAs, ~**2.9 million pregnant women** in
  scope (pharmanewsonline.com); independent evaluation by the Liverpool School of Tropical Medicine
  (lstmed.ac.uk); FG flagged off service-utilisation incentives **14 Sep 2026** (health.gov.ng).

## Evaluated analogues (the evidence base for any phone-based maternal CDS claim)

**PROMPTS, Kenya — Jacaranda Health (PLOS Medicine cluster RCT).**
Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11835334/
- Design: unmasked 1:1 parallel-arm cluster RCT, **40 facilities / 8 counties**, n=**6,139** consented;
  outcomes from phone surveys; ~85% take-up in treatment facilities.
- ITT effects (standardised indices): knowledge **+0.08 SD** (0.03-0.12, p=0.002); birth preparedness
  **+0.08 SD** (0.02-0.13, p=0.018); routine care seeking **+0.07 SD** (0.03-0.11, p=0.003); newborn
  care **+0.09 SD** (0.07-0.12, p<0.001); postpartum care content **+0.06 SD** (0.01-0.12, p=0.043).
- **Danger-sign care seeking: NOT significant** (−0.01 to 0.08, p=0.096) — the domain the intervention
  named as a priority is the one that did not move. Always state this.
- Magnitudes that hold up in prose: ~**20%** rise in the share receiving the guideline-recommended
  quantity of PNC visits; **>10%** increase in family planning counselling and physical examination.
- Cost: **74 US cents per participant** for their lifetime on the platform (message delivery,
  infrastructure, field enrolment, county engagement). Scale: >750,000 enrolled at study start
  (Nov 2021), >2 million since; ~**3 in 4** enrollees actively engage (administrative data).
- Limits: outcomes **self-reported**; trial "not powered to detect effects on health outcomes"; ITT
  captures the *offer*, not adoption (authors note scaling by ~1/0.76 as an alternative reading).

**m-mama, Tanzania — emergency transport (Delay 2).**
- 989 completed referrals in Shinyanga: **69.9%** community transport system vs **30.1%** standard
  ambulance; **US$170.40 vs US$472 per completed referral**. Source: Munishi et al., PLOS Global
  Public Health 2023, https://journals.plos.org/globalpublichealth/article?id=10.1371%2Fjournal.pgph.0001487
- What made it work (Shayo et al., BMJ Open 2024, https://bmjopen.bmj.com/content/14/2/e073859):
  24-hour toll-free line, dispatch centre with dedicated staff, **pre-notifying the receiving facility**
  so it can prepare, reimbursing community taxi drivers, radio/billboard awareness. The mechanism is
  coordination + reimbursement, **not** prediction.

## System-level caveats to quote

- WHO 2019, digital interventions for health system strengthening:
  "digital health interventions are **not a substitute for functioning health systems**, and ...
  significant limitations to what digital health is able to address".
  https://www.who.int/publications/i/item/9789241550505/
- GSMA Mobile Economy Africa 2026: almost **1 billion people in Africa (63% of the population)** are not
  using mobile internet despite living under coverage; barriers named are device affordability, digital
  skills and relevant content. https://www.gsma.com/mobileeconomy/sub-saharan-africa/
- WHO 2016 ANC: minimum **eight contacts**. https://www.who.int/news/item/07-11-2016-new-guidelines-on-antenatal-care-for-a-positive-pregnancy-experience

## Reusable measurement recipe (alert-load + precision arithmetic)

For any post claiming a screening/triage model helps frontline workers, compute instead of asserting:

1. **Action budget first.** contacts = caseload x contacts/week x weeks; per_day = contacts/(weeks x 7);
   max escalation rate = action_budget_per_day / per_day. (Verified fixture: 100 women, 1/week, 40
   weeks -> 4,000 contacts/year, **14.29/day**, **35.0%** ceiling inside a 5/day budget.)
2. **Precision at prevalence.** PPV = sens*p / (sens*p + (1-spec)*(1-p)). At 2% prevalence:
   90/95 -> **26.9%**, 90/90 -> **15.5%** (6.4 alerts per true case, 84.5% false), 90/80 -> **8.4%**
   (11.9 alerts per case); 70/90 -> 12.5%; 70/80 -> 6.7%. The same rule at 5% prevalence reaches 26.9%
   at 90/90 — state that thresholds need per-market recalibration.
3. Report **escalation rate alongside sensitivity** and pick the threshold from the worker's actionable
   count, not from recall.

## Differentiation map for this class

| Sibling post | Angle already taken | What is still free |
|---|---|---|
| `ai-primary-care-class-iib-ce` (Sep 18) | EU MDR Class IIb CE marking; threshold/specificity arithmetic | deployment/logistics angles |
| `who-prequalification-cad-tb-ai-screening` (Oct 1) | WHO prequalification of CAD software; procurement listing | service-delivery and financing angles |
| `goalkeepers-2026-ai-equity-pledge` (Sep 23) | Gates $1B AI-equity pledge; how to read report numbers | named in-country deployments |
| `ai-african-healthcare` (Jul 26) | continental landscape | any single verified programme |
| `nigeria-maternal-health-multiplier` (Oct 10) | MTN+Gates maternal health deployment; adoption bottleneck | other countries, other service lines |

## Source-access notes

- gatesfoundation.org, mtn.com, fortune.com, healthwise.punchng.com, businessamlive.com, bbc.com,
  who.int, gsma.com all fetch cleanly through `scripts/extract-web-text.py` (re-run with a different
  keyphrase to pull a different part of a long page).
- PMC articles fetch fine; `pdftotext` still needed for PDFs. `web_extract` remains search-only in this
  environment.
