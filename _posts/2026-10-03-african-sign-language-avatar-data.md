---
title: "300 Deaf Students, One Interpreter: What It Takes to Scale African Sign Language AI"
date: 2026-10-03 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [African Sign Language, Accessibility, Assistive AI, Motion Capture, Datasets, Kenya, Offline AI]
image:
  path: /assets/img/cover-african-sign-language-avatar-data.webp
  alt: "Cover: a signing hand with motion-capture landmark dots, wired to a corpus of captured sign sequences, next to the ratio 300 students : 1 interpreter"
---

## A classroom in northern Kenya, and the number the demo rests on

On 21 September 2026, on the Unstoppable Africa mainstage in New York, a Kenyan founder did something more useful than unveil a model. Elly Savatia demonstrated Terp 360 — a platform that turns speech and text into Kenyan Sign Language performed by a 3D avatar — and announced two things: work has started on two-way translation, and the platform is expanding beyond Kenya into Rwanda, Uganda and South Africa [1][2].

> **Why this is worth reading as engineering**
> The story behind Terp 360 is not a funding announcement. It is a ratio: roughly **300 deaf students, one interpreter** in the northern Kenya classroom where the idea started [3]. Everything else in the product follows from trying to make that ratio survivable — a corpus captured from real signers, an offline byte budget, and a latency budget that decides whether a deaf student waits for a perfect sign or gets a slightly simpler one immediately.
{: .prompt-info }

Savatia's own framing is the design principle, and it is the one sentence worth stealing: *"you don't build solutions and take them to the people; you build with them"* [2]. More than 30 deaf people shaped the product directly — validating signs before they go live, reviewing animations, testing the experience [9].

Positive-lane caveats first, because this is a demo-stage claim and not a finished service: Terp 360 was still described as being in testing when it won the 2025 Africa Prize, and as of August 2025 it worked in one direction only — speech and text into KSL — with signing back into text explicitly "on the roadmap but technically harder" [3][9]. The September 2026 announcement says work on that second direction has started [1]. Read the rest of this post with that in mind: the interesting part is the engineering, not the demo.

## The gap is a ratio, not a slogan

Kenya does not lack legal recognition of Kenyan Sign Language. Article 7(3)(b) of the Constitution requires the State to promote its development and use, Article 54(1)(d) guarantees persons with disabilities the right to use it, and Article 120 recognises KSL as an official language of Parliament [4]. What is missing is delivery capacity, and the National Gender and Equality Commission said so bluntly in its International Day of Sign Languages statement on 23 September 2026: *"A right that cannot be exercised because an interpreter is unavailable, information is inaccessible or a service provider cannot communicate in Kenyan Sign Language remains an unfulfilled right"* [4].

The commission's figures make the sizing exercise concrete:

| Measure | Figure | Source |
|---|---|---|
| Kenyans aged 5+ with a hearing disability | 153,361 | 2019 census, cited by NGEC [4] |
| Kenyans aged 5+ reporting some degree of hearing loss | 4% | 2022 KDHS, cited by NGEC [4] |
| Deaf students in the classroom that started Terp 360 | ~300 | Royal Academy of Engineering [3] |
| Interpreters in that classroom | 1 | Royal Academy of Engineering [3] |
| Interpreters serving ~80,000 hearing-impaired people, Germany | ~850 | German interpreters' association, cited in [8] |

Two of those rows describe the same country with two different definitions of hearing loss, and they land an order of magnitude apart. That is not a footnote: it decides the addressable population, the budget, and what "coverage" means for any assistive product aimed at Deaf users. Run the arithmetic on the published numbers — the only assumptions are the length of the teaching week and a full-time interpreter's hours, both marked in the code:

```python
# Demand side: interpretation throughput, not model quality, is the binding constraint.
# Published inputs; assumptions are marked.
CENSUS_HEARING_DISABILITY = 153_361     # 2019 census, people aged 5+ with a hearing disability (NGEC, 23 Sep 2026)
CENSUS_POPULATION = 47_564_296          # 2019 Kenya Population and Housing Census (KNBS)
KDHS_SHARE = 0.04                       # 2022 KDHS: 4% of people aged 5+ report some hearing loss

CLASS_STUDENTS = 300                    # northern Kenya classroom (Royal Academy of Engineering, 2025)
CLASS_INTERPRETERS = 1
TEACHING_HOURS_WEEK = 30                # assumption
INTERPRETER_FTE_HOURS = 40              # assumption

weekly_minutes = CLASS_INTERPRETERS * TEACHING_HOURS_WEEK * 60
per_student = weekly_minutes / CLASS_STUDENTS
print("one interpreter, 300 students")
print(f"  interpreted teaching per student : {per_student:.1f} min/week")
print(f"  the same room, hearing classmate : {TEACHING_HOURS_WEEK * 60} min/week")
print(f"  access ratio                     : {TEACHING_HOURS_WEEK * 60 / per_student:.0f} : 1")

fte = CENSUS_HEARING_DISABILITY / INTERPRETER_FTE_HOURS
print("\ncost of one interpreted hour per person, per week")
print(f"  {CENSUS_HEARING_DISABILITY:,} people -> {fte:,.0f} full-time interpreter posts")

kdhs_count = KDHS_SHARE * CENSUS_POPULATION
print("\nwhy two national numbers can differ by an order of magnitude")
print(f"  4% of {CENSUS_POPULATION:,}       : {kdhs_count:,.0f} people")
print(f"  census 'hearing disability'      : {CENSUS_HEARING_DISABILITY:,} ({CENSUS_HEARING_DISABILITY / CENSUS_POPULATION * 100:.2f}% of population)")
print(f"  gap between the two definitions  : {kdhs_count / CENSUS_HEARING_DISABILITY:.1f}x")
```

```text
one interpreter, 300 students
  interpreted teaching per student : 6.0 min/week
  the same room, hearing classmate : 1800 min/week
  access ratio                     : 300 : 1

cost of one interpreted hour per person, per week
  153,361 people -> 3,834 full-time interpreter posts

why two national numbers can differ by an order of magnitude
  4% of 47,564,296       : 1,902,572 people
  census 'hearing disability'      : 153,361 (0.32% of population)
  gap between the two definitions  : 12.4x
```

Three things fall out of that. Six minutes of interpreted teaching per student per week is the ratio the product exists to attack. Giving every person in the census group a single interpreted hour per week would take roughly **3,834 full-time interpreter posts** — which is why "just hire more interpreters" is a position, not a plan, in a country where the profession is thin and the queue is long. And the 12.4x definitional gap is the first thing any team should resolve before quoting an addressable market.

For scale, the German comparison in the same literature is instructive: roughly 850 interpreters serve about 80,000 hearing-impaired people — one per ~94 people — in a country with a funded interpreter service, and supply is still contested [8]. Kenya's 300:1 classroom is what a scarce profession looks like when the alternative to automation is no communication at all.

## What Terp 360 actually is

The product is a web-based platform: you type or speak, the system processes the input through Signvrse's sign-language database and a translation model, and a 3D avatar signs the result [9]. The part worth studying is how the dataset was built, because the team's first approach failed on its own terms.

| Element | What the record shows | Source |
|---|---|---|
| Input / output | English and Swahili in, Kenyan Sign Language out (one-way as of Aug 2025) | [9] |
| Original approach | Computer vision on hand shapes — abandoned when it became clear how much meaning sits outside the hands | [9] |
| Current approach | Motion capture of skilled deaf signers, replayed by 3D avatars | [9] |
| Corpus | 2,300+ locally recorded signs; 20,000+ professionally captured sequences | [3][9] |
| Dataset composition | Signers from across Kenya — 60% urban, 40% rural | [9] |
| Validation | Every sign validated by deaf community partners before release | [9] |
| Recognition | 2025 Africa Prize for Engineering Innovation (£50,000, Dakar, 16 Oct 2025) | [3][5] |
| Funding | Google.org Accelerator: Generative AI, June 2025 cohort (20 organisations, share of $30M) | [10] |

On the money, be careful with attribution. Google's own announcement lists Signvrse as one of five organisations in the cohort with impact in Sub-Saharan Africa, describing the work as *"real-time, offline sign language avatars to overcome communication challenges and interpreter shortages for millions of deaf individuals across Africa"* [10]. The accelerator's 20 recipients shared $30M, with individual awards ranging from $500,000 to over $2M, and Google has not disclosed the amount allocated specifically to Signvrse [1]. The UN's *Africa Renewal* reports Savatia citing a **US$2 million** Google investment supporting what it describes as the largest publicly documented database for African sign language [6]. Report the number as the founder's stated figure, not as a Google disclosure — those are different claims.

Keep the units straight: ~2,300 *signs* is the vocabulary, more than 20,000 *captured sequences* is the recording volume — the same signs performed by different signers, from different regions, multiple times over [3][9].

## Two directions of the same bridge

Terp 360 goes speech → sign. Google DeepMind's SL2T, launched 12 August 2026, goes the other way: sign → text, initially American Sign Language to English, shipping inside Gboard and Live Transcribe on Pixel 11 — the first time a sign-language model has reached a mainstream consumer product [7]. DeepMind's own framing of the problem is the honest one: the AI boom in spoken languages *"has not reached the world's more than 200 sign languages — and the estimated 70 million Deaf and hard of hearing people who use them"* [7].

Both halves matter, and they are complements rather than competitors: a Deaf user signing into a phone needs sign → text; a hearing teacher, clinician or bank teller needs speech → sign. Neither replaces an interpreter.

The corpus gap is what they share. KSL is not ASL, and neither is Tanzanian or South African Sign Language — off-the-shelf models trained on Western signing data routinely fail on regional dialects and regional grammar [11]. That is why the same month that brought Terp 360's expansion announcement also brought at least three more African projects pointing at the same missing infrastructure [11]: a Kenyan team (ZeroBionic) rendering speech through a locally 3D-printed, multi-jointed robotic arm designed for offline classrooms; a University of KwaZulu-Natal graduate's system converting South African Sign Language into spoken English, built after watching his parents struggle at a social grant office; and Arusha Technical College in Tanzania, funded under LINGUA Africa to create the first open datasets for Tanzanian Sign Language. Kenya's iHUB and Mastercard Foundation EdTech Fellowship selected Signvrse for sign language translation work alongside DEAFHEALTH and Deaf Outreach Program [11].

One more thing in that set is worth copying: UNDP's HAIDI Innovation Track in Kenya makes working directly with disability communities on testing and validation a *funding condition*, not a nice-to-have [11].

## The constraint that decides whether any of it works

Sign languages are not gesture libraries. Grammar lives in facial expression, head position, body orientation, and the scale and speed of movement — non-manual markers carry negation, questions and intensity [11]. A system that renders hands perfectly and mouths nothing is not signing; it is a mime of signing.

The peer-reviewed record is unusually clear about the consequence. An evaluation of a German Sign Language avatar on HoloLens 2, published in 2025 by researchers at TU Berlin and DFKI, gave expert Deaf users adjustable settings they preferred — and still found *"no significant improvements in UX or comprehensibility were observed, which remained at low levels, amid missing SL elements (mouthings and facial expressions) and implementation issues (indistinct hand shapes, lack of feedback and menu positioning)"* [8]. The paper's rule is the one to take away: *"personalisation alone is insufficient, and that SL avatars must be comprehensible by default"* [8]. A related study in the *Journal on Multimodal User Interfaces*, run with deaf signers comparing three signing agents against a human interpreter, opened by naming the field's real deficit — the *"notable lack in the signing systems evaluation by individuals who utilize sign language"* [12].

Which means comprehension testing with Deaf raters, against a human-signer baseline, is the evaluation that matters — not BLEU against a text reference, and not how good the avatar looks in a launch video.

The second constraint is bytes. A landmark-based sign corpus is never as small as the demo makes it look:

```python
# Supply side: what a landmark-based sign corpus costs in bytes.
# MediaPipe Holistic emits 543 landmarks (33 pose + 468 face + 21 per hand) per frame.
LANDMARKS, COORDS = 543, 3
DISTINCT_SIGNS = 2_300          # locally recorded signs reported for Terp 360 (Royal Academy of Engineering, 2025)
CAPTURED_SEQUENCES = 20_000     # professionally captured sequences in the dataset (TechCabal, Aug 2025)
MAX_VOCAB, FPS, CLIP_SECONDS = 2_300, 30, 2.0   # assumption: one shipped clip per distinct sign

frames = int(CLIP_SECONDS * FPS)
values = LANDMARKS * COORDS
print(f"{LANDMARKS} landmarks x {COORDS} coordinates = {values:,} values/frame; {frames} frames per {CLIP_SECONDS:g}s clip")
print()

head = f"{'encoding':>9} {'bytes/frame':>12} {'KB/clip':>9} {'GB to collect':>14} {'GB vocabulary':>14}"
print(head)
print("-" * len(head))
for name, width in (("float32", 4), ("float16", 2)):
    per_frame = values * width
    per_clip = per_frame * frames
    collect = per_clip * CAPTURED_SEQUENCES
    vocab = per_clip * MAX_VOCAB
    print(f"{name:>9} {per_frame:>12,} {per_clip / 1024:>9.1f} {collect / 1024**3:>14.2f} {vocab / 1024**3:>14.2f}")

# Temporal compression: keep one keyframe, then 1 byte per coordinate for the deltas.
delta = (values * 2 + values * 1 * (frames - 1)) * MAX_VOCAB
flat = values * 2 * frames * MAX_VOCAB
print()
print(f"delta-encoded vocabulary (keyframe + 1 byte/coordinate): {delta / 1024**3:.2f} GB")
print(f"reduction vs float16 clips: {100 * (1 - delta / flat):.0f}%")
```

```text
543 landmarks x 3 coordinates = 1,629 values/frame; 60 frames per 2s clip

 encoding  bytes/frame   KB/clip  GB to collect  GB vocabulary
--------------------------------------------------------------
  float32        6,516     381.8           7.28           0.84
  float16        3,258     190.9           3.64           0.42

delta-encoded vocabulary (keyframe + 1 byte/coordinate): 0.21 GB
reduction vs float16 clips: 49%
```

Read those columns as a product decision. Collecting a 20,000-sequence corpus even at float16 costs **3.64 GB** of raw pose data before you add video, audio and annotations — a recording campaign, not a side quest. Compressing the vocabulary to 0.21 GB is what makes an offline-first client plausible on a phone that is also carrying someone's life. This is the same lever [the TranslatePsy-AfriSLM release](/posts/translatepsy-afrislm-offline-translation/) pulled for translation: the model has to be small enough to live on the device, because the connectivity is not going to arrive first.

And the operational decision that follows is already documented in the product: Signvrse pre-computes common sign combinations for a **35% reduction in processing time**, and streams essential parts first — *"It's better to show a slightly simpler sign immediately than a perfect one after a long delay"* [9]. Quality and delivery latency trade against each other in real time, and in a lecture the deadline is conversational.

## How to apply this

The reusable pattern here is bigger than sign language. It applies to any assistive AI aimed at a community whose language, accent or modality is not in the training data.

| Practice | Why it matters | Concrete test |
|---|---|---|
| Build with the community, not for it | Reference data that is not community-validated fails on grammar, not vocabulary | Are members of the community on the review path for every release, as with Signvrse's pre-release sign validation [9]? |
| Fix your population definition before quoting a market | Census disability counts and self-reported hearing loss can sit 12x apart [4] | State the definition and the denominator in the same sentence as the number |
| Treat the corpus as the product | The model is reproducible; the captured, consented, regional corpus is not | Do you have urban *and* rural signers, and multiple takes per sign [9]? |
| Budget bytes and latency for the worst device | Offline-first is a size constraint, not a slogan | Can the vocabulary fit the phone, and does the first sign appear within a conversational beat [9]? |
| Evaluate comprehension with Deaf raters | Adjustability and polish do not fix missing non-manual markers [8] | Retelling/comprehension scores against a human-signer baseline, not BLEU [8][12] |
| Wire co-design into the funding condition | Voluntary consultation is the first thing dropped under delivery pressure | Is community testing a milestone the funder checks, as UNDP's HAIDI track requires [11]? |

## Key takeaways

| Takeaway | Detail |
|---|---|
| The binding constraint is interpreter throughput | 300 deaf students to one interpreter in the classroom behind Terp 360 — 6 minutes of interpreted teaching per student per week [3] |
| Automation has to be complementary | Speech → sign (Terp 360) and sign → text (SL2T) are two halves of one conversation, neither a replacement for an interpreter [7] |
| The corpus, not the renderer, is the moat | 2,300+ signs / 20,000+ captured sequences, validated by deaf partners, 60/40 urban-rural [3][9] |
| Comprehension is the metric | The HoloLens 2 study found UX and comprehensibility staying low even with the settings users asked for [8] |
| Offline is a byte budget | ~3.64 GB to collect at float16; 0.21 GB delta-encoded for the shipped vocabulary |
| Co-design is enforceable | UNDP HAIDI makes community testing a funding condition; Signvrse has 30+ deaf contributors on the product [9][11] |

## References

1. Birr Metrics — *Google-Backed Signvrse Expands AI Sign Language in Africa*, 22 Sep 2026: [birrmetrics.com](https://birrmetrics.com/google-backed-signvrse-expands-ai-sign-language/)
2. Global Africa Business Initiative — Unstoppable Africa 2026 programme, "Unstoppable Africans: Elly Savatia on Turning Silence Into Signal": [gabi.unglobalcompact.org](https://gabi.unglobalcompact.org/unstoppableafrica2026/Programme-2026)
3. Royal Academy of Engineering — *Kenyan Innovator Elly Savatia named the 2025 Africa Prize for Engineering Innovation winner*, 16 Oct 2025, and the awardee profile: [raeng.org.uk](https://raeng.org.uk/news/kenyan-innovator-elly-savatia-named-the-2025-africa-prize-for-engineering-innovation-winner/) · [africaprize.raeng.org.uk](https://africaprize.raeng.org.uk/2025-cohort/elly-savatia/)
4. People Daily — *Gender and equality commission calls for more sign language interpreters to protect deaf rights*, 23 Sep 2026 (NGEC statement; 153,361 census figure; 2022 KDHS 4%): [peopledaily.digital](https://peopledaily.digital/news/gender-and-equality-commission-calls-for-more-sign-language-interpreters-to-protect-deaf-rights)
5. Disrupt Africa — *Kenyan innovator Elly Savatia wins $67k Africa Prize for Engineering for sign-language app*, 23 Oct 2025 (£50,000 / US$67,000): [disruptafrica.com](https://disruptafrica.com/2025/10/23/kenyan-innovator-elly-savatia-wins-67k-africa-prize-for-engineering-for-sign-language-app/)
6. UN *Africa Renewal* — *Africa's business leaders push for growth that stays on the continent* (Savatia on the US$2M Google investment and the sign-language database): [africarenewal.un.org](https://africarenewal.un.org/en/magazine/africas-business-leaders-push-growth-stays-continent)
7. Google DeepMind — *Putting sign language AI into users' hands* (SL2T), 12 Aug 2026: [deepmind.google](https://deepmind.google/blog/putting-sign-language-ai-into-users-hands/)
8. Wasserroth, Avramidis, Czehmann, Kojic, Nunnari & Möller — *Evaluation of a Sign Language Avatar on Comprehensibility, User Experience & Acceptability*, arXiv:2508.05358: [arxiv.org](https://arxiv.org/html/2508.05358v1)
9. TechCabal — *Signvrse is pushing for real-time sign language interpretation*, 21 Aug 2025 (20,000+ captured sequences; 60/40 urban-rural; 35% pre-computation gain; streaming approach; 30+ deaf contributors): [techcabal.com](https://techcabal.com/2025/08/21/signvrse-sign-language-interpretation/)
10. Google — *The Google.org Accelerator: Generative AI welcomes 5 new organizations having impact in SSA*: [blog.google](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/the-googleorg-accelerator-generative-ai-welcomes-5-new-organizations-having-impact-in-ssa/)
11. iAfrica — *Kenyan Startup Builds Robotic Sign Language Interpreter — and the African Dataset to Train It* (ZeroBionic, UKZN, Arusha Technical College, UNDP HAIDI; iHUB/EdTech Fellowship): [iafrica.com](https://iafrica.com/kenyan-startup-builds-robotic-sign-language-interpreter-and-the-african-dataset-to-train-it/)
12. Imashev, Oralbayeva, Baizhanova & Sandygulova — *Assessment of comparative evaluation techniques for signing agents: a study with deaf adults*, *Journal on Multimodal User Interfaces* 19 (2025): [link.springer.com](https://link.springer.com/article/10.1007/s12193-024-00442-z)
13. Kenya National Bureau of Statistics — 2019 Kenya Population and Housing Census results (47,564,296): [knbs.or.ke](https://www.knbs.or.ke/2019-kenya-population-and-housing-census-results/)

## Related posts

- [Nineteen Languages, One Download: Reading the TranslatePsy-AfriSLM Release Past the Headline](/posts/translatepsy-afrislm-offline-translation/) — the same offline-first byte budget, applied to translation
- [African AI Communities: Masakhane, Lacuna Fund, and Local Dataset Initiatives](/posts/african-ai-communities/) — why community-held corpora decide what African models can do
- [AI in African Education: Personalised Learning for a Young Continent](/posts/ai-african-education/) — the classroom side of the access problem
- [Prequalified: WHO Opens Its Procurement List to AI TB Screening Software](/posts/who-prequalification-cad-tb-ai-screening/) — what it takes for an AI health tool to become purchasable rather than demoable
