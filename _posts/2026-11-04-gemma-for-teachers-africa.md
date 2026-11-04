---
title: "500 Teachers, 15,000 Students, One Shared Tablet"
date: 2026-11-04 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [education, small language models, gemma, evaluation, phonics, kenya]
image:
  path: /assets/img/cover-gemma-for-teachers-africa.webp
  alt: A shared classroom tablet split into five student slots beside a phonics ladder and a memory bar showing which open model build fits
---

## Introduction

> **The pilot is 500 teachers. The artifact that outlives it is the eval suite.**
> On 4 November 2026, Google's Africa blog announced that Fab AI will lead a cohort of local innovators in Kenya, Ghana and South Africa to build classroom tools on open Gemma models, funded jointly by Google.org and the Gates Foundation, with compute credits and technical support from Google DeepMind. The first phase aims at up to 500 teachers and 15,000 students, with a stated ceiling of more than 2 million learners later. Datasets, evaluation suites and models are to be published.
{: .prompt-info }

An education announcement is easy to write and hard to check, because almost every number in it is a target rather than a measurement. This one is different in a useful way: the organisation running the cohort already publishes benchmarks for education models in low- and middle-income countries, and it ran the randomised trial that the funding is implicitly built on. That gives us two things to read against each other, the promise and the evidence.

The engineering question underneath is also concrete. If a teacher-facing tool has to run on a low-cost Android tablet shared by several students, then the model size, the quantisation and the context budget are decided by the device, not by the demo. And if the tool is meant to support early reading, then the thing you must measure is whether the text it generates can be sounded out by a child who has been taught only part of the alphabet.

## What was announced, and who signed it

The 4 November post is written by Benson Kinoti, Google for Education lead for Sub-Saharan Africa. It pairs a funded cohort with a training programme:

| Partner | Role in the announcement |
|---------|--------------------------|
| Fab AI | Leads a cohort of local innovators in Kenya, Ghana and South Africa building bespoke tools on open Gemma models |
| Google DeepMind | Compute credits and technical expertise |
| Google.org and the Gates Foundation | Joint funding |
| Google AI Educator Series | Expands across Sub-Saharan Africa, starting in South Africa with the South African Council for Educators, then Nigeria, Kenya and Ghana in 2027 with Ubuntu Hub |

The three product areas named for the cohort are curriculum-aligned lesson planning, timely diagnostic feedback and assessment, and practice-based teacher coaching. The commitment to publish "datasets, evaluation suites, and open-source models" is the sentence worth underlining, because those are the parts another country's ministry can inspect and reuse after the pilot ends.

The funding sits inside a wider round announced a day earlier. Google.org's education post on 3 November lists more than $9.5 million across five organisations: CUNY's Computing Integrated Teacher Education initiative, Fab AI, the Krause Center for Innovation, Teaching Lab's Coherence OS and 1EdTech Global. One of those five is the Africa cohort. The rest are mostly United States institutions, which is worth knowing when a headline reads "AI for teachers" and the African piece turns out to be a fifth of a $9.5 million round.

Context from the same post: more than 60 percent of Sub-Saharan Africa's population is under 25. That is Google's figure, and it is the reason a 500-teacher pilot is being described as phase one rather than the programme.

## The pilot numbers, and the part that outlives them

Three numbers in the announcement carry most of the meaning, and each is a target:

- **500 teachers**: the number who receive tools and training in the first phase.
- **15,000 students**: the learners behind those teachers, so about 30 per teacher.
- **2 million learners**: the stated long-run ceiling, which is roughly 130 times the first phase.

Nothing here is a result. A pilot of this shape produces three kinds of evidence, and they are not interchangeable. Usage evidence (how many hours teachers and students put in), quality evidence (whether the lesson plans and reading texts are correct and usable), and outcome evidence (whether children learn more). The Sierra Leone trial is the only one of the three that exists yet, and it is the reason the funding is defensible.

## What the evidence already shows

The technical report behind Google's May 2026 education post describes a pre-registered, cluster-randomised trial run with Fab AI and local teachers. The design is worth reading before any of the effect sizes:

| Item | Detail |
|------|--------|
| Design | Two-arm cluster-randomised trial, classrooms as clusters, pre-registered with the AEA RCT Registry (AEARCTR-0016651) |
| Setting | 12 government-supported junior secondary schools, Port Loko District, Sierra Leone |
| Timing | Nine weeks, 6 October to 5 December 2025, second term of the 2025 academic year |
| Sample | 1,763 students aged 13 or older in 48 classrooms (16 grade 7, 32 grade 8) |
| Allocation | 24 treatment and 24 control classrooms, half of each school-grade's maths classrooms per arm |
| Treatment | Gemini's Guided Learning feature used in half of weekly maths lessons, 12 hours requested over the trial |
| Analysis panel | 1,423 students present at both baseline and endline |

Teacher training ran before randomisation, through a cascade: a five-hour session for trainers, then roughly five to six hours per maths teacher, for both arms. That detail matters, because it means the comparison is not "AI versus nothing". It is "AI plus trained teachers versus trained teachers".

The results, as reported:

| Estimate | Effect (SD) | 95% CI | p |
|----------|-------------|--------|---|
| Intent to treat | +0.258 | [0.027, 0.488] | 0.029 |
| Per additional hour of use | +0.016 | [0.002, 0.031] | 0.026 |
| Completing the requested 12 hours (treatment on the treated) | +0.380 | [0.040, 0.719] | 0.029 |
| Interaction with baseline maths score | +0.195 | [0.074, 0.315] | 0.002 |

The headline figure converts to a student at the 50th percentile moving to about the 60th, or roughly 1.2 to 1.7 years of typical learning progress in low- and middle-income countries. Uptake was unusually high for an education technology trial: 69.0 percent of the 871 students in treatment classrooms reached the 12-hour threshold, and treatment classrooms averaged about 15 hours, a quarter more than requested. Attrition was 124 students from a 1,547-student baseline sample, with 214 new entrants mid-trial.

Three caveats belong next to those numbers rather than in a footnote.

**The interaction term is the equity question.** Students who started stronger gained more: each additional standard deviation of baseline maths proficiency added 0.195 SD to the treatment effect. An intervention that helps the top of the class more than the bottom can raise the average and widen the gap at the same time. The report says it analyses other heterogeneity in an appendix, but on this axis the direction is clear.

**Baseline reading was far below the maths baseline.** The balance table puts mean baseline reading at about minus 1.35 SD in both arms, against maths near zero, because the maths score is standardised within this cohort while reading is on a different scale. Read it as a description of the cohort rather than a deficit claim: these students were in grade 7 and 8 in a system where reading is the binding constraint. That is precisely why the new cohort's brief includes early reading, and why a maths result cannot be transferred to a reading tool.

**It was Gemini, not Gemma, and it was one district.** The trial tested a hosted model in classrooms with devices and connectivity supplied by the project. The Africa cohort is being asked to build on open weights that run locally. Different model, different failure modes, different support burden.

## The reading target is the harder one

Fab AI's public benchmark pages are the most useful thing to read before the cohort ships anything, because they document where models fail at exactly the tasks the cohort has been funded to do. The suite includes pedagogy questions drawn from Chile's Ministry of Education, a special-educational-needs subset, visual maths and visual reasoning, multilingual pedagogy, linguistic knowledge and a decodability assessment.

Two of the descriptions are unusually blunt:

- On visual maths for the early grades: leading models are acing the international maths olympiad while "all models are still struggling with the kind of early grade visual maths taught in low- and middle-income countries".
- On multilingual pedagogy: the labs claim growing language coverage, and the benchmark asks whether a model can support a teacher in the language that teacher actually uses in the classroom.

That framing is the right one for a reading tool, and it points at a measurement most education teams skip. A generated story can be grammatical, engaging and useless to a beginner, because the child cannot decode it. Decodability is computable: a word counts if every grapheme in it has been taught by that point in the phonics sequence. A passage that is 95 percent decodable supports independent reading; one at 70 percent sends the child to guess from pictures and context.

The script below implements that check with a cumulative scope and sequence and a greedy grapheme matcher. It scores an unconstrained, plausible-sounding passage against a controlled one.

{% raw %}
```python
import re

# Grapheme inventory, longest first so greedy matching reads "sh" as one unit.
GRAPHEMES = ["igh", "air", "ear", "ck", "sh", "ch", "th", "ng", "ai", "ee", "oo",
             "oa", "ou", "oi", "ar", "or", "ur", "ow", "ea", "qu",
             "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
             "n", "o", "p", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Cumulative scope and sequence: what a class has been taught by each week.
WEEK3 = {"s", "a", "t", "p", "i", "n", "m", "d"}
WEEK5 = WEEK3 | {"g", "o", "c", "k", "e", "u", "r", "h", "b", "f", "ck"}
WEEK8 = WEEK5 | {"l", "j", "v", "w", "x", "y", "z",
                 "sh", "ch", "th", "ng", "ai", "ee", "oo", "oa"}
TAUGHT = {3: WEEK3, 5: WEEK5, 8: WEEK8}

# High-frequency words taught as whole shapes rather than sounded out.
TRICKY = {"the", "a", "i", "you", "was", "said", "are", "of", "to", "do", "he",
          "she", "we", "me", "my", "they", "their", "there", "one", "two",
          "some", "come"}


def segment(word):
    """Greedy longest-match segmentation into graphemes."""
    out, i = [], 0
    while i < len(word):
        for g in GRAPHEMES:
            if word.startswith(g, i):
                out.append(g)
                i += len(g)
                break
        else:
            return None            # a letter the inventory cannot cover
    return out


def decodable(word, taught):
    if word in TRICKY:
        return True                # taught as a whole word
    segs = segment(word)
    return bool(segs) and all(g in taught for g in segs)


def score(text, taught):
    words = re.findall(r"[a-z]+", text.lower())
    ok = [w for w in words if decodable(w, taught)]
    return len(ok), len(words), sorted({w for w in words if not decodable(w, taught)})


GENERATED = ("The beautiful elephant wandered past the mountain because the weather was "
             "pleasant and the farmers were celebrating a plentiful harvest near the river.")
CONTROLLED = ("The cat sat on the mat. The dog ran to the log. Mum had a big red hen "
              "and the hen sat in the sun.")

print(f"{'week':>4} {'passage':<11} {'decodable':>9} {'words':>6} {'share':>6}")
for week in (3, 5, 8):
    for name, text in (("generated", GENERATED), ("controlled", CONTROLLED)):
        ok, total, bad = score(text, TAUGHT[week])
        print(f"{week:>4} {name:<11} {ok:>9} {total:>6} {ok / total:>6.0%}")

ok, total, bad = score(GENERATED, TAUGHT[8])
print("still not decodable at week 8:", ", ".join(bad))
ok, total, _ = score(CONTROLLED, TAUGHT[3])
print(f"controlled passage clears a 90% target at week 3: {ok / total >= 0.90}")
ok, total, _ = score(GENERATED, TAUGHT[8])
print(f"generated passage clears a 90% target at week 8: {ok / total >= 0.90}")
```
{% endraw %}

```text
week passage     decodable  words  share
   3 generated           9     23    39%
   3 controlled         13     25    52%
   5 generated          10     23    43%
   5 controlled         24     25    96%
   8 generated          16     23    70%
   8 controlled         25     25   100%
still not decodable at week 8: beautiful, farmers, harvest, mountain, near, pleasant, weather
controlled passage clears a 90% target at week 3: False
generated passage clears a 90% target at week 8: False
```

The interesting part is not that the unconstrained passage scores worse. It is the shape of the two curves. The controlled passage goes 52, 96, 100 percent across the three weeks: the same text fails at week 3 because "big", "red", "hen", "log" and "sun" use letters not yet taught, then becomes fully decodable later. Decodability is a property of a passage and a point in the sequence, never of the passage alone. A tool that generates once and caches forever will be wrong for nine weeks out of ten.

The unconstrained passage climbs to 70 percent and stops, held back by seven words that no amount of phonics teaching resolves: beautiful, farmers, harvest, mountain, near, pleasant, weather. A prompt that says "write a short story about a harvest for grade 2" produces exactly this register, and it reads perfectly to the adult who wrote the prompt.

## Which model actually fits on the device

The cohort is being asked for small-model tooling, and the device budget decides most of the engineering. Gemma 4 is the current open-weight line: Apache 2.0, released in five sizes (E2B, E4B, 12B, 26B A4B and 31B), with audio input on the smaller three, 128K context on the small models and 256K on the medium ones, and a separate speculative-decoding draft model bundled with each.

The memory table in Google's documentation is the number to plan against, since it already includes a 20 percent overhead for runtime:

| Build | Gemma 4 E2B | Gemma 4 E4B |
|-------|-------------|-------------|
| BF16 (16-bit) | 11.4 GB | 17.9 GB |
| SFP8 (8-bit) | 5.7 GB | 8.9 GB |
| Q4_0 (4-bit) | 2.9 GB | 4.5 GB |
| Mobile build (LiteRT-LM) | 1.1 GB | 2.5 GB |
| Mobile build, text only | 0.84 GB | 2.2 GB |

Read that table as a device filter. A 3 GB Android tablet with roughly 1.3 GB free after the operating system cannot load the 4-bit E2B build at 2.9 GB, and the 0.84 GB text-only mobile build is the only variant with room for the app itself. A 4 GB device with about 2 GB free can just hold the text-only E4B at 2.2 GB and nothing else. The 12B at 6.7 GB in Q4 is a laptop or a server, not a shared classroom tablet, whatever the context-length headline says.

Two details in the docs keep this honest. Those figures cover static weights only and exclude the KV cache, which grows with prompt and response length, so a 128K context window is a capability claim rather than a device budget. And the E2B and E4B "E" stands for effective parameters: per-layer embeddings keep the compute small while the resident embedding tables make the loaded footprint larger than the effective parameter count suggests. The number that matters on a 3 GB tablet is the one in the table, not the one in the name.

The practical consequence for a cohort targeting large, resource-constrained classrooms: a text-only mobile build of the smallest model, short prompts, and retrieval of curriculum content rather than pasting it into context. Anything that needs the vision or audio path moves the memory floor from 0.84 GB to 1.1 GB and needs a device class above the cheapest tablets in the room.

## Two dose numbers, one lesson

The trial reports a per-hour effect and a threshold effect, and it is tempting to treat the second as a bigger version of the first. It is not, and the arithmetic shows why.

{% raw %}
```python
ITT = 0.258            # intent-to-treat, SD
CI_ITT = (0.027, 0.488)
HOURS_MEAN = 15.0      # treatment classrooms averaged ~15 hours of uptake
TOT_12H = 0.380        # treatment-on-the-treated at the 12-hour threshold
CI_TOT = (0.040, 0.719)
PER_HOUR = 0.016       # reported slope, SD per additional hour
REQ_HOURS = 12
WEEKS = 9

slope_itt = ITT / HOURS_MEAN
slope_tot = TOT_12H / REQ_HOURS

print(f"requested dose: {REQ_HOURS} h over {WEEKS} weeks = {REQ_HOURS * 60 / WEEKS:.0f} min/week")
print(f"                          = {REQ_HOURS * 60 / WEEKS / 40:.1f} forty-minute lessons/week")
print()
print(f"{'hours':>5} {'from ITT slope':>15} {'from 12h TOT':>13} {'reported slope':>15}")
for h in (0, 3, 6, 12, 15, 24):
    print(f"{h:>5} {slope_itt * h:>15.3f} {slope_tot * h:>13.3f} {PER_HOUR * h:>15.3f}")
print()
print(f"slope implied by ITT / mean hours : {slope_itt:.4f} SD per hour")
print(f"slope implied by 12h treated effect: {slope_tot:.4f} SD per hour")
print(f"reported per-hour slope            : {PER_HOUR:.4f} SD per hour")
print(f"ratio of the two implied slopes    : {slope_tot / slope_itt:.2f}x")
print(f"ITT slope within 10% of reported slope: {abs(slope_itt - PER_HOUR) / PER_HOUR < 0.10}")
print(f"12h slope within 25% of reported slope: {abs(slope_tot - PER_HOUR) / PER_HOUR < 0.25}")
print()
print(f"ITT effect 95% CI: {CI_ITT}  width {CI_ITT[1] - CI_ITT[0]:.3f} SD")
print(f"12h TOT 95% CI   : {CI_TOT}  width {CI_TOT[1] - CI_TOT[0]:.3f} SD")
print(f"CI widths overlap: {CI_ITT[1] > CI_TOT[0] and CI_TOT[1] > CI_ITT[0]}")
```
{% endraw %}

```text
requested dose: 12 h over 9 weeks = 80 min/week
                          = 2.0 forty-minute lessons/week

hours  from ITT slope  from 12h TOT  reported slope
    0           0.000         0.000           0.000
    3           0.052         0.095           0.048
    6           0.103         0.190           0.096
   12           0.206         0.380           0.192
   15           0.258         0.475           0.240
   24           0.413         0.760           0.384

slope implied by ITT / mean hours : 0.0172 SD per hour
slope implied by 12h treated effect: 0.0317 SD per hour
reported per-hour slope            : 0.0160 SD per hour
ratio of the two implied slopes    : 1.84x
ITT slope within 10% of reported slope: True
12h slope within 25% of reported slope: False

ITT effect 95% CI: (0.027, 0.488)  width 0.461 SD
12h TOT 95% CI   : (0.04, 0.719)  width 0.679 SD
CI widths overlap: True
```

The per-hour slope reproduces from the intent-to-treat estimate: 0.258 SD divided by the 15 hours treatment classrooms averaged is 0.0172, within 8 percent of the 0.016 the report fits. The threshold estimate implies 0.0317 per hour, nearly twice that. Both are correct as estimands and they answer different questions. Intent to treat averages over everyone in a treatment classroom, including students who used the tool for four hours. Treatment on the treated at the 12-hour threshold conditions on students who completed the dose, which selects for classrooms where implementation worked.

So the threshold number is not a steeper dose-response curve, and multiplying 0.0317 by 24 should not become a plan. For planning within the observed range, use about 0.016 SD per hour and treat it as a bound, not a forecast: 12 hours lands near 0.19 SD, and the trial's own confidence interval on that per-hour slope runs from 0.002 to 0.031, a factor of fifteen.

One organisational number is more useful than either. Twelve hours over nine weeks is 80 minutes a week, or two forty-minute lessons. A tool that needs a whole period per session asks a school to give up twice that.

## What to watch when the artifacts land

Four things decide whether this cohort produces something reusable:

- **The eval suite.** A published suite with items, rubrics and per-language coverage is the difference between a report and a public good. Watch for conditional coverage: items per grade, per language, and per task type, not a single aggregate score.
- **The licence on the models.** Gemma 4 is Apache 2.0, which permits redistribution and fine-tuning without the resale carve-outs that older Gemma terms and the current Mistral weights carry. Confirm it for whatever checkpoint the cohort actually ships.
- **Usage logs, not satisfaction scores.** The Sierra Leone result is a dose-response finding. A pilot that reports teacher approval but not hours of use cannot be compared to it.
- **The reading claim.** If early-reading tools are in scope, decodability against the national scope and sequence is the minimum defensible quality check, and it is cheap to run.

## How to apply this

If you are building for a cohort like this one, the ordering below saves redesigns.

1. **Pick the model from the device, not the benchmark.** Start with the free-RAM figure on the tablet the programme will actually buy, then choose the build. On a 3 GB device that is a 0.84 GB text-only mobile build, and every other decision follows from it.
2. **Run the public education benchmarks before writing your own eval.** Fab AI's pedagogy, multilingual pedagogy, linguistic knowledge and decodability sets exist to be reused, and they are aimed at the exact failure modes of the one cohort being funded.
3. **Measure the property your user needs, not the property that is easy to log.** For a reading tool that is the share of generated words a child at week N can decode. For a lesson planner it is curriculum alignment against the national syllabus, checked item by item.
4. **Log dose.** Minutes of use per student per week, with attendance corroboration, is the variable that predicted outcomes in the only trial run in this class of tool.
5. **Plan for the bottom of the class.** The interaction term says the gains concentrated where baseline proficiency was higher. Pair the tool with teacher-led grouping, or measure the bottom quartile separately and publish it.
6. **Budget the teacher's time honestly.** Five to six hours of training per teacher preceded the trial, and both arms got it. A tool that arrives without that hour budget is a different intervention with unknown effects.

## Key takeaways

| Claim | What the source supports |
|-------|--------------------------|
| A funded cohort will build Gemma-based classroom tools in Kenya, Ghana and South Africa | Announced 4 November 2026, first phase targeted at 500 teachers and 15,000 students |
| The evidence base is a randomised trial in Sierra Leone, not the cohort itself | 48 classrooms, 1,763 students, +0.258 SD intent to treat, published in a DeepMind technical report |
| Gains scaled with use, on a wide interval | +0.016 SD per hour, 95% CI [0.002, 0.031] |
| The equity direction is unfavourable | +0.195 SD treatment effect per baseline SD of maths, 95% CI [0.074, 0.315] |
| The device sets the model size | 0.84 GB text-only mobile build for a 3 GB tablet, 2.9 GB for the 4-bit build of the same model |
| Decodability is measurable before shipping | A plausible generated passage scored 70 percent at week 8 against a 90 percent target |

## References

- Google Africa blog, Empowering Educators and Learners in Sub-Saharan Africa (4 November 2026) — https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/empowering-educators-and-learners-in-sub-saharan-africa-2026/
- Google, We're providing funding for organizations building AI for teachers (3 November 2026) — https://blog.google/products-and-platforms/products/education/google-org-funding-ai-education/
- Google DeepMind, Teaching with Gemini: Measuring the impact of Guided Learning on student mathematics progress in Sierra Leone, technical report — https://storage.googleapis.com/deepmind-media/LearnLM/learnLM_sierraleone_may26.pdf
- Google DeepMind, Rapid evidence for AI in education: an RCT playbook — https://storage.googleapis.com/deepmind-media/LearnLM/learnLM_sierraleone_playbook_jun26.pdf
- Google DeepMind blog, Gemini's guided learning: results from a randomized controlled trial in Sierra Leone — https://deepmind.google/blog/measuring-the-impact-of-learning-with-ai-in-sierra-leone-and-beyond/
- Stanford SCALE, repository entry for the Sierra Leone trial (N = 1763, 48 classrooms) — https://scale.stanford.edu/ai/repository/teaching-gemini-measuring-impact-guided-learning-student-mathematics-progress-sierra
- Google, Gemma 4 model card (Apache 2.0, five sizes, context windows, modalities) — https://ai.google.dev/gemma/docs/core/model_card_4
- Google, Gemma models overview, inference memory requirements table — https://ai.google.dev/gemma/docs/core
- Fab AI, AI benchmark leaderboards for education in LMICs — https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/benchmarks/about
- Fab AI, organisation overview — https://www.fab-ai.org/

## Related posts

- [Small AI by design: ten Kenyan startups and the offline-first constraint](/posts/kenya-ai-academy-offline-first/) — the device and connectivity constraints that decide what ships in Kenyan classrooms.
- [The Tokenizer Was the Point: Measuring MORENA, an Open 1.5B Model for 12 African Languages](/posts/morena-open-african-language-model-tokenizer/) — what language coverage costs before it reaches a model this small.
- [How Much Human Support Does AI Advice Need? 605 Weighed Harvests in Kenya and Uganda](/posts/human-support-dosage-ai-farm-advice/) — the same dose question, asked of advisory services.
- [635 gigabytes of free: what a trillion-parameter open release costs to own](/posts/open-weights-download-cost/) — why open weights are a capital question as much as a licence question.
