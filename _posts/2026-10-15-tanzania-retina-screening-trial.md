---
title: "The Model Was Not the Intervention: Reading Tanzania's Retina Screening Trial"
date: 2026-10-15 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [medical ai, diabetic retinopathy, randomised controlled trial, tanzania, ai in africa, clinical validation, health systems, screening, lancet global health]
image:
  path: /assets/img/cover-tanzania-retina-screening-trial.webp
  alt: A dark retinal image at left with vessel branches, an optic disc and small red lesions, a purple tag reading SELENA+ v1.7.0 offline, and two diverging pathways to the right, a cyan one reaching 10.9 days and 35.9 percent attendance and a dashed yellow one looping out to 36.9 days and 18.2 percent
---

## Introduction

> **The trial, in one line**
> On 13 October 2026 *The Lancet Global Health* published a randomised controlled trial of AI-supported diabetic retinopathy screening across 24 hospitals in Tanzania. Among the people that masked UK-certified graders confirmed had referable retinopathy, **157 of 437 (35.9%)** reached the central eye clinic within eight weeks on the AI-supported pathway, against **85 of 467 (18.2%)** on the standard pathway (odds ratio 2.52, 95% CI 1.86 to 3.42).
{: .prompt-info }

That is the number travelling through the health-technology press this week, and it is a good number: a controlled, preregistered, peer-reviewed doubling of the share of high-risk patients who actually reached specialist care. It is also the third time in October that a screening result has landed on this blog, so it is worth being precise about what is new here. [WHO's prequalification of CAD software for TB](/posts/who-prequalification-cad-tb-ai-screening/) was about the procurement gate. [Nigeria's maternal health multiplier](/posts/nigeria-maternal-health-multiplier/) was about adoption arithmetic and the unglamorous device-distribution work behind an "AI-enabled" label. This trial is different from both: it is a randomised experiment whose primary endpoint was not the model's accuracy but whether people showed up.

That distinction is the whole story, and it is easy to lose. Read the paper's own numbers closely and three things appear that the headline hides: the randomised comparison bundled three interventions rather than one, the control arm came in at less than half the attendance the trial's design assumed, and the diagnostic performance that everyone quotes as the reason for AI is filed as a secondary outcome with a specificity that is worse than the human graders'.

## What the trial measured

The primary outcome was attendance at the central ophthalmology clinic within eight weeks of screening, restricted to participants with true referable diabetic retinopathy as judged by the reference standard. The design paper, published in *BMJ Open* two years before the results, is explicit about why that endpoint and not another: these are the people at highest risk of vision loss, and connecting them to eye care is what a screening programme is for.

Everything about the model's accuracy came second. The secondary outcomes were the attendance rate among all those referred, attendance at 16 weeks, the prospective sensitivity and specificity of the AI software, the number of false positives who attended, ungradable results, time to presentation, and the cost-effectiveness, acceptability and fidelity of the intervention.

| Outcome (true referable retinopathy unless stated) | AI-supported | Standard | Effect | p |
|---|---|---|---|---|
| Attended clinic within 8 weeks | 157/437 (35.9%) | 85/467 (18.2%) | OR 2.52 (1.86 to 3.42) | <0.0001 |
| Attended within 8 weeks, all those referred | 234/640 (36.6%) | 126/605 (20.8%) | OR 2.19 (1.70 to 2.82) | <0.0001 |
| Attended within 16 weeks | 171/437 (39.1%) | 110/467 (23.6%) | OR 2.09 (1.57 to 2.78) | <0.0001 |
| False positives attending within 8 weeks | 84/755 (11.1%) | 48/719 (6.7%) | OR 1.75 (1.21 to 2.53) | 0.0031 |
| Final grade ungradable by reference standard | 82/1192 (6.9%) | 109/1186 (9.2%) | OR 0.73 (0.54 to 0.98) | 0.039 |
| Days to presentation (mean) | 10.9 (SD 11.7) | 36.9 (SD 11.0) | HR 2.47 (1.89 to 3.21) | <0.0001 |

A note on that last row before anyone reads it as "AI saw people 26 days sooner". The 8-week attendance figures are the endpoint; the 26-day gap is the mean among the people who did attend. It describes the two pathways, not a causal saving in the way a survival analysis would.

## Two paths, thirty seconds apart

The trial randomised 2,378 people 1:1 using sequentially numbered opaque envelopes and a variable block size of four to eight, and it did not stratify by recruitment site. Because the intervention includes the patient being handed a result in the room, neither participants nor screening staff could be masked. The person collecting the primary outcome at the clinic was masked and had no other role in the trial, which is the right way to protect an outcome that is otherwise self-evident.

What the two arms received is where the interesting engineering sits.

On the AI-supported side, retinal photographers took two 45-degree images per eye on a Topcon NW8 fundus camera after dilation with 1% tropicamide. The image files went to SELENA+ version 1.7.0, made by EyRIS in Singapore, installed on an HP laptop with an Intel Core i3 processor. The software returned non-referable, referable or ungradable in about 30 seconds. If the output was ungradable the photographer tried again and resubmitted. The patient got the result and face-to-face counselling before leaving the photography room.

On the standard-of-care side, the same images were graded by an ophthalmology resident at Kilimanjaro Christian Medical Centre within four weeks, after which the patient was telephoned with the result and counselled on the phone.

So the randomised contrast is not "AI versus no AI". It is a bundle: an immediate result, an automated triage decision, and a counselling conversation that happens while the patient is still in the building. The paper's discussion attributes the effect to how results were delivered, and the point of screening at all is to shorten the distance between detection and treatment. But it means the trial cannot tell you how much of the gain came from the model and how much from the room it was standing in. A cheap alternative design, say a human grader who is simply faster, is untested here.

## The control arm arrived at less than half the assumed baseline

Here is the part most coverage skipped. The trial was powered assuming 42% attendance in the standard-of-care group, drawn from previous programme data, and a 33% relative increase was agreed with Tanzanian partners as the level worth detecting. That put the design target at 56%.

The observed standard-of-care attendance was 18.2%.

The AI-supported arm reached 35.9%, which is a large and statistically unambiguous effect, and also 6.1 points below the level the programme believed it was already achieving before any AI was introduced, and 20 points below the target the trial was powered to detect. That is not a criticism of the trial. It is a statement about how much worse the unsupported pathway was than anyone had measured, and it changes what the intervention is worth: AI here is less a leap forward than a partial repair of a follow-up process that was largely failing.

You can rebuild the design arithmetic from the published parameters in a few lines.

```python
import math

z_alpha, z_beta = 1.959964, 1.281552   # two-sided 5%, 90% power
p_soc, p_target = 0.42, 0.56           # the design's assumed baseline and its target

n_group = (z_alpha + z_beta)**2 * (p_soc*(1-p_soc) + p_target*(1-p_target)) / (p_target-p_soc)**2
print(f"needed for the outcome comparison: {math.ceil(n_group)} per group")

print(f"observed attendance, standard arm: {100*85/467:.1f}%  (design assumed {100*p_soc:.0f}%)")
print(f"observed attendance, AI arm:       {100*157/437:.1f}%  (design target {100*p_target:.0f}%)")

referable = 266          # cases with true referable DR needed per group
prevalence = 0.225       # assumed share of those screened who have it
per_arm = math.ceil(referable/prevalence)
print(f"referable cases needed per group:  {referable}")
print(f"implied recruitment: {per_arm} per arm, {2*per_arm} total (published: 2364)")
```

Output:

```text
needed for the outcome comparison: 263 per group
observed attendance, standard arm: 18.2%  (design assumed 42%)
observed attendance, AI arm:       35.9%  (design target 56%)
referable cases needed per group:  266
implied recruitment: 1183 per arm, 2366 total (published: 2364)
```

The enrolment figure reproduces to within two participants, and the difference is only rounding: the paper's 1,182 per arm times the assumed 22.5% prevalence gives 265.95 referable cases, which it reports as 266. The useful line is the middle pair. A screen that was assumed to connect 42 in 100 high-risk patients was connecting 18 in 100, and nobody had noticed until a randomised trial measured it.

The same arithmetic is a warning about effect sizes. With the observed control rate of 18.2% instead of the assumed 42%, the study retained ample power for the effect it found, because the effect was much larger than the design's minimum. The reverse case, a smaller true effect against a low baseline, is where trials like this lose their ability to detect anything, and it is the reason measuring your own baseline before a trial or a procurement decision matters more than the vendor's benchmark table.

## The accuracy numbers are secondary, and they are honest

The diagnostic performance was measured against UK-certified reference grading on all 2,378 participants, for both the AI software and the Tanzanian graders, even in the standard-of-care arm where the AI result was not used clinically.

| Detection of referable retinopathy | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|
| AI software, all participants (n=2,378) | 91.6% (90.5 to 92.7) | 67.2% (65.3 to 69.1) | 63.1% | 92.9% |
| AI software, excluding ungradable (n=2,187) | 91.4% (90.3 to 92.6) | 67.2% (65.2 to 69.1) | 57.4% | 94.2% |
| Tanzanian graders (n=2,352) | 85.0% (83.5 to 86.4) | 85.8% (84.4 to 87.2) | 78.7% | not reported |

Read those two rows against each other and the AI is not the better clinician. It is the more sensitive one. It missed fewer cases, 91.6% against 85.0%, and referred considerably more people who turned out not to need referral, with a specificity of 67.2% against 85.8% and a positive predictive value of 63.1% against 78.7%. That is the trade the trial bought, and for a screening gate it is usually the trade you want: for a treatable cause of blindness, a missed case is worse than a wasted appointment.

The number that matters most for that argument is the sight-threatening end of the distribution. Of 131 participants with proliferative diabetic retinopathy, the AI system correctly identified 130, a sensitivity of 99.2% (98.9 to 99.6). The Tanzanian graders found 128 of the same 131. Almost everything the AI missed sat in the mild and moderate bands, where a 12-month re-screen is a reasonable answer.

Two operational side effects are worth noting. Ungradable results fell from 9.2% to 6.9%, because the workflow allowed an immediate rephotograph and resubmission rather than sending an unreadable image off to be graded. And the extra referrals are real work: 84 of 755 false positives, 11.1%, turned up at the clinic within eight weeks, against 48 of 719, 6.7%, in the standard arm. False positives do not disappear into the database. They occupy appointment slots.

## The advantage is not just a head start

One tempting explanation for the doubling is pure speed: the AI arm started its eight-week clock 26 days earlier in practice, so more of its attendance had time to land inside the window. If that were the whole mechanism, the AI arm's advantage would decay once both arms had a long enough window.

It does not. Attendance at 16 weeks was 39.1% against 23.6%, an odds ratio of 2.09, only slightly smaller than the 2.52 at eight weeks. The gap persists and the AI arm keeps converting a higher share of its referrals. A simple constant-hazard fit to the four published attendance points makes the shape visible.

```python
def weekly_hazard(attendance, weeks):
    """Per-week conversion rate that reproduces the observed attendance."""
    return 1 - (1 - attendance) ** (1 / weeks)

arms = {
    "AI-supported": (0.359, 0.391),   # 8-week and 16-week attendance, true referable DR
    "standard":     (0.182, 0.236),
}
early, late = {}, {}
for arm, (a8, a16) in arms.items():
    early[arm] = weekly_hazard(a8, 8)
    late[arm]  = weekly_hazard(a16, 16)
    print(f"{arm:14s} weeks 1-8: {100*early[arm]:5.2f}%/week   weeks 9-16: {100*late[arm]:5.2f}%/week")

print()
print(f"hazard ratio, weeks 1-8:  {early['AI-supported']/early['standard']:.2f}")
print(f"hazard ratio, weeks 9-16: {late['AI-supported']/late['standard']:.2f}")
print(f"share of the AI arm's 16-week attendance that arrived inside 8 weeks: {100*0.359/0.391:.1f}%")
```

Output:

```text
AI-supported   weeks 1-8:  5.41%/week   weeks 9-16:  3.05%/week
standard       weeks 1-8:  2.48%/week   weeks 9-16:  1.67%/week

hazard ratio, weeks 1-8:  2.18
hazard ratio, weeks 9-16: 1.83
share of the AI arm's 16-week attendance that arrived inside 8 weeks: 91.8%
```

Two readings. First, the AI arm's conversion rate is higher than the standard arm's in both windows, 2.18 times higher in weeks one to eight and still 1.83 times higher in weeks nine to sixteen, so the effect is not a head start that flattens out. Second, both arms slow down sharply after eight weeks: conversion falls to 56% and 67% of its early rate. Whatever moves people to a clinic is front-loaded, which is why the eight-week window captures 91.8% of the AI arm's eventual 16-week attendance and why a programme that cannot deliver a result quickly is leaving most of its attendance on the table.

Treat this as a shape, not a reanalysis. It is a single-parameter exponential fit to four aggregate published points, assuming a constant hazard inside each window and no censoring or competing risks. The trial's own protocol specifies proper time-to-event methods, and those are what should be quoted.

## If you are deploying screening AI, copy the selection criteria

The intervention is described in enough detail to be reusable, and four of the choices transfer directly to any deployment outside a trial.

**Pick the system against your constraints, not the leaderboard.** Before the trial the team reviewed commercially available systems against four criteria: regulatory approval, ability to work offline, ability to work with different fundus cameras, and evidence of performance in an African population. SELENA+ was the only device meeting all four at the time of selection, and it was provided free for academic use. Offline operation is the criterion most likely to be skipped and most likely to end a deployment once the pilot funding stops. A model that needs a round trip to a server is not equivalent to one running on the laptop in the photography room.

**Instrument the pathway, not just the model.** The endpoint here was attendance, which required an administrator with access to the hospital's electronic medical record cross-checking every referral. That is unglamorous work and it is what turned a plausible story into a measured one.

**Budget the referrals you are about to create.** Both the false-positive rate and the doubled attendance land on the specialist clinic. The reference clinic in this trial had cameras, optical coherence tomography, laser and anti-VEGF treatment and vitreoretinal surgery. The authors say outright that improved screening needs to be accompanied by better access to specialist clinics and treatments; a screening programme that outruns its treatment capacity converts earlier detection into longer waiting lists. The Tanzanian Ministry of Health, a collaborator on the study, has committed to integrating AI diagnostics into diabetic eye screening in multiple regions, so that commitment is now the constrained resource to watch.

**Read the sensitivity at the sight-threatening end.** An average sensitivity across all grades hides the case you care about. The 99.2% figure for proliferative retinopathy, 130 of 131, is the number to hold a vendor to, and it is computed on a subgroup of 131 people, so demand the denominator.

## Key takeaways

| Point | Detail |
|---|---|
| The headline result | Attendance within 8 weeks among people with true referable retinopathy rose from 18.2% (85/467) to 35.9% (157/437); OR 2.52 (1.86 to 3.42) |
| The endpoint was a pathway metric | Primary outcome was clinic attendance, not model accuracy; accuracy was a secondary outcome |
| The intervention was a bundle | Immediate result, automated triage, and face-to-face counselling in the photography room, against a deferred telephone result |
| The baseline was the surprise | The design assumed 42% standard-arm attendance; the trial measured 18.2%, and the AI arm's 35.9% stayed under the 56% the trial was powered for |
| Accuracy traded specificity for sensitivity | AI sensitivity 91.6% versus 85.0% for graders, specificity 67.2% versus 85.8%; 130 of 131 proliferative cases found by AI |
| The effect is not only speed | The 16-week gap persists (39.1% versus 23.6%), and 91.8% of the AI arm's eventual attendance arrived inside eight weeks |
| What remains unproven | The trial was not powered for treatment uptake or visual outcomes, so sight saved is still an inference rather than a measured result |

## References

- Cleland CR et al., *Artificial intelligence-supported diabetic retinopathy screening in Tanzania: a randomised controlled trial*, The Lancet Global Health, 2026 — https://www.sciencedirect.com/science/article/pii/S2214109X2600224X
- London School of Hygiene & Tropical Medicine, *AI screening helps more people with diabetes access sight-saving eye care*, 13 October 2026 — https://www.lshtm.ac.uk/newsevents/news/2026/ai-screening-helps-more-people-diabetes-access-sight-saving-eye-care
- Medical Xpress, *AI screening helps more people with diabetes access sight-saving eye care*, 13 October 2026 — https://medicalxpress.com/news/2026-10-ai-screening-people-diabetes-access.html
- Cleland CR et al., *Artificial intelligence-supported diabetic retinopathy screening in Tanzania: rationale and design of a randomised controlled trial*, BMJ Open 2024;14:e075055 — https://bmjopen.bmj.com/content/14/1/e075055
- ISRCTN Registry, trial registration ISRCTN18317152 — https://trialscreen.org/trials/diabetic-retinopathy-artificial-intelligence-supported-complication-diabetes-trial-isrctn18317152
- Head-to-head comparative evaluation of four commercially available AI systems for detecting referable diabetic retinopathy in a Tanzanian population, Diabetes Care 2026;49(8):1451 — https://diabetesjournals.org/care/article/49/8/1451/171992/Head-to-Head-Comparative-Evaluation-of-Four
- Kilimanjaro Christian Medical Centre — https://www.kcmc.ac.tz/
- International Centre for Eye Health, LSHTM — https://iceh.lshtm.ac.uk/

## Related posts

- [Prequalified: WHO Opens Its Procurement List to AI TB Screening Software](/posts/who-prequalification-cad-tb-ai-screening/) — the procurement gate that decides which screening software a ministry can buy
- [The Bottleneck Is Adoption, Not Innovation: Reading Nigeria's AI Maternal Health Multiplier](/posts/nigeria-maternal-health-multiplier/) — the same question asked of a maternal health platform: what actually has to exist for the model to matter
- [AI in African Healthcare](/posts/ai-african-healthcare/) — where imaging and triage tools are already running on the continent
- [Certified in Europe, Judged at the Health Post](/posts/ai-primary-care-class-iib-ce/) — what a regulatory mark does and does not transfer to a rural clinic

**Next in this series:** a look at what changes when the screening model runs on the phone in the photographer's pocket rather than a laptop on the table, and when a country has to maintain a hundred deployments instead of one trial.
