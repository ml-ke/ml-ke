---
title: "Satellites Guess, Agents Confirm: Nigeria's Crop Map Starts With Ground Truth"
date: 2026-11-25 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [agriculture, remote-sensing, ground-truth, nigeria, geospatial, data-quality, smallholders]
image:
  path: /assets/img/cover-naps-nigeria-plot-level-crop-map.webp
  alt: "Satellite pass over a grid of farm parcels, one parcel outlined and confirmed by an agent on the ground, with a five-day refresh ring"
---

## A national system starts with a map, not a model

> **The short version**
> On 16 November 2026 the steering committee for Nigeria's National Agro-Productivity System met for the first time and adopted a Phase One plan whose deliverables run to a sixth-month handover. Two days later the first agents completed training in Dutse, and field observations on rice, wheat and maize are now being collected in 59 communities across 11 of Jigawa State's 27 local government areas. Satellite imagery proposes what is planted; agents confirm it on the ground. The design target is a plot-level picture refreshed every five days from around January 2027, with a validation report in month five.

The interesting thing about NAPS is not that a government bought satellite analytics. It is the order of operations. The first deliverable, due in month two, is a baseline land-cover map of one state, not a national yield forecast. The second is a crop map. The five-day monitoring cadence that the announcement leads with does not begin until roughly month three. Someone sequenced this the way a working remote-sensing team would sequence it, and the schedule says so in public.

That sequencing is worth reading closely, because it is also the part most likely to slip. This post walks through what has started, what the published milestones commit to, and what the arithmetic of 25 agents and 59 communities can and cannot establish. It is a different question from the one asked in [Seven Countries, One Set of Weights](/posts/crop-model-seven-countries/), which was about carrying a trained crop classifier across borders. Here the model is the cheap part. The labels are the project.

## What started

The system launched in July 2026 as a memorandum of understanding, and the reporting since then has been consistent about the partners and the structure. What changed this month is that field work began.

| Item | As published |
|---|---|
| System | National Agro-Productivity System (NAPS) |
| Coordinating body | Presidential Food Systems Coordinating Unit (PFSCU), chaired by Vice President Kashim Shettima |
| Technical partners | OCP Africa and Ground Truth Analytics |
| MoU | Signed 17 July 2026 at Mohammed VI Polytechnic University, Ben Guerir, Morocco, by Sen. Ibrahim Hassan Hadejia for the federal government |
| Stated MoU scope | Technology localisation, dataset calibration, ground-truthing, institutional capacity-building, knowledge transfer |
| Pilot state | Jigawa, with 11 of its 27 local government areas covered |
| Crops under observation | Rice, wheat, maize |
| Field scope | 59 communities across the state's four agricultural zones |
| Field agents | 25 trained extension agents, coached by OCP Africa agronomists |
| Refresh target | Every five days, from approximately January 2027 |
| Phase 1 window | November 2026 to April 2027, the dry season |
| Next committee milestone | 31 January 2027 |
| Original deployment plan | Phased across 15 priority states before wider expansion |

The steering committee is not a single ministry. It seats the Federal Ministry of Agriculture and Food Security, the Federal Ministry of Water Resources and Sanitation, the National Space Research and Development Agency, and the Nigeria Governors' Forum, alongside one state per geopolitical zone: Jigawa for the North-West, Borno for the North-East, Plateau for North-Central, Ekiti for South-West, Ebonyi for South-East and Cross River for South-South. Putting the space agency at the table is the structural signal that imagery access is meant to sit inside government rather than arrive as a vendor subscription.

One date in the reporting does not line up cleanly. Punch, writing on 17 November, describes 25 trained agents as already deployed on 16 November. Leadership and Independent both date the Dutse training to 18 November. The pilot is under way either way, and the sequencing ambiguity is a small reminder that a press release is not a project plan.

## The gap it targets: data that arrives after the season

The PFSCU's own account of the problem is unusually blunt for a government statement. Executive Secretary Marion Moon described monitoring during the season as the unit's biggest weakness: "We reach the end of a season and find we did not hit the targets we saw at the beginning." Her framing of what the system is for is narrower and more useful than "AI for agriculture" suggests: if a farmer said they were going to plant maize, did they actually plant maize, and where.

That is a label-lag problem before it is a modelling problem. A system trained on last season's ground truth forecasts this season's harvest using a picture of intentions that was never checked. Nigeria declared a food-security emergency in July 2023, and ThisDay's account of the launch notes that food inflation had climbed above 40 per cent year on year at the height of the crisis in early 2025. When the reserve, import and export decisions are being made on declarations that nobody verified, the error compounds at exactly the moment it is most expensive.

In-season plot-level monitoring is also harder than it sounds, for reasons that have nothing to do with model choice. Rainy-season cloud cover removes optical imagery for stretches of weeks. A smallholder field of a fraction of a hectare covers only tens of pixels at the 10-metre resolution of the freely available Sentinel-2 bands, so boundaries are mixed pixels by construction. Mixed cropping and staggered planting dates mean one pixel can be a blend of two crops at two growth stages. Any of those alone is enough to make a national crop-area estimate wobble.

## The sequence matters more than the model

The published milestones read like a project plan, which is why they are worth tabulating. Each one is a checkpoint that can fail visibly.

| Month | Deliverable | What it proves |
|---|---|---|
| 1 | Field agents mobilised, observations collected | The field layer exists and can be paid for |
| 2 | Baseline land-cover map of Jigawa | The cultivated mask is defensible |
| 3-4 | First crop maps produced | Crops can be separated within the mask |
| ~3 onward | Five-day monitoring begins | The cadence is operationally real, not aspirational |
| 5 | Yield data collection and a validation report | Someone measured the error |
| 6 | Full platform and Phase 1 report to the committee | The system can be handed over |

Land cover before crop type is the right order and it is routinely skipped. A crop classifier is only as good as the mask it runs inside: if fallow land is classified as cultivated, crop area inflates, and every downstream number inherits the error. The mask is also the cheapest thing to validate, because "is this parcel farmed at all" is a question an agent can answer from a motorcycle with a phone.

The month-five validation report is the deliverable to watch. A pilot that produces maps but no error estimate has produced a demo. A pilot that publishes precision and recall on a held-out set of agent-verified plots has produced an instrument.

## What 25 agents and 59 communities can verify

The scale figures are worth doing arithmetic on, because they bound what the pilot can claim. Every input below is from the published statement as reported by Punch, Leadership and Independent.

```python
from datetime import date

# Figures as published by the PFSCU statement, via Punch / Leadership / Independent
LGA_TOTAL, LGA_PILOT = 27, 11
COMMUNITIES, AGENTS, CROPS = 59, 25, 3
REFRESH_DAYS = 5
TRAINING = date(2026, 11, 18)   # extension agents trained in Dutse
MILESTONE = date(2027, 1, 31)   # next Steering Committee milestone

cycles = (MILESTONE - TRAINING).days / REFRESH_DAYS
visits_per_day = COMMUNITIES / REFRESH_DAYS
visits_per_agent_day = visits_per_day / AGENTS

print(f"Phase 1 LGA coverage: {100 * LGA_PILOT / LGA_TOTAL:.1f}% of {LGA_TOTAL} LGAs")
print(f"Communities per agent: {COMMUNITIES / AGENTS:.2f}")
print(f"Five-day cycles before 31 Jan 2027: {int(cycles)}")
print(f"One full community sweep per cycle: {visits_per_day:.1f} visits/day")
print(f"Per-agent load for that sweep: {visits_per_agent_day:.2f} visits/day")
print(f"Annual refreshes at a five-day cadence: {int(365 / REFRESH_DAYS)}")
print(f"Sparse audit, not a census: {COMMUNITIES / AGENTS < 3}")
print(f"Phase 1 proves a national rate: {LGA_PILOT / LGA_TOTAL > 0.5}")
```

```text
Phase 1 LGA coverage: 40.7% of 27 LGAs
Communities per agent: 2.36
Five-day cycles before 31 Jan 2027: 14
One full community sweep per cycle: 11.8 visits/day
Per-agent load for that sweep: 0.47 visits/day
Annual refreshes at a five-day cadence: 73
Sparse audit, not a census: True
Phase 1 proves a national rate: False
```

Two readings follow, and both matter for how the pilot should be judged.

The first is that 25 agents cannot enumerate 59 communities every five days at meaningful density. A full sweep works out to about 0.47 visits per agent per day, which is plausible as a call round, but it is a sparse sample of a state, not a census of it. Phase 1 covers 41 per cent of Jigawa's local government areas, so any number it produces is a Jigawa dry-season number. It cannot be scaled to a national rate by assertion.

The second reading is that sparse sampling is still enough, if it is designed rather than improvised. You do not need to visit every field to estimate a classifier's error rate; you need a sample stratified by agro-ecological zone, by crop, and by the model's own confidence, with the disagreements oversampled because that is where the information is. A pilot that spends its 59 communities confirming the plots the model found easy has bought a press release. One that spends them on the boundary cases has bought an error bar.

### How many verified plots an error bar costs

The follow-up question is how much field work an error bar actually costs, and that one is answerable with the standard library. A proportion estimated to within a few percentage points needs a few hundred independent observations, before any clustering penalty.

```python
import math

Z = 1.96                    # 95% confidence
P = 0.5                     # worst-case variance for a proportion
COMMUNITIES = 59
ZONES, CROPS = 4, 3
PLOTS_PER_COMMUNITY = 12

def n_for(margin, p=P, z=Z):
    return math.ceil(z**2 * p * (1 - p) / margin**2)

strata = ZONES * CROPS
capacity = COMMUNITIES * PLOTS_PER_COMMUNITY
stratified = n_for(0.07) * strata

for m in (0.07, 0.05, 0.03):
    print(f"Verified plots for +/-{m*100:.0f} pp at 95%: {n_for(m)}")

print(f"Strata if crossed by zone and crop: {strata}")
print(f"Communities per stratum: {COMMUNITIES / strata:.1f}")
print(f"Plots available at {PLOTS_PER_COMMUNITY} per community: {capacity}")
print(f"Design effect of 1.5 pushes +/-5 pp to: {math.ceil(n_for(0.05) * 1.5)} plots")
print(f"Reaches +/-5 pp statewide: {capacity >= n_for(0.05)}")
print(f"Reaches +/-3 pp statewide: {capacity >= n_for(0.03)}")
print(f"Plots per community for +/-7 pp in all strata: {math.ceil(stratified / COMMUNITIES)}")
```

```text
Verified plots for +/-7 pp at 95%: 196
Verified plots for +/-5 pp at 95%: 385
Verified plots for +/-3 pp at 95%: 1068
Strata if crossed by zone and crop: 12
Communities per stratum: 4.9
Plots available at 12 per community: 708
Design effect of 1.5 pushes +/-5 pp to: 578 plots
Reaches +/-5 pp statewide: True
Reaches +/-3 pp statewide: False
Plots per community for +/-7 pp in all strata: 40
```

The reading is narrow but useful. At 12 verified plots per community, the pilot's 59 communities support roughly 708 reference plots, which is enough for a single statewide estimate at plus or minus 5 points and not enough for plus or minus 3. Crossing four agricultural zones with three crops produces 12 strata; at that 708-plot budget, a per-stratum estimate at plus or minus 7 points is out of reach, and reaching it needs about 40 plots per community instead of 12. None of this is a flaw in the plan, because the plan does not yet promise stratified error bars. It is the arithmetic that decides which claims the month-five validation report can make, and it is cheaper to settle now than after the report is written.

There is a third reading the press coverage skips. Agent observations are the reference standard here, so the reference has its own error rate. Two agents scoring the same plot, on a subset, turns "the model is 88 per cent accurate" into "the model agrees with trained observers 88 per cent of the time, and the observers agree with each other 91 per cent of the time." The second sentence is the one a bank can act on. The mechanics of that second number are the same ones in [Agreement Before Accuracy: Auditing the Labels You Are About to Train On](/posts/annotator-agreement-before-you-train/).

## The dry season is the tutorial, the rainy season is the exam

Phase 1 runs from November to April, which is the kindest possible window for an optical satellite system in northern Nigeria. Skies are largely clear, so a five-day refresh can actually be met. Irrigated wheat and rice sit in discrete blocks with relatively uniform planting dates, which keeps parcels separable at 10 metres. Phenology is compressed and legible: a wheat field that has been harvested looks nothing like one that has not.

The rainy season is a different problem. Persistent cloud removes optical passes for stretches long enough that a five-day cadence simply cannot be honoured from Sentinel-2 alone. Farmers interplant, stagger planting across weeks, and leave fields fallow in patches, so a single parcel can carry two crops at two growth stages and a fallow strip. Weeds and volunteer crops confuse the spectral signature at exactly the point where the classifier has the least label support. Radar imagery sees through cloud, but it is coarser and harder to read for crop type, which is why operational systems fuse it with optical rather than replacing one with the other.

None of this makes the dry-season pilot pointless. It makes it a tutorial. A team that cannot get a state-level dry-season map right will not get a rainy-season one right either. But the national question, which is whether Nigerian staple production can be tracked in-season rather than reconstructed afterwards, is a rainy-season question, and the pilot's results will not answer it. The 31 January 2027 milestone and the month-five validation report are the points at which it becomes reasonable to ask for a rainy-season trial.

## Why a verified planting record is worth more than a yield map

The PFSCU statement lists the benefits in a deliberate order: land-use planning, irrigation targeting, input distribution, food supply planning, early shortage warning, trade and research policy, and then the one that pays for the system. Verified records of what a farmer has planted, the statement argues, give banks the confidence to lend and insurers the evidence to provide cover.

That is the durable product. A yield forecast is consumed once a season and is wrong often enough that nobody builds a credit product on it. A parcel-level record of who farms what, where, and since when is collateral infrastructure. It is also the harder thing to build, because it requires the parcel-to-owner link that the system's technical partner described in Punch's account: knowing the who, the crop, the location, and the phenological stage across the season.

The parcel-to-owner link is also personal data, and the same table that makes a farmer bankable makes them traceable. Any national registry that wires verified holdings into lending and insurance needs an access-control design before the first lender is connected, not after. That is a policy question, and it is the one place where the pilot's institutional breadth is an advantage: the Governors' Forum and six state governments are already in the room.

## What is not published yet

The honest summary is that NAPS has published a plan and a schedule, not a result.

- **No accuracy figures.** No precision, recall, confusion matrix, or area error estimate has been released for any crop, in any local government area.
- **No independent evaluation.** Every figure in this post traces to the PFSCU statement as reported by Nigerian outlets. There is no third-party audit and no technical report.
- **One state, one season, three crops.** Rice, wheat and maize in the dry season. Rainy-season performance, which is the harder case because of cloud cover and crop mixing, is untested.
- **A 15-state plan with 11 LGAs in hand.** The July launch described phased deployment across 15 priority states. Field work today is confined to 11 of 27 local government areas in one of them.
- **A cadence target, not a cadence measured.** Five-day refresh is scheduled from around January 2027. Nobody has yet shown the pipeline sustaining it.
- **No cost per hectare.** Which decides whether this is a system Nigeria can run nationally or a well-instrumented pilot.

None of that is a criticism of the pilot. It is the normal state of a programme in month one, and publishing the milestone dates is what makes the eventual claims checkable. The mistake would be reading the launch coverage as evidence that the hard part is done.

## How to apply this

If you are building a monitoring programme anywhere with smallholders and thin public data, the pilot's structure transfers even where the politics do not. Six practices worth copying:

| Practice | Why it survives a real season |
|---|---|
| Publish the label schema before the model | "Cultivated" and "planted" mean different things to an agronomist, a lender and a tax office |
| Baseline before classification | A crop classifier inherits every error in the mask beneath it |
| Budget the field layer as an error instrument | The point of a visit is to measure the model, not to fill a database |
| Stratify the sample and oversample disagreement | Error estimates need the hard plots, not the easy ones |
| Double-code a share of ground truth | A reference standard with unknown error cannot validate anything |
| Write the validation date into the plan | A report nobody scheduled is a report nobody writes |

The cadence question deserves its own line. Seventy-three refreshes a year sounds like an engineering target. In practice it is a budget line: fuel, per diems, phone data, and the agronomist time to coach the agents. The pilot's six-month schedule is the right way to find out the real number before committing to a national one.

## Key takeaways

| Takeaway | Detail |
|---|---|
| What launched | A six-month, one-state pilot of in-season, plot-level crop monitoring, with field work under way in Jigawa |
| Why the order matters | Baseline land cover precedes crop classification, and a validation report precedes any national claim |
| What the scale bounds | 25 agents over 59 communities is a stratified audit, not a census; 41 per cent of one state's LGAs |
| Where the value sits | A verified parcel-to-owner record supports credit and insurance in a way a yield forecast does not |
| What is missing | Accuracy figures, independent evaluation, rainy-season results, cost per hectare |
| The date to watch | 31 January 2027 for the next milestone, month five for the validation report |

## References

- Punch, "FG deploys AI to track Jigawa crops, farmland", 17 November 2026 — https://punchng.com/fg-deploys-ai-to-track-jigawa-crops-farmland/
- Leadership, "Nigeria deploys AI, satellites to monitor farmland as agricultural intelligence system begins in Jigawa", November 2026 — https://leadership.ng/nigeria-deploys-ai-satellites-to-monitor-farmland-as-agricultural-intelligence-system-begins-in-jigawa/
- Independent, "Nigeria puts satellites, artificial intelligence to work for its farmers as NAPS takes off in Jigawa", 24 November 2026 — https://independent.ng/nigeria-puts-satellites-artificial-intelligence-to-work-for-its-farmers-as-naps-takes-off-in-jigawa/
- Blueprint, "FG launches AI-powered satellite crop monitoring system, targets 15 states", 19 July 2026 — https://blueprint.ng/fg-launches-ai-powered-satellite-crop-monitoring-system-targets-15-states/
- ThisDay, "FG unveils AI-powered crop surveillance system to tackle food insecurity", 19 July 2026 — https://www.thisdaylive.com/2026/07/19/fg-unveils-ai-powered-crop-surveillance-system-to-tackle-food-insecurity/
- The Nation, "Fed govt unveils AI-powered crop surveillance system to tackle food insecurity", July 2026 — https://thenationonlineng.net/fed-govt-unveils-ai-powered-crop-surveillance-system-to-tackle-food-insecurity-2/
- Copernicus Sentinel-2 mission specifications, 10 m visible and near-infrared bands — https://sentinel.esa.int/web/sentinel/missions/sentinel-2

## Related posts

- [Seven Countries, One Set of Weights: What 96% Has to Survive at the Border](/posts/crop-model-seven-countries/)
- [Four Times the Fields: The Delivery Stack Behind the Google–Gates Farmer AI Plan](/posts/gates-google-farmer-ai-stack/)
- [How Much Human Support Does AI Advice Need? 605 Weighed Harvests in Kenya and Uganda](/posts/human-support-dosage-ai-farm-advice/)
- [Agreement Before Accuracy: Auditing the Labels You Are About to Train On](/posts/annotator-agreement-before-you-train/)
- [AI for Agriculture & Climate in Africa: Predicting the Unpredictable](/posts/ai-africa-agriculture-climate/)
