---
title: "Seven Countries, One Set of Weights: What 96% Has to Survive at the Border"
date: 2026-11-06 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [agriculture, computer vision, evaluation, calibration, deployment, model cards]
image:
  path: /assets/img/cover-crop-model-seven-countries.webp
  alt: One identical model weight block on the left of a border line, and seven different disease-mix bar charts on the right
---

## Introduction

> **The weights crossed the border. The priors did not.**
> On 3 November 2026, the UNDP AI Hub for Sustainable Development and DeepLeaf, a Moroccan agritech company, launched an AI foundation model for food security. It is built on a DeepLeaf model already validated in Kenya, Morocco and Italy. In seven countries, local agritech startups will build their own products on that single shared model, with data shared by consent and open components returned as digital public goods.

That is a better class of announcement than a pilot, because it describes a distribution model rather than a demonstration. The part worth reading closely is the word "shared". One set of weights is about to see Tunisian greenhouses, Ghanaian maize plots and Rwandan hillside fields, and each of those is a different mixture of diseases, a different set of phone cameras, and a different stage at which farmers take the photo.

The company reports a field-validated accuracy around 96 percent from a smartphone photo in under three seconds. That number can be entirely true and still be a poor predictor of what the same model does in a country whose dominant disease is different from the one that dominated the validation set. This post reads the announcement, then works out what the shared model's audit has to cover if the number is going to mean anything in seven places at once.

## What actually launched

The convening was called *AI Diffusion Pathways: The Frontier is Collaboration*, held at Luiss Guido Carli University. The food-security model was the concrete artifact it produced, and the hub it belongs to was launched with UNDP during Italy's G7 presidency.

| Item | What the record says | Source |
|------|----------------------|--------|
| Shared base model | An AI foundation model for food security, built on DeepLeaf's African-developed model, already validated in Kenya, Morocco and Italy | UNDP, 3 November 2026 |
| Countries building on it | Italy, Morocco, Tunisia, Kenya, Ghana, Tanzania and Rwanda, where local agritech startups build their own solutions on one shared model | UNDP |
| Data terms | Shared with consent, solutions stay locally owned, open components return as digital public goods | UNDP |
| Hub scale to date | 1,100+ African innovators engaged, 150+ Italian organisations, 7,000+ startups registered, 49 startups in Cyber4Africa across 13 countries | UNDP, quoting Italy's enterprise minister |
| Product claim | Smartphone photo, under three seconds, field-validated accuracy around 96 percent | UNA Boulder, describing the cohort |
| Reach claim | The opened foundational model turns a single product into shared infrastructure that "now reaches millions of farmers" | UNDP, later readout on the same hub |
| Coverage claim | 55+ crops in one catalogue, 800 anomaly types in the training data; 80 crops and 1,000+ disorders in the company's own count | IRCAI; The Next Africa |

The coverage figures disagree, and that disagreement is informative rather than embarrassing. An 800-anomaly training set, a 55-crop shipped catalogue and a 1,000-disorder marketing count are three different objects: training data, a released product surface, and a roadmap. When seven startups inherit one model, which of those three they get determines what they can actually build. That is the first question a local team should ask, and it is answerable from a model card in one line.

A second UNDP readout explains what the hub means by the phrase in the event title. It calls these *AI diffusion pathways*: reusable routes that record how a proven application reaches widespread adoption, so the next adopter starts from what already worked instead of from zero. That is a more useful frame than "scaling", because it names the artifact being shared. The weights are one component of the route. The evaluation, the diagnosis workflow, the confirmation step and the commercial model are the rest, and they are the parts a founder cannot download.

The second number in the same announcement is the one the model's economics rest on. Italy's enterprise minister put the hub's reach at more than 1,100 African innovators and 150 Italian organisations. A UNA Boulder write-up of the hub adds $3 million in compute credits, 1.5 million GPU hours made available, and 25+ new partnerships with G7 governments, universities and companies. Against that, only about 5 percent of African AI innovators have access to the compute they need to scale, and the continent holds less than 2 percent of the world's data-centre capacity. A model that arrives pretrained is one way to spend the missing compute once instead of seven times.

## Three numbers that get conflated

A press release about a diagnostic system usually carries accuracy, latency and class count in the same sentence, and they measure different things.

- **96 percent accuracy** is an aggregate over whatever class mix the validation set contained. It says nothing about the class you care about, and nothing about the base rate in your field.
- **Under three seconds** is a latency budget, and on a phone it depends on image size and whether inference runs on device or server. It is a UX number, not a quality number.
- **1,000+ disorders** is a label space. Every additional near-neighbour class makes the same accuracy harder to hold, because more of the probability mass sits between classes that look alike.

The aggregate is the problem. Suppose the shipped triage is "diseased or healthy" at 92 percent sensitivity and 97 percent specificity, both plausible for a field-validated model. Now apply it to two fields with different disease prevalence. The accuracy barely moves. The precision moves by a factor of two.

## The arithmetic that turns 96% into 38%

```python
def ppv(sens, spec, prev):
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return tp / (tp + fp)

def acc(sens, spec, prev):
    return sens * prev + spec * (1 - prev)

print("published model: sensitivity 0.92, specificity 0.97")
print(f"{'prevalence':>10} {'accuracy':>9} {'precision':>10} {'false alarms per 1,000 flagged':>31}")
for prev in (0.02, 0.05, 0.10, 0.25, 0.50):
    a, p = acc(0.92, 0.97, prev), ppv(0.92, 0.97, prev)
    print(f"{prev:>10.2f} {a:>9.3f} {p:>10.3f} {1000*(1-p):>31.0f}")

lo, hi = acc(0.92, 0.97, 0.02), acc(0.92, 0.97, 0.25)
print(f"\naccuracy across the two mixes: {lo:.3f} to {hi:.3f}")
print(f"precision across the two mixes: {ppv(0.92,0.97,0.02):.3f} to {ppv(0.92,0.97,0.25):.3f}")
```

```
published model: sensitivity 0.92, specificity 0.97
prevalence  accuracy  precision  false alarms per 1,000 flagged
      0.02     0.969      0.385                             615
      0.05     0.968      0.617                             383
      0.10     0.965      0.773                             227
      0.25     0.958      0.911                              89
      0.50     0.945      0.968                              32

accuracy across the two mixes: 0.969 to 0.958
precision across the two mixes: 0.385 to 0.911
```

Read the first and last rows together. At 50 percent prevalence this model looks excellent: 96.8 percent of its flags are correct. At 2 percent prevalence, 61.5 percent of flags are false, while reported accuracy has gone *up* to 96.9 percent. The accuracy number is highest exactly where the tool is least trustworthy, because a mostly-healthy population is easy to classify and the errors concentrate in the small diseased group.

For an advisory product that answers "what is wrong with this plant", a 2 percent prevalence is not unusual for any single disease. That is why the same 96 percent shows up in case studies and in complaints without either side being wrong. They are measuring at different base rates.

The practical consequence for a shared model is direct. A startup in a country where the flagged disease is rarer than in the validation set inherits a precision problem it did not create, and no amount of weight-sharing fixes it. What fixes it is knowing the local mix and recomputing the operating point.

## When the model moves, the priors stay behind

Two different shifts are hidden in the phrase "seven countries".

**Label shift** is a change in the mix of classes. A region with more fungal pressure and less viral pressure moves the base rates the model was trained under. This shift is cheap to handle: estimate the local class frequencies from a sample, then re-weight the model's decisions. No new weights, no fine-tuning.

**Covariate shift** is a change in the images themselves. Different phone sensors, different lighting, a different growth stage at the moment of capture, a different background of soil and trellis. This one needs data, because the model has never seen the condition and no prior correction compensates for a feature it cannot read.

Label shift alone is enough to change which classes are trustworthy. The demo below uses a planted six-class model: each class has one overlapping symptom score, decisions are made by weighting the class densities by a prior, and everything is integrated exactly rather than sampled, so the numbers are deterministic.

```python
import math

classes     = ["healthy", "nutrient", "fungal", "pest", "viral", "water"]
means       = [0.00, 0.45, 0.90, 1.35, 1.80, 2.25]   # planted, not DeepLeaf's weights
sigma       = 0.15
train_prior = [0.40, 0.10, 0.20, 0.15, 0.10, 0.05]   # mix the model was fitted on
local_prior = [0.70, 0.06, 0.10, 0.07, 0.03, 0.04]   # a healthier field, different mix

def confuse(prior):
    """Exact confusion matrix: integrate each class density over the region it wins."""
    def phi(x, m):
        return math.exp(-0.5 * ((x - m) / sigma) ** 2)
    C = [[0.0] * 6 for _ in range(6)]
    step, x = 0.001, -2.0
    while x < 4.0:
        dens = [phi(x, m) for m in means]
        best = max(range(6), key=lambda j: prior[j] * dens[j])
        for i in range(6):
            C[i][best] += dens[i] * step
        x += step
    for i in range(6):
        row = sum(C[i])
        C[i] = [v / row for v in C[i]]
    return C

def report(C, prior, label):
    accuracy = sum(prior[i] * C[i][i] for i in range(6))
    print(f"{label:<24} accuracy {accuracy:.3f}")
    for j, c in enumerate(classes):
        flagged = sum(prior[i] * C[i][j] for i in range(6))
        precision = prior[j] * C[j][j] / flagged
        print(f"  {c:<9} prevalence {prior[j]:.2f}  recall {C[j][j]:.2f}  precision {precision:.3f}")
    return accuracy

base  = confuse(train_prior)
shift = confuse(local_prior)          # same weights, decision boundary moved to the local mix
a1 = report(base,  train_prior, "trained mix")
print()
a2 = report(base,  local_prior, "local mix, as shipped")
print()
a3 = report(shift, local_prior, "local mix, corrected")
print()
print(f"accuracy: {a1:.3f} -> {a2:.3f} -> {a3:.3f} (trained, local, corrected)")
print(f"viral recall after correction: {shift[4][4]:.2f} (was {base[4][4]:.2f} uncorrected)")
print(f"nutrient recall after correction: {shift[1][1]:.2f} (was {base[1][1]:.2f} uncorrected)")
```

```
trained mix              accuracy 0.908
  healthy   prevalence 0.40  recall 0.98  precision 0.963
  nutrient  prevalence 0.10  recall 0.75  precision 0.804
  fungal    prevalence 0.20  recall 0.90  precision 0.890
  pest      prevalence 0.15  recall 0.87  precision 0.869
  viral     prevalence 0.10  recall 0.87  precision 0.873
  water     prevalence 0.05  recall 0.90  precision 0.915

local mix, as shipped    accuracy 0.941
  healthy   prevalence 0.70  recall 0.98  precision 0.987
  nutrient  prevalence 0.06  recall 0.75  precision 0.676
  fungal    prevalence 0.10  recall 0.90  precision 0.885
  pest      prevalence 0.07  recall 0.87  precision 0.882
  viral     prevalence 0.03  recall 0.87  precision 0.774
  water     prevalence 0.04  recall 0.90  precision 0.966

local mix, corrected     accuracy 0.946
  healthy   prevalence 0.70  recall 0.99  precision 0.979
  nutrient  prevalence 0.06  recall 0.66  precision 0.769
  fungal    prevalence 0.10  recall 0.90  precision 0.888
  pest      prevalence 0.07  recall 0.88  precision 0.877
  viral     prevalence 0.03  recall 0.81  precision 0.834
  water     prevalence 0.04  recall 0.94  precision 0.940

accuracy: 0.908 -> 0.941 -> 0.946 (trained, local, corrected)
viral recall after correction: 0.81 (was 0.87 uncorrected)
nutrient recall after correction: 0.66 (was 0.75 uncorrected)
```

The first result is the counter-intuitive one. Accuracy *rose* from 0.908 to 0.941 when the field mix changed, because the new mix is 70 percent healthy and healthy is the class the model reads almost perfectly. Any dashboard tracking the headline would report an improvement. Underneath it, precision on nutrient deficiency fell from 0.804 to 0.676 and precision on the viral class fell from 0.873 to 0.774. The model did not get better. The population got easier.

The second result is that correcting for the mix is not free either. Re-weighting the decisions with the local priors recovered precision on both weak classes, to 0.769 and 0.834, and nudged accuracy to 0.946. It paid for that with recall: viral fell from 0.87 to 0.81 and nutrient from 0.75 to 0.66. For a screening tool whose job is to surface cases for a human to confirm, moving the decision boundary toward precision is a trade you have to choose deliberately, not a correction you apply because it sounds rigorous.

There is also a third effect the simulation cannot show, and it is the one that decides deployments. The local mix has to be estimated from something. If the estimate comes from the same flagged pool the model produced, the correction is fitting the model's own errors, and the precision gain quietly evaporates in the field.

## Choosing an operating point when the errors cost differently

The demo above shows prior correction pushing recall down. That is not an argument against correction, it is an argument for choosing the trade on purpose. A false flag costs an agronomist visit. A missed case costs a harvest, and with a spreading disease it costs the neighbouring farms as well.

The threshold is not part of the weights, and it is where most of the local decision actually lives. Take two overlapping score distributions, one for diseased plants and one for healthy plants, and sweep the cut-off at a 5 percent prevalence.

```python
from statistics import NormalDist

Phi = NormalDist().cdf
prev = 0.05        # one disease class in a mixed field
visit = 400        # KES per agronomist confirmation; an assumption, not a quote

def curve(t):
    sens = 1 - Phi(t - 1.0)      # diseased scores ~ N(1, 1)
    spec = Phi(t + 1.0)          # healthy scores ~ N(-1, 1)
    tp, fp = sens * prev, (1 - spec) * (1 - prev)
    return sens, spec, tp / (tp + fp)

print(f"prevalence {prev:.2f}, disease and healthy scores overlapping")
print(f"{'threshold':>10}{'recall':>8}{'precision':>11}{'flags/1,000':>13}{'confirmation cost':>19}")
for t in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
    sens, spec, prec = curve(t)
    flags = 1000 * (sens * prev + (1 - spec) * (1 - prev))
    print(f"{t:>10.1f}{sens:>8.3f}{prec:>11.3f}{flags:>13.0f}{visit*flags:>17,.0f}")

best = next((-1.0 + i * 0.01 for i in range(301)
             if curve(-1.0 + i * 0.01)[2] >= 0.80))
sens, spec, prec = curve(best)
print(f"\nmost sensitive threshold holding precision at 0.80: {best:.2f}")
print(f"  recall {sens:.3f}, so {100*(1-sens):.1f}% of cases are missed")
print(f"  cost of confirmation per 1,000 plants: KES {visit*1000*(sens*prev+(1-spec)*(1-prev)):,.0f}")
```

```
prevalence 0.05, disease and healthy scores overlapping
 threshold  recall  precision  flags/1,000  confirmation cost
      -0.5   0.933      0.137          340          135,908
       0.0   0.841      0.218          193           77,116
       0.5   0.691      0.353           98           39,216
       1.0   0.500      0.536           47           18,645
       1.5   0.309      0.723           21            8,530
       2.0   0.159      0.861            9            3,686

most sensitive threshold holding precision at 0.80: 1.75
  recall 0.227, so 77.3% of cases are missed
  cost of confirmation per 1,000 plants: KES 5,665
```

At 5 percent prevalence, holding precision at 0.80 requires a threshold that misses 77.3 percent of cases. That is the trap in the "high precision or nothing" instinct: at low prevalence you can buy precision with recall, and the model still reports 96 percent on a balanced benchmark. At the other end, the most sensitive setting catches 93.3 percent of cases and sends 340 of every 1,000 plants for confirmation, which at an illustrative KES 400 a visit is KES 135,908 per 1,000 plants.

Whether that is affordable is a crop-economics question, not a model question. For a high-value greenhouse crop, a few hundred shillings of confirmation against a lost season is obviously worth it. For a low-margin staple field, 340 visits per 1,000 plants is not, and the honest conclusion is that the model has to improve rather than the threshold move. Either way, the threshold and its cost belong in the deployment record, because a recall number without a threshold and a prevalence is not a specification.

## Estimating the local mix without labelling the country

Every correction above needs the local class frequencies, and that is the step teams most often fake. Three routes are available, and each carries a bias worth naming.

| Route | What you do | Bias it introduces |
|-------|-------------|--------------------|
| Expert-confirmed subsample | An agronomist labels a few hundred photos, stratified by site and crop stage | Sampling bias if the photos come from app users, who photograph plants that already look wrong |
| Extension and clinic records | Use scouting reports or plant-clinic logs as the class frequencies | Reporting bias: only diseases someone recognised enter the record at all |
| Model output, iterated | Estimate the mix from predicted labels and re-weight until stable | Circular: the model's errors get baked into the prior that is supposed to correct them |

The first bias is the one that flatters the system. A farmer opens the app because something looks wrong, so the images reaching the model are enriched for disease relative to the field. Prevalence measured on that sample is higher than the field's, and higher prevalence mechanically produces better precision. A product can therefore look well calibrated in its own analytics and underperform in a random field survey, with no bug anywhere in the pipeline.

The fix is unglamorous: a small random sample collected on a fixed protocol, one photo per plant along a transect, no self-selection, stratified by agro-ecology, with the agronomist's diagnosis recorded independently of the model's output. Thirty plants per site is enough to see whether the mix is the one the marketing assumes. It is not enough to validate per-class recall, which is what the next section is about.

## What a local audit costs

The last question is the practical one, and it is where most seven-country plans fail. To say "we validated the shared model locally" you need enough confirmed cases per class to detect a meaningful change.

```python
import math
from statistics import NormalDist

z = NormalDist().inv_cdf
alpha, power = 0.05, 0.80
z_alpha, z_beta = z(1 - alpha / 2), z(power)
p0 = 0.87          # the recall a shared model reports for one disease class

def detectable(n, p=p0):
    """Smallest recall drop away from p that n confirmed cases can detect at 80% power."""
    return (z_alpha + z_beta) * math.sqrt(p * (1 - p) / n)

print(f"{'confirmed cases':>16}{'half-width (95% CI)':>22}{'smallest detectable drop':>26}")
for n in (50, 100, 200, 400, 800):
    hw = z_alpha * math.sqrt(p0 * (1 - p0) / n)
    print(f"{n:>16}{100*hw:>20.1f} pp{100*detectable(n):>24.1f} pp")

target = 0.07
n_needed = math.ceil((z_alpha + z_beta) ** 2 * p0 * (1 - p0) / target ** 2)
print(f"\ncases needed to detect a {100*target:.0f}-point recall drop: {n_needed}")

print(f"{'field prevalence':>18}{'photos screened to find them':>32}")
for prev in (0.03, 0.01, 0.005):
    print(f"{prev:>18.3f}{math.ceil(n_needed / prev):>32,}")

print(f"\nseven separate national audits: {7 * math.ceil(n_needed / 0.03):,} photos")
print(f"one pooled audit, common recall: {n_needed} confirmed cases total")
```

```
confirmed cases   half-width (95% CI)  smallest detectable drop
              50                 9.3 pp                    13.3 pp
             100                 6.6 pp                     9.4 pp
             200                 4.7 pp                     6.7 pp
             400                 3.3 pp                     4.7 pp
             800                 2.3 pp                     3.3 pp

cases needed to detect a 7-point recall drop: 182
  field prevalence    photos screened to find them
             0.030                           6,067
             0.010                          18,200
             0.005                          36,400

seven separate national audits: 42,469 photos
one pooled audit, common recall: 182 confirmed cases total
```

A hundred confirmed cases of one disease gives a 95 percent interval of plus or minus 6.6 points and can only detect a drop larger than 9.4 points. That is roughly the size of every accuracy difference discussed in this post, which means a 100-image pilot cannot confirm or refute a published 96 percent. It can only produce a number that looks like validation.

To detect a 7-point recall drop you need about 182 confirmed cases of that class. At 3 percent field prevalence that is 6,067 photos screened and labelled, per country, for one class. Seven separate national audits of that shape come to roughly 42,469 photos of collection and expert labelling, before anyone has built a product. This is the number that makes the shared-model design rational: a single pooled audit of 182 confirmed cases per class is affordable across a seven-country programme and impossible for one startup, provided the countries accept a common recall estimate. If they want per-country estimates, the pooling saves nothing and the cost returns.

The consent-based data sharing in the announcement is what funds that audit. The audit is also the most useful thing the programme can publish as a digital public good, more useful than the weights, because the weights are only as portable as the evidence that they travel.

## How to apply this

If you are one of the teams inheriting a shared diagnostic model, the order below avoids rebuilding twice.

1. **Ask which build you are getting.** A training catalogue, a shipped product and a roadmap have different numbers. Request the model card with the class list, the release date and the input specification, then check the class count against the catalogue you were promised.
2. **Request per-class recall at a stated prevalence.** Aggregate accuracy is not a quality signal for a diagnostic tool. If the vendor cannot supply per-class figures, treat the headline as a marketing figure and budget your own audit.
3. **Recompute precision for your own field mix.** Two numbers are enough: the model's sensitivity and specificity, and your best estimate of local prevalence for the classes you care about. The arithmetic above takes a minute and changes the operating point.
4. **Set the threshold against your cost ratio, and publish it.** A false flag costs a visit, a missed case costs a harvest. At 5 percent prevalence, a threshold holding precision at 0.80 misses 77 percent of cases, so decide which error your crop can absorb before you ship the number.
5. **Design a two-stage flow and budget the second stage.** The model flags, an extension officer or agronomist confirms. The human stage is not a courtesy, it is the control that absorbs the precision loss at low prevalence, and it has a real cost per confirmed case.
6. **Size the local audit before you promise a result.** 182 confirmed cases per class to detect a 7-point change at 80 percent power; 400 to detect 4.7 points. Convert to field photos at your own prevalence and put that number in the plan.
7. **Watch for covariate shift that priors cannot fix.** A new phone, a different growth stage or a new symptom presentation needs relabelled images, not a new prior. Track confidence distributions by site; a drop in mean top-class score on the same crop usually means the input changed, not the disease.
8. **Publish the audit, not just the model.** If the licence says open components return as digital public goods, the evaluation belongs in that set. A published per-class audit is the only artifact that lets the next country skip straight to deployment.

## Key takeaways

| Claim | What the sources support |
|-------|--------------------------|
| A shared African-built crop model is being distributed to seven countries | UNDP, 3 November 2026: Italy, Morocco, Tunisia, Kenya, Ghana, Tanzania and Rwanda |
| The base model was validated in three of those countries | Kenya, Morocco and Italy, per the UNDP release |
| A 96 percent accuracy figure can coexist with 38 percent precision | At 2 percent prevalence with 0.92 sensitivity and 0.97 specificity, precision is 0.385 |
| Changing the disease mix can raise headline accuracy while lowering trust | Planted six-class demo: accuracy 0.908 to 0.941, viral precision 0.873 to 0.774 |
| Prior correction trades recall for precision | Viral recall 0.87 to 0.81, nutrient recall 0.75 to 0.66 in the same demo |
| Holding precision at low prevalence costs most of the recall | At 5 percent prevalence a 0.80-precision threshold catches 22.7 percent of cases |
| The mix you estimate from app users is not the field's mix | Self-selected photos are enriched for disease, which makes precision look better than it is |
| A small local pilot cannot verify the published number | 100 confirmed cases gives a ±6.6 point interval and detects only drops above 9.4 points |
| A pooled seven-country audit is the affordable design | 182 confirmed cases per class, about 6,067 photos per country at 3 percent prevalence |

## References

- UNDP Rome Centre, African AI companies and G7 industry move from partnership to deployment in Rome (3 November 2026) — https://www.undp.org/romecentre/press-releases/african-ai-companies-and-g7-industry-move-partnership-deployment-rome
- UNDP Rome Centre, South Korea governor of Chungcheongnam-do province visits the AI Hub for Sustainable Development in Rome (readout on DeepLeaf's opened foundational model and diffusion pathways) — https://www.undp.org/romecentre/press-releases/south-korea-governor-chungcheongnam-do-province-visits-ai-hub-sustainable-development-rome
- UNA Boulder, Bringing Africa's AI innovators to Boulder: inside the UNDP AI Hub for Sustainable Development — https://www.unaboulder.org/blog/bringing-africas-ai-innovators-to-boulder-inside-the-undp-ai-hub-for-sustainable-development
- The Next Africa, DeepLeaf partners with Agrivi to scale AI agriculture solutions across Europe and Africa — https://magazine.thenextafrica.com/deepleaf-partners-with-agrivi-to-scale-ai-agriculture-solutions-across-europe-and-africa/
- IRCAI Global Top 100, DeepLeaf AI entry (55+ crops, 800 anomaly types) — https://ircai.org/top100/entry/deepleaf-ai/
- AI Hub for Sustainable Development, partner countries and programme — https://aihubfordevelopment.org/
- UNDP, G7-endorsed AI Hub for Sustainable Development — https://aihubfordevelopment.org/highlights

## Related posts

- [How Much Human Support Does AI Advice Need? 605 Weighed Harvests in Kenya and Uganda](/posts/human-support-dosage-ai-farm-advice/) — the human stage in the loop, costed across two countries.
- [44,000 Seeds, One Open Map: What the Dryland Sorghum Pangenome Changes for Breeders](/posts/sorghum-pangenome-genomic-selection/) — the upstream data release this class of model depends on.
- [A Mine, Forty Clinics, Ten Million Downloads: What Open Weights Did for TB Screening](/posts/open-weights-tb-screening-front-line/) — an open diagnostic model with a published deployment record.
- [Small AI by design: ten Kenyan startups and the offline-first constraint](/posts/kenya-ai-academy-offline-first/) — why latency and device budgets decide what reaches a farmer.
