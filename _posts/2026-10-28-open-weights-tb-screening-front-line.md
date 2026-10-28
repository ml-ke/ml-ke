---
title: "A Mine, Forty Clinics, Ten Million Downloads: What Open Weights Did for TB Screening"
date: 2026-10-28 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [tb screening, medical ai, medgemma, medsiglip, open weights, edge inference, ai in africa, chest radiography, capacity planning, clinical validation]
image:
  path: /assets/img/cover-open-weights-tb-screening-front-line.webp
  alt: A dark mine tunnel cross-section at four kilometres depth with a portable X-ray detector and a small inference box marked offline, three stacked model cards labelled image encoder, vision-language and cough audio to the right, and a 45-second clock face
---

## Introduction

> **The announcement, in one line**
> At IDWeek 2026 in Washington, a research team described how one chest X-ray model now reads films in about 40 clinics across six African countries, screens miners nearly four kilometres underground at Mponeng, and underpins a national programme in Indonesia that aims to screen 50 million people a year. The weights behind those deployments are given away, not rented.
{: .prompt-info }

This blog has covered the same technology twice from other directions. [Prequalified](/posts/who-prequalification-cad-tb-ai-screening/) followed the September decision to let computer-aided detection software onto the WHO procurement list, which is the question of who is allowed to buy the thing. [The Model Was Not the Intervention](/posts/tanzania-retina-screening-trial/) read a Tanzanian trial where the randomised comparison bundled a result, a counsellor and a pathway, and the model was the least interesting part. Neither post asked the question an engineer actually faces when a ministry hands over a project brief: what do we install, on what box, at what cost per film, and what has to be true before the output is allowed to touch a patient.

That is this post. It is a good-news story, and the parts worth studying are the unglamorous ones: a Pretoria company that took a licensed research model through European device certification, a pricing model that works out at roughly 70 US cents a film, and a national programme whose constraint turns out to be memory bandwidth rather than model accuracy.

## What was actually announced at the conference

The Google Research post published on 23 October is a retrospective rather than a launch. It walks through a decade of work on chest radiograph interpretation for tuberculosis, from a 2019 benchmark in *Radiology*, through a 2021 Stop TB Partnership evaluation that ranked the model among the top performers globally, to a prospective multi-site validation in *NEJM AI* in 2024. What is new is the packaging: instead of shipping a bespoke model per partner, the team now publishes open foundation models that partners adapt themselves.

| Piece | What it is | Why it matters for a deployment |
|---|---|---|
| MedSigLIP | Two-tower encoder, 400M parameters each side, 448x448 pixels, up to 64 text tokens | Lets a team classify or search radiographs with plain-language labels, and fine-tune on a few thousand local films |
| MedGemma | Multimodal family: 4B, 27B text-only, 27B multimodal | Handles the clinical reasoning layer, and the 4B variant is small enough to run on hardware a district hospital can afford |
| HeAR | Audio encoder trained on more than 300 million audio segments | Screens from a cough recorded on an ordinary phone, for posts that will never own an X-ray machine |

The published download figure is the one to hold on to: more than 10 million downloads for MedGemma, across thousands of research adaptations. Downloads are not deployments, and the number says nothing about how many of those runs happened in a clinic rather than a notebook. But it does show the distribution effect of open weights. A ministry that wants a model fine-tuned on its own films, running on its own server, can do that without negotiating a licence for each site.

MedSigLIP is the piece that gets skipped in most summaries, and it is the one that changes the engineering. A two-tower encoder turns an image and a phrase into vectors in the same space, so "cavitary lesion in the right upper lobe" becomes a query rather than a class label. The model card is explicit that it is built for data-efficient classification, zero-shot classification and semantic retrieval. In a screening programme, that combination gives you three products from one model: a triage score, a search index over the archive, and a quality-control tool for finding the films that look nothing like the rest.

## The deployment record on the continent

The clinical translation was done by Nexus Intelligence, a Pretoria company founded in 2022 by a physician and a clinical engineer. Their product runs two models in parallel on every film, one that decides whether the lungs look normal and highlights what it finds, and one tuned to patterns suggestive of pulmonary tuberculosis. The clinician sees the original image with heatmaps and contours, and in the published description the whole cycle takes as little as 45 seconds.

The footprint, as described in the 23 October post:

| Measure | Reported figure |
|---|---|
| Countries live | South Africa, Sierra Leone, Côte d'Ivoire; expansions in the Democratic Republic of the Congo, Zambia and Cameroon |
| Facilities | Approximately 40 |
| Clinicians using it | More than 90 |
| Radiographs evaluated by those clinicians | More than 25,000 |
| Turnaround | From weeks to under 45 seconds |
| Free screenings delivered | More than 3,000, against a goal of 10,000 a year |
| Regulatory status | CE MDR Class IIb, ISO 13485 quality management, WHO prequalification pathway via an agreement with FIND, US FDA 510(k) in progress |

One number in that table does not agree with a September profile of the same company in Streamlinefeed, which put cumulative volume at more than 50,000 chest X-rays across about 47 sites. That profile predates the October post by seven weeks, so the two are not a snapshot and a later snapshot. They are either counting different things, radiographs read by clinicians in the deployment programme versus analyses the platform has run, or one of them is wrong. I could not reconcile them from public sources, so treat 25,000 as the clinician-reviewed figure and 50,000 as the platform's own cumulative claim.

The Mponeng deployment is the one that reads like a stunt and is not. Mponeng is the deepest mine in the world, past four kilometres down by Guinness's record, and deep-shaft miners breathe silica dust for a living. Nexus put ultra-portable digital radiography kits and offline inference underground with the mine operator, Harmony Gold, and the South African health department, so a miner is screened where he works and flagged cases go straight to molecular testing. Screening, triage and a confirmatory test in one visit, on a site with no network.

That is the argument for edge inference in one paragraph. Not privacy, not latency in the abstract, but the fact that a referral chain which normally takes weeks can compress into a single underground shift when the model is on the same table as the detector.

## What a screen costs, and what a specialist costs

The price list is the part nobody publishes, and Streamlinefeed got it: 0.70 US dollars per screen under a pay-per-study model, about 90 Kenyan shillings, dropping to 0.62 in bundles, with an unlimited annual licence at 7,000 dollars per modality for high-volume facilities. The same piece reports the founders' arithmetic on scarcity: a well-resourced system needs 100 to 120 radiologists per million people, many low-income countries have fewer than two, Nigeria has roughly 300 for more than 240 million people, South Africa about 700 for 62 million, and at least 14 African countries have reportedly had none practising at all.

Those density figures are the company's, not an independent measurement, and I would want to see the source before quoting them in a policy document. The direction is not in dispute. Neither is the throughput problem: at 0.62 dollars a study, a 25,000-film programme costs about 15,500 dollars in reading fees, which is less than the salary of a single radiologist in most of the countries where the software is deployed.

| Item | Reported cost | What it buys |
|---|---|---|
| Pay per study | $0.70 (about KSh 90) | One film read, no commitment |
| Bundled per study | $0.62 | Same, at volume |
| Annual per modality | $7,000 | Unlimited studies at one high-volume site |
| Certification | Not published | CE MDR Class IIb, ISO 13485 audits, clinical evaluation |

Now put the accuracy numbers beside the price. The independent evaluation in *The Lancet Digital Health* reported an AUC of 0.897 and the highest sensitivity among 12 commercial products, at 89.1 percent sensitivity with the threshold fixed at 0.5. The prospective study across three sites in Lusaka, published in *NEJM AI*, reported 87 percent sensitivity and 70 percent specificity at the high-sensitivity setting, against 84 percent sensitivity and 68 percent specificity for the best of ten independent radiologists. None of those radiologists reached the 90 percent sensitivity target that WHO sets for triage, the bar the comparison used.

Read the second number twice. Seventy percent specificity means that for every 100 people without TB who are screened, 30 get flagged. In a programme that screens 50 million people a year at a prevalence of, say, half a percent, that is 250,000 true cases and roughly 15 million false alarms. No health system absorbs 15 million unnecessary molecular tests. The specificity figure, not the sensitivity figure, is what decides whether a national programme is logistically possible, and it is the number least likely to appear in a press release.

## Sizing the box: three numbers that decide the hardware

Here is where the maths gets concrete and where most project briefs go wrong. Parameter counts convert to memory with one multiplication, and memory decides the device class before anyone argues about accuracy.

{% raw %}
```python
"""What the open weights cost in memory, and what an embedding index costs in bytes."""
GB = 1000 ** 3

MODELS = {
    "MedGemma 4B (multimodal)": 4.0e9,
    "MedGemma 27B (multimodal)": 27.0e9,
    "MedSigLIP (400M vision + 400M text)": 0.8e9,
}
PRECISIONS = {"bf16": 2.0, "int8": 1.0, "int4": 0.5}
HARDWARE = {"8 GB edge box": 8, "16 GB clinic laptop": 16, "24 GB workstation GPU": 24, "80 GB server GPU": 80}
HEADROOM = 1.3   # weights x 1.3 as a floor for activations, KV cache and the rest of the process

print("weights only, decimal GB")
print(f"{'model':38s}" + "".join(f"{p:>9s}" for p in PRECISIONS))
for name, params in MODELS.items():
    row = "".join(f"{params * b / GB:9.1f}" for b in PRECISIONS.values())
    print(f"{name:38s}{row}")
print()
print("fits a device class? (weights x 1.3 must be under the memory)")
for name, params in MODELS.items():
    picks = []
    for hw, mem in HARDWARE.items():
        ok = [p for p, b in PRECISIONS.items() if params * b / GB * HEADROOM <= mem]
        if ok:
            picks.append(f"{hw}: {'/'.join(ok)}")
    print(f"{name:38s}" + ("; ".join(picks) if picks else "no consumer device"))
print()
print("image-embedding index, bytes per 1,000,000 images")
print(f"{'dim':>5s} {'fp32':>10s} {'fp16':>10s} {'int8':>10s}")
for dim in (512, 768, 1152):
    row = [1_000_000 * dim * b / GB for b in (4.0, 2.0, 1.0)]
    print(f"{dim:5d}" + "".join(f"{v:9.2f}G" for v in row))
```
{% endraw %}

```text
weights only, decimal GB
model                                      bf16     int8     int4
MedGemma 4B (multimodal)                    8.0      4.0      2.0
MedGemma 27B (multimodal)                  54.0     27.0     13.5
MedSigLIP (400M vision + 400M text)         1.6      0.8      0.4

fits a device class? (weights x 1.3 must be under the memory)
MedGemma 4B (multimodal)              8 GB edge box: int8/int4; 16 GB clinic laptop: bf16/int8/int4; 24 GB workstation GPU: bf16/int8/int4; 80 GB server GPU: bf16/int8/int4
MedGemma 27B (multimodal)             24 GB workstation GPU: int4; 80 GB server GPU: bf16/int8/int4
MedSigLIP (400M vision + 400M text)   8 GB edge box: bf16/int8/int4; 16 GB clinic laptop: bf16/int8/int4; 24 GB workstation GPU: bf16/int8/int4; 80 GB server GPU: bf16/int8/int4

image-embedding index, bytes per 1,000,000 images
  dim       fp32       fp16       int8
  512     2.05G     1.02G     0.51G
  768     3.07G     1.54G     0.77G
 1152     4.61G     2.30G     1.15G
```

Three things fall out of that output. The 4B multimodal model fits in eight gigabytes at int8, which is the difference between a hospital buying a small edge box and buying a server. The 27B model does not fit anywhere useful below a 24 GB card, and only at int4 there, which is why the national-scale deployments fine-tune the small model rather than hosting the big one. And the embedding index is cheap at a million images and stops being cheap later: at 1,152 dimensions in fp16, a million films cost 2.3 GB, so ten million cost 23 GB and a hundred million cross into a sharded index that needs its own operations plan.

The model card does not publish the projection width for MedSigLIP, so the three rows above are there to let you plug in whatever your deployment actually produces. The arithmetic per dimension is linear and easy to redo.

## National scale, and why sovereignty is a memory number

Indonesia has the second-highest TB burden in the world, and its health ministry has said it wants to screen 50 million people a year. Google's description of the programme is that the ministry fine-tunes MedGemma and MedSigLIP on domestic chest radiographs and runs them on local infrastructure, so patient data stays inside the country.

Sovereignty is usually discussed as a legal question. It is also a capacity question, and the capacity arithmetic is simple enough to run before a minister signs anything.

{% raw %}
```python
"""Capacity math for a national screening programme: how much compute does 50 million X-rays a year need?"""
import math
TARGET = 50_000_000          # images per year (Indonesia MoH goal, per Google)
WORKDAYS = 250               # working days a year
SHIFT_H = 8                  # hours per working shift
SECONDS_PER_DEVICE_HOUR = 3600

# per-image wall clock, seconds
SCENARIOS = {
    "national CAD tool, 5 s/image": 5,
    "pocket X-ray + offline box, 45 s/image": 45,
    "batched GPU site, 2 s/image": 2,
}

annual_seconds = WORKDAYS * SHIFT_H * SECONDS_PER_DEVICE_HOUR
rate_8h = TARGET / annual_seconds                 # images per second needed in-shift
rate_247 = TARGET / (365 * 24 * SECONDS_PER_DEVICE_HOUR)

print(f"target: {TARGET:,} images/year over {WORKDAYS} days x {SHIFT_H} h")
print(f"required rate in-shift: {rate_8h:.2f} images/s")
print(f"required rate around the clock: {rate_247:.2f} images/s")
print()
print(f"{'device profile':44s} {'img/h':>7s} {'devices':>8s} {'utilisation':>11s}")
for name, secs in SCENARIOS.items():
    per_hour = SECONDS_PER_DEVICE_HOUR / secs
    capacity_per_device = per_hour * SHIFT_H * WORKDAYS
    devices = math.ceil(TARGET / capacity_per_device)
    served = capacity_per_device * devices
    print(f"{name:44s} {per_hour:7.0f} {devices:8d} {TARGET / served:11.2%}")
print()
# what a single 8-hour shift at each profile actually clears
for name, secs in SCENARIOS.items():
    print(f"one device, one shift ({name}): {int(SECONDS_PER_DEVICE_HOUR / secs * SHIFT_H):,} images")
```
{% endraw %}

```text
target: 50,000,000 images/year over 250 days x 8 h
required rate in-shift: 6.94 images/s
required rate around the clock: 1.59 images/s

device profile                                 img/h  devices utilisation
national CAD tool, 5 s/image                     720       35      99.21%
pocket X-ray + offline box, 45 s/image            80      313      99.84%
batched GPU site, 2 s/image                     1800       14      99.21%

one device, one shift (national CAD tool, 5 s/image): 5,760 images
one device, one shift (pocket X-ray + offline box, 45 s/image): 640 images
one device, one shift (batched GPU site, 2 s/image): 14,400 images
```

The 45-second figure that reads as fast in a press release is a disaster at national scale. A device that takes 45 seconds per film clears 640 films in an eight-hour shift, so hitting 50 million a year needs 313 of them running every working day with almost no idle time. At five seconds per film, the same target needs 35 devices, and a batched GPU site needs 14. Same model family, same accuracy, three completely different procurement bills.

That is the real reason the Indonesian ministry fine-tunes rather than simply licenses: at 50 million screens a year, inference throughput becomes the binding constraint, and a team that owns the weights can batch, quantise and tune the pipeline instead of accepting whatever latency the vendor's cloud offers.

## The translation gap has a price tag

The certification trail is worth listing because it is what separates a model from a product, and it is usually omitted from the good-news version of these stories. Nexus holds CE MDR Class IIb certification, obtained in December 2025 according to the Streamlinefeed profile, plus ISO 13485 quality management, and it signed an agreement with FIND to pursue WHO prequalification while working toward a US 510(k). Its development and validation programme incorporates more than 460,000 chest X-rays.

Class IIb is not a formality. Under the European framework it requires a clinical evaluation, a quality management system audited by a notified body, post-market surveillance, and a technical file that survives inspection. A research group with a strong AUROC and a GitHub repository is several years and a seven-figure budget away from that. The reason to say this plainly is that African teams keep being told the barrier is compute. It is not. The barrier is the paperwork that makes a regulator comfortable, and the companies that clear it are the ones that get to charge per study.

## How to apply this: five gates before you ship a model

If you are building in this space, whether for a ministry, an NGO or a private clinic chain, these are the five checks that separate a pilot from a programme.

**One: re-measure sensitivity and specificity at your prevalence.** The Lusaka numbers, 87 percent sensitivity and 70 percent specificity, and the 89.1 percent sensitivity at a fixed 0.5 threshold from the multi-product evaluation, are not the same operating point. A vendor quoting one number is hiding a threshold choice. Ask for the full curve, then pick your cut-off on your own films.

**Two: audit the image pipeline before the model.** Ungradable films are the silent failure mode. If your detector produces images with the wrong exposure, the wrong positioning or a compressed JPEG, the model's output is not wrong so much as meaningless, and it will still be reported with a confidence score.

**Three: budget the box, not the benchmark.** The table above is the whole argument. Decide the device class from memory and throughput requirements first, then choose the model size that fits. A 4B model at int8 in an 8 GB edge box, or a 27B at int4 in a 24 GB card, are the two realistic tiers, and the second one does not survive a rural clinic's power supply.

**Four: read the certificate, then the post-market plan.** A CE mark tells you the device passed an evaluation on the date it was filed. It does not tell you what happens when the manufacturer ships version 3.1. [The prequalification post](/posts/who-prequalification-cad-tb-ai-screening/) covers the version-drift problem in detail; the short version is that any programme buying software needs a contractual right to re-validate after an update, and the test set to do it with.

**Five: fund the referral, not only the screen.** Seventy percent specificity at scale generates more false positives than any laboratory network can absorb. The Tanzanian trial showed that the pathway, not the model, decided whether people reached care. A screening programme that cannot test the people it flags is a database, not a health service.

## Key takeaways

| Point | Detail |
|---|---|
| The distribution model changed | Open weights (MedGemma, MedSigLIP, HeAR) let a health system fine-tune locally and run on-device, instead of licensing a cloud endpoint per site |
| Africa is the proving ground, not the pilot | About 40 facilities in six countries, more than 25,000 clinician-reviewed films, and a deployment four kilometres underground at Mponeng |
| Unit economics are the story | Roughly $0.70 per study, $7,000 per modality per year, against a radiologist density under two per million in many low-income countries |
| Specificity decides feasibility | 70 percent specificity means 30 false positives per 100 healthy people screened; at national volume that is the binding constraint, not sensitivity |
| Throughput beats accuracy at scale | At 45 seconds per film, 50 million screens a year needs 313 devices; at 5 seconds it needs 35 |
| Certification is the moat | CE MDR Class IIb plus ISO 13485 plus a prequalification pathway is what turns a model into something a ministry can buy |

## References

- Google Research, "Bringing foundational models to the front lines of TB screening", 23 October 2026 — https://research.google/blog/bringing-foundational-models-to-the-front-lines-of-tb-screening/
- Google, "MedGemma is helping global healthcare providers deliver better care", 23 September 2026 — https://blog.google/innovation-and-ai/technology/health/medgemma-global-healthcare/
- Google Africa, "Meet the Pretoria team using AI to spot lung disease in as little as 45 seconds", 16 June 2026 — https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-pretoria-team-using-ai-to-spot-lung-disease-in-as-little-as-45-seconds/
- Streamlinefeed, "Nexus bets AI chest screens can fill Africa's radiologist void", 2 September 2026 — https://streamlinefeed.co.ke/news/nexus-bets-ai-chest-screens-can-fill-africas-radiologist-void
- Techpoint Africa, "Nexus AI is bringing AI-powered X-ray screening to places with few radiologists" — https://techpoint.africa/feature/nexus-ai-powered-x-ray-screening/
- Health AI Developer Foundations, MedGemma model card — https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
- Health AI Developer Foundations, MedSigLIP model card — https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card
- WHO, Global tuberculosis report 2025, key facts — https://cdn.who.int/media/docs/default-source/global-tuberculosis-report-2025/global-tb-report-2025_factsheet.pdf
- WHO, consolidated guidelines on tuberculosis, module 2: screening — https://www.who.int/publications/i/item/9789240022676
- Guinness World Records, deepest mine — https://www.guinnessworldrecords.com/world-records/66169-deepest-mine
- Nexus Intelligence, product and certification page — https://nexus-health.ai/
- World Today News, coverage of the IDWeek 2026 TB screening session — https://www.world-today-news.com/google-unveils-results-of-ai-driven-tuberculosis-screening-technology/

## Related posts

- [Prequalified: WHO Opens Its Procurement List to AI TB Screening Software](/posts/who-prequalification-cad-tb-ai-screening/) — the procurement gate this post assumes
- [The Model Was Not the Intervention](/posts/tanzania-retina-screening-trial/) — the randomised trial that measured attendance rather than accuracy
- [Certified in Europe, Judged at the Health Post](/posts/ai-primary-care-class-iib-ce/) — what a Class IIb certificate transfers to a rural clinic
- [AI in African Healthcare](/posts/ai-african-healthcare/) — the wider survey of imaging and triage tools on the continent
- [Small AI by design: ten Kenyan startups and the offline-first constraint](/posts/kenya-ai-academy-offline-first/) — the edge-inference economics from the builder's side
