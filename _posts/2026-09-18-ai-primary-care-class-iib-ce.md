---
title: "Certified in Europe, Judged at the Health Post: What a Class IIb CE Mark Really Buys Primary Care"
date: 2026-09-18 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [medical ai, regulation, eu mdr, eu ai act, primary care, cad, tb screening, kenya]
math: true
image:
  path: /assets/img/cover-ai-primary-care-class-iib-ce.webp
  alt: A four-step ladder of EU MDR software classes rising from Class I to Class III, with a CE certification seal hovering over the Class IIb step, beside a gauge labelled Se 90% / Sp 70% whose output splits into a thin bar of 180 cases found and a thick bar of 2,940 confirmatory tests per 10,000 people screened
---

## Introduction

On 17 September 2026, [Qure.ai announced](https://www.qure.ai/us/news-press-coverages/qureai-clinical-decision-support-system-aira-among-worlds-first-to-win-class-llb-ce-certification) that its primary-care clinical decision support system, Aira, had received **Class IIb CE certification under the EU Medical Device Regulation (MDR)**. The release describes it as the first AI-powered clinical decision support system (CDSS) "to combine documentation, workflow automation and analysis of patient EHR data with clinical decision-making under a single CE mark" ([USA Today/EZ Newswire copy](https://www.usatoday.com/press-release/story/43607/qure-ais-clinical-decision-support-system-aira-among-worlds-first-to-win-class-iib-ce-certification/), [CXOtoday](https://cxotoday.com/media-coverage/qure-ais-clinical-decision-support-system-aira-among-worlds-first-to-win-class-iib-ce-certification/)).

That is a press release, and an honest reading of the coverage is that one news release has been syndicated across a lot of mastheads. The reason it still deserves a post is the class it names, not the announcement. **Class IIb is not a badge; it is the doorway to a second, stricter regime**, and the country where the software actually runs has a say that no EU certificate can override.

> **The framing**
> [AI in African Healthcare](/posts/ai-african-healthcare/) surveyed what works across the continent's clinics — imaging, triage, maternal care. This post is narrower and harder-edged: it takes one regulatory milestone and asks what a certificate does and does not transfer to a health ministry, then does the triage arithmetic that the certificate deliberately leaves to the operator. The useful half is a downloadable threshold calculation and a six-question procurement check.
{: .prompt-info }

## What actually cleared the bar

Qure.ai is an Indian company whose imaging models (qXR for chest X-ray, qTrack for follow-up) already clear regulatory review at scale: the release states its portfolio spans 105+ countries, 26+ FDA-cleared findings and CE marking for 60+ indications, and [MobiHealthNews covered](https://www.mobihealthnews.com/news/asia/ce-mark-india-made-ai-screening-tb-toddlers) a prior Class IIb CE for paediatric TB screening. Aira is the different product line: not an image reader, a **primary-care co-pilot** launched at the World Health Assembly in 2025, [reported by MobiHealthNews](https://www.mobihealthnews.com/news/asia/qureai-unveils-ai-co-pilot-community-health-workers) as an LLM-based assistant trained on data from health systems in low- and middle-income countries, aimed at the finding that "more than 40% of community health workers' time is spent on manual data collection."

The company's own first-year numbers, all reported by Qure.ai and none independently audited:

| Setting | Reported result | Source |
|---|---|---|
| Kenya outreach programme | 32% less time on protocol-driven documentation, 98% task-completion rate | Qure.ai release |
| Nigeria, primary HIV care facilities | Clinic-to-admin time ratio improved 67% in favour of patient-facing time; admin time down 23% | Qure.ai release |
| Mozambique | Integrated into AlôVida, the Ministry of Health (MISAU) national platform, with VillageReach implementing | Qure.ai release |
| Footprint | Ten live pilot sites across Nigeria, Kenya, Mozambique, Solomon Islands and Bangladesh | Qure.ai release |

Kenya is named as a deployment country — "deployed with local implementation partners and Ministry of Health departments." That sentence matters more than any accuracy figure, because it is where the certificate stops and local law begins.

## Why the class is the interesting part

EU MDR Annex VIII **Rule 11** classifies medical software by what its output is used to decide, not by how clever the model is. Class IIb is the second-highest rung:

| Rule 11 outcome | Trigger | Conformity route |
|---|---|---|
| Class III | Information used for decisions that affect the life of a patient or have a serious impact on health | Notified body, most intensive |
| **Class IIb** | Information used for decisions liable to cause serious deterioration of health or a surgical intervention | Notified body assessment, ISO 13485 quality system |
| Class IIa | All other diagnostic/therapeutic decision support | Notified body (lighter) |
| Class I | Monitoring, or storage, archiving, communication and search functions | Self-certification |

The classification is a design decision made before a line of code, and getting it wrong is expensive in both directions — under-classified software is discovered during notified-body review, over-classified software burns years on documentation it never needed ([Rule 11 classification guide](https://trustedtracemed.com/resources/eu-mdr-rule-11-samd-classification.html)). Class IIb also means a quality management system per EN ISO 13485 and a clinical evaluation under Article 61, not a test-set report.

Qure.ai's release calls Class IIb "one of the EU's highest levels of regulatory scrutiny for software supporting decisions where accuracy and reliability are critical to patient outcomes." In the ladder above, that is accurate without being singular: it is *second* to Class III. What it unambiguously buys a ministry is what the release says it buys — "an independently assessed foundation for evaluation and procurement."

## The clause almost nobody quotes

Certification under MDR has a consequence in AI-specific law, and it is written plainly in the European Commission's own guidance. [MDCG 2025-6](https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en?filename=mdcg_2025-6_en.pdf), the FAQ on the interplay between the MDR/IVDR and the AI Act, states that a medical device with AI ("MDAI") is a high-risk AI system under **Article 6(1)** if it meets *both* conditions:

1. the AI system is itself a medical device, or is a safety component of one; **and**
2. it is subject to a **third-party conformity assessment by a notified body** under the MDR or IVDR.

The second condition is the point. A Class I self-certified tool never reaches Article 6(1); a Class IIb device by definition went through a notified body, so it lands in the AI Act's high-risk tier automatically. The obligations that follow are the ones health-ministry lawyers will learn to read: a risk management system (Art. 9), training-data governance (Art. 10), technical documentation (Art. 11), logging (Art. 12), deployer information (Art. 13), human oversight (Art. 14), and accuracy, robustness and cybersecurity (Art. 15).

The timing is still open. [Article 113(c)](https://artificialintelligenceact.eu/article/113/) sets Article 6(1) applications from 2 August 2027, and the 2026 amendment recorded on the AI Act Explorer pushes Annex I / Article 6(1) systems to **2 August 2028**, with stand-alone Annex III high-risk systems at 2 December 2027. So the regime is not fully live — which is exactly why a September 2026 certification is a rehearsal a regulator built early, not a scramble.

## What the stamp does not cover

Three things, and the third is the one that decides outcomes.

**Your own regulator.** Kenya is not waiting for Brussels. The Pharmacy and Poisons Board (PPB) has published a [guideline on regulation of Medical Device Software](https://web.pharmacyboardkenya.org/download/guideline-on-regulation-of-medical-device-software-in-kenya-mdsw/), and [reported in the Kenyan trade press](https://healthbusiness.co.ke/10137/kenya-tightens-oversight-of-medical-device-software/), PPB CEO Dr Ahmed Mohamed describes a **risk-based** framework that regulates SaMD separately from software embedded in hardware, aligns with the International Medical Device Regulators Forum, and draws on the Digital Health Act (2023) and the Kenya AI Strategy (2025–2030). Obligations include IEC 62304 and ISO 14971 compliance, version control, clinical evidence, post-market surveillance, and — notably — secure-by-design, role-based access control and encryption for anything network-connected. Qure.ai's own release concedes the limit: certification gives ministries a foundation "subject to applicable local rules."

**The operating point.** A CE certificate covers a declared intended use, a software version and a validated algorithm. It does not choose the score threshold at which a chest X-ray is flagged for confirmatory testing. That is the ministry's, the programme's, or the radiographer's decision, and it trades sensitivity against specificity with real consequences.

**The confirmatory queue.** Screening is not diagnosis. WHO's own [consolidated guidelines module](https://tbksp.who.int/en/node/1313) is explicit: the minimal requirements for a target screening test are "an overall sensitivity of 90% and a specificity of 70%," and screening tests "are not intended to provide a definitive diagnosis." Every false positive it emits has to be absorbed by a GeneXpert cartridge, a clinician's hour and a patient's travel.

## The arithmetic the certificate leaves to you

Here is the part a procurement team can run before signing. The inputs are **measured**, not hypothetical: 12 CAD products and 11 radiologists reading the same 774 chest X-rays from the South African National TB Prevalence Survey, against a composite microbiological reference standard ([Sci Rep, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12215708/)). Positive predictive value at a screening prevalence $p$ is

$$\text{PPV} = \frac{\text{Se}\cdot p}{\text{Se}\cdot p + (1-\text{Sp})(1-p)}$$

and notice what it does at low $p$: a 90/70 test in a 2% queue predicts at 5.8%, because specificity enters the denominator multiplied by the large healthy pool.

```python
"""What a screening operating point costs downstream. Stdlib only."""

OPERATING_POINTS = [                      # measured pairs, one study
    ("WHO screening TPP floor",           0.900, 0.700),
    ("Radiologist, UK, youngest band",    0.857, 0.889),
    ("Radiologist, India, youngest band", 0.750, 0.926),
    ("Radiologist, UK, oldest band",      0.662, 0.745),
]
PREVALENCES = [0.005, 0.02, 0.10]         # queue prevalence, not study prevalence
N = 10_000


def triage(sens, spec, prev, n=N):
    cases = prev * n
    tp = cases * sens
    fn = cases - tp
    fp = (n - cases) * (1 - spec)
    tn = (n - cases) - fp
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                ppv=tp / (tp + fp), npv=tn / (tn + fn),
                screens_per_case=n / tp, fp_per_case=fp / tp)


for prev in PREVALENCES:
    print(f"\n-- prevalence {prev*100:.1f}%  ({prev*N:.0f} cases per {N:,}) --")
    print(f"{'operating point':33s} {'TP':>5s} {'FP':>6s} {'FN':>4s} "
          f"{'PPV':>6s} {'screen/case':>11s} {'FP per case':>11s}")
    for label, sens, spec in OPERATING_POINTS:
        r = triage(sens, spec, prev)
        print(f"{label:33s} {r['tp']:5.0f} {r['fp']:6.0f} {r['fn']:4.0f} "
              f"{r['ppv']*100:5.1f}% {r['screens_per_case']:11.0f} "
              f"{r['fp_per_case']:11.1f}")
```

Run as-is, the output is:

```text
-- prevalence 2.0%  (200 cases per 10,000) --
operating point                      TP     FP   FN    PPV screen/case FP per case
WHO screening TPP floor             180   2940   20   5.8%          56        16.3
Radiologist, UK, youngest band      171   1088   29  13.6%          58         6.3
Radiologist, India, youngest band   150    725   50  17.1%          67         4.8
Radiologist, UK, oldest band        132   2499   68   5.0%          76        18.9
```

Read the last two columns against each other, because that is the decision. **The test that just meets the WHO floor is not the cheapest to operate.** At 2% prevalence it flags 16.3 people for confirmatory testing per case it finds; the Indian radiologists' operating point (75.0% sensitivity, 92.6% specificity) flags 4.8 — three times less downstream load, at the cost of missing 30 more cases per 10,000 and finding one case per 67 people screened instead of per 56. In a district where a GeneXpert cartridge is the scarce resource, that trade is a budget line; in one where a missed case means a year of transmission, it is a different argument entirely.

The same table also shows why consistency, not superhuman accuracy, is the honest case for CAD. In the South African survey, the highest-AUC CAD product (Lunit, AUC 0.902) beat every radiologist, the second (Nexus, 0.897) matched or beat them, and **qXR and most other products statistically overlapped with the human readers**. What does not overlap is stability across patient bands: the same UK radiologists held 85.7% sensitivity in the youngest band and 66.2% in the oldest, while the model's score distribution is fixed at release. The full sweep in the script above runs the same comparison at 0.5%, 2% and 10% prevalence — at 0.5% even the best operating point needs ~200 people screened per case found, which is the honest number for population-wide screening and the reason targeted queues exist at all.

## How to apply this: six questions before you sign

| Question | What to ask for | Why it is the real control |
|---|---|---|
| 1. Intended use, verbatim | The declared intended-use statement and the software version covered | The certificate is void outside it; "AI for lung health" is not an intended use |
| 2. Class and certificate | Class (e.g. IIb), notified body name and certificate number | Lets you verify the certification exists, and predicts your AI Act exposure |
| 3. Clinical evidence | The clinical evaluation summary, and the population it was validated on | Accuracy measured on a different prevalence is not your accuracy |
| 4. Threshold policy | The operating point supplied by default, and how it is changed | This single number sets your confirmatory-queue load (table above) |
| 5. Local registration | PPB MDSW registration status and the class PPB assigns it | CE is European; the licence to operate is national |
| 6. Post-market plan | Who monitors live performance, on what cadence, and what triggers a rollback | Drift and version changes are the failure modes certification cannot prevent |

Question 6 is where the AI Act and PPB guidance converge — both demand ongoing clinical evaluation after market entry rather than a one-time stamp. For the engineering side of that loop, [RegTech and Model Governance for MLOps](/posts/mlops-regtech-model-governance/) covers the control set, and [Edge AI in African Markets](/posts/edge-ai-mobile-african-markets/) covers why connectivity and device constraints belong in the intended-use statement rather than the deployment plan.

## Key takeaways

| Takeaway | Detail |
|---|---|
| Certification is a class, not a score | Class IIb under EU MDR means notified-body assessment, ISO 13485 and clinical evaluation under Article 61 — not a claim of better accuracy |
| Class IIb automatically makes it AI Act high-risk | MDCG 2025-6: a medical device with AI under third-party conformity assessment meets Article 6(1) |
| The clock is not fully live | Article 113(c) dates Article 6(1); Annex I systems apply from 2 August 2028, stand-alone Annex III from 2 December 2027 |
| The threshold, not the model, sets your operating cost | At 2% prevalence the WHO-floor 90/70 test spends 16.3 confirmatory tests per case found; a 75/92.6 point spends 4.8 |
| Consistency is the honest case for CAD | In the South African survey CAD matched rather than crushed radiologists — but its sensitivity did not fall from 85.7% to 66.2% with patient age |
| Local registration is not implied | Kenya's PPB runs a risk-based MDSW framework (IEC 62304, ISO 14971, post-market surveillance); CE is not a Kenyan licence |

## References

1. [Qure.ai — Aira receives Class IIb CE Mark under EU MDR](https://www.qure.ai/us/news-press-coverages/qureai-clinical-decision-support-system-aira-among-worlds-first-to-win-class-llb-ce-certification), 17 September 2026 (primary; company-reported figures).
2. [USA Today / EZ Newswire — syndicated release text](https://www.usatoday.com/press-release/story/43607/qure-ais-clinical-decision-support-system-aira-among-worlds-first-to-win-class-iib-ce-certification/) and [CXOtoday coverage](https://cxotoday.com/media-coverage/qure-ais-clinical-decision-support-system-aira-among-worlds-first-to-win-class-iib-ce-certification/), 17 September 2026.
3. [MobiHealthNews — Qure.ai unveils AI co-pilot for community health workers](https://www.mobihealthnews.com/news/asia/qureai-unveils-ai-co-pilot-community-health-workers), 28 May 2025 (independent).
4. [MobiHealthNews — CE mark for India-made AI for screening TB in toddlers](https://www.mobihealthnews.com/news/asia/ce-mark-india-made-ai-screening-tb-toddlers).
5. [MDCG 2025-6 — Interplay between the MDR/IVDR and the AI Act](https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en?filename=mdcg_2025-6_en.pdf), European Commission (Article 6(1) conditions).
6. [EU AI Act Article 113 — entry into force and application](https://artificialintelligenceact.eu/article/113/) and [Article 6 — classification of high-risk systems](https://artificialintelligenceact.eu/article/6/).
7. [EU MDR Rule 11 Software as a Medical Device classification guide](https://trustedtracemed.com/resources/eu-mdr-rule-11-samd-classification.html) (class ladder and conformity routes).
8. [WHO TB Knowledge Sharing — screening tools and the 90%/70% target product profile](https://tbksp.who.int/en/node/1313); [WHO target product profiles for TB screening tests](https://www.who.int/publications/i/item/9789240113572).
9. [WHO policy statement — use of computer-aided detection software for TB screening](https://www.who.int/publications/i/item/9789240110373).
10. [Scientific Reports — accuracy of CAD software versus radiologists in chest X-ray TB detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC12215708/) (774 chest X-rays, South African National TB Prevalence Survey; all sensitivity/specificity pairs used in the calculation).
11. [KEMSA — Kenya rolls out 80 AI-powered digital X-ray units](https://kemsa.go.ke/content/1100295/news/kenya-rolls-out-80-ai-powered-digital-x-ray-units-to-boost-tb-and-lung-disease-diagnosis) and [The Star — Kenya flags off 80 ultra-portable digital X-ray systems to 43 counties](https://www.the-star.co.ke/news/2025-10-13-kenya-flags-off-80-ultra-portable-digital-x-ray-systems-to-combat-tb-to-counties), 13 October 2025.
12. [The Standard — Amref's KES 154.4m AI TB-screening programme](https://www.standardmedia.co.ke/health/health-science/article/2001497328/sh154m-ai-programme-promises-to-cure-tb-screening-headache) (CAD installed in digital X-ray machines, Global Fund financing).
13. [Pharmacy and Poisons Board — Guideline on Regulation of Medical Device Software in Kenya](https://web.pharmacyboardkenya.org/download/guideline-on-regulation-of-medical-device-software-in-kenya-mdsw/); [Health Business — Kenya tightens oversight of medical device software](https://healthbusiness.co.ke/10137/kenya-tightens-oversight-of-medical-device-software/).

## Related posts

- [AI in African Healthcare: Diagnostics, Drug Discovery and Care Delivery](/posts/ai-african-healthcare/)
- [RegTech and Model Governance for MLOps](/posts/mlops-regtech-model-governance/)
- [Edge AI in African Markets](/posts/edge-ai-mobile-african-markets/)
- [The Lab Partner That Never Sleeps: How AI Now Designs Physics Experiments](/posts/ai-designed-physics-experiments/)
