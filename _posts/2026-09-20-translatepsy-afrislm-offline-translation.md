---
title: "Nineteen Languages, One Download: Reading the TranslatePsy-AfriSLM Release Past the Headline"
date: 2026-09-20 00:00:00 +0300
categories: [AI in Africa, Machine Learning]
tags: [machine translation, low-resource nlp, african languages, on-device ai, quantization, benchmarks]
math: true
image:
  path: /assets/img/cover-translatepsy-afrislm-offline-translation.webp
  alt: A smartphone outline holding nineteen language-code chips, with a dashed cloud behind it struck through by a red slash labelled no uplink, beside a size bar comparing a 21 MB language-pair model with a 641 MB quantized one
---

## Introduction

On 2 September 2026, Tether AI Research published a family of open translation models called **TranslatePsy-AfriSLM**, covering **19 Sub-Saharan African languages** and designed to run entirely on a phone or laptop ([Tether](https://tether.io/news/tether-releases-open-source-ai-translation-models-for-african-and-european-languages/), [Crypto Briefing](https://cryptobriefing.com/tether-open-source-ai-translation-african-languages/)). The weights are on Hugging Face under Apache-2.0, and the research behind them was accepted at the EMNLP 2026 main conference ([arXiv:2608.18655](https://arxiv.org/abs/2608.18655)). The release supports Hausa, Amharic, Yoruba, Lingala, Swahili, Igbo, Zulu, Somali, Oromo, Malagasy, Kinyarwanda, Xhosa, Afrikaans, Wolof, Luganda, Nyanja, Shona, Tswana and Southern Sotho.

Most coverage framed this as a benchmark story — "800 million parameters beats a 122-billion-parameter model." That is what the paper says, and the numbers are real. But the benchmark is the least interesting thing in the release, and reading only the headline hides the part that matters for anyone deploying in Kenya, Nigeria or Malawi: **the smallest deployment fits in tens of megabytes and never asks the network for permission.**

> **The framing**
> This post is about a *release*, not a technique: what shipped, where offline translation becomes the difference between usable and unusable, and how to read the vendor benchmark without either dismissing or swallowing it. It is not low-resource NLP basics ([Low-Resource NLP](/posts/low-resource-nlp/)), a RAG build ([RAG for Low-Resource African Languages](/posts/rag-low-resource-african-languages/)) or a fine-tuning walkthrough ([Fine-Tuning African Language LLMs](/posts/fine-tuning-african-language-llms/)) — those assume a model you can reach over the network. This is about the case where there is no network.
{: .prompt-info }

## What actually shipped

Four artifacts, and they are not interchangeable:

| Release | What it is | Size | Languages | Licence |
|---|---|---|---|---|
| `TranslatePsy-AfriSLM` | Full-parameter SFT of Qwen3.5 (0.8B / 2B / 4B) | 0.8B–4B params, plus Q4/Q8 GGUF builds | 19 African + English | Apache-2.0 |
| `TranslatePsy-AfriNano` | Marian/Bergamot compact MT, 17M–43M params | 21–35 MB per language pair | 8 African + English | Apache-2.0 |
| `TranslatePsy-EuroNano` | Same design, English as pivot | compact | 9 European, 90 directions | Apache-2.0 |
| `TranslatePsy-AfriSLM-Synthetic-Mix` | The training data | — | English ↔ 19 African | CC BY-NC 4.0 |

The **Nano** family is the browser-and-feature-phone answer: a 17M–43M parameter Marian model whose card reports it retains up to **96.24% of NLLB-200 translation accuracy at 56.7× smaller deployment size and 3.53× lower peak CPU RAM** ([model card](https://huggingface.co/qvac/TranslatePsy-AfriNano)). The **SLM** family is the conversational answer: a full-parameter fine-tune, quantised to a 641 MB Q4_K_M GGUF, which we downloaded and ran (below).

The licence split matters. The **models** are Apache-2.0 — commercial use is fine. The **synthetic training mix** is CC BY-NC 4.0, non-commercial only, and the paper states plainly that the raw open-source mix "will not be released due to licensing/provenance issues." If you are building a product, you can ship the weights; you cannot build a commercial training pipeline on the published data.

## Why offline is the headline

The 2026 edition of the GSMA's *State of Mobile Internet Connectivity* puts Sub-Saharan Africa at **25% of the population using mobile internet on their own device**, with **66% — roughly 820 million people — living inside mobile broadband coverage and not using it** ([Capital Ethiopia, 20 Sep 2026](https://capitalethiopia.com/2026/09/20/sub-saharan-africa-faces-digital-divide-as-handset-costs-and-usage-gap-persist/); [Nairametrics, 17 Sep 2026](https://nairametrics.com/2026/09/17/nigeria-accounts-for-140-million-mobile-internet-usage-gap-gsma-says/) reports the same report's 3.1 billion global usage-gap figure, with Nigeria accounting for about 140 million of it).

The same report identifies what is actually stopping people, and it is not coverage:

- The retail price of an entry-level internet-enabled handset is **76% of average monthly income for the poorest fifth of Sub-Saharan Africa** (44% across low- and middle-income countries).
- Global shipments of sub-$100 smartphones are projected to **fall 36% in 2026** as AI datacentre demand drives memory prices up; Sub-Saharan Africa is expected to take 16 million fewer shipments, a decline of more than a quarter.
- About **58% of the adult usage gap in LMICs is women**, and rural residents are 28% less likely to adopt mobile internet than urban ones.

> **Why a download beats an API here**
> Every barrier above is about the *device and the bill*, not the network. A 21–35 MB model inside an app works on the phone a farmer already owns, offline, with no per-call cost and no data leaving the handset. The cloud version of the same model is better and irrelevant to that user.
{: .prompt-tip }

That is the honest argument for this release: not that a 0.8B model dethroned a 122B one, but that the useful size class now fits where the users are.

## The claim that carries the release: data quality, not parameters

The paper's methodological bet is that **curation beats scale**. Its unified quality-estimation filter removes up to **96% of training tokens without degrading quality**: a filtered configuration reaches an SSA-COMET score of **0.530 versus 0.528** for the unfiltered one, using **1.76B tokens instead of 44.93B**. The authors' conclusion is that raw open-source parallel text "contains a weak training signal that can be concentrated via dedicated curation."

Their filtering choice is also instructive. No single quality estimator dominates: SSA-COMET filtering produced the best SSA-COMET gain (+4.85% over random filtering) but a weaker COMET-22 gain (+0.94%), while MetricX filtering produced the best MetricX gain (+13.3%) and the *only negative* chrF++ result (−0.16%). They therefore filter on an **average robust z-score across estimators, computed in the same direction as training** — the aligned configuration. If you curate parallel data for any low-resource pair, that is the transferable result: combining estimators on the training direction is more robust than trusting one metric's opinion of itself.

## We ran the smallest quantised model on a laptop

Benchmarks are claims; downloads are artifacts. So we took the 0.8B Q4_K_M GGUF (`TranslatePsy-AfriSLM-0.8B-Q4_K_M-imat.gguf`, **672,329,792 bytes = 641.2 MiB**), downloaded `llama.cpp` release `b11062`, and ran inference on **4 CPU threads with no GPU, no network, and the paper's own prompt template from Figure 9**:

{% raw %}
```bash
# offline, CPU-only, greedy decoding; the system+user prompt are the paper's Fig. 9
SYS="You are a professional English to Swahili translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Swahili grammar, vocabulary, and cultural sensitivities. Produce only the Swahili translation, without any additional explanations or commentary."
TXT="The rains have come early this year, and the farmers need to know which seeds to plant."
llama-cli -m afrislm-q4.gguf -sys "$SYS" \
  -p "Please translate the following English text into Swahili: $TXT Translation:" \
  -n 80 --temp 0 --top-k 1 -c 1024 -t 4 -st
```
{% endraw %}

The real output, verbatim:

```text
Mvua zimeanza mapema mwaka huu, na wakulima wanahitaji kujua ni mbegu gani wanapaswa kupanda.

[ Prompt: 100.1 t/s | Generation: 28.7 t/s ]
```

Four more directions, same binary and settings, all produced in a single sitting:

| Direction | Input | Output (verbatim) |
|---|---|---|
| English → Hausa | Wash your hands with soap before you prepare food. | `Ku wanke hannayenku da sabulu kafin ku dafa abinci.` |
| English → Zulu | The clinic will open at eight in the morning. | `Umtholampilo uzovulwa ngo-8 ekuseni.` |
| Swahili → English | Mvua zimeanza mapema mwaka huu, na wakulima wanahitaji kujua ni mbegu gani wanapaswa kupanda. | `The rains have started earlier this year, and farmers need to know which seeds to plant.` |
| Swahili → Hausa | Karibu, tafadhali kaa hapa na unywe maji. | `Karibu, don Allah ka zauna a nan na shan ruwa.` |

Measured on that machine: **100.1 tokens/s prompt processing, 28.7 tokens/s generation, 1.05 GB peak RSS** (`/usr/bin/time -v`, 0.8B Q4_K_M, 1024-token context, 4 threads), with repeat runs landing in the 91–103 t/s prompt and 25.9–28.8 t/s generation range — about three seconds wall-clock for a load-and-translate. A mid-range laptop from 2019 ran a 19-language translator with the Wi-Fi off.

Two honest observations from the outputs:

- **English → Swahili preserved the hard parts.** The date and number both survived: `...haina haja ya kurudi hadi Juni 2027` keeps "June 2027" intact and negates correctly. The agreement on `haina haja` ("it has no need") does not match the woman, where Swahili expects `hana haja` — the kind of slip only a human reviewer catches.
- **Zero-shot Swahili → Hausa is where it frays.** The greeting came back as `Karibu`, the Swahili form rather than the Hausa `Barka da zuwa`. The paper predicts this: its out-of-distribution languages show "smaller and more heterogeneous" gains with "metric-specific regressions." Training was English ↔ African only; African → African was never seen.

That second point is not a defect to hide; it is the release's actual frontier, and the authors say so.

## How to read the benchmark table

Here is the paper's Table 3 (SSA-COMET, 0–1, averaged across available test sets). The headline claim lives in the "Ours" rows:

| Model | Flores-200 | BOUQuET | Smol |
|---|---|---|---|
| Qwen3.5-122B-A10B | 0.5505 | 0.5716 | 0.4574 |
| TranslateGemma-27B | 0.5455 | 0.5677 | 0.4608 |
| NLLB-3.3B | 0.5944 | 0.6178 | 0.4909 |
| AfriqueGemma-12B | 0.5655 | 0.5892 | 0.4649 |
| **TranslatePsy-AfriSLM-0.8B** | **0.5944** | **0.6223** | **0.4973** |
| TranslatePsy-AfriSLM-2B | 0.6070 | 0.6322 | 0.5074 |
| TranslatePsy-AfriSLM-4B | 0.6143 | 0.6391 | 0.5136 |

The 0.8B genuinely beats the 122B specialised-LLM baseline on all three, and beats NLLB-3.3B on BOUQuET and Smol. Now read the *same* paper's significance table, which reports five metrics with bootstrap p-values. "Outperforms" is a metric-specific claim, and one column can quietly contradict another:

{% raw %}
```python
# Reading a "we beat much larger models" claim without swallowing it.
# Numbers transcribed from Tables 25 (FLORES-200) in Gritta et al. 2026,
# arXiv:2608.18655. Direction matters: MetricX is lower-is-better.
HIGHER_IS_BETTER = {"COMET-22": True, "SSA-COMET": True, "chrF++": True,
                    "spBLEU": True, "MetricX": False}

FLORES = {
    "vs Qwen3.5-122B-A10B": {
        "COMET-22":  (-0.0001, 0.452),
        "SSA-COMET": (+0.0413, 0.0005),
        "MetricX":   (-0.548,  0.0005),
        "chrF++":    (+3.30,   0.0005),
        "spBLEU":    (+3.95,   0.0005),
    },
    "vs NLLB-3.3B": {
        "COMET-22":  (-0.0093, 0.0005),
        "SSA-COMET": (+0.0000, 0.483),
        "MetricX":   (+0.284,  0.0005),
        "chrF++":    (-1.45,   0.0005),
        "spBLEU":    (-0.93,   0.0005),
    },
}

def verdict(metric, delta, p, alpha=0.05):
    if p > alpha:
        return "TIE (not significant)"
    better = (delta > 0) if HIGHER_IS_BETTER[metric] else (delta < 0)
    return "supported" if better else "REGRESSED"

print("claim: TranslatePsy-AfriSLM-0.8B outperforms, on FLORES-200")
for baseline, metrics in FLORES.items():
    print(f"  {baseline}")
    for metric, (delta, p) in metrics.items():
        tag = "p<0.001" if p < 0.001 else f"p={p}"
        print(f"    {metric:<10} {delta:+7.4f}  {tag:<8} -> {verdict(metric, delta, p)}")
```
{% endraw %}

Run it and the release gets more interesting, not less:

```text
claim: TranslatePsy-AfriSLM-0.8B outperforms, on FLORES-200
  vs Qwen3.5-122B-A10B
    COMET-22   -0.0001  p=0.452  -> TIE (not significant)
    SSA-COMET  +0.0413  p<0.001  -> supported
    MetricX    -0.5480  p<0.001  -> supported
    chrF++     +3.3000  p<0.001  -> supported
    spBLEU     +3.9500  p<0.001  -> supported
  vs NLLB-3.3B
    COMET-22   -0.0093  p<0.001  -> REGRESSED
    SSA-COMET  +0.0000  p=0.483  -> TIE (not significant)
    MetricX    +0.2840  p<0.001  -> REGRESSED
    chrF++     -1.4500  p<0.001  -> REGRESSED
    spBLEU     -0.9300  p<0.001  -> REGRESSED
```

Read that second block carefully, because it is the part no press release carries. Against **NLLB-3.3B on Flores-200**, the 0.8B model is a **statistical tie on SSA-COMET** ($\Delta = 0.0000$, $p = 0.483$) and **loses on chrF++, spBLEU, MetricX and COMET-22**. The paper's own phrasing is precise — it "exceeds NLLB-3.3B on BOUQuET and Smol and **matches** it on Flores-200" — and a reader who skips to the abstract reads something stronger than the tables support.

The lesson generalises past this release. "Beats models 100× larger" is a defensible statement about *specific benchmarks and specific metrics*, chosen by the authors, on in-distribution languages. It is not a claim about your sentence, your dialect or your domain. The paper says this itself: the strongest per-language gains land on low-baseline languages (Oromo, Malagasy, Lingala, Tswana, Zulu), Afrikaans gains least, and held-out languages show "metric-specific regressions for Nigerian Pidgin, Sudanese Arabic, and Tamazight."

## What the authors themselves flag

Three limitations from the paper that a deployment decision needs:

1. **No human evaluation yet.** The authors state that reference-based metrics suggest they "have not yet reached the translation quality of European and Asian languages," and that resolving absolute quality requires expert annotators, not more metrics. The models' own benchmark is built from the same metric families used to filter the training data — a circularity the authors test for, but do not fully escape.
2. **Dialects are under-represented.** "The exact dialect tracking is unavailable for our web-curated corpora; models may over-reflect standardized written forms and under-represent regional dialects or oral traditions." They advise auditing before deployment. A model that translates textbook Swahili is not automatically a model that handles Sheng or coastal dialects.
3. **Some of the data is synthetic, and provenance is partly unknown.** The "open-source" mix is not released at all; the Fine Translations corpus is entirely synthetic; and an unknown share of the "open-source" subset is at least partially synthetic due to LLM-based rewriting upstream.

None of that undoes the release. It does mean the right posture is "powerful new component, needs a local acceptance test" — a posture to hold about every model you did not evaluate yourself.

## How to apply this: pick the size class, then test it

**Step 1 — choose by device, not by benchmark rank.** The two families are not a quality ladder; they are different deployment targets:

| Your constraint | Use | Why |
|---|---|---|
| Low-end Android or in-browser, no install budget | `TranslatePsy-AfriNano` (21–35 MB per pair) | Fits an app bundle; ~3.5× lower CPU RAM than NLLB-class |
| Laptop or server, conversation, African → African attempts | `TranslatePsy-AfriSLM` 0.8B–4B (Q4 GGUF, 641 MB measured for 0.8B) | Independent generation, best quality per parameter reported |
| European languages with English as pivot | `TranslatePsy-EuroNano` | 90 directions from two models per tier |
| Commercial training on the published data | — | Weights are Apache-2.0; the synthetic mix is **CC BY-NC 4.0** |

**Step 2 — run the release audit, not the leaderboard.** Five checks, all cheap:

1. Download the *smallest* quantisation that meets your latency budget and run 20 real sentences from your domain through it.
2. Score with a metric family you did **not** see in the release notes (if every published metric shares a family with the training filter, use chrF or a small human panel).
3. Check the per-language rows, never the average. One 0.62 aggregate can hide a 0.41 language.
4. Test your language pair in *both* directions, and test African → African separately — the training pairs were English-centred.
5. Write down your acceptance threshold **before** you look at the output. Two of the verdicts above flip purely on which metric you chose.

**Step 3 — decide what offline actually buys you.** A device-side model that degrades gracefully beats a cloud model that fails completely: `hana haja` rather than no translation at all is an easy trade for a health-education screen or a market-price lookup.

## Key takeaways

| Takeaway | Evidence |
|---|---|
| The small-size class is now genuinely deployable offline | 19 languages; 0.8B Q4 = 641 MB; 1.05 GB peak RSS; 28.7 tok/s on 4 CPU threads |
| Offline is the point, not a bonus | GSMA 2026: 25% of Sub-Saharan Africa uses mobile internet; 66% usage gap; handset costs 76% of the poorest quintile's monthly income |
| Curation beat scale | 96% of training tokens filtered away with SSA-COMET 0.530 vs 0.528 (1.76B vs 44.93B tokens) |
| "Outperforms much larger models" is metric-specific | 0.8B beats Qwen3.5-122B on SSA-COMET/chrF++/spBLEU, **ties** on COMET-22 ($p=0.452$); ties NLLB-3.3B on SSA-COMET and loses on four other metrics (Flores-200) |
| Know which licence covers what | Weights Apache-2.0; synthetic data CC BY-NC 4.0; the raw open-source mix is unreleased |
| Human and dialect evaluation is still open | Paper limitations: quality below European/Asian benchmarks; dialects and oral forms under-represented |

## References

- Gritta, Lambert, Back & Nazir, *TranslatePsy-AfriSLM: High-Quality Data Scaling For Low-Resource Machine Translation*, EMNLP 2026 — [arXiv:2608.18655](https://arxiv.org/abs/2608.18655)
- Tether, *Tether Releases Open-Source AI Translation Models for African and European Languages* — [tether.io](https://tether.io/news/tether-releases-open-source-ai-translation-models-for-african-and-european-languages/)
- Model cards and weights: [AfriSLM-0.8B](https://huggingface.co/qvac/TranslatePsy-AfriSLM-0.8B), [Q4 GGUF](https://huggingface.co/qvac/TranslatePsy-AfriSLM-0.8B-Q4-GGUF), [AfriNano](https://huggingface.co/qvac/TranslatePsy-AfriNano)
- *Tether releases open-source AI translation models for African and European languages* — [Crypto Briefing, 2 Sep 2026](https://cryptobriefing.com/tether-open-source-ai-translation-african-languages/) (21–35 MB per pair)
- *Tether Releases Open Offline Translation Models for 19 African Languages* — [iAfrica, 8 Sep 2026](https://iafrica.com/tether-releases-open-offline-translation-models-for-19-african-languages-with-peer-reviewed-benchmarks/)
- *Sub-Saharan Africa faces digital divide as handset costs and usage gap persist* — [Capital Ethiopia, 20 Sep 2026](https://capitalethiopia.com/2026/09/20/sub-saharan-africa-faces-digital-divide-as-handset-costs-and-usage-gap-persist/)
- *Nigeria accounts for 140 million mobile internet usage gap, GSMA says* — [Nairametrics, 17 Sep 2026](https://nairametrics.com/2026/09/17/nigeria-accounts-for-140-million-mobile-internet-usage-gap-gsma-says/); report: [GSMA, *State of Mobile Internet Connectivity 2026*](https://www.gsma.com/somic/)
- Benchmarks used: [FLORES-200, arXiv:2207.04672](https://arxiv.org/abs/2207.04672) · [BOUQuET, arXiv:2502.04314](https://arxiv.org/abs/2502.04314) · [SMOL, arXiv:2502.12301](https://arxiv.org/abs/2502.12301)
- Inference runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Related posts

- [RAG for Low-Resource African Languages](/posts/rag-low-resource-african-languages/)
- [Fine-Tuning African Language LLMs](/posts/fine-tuning-african-language-llms/)
- [Evaluating LLMs for African Use Cases](/posts/evaluating-llms-african-use-cases/)
- [Swahili NLP](/posts/swahili-nlp/)
- [Low-Resource NLP](/posts/low-resource-nlp/)
- [Edge AI for African Markets](/posts/edge-ai-mobile-african-markets/)
