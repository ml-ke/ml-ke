---
title: "Wolof in Five Sizes: What Orange's Open-Weight Release Actually Gives You"
date: 2026-11-22 00:00:00 +0300
categories: [AI in Africa, LLM]
tags: [wolof, african-languages, open-weights, licensing, senegal, model-evaluation]
image:
  path: /assets/img/cover-wolof-five-sizes-orange-release.webp
  alt: Five model sizes as rising bars with a score curve and a padlock labelled CC-BY-NC-SA on the licence gate
---

## A telecom ships a language, not a demo

> **The short version**
> On 17 November 2026, Orange published five open-weight language models for Wolof, sized from 0.6B to 14B parameters, on Hugging Face. They are free for non-commercial use. A technical report puts Wolof at 23M speakers and says the 8B and 14B checkpoints come close to commercial API systems on the team's own 63-task suite. A blind human evaluation with 12 Wolof speakers backs part of that claim: the 8B model scored 1051 Elo overall against Gemini-2.5-flash at 1112.
{: .prompt-info }

The press release takes five minutes to read. The artifacts behind it do not. Five model cards, five repositories, a 53-page technical report with appendices, and a licence that decides who can build a business on any of it. If you want to use one of these checkpoints, the announcement tells you almost nothing you need. The card and the report tell you most of it.

What follows is what a builder needs: what shipped, how the recipe changed with size, what the evaluation measured, what the card admits the model cannot do, what each size costs to run, and where the licence puts a wall. The story is good news for African language technology. The caveats are specific enough to plan around, which is the more useful kind of good news.

## The repositories predate the announcement

Start with dates, because they change how you read the release. The Hugging Face API records when each repository became public:

| Checkpoint | Public on Hugging Face | Downloads at check |
|---|---:|---:|
| Wolof-Qwen3-8B-v2610 | 9 October 2026 | 138 |
| Wolof-Qwen3-4B-v2610 | 14 October 2026 | 463 |
| Wolof-Qwen3-14B-v2610 | 14 October 2026 | 99 |
| Wolof-Qwen3-0.6B-v2610 | 16 November 2026 | 108 |
| Wolof-Qwen3-1.7B-v2610 | 16 November 2026 | 168 |

The 8B weights went up more than five weeks before the press release that "released" them. That is normal for a coordinated family launch, and it matters for one practical reason: the announcement date is not the date you should quote when you need a citation with a timestamp. The API date is. It also means the family was already visible to anyone watching the Hugging Face firehose, and the download counts above suggest it has not been widely picked up yet. A few hundred downloads per checkpoint is a real number, not a rounding error, and it is worth knowing before you assume there is a community around these weights.

The press release itself is accurate about the parts it covers. Five fine-tuned open-weight models for Wolof. Built by Orange's in-house AI teams. Available free for non-commercial use. Applications named: conversational assistants, FAQ tools, guided digital service journeys. Two quotes, from Usman Javaid, Chief AI Officer, and Ben Cheick Haidara, Deputy CEO of Orange Middle East Africa, both framing this as digital inclusion across the group's 18 markets in the region. The release also says Orange is joining a coalition of more than 60 organisations working on linguistic diversity in AI models.

That coalition is worth a sentence of context, because the release does not explain it. It is the five-year language-access push convened by the Gates Foundation, announced in September 2026, with a stated target of reaching an estimated 3.4 billion people whose languages are poorly served by current models. Reported initial signatories include Anthropic, Google, Microsoft, NVIDIA, UNICEF, the World Bank Group and the UK Foreign, Commonwealth and Development Office. Orange joining it is a commitment, not a deliverable. Treat it as a signal about intent.

Also buried near the end of the release: Wolof is one of six African languages Orange says it is working on. The others are Lingala, Swahili, Darija, Mooré and Dioula. If the Wolof recipe generalises, and the report gives you enough detail to judge whether it might, those are the next places to look.

## Five checkpoints, and a recipe that changes with size

All five models are adapted from Qwen3 dense checkpoints, so the architecture is inherited rather than invented. Reading the config files confirms it: `Qwen3ForCausalLM`, a 151,936-token vocabulary, 40,960-token maximum position embeddings, and grouped-query attention with 8 key/value heads at head dimension 128 across every size. What changes is depth and width:

| Size | Parameters | Layers | bf16/f16 weights (GiB) |
|---|---:|---:|---:|
| 0.6B | 596,049,920 | 28 | 1.11 |
| 1.7B | 1,720,574,976 | 28 | 3.20 |
| 4B | 4,022,468,096 | 36 | 7.49 |
| 8B | 8,190,735,360 | 36 | 15.26 |
| 14B | 14,768,307,200 | 40 | 27.51 |

The training recipe is documented in more detail than most commercial releases ever get. The 1.7B card lists the supervised fine-tuning mixture as roughly 446,000 samples and 109.4M tokens, split into four groups: 182K bilingual French-to-Wolof and Wolof-to-French translation instructions (10.0M tokens), 215K translated general instructions covering QA, summarisation, dialogue, knowledge, coding and reasoning (78.8M tokens), 7K multi-turn Wolof conversations (5.7M tokens), and 44K English tool-use and function-calling examples (14.9M tokens). Three epochs, cosine decay with 5% warmup, peak learning rate 2e-5, effective batch 1,024, maximum sequence length 2,048, bfloat16, DeepSpeed ZeRO-3, FlashAttention2, seed 42. Some samples were machine-translated into Wolof, and the card says so. Raw internal customer conversations were not released.

Then there is the preference stage, and this is where a reader should slow down. The DPO data is DOLCI, 233,804 preference pairs, and roughly 98% of it is in English. None of it is in Wolof. The card is explicit that this stage is general-capability alignment, not additional Wolof learning. So the model's Wolof ability comes entirely from one stage of one pipeline. Everything after it is damage control for the base model's other skills.

The last stage differs by size, which the family-level framing hides. The 1.7B card says model merging was evaluated and not retained. The technical report's stage-by-stage appendix shows the 14B row where merging at a 40/60 ratio beats DPO on the aggregate score, 0.47 against 0.42, and beats it on Wolof understanding and generation too. The report also admits the hyperparameters were only tuned on validation data for the 8B size. So the five checkpoints are not the same recipe at five scales. They are five related experiments, and the report says as much.

One more detail from the appendices that a builder should internalise: continued pre-training was tried and dropped. The team could collect only 13M raw Wolof tokens, which they judged insufficient. That is the honest version of the data-scarcity story. Wolof has 23M speakers, and the largest clean raw text corpus the team could assemble for domain adaptation was 13M tokens, which is smaller than the instruction mixture they built on top of it.

## What the evaluation shows, and what the headline number hides

The evaluation suite is large: 63 public and in-house tasks, 35 of which include Wolof, 13 using in-house data. Scores are normalised per task so that a random predictor maps to 0 and the metric maximum maps to 1, which means negative numbers are worse than guessing. The 1.7B card publishes the aggregates against its own Qwen3 base:

{% raw %}
```python
# Aggregate scores published in the Wolof-Qwen3-1.7B model card,
# as (Qwen3-1.7B base, Wolof-Qwen3-1.7B adapted).
scores = {
    "overall":               (0.23, 0.31),
    "wolof understanding":   (0.04, 0.22),
    "wolof generation":     (-0.04, 0.22),
    "english understanding": (0.40, 0.37),
    "english generation":    (0.41, 0.46),
    "french understanding":  (0.32, 0.31),
    "french generation":     (0.39, 0.43),
    "conversation":          (0.23, 0.36),
    "instruction following": (0.50, 0.25),
    "function calling":      (0.61, 0.24),
    "safety":                (0.32, 0.30),
}

print(f"{'aggregate':<24}{'base':>6}{'adapted':>9}{'delta':>7}")
for name, (base, adapted) in scores.items():
    print(f"{name:<24}{base:>6.2f}{adapted:>9.2f}{adapted - base:>+7.2f}")
```
{% endraw %}

```text
aggregate                 base  adapted  delta
overall                   0.23     0.31  +0.08
wolof understanding       0.04     0.22  +0.18
wolof generation         -0.04     0.22  +0.26
english understanding     0.40     0.37  -0.03
english generation        0.41     0.46  +0.05
french understanding      0.32     0.31  -0.01
french generation         0.39     0.43  +0.04
conversation              0.23     0.36  +0.13
instruction following     0.50     0.25  -0.25
function calling          0.61     0.24  -0.37
safety                    0.32     0.30  -0.02
```

Read the delta column, not the overall score. A +0.08 move at the top line is the sum of a +0.26 gain in Wolof generation and a set of losses elsewhere, two of them large. Instruction following drops by a quarter. Function calling drops by more than a third, which takes the model from the base checkpoint's best skill to one of its worst. The report attributes the tool-use decline mainly to false positives: the adapted models predict a tool call too often, including when none is needed. That is the exact failure mode that breaks an agent loop, because a spurious call is an action, not a wrong answer.

The card's own summary of the trade is one sentence long: some broader capabilities decline relative to the base model. The report adds the shape of the decline. At 4B and above, the losses are contained and the Wolof gains are large. At 0.6B, the model improves on Wolof but degrades on several other skills, which makes the smallest checkpoint unsuitable as a general assistant even though it is the one that fits the smallest device. The paper is also clear that the targeted applications are not agentic, and that autonomous planning, multi-step tool use, prompt-injection resistance and safe control of external actions were not evaluated at all. If your product needs a Wolof-speaking agent that touches tools, this release does not clear that bar, and Orange says so before you discover it.

Within the family, the size gradient is real. The report describes 8B and 14B as particularly competitive with substantially larger commercial systems on the suite, and the 4B as holding most of the base model's skills. Prior Wolof-aware open releases are included in the comparison: Oolel shows similar Wolof linguistic performance but ranks lower on most other skills, and tiny-aya-earth, which is not Wolof-specific, performs poorly. Those are the team's numbers on the team's tasks, but they are published per task with descriptions, so you can check what each one asks before you trust the aggregate.

## Twelve annotators, nine tasks, one blind comparison

Automatic metrics on a low-resource language are weak evidence on their own, so the report adds a human evaluation. Twelve Wolof-speaking annotators compared outputs from five systems: Gemini-2.5-pro, Gemini-2.5-flash, the final Wolof-Qwen3-8B, its unadapted Qwen3-8B base, and Oolel-v0.1. Annotators did not know which system produced which output. Each of nine tasks contributed 100 samples, with four criteria: correctness, formatting, Wolof quality, and overall preference. Pairwise outcomes became Elo scores with ties split evenly.

| Model | Overall | Correctness | Formatting | Wolof quality |
|---|---:|---:|---:|---:|
| Gemini-2.5-flash | 1112 | 1104 | 1072 | 1066 |
| Gemini-2.5-pro | 1093 | 1082 | 1090 | 1062 |
| Wolof-Qwen3-8B | 1051 | 1035 | 1030 | 1058 |
| Oolel-v0.1 | 1002 | 994 | 1030 | 1038 |
| Qwen3-8B (base) | 742 | 785 | 777 | 777 |

Win probabilities are in parentheses in the report; 0.5 means an even chance. The 8B model sits at 0.57 overall against the field, the Gemini systems at 0.64 and 0.67, Oolel at 0.48, and the base model at 0.14. The interesting column is the last one. On Wolof quality alone, the gap narrows to eight Elo points against Gemini-2.5-flash and four against Gemini-2.5-pro. That is a rounding error next to the 309-point gap between the adapted model and its base. The overall gap is carried by correctness and formatting, not by the language. That is a meaningful distinction: the model speaks Wolof at close to commercial quality in this evaluation, and it is slightly more likely to be wrong about the task or to format the answer poorly.

The base model row is the control, and it is brutal in the useful way. An unadapted Qwen3-8B scores 742 overall and 777 on Wolof quality against systems in the 1000s. Whatever else is true, the adaptation did something real.

## The licence is the load-bearing detail

Every number above describes a model you cannot build a paid product on without a conversation. The 1.7B card's licence section is short and unambiguous: non-commercial use is governed by Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, while commercial use requires separate terms with Orange. It then adds the sentence most releases leave implicit: the model is not distributed under Apache License 2.0, which permits commercial use. Users must also comply with the Qwen3 base-model licence and applicable training-data terms.

Three practical consequences. First, a pilot, a research project, an NGO deployment or a public-sector trial is inside the licence. A startup charging for a Wolof assistant is not. Second, share-alike travels: if you fine-tune and distribute a derivative, the derivative carries the same non-commercial share-alike terms, so you cannot use the release as a free starting point for a proprietary model. Third, the base-model obligation stacks on top, so the Qwen3 terms apply regardless of what you do with the adapter.

This is not unusual in African language technology, and the comparison is the part builders most often skip. Checking the canonical repositories on 22 November 2026:

| Release | Language coverage | Licence on the model repository |
|---|---|---|
| Wolof-Qwen3 (Orange) | Wolof | CC-BY-NC-SA 4.0, commercial by separate terms |
| MORENA (Vambo AI) | 12 African languages | Apache 2.0 |
| InkubaLM (Lelapa AI) | 5 African languages | CC-BY-NC 4.0 |
| tiny-aya-earth (Cohere Labs) | 67 languages, several African | CC-BY-NC 4.0 |
| Oolel-v0.1 (Soynade Research) | Wolof | none declared on the model repository |

One of the five is commercially usable without asking anyone. Three carry non-commercial terms, one of them with share-alike. One declares nothing, which in practice means you carry the risk. The most permissive of them, MORENA, is a 1.5B model built from scratch for twelve languages under Apache 2.0.

None of this makes the Orange release less useful. It makes it a research artifact and a public-service artifact, which is exactly what a state-owned or partly state-owned telecom releasing language infrastructure would produce. If you are a Kenyan county government, a school, a health NGO or a university lab working in Wolof, you have a usable model family today, with published evaluation and a licence that permits your work. If you are a founder, the same release is a benchmark and a recruitment signal, and a phone call to Orange is the actual next step.

## What each size costs to run

Sizing is arithmetic once you have the configs, and the quantised files third parties published let you replace estimates with measurements. The block below uses the parameter counts and layer/head geometry from each repository's `config.json`, the Q4_K_M file sizes from the quantisation repositories on Hugging Face, and a 4,096-token context window with bf16 key/value cache.

{% raw %}
```python
# Sizing the Wolof-Qwen3 family from its own configs plus the Q4_K_M files
# third-party quantisers published on Hugging Face, checked 2026-11-22.
models = [
    # size, params, layers, kv_heads, head_dim, q4_k_m_bytes
    ("0.6B",   596_049_920, 28, 8, 128,   396_700_000),
    ("1.7B", 1_720_574_976, 28, 8, 128, 1_107_400_000),
    ("4B",   4_022_468_096, 36, 8, 128, 2_497_300_000),
    ("8B",   8_190_735_360, 36, 8, 128, 5_030_000_000),
    ("14B", 14_768_307_200, 40, 8, 128, 9_000_000_000),
]

MIB, GIB, CTX = 1024 ** 2, 1024 ** 3, 4096

print(f"{'size':>5} {'params':>14} {'bf16 GiB':>9} {'Q4 GiB':>7} {'KV/1k MiB':>10} {'4k total GiB':>12}")
for size, params, layers, kv_heads, head_dim, q4 in models:
    weights = params * 2 / GIB
    # two tensors (K and V) per layer, 2 bytes each, 1000 tokens
    kv_per_1k = 2 * layers * kv_heads * head_dim * 2 * 1000 / MIB
    total = q4 / GIB + kv_per_1k * (CTX / 1000) / 1024
    print(f"{size:>5} {params:>14,} {weights:>8.2f} {q4/GIB:>6.2f} {kv_per_1k:>10.1f} {total:>11.2f}")
```
{% endraw %}

```text
 size         params  bf16 GiB  Q4 GiB  KV/1k MiB 4k total GiB
 0.6B    596,049,920     1.11   0.37      109.4        0.81
 1.7B  1,720,574,976     3.20   1.03      109.4        1.47
   4B  4,022,468,096     7.49   2.33      140.6        2.89
   8B  8,190,735,360    15.26   4.68      140.6        5.25
  14B 14,768,307,200    27.51   8.38      156.2        9.01
```

Three things follow from that table. The 0.6B and 1.7B checkpoints share the same layer geometry, so their KV cache cost is identical, and the difference between them is weight memory alone. The jump from 4B to 8B costs about 2.36 GiB of quantised weights for the same cache footprint, which is the cheapest step in the family. The 14B lands at 9.01 GiB resident at a 4k context, which is server territory, not phone territory.

The practical read: 1.7B under a 2 GB budget, 4B under 3 GiB, 8B under 6 GiB, and 14B only where you have a GPU or an 8-core box with 16 GB to spare. Below 4B, the evaluation says you are giving up general capability, so the choice is between a small model that only does Wolof-adjacent tasks well and a bigger model that needs real hardware. For a call-centre FAQ bot handling a narrow intent set, the 1.7B is the right shape. For anything that needs to reason about the answer, the report's own numbers push you to 4B or above.

One cost the release does not address is tokenisation, and the report's appendix shows the team knows about it. In a side experiment on Qwen2.5-7B, adding 380 Wolof tokens to the vocabulary raised the average score from 0.35 to 0.39. That change is not in the main recipe, and the released Qwen3 models inherit the base vocabulary unchanged at 151,936 tokens. So Wolof text still pays the base tokenizer's subword penalty on every prompt and every completion, and the report leaves that 0.04 on the table for someone else to collect. If you are building on these weights at volume, vocabulary work is where your next margin is.

## How to use this release

If you are evaluating whether to build on it, the order of operations matters more than the model choice.

1. **Check the licence against your revenue model before you download.** Non-commercial, share-alike, base-model terms stacked. Prototype inward, then talk to Orange if the prototype works.
2. **Skip 0.6B unless your task is genuinely narrow.** The report says it degrades on several skills. It is the only size that fits a cheap phone, and that is its whole case.
3. **Start at 1.7B for FAQ and intent work, move to 8B for anything generative.** The human evaluation used 8B and found Wolof quality at parity with commercial systems, with correctness and formatting a step behind.
4. **Do not ship it with tools enabled.** Function calling fell from 0.61 to 0.24, mostly from false-positive calls. If you need tools, put a validation layer in front of every call and log the rejections.
5. **Evaluate on your own Wolof, from your own users.** The card warns that no single dataset represents every dialect, register, spelling convention or population, and that part of the training data was machine-translated. A 200-example evaluation set drawn from your support tickets will tell you more than the 63-task aggregate.
6. **Keep a French or English baseline in the harness.** The adaptation cost English understanding 0.03 and French understanding 0.01. In a bilingual deployment, route by confidence rather than forcing everything through the Wolof path.
7. **Watch the six-language roadmap.** Lingala, Swahili, Darija, Mooré and Dioula are named as next. If a Swahili checkpoint appears with the same documentation quality, it will be the most useful East African release of the cycle.

## Key takeaways

| Point | Detail |
|---|---|
| What shipped | Five Qwen3-based Wolof checkpoints, 0.6B to 14B, plus a 63-task technical report |
| Announcement versus availability | Press release 17 Nov 2026; repositories public 9 Oct to 16 Nov 2026 |
| Wolof gain | Generation -0.04 to 0.22 and understanding 0.04 to 0.22 against the base, on normalised scores |
| What it cost | Function calling 0.61 to 0.24, instruction following 0.50 to 0.25 |
| Human check | 12 annotators, Elo 1051 for 8B against 1112 for Gemini-2.5-flash, Wolof quality within 8 points |
| Licence | CC-BY-NC-SA 4.0, commercial use by separate terms, Qwen3 terms stack on top |
| Deployment | 1.7B at 1.47 GiB resident for 4k context, 4B at 2.89 GiB, 8B at 5.25 GiB, 14B at 9.01 GiB |
| Biggest miss | Vocabulary work worth +0.04 average was measured and not shipped |

The release is worth the attention it got, for reasons that have little to do with the announcement. A telecom published its data mixture, its DPO caveat, its per-task scores, its annotator protocol, and a section listing what the model should not be used for. That is more disclosure than most open-weight releases from any continent carry, and it is what makes the family usable by someone who is not on the team. The licence limits who can commercialise it. The documentation is what lets everyone else decide whether to try.

## References

- Orange, "Orange releases African-language AI models to advance digital inclusion in Africa", 17 Nov 2026 — [orange.com](https://www.orange.com/en/press-release/orange-releases-african-language-ai-models-to-advance-digital-inclusion-in-africa-520346)
- Orange AI, "Wolof-Adapted Industry-Ready Open-Weight Language Models", arXiv:2610.20824 — [arxiv.org](https://arxiv.org/abs/2610.20824)
- Hugging Face model card, `Orange/Wolof-Qwen3-1.7B-v2610` — [huggingface.co](https://huggingface.co/Orange/Wolof-Qwen3-1.7B-v2610)
- Hugging Face model card, `Orange/Wolof-Qwen3-14B-v2610` — [huggingface.co](https://huggingface.co/Orange/Wolof-Qwen3-14B-v2610)
- Hugging Face API, repository creation dates and download counts for the five Wolof-Qwen3 repositories, checked 22 Nov 2026 — [huggingface.co/api/models](https://huggingface.co/api/models?author=Orange)
- TechAfrica News, "Orange Releases Five Open-Weight AI Models for Wolof Language", 18 Nov 2026 — [techafricanews.com](https://techafricanews.com/2026/11/18/orange-releases-five-open-weight-ai-models-for-wolof-language/)
- EdTech Innovation Hub, "OpenAI Foundation and Gates Foundation back five-year AI language access push for 3.4bn people", 24 Sep 2026 — [edtechinnovationhub.com](https://www.edtechinnovationhub.com/news/openai-foundation-and-gates-foundation-back-five-year-ai-language-access-push-for-34bn-people)
- Creative Commons, "Attribution-NonCommercial-ShareAlike 4.0 International" deed — [creativecommons.org](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Hugging Face repositories `vamboai/morena-1.5b-instruct`, `lelapa/InkubaLM-0.4B`, `CohereLabs/tiny-aya-earth`, `soynade-research/Oolel-v0.1`, licence metadata checked 22 Nov 2026
- Qwen Team, "Qwen3 Technical Report", arXiv:2505.09388 — [arxiv.org](https://arxiv.org/abs/2505.09388)

## Related posts

- [The Tokenizer Was the Point: Measuring MORENA, an Open 1.5B Model](/posts/morena-open-african-language-model-tokenizer/)
- [Fine-Tuning LLMs on African Language Datasets](/posts/fine-tuning-african-language-llms/)
- [Swahili NLP: Building Language Models for African Languages](/posts/swahili-nlp/)
- [Low-Resource NLP: Transfer Learning and Data Augmentation](/posts/low-resource-nlp/)
- [Nineteen Languages, One Download: The TranslatePsy-AfriSLM Release](/posts/translatepsy-afrislm-offline-translation/)
