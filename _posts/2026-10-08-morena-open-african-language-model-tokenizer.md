---
title: "The Tokenizer Was the Point: Measuring MORENA, an Open 1.5B Model for 12 African Languages"
date: 2026-10-08 00:00:00 +0300
categories: [AI in Africa, LLM]
tags: [morena, vambo ai, tokenizer, african languages, swahili, open weights, token efficiency, foundation model, multilingual nlp]
math: false
image:
  path: /assets/img/cover-morena-open-african-language-model-tokenizer.webp
  alt: Swahili text passing through a tokenizer gate and emerging as chunky tokens, beside a per-character cost meter
---

> **In one paragraph:** On 18 September 2026, Vambo AI, a South African startup founded in April 2023, released MORENA, an Apache-2.0 language model trained from scratch for twelve African languages plus English, French and code. Most coverage led with the headline benchmark: 1.408 bits per byte on African text, the lowest of 26 models measured, beating an 8B rival. That result is real, and it is in the model card. But the number that changes what a builder pays is upstream of every benchmark: the tokenizer. This post reads the release through that lens and measures the effect here rather than repeating the vendor's table.
{: .prompt-info }

This is the third open multilingual release covered in this series in three weeks, so it is worth saying what is different about this one. The [TranslatePsy-AfriSLM release](/posts/translatepsy-afrislm-offline-translation/) was a translation model you can run offline on a laptop, and [MiMo-V2.6](/posts/mimo-v26-open-release-builders/) was a Chinese frontier release with an unusually open training loop. MORENA is neither. It is a small foundation model built specifically so that African languages are not an afterthought in the vocabulary, and it is the tokenizer, not the parameter count, that carries that decision.

## What actually shipped

| Item | Detail |
|---|---|
| Released | 18 September 2026 (Vambo AI technical report, *MORENA: An African Foundation Model*) |
| Licence | Apache 2.0, open weights |
| Sizes | 1.5B (base and instruct), 0.5B mini, 0.2B nano |
| Quantised builds | GGUF `f16`, `Q8_0` and `Q4_K_M` for the instruct model; a community MLX 4-bit quant for Apple silicon |
| Languages | ChiShona, Kiswahili, Hausa, Yorùbá, Igbo, isiZulu, isiXhosa, Kinyarwanda, Setswana, Afrikaans, isiNdebele and Nigerian Pidgin, plus English and French |
| Architecture | 28 layers × 2,048, GQA 16/4, SwiGLU 6,144, RoPE θ = 500,000, 4,096-token context, tied embeddings |
| Vocabulary | 65,536 entries, trained on the target language mix |
| Training | 251.7B tokens of pretraining, then 63B of mid-training — 315B tokens seen, 12,661 A100 GPU-hours |

The name is not an acronym. Per the model card, *morena* is Sesotho and Setswana for a king, a lord or a chief, the register you would use for someone honoured. The ambition is a model that addresses these languages as first-class, not as translation targets.

One detail from the training write-up deserves more attention than it usually gets. The mixture moved in three regimes as machine-translated material landed: African text was 24.8% of tokens seen (14.8% machine-translated) for the first 25,304 steps, 39.1% (31.0% machine-translated) to step 60,000, and 50.2% (41.6% machine-translated) during mid-training. Nine languages were machine-translated from English documents, which is 24% of pretraining tokens and 28% including mid-training. The authors disclose it and the numbers are specific, which is more than most releases manage. It is also the release's biggest open question: translated text can extend coverage where native text is scarce, but it is not the same as natural local usage.

## The benchmark everyone quoted, and the number that actually pays the bill

The figure in every headline is **1.408 bits per byte**, the mean across the twelve African languages, measured on the base checkpoint. The model card lists it as the lowest of 26 models measured, ahead of Lugha-Llama-8B at 1.423 and gemma-3-12b-it at 2.159. The instruction-tuned sibling lands at 1.441 on the same metric. On translation, the instruct model reports a FLORES+ chrF++ of 45.8 for English into five African languages, against 37.8 for MADLAD-400-3B and 36.8 for Lugha-Llama-8B.

Two things are true about those numbers at the same time. They are developer-published, and the parameter advantage over a 12B model is large. Vambo's co-founder Isheanesu Misi told Disrupt Africa the model "beats a Google model eight times its size on African text modelling", that it is "over three times the size of the second largest African language model", and that it demonstrates other African efforts "do not necessarily need billion dollar budgets". Those are company claims about a company's artifact, and they should be read that way. What the numbers do not require you to take on trust is the mechanism, because the mechanism is arithmetic you can check.

Every token a model emits costs money and occupies context. If a vocabulary was built around English and code, then Swahili, Yoruba or Hausa text arrives as more, smaller pieces. The same sentence costs more tokens, and your 4,096-token window holds less of it. Vambo's stated fix was to choose the tokenizer, the data mixture and the language list before training began, after comparing vocabulary sizes for cost and efficiency. The card's claims: MORENA encodes African text with 1.39 times fewer tokens than Gemma 3 and 1.53 times fewer than Llama 3.2 on identical passages, while African text still costs 0.249 tokens per byte against 0.234 for English, about 6% more, a gap the team says it cannot fully explain.

That last sentence is the kind of thing a benchmark table usually omits, and it is the reason this release is worth more than its leaderboard position.

## Measure the tokenizer yourself

You do not need the weights to test the cost claim. The tokenizer is a 4.7 MB JSON file, and it is enough to answer the only question that matters commercially: *does this vocabulary charge me more for Swahili than for English?*

{% raw %}
```python
"""Tokens per UTF-8 byte: MORENA's tokenizer against two English-centric ones."""
import urllib.request

import tiktoken
from tokenizers import Tokenizer

MORENA_TOK = "https://huggingface.co/vamboai/morena-1.5b-base/resolve/main/tokenizer.json"

# First sentences of the Wikipedia summaries for Nairobi (en/sw) and Nigeria (ha/yo),
# fetched 2026-10-08 and frozen here so these numbers reproduce exactly.
TEXTS = {
    "English": "Nairobi is the capital and largest city of Kenya, located in the south-central part of the country.",
    "Swahili": "Nairobi ni mji mkuu na jiji kubwa zaidi la Kenya, lililoko katika sehemu ya kusini-kati ya nchi.",
    "Hausa": "A Gwamnatance Tarayyar Najeriya kasa ce da take a Afirka ta yamma.",
    "Yoruba": "Nàìjíríà jẹ́ Orílẹ̀-èdè Olómìnira Ìjọba Àpapọ̀ ilẹ̀ Nàìjíríà jẹ́ orílẹ̀-èdè ìjọba àpapọ̀ olómìnira.",
}

req = urllib.request.Request(MORENA_TOK, headers={"User-Agent": "tokenizer-demo/1.0"})
morena = Tokenizer.from_str(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))
refs = {"o200k": tiktoken.get_encoding("o200k_base"), "cl100k": tiktoken.get_encoding("cl100k_base")}

print(f"MORENA vocab: {morena.get_vocab_size()}  |  o200k vocab: {refs['o200k'].n_vocab}"
      f"  |  cl100k vocab: {refs['cl100k'].n_vocab}")
print(f"{'language':8s} {'bytes':>6s} {'MORENA':>8s} {'tpB':>7s} {'o200k':>7s} {'tpB':>7s} {'cl100k':>7s} {'tpB':>7s}")
tpB = {}
for lang, text in TEXTS.items():
    nb = len(text.encode("utf-8"))
    m = len(morena.encode(text).ids)
    o = len(refs["o200k"].encode(text))
    c = len(refs["cl100k"].encode(text))
    tpB[lang] = (m / nb, o / nb, c / nb)
    print(f"{lang:8s} {nb:6d} {m:8d} {m/nb:7.3f} {o:7d} {o/nb:7.3f} {c:7d} {c/nb:7.3f}")

print()
print("Cost penalty vs English, tokens per byte ÷ English's")
print(f"{'language':8s} {'MORENA':>8s} {'o200k':>8s} {'cl100k':>8s}")
for lang in TEXTS:
    if lang != "English":
        r = [tpB[lang][i] / tpB["English"][i] for i in range(3)]
        print(f"{lang:8s} {r[0]:8.2f} {r[1]:8.2f} {r[2]:8.2f}")

afr = [k for k in TEXTS if k != "English"]
for i, name in enumerate(["MORENA", "o200k", "cl100k"]):
    mean = sum(tpB[k][i] for k in afr) / len(afr) / tpB["English"][i]
    print(f"{name} mean penalty across the three African samples: {mean:.2f}x")
```
{% endraw %}

Run it with `uv run --with=tokenizers --with=tiktoken python3 tokenizer_cost.py`. The comparison is against GPT-4o's `o200k_base` and GPT-4's `cl100k_base`, the two vocabularies most readers will already be paying for. Verbatim stdout:

```
MORENA vocab: 65536  |  o200k vocab: 200019  |  cl100k vocab: 100277
language  bytes   MORENA     tpB   o200k     tpB  cl100k     tpB
English      99       23   0.232      21   0.212      21   0.212
Swahili      96       23   0.240      27   0.281      38   0.396
Hausa        66       15   0.227      20   0.303      25   0.379
Yoruba      148       33   0.223      60   0.405      79   0.534

Cost penalty vs English, tokens per byte ÷ English's
language   MORENA    o200k   cl100k
Swahili      1.03     1.33     1.87
Hausa        0.98     1.43     1.79
Yoruba       0.96     1.91     2.52
MORENA mean penalty across the three African samples: 0.99x
o200k mean penalty across the three African samples: 1.56x
cl100k mean penalty across the three African samples: 2.06x
```

Read the bottom block, not the row-by-row counts. The absolute tokens-per-byte figures are not where MORENA wins. Its 65,536-entry vocabulary is a quarter the size of `o200k`'s 200,019, so on plain English it is fractionally *worse* (0.232 against 0.212). That is expected and it is the trade-off every small-vocabulary model makes: a tighter vocabulary spends more tokens on English to spend far fewer on everything else.

The penalty ratio is the tell. English-centric vocabularies charge a surcharge for African scripts: Swahili costs 1.33 times more per character under `o200k` and 1.87 times more under `cl100k`; Yoruba costs 1.91 and 2.52 times more respectively, because its diacritics and tone marks fragment aggressively. Under MORENA's tokenizer the same three samples cost 0.99 times the English rate: the surcharge is gone. In production terms, a Swahili customer-support queue or a Yoruba document pipeline tokenised with `cl100k` is paying roughly double what the English equivalent costs for the same information, and that premium is a property of the vocabulary, not of the task.

Treat the size of the effect as indicative, not precise. Four one-sentence samples are not a benchmark, and the vendor's own measurement over full corpora is 1.39× against Gemma 3 and 1.53× against Llama 3.2. What this block establishes is the direction and the mechanism on your own machine, in about forty lines with no GPU.

## What the model card says the model cannot do

The most unusual page in this release is the section of the model card headed "What it is not good at", and it is worth reading in full before deploying anything. From the instruction-tuned card:

- **Retrieval-augmented QA is at chance** in African languages, even though the model demonstrably reads the passage (grounding 0.73). If your product is RAG over Swahili documents, the released model is not ready for it.
- **Grounded generation is fully faithful to given facts in 23% of attempts**, down from 31% in the previous version.
- **Over-refusal persists.** The model answers 58.9% of ordinary benign requests well; most failures are refusals. The card's example: an earlier version once refused to recommend a dry cleaner, citing "illegal substances or services".
- **Multiple-choice comprehension in African languages is at chance** for this model and for every model under 12B that Vambo measured. That bounds the size class, not just this checkpoint.
- **Tool calling is measured with the tool marker prefilled.** Left to decide for itself, the model almost never calls a tool, so agentic use is not what this checkpoint is for.
- **Every capability number is a model judging a model.** The judge is `google/gemma-3-12b-it`, and no native speaker has yet rated an answer. Safety scoring covers 3,341 attempts across 13 languages and 11 categories, with 89.0% handled well; the weakest languages are Igbo (75%), Yoruba (82%) and Setswana (83%).

A model card that reports its own regression, states which evaluator produced its scores, and names the languages where safety is weakest is doing the work the genre was invented for. If you are choosing a model for a Kenyan or Nigerian deployment, that list tells you more than any leaderboard row: translate, summarise and fine-tune, yes; unsupervised RAG and autonomous tool use, not yet.

## How to apply this release

The released artifacts support three uses well, in rough order of how much the repo helps you do them: fine-tune the base checkpoint, run the instruct model locally, or lift the tokenizer on its own. A hosted API is a fourth option, with different trade-offs.

**Fine-tune the base checkpoint.** Misi names fine-tuning as "our recommended way to use this" for startups and researchers, and the base card agrees. It is explicitly "not a chat model" and exists as a starting point for continued pretraining and domain adaptation. That makes it a candidate for the workflow in [Fine-Tuning LLMs on African Language Datasets](/posts/fine-tuning-african-language-llms/), with one difference worth noting: because the vocabulary was trained on the target mix rather than extended onto an English tokenizer, you inherit a tokenizer that already encodes Swahili and Yoruba at roughly English cost. You do not have to budget for re-learning embeddings for the base vocabulary.

**Run the instruct model locally.** The GGUF builds (`f16`, `Q8_0`, `Q4_K_M`) exist for exactly this, and the parameter count puts it in territory a laptop or a modest edge box can hold. The measured reference point from a sibling post in this series is a 0.8B Arabic model at 28.7 tokens/s on a 2019 mid-range CPU with 1 GB peak RSS; a 1.5B Q4_K_M on the same class of machine is the same order of magnitude, not a data-centre workload.

**Mind the chat format.** The instruct checkpoint does not use the `<|user|>`-style markers that most inference stacks assume. Turns are marked by two reserved tokens: `<reserved_0>` (id 3) opens a user turn, `<reserved_1>` (id 4) opens the assistant turn. The model card is explicit that the familiar special strings "are not in the vocabulary and produce degenerate output". The repo ships a `load_example.py` with a full worked prompt; use it rather than hand-rolling a template, because a wrong marker here produces text that looks like a model failure rather than a formatting one.

**Use the tokenizer on its own.** Even if you never load MORENA's weights, the tokenizer is worth a look. If you are serving Swahili, Hausa or Yoruba text through a general-purpose model, the block above is a rough quote for the vocabulary premium you are paying, and the cheapest win available is often a finer-grained routing decision: send short, high-volume African-language traffic to a vocabulary that codes it efficiently, and keep the large frontier model for the tasks that need it.

The premium shows up twice in a bill. Per-token pricing means a 1.56× tokens-per-character rate is a 56% surcharge on the text-processing portion, before any caching helps. Context is the quieter cost: at the rates measured above, a 4,096-token window holds roughly 17,000 bytes of Swahili under MORENA's tokenizer and about 10,300 under `cl100k`, so the same window carries roughly 65% more of the document. Truncation you blame on a small context window is sometimes a vocabulary decision instead.

The OpenAI-compatible alternative, Vambo's own hosted API, covers translation and speech across 60+ languages, but that is a different proposition from open weights and it puts you back on someone else's tokenizer and pricing.

## Why this matters beyond one release

Two decades of African-language NLP have mostly meant adapting someone else's model: continuing a Llama variant, or bolting extra tokens onto an English vocabulary, as the [low-resource](/posts/rag-low-resource-african-languages/) and [Swahili NLP](/posts/swahili-nlp/) posts here describe. MORENA's contribution is not that it solves African-language AI. Its own card says it does not. It is that the vocabulary, the mixture and the evaluation were decided together, from scratch, for these languages, and the resulting artifact is small enough and open enough for a Nairobi or Lagos team to fine-tune without a funding round.

The honest caveats stand: the benchmarks are developer-published, the variance across tasks is real, nine of the target languages lean on machine-translated data, and the safety evaluation has no native-speaker ratings yet. A 1.5B model that is at chance on African-language multiple choice and RAG is a foundation, not a product. But foundations are the thing this ecosystem has been short of, and the tokenizer is the part of this one that will quietly change the economics of every African-language pipeline that adopts it.

## Key takeaways

| Takeaway | Evidence |
|---|---|
| The release is real and open | Apache 2.0, eight checkpoints from 0.2B to 1.5B on Hugging Face, GGUF builds included |
| The headline benchmark is developer-published | Base checkpoint 1.408 bits per byte, lowest of 26 models measured; instruct 1.441 |
| The claim with a checkable mechanism is the tokenizer | Measured here: African samples cost 0.99× the English rate under MORENA, 1.56× under GPT-4o, 2.06× under GPT-4 |
| Small vocabulary is a deliberate trade | 65,536 entries vs 200,019 for `o200k`; marginally worse on English, far better on the target languages |
| The model card is unusually honest | RAG at chance, grounded generation 23%, benign requests answered well 58.9%, scores judged by another model |
| Fine-tuning, not chat, is the intended use | The base checkpoint is the recommended starting point; the instruct card flags over-refusal and prefilled-marker tool calls |

## References

- [Africa's AI moment has arrived — this 1.5B model built from scratch for 12 African languages beats Google, Meta, and Alibaba while being 8x smaller](https://www.techradar.com/pro/africas-ai-moment-has-arrived-this-1-5b-model-built-from-scratch-for-12-african-languages-beats-google-meta-and-alibaba-while-being-8x-smaller) — TechRadar, 23 September 2026
- [SA's Vambo AI releases language model trained for 12 African languages](https://disruptafrica.com/2026/10/07/sas-vambo-ai-releases-language-model-trained-for-12-african-languages/) — Disrupt Africa, 7 October 2026
- [vamboai/morena-1.5b-base model card](https://huggingface.co/vamboai/morena-1.5b-base) and [vamboai/morena-1.5b-instruct model card](https://huggingface.co/vamboai/morena-1.5b-instruct) — architecture, training mixture, benchmark table and limitations
- [MORENA collection on Hugging Face](https://huggingface.co/collections/vamboai/morena) — the eight released checkpoints, including the GGUF and MLX builds
- [Africa's MORENA Model Bets on Smaller Local-Language AI](https://streamlinefeed.co.ke/news/africas-morena-model-bets-on-smaller-local-language-ai) — Streamlinefeed, 30 September 2026, on developer-reported benchmarks and the machine-translation caveat
- [Vambo AI](https://vambo.ai/) — platform and API documentation
- [OpenAI tiktoken](https://github.com/openai/tiktoken) and the [Hugging Face tokenizers](https://huggingface.co/docs/tokenizers/) library used in the measurement above

## Related posts

- [Nineteen Languages, One Download: Reading the TranslatePsy-AfriSLM Release Past the Headline](/posts/translatepsy-afrislm-offline-translation/)
- [One Licence, Three Checkpoints: What the MiMo-V2.6 Open Release Hands Builders](/posts/mimo-v26-open-release-builders/)
- [Fine-Tuning LLMs on African Language Datasets](/posts/fine-tuning-african-language-llms/)
- [Swahili NLP: Building Language Models for African Languages](/posts/swahili-nlp/)
- [Building RAG Systems for Low-Resource African Languages](/posts/rag-low-resource-african-languages/)
