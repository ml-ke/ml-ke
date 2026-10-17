---
title: "Same Script, Fifty-One Answers: What Morocco's Open Darija Release Actually Solves"
date: 2026-10-17 00:00:00 +0300
categories: [AI in Africa, LLM]
tags: [morocco, darija, mistral, voxtral, speech recognition, language identification, open weights, multilingual nlp, mena]
math: false
image:
  path: /assets/img/cover-morocco-darija-speech-stack.webp
  alt: A row of words in different languages all written in the same Arabic script converging on a classifier rack of many labelled slots, with one slot lit and a live waveform feeding a 480 millisecond tag
---

> **In one paragraph:** On Tuesday 29 September 2026, Morocco's Ministry of Digital Transition and Administrative Reform released two open-source language tools built with Mistral: a 51-class Arabic language and dialect classifier, and a Darija speech recogniser built on Voxtral. Both are Apache 2.0, and both now sit in Mistral's Hugging Face organisation. The weights are only half the contribution. The release also ships the parts that usually stay inside the lab: two evaluation sets built with the ministry from real recordings, a published scoring normaliser, and a classifier designed to sit in front of the recogniser rather than beside it.
{: .prompt-info }

This lane has covered four open multilingual releases in five weeks: [MORENA's tokenizer](/posts/morena-open-african-language-model-tokenizer/), the [TranslatePsy offline translation family](/posts/translatepsy-afrislm-offline-translation/), and before those the [sign-language corpus behind Terp 360](/posts/african-sign-language-avatar-data/). This one is different in kind. Those were text-in, text-out. This is audio in, and the first thing it ships is a decision about which language the audio is in before a single word is transcribed.

## What landed

The ministry announced the tools on 29 September (Morocco World News, TechReview Africa, Ecofin all carry the same release), positioning them under the Maroc Digital 2030 strategy and the AI Made in Morocco roadmap. The Hugging Face checkpoints followed in the first half of October.

| Artifact | What it is | Published numbers |
|---|---|---|
| `mistralai/LIDstral-Arabic` | Text classifier over Arabic-script languages: fastText one-vs-all base, a 64-unit LayerNorm MLP stacker over 144 derived features, then per-class isotonic calibration | Darija F1 **88.67**, against **72.65** for LahjatBERT ALDi CL and **71.28** for GlotLID v3; evaluated on 84,870 Arabic examples |
| `mistralai/Voxtral-Mini-4B-Realtime-Arabic` | Streaming ASR fine-tuned from Voxtral-Mini-4B-Realtime-2602, ~4.4B parameters in BF16, 16 kHz causal encoder, configurable transcription delay | Average CER **8.82%** across seven Arabic benchmarks at a **480 ms** delay; the offline sibling it is measured against sits at **7.91%** |
| `normalization.py` | The scoring normaliser used for every CER figure above, published in the model repository | Covers dialect spelling, clitics, loanwords and spoken numbers |
| ONNX checkpoint + WebGPU demo | Community ONNX export of the Arabic ASR, running in a browser Space | No host required to try it |

Two new benchmarks sit inside those seven: **ISMA** and **Darija in the Wild**, both co-developed with the ministry. That is the difference between a benchmark and a demo.

## The gate matters more than the transcript

A speech recogniser for Moroccan Darija is not a translation problem. Darija is an Arabic dialect that diverges from Modern Standard Arabic in vocabulary, morphology and phonology, and it is routinely mixed with French and English inside a single sentence. Ecofin's write-up makes the operational point plainly: the tools adapt existing open models to a local linguistic environment rather than building a foundation model from nothing.

That framing explains the shape of the release. The classifier is not a novelty. It is the component that decides which audio goes where.

The hard part is visible in the label set. LIDstral's card lists 46 language codes grouped by family: sixteen Arabic varieties under MSA, Maghreb, Gulf, Nile Valley, Levantine and Chadian headings; two Berber languages; five Turkic; eleven Indo-Iranian; ten African; two Austronesian, Malay among them in Jawi script. The model's output layer has 51 labels, and in the printed output a class is sometimes a country (`Morocco`, `Algeria`, `Iraq`) and sometimes a language (`MSA`, `Urdu`, `Punjabi_Shahmukhi`). The card states the design reason for the non-Arabic half directly: those languages were included in training "to help the model distinguish languages that share a script and avoid classifying text as Arabic based on its script alone."

That sentence is the whole engineering argument. Persian, Urdu, Pashto, Sindhi, Kashmiri, Punjabi in Shahmukhi and Sorani Kurdish are written in Arabic script. So are Hausa, Wolof, Fula and Kanuri in their Ajami tradition. A script detector sees one language family where there are dozens, and Darija against MSA is the same problem one level down: identical alphabet, different words.

The card's preprocessing step shows the same concern from the other side. It strips web and ASR artefacts, diacritics, decorative elongation and repeated punctuation, and it preserves alef variants. That last choice is deliberate: for identification, keeping أ against ا is signal, because the variant tells you something about how the writer spells. For scoring a transcript, you usually want the opposite, which is where the normaliser comes in.

Read the shipped `classify.py` and one feature stands out. Among the 144 the stacker consumes is a hand-built Saudi-versus-other-Gulf margin, computed as the Saudi score minus the best score among Bahrain, Kuwait, Qatar, the UAE, Oman and Yemen. Someone found that confusion in development and gave the stacker a feature for it, which is a more useful disclosure than a single average.

## The number that decides whether you can trust the number

CER is a character-level edit distance divided by the length of the reference. It sounds atomic. It is not, because the strings you compare are whatever your normaliser produces.

Mistral's own card says it outright: the 8.82% figure is computed "with normalization co-developed with MTNRA, covering Arabic dialect-specific spelling, clitics, loanwords, and spoken numbers", and "scores may differ under other normalization methods." The code that does it ships as `normalization.py` in the model repository, which converts a caveat into something a reviewer can rerun.

For anyone comparing vendors, this is the practical lesson of the release. Two teams can run the same checkpoint over the same audio, publish different CER numbers, and both be correct. Before you compare an 8.82% to anything else, find out which normalisation produced it. The gap between the realtime model and its offline sibling is 0.91 percentage points on identical data and scoring, and that comparison is the vendor's own, which is what makes it usable: it measures what sub-second latency costs on this language family rather than what one vendor's marketing costs another's.

The second caveat is structural. Two of the seven benchmarks were built with the ministry whose project the model serves. That is not a scandal, it is how a domain benchmark gets made when the domain has no public test set, but it does mean part of the suite has no external history. OrcaRouter's teardown puts the whole evidence base in one line: nothing on that card has been reproduced by a third party yet, and there is no hosted endpoint to reproduce it against.

In operator units, that headline reads: at 8.82% CER a 1,000-character transcript carries roughly 88 characters for a reviewer to fix, and at 7.91% the offline model asks for about 79. The card plots performance against delay in an image and quotes only the 480 ms point in text, so if your product can absorb a two-second delay, that number is yours to measure.

## We ran the classifier

The classifier is small enough to run on a laptop, so we did. The bundle is a 1.07 GB fastText model plus a 62 KB MLP checkpoint and a 21 KB calibration pickle, and `classify.py` will read a sentence from the command line, a JSONL file, or a list in Python. We installed it in a Python 3.12 environment with `fasttext-wheel`, NumPy, scikit-learn and CPU-only PyTorch, and ran ten inputs through the Python API.

```bash
# the classifier bundle and its three checkpoints
pip install fasttext-wheel==0.9.2 numpy==1.26.4 scikit-learn==1.4.0 "torch>=2.6,<3"
hf download mistralai/LIDstral-Arabic --local-dir LIDstral-Arabic
cd LIDstral-Arabic

# one sentence, or a JSONL file of {"text": ...} objects
python classify.py --text "ياك نتا بخير؟ آش خبارك مع الخدمة؟"
python classify.py --input batch.jsonl > predictions.jsonl
```

Ten inputs through the Python API (`ArabicLID(".").load()`, then `predict`):

| Input | Script | We expected | Returned | Confidence | Runner-up |
|---|---|---|---|---|---|
| Darija, the card's own example | Arabic | Morocco | **Morocco** | 0.952 | Algeria 0.013 |
| MSA, the card's own example | Arabic | MSA | **MSA** | 0.730 | Iraq 0.120 |
| MSA, a ministry-style news sentence | Arabic | MSA | **MSA** | 0.698 | Mauritania 0.121 |
| Egyptian colloquial | Arabic | Egypt | **Egypt** | 0.859 | Palestine 0.048 |
| Gulf colloquial | Arabic | Saudi_Arabia | **Saudi_Arabia** | 0.440 | Bahrain 0.112 |
| Persian | Arabic | Persian | **Persian** | 0.980 | Kashmiri 0.010 |
| Urdu | Arabic | Urdu | **Urdu** | 0.993 | Punjabi_Shahmukhi 0.005 |
| Swahili | Latin | Swahili | **Swahili** | 0.949 | Yemen 0.023 |
| Latin-script Darija (Arabizi) | Latin | out of scope | Wolof | 0.146 | Afar 0.123 |
| English | Latin | out of scope | Sindhi | 0.889 | Urdu 0.087 |

Eight of ten matched. The first row is the calibration check that matters: the card's Darija example comes back at 0.952, against the 0.9523 the card prints as illustrative output, so the checkpoint on the Hub behaves like the one behind the card.

The closest call among the in-scope inputs is more instructive than the clean hits. Gulf colloquial scored lowest of the seven Arabic-script inputs at 0.440, with Bahrain taking 0.112, which is the confusion the shipped `classify.py` builds a dedicated feature for. A country-level label is a routing guess, not a fact about the speaker.

Swahili came back at 0.949, the quietly useful part for anyone reading this in Nairobi. The release is framed around Darija, but the label set also covers Hausa, Wolof, Oromo, Tigrinya, Kanuri, Afar, Dinka and N'Ko, plus two Tamazight labels, one in Tifinagh script. That is broader than the "text written in Arabic script" line on the card suggests.

The two inputs we expected to be rejected were not. English was labelled Sindhi at 0.889, and Latin-script Darija landed on Wolof at 0.146 with Afar close behind. Neither is a bug, since the card says the model "always assigns one of its 51 supported classes" and may misclassify unsupported or mixed-language text. But an 0.889 on a language it never trained on is the argument for an abstention layer in one row: route on labels alone and that English sentence goes to a Sindhi pipeline with high confidence. Two runs of the same script produced byte-identical output, so these numbers are re-checkable rather than sampled.

## How to put it in front of a recogniser

The deployment shape the release implies is a gate. Classify the text first, then choose the transcriber, then let the normaliser decide how you score the result.

```text
audio ──► ASR (Arabic realtime, 480 ms) ──► text ──► LIDstral ──► route
                                                          ├─ Darija-labelled and confident → Darija post-processing, Darija reviewers
                                                          ├─ MSA-labelled → standard Arabic pipeline
                                                          └─ low confidence / high entropy → queue for a human, or re-run offline
```

Four details matter if you copy that diagram.

**Calibrate the abstain rule yourself.** The card is explicit that the model "always assigns one of its 51 supported classes" and may therefore misclassify unsupported languages or mixed-language text. A 51-class classifier with no abstain path will label Kinyarwanda as something. Per-class isotonic calibration gives you probabilities you can threshold, but the threshold is yours to choose against your own cost of being wrong.

**Do not assume the label is a dialect.** The printed labels are a mix of country names and a language name (`Morocco`, `Algeria`, `Mauritania`, `Libya`, `Iraq`, `Egypt`, `MSA`). Country-level routing is what most products need, and it is what the card delivers. It is not a linguistic claim about the varieties inside a country.

**Treat code-switching as the default case, not the edge case.** The ASR card is built for speakers who alternate between Arabic, French and English inside a conversation, and the ministry's stated use cases include public services, customer support and document digitisation, which is exactly where that happens.

**Run it as a batch job, not a microservice.** `classify.py` accepts JSONL on stdin, writes one JSON object per line, defaults to batches of 256 and pins PyTorch to a single thread. That shape suits a nightly sweep of yesterday's contact-centre exports better than a per-request hop in a latency budget. For inline routing, load `ArabicLID` once in your process and call `predict` on batches instead of shelling out per sentence.

## Five checks before you trust the router

The release gives you a classifier and an evaluator. It does not give you a number for your audio, on your channel, with your speakers.

1. **Record on the channel you will ship on.** A model evaluated on broadcast audio and deployed on 8 kHz phone calls is being tested on the wrong distribution. Two hundred utterances per dialect is a reasonable start; matching the microphone matters more than adding volume.
2. **Label with two people and report agreement.** If your annotators disagree on whether a sentence is Darija or MSA, no classifier can be scored honestly against that label set. The [annotator agreement audit](/posts/annotator-agreement-before-you-train/) shows where the ceiling sits even when every annotator is careful.
3. **Score CER twice.** Once with the normaliser shipped in the repository, once with your own, and keep both. The difference is a definition, not noise.
4. **Measure the abstain rate at your threshold.** Sweep the confidence cutoff and plot how much traffic falls through to a human. Where the classifier stops being useful is a business decision, not a model property.
5. **Score the task, not the transcript.** Whether the caller's request was resolved is the metric that pays for the system. CER is the proxy that helps you debug it.

That list is generic on purpose. It applies to a Darija contact centre, a Swahili voice assistant and an Arabic claims line equally, and it is the part of this release that transfers to a team with no interest in Morocco at all.

## What the release does not give you

- **No independent evaluation.** Every figure above is the vendor's or the ministry's. That is normal at release, and a reason to run your own sample before betting a roadmap on it.
- **No production deployment to point at.** Ecofin is careful: the use cases "represent potential integration avenues rather than large-scale operational deployments."
- **No price, because there is no service.** Both checkpoints are self-hosted, through vLLM with the `mistral-common` audio extras or Transformers 5.2.0 and later. The GPU bill is yours, and the card ships BF16 only, so any int8 or fp8 build and its accuracy cost are also yours.
- **A language boundary.** The ASR targets Arabic dialects and MSA; for anything else the card points you back to the base Voxtral model.
- **One headline number for 51 classes.** The Darija F1 is quoted in text because Darija is the ministry's target. The rest of the label set lives in a chart, so figures for the Berber, Turkic or South Asian classes are yours to measure.

None of those gaps is unusual for a sovereign-AI release in its first fortnight. What is unusual is that the evaluation normalisation shipped alongside the weights, because that is what makes the numbers checkable at all.

## Why a ministry building this is the story

There is a version of AI localisation where a global vendor ships a model, a local team writes a prompt, and the vendor's benchmark quietly becomes the country's benchmark. Morocco's release is the other version: the ministry is named on two of the seven ASR benchmarks, the scoring code is in the repository, and the licence is Apache 2.0 rather than a research-only grant. Three decisions carry that, and each is procurement rather than research. The state commissioned a capability instead of buying a licence, so it is not renting access to its own language. It insisted on test data from its own speakers, which is why ISMA and Darija in the Wild exist. And it published the scoring code, which turns a claim into something a competitor, a regulator or a journalist can rerun at no cost.

The comparison with this blog's earlier coverage is sharper. [MORENA](/posts/morena-open-african-language-model-tokenizer/) made its argument through a tokenizer that does not punish African text with a byte premium. LIDstral makes its argument through a gate that keeps one dialect from being silently overwritten by another. Both are about the same thing: the cost of a language not being in the default path, paid in tokens in one case and in mis-transcribed speech in the other.

There is a limit to that optimism. Morocco's release is two checkpoints for one dialect family, and the ministry's own framing treats them as building blocks for its AI marketplace and public services rather than a finished system. A country that stops at "we have an open model" has a press release. A country that keeps the test set, the normaliser and the abstention policy has something it can put into a service-level agreement.

## Key takeaways

| Question | Answer from this release |
|---|---|
| What shipped | 51-class Arabic-script language and dialect classifier, plus a 4.4B realtime Darija ASR, both Apache 2.0 |
| Headline quality | Darija F1 88.67 (vs 72.65 strongest baseline); ASR CER 8.82% at 480 ms delay |
| Realtime cost of latency | 0.91 percentage points against the offline sibling, on identical data and scoring |
| Where the risk sits | The CER normaliser. Numbers are only comparable under the same one |
| What to run first | Your own audio through the classifier, checking confidence and abstention before routing |
| What is still missing | Independent evaluation, a hosted endpoint, quantised builds, and any production deployment |

## References

- Mistral AI, [Voxtral-Mini-4B-Realtime-Arabic model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-Arabic) — CER figures, 480 ms delay, ISMA and Darija in the Wild, `normalization.py`, licence
- Mistral AI, [LIDstral-Arabic model card](https://huggingface.co/mistralai/LIDstral-Arabic) — 51-class pipeline, 144 features, Darija F1 and baselines, preprocessing and limitations
- Morocco World News, [Morocco, Mistral AI release first open-source Darija AI tools](https://www.moroccoworldnews.com/2026/09/340193/morocco-mistral-ai-release-first-open-source-darija-ai-tools/) — 29 September 2026 announcement, Maroc Digital 2030 framing
- TechReview Africa, [Morocco, Mistral AI release first open-source AI tools under AI made in Morocco roadmap](https://techreviewafrica.com/news/7504/morocco-mistral-ai-release-first-open-source-ai-tools-under-ai-made-in-morocco-roadmap) — ministry statement, intended sectors
- Ecofin Agency, [Morocco and Mistral AI launch open-source tools for local dialect processing](https://www.ecofinagency.com/news-digital/0110-59399-morocco-and-mistral-ai-launch-open-source-tools-for-local-dialect-processing) — release date, "potential integration avenues" caveat, Apache 2.0 lineage
- OrcaRouter, [Voxtral-Mini-4B-Realtime-Arabic: a quiet Arabic ASR drop](https://www.orcarouter.ai/blog/voxtral-mini-4b-realtime-arabic-quiet-release) — vendor-claim audit, missing independent evaluation, BF16 only
- iAfrica, [Morocco releases first open-source Darija AI tools from Mistral partnership](https://iafrica.com/morocco-releases-first-open-source-darija-ai-tools-from-mistral-partnership/) — partnership context
- Hugging Face, [Voxtral-Mini-4B-Realtime-Arabic ONNX](https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-Arabic-ONNX) and the [WebGPU demo Space](https://huggingface.co/spaces/mistralai/Voxtral-Mini-4B-Realtime-Arabic-WebGPU)
- Evaluation corpora named on the classifier card: [NADI](https://figshare.com/articles/dataset/NADI_dataset/27935430), [MADAR](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), [QADI](https://github.com/qcri/QADI), [Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca), [FLEURS](https://huggingface.co/datasets/google/fleurs)

## Related posts

- [The Tokenizer Was the Point: Measuring MORENA](/posts/morena-open-african-language-model-tokenizer/) — what a vocabulary costs a language
- [TranslatePsy-AfriSLM: offline translation for 19 African languages](/posts/translatepsy-afrislm-offline-translation/) — the on-device translation family
- [The corpus is the product: African sign language data](/posts/african-sign-language-avatar-data/) — where the training data has to come from
- [Swahili NLP](/posts/swahili-nlp/) — the low-resource pipeline, one language at a time
