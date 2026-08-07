---
title: "Evaluating LLMs: Benchmarks That Matter for African Use Cases"
date: 2026-08-07 00:00:00 +0300
categories: [AI Engineering, LLM]
tags: [evaluation, benchmarks, africa-nlp, llm-eval, african-languages, model-comparison]
image:
  path: /assets/img/cover-series-practical-playbook.webp
  alt: A mix of benchmark leaderboard scores alongside African language text samples and flag icons
---

## The Benchmark Blind Spot

A model scores 90% on MMLU. It reasons through physics problems, parses legal documents, and writes competent Python. You deploy it for a Kenyan agritech startup — and it cannot reliably answer "Je, mboga hizi zinahitaji maji kiasi gani?" (How much water do these crops need?) in Swahili.

This gap is not a bug. It is a feature of how LLMs are evaluated.

The standard suite of benchmarks — **MMLU** (massive multitask language understanding), **HellaSwag** (commonsense reasoning), **GSM8K** (grade-school math), and **HumanEval** (code generation) — are overwhelmingly English-centric. They measure what a model knows about Western exam questions, Western common sense, and English-language programming. They tell you almost nothing about how a model performs in Kikuyu, Hausa, or a Nairobi matatu conductor's code-switched Sheng.

## Why Standard Benchmarks Fail for African Contexts

**MMLU** covers 57 subjects from US history to college biology. Zero African languages. If a model has never seen Swahili or Yoruba in its training data, a 90% MMLU score is irrelevant — the model simply cannot process the input.

**HellaSwag** tests whether a model can complete a sentence about everyday scenarios. The "everyday" is American: microwaving popcorn, mowing lawns, filling a car at a gas station. A farmer in rural Uganda does not share those reference points. The model might fail a commonsense test that is actually a cultural knowledge test in disguise.

**GSM8K** asks math questions about pizzas, baseball cards, and movie tickets. The arithmetic works, but the scenarios are alien — and a model that only saw Western currency and measurement systems will struggle with "Unaweza kununua ndizi ngapi na shilingi 500?" (How many bananas can you buy with 500 shillings?).

**HumanEval** tests Python functions with docstrings in English. It captures nothing about instruction following in mixed-language prompts — the code-switching patterns that are natural in Lagos, Nairobi, or Johannesburg.

The fundamental issue: **benchmark scores do not transfer across linguistic and cultural boundaries**. A high score on a Western benchmark is necessary but not sufficient for African deployment.

## African-Focused Evaluation Efforts

The good news is that the African NLP community — led by groups like **Masakhane**, **Lacuna Fund**, and **AI4D Africa** — has been building evaluation resources that actually reflect African use cases:

### AfriQA
A question-answering dataset covering 10 African languages including Swahili, Yoruba, Hausa, and Amharic. Built by culturally adapting the SQuAD format — questions are relevant to African contexts, not just translated English queries. This tests whether a model can extract answers from passages in local languages.

### MasakhaNER
Named Entity Recognition annotations for 20+ African languages. If a model cannot identify "Nairobi" as a location or "Kenyatta" as a person in a Swahili sentence, it is not ready for any production NLP pipeline on the continent. MasakhaNER revealed that even top multilingual models degrade 15–30 points in F1 when moving from English to African languages.

### SERAKWAYE
A benchmark for African language understanding covering 42 languages across three tasks: sentiment analysis, topic classification, and NER. Designed to evaluate multilingual models systematically across a wide range of African languages — from widely spoken ones like Hausa and Swahili to lower-resource languages like Ewe and Dagbani.

### AfriMMLU
A recent effort extending the MMLU framework to African contexts. Instead of translated US exam questions, AfriMMLU sources questions from African educational curricula — West African Examination Council (WAEC) materials, Kenyan KCPE exams, and Nigerian WASSCE papers. This is what a culturally grounded benchmark looks like.

## What to Actually Evaluate

If you are deploying an LLM for African users, your evaluation rubric should include:

**Language understanding in local languages.** Can the model classify sentiment in Yoruba tweets? Extract named entities from an Amharic news article? Translate a Hausa WhatsApp message to English? Run evaluations with AfriQA, MasakhaNER, and SERAKWAYE.

**Cultural knowledge, not just translated benchmarks.** A translated MMLU question about "which US president signed the Civil Rights Act" tests translation ability, not cultural relevance. Use AfriMMLU or build your own evaluation set from local curricula and domain-specific materials.

**Instruction following in mixed-language prompts.** Code-switching is normal across Africa — "Nipe summary ya hii report" or "Translate this paragraph to Kiswahili." Models trained on monolingual English instructions often fail when the instruction and content switch languages mid-sentence.

**Safety and bias.** A model fine-tuned primarily on Western data may carry assumptions about family structures, gender roles, or religious norms that are inappropriate for African contexts. Evaluate for harmful stereotypes, cultural misrepresentations, and refusal patterns that change when the language switches.

## A Practical Evaluation Pipeline with lm-eval-harness

The [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) by EleutherAI is the standard framework for running LLM evaluations. Setting up an African-language evaluation pipeline is straightforward:

```bash
# Install the harness
pip install lm-eval

# Run a standard benchmark
lm_eval --model hf --model_args pretrained=mistralai/Mistral-7B-v0.3 \
  --tasks mmlu,hellaswag,gsm8k --num_fewshot 5 \
  --output_path results/standard/

# Run African-language benchmarks (after registering custom tasks)
lm_eval --model hf --model_args pretrained=mistralai/Mistral-7B-v0.3 \
  --tasks afriqa,masakhaner,serakwaye \
  --output_path results/african/
```

For African-language tasks, you may need to register custom task YAML files pointing to the relevant Hugging Face datasets. A minimal task definition for AfriQA might look like:

```yaml
# tasks/afriqa.yaml
task: afriqa
dataset_path: afriqa/afriqa
dataset_name: swahili
output_type: generate_until
training_split: train
validation_split: validation
doc_to_text: "Context: {{context}}\nQuestion: {{question}}\nAnswer:"
doc_to_target: " {{answers.text | join(', ')}}"
metric_list:
  - metric: exact_match
  - metric: f1
```

The key insight: **run both**. Compare the standard benchmark score against the African-language benchmark score. A wide gap tells you exactly how much the model relies on English-centric training — and where it will fail in production.

## Building on Earlier Work

This post is part of a practical series on building with LLMs in African contexts. We covered the fundamentals in [Low-Resource NLP](/posts/low-resource-nlp/) — transfer learning and data augmentation strategies — and took a deep dive into [Swahili NLP](/posts/swahili-nlp/) with datasets and model-building techniques. The next step is evaluation: measuring whether your efforts actually work.

Without African-language benchmarks in your evaluation loop, you are flying blind — no matter how high your MMLU score looks.

---

*Have you run into evaluation gaps deploying LLMs for African languages? What benchmarks do you wish existed? Share your experience in the comments below.*
