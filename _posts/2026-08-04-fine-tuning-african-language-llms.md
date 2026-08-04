---
title: "Fine-Tuning LLMs on African Language Datasets"
date: 2026-08-04 00:00:00 +0300
categories: [AI Engineering, LLM]
tags: [fine-tuning, african-languages, nlp, swahili, low-resource, transformers]
image:
  path: /assets/img/cover-series-practical-playbook.webp
  alt: Fine-tuning pipeline diagram — base LLM adapted via LoRA for African language tasks
---

## Why Fine-Tune?

A foundation model like Llama 3 70B or Mistral generates fluent English, French, and Arabic. Ask it to write in Swahili or translate a sentence to Yoruba, and quality drops off a cliff.

The reason is straightforward: **training data distribution**. The largest models are trained on corpora where English comprises 50-80% of tokens. African languages — even widely spoken ones like Swahili (200M speakers) — account for fractions of a percent. The model simply hasn't seen enough examples to learn the grammar, idioms, and vocabulary of these languages.

Fine-tuning changes that. With as little as 10,000-50,000 high-quality examples in a target language, you can dramatically improve a model's performance on African language tasks — translation, classification, summarisation, and generation. And thanks to Parameter-Efficient Fine-Tuning (PEFT), this is achievable on hardware you probably already have.

This builds on our earlier deep dives into [Swahili NLP](/posts/swahili-nlp/) and [Low-Resource NLP Techniques](/posts/low-resource-nlp/). Fine-tuning is the next logical step: once you know the landscape and the transfer strategies, you need a practical pipeline to adapt models to your specific language.

## Available Datasets

The African NLP community — particularly the **Masakhane** project — has done the hard work of curating and annotating datasets. Here are the key resources for fine-tuning:

- **Masakhane datasets**: A growing collection of annotated datasets for translation, NER, sentiment, and news classification across 50+ African languages. The [Masakhane GitHub](https://github.com/masakhane) hosts the primary repositories.

- **MasakhaNER** (2021, updated 2024): Named Entity Recognition annotations for 20+ African languages including Yoruba, Hausa, Igbo, Swahili, Amharic, and Wolof. Each language has ~10K-15K annotated sentences.

- **AfriQA**: A question-answering dataset covering 10 African languages, built by translating and culturally adapting the SQuAD format. Useful for instruction-tuning LLMs to answer questions in African languages.

- **MasakhaNEWS**: News topic classification (5 categories) for 15 African languages, with ~10K articles per language.

- **SERAKWAYE**: A benchmark for African language understanding covering 42 languages across 3 tasks (sentiment, topic classification, NER). Designed specifically to evaluate multilingual models on African languages.

- **JW300**: A parallel corpus of ~3.5M sentence pairs between English and 300+ languages, including extensive African language coverage. Ideal for translation fine-tuning.

For Swahili specifically, the datasets covered in our [Swahili NLP post](/posts/swahili-nlp/) (CC-100, OSIAN, GlobalVoices, NLLB) provide additional pre-training material.

## Three Key Techniques

### 1. LoRA and QLoRA (Parameter-Efficient Fine-Tuning)

LoRA (Low-Rank Adaptation) freezes the base model weights and injects trainable low-rank matrices into each transformer layer. Instead of updating 70B parameters, you train a few million. This means:

- **Memory efficiency**: A 7B model fine-tunes on 16GB GPU memory with LoRA
- **Speed**: Training is 2-4x faster since gradients only flow through LoRA weights
- **Portability**: The LoRA adapter is a ~25MB file — share it independently of the base model

QLoRA goes further by quantizing the base model to 4-bit (NF4 format) before applying LoRA. This enables fine-tuning a 70B parameter model on a single **RTX 4090 (24GB)** — something that would otherwise require 8× A100s.

### 2. Continued Pre-Training with Tokenizer Extension

Base model tokenizers rarely include African language vocabulary efficiently. Our Swahili example from the earlier post showed one word (`hawakuwasomea`) fragmenting into 7 subword tokens. This wastes context window and increases compute.

The solution is **tokenizer extension**:
1. Train a new BPE tokenizer on a large African language corpus (e.g., OSIAN Swahili)
2. Merge the new tokens into the base model's tokenizer
3. Run continued pre-training (masked language modeling or causal LM) on the extended vocabulary

This adds only the tokens that the original tokenizer handled poorly — typically a few thousand new tokens.

### 3. Instruction Tuning on Translated or Adapted Data

For task-specific fine-tuning (translation, QA, summarisation), convert your datasets into instruction format:

```
### Instruction:
Translate the following sentence from English to Swahili.

### Input:
The agricultural sector employs 60% of the workforce in East Africa.

### Response:
Sekta ya kilimo inaajiri asilimia 60 ya wafanyakazi katika Afrika Mashariki.
```

This format works with chat-tuned models (Llama, Mistral, Qwen) and produces dramatically better results than standard fine-tuning.

## Practical Pipeline

Here's an end-to-end pipeline using HuggingFace TRL (Transformer Reinforcement Learning) for QLoRA fine-tuning of a translation model:

```python
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ===== 1. Dataset Preparation =====
# Load JW300 Swahili-English parallel data
dataset = load_dataset("masakhane/jw300", "sw-en", split="train")

def format_instruction(example):
    return {
        "text": f"### Instruction:\nTranslate English to Swahili.\n\n### Input:\n{example['en']}\n\n### Response:\n{example['sw']}"
    }

dataset = dataset.map(format_instruction)

# ===== 2. Tokenizer Extension (if needed) =====
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# Check tokenizer efficiency on Swahili text
test_word = "hawakuwasomea"
tokens = tokenizer.tokenize(test_word)
print(f"Base tokenizer: {len(tokens)} tokens for '{test_word}'")
# If token count is high, extend the tokenizer here
# new_tokens = train_new_bpe(swahili_corpus)
# tokenizer.add_tokens(new_tokens)

tokenizer.chat_template = None  # Use default

# ===== 3. QLoRA Configuration =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=16,           # rank of the LoRA matrices
    lora_alpha=32,  # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ===== 4. Training =====
trainer = SFTTrainer(
    model="meta-llama/Llama-3.2-3B",
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./swahili-llm-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
    ),
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,
)

trainer.train()

# ===== 5. Merge and Export =====
model = trainer.model.merge_and_unload()
model.save_pretrained("./swahili-llm-merged")
tokenizer.save_pretrained("./swahili-llm-merged")
```

## Hardware Reality Check

The pipeline above uses **Llama 3.2 3B** as an example — it fine-tunes in about 2-3 hours on a single RTX 4090 (24GB) or RTX 3090.

For larger models, here's the QLoRA footprint:

- **3B**: 8-10 GB — RTX 3060 / any 8GB+ — ~2 hours (3 epochs, 50K examples)
- **7-8B**: 14-16 GB — RTX 3090/4090 — ~4 hours
- **13B**: 18-20 GB — RTX 4090 — ~6-7 hours
- **70B**: 22-24 GB — RTX 4090 (just barely) — ~20-24 hours

This is the core message of our **constrained-environment ML** theme: you don't need a cluster. A single consumer GPU, careful use of 4-bit quantization, and LoRA adapters let you produce production-quality African language models on hardware that fits under a desk.

The real constraint isn't compute — it's **data**. Investing in building high-quality African language datasets (via Masakhane, local universities, or community annotation drives) delivers far more impact than buying more GPUs.

## Testing Your Fine-Tuned Model

After merging, evaluate on the [SERAKWAYE benchmark](https://github.com/masakhane/serakwaye) or run a quick qualitative test:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="./swahili-llm-merged")

prompt = "### Instruction:\nTranslate to Swahili.\n\n### Input:\nI would like to buy maize and beans.\n\n### Response:\n"
result = pipe(prompt, max_new_tokens=50)[0]["generated_text"]
print(result)
# Expected: "Ningependa kununua mahindi na maharagwe."
```

Compare the output against the base model without fine-tuning. The difference is dramatic — and it's achievable with a weekend's work and a single GPU.

## Summary

Fine-tuning LLMs on African languages is one of the highest-leverage activities in NLP today. The community datasets exist, the techniques (LoRA/QLoRA, tokenizer extension, instruction tuning) are mature, and the hardware requirements are modest. The gap between base model performance and what's possible after fine-tuning is enormous — largely because the base models were never trained on your target language in the first place.

If you've been reading through this series — from [Swahili NLP](/posts/swahili-nlp/) through [Low-Resource NLP](/posts/low-resource-nlp/) to this post — you now have the full toolkit: understanding the linguistic challenges, transfer learning strategies, and a practical fine-tuning pipeline. The next step is picking a language and running the experiment.

*Next in the series: Evaluating African Language LLMs — benchmarks, human evaluation protocols, and building trust in model outputs for production deployment.*
