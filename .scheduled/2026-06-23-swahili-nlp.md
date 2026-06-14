---
title: "Swahili NLP: Building Language Models for African Languages"
date: 2026-06-23 00:00:00 +0300
categories: [Machine Learning, AI in Africa]
tags: [africa, swahili-nlp, natural-language-processing, machine-translation, nlp]
image:
  path: /assets/img/cover-swahili-nlp.webp
  alt: Swahili text in Arabic and Latin script with NLP pipeline diagram
---

## Why Swahili Matters for NLP

Swahili (Kiswahili) is the most widely spoken African language in the world, with over 200 million speakers across East and Central Africa. It's the national or official language of Tanzania, Kenya, Uganda, the Democratic Republic of Congo, Rwanda, Burundi, and the African Union itself. Unlike many African languages, Swahili has a long written tradition — it was first written in Arabic script (Ajami) in the 10th century, and it's now written in Latin script with a standardized orthography.

This makes Swahili a unique test case for African NLP: it has *more* resources than virtually any other indigenous African language, but still far fewer than any European language of comparable speaker count. If we can't build good NLP systems for Swahili, we have no hope for the other 2,000+ African languages.

## The Linguistic Challenges

### Agglutinative Morphology

Swahili is a **Bantu agglutinative language**, meaning it builds complex words by chaining together prefixes, infixes, and suffixes onto a verb root. Consider the verb *kusoma* (to read):

| Swahili | Gloss |
|---------|-------|
| ninasoma | I read (present) |
| nilisoma | I read (past) |
| nitasoma | I will read |
| ninakusoma | I read you |
| hawakuwasomea | They did not read for them |

The last example, *hawakuwasomea*, is a single word in Swahili containing the subject prefix (*ha-* negative), tense marker (*-ku-* past), object prefix (*-wa-* them), verb root (*-som-* read), and applicative suffix (*-ea* for/at). In English, this entire paradigm would require multiple sentences.

**Why this matters for tokenization:** Standard subword tokenizers like BPE (Byte-Pair Encoding) used in BERT and GPT models were designed for English. They tend to fragment Swahili words into many small tokens, inflating the effective sequence length. A sentence of 20 Swahili words might contain 60+ subword tokens — eating into the 512-token limit of many transformer models.

```python
# BPE tokenization example (conceptual)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
swahili_text = "hawakuwasomea"
tokens = tokenizer.tokenize(swahili_text)
# mBERT splits this into: ['ha', '##wa', '##ku', '##wa', '##so', '##me', '##a']
# That's 7 tokens for a single word!
```

### Code-Switching

Swahili speakers in East Africa rarely speak "pure" Swahili. Code-switching between Swahili and English — known as **Sheng** in Kenya — is the norm, especially in urban areas:

{% raw %}
```
"Sasa, the meeting imeanza already, so usichelewe next time."
```
{% endraw %}

("Now, the meeting has already started, so don't be late next time.")

This mixed-language pattern breaks the assumptions of monolingual tokenizers and language models. Models trained on "clean" Swahili text fail on real-world social media content. A 2024 study by Masakhane researchers found that sentiment analysis accuracy on Sheng text dropped 23% compared to standard Swahili.

### Dialectal Variation

Swahili has multiple dialects:
- **Kiunguja** (Zanzibar): The standard dialect
- **Kimvita** (Mombasa): Significant lexical and phonetic differences
- **Kiamu** (Lamu): Archaic features preserved
- **Congo Swahili**: Influenced by French and Lingala
- **Sheng** (Urban Kenyan slang): Highly fluid and rapidly evolving

Building a single model that handles all these varieties is a major challenge.

## Available Resources

Despite the challenges, Swahili is relatively well-resourced compared to other African languages:

### Text Datasets

| Dataset | Size | Source | License |
|---------|------|--------|---------|
| Swahili CC-100 | 13M sentences | Common Crawl | Public |
| OSIAN Swahili | 1.7M sentences | News, Wikipedia, religious texts | CC-BY |
| Masakhane News Swahili | 25K sentences | East African newspapers | CC-BY-SA |
| MaCoCu Swahili | 5.2M sentences | Web crawl | CC-BY |
| Swahili NER Corpus | 15K annotations | News text | Research |

### Parallel Corpora (for Translation)

| Dataset | Size | Target Languages |
|---------|------|------------------|
| JW300 Sw-En | 3.5M sentence pairs | English |
| GlobalVoices Sw-En | 450K sentence pairs | English |
| Tanzil Sw-En | 60K sentence pairs | English |
| NLLB Seed Swahili | 18K sentence pairs | 40+ languages |

### Pretrained Models

Several multilingual models include Swahili in their training data:
- **mBERT** (110 languages): Includes Swahili
- **XLM-R** (100 languages): 50GB of filtered CommonCrawl per language
- **NLLB-200** (200 languages): Swahili-to-X translation model
- **AfriBERTa** (11 African languages): Trained specifically on African languages
- **BERT-Swahili** (Kinyonga lab, 2023): RoBERTa-base trained on 10GB of Swahili text

### Limitations of Existing Datasets

Most "Swahili" datasets are:
1. **Heavily English-influenced**: Much of the CC-100 Swahili is actually translated English content, not naturally produced Swahili
2. **Formal register only**: News and religious texts dominate; there's almost no conversational Swahili
3. **Dated**: The Masakhane news corpus was collected in 2020-2021
4. **Missing code-switching**: Almost no datasets include Swahili-English mixing

## Practical Pipeline: Building a Swahili Text Classifier

Let's walk through a real pipeline for fine-tuning a Swahili sentiment classifier:

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import datasets
import numpy as np

# Load Swahili sentiment dataset
dataset = datasets.load_dataset("masakhane/swahili-sentiment", split="train")

# Use a multilingual model with Swahili support
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Swahili sentences can be long due to agglutination
    # We increase max_length and use truncation
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Evaluate with F1 score (class imbalance is common)
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"f1": f1_score(labels, predictions, average="weighted")}

training_args = TrainingArguments(
    output_dir="./swahili-sentiment-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    ),
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # In practice, use a separate eval split
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Evaluation Considerations

Standard NLP evaluation metrics don't tell the whole story for Swahili:

**BLEU for translation**: BLEU correlates poorly with human judgment for Swahili because it rewards token-level n-gram overlap, but Swahili's agglutinative morphology means multiple valid translations of a single word. A Swahili-English translator might render *nisingekuwa* as either "if I were not" or "if I wasn't" — identical meaning, different BLEU scores.

**Accuracy metrics**: For classification tasks, test sets inevitably contain more formal Swahili than real-world usage. A model scoring 0.95 on news sentiment may drop to 0.65 on Twitter Sheng.

**Human evaluation protocols**: Masakhane publishes guidelines for human evaluation of Swahili NLP that account for:
- Dialect tolerance (accept multiple correct translations)
- Code-switching acceptance
- Cultural context awareness

## The Road Ahead for Swahili NLP

The most promising directions:

1. **Better tokenization**: Byte-level and character-level models (CANINE, ByT5) avoid subword fragmentation entirely and work well for agglutinative languages
2. **Sheng and code-switching datasets**: Masakhane's current project is building a 100K-sentence Swahili-English code-switching corpus
3. **Speech NLP**: Swahili is a spoken-first language for most of its speakers; ASR and text-to-speech are underserved
4. **Cross-lingual transfer**: Using English or French models to bootstrap Swahili systems via parallel data

Swahili NLP is the canary in the coal mine for African language technology. If we get this right — with rigorous evaluation, culturally aware dataset creation, and linguistically informed model architecture choices — the methods will generalize to hundreds of other African languages. If we get it wrong, we risk building technology that doesn't serve the 200 million people who speak Swahili every day.

*Next in the series: Low-resource NLP techniques — transfer learning and data augmentation for languages with almost no training data.*
