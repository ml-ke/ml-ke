---
title: "Contamination Checks Before You Trust a Fine-Tune Score"
date: 2026-10-26 00:00:00 +0300
categories: [Machine Learning, AI Engineering]
tags: [evaluation, data contamination, decontamination, fine-tuning, data quality, benchmarks]
image:
  path: /assets/img/cover-contamination-checks-before-you-trust-a-score.webp
  alt: An eval item passing through a 13-gram gate that catches its verbatim copy while a paraphrased twin slips through
---

> **The short version**
> A fine-tune that scores 89 on a benchmark may be scoring 62 on text it has never seen. This post builds the check that tells the difference: a 13-gram overlap gate, measured false-alarm and recall behaviour as the n-gram size moves, the paraphrase and translation cases the gate cannot see, and the subset-scoring arithmetic that turns "contaminated" into an actual number with a confidence interval. Every figure below comes from the code in this post, run on this laptop, and each block re-runs byte-identical.
{: .prompt-info }

Two things go wrong when you evaluate a fine-tune, and they pull in opposite directions.

The first is that your training data contains the test set, or a translation of it, or a paraphrase of it. The score then measures memorisation and the number you report is an upper bound you did not label as one. The second is that your decontamination gate is too blunt, deletes legitimate items, and leaves you evaluating on a smaller set that is now easier or weirder than the original, with a confidence interval nobody widened to match.

Both failures are measurable, and both are cheap to measure at the scale most teams actually operate: thousands of training documents, a few hundred to a few thousand eval items. This post builds the measurement rather than arguing about it. If you want the version with the adversarial framing, the [benchmark poisoning post](/posts/eval-benchmark-poisoning/) covers contamination as an attack. This one is the build-time hygiene version: what to run, what the numbers mean, and what to report.

One clarification up front, because the words get used for different jobs. Corpus deduplication removes near-duplicate documents from inside your training set; the [MinHash post](/posts/minhash-lsh-corpus-dedup/) covers that. Decontamination compares your training set against an eval set you do not want to memorize, and the unit of interest is the eval item's span of tokens, not the document. Different comparison, different threshold, different failure mode.

## Two checks, two jobs

There are two families of check and they are not substitutes.

| Check | Needs | Cost | What it catches | What it misses |
|---|---|---|---|---|
| n-gram overlap | training text and eval text | linear in both, one hash set | verbatim copies, copies with a few edits | paraphrase, translation, reformatting |
| Semantic match | embeddings or an LLM judge | one pass plus a threshold to defend | paraphrases and translations | precision, without a second pass to confirm |

The arithmetic matters here. Exact n-gram matching is cheap enough to run on every ingest, which is why it ends up in almost every data pipeline. The semantic check is expensive enough that it belongs only on the output of the cheap one, as a second opinion on a much smaller set. That is the shape of the pipeline: gate broadly and cheaply, then review narrowly and expensively.

## The n-gram gate, and where the threshold sits

The rule most pipelines implement is this: tokenise the eval item, enumerate its n-grams, and ask what fraction of them also appear in a training document. Above some share, flag the document. The published thresholds are 13-grams in the GPT-3 contamination analysis and a 50-character overlap in the GPT-4 technical report, both summarised in [Yang et al.](https://arxiv.org/abs/2311.04850). Marin's open data pipeline uses the same 13-gram unit with an explicit rule: for each paragraph, if at least half of its eligible 13-gram occurrences appear in the evaluation material, flag the document, because a lighter rule "catches obvious copies, but misses paraphrases and some embedded examples", [their words](https://openathena.ai/blog/marin-data-pipeline-overview/).

Here is that gate, plus the character rule, run against five training documents that all relate to one eval item in different ways.

{% raw %}
```python
import difflib
import re

EVAL = """Question: A clinic in Kisumu records 240 patient visits. 15 percent are referrals. How many referrals were recorded?
A. 24
B. 36
C. 48
D. 60
Answer: B"""

CORPUS = {
    "verbatim copy": "notes: " + EVAL,
    "one word changed": EVAL.replace("Kisumu", "Nakuru"),
    "paraphrased": """Question: At a health centre in Kisumu, 240 visits were logged last month and 15 percent of them were referrals. What number of referrals does that give?
A. 24
B. 36
C. 48
D. 60
Answer: B""",
    "translated to Swahili": """Swali: Kliniki moja Kisumu huandika ziara 240 za wagonjwa. Asilimia 15 ni rufaa. Ni rufaa ngapi ziliandikwa?
A. 24
B. 36
C. 48
D. 60
Jibu: B""",
    "same template, new numbers": """Question: A school in Kisumu buys 240 exercise books. It keeps 15 percent as spares. How many books are kept as spares?
A. 24
B. 36
C. 48
D. 60
Answer: B""",
}


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def grams(t, n):
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


def share(eval_item, train_doc, n):
    ig = grams(toks(eval_item), n)
    if not ig:
        return 0.0
    tg = set(grams(toks(train_doc), n))
    return sum(1 for g in ig if g in tg) / len(ig)


def longest_span(a, b):
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


NS = (3, 5, 8, 10, 13, 20)
print("Token rule: share of the eval item's n-grams also found in one training doc")
print((f"{'training doc holds':26s} " + " ".join(f"n={n:<3d}" for n in NS)).rstrip())
for name, doc in CORPUS.items():
    print((f"{name:26s} " + " ".join(f"{share(EVAL, doc, n):.2f}" for n in NS)).rstrip())

print()
print("Character rule (GPT-4 report's 50-character overlap test)")
for name, doc in CORPUS.items():
    span = longest_span(EVAL, doc)
    print(f"  {name:26s} longest shared span {span:4d} chars -> flagged at 50 chars: {span >= 50}")

print()
for n in (3, 13):
    flagged = [k for k, d in CORPUS.items() if share(EVAL, d, n) >= 0.5]
    print(f"n={n:<3d} with the 50% share rule flags: {flagged}")
```
{% endraw %}

```text
Token rule: share of the eval item's n-grams also found in one training doc
training doc holds         n=3   n=5   n=8   n=10  n=13  n=20
verbatim copy              1.00 1.00 1.00 1.00 1.00 1.00
one word changed           0.88 0.79 0.76 0.74 0.69 0.44
paraphrased                0.31 0.25 0.14 0.05 0.00 0.00
translated to Swahili      0.23 0.17 0.05 0.00 0.00 0.00
same template, new numbers 0.31 0.25 0.14 0.05 0.00 0.00

Character rule (GPT-4 report's 50-character overlap test)
  verbatim copy              longest shared span  150 chars -> flagged at 50 chars: True
  one word changed           longest shared span  123 chars -> flagged at 50 chars: True
  paraphrased                longest shared span   35 chars -> flagged at 50 chars: False
  translated to Swahili      longest shared span   26 chars -> flagged at 50 chars: False
  same template, new numbers longest shared span   35 chars -> flagged at 50 chars: False

n=3   with the 50% share rule flags: ['verbatim copy', 'one word changed']
n=13  with the 50% share rule flags: ['verbatim copy', 'one word changed']
```

Four things in that table are worth keeping.

A verbatim copy scores 1.00 at every n-gram size, so for the case you care most about, the choice of n barely matters. A single substituted word is still caught at n=13, at 0.69, because one edit only destroys the 13-grams that contain that token; push n to 20 and the share falls to 0.44 and the item drops below the threshold. So the n-gram size is a tolerance setting more than a purity one: larger n narrows the window of edits that still register.

The paraphrase scores 0.00 from n=13 up, and the char rule sees only a 35-character shared span, under the 50-character bar. The residual 0.31 at n=3 is not wording: in this corpus the matches at low n are entirely the `A. 24 / B. 36 / C. 48 / D. 60 / Answer: B` block, which is shared by every item in the file. That is the mechanism behind the false positives Yang et al. describe on multiple-choice benchmarks, where shared option text and repeated numbers trigger overlap with no contamination present.

And the translated item scores 0.00 at n=10 while being the same question. Translation is the cheapest possible evasion, which matters if your corpus includes machine-translated material, a common pattern in African-language datasets where English benchmarks get translated to build training pairs.

## At scale, the threshold decides your false-alarm rate

One item proves the gate has a blind spot. It does not tell you what the gate does to four hundred items. This second block builds a 400-item eval set in one template, leaks 20 of those items verbatim into a 320-document training corpus, and runs the same 50%-share rule at six values of n.

{% raw %}
```python
import difflib
import random
import re

random.seed(20261026)

TOPICS = ["maize yields", "clinic staffing", "school fees", "fuel prices", "water points",
          "boda licences", "market stalls", "tea deliveries", "fishery permits", "solar kits"]
NUMS = list(range(11, 99))


def item(i):
    t = TOPICS[i % len(TOPICS)]
    a, b = random.sample(NUMS, 2)
    p = random.choice([5, 10, 15, 20, 25, 30])
    return (f"Question: A report on {t} in county {i} records {a * 10} cases. "
            f"{p} percent are flagged. How many cases are flagged?\n"
            f"A. {a * p // 10}\nB. {b}\nC. {a}\nD. {b * 2}\nAnswer: A")


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def grams(t, n):
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


N_EVAL, N_COPIED = 400, 20
eval_items = [item(i) for i in range(N_EVAL)]
copied = random.sample(range(N_EVAL), N_COPIED)
train_docs = [item(i + 5000) for i in range(300)] + [eval_items[i] for i in copied]

print(f"eval items {N_EVAL}, leaked into training {N_COPIED}, training docs {len(train_docs)}")
print()
print(f"{'n':>3s} {'flagged clean':>14s} {'caught leaked':>14s}")
for n in (3, 5, 8, 10, 13, 20):
    tg = set()
    for d in train_docs:
        tg.update(grams(toks(d), n))

    def flagged(doc):
        g = grams(toks(doc), n)
        return bool(g) and sum(1 for x in g if x in tg) / len(g) >= 0.5

    hits = [i for i, d in enumerate(eval_items) if i not in copied and flagged(d)]
    caught = sum(1 for i in copied if flagged(eval_items[i]))
    print((f"{n:3d} {len(hits):>8d}/{N_EVAL - N_COPIED:<5d} {caught:>8d}/{N_COPIED:<5d}").rstrip())

n = 13
tg = set()
for d in train_docs:
    tg.update(grams(toks(d), n))
print()
print("the clean items still flagged at n=13, with their longest shared character span:")
for i, d in enumerate(eval_items):
    if i in copied:
        continue
    g = grams(toks(d), n)
    if g and sum(1 for x in g if x in tg) / len(g) >= 0.5:
        span = max(difflib.SequenceMatcher(None, d, t, autojunk=False)
                   .find_longest_match(0, len(d), 0, len(t)).size for t in train_docs)
        print(f"  eval item {i}: {span} of {len(d)} characters shared with a training doc")
```
{% endraw %}

```text
eval items 400, leaked into training 20, training docs 320

  n  flagged clean  caught leaked
  3      380/380         20/20
  5      175/380         20/20
  8        2/380         20/20
 10        2/380         20/20
 13        2/380         20/20
 20        0/380         20/20

the clean items still flagged at n=13, with their longest shared character span:
  eval item 89: 106 of 151 characters shared with a training doc
  eval item 203: 108 of 154 characters shared with a training doc
```

At n=3 every single clean item is flagged, 380 of 380, so the gate is useless: it flags the template, not the leak. At n=5 that drops to 175 of 380, still a coin flip. From n=8 the false-alarm count collapses to 2 of 380, and recall on the leaked items stays at 20 of 20 all the way up to n=20, because verbatim copies are long enough that widening the window never costs them entirely.

The two survivors at n=13 deserve attention rather than a shrug. They are not copies. They share 106 and 108 characters with a training document out of 151 and 154, which means same stem, different county number: near-duplicates generated by the template itself. A gate that flags them is doing something useful, since a benchmark full of near-identical stems wastes eval capacity and inflates apparent consistency, but calling them contamination would be wrong. The output of this gate is a review queue, not a verdict, and the queue is short enough to read.

Two honest caveats about this measurement. The corpus is synthetic and templated, which exaggerates the low-n false-alarm rate compared with natural prose; what transfers is the shape of the curve, not the exact counts. And recall of 20 of 20 is a property of verbatim leakage. Change one word in the copies and recall holds at n=13 but breaks at n=20, as the first block showed. If your leak is a reworded copy, this gate is not the check you need.

## The blind spot the gate cannot fix

Yang et al. trained Llama-2 on rephrased test sets, using paraphrases and translations, and pushed the rephrased data through the same n-gram detection that would normally catch a copy. On MMLU, Llama-2-7B goes from 45.3 to 88.5 when trained on question-only rephrased samples. Llama-2-13B goes from 54.8 to 85.9 on the full-prompt variant and 89.9 on question-only, while the n-gram check sees nothing. On GSM-8K the same recipe takes 13B from 28.7 to 95.3. On HumanEval, CodeLlama-7B climbs from 32.9 to 67.7 with rephrased Python, and reaches 100 when the test set itself is used as fine-tuning data.

The translation condition is the one to internalise, because it is the cheapest to produce. With the MMLU test set translated into Chinese and used as fine-tuning data, Llama-2-7B answers the original test set at 100, and 91.1 and 74.3 on the question-only and full-prompt forms of the translated set, while no 13-gram match fires. Translation moves the text out of the matcher's reach and leaves the meaning where the model can still memorise it.

Their survey of the same problem in public data found 8 to 18 percent of HumanEval overlapping in the pretraining sets they examined, with the per-dataset figures reaching 18.9 percent for The Stack, 15.9 percent for StarCoder-Data, 12.8 percent for CodeAlpaca and 8.53 percent for RedPajama-Data. Those are not adversarial inserts. They are artefacts of scraping and of synthetic data generated by models that had already seen the benchmark.

The mitigation they propose is the tiered shape from earlier: use embedding similarity to generate a broad candidate set, then refine it with an LLM judge, because similarity search alone over-flags and n-gram alone under-flags. Their cost table is blunt about the trade: n-gram overlap is linear in training size times eval size and needs only the text; embedding search adds an index but still needs the text; decoding-based and influence-function methods need model access, which third parties rarely have.

Anyone fine-tuning on a budget is usually the third party in exactly that position: no pretraining data, no labelled clean slice of their own. The move available is to document what you checked and report the number you can defend.

## Score the subsets, then report the gap

Llama 2's own contamination appendix is the most useful template I have found, because it turns the flag into a score rather than a delete. Their definition works bottom up: a token is contaminated if it appears in any token n-gram longer than 10 tokens in both the eval sample and the training set, and a sample's contamination percentage is the share of its tokens that are contaminated. They allow a skipgram budget of four tokens across, no mismatches in the first ten tokens, no trailing mismatches, because evaluation prompts get reformatted between runs.

Then the two-condition test, which is the part most teams skip. If contamination is driving the score, the cleanest samples should score below the overall average and the dirtiest should score above it, and if only one of those holds, the evidence is insufficient. They report, for MMLU overall at 70B, a clean subset of 11,862 items at 68.0 and a dirty subset of 1,536 items at 78.2; on HellaSwag 70B the same comparison runs 80.0 against 92.2; on MMLU-Humanities at 7B it is 40.8 against 56.9.

You can compute the same gap for your own eval with three lines of arithmetic. The expected inflation from a leak of share q, when clean items are answered at p-clean and leaked items at p-dirty, is q times the gap between them.

{% raw %}
```python
import math
import random

random.seed(20261026)
P_CLEAN, P_DIRTY, N = 0.62, 0.94, 1200


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


print(f"eval set of {N} items; clean items answered at {P_CLEAN:.2f}, leaked items at {P_DIRTY:.2f}")
print(f"closed form: reported = (1-q)*{P_CLEAN:.2f} + q*{P_DIRTY:.2f}")
print()
print(f"{'leaked share q':>14s} {'reported':>9s} {'clean-only':>11s} {'inflation':>10s} {'clean n':>8s}")
for q in (0.02, 0.05, 0.15, 0.30):
    reported = (1 - q) * P_CLEAN + q * P_DIRTY
    n_clean = round(N * (1 - q))
    print(f"{q:13.0%} {reported:9.3f} {P_CLEAN:11.3f} {q * (P_DIRTY - P_CLEAN):+10.3f} {n_clean:8d}")

print()
print("sampled run at q=15%, seed 20261026:")
q, n_dirty = 0.15, round(N * 0.15)
n_clean = N - n_dirty
k_clean = sum(random.random() < P_CLEAN for _ in range(n_clean))
k_dirty = sum(random.random() < P_DIRTY for _ in range(n_dirty))
overall = (k_clean + k_dirty) / N
lo, hi = wilson(k_clean, n_clean)
print(f"  reported score over all {N} items: {overall:.3f}")
print(f"  clean subset: {k_clean}/{n_clean} = {k_clean / n_clean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print(f"  leaked subset: {k_dirty}/{n_dirty} = {k_dirty / n_dirty:.3f}")
print(f"  both conditions hold (clean below overall AND leaked above overall): "
      f"{k_clean / n_clean < overall < k_dirty / n_dirty}")
```
{% endraw %}

```text
eval set of 1200 items; clean items answered at 0.62, leaked items at 0.94
closed form: reported = (1-q)*0.62 + q*0.94

leaked share q  reported  clean-only  inflation  clean n
           2%     0.626       0.620     +0.006     1176
           5%     0.636       0.620     +0.016     1140
          15%     0.668       0.620     +0.048     1020
          30%     0.716       0.620     +0.096      840

sampled run at q=15%, seed 20261026:
  reported score over all 1200 items: 0.670
  clean subset: 632/1020 = 0.620  95% CI [0.589, 0.649]
  leaked subset: 172/180 = 0.956
  both conditions hold (clean below overall AND leaked above overall): True
```

The table explains why contamination goes unnoticed for so long. At a 5 percent leak the inflation is 0.016, which is smaller than the noise on a few hundred items and will be attributed to a seed. At 15 percent it is 0.048. At 30 percent it is 0.096, and a 9.6-point lift is larger than the difference between most pairs of models anyone is choosing between. The leak does not have to be large to change a decision, it only has to be larger than the decision's margin.

The adjustment is not free either, and the sampled run shows the price: dropping the leaked items leaves 1,020 clean items, the clean score is 0.620 with a 95 percent interval from 0.589 to 0.649, and that interval is the honest number to ship. If your eval set is 200 items and you find a 15 percent leak, the clean subset is 170 items and the interval on it is wide enough that a two-point model difference disappears into it. Contamination checks and eval-set size are the same problem, and a small eval set is contaminated in a second sense: it cannot resolve what you are asking it.

## A protocol you can put in the pipeline

| Step | What to run | Why it earns its place |
|---|---|---|
| Pin the eval set | hash every item, keep the set out of every training crawler path | a gate cannot compare against a file that changed last week |
| Gate at ingest | 13-gram share per document, flag at 50 percent | cheap enough to run on every append, catches copies |
| Review the queue | read the flagged documents, not just the count | the small residue is near-duplicates and template collisions, and reading them is minutes |
| Add a semantic pass | embed flagged documents, confirm with a judge or a human | catches paraphrase and translation, which n-gram cannot |
| Report two numbers | leaked share q, plus clean-subset score with its interval | turns contamination into a caveat a reader can price |
| Re-run on every append | the same gate, on the new shard only | contamination arrives with data, not with the first build |

The step that gets skipped is the last one, and it is the one that keeps the first five honest. Data arrives after the pipeline is built, usually from a scrape that nobody annotated, and a decontamination run that happened once at project start says nothing about the shard added last week.

There is a second habit worth copying from Llama 2's appendix: they decontaminated their internal development sets too, not only the benchmarks they published against. A dev set you have leaked into is worse than a benchmark you have leaked into, because you tune against it repeatedly and the leak compounds.

## Key takeaways

| Finding | Measured here | What to do |
|---|---|---|
| Low n-gram sizes flag everything | n=3 flagged 380 of 380 clean items, n=5 flagged 175 | Use 13-grams or longer, never 3 to 5 |
| A 13-gram gate catches copies and near-copies | verbatim 1.00 at every n, one-word edit 0.69 at n=13 | Keep it as the first gate, at 50 percent share |
| Paraphrase and translation defeat both text rules | 0.00 n-gram share at n=13, 35 and 26 character spans against a 50-character rule | Add a semantic pass on the flagged set |
| The residue needs reading, not deleting | 2 of 380 flags at n=13 were same-stem near-duplicates | Treat output as a review queue |
| Leak size decides whether it matters | +0.016 at 5 percent, +0.096 at 30 percent | Report q next to the score |
| The fix costs precision | clean n fell from 1,176 to 840, interval 0.589 to 0.649 | Widen the interval, or grow the eval set |

A score is a claim about the data the model has not seen. Contamination checks are how you find out whether that sentence is true, and the whole apparatus needed for a small fine-tune is a 13-gram set, a review of the short queue it produces, and one extra number in the results table. The alternative, which is what most released numbers do today, is to report a score with no statement about overlap at all: of 30 model developers documented by [Zhang et al.](https://arxiv.org/abs/2410.08385), just 9 report train-test overlap in any form, four by releasing training data and five by publishing a methodology.

## References

- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", Appendix A.6 dataset contamination (token-level rule, skipgram budget, two-condition test, Table 51) — https://arxiv.org/abs/2307.09288
- Yang, Chiang, Zheng, Sze and Gonzalez, "Rethinking Benchmark and Contamination for Language Models with Rephrased Samples", arXiv:2311.04850 — https://arxiv.org/abs/2311.04850
- Zhang, Klyman, Mai, Levine, Zhang, Bommasani and Liang, "Language model developers should report train-test overlap", arXiv:2410.08385 — https://arxiv.org/abs/2410.08385
- Brown et al., "Language Models are Few-Shot Learners" (13-gram contamination criterion) — https://arxiv.org/abs/2005.14165
- OpenAI, "GPT-4 Technical Report" (50-character overlap rule) — https://arxiv.org/abs/2303.08774
- Marin / Open Athena, "Curating 25 Trillion Tokens for LLM Pretraining" (13-gram, 50 percent share per paragraph) — https://openathena.ai/blog/marin-data-pipeline-overview/
- "A Survey on Data Contamination for Large Language Models", arXiv:2406.04244 — https://arxiv.org/abs/2406.04244
- lm-sys, llm-decontaminator (reference implementation of the LLM-based refinement step) — https://github.com/lm-sys/llm-decontaminator

## Related posts

- [Benchmark Poisoning: when the leaderboard is the attack](/posts/eval-benchmark-poisoning/) — the adversarial version of the same leak
- [MinHash and LSH for corpus deduplication](/posts/minhash-lsh-corpus-dedup/) — removing near-duplicates inside the training set
- [Paired Eval Significance: is the difference real?](/posts/paired-eval-significance/) — what to run once your eval set is clean
- [Annotator Agreement Before You Train](/posts/annotator-agreement-before-you-train/) — checking the labels the eval set rests on
