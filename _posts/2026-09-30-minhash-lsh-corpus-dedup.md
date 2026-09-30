---
title: "Your Corpus Is Duplicating Itself — MinHash + LSH Dedup You Can Measure Tonight"
date: 2026-09-30 00:00:00 +0300
categories: [Machine Learning, Data Science]
tags: [deduplication, minhash, lsh, data-quality, evaluation, contamination, jaccard, mlops]
mermaid: false
math: true
image:
  path: /assets/img/cover-minhash-lsh-corpus-dedup.webp
  alt: Document rows feeding a 112-hash MinHash signature strip, with one band of eight hashes matched, collapsing into a single kept row
---

> **The one-line version**
> Near-duplicate rows double-count your corpus, inflate your evaluation numbers, and make memorised text more likely. MinHash + LSH finds them while comparing a fraction of one percent of the row pairs. Below: the whole audit run on a 350-row corpus with exact ground truth, the banding trade-off measured rather than assumed, and the five-line contamination gate that belongs in the same pipeline.
{: .prompt-info }

## Duplicates are a measurement problem, not a storage problem

A duplicated row costs you disk. A duplicated *corpus* costs you the ability to believe anything you print afterwards. The published numbers on this are blunter than most teams expect:

- Lee et al. (ACL 2022) found that over **1% of the unprompted output** of language models trained on standard web datasets is copied verbatim from training data, that removing duplicates makes models emit memorised text **ten times less often**, and that train–test overlap affects **more than 4% of the validation sets** of those datasets — one of their examples was a single 61-word English sentence repeated over **60,000 times** in C4 [1].
- Kandpal et al. (ICML 2022) showed the memorisation rate grows *superlinearly* with duplication: a sequence present **10 times** is regenerated on average **~1,000 times more often** than a sequence present once, and existing memorisation-detection methods are close to chance on sequences that appear only once [2].
- The Pile's builders measured a **28% duplicate rate** in OpenWebText2 and **26%** in their Common Crawl data using MinHashLSH at an approximate Jaccard of 0.5 — and noted that a plain quadratic comparison of all documents "would have taken several hundred thousand years" [3].
- A 2026 study of multilingual pretraining corpora records that SlimPajama's MinHash pass **removed 49% of RedPajama's content** [4].

The evaluation side is worse, because duplication there is invisible in the score. A 2026 systematic review of 55 contamination studies found **no detection method consistently reliable** across contamination tiers and model-access settings, flagged instruction tuning as a persistent blind spot, and reported test-score inflation estimates spanning roughly **6%–40%** depending on the benchmark and the setting [5].

| Symptom you can measure | What the literature reports | Source |
|---|---|---|
| Model emits training text verbatim | >1% of unprompted output; 10× less after dedup | Lee et al. 2022 [1] |
| Memorisation rate | 10× duplication → ~1,000× more regeneration | Kandpal et al. 2022 [2] |
| Validation set overlap | >4% of the split | Lee et al. 2022 [1] |
| Web-corpus duplication | 26–28% duplicate rate | The Pile [3] |
| Benchmarks | 6%–40% inflation, no reliable detector | Nourbakhsh et al. 2026 [5] |

## Two ideas, one formula

**Jaccard similarity** between two rows is the size of the shared set divided by the size of the union — on *shingles* (here, 5-grams of words), so a reordered or lightly edited row still scores high.

**MinHash** turns each row into a fixed-length signature: apply many different hash functions to a row's shingles, keep the minimum of each. The probability that two rows agree on one hash equals their Jaccard similarity, so the fraction of agreeing rows in a 112-row signature estimates it directly.

**Banding** then makes search sublinear. Split the 112 signature rows into 14 bands of 8. Two rows become a *candidate pair* if they agree on all 8 rows of **any** band, so the probability of a candidate pair is

$$P(\text{candidate}) = 1 - (1 - s^{8})^{14}$$

where $s$ is the true Jaccard similarity. That curve has the shape you want: near zero for unrelated rows, a steep jump around the threshold $(1/14)^{1/8} \approx 0.72$, and near-certainty above 0.85. FineWeb, which published this exact configuration — 5-grams, 112 hashes, 14 buckets of 8, "targeting documents that are at least 75% similar" — computes the matching probability as **56%** at $s = 0.70$, **77%** at 0.75, **92%** at 0.80 and **98.8%** at 0.85 [6]. Hugging Face's `datatrove` defaults to the same numbers (`n_grams=5, num_buckets=14, hashes_per_bucket=8`, i.e. 112 hashes) and even encodes them in the output folder name [7]. Milvus ships MinHash LSH as a native index type for the same reason its docs give: exact pairwise Jaccard is $O(n^2)$ in time and memory, which "makes it infeasible for use cases such as LLM training corpus cleaning" [8].

The formula is a *model*, and the model assumes the 8 rows in a band match independently. Cheap hash families are not exactly min-wise independent, so treat published probabilities as a design target and measure the candidate count on your own corpus — which is what the next section does.

## The audit: 350 rows, ground truth included

The script below builds a corpus whose duplicates I control exactly — 120 distinct documents, 30 verbatim copies, and 200 edited near-copies at substitution rates of 0.5%, 2%, 5%, 10% and 20% — then computes every pair's exact Jaccard on 5-gram sets as ground truth and asks how well each LSH setting recovers it. Nothing here needs a GPU, a model, or the network.

{% raw %}
```python
"""Near-duplicate audit of a 350-row corpus: MinHash signatures + LSH banding.
Deterministic, stdlib only, no network. Ground truth = exact Jaccard on 5-gram sets."""
import hashlib, random, re
from collections import defaultdict

K = 5                      # shingle size; datatrove's default n_grams
BANDS, ROWS = 14, 8        # datatrove's / FineWeb's default: 14 buckets x 8 hashes = 112 hashes
MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def shingles(tokens, k=K):
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def mix(x, seed):
    """splitmix64: cheap, strongly mixing, one independent permutation per seed."""
    x = (x + seed) & MASK
    z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def signature(sh, rows=BANDS * ROWS):
    h = [h64(s) for s in sh]
    return tuple(min(mix(x, (i + 1) * GOLDEN) for x in h) for i in range(rows))


def jaccard(a, b):
    return len(a & b) / len(a | b)


def lsh_pairs(sigs, bands=BANDS, rows=ROWS):
    if bands * rows > len(next(iter(sigs.values()))):
        return None
    table = defaultdict(list)
    for doc, sig in sigs.items():
        for i in range(bands):
            table[(i, sig[i * rows:(i + 1) * rows])].append(doc)
    pairs = set()
    for members in table.values():
        members.sort()
        for i, x in enumerate(members):
            for y in members[i + 1:]:
                pairs.add((x, y))
    return pairs


VOCAB = ("maize rainfall harvest aflatoxin soil moisture extension farmer cooperative kiswahili "
         "fertilizer yield season satellite plot irrigation seed market price naivasha nakuru "
         "drought resilience agronomy sensor dataset model label feature training validation token "
         "embedding latency throughput index query corpus annotation batch drift retrain pipeline").split()
rng = random.Random(7)
tokens = {}
for i in range(120):                                   # 120 distinct documents
    tokens["base%03d" % i] = [rng.choice(VOCAB) for _ in range(rng.randint(120, 200))]
for i in range(30):                                    # 30 verbatim copies
    tokens["copy%03d" % i] = list(tokens["base%03d" % i])
for rate in (0.005, 0.02, 0.05, 0.10, 0.20):           # 200 edited near-copies
    for i in range(40):
        d = list(tokens["base%03d" % (i % 120)])
        for j in range(len(d)):
            if rng.random() < rate:
                d[j] = rng.choice(VOCAB)
        tokens["sub%03d_%03d" % (int(rate * 1000), i)] = d

sh = {name: shingles(t) for name, t in tokens.items()}
names = sorted(sh)
n = len(names)
total = n * (n - 1) // 2
print("corpus: %d rows, %d distinct 5-grams per row on average" % (n, sum(len(s) for s in sh.values()) // n))
print("full pairwise matrix: %d pairs" % total)

truth = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        j = jaccard(sh[a], sh[b])
        if j >= 0.2:
            truth[(a, b)] = j
print("brute force: %d pairs above 0.2 Jaccard" % len(truth))

sigs = {name: signature(s) for name, s in sh.items()}
gold = {p for p, j in truth.items() if j >= 0.8}
print("%-10s %-7s %-8s %-11s %s" % ("bands x r", "thr", "pairs", "recall@0.8", "prec@0.8"))
cand = None
for bands, rows in ((14, 8), (10, 10), (16, 4), (8, 4), (20, 6)):
    c = lsh_pairs(sigs, bands, rows)
    if c is None:
        print("%-10s skipped: needs %d hashes, signature holds %d"
              % ("%dx%d" % (bands, rows), bands * rows, len(next(iter(sigs.values())))))
        continue
    hit = gold & c
    print("%-10s %-7.2f %-8d %-11.3f %.3f" % ("%dx%d" % (bands, rows), (1.0 / bands) ** (1.0 / rows),
                                              len(c), len(hit) / len(gold), len(hit) / max(len(c), 1)))
    if (bands, rows) == (14, 8):
        cand = c

est = lambda p: sum(1 for x, y in zip(sigs[p[0]], sigs[p[1]]) if x == y) / len(sigs[p[0]])
kept = {p for p in cand if est(p) >= 0.8}
print("default config: %d candidate pairs (%.2f%% of the matrix); estimate >= 0.8 keeps %d" % (
    len(cand), 100 * len(cand) / total, len(kept)))
print("end to end: recall@0.8 = %.3f, precision = %.3f" % (
    len(gold & kept) / len(gold), len(gold & kept) / len(kept)))

print("%-8s %-9s %-8s %s" % ("hashes", "thr", "kept", "recall@0.8"))
for H in (64, 112, 256):
    s2 = {name: signature(s, H) for name, s in sh.items()}
    k2 = {p for p in lsh_pairs(s2, H // ROWS, ROWS) if sum(1 for x, y in zip(s2[p[0]], s2[p[1]]) if x == y) / H >= 0.8}
    print("%-8d %-9.2f %-8d %.3f" % (H, (ROWS / H) ** (1.0 / ROWS), len(k2),
                                     len(gold & k2) / (len(gold) + 1e-9)))

parent = {name: name for name in names}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b in kept:
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
groups = defaultdict(list)
for name in names:
    groups[find(name)].append(name)
dupes = [c for c in groups.values() if len(c) > 1]
drop = sum(len(c) - 1 for c in dupes)
print("clusters: %d duplicate groups covering %d rows; %d rows dropped (%.1f%% of the corpus)" % (
    len(dupes), sum(len(c) for c in dupes), drop, 100 * drop / n))
print("largest group: %d rows" % max(len(c) for c in dupes))
```
{% endraw %}

Output on this machine (Python 3, one core, no third-party packages):

```text
corpus: 350 rows, 157 distinct 5-grams per row on average
full pairwise matrix: 61075 pairs
brute force: 605 pairs above 0.2 Jaccard
bands x r  thr     pairs    recall@0.8  prec@0.8
14x8       0.72    233      1.000       0.712
10x10      0.79    194      0.970       0.830
16x4       0.50    403      1.000       0.412
8x4        0.59    337      1.000       0.493
20x6       skipped: needs 120 hashes, signature holds 112
default config: 233 candidate pairs (0.38% of the matrix); estimate >= 0.8 keeps 163
end to end: recall@0.8 = 0.964, precision = 0.982
hashes   thr       kept     recall@0.8
64       0.77      162      0.916
112      0.72      163      0.964
256      0.65      164      0.976
clusters: 40 duplicate groups covering 136 rows; 96 rows dropped (27.4% of the corpus)
largest group: 4 rows
```

## Reading the numbers

**The candidate set is tiny.** 233 candidate pairs out of 61,075 — **0.38%** of the matrix. Everything downstream only ever looks at those.

**Banding should be tuned for recall, not precision.** At 14×8 the bands recovered **100%** of the 605 pairs at or above 0.8 Jaccard, with 71.2% of candidates true. Tightening to 10×10 (threshold 0.79) cut candidates to 194 and lifted precision to 0.830, but missed 3% of the true pairs. Loosening to 16×4 (threshold 0.50) found everything and doubled the candidate list, dropping precision to 0.412. Cheap-and-wide is the right side to err on: a false candidate costs one exact Jaccard computation, a missed duplicate costs a duplicated row in the corpus forever.

**One extra filter converts candidates into near-certain pairs.** Estimating Jaccard from the signature itself (the fraction of the 112 rows that agree) and dropping anything below 0.8 took the same candidate set to **recall 0.964 and precision 0.982** — 163 pairs, of which essentially all are real. Note the recall cost: the estimate is a 112-sample binomial, so a handful of true pairs estimate just under the line.

**Signature length buys precision.** With 64 rows the estimate is noisy and recall@0.8 fell to **0.916**; 112 rows gives 0.964; 256 rows gives 0.976. If your evals are sensitive to a few leaked rows, pay for the hashes.

**Watch the configuration guard.** The `20x6` row in the sweep is skipped rather than computed, because 20 × 6 = 120 hashes exceeds the 112-row signature — the band keys go empty and every row collides with every other row. The first version of this script did that silently and compared all 36,315 pairs at the time; one `if bands * rows > len(signature)` check catches it.

**The clusters are where the payoff is.** 40 duplicate groups covering 136 rows; after keeping the shortest row per group, **96 rows (27.4%)** leave the corpus. That is close to the 26–28% duplicate rates the Pile's builders reported on real web data [3] — and mine is a corpus I seeded deliberately, which is the point: you cannot tell by looking, and neither can a data-cleaning job that only removes exact matches.

**Cost.** Measured here: brute force over 61,075 pairs took **0.89 s** (about 14.6 µs per pair), the signature pass took **4.66 s** for 350 rows (≈13 ms per row), and the whole script ran in **22.0–22.2 s** across three runs. Straight-line arithmetic from those two rates: at a million rows the pairwise pass is ~5 × 10¹¹ comparisons — roughly **85 days** on one core — while the signature pass is about **3.7 hours**, parallelisable, and the LSH candidate count grows with the number of *duplicate* pairs rather than the square of the corpus.

## What this pass cannot see

MinHash on word shingles measures **surface overlap**. It catches reformatted, truncated, boilerplate-wrapped, translated-and-back, and lightly edited copies — exactly the population that inflates counts. A paragraph rewritten from scratch scores far below any practical threshold and stays in the corpus. Commercial pipelines close that hole with embedding-based clustering, which costs a model forward pass per row, and the two are complements rather than substitutes: run the shingle pass first because it is cheap, deterministic and explainable, then spend embeddings only on the rows it leaves standing. Keep the decision explicit either way — "duplicate" is defined by the metric you choose, and a dedup rule nobody wrote down is a licence to drop rows for reasons a reviewer cannot reconstruct.

## The five-line gate for eval sets

Corpus dedup fixes the training side. The same machinery catches the evaluation side, and there is a weaker check you should run even if you never build an index: 13-gram overlap between each eval item and the whole training corpus. This is the measure Lee et al. and the contamination literature use, and it is a few lines:

{% raw %}
```python
"""13-gram eval/train overlap check — the cheapest contamination gate."""
import re

EVAL = {
    "eval-01": "Which pest causes the largest yield loss in rainfed maize in western Kenya, and what "
               "spraying interval does the county extension service recommend once the whorl is infested?",
    "eval-02": "What plant spacing and seed rate does the county agronomy office recommend for hybrid "
               "maize planted in the highlands of Nakuru during the long rains season?",
    "eval-03": "How much aflatoxin is permitted in maize flour offered for sale in Kenya, and which "
               "laboratory carries out the confirmatory test for millers in the eastern region?",
    "eval-04": "Explain how the stem borer was controlled in the trial plots in Kakamega and what "
               "yield difference the trial recorded between the treated and untreated blocks.",
    "eval-05": "Which soil test result would stop a farmer from top-dressing nitrogen at tasselling, "
               "and what alternative does the agronomy manual suggest for that field?",
    "eval-06": "How many consecutive days of dry spell trigger the drought early warning alert in "
               "Kitui, and which office publishes the alert bulletin during the short rains?",
    "eval-07": "What is the subsidy price for a fifty kilogram bag of planting fertiliser this season, "
               "and which depot network is authorised to redeem the subsidy vouchers?",
    "eval-08": "Which agronomy extension channel reached the most farmers in Nyanza last year, and "
               "what share of the sampled households reported using the advice on their own plots?",
}
FILLER = ("field notes from the extension round report that soil moisture recovered after the "
          "short rains and the cooperative recorded higher delivery volumes at the depot gate ")

TRAIN = {"train%02d" % i: FILLER * 6 for i in range(30)}
# three rows carry an eval item: verbatim, with one word changed, and rephrased
TRAIN["train07"] += EVAL["eval-01"] + " " + FILLER
TRAIN["train11"] += EVAL["eval-02"].replace("recommend", "prefer") + " " + FILLER
TRAIN["train19"] += ("In western Kenya under rainfed conditions the pest with the largest impact on "
                     "maize yield is the stem borer, followed by fall armyworm, and the trial blocks "
                     "that were sprayed twice recorded a higher yield than the untreated blocks.")


def grams(text, k):
    t = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(t[i:i + k]) for i in range(len(t) - k + 1)}


K = 13
train_grams = set()
for row in TRAIN.values():
    train_grams |= grams(row, K)
print("train: %d rows, %d distinct %d-grams" % (len(TRAIN), len(train_grams), K))
print("%-8s %-9s %-9s %s" % ("item", "13-grams", "overlap", "verdict"))
for key in sorted(EVAL):
    g = grams(EVAL[key], K)
    ov = len(g & train_grams) / len(g)
    verdict = "CONTAMINATED" if ov >= 0.5 else ("partial" if ov > 0 else "clean")
    print("%-8s %-9d %-9.3f %s" % (key, len(g), ov, verdict))
print("verbatim copies found:", sum(1 for key in EVAL
                                    if len(grams(EVAL[key], K) & train_grams) / len(grams(EVAL[key], K)) >= 0.5))
REPHRASED = ("In the western counties, when maize is grown without irrigation, which insect pest "
             "does the most damage to the harvest, and how often should the crop be sprayed after "
             "the growing point of the plant has been attacked?")
ov = len(grams(REPHRASED, K) & train_grams) / len(grams(REPHRASED, K))
print("rephrased version of eval-01: %d 13-grams, overlap %.3f -> overlap cannot see paraphrase" % (
    len(grams(REPHRASED, K)), ov))
```
{% endraw %}

```text
train: 30 rows, 143 distinct 13-grams
item     13-grams  overlap   verdict
eval-01  16        1.000     CONTAMINATED
eval-02  14        0.143     partial
eval-03  15        0.000     clean
eval-04  14        0.000     clean
eval-05  13        0.000     clean
eval-06  14        0.000     clean
eval-07  14        0.000     clean
eval-08  15        0.000     clean
verbatim copies found: 1
rephrased version of eval-01: 26 13-grams, overlap 0.000 -> overlap cannot see paraphrase
```

Three lessons are visible in fourteen lines of output. A verbatim leak scores **1.000** — no threshold argument, the gate just fires. Changing **one word** ("recommend" → "prefer") collapses the overlap from 1.000 to **0.143**, because every 13-gram spanning that word is destroyed; the item still trips a 0.5 rule, but a heavier synonym pass would not. And a genuinely **rephrased** version of the leaked question scores **0.000** against the corpus that contains the original verbatim. That last line is the honest limit of the method, and it matches what the 2026 review concluded from 55 studies: string-matching, likelihood-based, membership-inference and auditing families all have failure modes, and none dominates [5]. So use the 13-gram gate as a tripwire for the easy cases — and use the MinHash index, with a lower similarity threshold, when you want to catch the rewritten ones.

## How to wire this into a pipeline

1. **Deduplicate before you split.** Run the pass on the raw pool, then split. Deduplicating train and test separately leaves the cross-split overlap exactly where it hurts most.
2. **Cache the signatures, and reuse the index across splits.** `datatrove`'s pipeline writes signatures per bucket and lets a later run take an `index_folder` — "load all index files in this folder and use them as a reference … remove any matches on our dataset with signatures from the index" [7]. That is the mechanism for filtering a new batch, or a validation set, against the corpus you already trained on.
3. **Pick bands for recall, filter for precision.** Bands are cheap; the exact Jaccard check (or the signature estimate) is the expensive step, so it belongs after the candidate set, and its threshold can be set from your own numbers rather than the textbook curve.
4. **Decide the keep rule explicitly and log it.** Keeping the shortest row of each cluster is a reasonable default; keeping the highest-quality row needs a quality signal you trust. Whatever you choose, record the cluster ID on the surviving row — otherwise a future metric jump has no explanation attached.
5. **Record three numbers in your datasheet:** candidate-pair percentage, rows removed, and the banding configuration. `datatrove` encodes the config in its folder name (`5ng_14bs_8hs`) for exactly this reason [7]. A corpus with 27% of rows removed and one with 2% removed are different corpora, even at the same token count.
6. **Guard the configuration and re-measure on your own hash family.** Assert `bands × rows ≤ signature length`, and check the per-row agreement rate against known-Jaccard pairs before trusting a threshold: the banding formula assumes independent rows, which holds only approximately for cheap hash families.

## Key takeaways

| Question | Answer from this run |
|---|---|
| How much of the pairwise matrix must I compare? | 0.38% (233 candidates out of 61,075 pairs) |
| Does banding miss real duplicates? | At 14×8, recall@0.8 was 1.000 on 605 known pairs |
| What do false candidates cost? | One exact Jaccard computation — 71.2% precision before filtering, 98.2% after |
| How many hashes? | 64 → recall 0.916; 112 → 0.964; 256 → 0.976 |
| What does the corpus lose? | 96 of 350 rows (27.4%) in 40 duplicate groups |
| Does it scale? | Pairwise ≈ 14.6 µs/pair (≈85 days at 1M rows, one core); signatures ≈ 13 ms/row (≈3.7 h at 1M rows, parallel) |
| Can the same code check contamination? | Yes — 13-gram overlap flags verbatim leaks instantly, and cannot see paraphrase |

## References

1. Lee et al., *Deduplicating Training Data Makes Language Models Better* (ACL 2022) — [arxiv.org/abs/2107.06499](https://arxiv.org/abs/2107.06499)
2. Kandpal et al., *Deduplicating Training Data Mitigates Privacy Risks in Language Models* (ICML 2022) — [arxiv.org/abs/2202.06539](https://arxiv.org/abs/2202.06539)
3. Gao et al., *The Pile: An 800GB Dataset of Diverse Text for Language Modeling* — [arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027)
4. Alrashed & Orabona, *Mix, MinHash, and Match: Cross-Source Agreement for Multilingual Pretraining Datasets* (KAUST, 2026) — [arxiv.org/abs/2512.18834](https://arxiv.org/abs/2512.18834)
5. Nourbakhsh et al., *Are LLM Benchmarks Already Contaminated? A Systematic Review of Contamination Detection Methods* (GEM 2026) — [aclanthology.org/2026.gem-main.50](https://aclanthology.org/2026.gem-main.50/)
6. Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (NeurIPS 2024) — [arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557)
7. Hugging Face `datatrove`, MinHash dedup configuration and pipeline — [github.com/huggingface/datatrove](https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py)
8. Milvus documentation, *MINHASH_LSH* index — [milvus.io/docs/minhash-lsh.md](https://milvus.io/docs/minhash-lsh.md)
9. Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, ch. 3 (banding and the S-curve) — [mmds.org](http://www.mmds.org/)
10. Ekzhu, *MinHash LSH* (datasketch) — false positives, false negatives and threshold behaviour — [ekzhu.com/datasketch/lsh.html](https://ekzhu.com/datasketch/lsh.html)

## Related posts

- [RAG Recall@k: The Denominator Nobody Prints](/posts/rag-recall-at-k-denominator/) — the metric bugs that duplicate and unlabelled rows hide inside
- [Benchmark Poisoning: When Your Eval Set Learns the Answers](/posts/eval-benchmark-poisoning/) — contamination detection methods and defences
- [Fraud Model Drift Monitoring](/posts/fraud-model-drift-monitoring/) — PSI and the data-quality gates that come before the score
- [Embedding Compression Audit](/posts/embedding-compression-audit/) — measuring a similarity-preserving transform instead of trusting the vendor
