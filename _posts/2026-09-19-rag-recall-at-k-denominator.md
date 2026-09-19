---
title: "The Denominator Is the Metric: Auditing a RAG Retriever Before You Blame the Model"
date: 2026-09-19 00:00:00 +0300
categories: [AI Engineering, Machine Learning]
tags: [rag, retrieval evaluation, information retrieval, recall at k, ndcg, llm evaluation, golden set]
math: true
image:
  path: /assets/img/cover-rag-recall-at-k-denominator.webp
  alt: A ranked top-5 result list with four green hits beside three denominator bars of shrinking length labelled 12 truly relevant, 8 judged and 5 capped at k, producing recall of 0.417, 0.500 and 0.800
---

## Introduction

Most arguments about retrieval-augmented generation start at the prompt, which is usually the wrong end. If the passage that answers the question is not in the context window, no prompt and no model swap gets it back. The [RAGAS authors](https://arxiv.org/abs/2309.15217) split the problem in three: whether the retrieval system "identifies relevant and focused context passages," whether the model "exploit[s] such passages in a faithful way," and the quality of the generation itself. Only the first is decided before the LLM sees anything.

So the retriever's number deserves more scrutiny than the prompt. Most retrieve-quality reports that look wrong are not retrieval failures — they are denominator failures. Recall is a fraction, and the numerator is usually honest while the denominator changed shape between the code that produced the number and the reality it claims to describe.

> **The framing**
> This is the measurement layer for RAG — not embedding choice for low-resource languages ([RAG for Low-Resource African Languages](/posts/rag-low-resource-african-languages/)), not graph-structured retrieval ([KG-LLM RAG](/posts/kg-llm-rag/)), not retrieval-time attacks ([RAG Security Attacks](/posts/rag-security-attacks/)) or model-level benchmarks ([Evaluating LLMs for African Use Cases](/posts/evaluating-llms-african-use-cases/)). It is about earning the number: four metrics in standard-library Python, three ways Recall@k inflates without anyone lying, and the audit function that makes the inflation impossible to ship.
{: .prompt-info }

## Why the retriever sets the ceiling

Every stage of the funnel can only lose documents:

$$\text{corpus} \;\rightarrow\; \text{fetch-}K \;\rightarrow\; \text{rerank} \;\rightarrow\; \text{keep-}n \;\rightarrow\; \text{prompt}$$

Recall@k is the fraction of the relevant documents for a query that appear in the top k results:

$$\text{Recall@}k = \frac{\lvert R_k \cap Rel \rvert}{\lvert Rel \rvert}$$

The numerator counts hits inside the cutoff. The denominator is the **complete judged relevant set for that query** — not k, not the candidate list you happened to return, not the number of documents your label file covers. [123ofAI's guide](https://123ofai.com/articles/blocks/recall-at-k) states it exactly: "The denominator is the total relevant per query, not K. This is the key difference between Recall@K and Precision@K, which divides by K."

That denominator also sets a ceiling nobody prints: the numerator cannot exceed k, so Recall@k can never beat $\min(1, k / \lvert Rel \rvert)$. With eight relevant chunks and a cutoff of five, the best any retriever on earth can score is 0.625 — so a report showing "recall@5 = 0.50" and sending you to retune the embedding model hides the more useful sentence: *80% of the ceiling, and five more relevant documents than the cutoff can hold.*

## Four metrics, one of which is a fraction

| Metric | The question it answers | Denominator | What it cannot see |
|---|---|---|---|
| Precision@k | Of the k slots I paid for, how many are relevant? | k | Anything below the cutoff — one perfect page scores 1.0 |
| **Recall@k** | Of all relevant documents, how many did I surface? | **\|Rel\| (judged)** | Needs complete labels; ceiling is k/\|Rel\| |
| MRR | How high is the *first* hit? | Queries, via 1/rank | Everything after the first hit |
| nDCG@k | How good is the *ordering*, with graded relevance? | The ideal DCG | Depends on agreed grades |

Two of the four are cut-off aware but order-blind. [Evidently's guide](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k) is blunt about the consequence: precision and recall at K "only reflect the number of relevant items in the top K without evaluating the ranking quality inside a list." NDCG adds that: it discounts each hit by $\log_2(\text{rank}+1)$ and normalises by the ideal ordering (DCG/IDCG, where 1.0 matches the ideal). MRR answers a narrower question — how quickly the first useful passage appears.

## The evaluation set: 24 chunks, four queries

You do not need 100,000 labelled examples to catch a denominator bug. Twenty-four chunks and four queries with hand-assigned grades is enough, and it fits in a file you can read.

{% raw %}
```python
import math

# ---------- the metrics (stdlib only) ----------
def precision_at_k(ranked, relevant, k):
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / k if k else 0.0


def recall_at_k(ranked, relevant, k):
    """Correct: denominator is the whole judged relevant set for this query."""
    if not relevant:
        raise ValueError("Recall@k undefined: no judged relevant docs for this query")
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def recall_at_k_capped(ranked, relevant, k):
    """WRONG: denominator capped at k. Inflates every query where |Rel| > k."""
    top = ranked[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / min(len(relevant), k)


def mrr(ranked, relevant):
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, grades, k):
    dcg = sum(grades.get(d, 0) / math.log2(i + 1)
              for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


# ---------- a 24-chunk corpus, labelled by hand ----------
# m = maize pests, s = storage/aflatoxin, d = dairy, p = poultry, x = soil/water
CORPUS = ([f"m{i}" for i in range(1, 7)] + [f"s{i}" for i in range(1, 5)]
          + [f"d{i}" for i in range(1, 6)] + [f"p{i}" for i in range(1, 6)]
          + [f"x{i}" for i in range(1, 5)])

# (query id, judged grades {doc: 0-3}, retrieved ranking, truly-relevant set)
EVAL_SET = [
    ("q_faw", {"m1": 3, "m2": 2, "m3": 1},
     ["m2", "x3", "m1", "p1", "d2", "m3", "s1", "x1", "p3", "d5"],
     {"m1", "m2", "m3"}),
    ("q_store", {"s1": 3, "s2": 2, "s3": 1},
     ["d2", "s2", "x4", "s1", "p2", "s3", "m4", "d1", "p5", "x2"],
     {"s1", "s2", "s3"}),
    ("q_milk", {"d1": 3, "d2": 2, "d3": 1},
     ["d1", "s2", "x1", "d2", "x3", "d3", "m1", "p4", "s4", "x2"],
     {"d1", "d2", "d3"}),
    ("q_broad", {"m1": 3, "m2": 3, "m3": 2, "m4": 2, "m5": 1, "m6": 1, "s1": 2, "s2": 1},
     ["m1", "s1", "m2", "m3", "x1", "m4", "m5", "m6", "s2", "d1"],
     {"m1", "m2", "m3", "m4", "m5", "m6", "s1", "s2", "s3", "x1", "x2", "s4"}),
]

K = 5
print(f"corpus: {len(CORPUS)} chunks | queries: {len(EVAL_SET)} | k = {K}")
print(f"\n{'query':8s} {'|Rel|':>5s} {'P@5':>6s} {'R@5':>6s} {'R@5cap':>7s} {'MRR':>6s} {'nDCG@5':>7s}")
for qid, grades, ranked, true_rel in EVAL_SET:
    rel = set(grades)
    print(f"{qid:8s} {len(rel):5d} "
          f"{precision_at_k(ranked, rel, K):6.3f} "
          f"{recall_at_k(ranked, rel, K):6.3f} "
          f"{recall_at_k_capped(ranked, rel, K):7.3f} "
          f"{mrr(ranked, rel):6.3f} "
          f"{ndcg_at_k(ranked, grades, K):7.3f}")

def mean(vals): return sum(vals) / len(vals)

print(f"\n{'metric':10s} {'correct':>8s} {'capped':>8s}")
print(f"{'mean R@5':10s} "
      f"{mean([recall_at_k(r, set(g), K) for _, g, r, _ in EVAL_SET]):8.3f} "
      f"{mean([recall_at_k_capped(r, set(g), K) for _, g, r, _ in EVAL_SET]):8.3f}")
print(f"{'mean P@5':10s} "
      f"{mean([precision_at_k(r, set(g), K) for _, g, r, _ in EVAL_SET]):8.3f}")
print(f"{'mean nDCG@5':10s} "
      f"{mean([ndcg_at_k(r, g, K) for _, g, r, _ in EVAL_SET]):8.3f}")

print("\nlabel coverage (judged vs truly relevant):")
print(f"{'query':8s} {'judged':>7s} {'true':>6s} {'R@5 judged':>11s} {'R@5 true':>9s}")
for qid, grades, ranked, true_rel in EVAL_SET:
    print(f"{qid:8s} {len(grades):7d} {len(true_rel):6d} "
          f"{recall_at_k(ranked, set(grades), K):11.3f} "
          f"{recall_at_k(ranked, true_rel, K):9.3f}")

print("\nstage-1 (fetch 10) vs stage-2 (keep 5) -- the reranker's ceiling:")
print(f"{'query':8s} {'R@10':>6s} {'R@5':>6s} {'lost by reranker':>17s}")
for qid, grades, ranked, true_rel in EVAL_SET:
    rel = set(grades)
    r10, r5 = recall_at_k(ranked, rel, 10), recall_at_k(ranked, rel, K)
    print(f"{qid:8s} {r10:6.3f} {r5:6.3f} {r10 - r5:17.3f}")

print("\nsame top-5 set, reordered (d1 moved from rank 1 to rank 5):")
qid, grades, ranked, true_rel = EVAL_SET[2]
buried = [ranked[1], ranked[2], ranked[3], ranked[0], ranked[4]]
rel = set(grades)
print(f"  P@5     {precision_at_k(buried, rel, K):.3f}  (was {precision_at_k(ranked, rel, K):.3f})")
print(f"  R@5     {recall_at_k(buried, rel, K):.3f}  (was {recall_at_k(ranked, rel, K):.3f})")
print(f"  MRR     {mrr(buried, rel):.3f}  (was {mrr(ranked, rel):.3f})")
print(f"  nDCG@5  {ndcg_at_k(buried, grades, K):.3f}  (was {ndcg_at_k(ranked, grades, K):.3f})")
```
{% endraw %}

Run it and you get every number this post quotes:

```
corpus: 24 chunks | queries: 4 | k = 5

query    |Rel|    P@5    R@5  R@5cap    MRR  nDCG@5
q_faw        3  0.400  0.667   0.667  1.000   0.735
q_store      3  0.400  0.667   0.667  0.500   0.536
q_milk       3  0.400  0.667   0.667  1.000   0.811
q_broad      8  0.800  0.500   0.800  1.000   0.880

metric      correct   capped
mean R@5      0.625    0.700
mean P@5      0.500
mean nDCG@5    0.741

label coverage (judged vs truly relevant):
query     judged   true  R@5 judged  R@5 true
q_faw          3      3       0.667     0.667
q_store        3      3       0.667     0.667
q_milk         3      3       0.667     0.667
q_broad        8     12       0.500     0.417

stage-1 (fetch 10) vs stage-2 (keep 5) -- the reranker's ceiling:
query      R@10    R@5  lost by reranker
q_faw     1.000  0.667             0.333
q_store   1.000  0.667             0.333
q_milk    1.000  0.667             0.333
q_broad   1.000  0.500             0.500

same top-5 set, reordered (d1 moved from rank 1 to rank 5):
  P@5     0.400  (was 0.400)
  R@5     0.667  (was 0.667)
  MRR     0.333  (was 1.000)
  nDCG@5  0.481  (was 0.811)
```

## Bug one: the denominator capped at k

Look at the `R@5cap` column. For `q_broad` the honest score is **0.500** and the capped variant reports **0.800** — a 60% overstatement from a single `min()`. Across all four queries the mean moves from 0.625 to 0.700. Nobody had to fabricate a hit; the implementation just divided by the wrong thing.

The bug is easy to write and hard to notice, because `min(len(relevant), k)` looks like defensive programming. But it silently rescales every query with more relevant documents than the cutoff — exactly the queries where coverage is hard and the metric matters most. With a fine-grained corpus (chunks, not documents) and a small cutoff, most queries have more relevant chunks than slots, so the bug is systematic rather than occasional.

## Bug two: labels you never collected

The second inflation needs no code at all, just a label file less complete than reality. For `q_broad`, 8 documents were judged and 12 are genuinely relevant, so recall@5 reads **0.500** against the label file and **0.417** against the truth. 123ofAI's guide names the direction of the error: with partial labels "your Recall@K will be an overestimate because the true \|Rel\| is larger than what you measured."

This is not a rookie mistake — it is how every large test collection is built. Judgments come from *pooling*: you collect the top results of a set of submitted runs, judge only those, and treat unjudged documents as non-relevant. Buckley and Voorhees showed the limit of that assumption in [Bias and the limits of pooling for large collections](https://link.springer.com/article/10.1007/s10791-007-9032-x) (*Information Retrieval* 10:491–508, 2007): as document sets grow a constant-size pool "represents an increasingly small" sample, and judgment sets "can be biased in that they favor relevant documents that contain topic title words." A pooled judgment set is not the relevant set; it is a sample drawn around what some earlier system already found.

For an African-language or domain-specific build your label file is the weakest link — and it was probably written by the same retriever you are evaluating, since you judged what it returned. Two cheap mitigations:

- **Pool from more than one retriever.** Judge the union of BM25, your dense retriever and one reranker's top-20; different systems miss different documents, and the union is what makes the denominator honest.
- **Borrow a public collection to check the shape of your numbers.** [CIRAL](https://huggingface.co/datasets/CIRAL/ciral) (SIGIR '24, pp. 293–302) provides English queries with query-passage judgments for **Hausa, Somali, Swahili and Yoruba** in TREC qrels format, so you can score a multilingual retriever against a collection you did not build. Carry one caveat into the write-up: CIRAL's judgments are themselves pooled, so a recall figure against it is a lower bound, not the truth.

## Bug three: duplicates, and the ceiling nobody prints

The last silent inflator is an index holding the same chunk twice — a re-ingest that did not delete the old copy, or a document in two source folders. The audit function below prints the numbers that make a recall figure readable, and catches the duplicate:

{% raw %}
```python
def audit(query_id, grades, ranked, corpus_size, k):
    """Never report Recall@k without: |Rel|, k, label coverage and the dedup count."""
    problems, seen, dupes = [], set(), []
    for doc in ranked:
        if doc in seen:
            dupes.append(doc)
        seen.add(doc)
    if not grades:
        problems.append("no judged relevant docs -> Recall@k is undefined")
    if dupes:
        problems.append(f"duplicate ids in the ranking: {sorted(set(dupes))} - a hit counted twice")
    if 0 < len(grades) < k:
        problems.append(f"|Rel|={len(grades)} < k={k}: Precision@k can never exceed "
                        f"{len(grades)/k:.2f}, whatever the retriever does")
    ceiling = min(1.0, k / len(grades)) if grades else 0.0   # Recall@k can never exceed k/|Rel|
    distinct_top = list(dict.fromkeys(ranked[:k]))            # dedupe, keep order
    positional = sum(1 for d in ranked[:k] if d in grades)    # a duplicate scores twice
    hits = sum(1 for d in distinct_top if d in grades)
    achieved = hits / len(grades) if grades else 0.0
    if ceiling:
        print(f"{query_id:9s} |Rel|={len(grades):2d} k={k} labels={len(grades)/corpus_size:6.1%} "
              f"ceiling={ceiling:.3f} R@k={achieved:.3f} ({achieved/ceiling:.0%} of ceiling)")
    else:
        print(f"{query_id:9s} Recall@k UNDEFINED")
    if positional != hits:
        problems.append(f"positional count says {positional} hits, "
                        f"distinct documents say {hits}")
    for p in problems:
        print(f"          ! {p}")


audit("q_faw",   {"m1": 3, "m2": 2, "m3": 1}, ["m2", "x3", "m1", "p1", "d2"], 24, 5)
audit("q_store", {"s1": 3, "s2": 2, "s3": 1}, ["d2", "s2", "x4", "s1", "p2"], 24, 5)
audit("q_milk",  {"d1": 3, "d2": 2, "d3": 1}, ["d1", "s2", "x1", "d2", "x3"], 24, 5)
audit("q_broad", {"m1": 3, "m2": 3, "m3": 2, "m4": 2, "m5": 1, "m6": 1, "s1": 2, "s2": 1},
      ["m1", "s1", "m2", "m3", "x1"], 24, 5)
audit("q_dupe",  {"d1": 3, "d2": 2, "d3": 1}, ["d1", "d1", "x3", "d2", "x1"], 24, 5)
```
{% endraw %}

```
q_faw     |Rel|= 3 k=5 labels= 12.5% ceiling=1.000 R@k=0.667 (67% of ceiling)
          ! |Rel|=3 < k=5: Precision@k can never exceed 0.60, whatever the retriever does
q_store   |Rel|= 3 k=5 labels= 12.5% ceiling=1.000 R@k=0.667 (67% of ceiling)
          ! |Rel|=3 < k=5: Precision@k can never exceed 0.60, whatever the retriever does
q_milk    |Rel|= 3 k=5 labels= 12.5% ceiling=1.000 R@k=0.667 (67% of ceiling)
          ! |Rel|=3 < k=5: Precision@k can never exceed 0.60, whatever the retriever does
q_broad   |Rel|= 8 k=5 labels= 33.3% ceiling=0.625 R@k=0.500 (80% of ceiling)
q_dupe    |Rel|= 3 k=5 labels= 12.5% ceiling=1.000 R@k=0.667 (67% of ceiling)
          ! duplicate ids in the ranking: ['d1'] - a hit counted twice
          ! |Rel|=3 < k=5: Precision@k can never exceed 0.60, whatever the retriever does
          ! positional count says 3 hits, distinct documents say 2
```

Three things this output buys you. The **ceiling** turns a bare 0.500 into "80% of the maximum score at this cutoff" — a different engineering conversation. The **label coverage** (12.5%, 33.3%) shows which queries are measured and which are guessed. The **duplicate check** catches the worst case: `q_dupe` scores 0.667 on distinct documents while a positional count reports 3 hits from 3 relevant — a duplicated chunk counting a hit twice.

## Measure the two stages separately

The stage-1/stage-2 table ends the "the retriever is bad" argument: all four queries have **R@10 = 1.000**, so at fetch depth 10 every relevant document was already in the candidate list. After the cutoff to 5, `q_broad` falls to 0.500 and the others to 0.667. The loss is between fetching and keeping — the reranker or the keep-n choice — and retraining the embedding model will not fix it.

Hence the two numbers must be logged apart, per query, from day one. Pinecone's [reranker walkthrough](https://www.pinecone.io/learn/series/rag/rerankers/) frames them as different jobs: "maximize retrieval recall by retrieving plenty of documents and then maximize LLM recall by minimizing the number of documents that make it to the LLM." One end-to-end recall number cannot say which job failed. The Neural Base's retrieval course puts the stage-1 bar at "recall@k in stage 1 must be >95% or reranking cannot compensate," measured by "logging whether the gold answer was in your top-k before reranking" — the R@10 column above.

## Coverage is not order

The reordered block at the end of the first script keeps the **same five documents** in the top-5 and only changes their order — `d1` moves from rank 1 to rank 5. Precision@5 stays 0.400 and Recall@5 stays 0.667, because neither metric can see inside the list. MRR falls from 1.000 to 0.333 and nDCG@5 from 0.811 to 0.481.

That gap is why both metric families are worth keeping:

- Recall high, nDCG low: the reranker buried documents it had — fix the reranker or raise `n`, not the retriever.
- Both low: the documents never reached the candidate list — fix chunking, the embedding model or the fetch depth `K`.
- MRR sits in between: how quickly the first useful passage appears.

## Judging the generator

Retriever metrics are computable once you have labels. Faithfulness is not, which is why LLM-judge frameworks exist — and why they should be treated as instruments with error bars, not oracles.

- **RAGAS** ([arXiv 2309.15217](https://arxiv.org/abs/2309.15217)) evaluates the same three dimensions reference-free, "without having to rely on ground truth human annotations." **ARES** ([arXiv 2311.09476](https://arxiv.org/abs/2311.09476), NAACL 2024) fine-tunes lightweight judges, then corrects them with prediction-powered inference from "a small set of human-annotated datapoints" (a few hundred, across eight knowledge-intensive tasks); its judges "remain effective across domain shifts."
- **RAGBench** ([arXiv 2407.11005](https://arxiv.org/abs/2407.11005)) is the sobering one: across a 100k-example benchmark over five industry domains, "LLM-based RAG evaluation methods struggle to compete with a finetuned RoBERTa model on the RAG evaluation task." A general-purpose model is not automatically the better judge.
- **Judges have position bias.** [Judging the Judges](https://aclanthology.org/2025.ijcnlp-long.18.pdf) (IJCNLP 2025) measures it by swapping the compared items and checking *position consistency*, and finds some judges consistently prefer whichever answer came first. Swap-and-average before trusting a pairwise score.
- **Calibrate against humans and publish the agreement.** Sample 50–100 traces a week, label them on the same rubric, and compute Cohen's kappa: "Target kappa is 0.6 or higher. Below 0.6, the judge is too noisy to trust."

Same discipline as the denominator: report the label size, judge model and agreement number next to the score.

## How to apply this: a five-step harness

1. **Write 30–50 queries before any retrieval code.** Record who judged, what they saw, and the grade scale; a `golden.yaml` of `{query, grades: {doc_id: 0-3}}` is enough.
2. **Log four numbers per query per run:** fetch-recall@K, keep-recall@n, MRR, nDCG@n — plus `|Rel|`, `k` and label coverage, in a table so a regression is a diff, not a vibe.
3. **Guard the denominator in code.** Fail the build on `|Rel| < k`, duplicate ids in a ranking, or label coverage below threshold. A fixture with the obvious cases — all relevant retrieved, none retrieved, one of three, a hit just below the cutoff, fewer results than k, an empty relevant set — stops it drifting silently.
4. **Sanity-check the shape against a public collection.** [BEIR](https://arxiv.org/abs/2104.08663) scored 18 datasets and 10 retrieval systems on nDCG@10 and found BM25 "a robust baseline" while re-ranking models led zero-shot "at high computational costs" — a reminder that a fancy retriever losing to BM25 is telling you about your chunking, not embeddings. For Hausa, Somali, Swahili or Yoruba, use CIRAL.
5. **Give the judge its own error bar.** Pin the judge model, report Cohen's kappa against human labels, and re-calibrate on every judge change — otherwise a judge upgrade looks like a quality change.

| Layer | Metric | Working threshold | Source |
|---|---|---|---|
| Stage 1 (fetch-K) | Recall@K | > 0.95 before tuning the reranker | The Neural Base, IR course |
| Faithfulness | judge score | 0.7+ narrow domain, 0.5+ broad | Future AGI, RAG metrics |
| Coverage | Recall@k at k=20 | 0.8+ for broad corpora | Future AGI, RAG metrics |
| Judge trust | Cohen's kappa vs humans | ≥ 0.6, else re-tune or swap the judge | Future AGI, RAG metrics |

## Key takeaways

| # | Takeaway |
|---|---|
| 1 | Recall@k is a fraction. Capping its denominator at `k` inflates `q_broad` from 0.500 to 0.800, and the four-query mean from 0.625 to 0.700 — same hits, wrong divisor. |
| 2 | The denominator is the *complete judged* relevant set. Partial labels inflate the score (0.500 vs 0.417 here); pooled judgments are a sample, not the truth. |
| 3 | Print the ceiling $\min(1, k/\lvert Rel \rvert)$: with 8 relevant chunks and k=5, 0.625 is the best score available. |
| 4 | Log stage-1 and stage-2 recall separately. R@10 = 1.000 here, so the loss is in the cutoff — the retriever was never the problem. |
| 5 | Recall cannot see ordering: one document moved from rank 1 to rank 5 holds recall at 0.667 while nDCG@5 falls 0.811 → 0.481. |
| 6 | Deduplicate before scoring: a duplicated chunk scored 3 "hits" on 3 relevant documents — a reported 1.000 that is really 0.667. |
| 7 | Trust the judge only as far as its kappa: fine-tuned small judges beat general LLM judges on RAGBench, and pairwise judges carry position bias. |

## References

1. Es, James, Espinosa-Anke, Schockaert. [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), arXiv:2309.15217.
2. Saad-Falcon, Khattab, Potts, Zaharia. [ARES: An Automated Evaluation Framework for RAG](https://arxiv.org/abs/2311.09476), arXiv:2311.09476, NAACL 2024.
3. Friel, Belyi, Sanyal. [RAGBench: Explainable Benchmark for RAG Systems](https://arxiv.org/abs/2407.11005), arXiv:2407.11005.
4. Thakur, Reimers, Rücklé, Srivastava, Gurevych. [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models](https://arxiv.org/abs/2104.08663), arXiv:2104.08663, NeurIPS 2021.
5. Buckley, Dimmick, Soboroff, Voorhees. [Bias and the limits of pooling for large collections](https://link.springer.com/article/10.1007/s10791-007-9032-x), *Information Retrieval* 10:491–508, 2007.
6. [Recall@K in ML Systems](https://123ofai.com/articles/blocks/recall-at-k) — denominator and partial-label effects.
7. [Precision and recall at K](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k) and [NDCG explained](https://www.evidentlyai.com/ranking-metrics/ndcg-metric), Evidently AI.
8. Adeyemi, Oladipo, Zhang, … Lin. [CIRAL: A Test Collection for CLIR Evaluations in African Languages](https://doi.org/10.1145/3626772.3657884), SIGIR '24, 293–302; data: [CIRAL/ciral](https://huggingface.co/datasets/CIRAL/ciral).
9. [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://aclanthology.org/2025.ijcnlp-long.18.pdf), IJCNLP 2025.
10. [RAG evaluation metrics](https://futureagi.com/blog/rag-evaluation-metrics-2025/) — thresholds and judge-calibration targets.
11. [Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/), Pinecone; [Retrieve many, rerank top-k](https://theneuralbase.com/information-retrieval/learn/intermediate/retrieve-many-rerank-top-k/).

## Related posts

- [RAG for Low-Resource African Languages](/posts/rag-low-resource-african-languages/) — what to do when language coverage is the bottleneck.
- [KG-LLM RAG](/posts/kg-llm-rag/) — graph-structured retrieval, and how it changes the candidate set you measure.
- [Evaluating LLMs for African Use Cases](/posts/evaluating-llms-african-use-cases/) — model-level benchmarks, the layer beyond these retrieval metrics.
- [Temporal Validation for Fraud Models](/posts/temporal-validation-fraud-models/) — the same lesson one layer over: a split that borrows information produces a number you did not earn.
- [ML Monitoring](/posts/ml-monitoring/) — keeping these metrics on a schedule once the pipeline is live.
