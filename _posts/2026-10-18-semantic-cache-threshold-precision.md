---
title: "The Cache Was Confident and Wrong: Choosing a Semantic Cache Threshold You Can Defend"
date: 2026-10-18 00:00:00 +0300
categories: [LLM, AI Engineering]
tags: [semantic caching, embeddings, evaluation, llm cost, vector search, llmops]
image:
  path: /assets/img/cover-semantic-cache-threshold-precision.webp
  alt: A query card for account 8890 being sorted into a cache slot that holds account 4471's answer, stamped WRONG
---

> **A semantic cache is the only component in your stack whose failures are recorded as savings.** A hit is logged as an avoided call, a miss is logged as cost, and nothing in the default telemetry records that the answer was right for a different question. This post measures the threshold that decides it: a 12-intent FAQ, 24 true paraphrases, and 14 queries whose correct answer is not in the cache at all.
{: .prompt-info }

The reference design is simple enough that it gets copied without scrutiny: embed the question, search the cache, compare cosine similarity against a fixed threshold, serve the stored answer if it clears. The published numbers look excellent. GPT Semantic Cache reports 61.6%–68.8% cache hit rates and a positive hit rate above 97% using a threshold of 0.8 with `all-MiniLM-L6-v2` over 8,000 question-answer pairs ([Regmi & Pun, arXiv:2411.05276](https://arxiv.org/abs/2411.05276)).

Then you deploy it on real traffic and discover the second number. A banking team running 1,000 real queries through seven bi-encoders found false positive rates reaching 99% for some models at a threshold of 0.7, and got below 5% only after adding preprocessing, domain fine-tuning, multi-vector retrieval, cross-encoder reranking and a rule-based validation layer ([InfoQ, Nov 2025](https://www.infoq.com/articles/reducing-false-positives-retrieval-augmented-generation/)). Both results are honest. They answer different questions, and only one of them composes with your cost model.

## What the published numbers measure

| Source | Headline | What was measured |
|---|---|---|
| GPT Semantic Cache, arXiv:2411.05276 | 68.8% fewer API calls, hit rate 61.6%–68.8%, positive hit rate >97% | 8,000 generated QA pairs over four broad categories, `all-MiniLM-L6-v2`, threshold 0.8, each hit adjudicated by GPT-4o Mini |
| vCache, arXiv:2502.03771, ICLR 2026 | 12.5× higher cache hit and 26× lower error than static thresholds | Per-cached-prompt thresholds learned online against a user-specified error bound |
| InfoQ banking case study | Up to 99% false positives at threshold 0.7; sub-5% only with a six-layer stack | 1,000 real banking queries, seven bi-encoders |
| Redis embedding study | Model choice moves the usable threshold | Labelled duplicate / non-duplicate question pairs, confusion matrices across five models |

Three metrics are in play and they are not interchangeable. **Hit rate** is the share of queries the cache answered. **Positive hit rate** is the share of those answers that were acceptable. **Served-answer precision** is the same idea restricted to the traffic you serve, and it is the only one you can put into a cost equation.

The vCache authors are blunt about why the fixed threshold persists anyway: static thresholds "do not give formal correctness guarantees, result in unexpected error rates, and lead to suboptimal cache hit rates" ([arXiv:2502.03771](https://arxiv.org/abs/2502.03771)). Pick one number for the whole cache and you have picked which kind of error to live with.

## A fixture with confusions in it

The question is not "does my cache hit" but "which wrong answers does it serve". That needs a fixture where wrong answers exist. I built one for a solar-home-system support desk with 12 indexed FAQ entries, then added two sets of test queries:

- **24 true paraphrases**: same question, different words, the cached answer is correct.
- **14 danger queries**: 11 near-misses on the same topic plus 3 personalisation queries. Their correct answers are *not* in the index, so any hit is a wrong answer.

| Test query | Nearest cached question | Cosine | Should serve? |
|---|---|---|---|
| how many years is the battery covered for | how long is the battery warranty | 0.8465 | yes |
| how much credit do I have left now | how do I see how much credit I have left | 0.9102 | no: different account |
| how do I pay with cash instead of mpesa | how do I pay my instalment with M-PESA | 0.8081 | no: not in the index |
| can I get a refund instead of a replacement | can I return the system if I change my mind | 0.7861 | no: not in the index |
| what is the return period | can I return the system if I change my mind | 0.7005 | yes |

Read the last two rows together. The query that *should* be served scores 0.7005, and the query that must not be served scores 0.7861. Any threshold between them is decided by a margin of 0.08 in the wrong direction. This is the ordinary case, not an edge case: paraphrase tolerance and topic drift pull in opposite directions on the same axis.

## What one global threshold does

Embeddings are `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token window, MTEB average 62.17 per the [model card](https://huggingface.co/BAAI/bge-small-en-v1.5)), retrieved through [fastembed](https://github.com/qdrant/fastembed). Rank one candidate per query, then sweep the threshold:

```python
# pip install fastembed numpy
import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

CANON = {
    "pay_mpesa": "how do I pay my instalment with M-PESA",
    "reset_pin": "how do I reset my device PIN",
    "change_pin": "how do I change my device PIN",
    "panel_warranty": "how long is the panel warranty",
    "battery_warranty": "how long is the battery warranty",
    "payment_not_posted": "my payment did not reflect on my account",
    "technician_visit": "how do I book a technician visit",
    "clean_panel": "how should I clean the solar panel",
    "final_payment_unlock": "what happens after I finish the final payment",
    "return_window": "can I return the system if I change my mind",
    "add_light": "how do I add another bulb to my system",
    "credit_check": "how do I see how much credit I have left",
}
PARAPHRASE = [
    ("pay_mpesa", "how can I pay through mpesa"),
    ("pay_mpesa", "where do I send the mpesa payment"),
    ("reset_pin", "how do I reset the pin on my device"),
    ("reset_pin", "I forgot my device pin, how do I set a new one"),
    ("change_pin", "how can I change the pin on my device"),
    ("change_pin", "I want a different device pin"),
    ("panel_warranty", "what is the warranty period for the panel"),
    ("panel_warranty", "how many years is the panel covered for"),
    ("battery_warranty", "what is the warranty period for the battery"),
    ("battery_warranty", "how many years is the battery covered for"),
    ("payment_not_posted", "I paid but it is not showing on my account"),
    ("payment_not_posted", "my payment is missing from my account"),
    ("technician_visit", "how do I get a technician to come"),
    ("technician_visit", "can I request a technician"),
    ("clean_panel", "what is the right way to clean the panel"),
    ("clean_panel", "how do I wash the solar panel"),
    ("final_payment_unlock", "what happens once I finish paying"),
    ("final_payment_unlock", "what happens when the last payment is done"),
    ("return_window", "can I send the system back if I change my mind"),
    ("return_window", "what is the return period"),
    ("add_light", "how do I connect one more bulb"),
    ("add_light", "can I add a light to the system"),
    ("credit_check", "how do I check how much credit is left"),
    ("credit_check", "where can I see my remaining credit"),
]
# the correct answer is not in the index for any of these: a hit is a wrong answer
DANGER = [
    ("pay_mpesa", "how do I pay with cash instead of mpesa"),
    ("reset_pin", "how do I factory reset the whole device"),
    ("change_pin", "how do I change the phone number on my account"),
    ("clean_panel", "how do I clean the battery terminals"),
    ("return_window", "can I get a refund instead of a replacement"),
    ("pay_mpesa", "how do I cancel my mpesa subscription"),
    ("credit_check", "how do I check my mpesa balance"),
    ("panel_warranty", "what is the warranty on the control unit"),
    ("add_light", "how do I add a second panel"),
    ("final_payment_unlock", "when does the device lock"),
    ("panel_warranty", "how much does a replacement panel cost"),
    ("credit_check", "how much credit do I have left now"),
    ("credit_check", "how much do I owe this month"),
    ("credit_check", "what is my current balance"),
]
keys = list(CANON)
para_texts, para_gold = [q for _, q in PARAPHRASE], [g for g, _ in PARAPHRASE]
danger_texts = [q for _, q in DANGER]
PI, PD, NI = len(para_texts), len(danger_texts), len(keys)
LAM = 8.0  # LLM-call equivalents charged for one wrong answer

enc = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
unit = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
E = unit(np.array(list(enc.embed([CANON[k] for k in keys]))))
Qp = unit(np.array(list(enc.embed(para_texts))))
Qd = unit(np.array(list(enc.embed(danger_texts))))
Sp, Sd = Qp @ E.T, Qd @ E.T
top_p, sim_p = Sp.argmax(1), Sp.max(1)
top_d, sim_d = Sd.argmax(1), Sd.max(1)
gold = np.array([keys.index(g) for g in para_gold])
print(f"index {NI} | paraphrase tests {PI} | danger tests {PD} | fixture mix {PI/(PI+PD):.0%}/{PD/(PI+PD):.0%}")


def count(t):
    ap = int(((sim_p >= t) & (top_p == gold)).sum())
    wp = int(((sim_p >= t) & (top_p != gold)).sum())
    return ap, wp, int((sim_d >= t).sum())


def calls(ap, wp, ad, lam=LAM):
    return (wp * lam + (PI - ap - wp) + ad * lam + (PD - ad)) / (PI + PD)


print("\n[2] one global cosine threshold")
print("theta  para_ok  para_wrong  danger_hit  accuracy  calls/1k  saved/1k")
for t in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]:
    ap, wp, ad = count(t)
    c = calls(ap, wp, ad)
    print(f"{t:.2f}   {ap:>6d}   {wp:>9d}    {ad:>8d}    {(ap+PD-ad)/(PI+PD):>5.1%}    {c*1000:>6.0f}    {(1-c)*1000:>6.0f}")

print("\n[3] per-intent score ranges (bge-small)")
print("intent               para_min  danger_max")
for i, k in enumerate(keys):
    print(f"{k:<20} {sim_p[gold == i].min():>8.4f}   {sim_d[top_d == i].max() if (top_d == i).any() else 0.0:>9.4f}")
lo_p = min(sim_p[gold == i].min() for i in range(NI))
hi_d = max(sim_d[top_d == i].max() for i in range(NI) if (top_d == i).any())
print(f"lowest correct-pair score {lo_p:.4f} | highest wrong-pair score {hi_d:.4f} | window empty: {hi_d > lo_p}")

print("\n[4] personalisation: same wording, different correct answer")
for k in range(PD - 3, PD):
    print(f"  {sim_d[k]:.4f}  {danger_texts[k]:<34} -> {keys[top_d[k]]}")

ce = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
ce_p = np.array([float(next(iter(ce.rerank(q, [CANON[keys[i]]])))) for q, i in zip(para_texts, top_p)])
ce_d = np.array([float(next(iter(ce.rerank(q, [CANON[keys[i]]])))) for q, i in zip(danger_texts, top_d)])
print("\n[5] ms-marco cross-encoder on the top-1 candidate (raw logit)")
print(f"  paraphrase  min {ce_p.min():6.2f}  median {np.median(ce_p):6.2f}  max {ce_p.max():6.2f}")
print(f"  danger      min {ce_d.min():6.2f}  median {np.median(ce_d):6.2f}  max {ce_d.max():6.2f}")
print("  tau   para_ok  danger_hit  accuracy  saved/1k")
for tau in [-4, -2, 0, 2, 3, 4]:
    ap = int(((sim_p >= 0.82) & (top_p == gold) & (ce_p >= tau)).sum())
    wp = int(((sim_p >= 0.82) & (top_p != gold) & (ce_p >= tau)).sum())
    ad = int(((sim_d >= 0.82) & (ce_d >= tau)).sum())
    c = calls(ap, wp, ad)
    print(f"  {tau:>4d}  {ap:>6d}    {ad:>8d}    {(ap+PD-ad)/(PI+PD):>5.1%}    {(1-c)*1000:>6.0f}")

th = np.array([(sim_p[gold == i].min() + (sim_d[top_d == i].max() if (top_d == i).any() else 0.0)) / 2 for i in range(NI)])
ap = sum(1 for j in range(PI) if sim_p[j] >= th[top_p[j]] and top_p[j] == gold[j])
wp = sum(1 for j in range(PI) if sim_p[j] >= th[top_p[j]] and top_p[j] != gold[j])
ad = sum(1 for j in range(PD) if sim_d[j] >= th[top_d[j]])
print("\n[6] per-intent thresholds fitted on the 38 labelled pairs")
print(f"  para_ok={ap}/{PI}  para_wrong={wp}  danger_hit={ad}/{PD}  "
      f"accuracy={(ap+PD-ad)/(PI+PD):.1%}  saved/1k={(1-calls(ap,wp,ad))*1000:.0f}")

print("\n[7] savings per 1,000 queries (negative = worse than no cache)")
print("theta  lambda=1 @10%danger  lambda=8 @10%danger  lambda=8 @37%danger")
for t in [0.70, 0.78, 0.82, 0.85, 0.90]:
    ap, wp, ad = count(t)
    def sav(mix, lam):
        pm = (wp * lam + (PI - ap - wp)) / PI
        dm = (ad * lam + (PD - ad)) / PD
        return (1 - (mix * pm + (1 - mix) * dm)) * 1000
    print(f"{t:.2f}   {sav(0.90, 1):>18.0f}   {sav(0.90, 8):>19.0f}   {sav(0.63, 8):>16.0f}")

print("\n[8] served-answer precision and the break-even penalty")
print("theta  served  wrong_served  precision  break_even_lambda")
for t in [0.70, 0.78, 0.80, 0.82, 0.85, 0.90, 0.92]:
    ap, wp, ad = count(t)
    served, wrong = ap + ad, ad
    p = wrong / served if served else 0.0
    be = f"{1/p:>11.1f}" if p else "       none"
    print(f"{t:.2f}   {served:>6d}  {wrong:>12d}    {1-p:>7.1%}    {be}")
```

```text
index 12 | paraphrase tests 24 | danger tests 14 | fixture mix 63%/37%

[2] one global cosine threshold
theta  para_ok  para_wrong  danger_hit  accuracy  calls/1k  saved/1k
0.70       24           0          12    68.4%      2579     -1579
0.75       23           0          10    71.1%      2237     -1237
0.78       23           0           7    78.9%      1684      -684
0.80       23           0           3    89.5%       947        53
0.82       22           0           2    89.5%       789       211
0.85       18           0           1    81.6%       711       289
0.88       15           0           1    73.7%       789       211
0.90       13           0           1    68.4%       842       158
0.92       11           0           0    65.8%       711       289
0.95        5           0           0    50.0%       868       132
0.98        1           0           0    39.5%       974        26

[3] per-intent score ranges (bge-small)
intent               para_min  danger_max
pay_mpesa              0.8037      0.8081
reset_pin              0.9275      0.7974
change_pin             0.9113      0.6832
panel_warranty         0.8436      0.8206
battery_warranty       0.8465      0.0000
payment_not_posted     0.8607      0.0000
technician_visit       0.8387      0.0000
clean_panel            0.8802      0.7902
final_payment_unlock   0.9272      0.0000
return_window          0.7005      0.7861
add_light              0.8502      0.7592
credit_check           0.9298      0.9102
lowest correct-pair score 0.7005 | highest wrong-pair score 0.9102 | window empty: True

[4] personalisation: same wording, different correct answer
  0.9102  how much credit do I have left now -> credit_check
  0.7293  how much do I owe this month       -> credit_check
  0.7612  what is my current balance         -> credit_check

[5] ms-marco cross-encoder on the top-1 candidate (raw logit)
  paraphrase  min -10.55  median  -0.26  max   6.79
  danger      min -11.35  median  -9.61  max   1.05
  tau   para_ok  danger_hit  accuracy  saved/1k
    -4      19           1    84.2%       316
    -2      15           1    73.7%       211
     0      12           1    65.8%       132
     2       7           0    55.3%       184
     3       6           0    52.6%       158
     4       4           0    47.4%       105

[6] per-intent thresholds fitted on the 38 labelled pairs
  para_ok=22/24  para_wrong=0  danger_hit=2/14  accuracy=89.5%  saved/1k=211

[7] savings per 1,000 queries (negative = worse than no cache)
theta  lambda=1 @10%danger  lambda=8 @10%danger  lambda=8 @37%danger
0.70                  900                   300              -1590
0.78                  862                   513               -691
0.82                  825                   725                208
0.85                  675                   625                288
0.90                  488                   438                156

[8] served-answer precision and the break-even penalty
theta  served  wrong_served  precision  break_even_lambda
0.70       36            12      66.7%            3.0
0.78       30             7      76.7%            4.3
0.80       26             3      88.5%            8.7
0.82       24             2      91.7%           12.0
0.85       19             1      94.7%           19.0
0.90       14             1      92.9%           14.0
0.92       11             0     100.0%           none
```

Two things in that table are worth pausing on. First, **no paraphrase was ever routed to a sibling entry**: `para_wrong` is zero at every threshold, including 0.70. The battery/panel pair that looks like the obvious confusion never confused the model once both answers were in the index. Every error came from a query whose answer was never there. Second, there is no row where both columns are acceptable: the thresholds that preserve recall (0.70–0.80) serve 3 to 12 wrong answers, and the thresholds with zero wrong answers (0.92+) keep 11 of 24 paraphrases.

## The window is empty

The per-intent table explains why no single number can work. Each intent has its own score range, and those ranges do not line up:

- `reset_pin` accepts paraphrases down to 0.9275 and its worst confusion sits at 0.7974, a comfortable gap.
- `pay_mpesa` accepts down to 0.8037 and its worst confusion scores 0.8081: the confusion scores *higher than the weakest real paraphrase*.
- `return_window` accepts down to 0.7005 while its worst confusion reaches 0.7861.

Globally, the lowest acceptable paraphrase score is 0.7005 and the highest unacceptable score is 0.9102. A threshold above 0.9102 rejects almost everything; below 0.7005 it serves confusions. **The interval between the two distributions is empty**, so the search for one defensible threshold has no solution, only trade-offs.

This is exactly the argument behind vCache: thresholds must be estimated per cached prompt, under a stated error bound, rather than set once for the store. Its reported comparison against static-threshold and fine-tuned-embedding baselines — up to 12.5× higher hit rate at 26× lower error — comes from doing that online without extra training.

You can approximate it offline. Fit one midpoint threshold per intent from labelled paraphrases and confusions and the same fixture yields 22/24 paraphrases served correctly, 2/14 wrong answers, and 89.5% accuracy, against 18/24 and 1/14 at a global 0.85. Per-intent calibration buys four paraphrases back for one extra wrong answer. It is a real gain, and it is not a fix: it assumes you already own the labelled confusions.

## Three queries no threshold can fix

The personalisation rows are the ones to stare at:

| Query | Cosine | Served at 0.85? |
|---|---|---|
| how much credit do I have left now | 0.9102 | yes: someone else's balance |
| what is my current balance | 0.7612 | no |
| how much do I owe this month | 0.7293 | no |

The first query is answered by a *correct* entry with the highest score in the fixture, and the answer is wrong for the person asking, because the answer depends on the account. There is no rephrasing that changes this: two customers can send byte-identical text and need different answers. No threshold, no reranker and no fine-tuned embedding separates them, because the inputs are identical.

For a Kenyan instalment or savings product, that is not a corner of the traffic. Balance, instalment and "did my payment go through" questions are the most repeated things users ask, which means the most cacheable-looking traffic is the traffic you must never cache. The ceiling on your savings is set by policy before it is set by the encoder.

## A cross-encoder is not an answer-equivalence check

The obvious upgrade is a second-stage verifier: take the top candidate and rescore the pair with a cross-encoder before serving. It is the fifth layer in the banking team's six-layer stack, and it is what most tutorials recommend. Measured on this fixture with `Xenova/ms-marco-MiniLM-L-6-v2`, it does not work as a safety gate:

- Paraphrase logits: median −0.26, minimum −10.55. True paraphrases with little lexical overlap ("I forgot my device pin, how do I set a new one") are scored as irrelevant.
- Danger logits: median −9.61, but maximum +1.05. The highest-scoring danger query is *"how do I pay with cash instead of mpesa"*, which is the worst confusion in the set.

At a gate of τ = 0, recall falls from 22/24 to 12/24 while one wrong answer still passes. Tightening to τ = 2 does remove the wrong answer, and costs five more paraphrases. A reranker trained on query-passage relevance learns "is this document about the question", which is not the question a cache needs answered: "would this stored answer be correct for this new query". Getting the latter took the banking team domain fine-tuning plus a rule layer, not an off-the-shelf checkpoint.

## Normalise, or the threshold means nothing

One silent failure sits below all of this. A cosine threshold is only comparable to another cosine threshold if both sides are unit vectors. `fastembed` returns normalised vectors, so the fixture above is fine, but plenty of stores hand back raw encoder output and plenty of gateway code compares with a dot product:

```python
import numpy as np
from fastembed import TextEmbedding

enc = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qs = ["how can I pay through mpesa",
      "how do I pay my instalment with M-PESA",
      "how do I pay with cash instead of mpesa"]
V = np.array(list(enc.embed(qs)))
print("norms as returned:", [round(float(np.linalg.norm(v)), 4) for v in V])

# a store that skips normalisation hands back vectors of different lengths
S = V * np.array([[2.5], [1.0], [2.5]])
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print("pair                cosine   dot    cosine@0.85  dot@0.85")
for name, a, b in [("correct paraphrase", S[0], S[1]), ("wrong answer", S[2], S[1])]:
    print(f"{name:<18} {cos(a, b):>8.4f} {float(a @ b):>7.4f}   "
          f"{'serve' if cos(a, b) >= 0.85 else 'miss':<9}   {'serve' if a @ b >= 0.85 else 'miss'}")
```

```text
norms as returned: [1.0, 1.0, 1.0]
pair                cosine   dot    cosine@0.85  dot@0.85
correct paraphrase   0.8340  2.0851   miss        serve
wrong answer         0.8081  2.0202   miss        serve
```

Multiply two vectors by 2.5 and their dot product grows by 6.25 while their cosine does not move. A 0.85 dot-product gate then serves both the paraphrase and the confusion, and nothing in the logs distinguishes that from a well-tuned cache. Normalise on write, assert unit norm on read, and store the metric alongside the vectors.

## The threshold you can defend

Once you accept that precision is the only number that composes with cost, the arithmetic is short. A served query costs the full generation if the answer is wrong and nothing if it is right. If `p` is the share of served answers that are wrong and `λ` is what one wrong answer costs in call-equivalents, expected cost per served query is `pλ` and the saving is `1 − pλ`. The cache pays only while:

$$p < \frac{1}{\lambda}$$

At λ = 8 (a retry plus a human picking up the thread), served-answer precision must beat 88.5%. Table [8] applies that to the fixture: 0.70 needs λ under 3 to break even; 0.80 sits at exactly 8.7; 0.82 tolerates 12; 0.85 tolerates 19.

Pricing λ is the step teams skip. Take one wrong answer and count what it consumes: the retry generation, the agent minutes on the follow-up ticket, and the share of users who abandon the thread instead of asking a third time. A single escalation is worth several generations even in a cheap desk, which is why the break-even penalty is usually well above the numbers people assume when they set 0.75 and move on.

| Angle | 0.70 | 0.80 | 0.82 | 0.85 | 0.92 |
|---|---|---|---|---|---|
| Served answers | 36 | 26 | 24 | 19 | 11 |
| Wrong among served | 12 | 3 | 2 | 1 | 0 |
| Precision | 66.7% | 88.5% | 91.7% | 94.7% | 100% |
| Break-even λ | 3.0 | 8.7 | 12.0 | 19.0 | none |

Savings per 1,000 queries depend on both numbers, which is why the same threshold can look brilliant in one deployment and catastrophic in another:

| Threshold | λ = 1, 10% danger | λ = 8, 10% danger | λ = 8, 37% danger |
|---|---|---|---|
| 0.70 | 900 | 300 | −1,590 |
| 0.78 | 862 | 513 | −691 |
| 0.82 | 825 | 725 | 208 |
| 0.85 | 675 | 625 | 288 |
| 0.90 | 488 | 438 | 156 |

At λ = 1, where the user just asks again, 0.70 saves 900 calls per 1,000 queries and looks like the obvious choice. At λ = 8 the same threshold saves 300 when confusions are rare and loses 1,590 when they are not. The threshold did not change; the cost of being wrong did.

## Five gates before the cache goes live

1. **Label confusions, not paraphrases.** Twenty to forty near-miss pairs per intent, drawn from real logs, are the asset. Paraphrases prove the cache works; confusions prove it is safe. Keep both in CI next to the prompt tests.
2. **Price a wrong answer first.** Express it in call-equivalents (retry + escalation + the support ticket), then compute `1/λ`. That number is your precision floor, and it is the only budget your threshold has to meet.
3. **Calibrate per intent.** Fit a threshold from each intent's own paraphrase and confusion scores, or use vCache's online per-prompt estimation with an explicit error bound. A single store-wide number will be too tight for `return_window` and too loose for `pay_mpesa`.
4. **Exclude by policy, not by score.** Account-scoped intents (balance, amount due, payment status, anything carrying a phone number, account ID or date) never go in the cache. No similarity function can see the difference between two identical strings from two customers.
5. **Log the served answer.** Store the matched key, the score and the returned text, and not merely the hit counter. Sample-audit weekly and attach a TTL so stale answers expire. A cache whose audits only count hits cannot tell you its precision, and precision is the number you are defending.

## Key takeaways

| Takeaway | Evidence |
|---|---|
| Hit rate is a saving; precision is a risk | 24/24 paraphrases served at 0.70, with 12 of 14 confusions also served |
| One global threshold has no solution here | Lowest correct pair 0.7005, highest wrong pair 0.9102: empty interval |
| Per-intent calibration buys recall, not safety | 22/24 correct with 2 wrong, versus 18/24 with 1 wrong at a global 0.85 |
| Some intents must not be cached at all | Identical text, different account, cosine 0.9102: no gate can separate them |
| A generic reranker is the wrong verifier | True-paraphrase median −0.26; the worst confusion scores +1.05 |
| The threshold is a function of λ | 0.70 saves 900/1k at λ = 1 and loses 1,590/1k at λ = 8 with heavy confusions |

## References

- Regmi & Pun, *GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching* — [arXiv:2411.05276](https://arxiv.org/abs/2411.05276)
- Schroeder et al., *vCache: Verified Semantic Prompt Caching* (ICLR 2026) — [arXiv:2502.03771](https://arxiv.org/abs/2502.03771)
- Daivam, *Reducing False Positives in RAG Semantic Caching: a Banking Case Study* — [InfoQ](https://www.infoq.com/articles/reducing-false-positives-retrieval-augmented-generation/)
- Shelton, *What's the best embedding model for semantic caching?* — [Redis](https://redis.io/blog/whats-the-best-embedding-model-for-semantic-caching/)
- *BAAI/bge-small-en-v1.5* model card — [Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5)
- *fastembed* — [GitHub](https://github.com/qdrant/fastembed)

## Related posts

- [Your WHERE Clause Ate the Recall: Sizing the Candidate Pool Behind Filtered Vector Search](/posts/filtered-vector-search-pool-sizing/) — the same ranking problem one layer down, where the filter removes the answer
- [The Denominator Is the Metric: Auditing a RAG Retriever Before You Blame the Model](/posts/rag-recall-at-k-denominator/) — why a retrieval score needs its denominator printed beside it
- [Prompt Order Is a Cost Control: Measuring Prefix-Cache Reuse in Standard-Library Python](/posts/prompt-prefix-cache-order-cost/) — the other kind of cache, where reuse saves computation instead of answers
- [Your Model's 0.9 Is Not a 90% Chance: Calibrating Risk Scores Before You Set the Threshold](/posts/probability-calibration-risk-scores/) — threshold selection when the score is a probability rather than a similarity
