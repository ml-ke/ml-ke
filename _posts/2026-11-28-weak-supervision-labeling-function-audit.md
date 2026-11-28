---
title: "Three Rules and a Ballot Box: Auditing Labeling Functions Before You Train"
date: 2026-11-28 00:00:00 +0300
categories: [Machine Learning, AI Engineering]
tags: [weak supervision, programmatic labeling, data quality, annotation, evaluation, MLOps]
math: true
image:
  path: /assets/img/cover-weak-supervision-labeling-function-audit.webp
  alt: "Three keyword rule cards dropping labeled chips into a ballot box, two chips joined by a dashed link for a shared token, and a gold-sample scale reading 0.84 counted against 0.81 weighted"
---

> **The short version**
> A programmatic labeling project produces a label matrix, not a training set. Three numbers decide whether that matrix is worth training on: how much of the corpus the rules cover, how accurate each rule is, and whether the rules fail independently. This post builds a 1,200-ticket fixture, audits eight keyword rules against it, and measures the failure that catches most teams, where two rules sharing one ambiguous token make their agreement look like evidence. Every figure below comes from code you can run in twenty minutes with the standard library alone.

Weak supervision has a strong public case. Snorkel DryBell, deployed at Google, trained classifiers of comparable quality to ones built from tens of thousands of hand-labeled examples, and its authors report a 52% average F1 gain from wiring organisational sources that could not previously be served to a model at all ([arXiv:1812.00417](https://arxiv.org/abs/1812.00417)). A user study in the Snorkel paper found subject-matter experts building models 2.8 times faster, with 45.5% better predictive performance, than seven hours of hand labeling ([arXiv:1711.10160](https://arxiv.org/abs/1711.10160)).

The same literature is blunt about what the method hands back. A 2023 critical review ran weak-supervision methods across eight benchmark datasets and found that without clean validation samples, every method considered failed to work, performing similarly to or worse than the weak labels themselves ([arXiv:2305.17442](https://arxiv.org/abs/2305.17442)). The rules are not the deliverable. The deliverable is a matrix of votes whose quality you have measured.

## What the rules actually produce

A set of labeling functions applied to $m$ unlabeled points produces a label matrix $\Lambda \in (\mathcal{Y} \cup \{\varnothing\})^{m \times n}$: one row per example, one column per rule, and an abstain symbol wherever a rule declines to answer. Snorkel's framing of that object is the one worth keeping in your head, because every later decision is a decision about a column or a cell in it ([arXiv:1711.10160](https://arxiv.org/abs/1711.10160)).

The standard pipeline then has three stages: labeling functions, a label model that turns votes into probabilistic labels, and an end model trained on those labels ([arXiv:2202.05433](https://arxiv.org/abs/2202.05433)). Teams routinely skip straight from stage one to stage three, because a majority vote is one line of code. What they skip is the audit.

Three questions cover most of that audit:

- **Coverage:** what share of rows does each rule touch, and what share does the vote decide at all?
- **Accuracy:** on the rows a rule votes, how often is it right, and how wide is the interval around that estimate?
- **Independence:** when two rules fail, do they fail together?

Two neighbouring posts answered different questions on the same objects. [The label-budget post](/posts/annotation-budget-language-classifier/) asked which rows are worth paying for when humans do the labeling. [The annotator-agreement post](/posts/annotator-agreement-before-you-train/) asked whether two humans agree before you trust a label scheme. This one asks whether the machine rules agree with the truth, and what agreement among rules is actually worth.

## The fixture, and why it is deliberately plain

The corpus is 1,200 synthetic support tickets from a payments help desk, 40% of them refund requests and the rest billing, login, delivery and other queries collapsed into a single negative class. Each ticket is a bag of tokens drawn from a class-specific distribution. Nothing exotic: no model-based rules, no distant supervision from a knowledge base, no crowd labels.

The interesting parameter is the overlap. Refund tickets mention the app, invoices and delivery delays; non-refund tickets say "refund" 9% of the time, "money back" 7%, and "return" 28%. Two of the eight rules key on "return":

- **L2** votes refund when the ticket says "money back" or "return".
- **L3** votes refund when it says "return" or "send it back".

That shared token is the whole point of the exercise. Keyword and regex rules are the canonical text labeling function class in the survey, and duplicate keywords across rule authors are the ordinary way correlation enters a project ([arXiv:2202.05433](https://arxiv.org/abs/2202.05433)).

## Coverage first, accuracy second

Run the audit table before you run a single training job.

```python
import random

REF = [("refund", .55), ("money back", .38), ("return", .45), ("credit", .30),
       ("send it back", .12), ("charged twice", .14), ("invoice", .10),
       ("password", .08), ("delivery", .12), ("late", .10), ("app", .28), ("order", .40)]
OTH = [("invoice", .30), ("password", .32), ("otp", .28), ("delivery", .42),
       ("late", .30), ("charged twice", .18), ("return", .28), ("credit", .18),
       ("order", .35), ("refund", .09), ("money back", .07), ("app", .30)]


def fixture(n=1200, seed=7):
    rng = random.Random(seed)
    rows, truth = [], []
    for _ in range(n):
        r = rng.random() < .40                      # 40% of tickets are refund requests
        rows.append(" ".join(t for t, p in (REF if r else OTH) if rng.random() < p) or "hello")
        truth.append(1 if r else -1)
    return rows, truth


LFS = {"L1_refund_word": "refund",
       "L2_money_back": "money back|return",
       "L3_return": "return|send it back",
       "L4_double_charge": "charged twice",
       "L5_invoice": "invoice|statement",
       "L6_login": "password|otp",
       "L7_app_error": "app|error",
       "L8_logistics": "delivery|late"}
NEG = {"L5_invoice", "L6_login", "L7_app_error", "L8_logistics"}

rows, truth = fixture()
L = [[(-1 if k in NEG else 1) if any(t in s for t in pat.split("|")) else 0
      for k, pat in LFS.items()] for s in rows]

print(f"rows {len(rows)}   refund share {sum(t == 1 for t in truth) / len(truth):.3f}")
print("LF                 votes  coverage  accuracy  abstain")
for j, k in enumerate(LFS):
    idx = [i for i in range(len(rows)) if L[i][j] != 0]
    a = sum(1 for i in idx if L[i][j] == truth[i]) / len(idx)
    print(f"{k:18s}{len(idx):6d}{len(idx)/len(rows):10.3f}{a:10.3f}{len(rows)-len(idx):9d}")

mv = []
for i in range(len(rows)):
    cast = [v for v in L[i] if v != 0]
    mv.append((1 if sum(cast) > 0 else -1) if len(cast) >= 2 and sum(cast) != 0 else 0)
cov = [i for i in range(len(rows)) if mv[i] != 0]
acc = sum(1 for i in cov if mv[i] == truth[i]) / len(cov)
print(f"\nmajority vote covers {len(cov)} rows ({len(cov)/len(rows):.3f}), "
      f"undecided {len(rows)-len(cov)}")
print(f"accuracy on covered rows {acc:.3f}")
print(f"always-say-refund baseline "
      f"{max(sum(t == 1 for t in truth), sum(t == -1 for t in truth))/len(truth):.3f}")
```

```text
rows 1200   refund share 0.398
LF                 votes  coverage  accuracy  abstain
L1_refund_word       325     0.271     0.831      875
L2_money_back        528     0.440     0.581      672
L3_return            441     0.367     0.578      759
L4_double_charge     183     0.152     0.317     1017
L5_invoice           267     0.223     0.787      933
L6_login             417     0.347     0.890      783
L7_app_error         362     0.302     0.591      838
L8_logistics         510     0.425     0.818      690

majority vote covers 731 rows (0.609), undecided 469
accuracy on covered rows 0.833
always-say-refund baseline 0.602
```

Four things in that table are worth pausing on.

The two highest-coverage rules, L2 and L3 at 0.440 and 0.367 coverage, sit at 0.581 and 0.578 accuracy. They are barely better than a coin flip, and they cover 46.3% of the corpus between them once their overlap is collapsed.

L4 is worse than chance at 0.317. It votes refund on "charged twice", which appears in 18% of non-refund tickets and only 14% of refund tickets, so most of its votes are wrong. A rule that is reliably wrong is not rubbish: invert it and it becomes a rule that is right 68% of the time. What it must never do is keep its current weight.

The majority vote covers 61% of the corpus at 0.833 accuracy, against 0.602 for guessing the majority class every time. On the rows it decides, the matrix is worth roughly 23 points. That qualifier matters: coverage and accuracy trade against each other, and every headline accuracy figure below is computed on the rows a method actually labeled.

The undecided 469 rows are not errors, they are work. They are where a second rule, or a human, adds value, which is exactly the budget question the [label-budget post](/posts/annotation-budget-language-classifier/) prices.

## Agreement is not confirmation

Now the part that actually bites. Group the corpus by how many rules voted, and score the vote in each group.

```python
import random

REF = [("refund", .55), ("money back", .38), ("return", .45), ("credit", .30),
       ("send it back", .12), ("charged twice", .14), ("invoice", .10),
       ("password", .08), ("delivery", .12), ("late", .10), ("app", .28), ("order", .40)]
OTH = [("invoice", .30), ("password", .32), ("otp", .28), ("delivery", .42),
       ("late", .30), ("charged twice", .18), ("return", .28), ("credit", .18),
       ("order", .35), ("refund", .09), ("money back", .07), ("app", .30)]


def fixture(n=1200, seed=7):
    rng = random.Random(seed)
    rows, truth = [], []
    for _ in range(n):
        r = rng.random() < .40
        rows.append(" ".join(t for t, p in (REF if r else OTH) if rng.random() < p) or "hello")
        truth.append(1 if r else -1)
    return rows, truth


LFS = {"L1_refund_word": "refund", "L2_money_back": "money back|return",
       "L3_return": "return|send it back", "L4_double_charge": "charged twice",
       "L5_invoice": "invoice|statement", "L6_login": "password|otp",
       "L7_app_error": "app|error", "L8_logistics": "delivery|late"}
NEG = {"L5_invoice", "L6_login", "L7_app_error", "L8_logistics"}

rows, truth = fixture()
L = [[(-1 if k in NEG else 1) if any(t in s for t in p.split("|")) else 0
      for k, p in LFS.items()] for s in rows]


def mv(i):
    cast = [v for v in L[i] if v != 0]
    if len(cast) < 2 or sum(cast) == 0:
        return 0
    return 1 if sum(cast) > 0 else -1


print("voters  rows  decisive  accuracy")
for k in range(1, 8):
    grp = [i for i in range(len(rows)) if sum(1 for v in L[i] if v != 0) == k]
    dec = [i for i in grp if mv(i) != 0]
    if dec:
        a = sum(1 for i in dec if mv(i) == truth[i]) / len(dec)
        print(f"{k:6d}{len(grp):6d}{len(dec):10d}  {a:.3f}")

un = [i for i in range(len(rows)) if len({v for v in L[i] if v != 0}) == 1
      and sum(1 for v in L[i] if v != 0) >= 2]
sp = [i for i in range(len(rows)) if len({v for v in L[i] if v != 0}) == 2 and mv(i) != 0]
print(f"\nunanimous  {len(un):4d} rows  accuracy {sum(1 for i in un if mv(i)==truth[i])/len(un):.3f}")
print(f"conflicted {len(sp):4d} rows  accuracy {sum(1 for i in sp if mv(i)==truth[i])/len(sp):.3f}")

print("\npair       both vote  joint error  if independent  ratio")
for a, b in [(1, 2), (0, 3), (4, 7), (5, 6)]:
    ia = [i for i in range(len(rows)) if L[i][a] != 0]
    ib = [i for i in range(len(rows)) if L[i][b] != 0]
    ea = 1 - sum(1 for i in ia if L[i][a] == truth[i]) / len(ia)
    eb = 1 - sum(1 for i in ib if L[i][b] == truth[i]) / len(ib)
    both = [i for i in range(len(rows)) if L[i][a] != 0 and L[i][b] != 0]
    j = sum(1 for i in both if L[i][a] != truth[i] and L[i][b] != truth[i]) / len(both)
    print(f"L{a+1}-L{b+1}    {len(both):8d}  {j:11.3f}  {ea*eb:14.3f}  {j/(ea*eb):.2f}x")
```

```text
voters  rows  decisive  accuracy
     2   336       244  0.873
     3   295       295  0.793
     4   203       111  0.865
     5    70        70  0.843
     6    14         8  0.625
     7     3         3  0.667

unanimous   389 rows  accuracy 0.910
conflicted  342 rows  accuracy 0.746

pair       both vote  joint error  if independent  ratio
L2-L3         414        0.449           0.177  2.54x
L1-L4          44        0.227           0.116  1.97x
L5-L8         134        0.067           0.039  1.73x
L6-L7         119        0.109           0.045  2.42x
```

Accuracy does not climb with the number of voters. It falls, then wobbles: 0.873 with two voters, 0.793 with three, 0.865 with four, 0.625 with six. Two things explain that. A decisive two-voter row is unanimous by construction, since two voters who disagree produce a tie and drop out, so the first line is really a unanimity statistic. And adding voters adds the near-chance rules, L2 and L3 among them, so the fifth and sixth votes arrive carrying less information than the first two.

That is the practical form of a well-known trap. A widely used industry glossary states it plainly: when two functions are wrong for the same reason, their agreement looks like confirmation and is not ([Arize, weak supervision](https://arize.com/glossary/weak-supervision/)).

The joint-error table measures it. L2 and L3 both fire on 414 tickets and are both wrong on 44.9% of them. If their failures were independent, you would expect the product of their marginal error rates, 17.7%. The observed joint failure is 2.54 times higher, and the reason is a single shared token: every non-refund ticket that says "return" trips both rules at once. L1 and L4 show a second route to the same place at 1.97x. They share no pattern at all, but both vote refund, so both are wrong on the same refund-free tickets; a shared class prior correlates errors just as effectively as a shared keyword.

The consensus statistic still has a use. Unanimous rows come out at 0.910 against 0.746 for conflicted ones, a 16-point spread. That is a real signal and a cheap production filter: report accuracy per stratum rather than as one number, because a single global accuracy figure hides both ends of this distribution.

## What a gold sample buys, and when it does not

A few hundred hand labels turns every rule into a measured quantity. Here are 120 gold tickets, 10% of the corpus, with Wilson intervals on each rule's accuracy and a shrunk log-odds weight for the vote.

```python
import random, math

REF = [("refund", .55), ("money back", .38), ("return", .45), ("credit", .30),
       ("send it back", .12), ("charged twice", .14), ("invoice", .10),
       ("password", .08), ("delivery", .12), ("late", .10), ("app", .28), ("order", .40)]
OTH = [("invoice", .30), ("password", .32), ("otp", .28), ("delivery", .42),
       ("late", .30), ("charged twice", .18), ("return", .28), ("credit", .18),
       ("order", .35), ("refund", .09), ("money back", .07), ("app", .30)]


def fixture(n=1200, seed=7):
    rng = random.Random(seed)
    rows, truth = [], []
    for _ in range(n):
        r = rng.random() < .40
        rows.append(" ".join(t for t, p in (REF if r else OTH) if rng.random() < p) or "hello")
        truth.append(1 if r else -1)
    return rows, truth


LFS = {"L1_refund_word": "refund", "L2_money_back": "money back|return",
       "L3_return": "return|send it back", "L4_double_charge": "charged twice",
       "L5_invoice": "invoice|statement", "L6_login": "password|otp",
       "L7_app_error": "app|error", "L8_logistics": "delivery|late"}
NEG = {"L5_invoice", "L6_login", "L7_app_error", "L8_logistics"}
names = list(LFS)
rows, truth = fixture()
L = [[(-1 if k in NEG else 1) if any(t in s for t in p.split("|")) else 0
      for k, p in LFS.items()] for s in rows]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def run(size):
    gold = random.Random(11).sample(range(len(rows)), size)
    W = []
    for j in range(len(LFS)):
        idx = [i for i in gold if L[i][j] != 0]
        k = sum(1 for i in idx if L[i][j] == truth[i])
        shr = (k + 2) / (len(idx) + 4)          # shrink the estimate toward 0.5
        W.append(math.log(shr / (1 - shr)))
        if size == 120:
            lo, hi = wilson(k, len(idx))
            print(f"{names[j]:18s}{len(idx):5d}{k/len(idx):9.3f}  [{lo:.2f},{hi:.2f}]{W[-1]:+8.3f}")

    def vote(i, mode):
        cast = [(v, j) for j, v in enumerate(L[i]) if v != 0 and not (mode == "drop" and j == 3)]
        if len(cast) < 2:
            return 0
        s = sum(v * (W[j] if mode == "weight" else 1.0) for v, j in cast)
        return 1 if s > 0 else (-1 if s < 0 else 0)

    out = {m: [vote(i, m) for i in range(len(rows))] for m in ("equal", "weight", "drop")}
    common = [i for i in range(len(rows)) if all(out[m][i] != 0 for m in out)]
    line = []
    for m in out:
        cov = sum(1 for x in out[m] if x != 0)
        a = sum(1 for i in common if out[m][i] == truth[i]) / len(common)
        line.append(f"{m} cov {cov:4d} acc {a:.3f}")
    print(f"gold {size:4d}  common {len(common):4d}  " + "  ".join(line))


print("LF                 gold   acc   95% CI     weight")
run(120)
run(600)
```

```text
LF                 gold   acc   95% CI     weight
L1_refund_word       39    0.795  [0.64,0.89]  +1.194
L2_money_back        56    0.554  [0.42,0.68]  +0.201
L3_return            45    0.511  [0.37,0.65]  +0.041
L4_double_charge     25    0.360  [0.20,0.55]  -0.492
L5_invoice           27    0.852  [0.68,0.94]  +1.427
L6_login             38    0.895  [0.76,0.96]  +1.792
L7_app_error         35    0.657  [0.49,0.79]  +0.580
L8_logistics         53    0.774  [0.64,0.87]  +1.122
gold  120  common  702  equal cov  731 acc 0.842  weight cov  921 acc 0.812  drop cov  713 acc 0.842
gold  600  common  702  equal cov  731 acc 0.842  weight cov  921 acc 0.849  drop cov  713 acc 0.842
```

The intervals are the story. L4 covers 15% of the corpus, so 120 gold tickets give it 25 votes and a 95% interval of 0.20 to 0.55. That interval cannot separate "actively harmful" from "mildly useful", which is exactly the decision you are trying to make. L6, at 0.347 coverage, gets 38 votes and an interval of 0.76 to 0.96, tight enough to set a weight and move on. Low-coverage rules need more gold, not less, and the corpus-wide gold budget hides that.

The weighting result is the surprise. Evaluated on the 702 rows every scheme decides, the plain count scores 0.842 and the weighted vote scores 0.812 with 120 gold tickets. Counting wins. At 600 gold tickets the weighted vote reaches 0.849 and finally edges ahead.

This is not an argument against modeling accuracies, it is an argument about sample size. Snorkel's authors built an optimizer for exactly this fork, predicting when generative modeling beats a majority vote to within 2.16 accuracy points on average, because the modeling is not free and does not always pay ([arXiv:1711.10160](https://arxiv.org/abs/1711.10160)). With 25 to 56 gold observations per rule, the weights are mostly noise and the count is a better estimator. With 600, the weights have converged enough to help.

One evaluation detail matters as much as the result. The weighted vote also labels 190 more rows, because non-uniform weights break ties that a count leaves at zero. Comparing 0.812 against 0.842 across different covered sets would have been meaningless, so both were scored on the same 702 rows. Any time a weighting scheme changes your coverage, recompute on the intersection before you believe the difference.

## Estimating accuracy with no labels at all

There is a family of methods that recovers rule accuracies from agreement rates alone, no gold data. The cheapest is a closed form over triplets of conditionally independent sources ([arXiv:2002.11955](https://arxiv.org/abs/2002.11955)).

Encode each vote as $\pm 1$ and define scaled accuracy $a_i = \mathbb{E}[v_i Y] = 2p_i - 1$. If three sources fail independently given the truth, then $\mathbb{E}[v_i v_j] = a_i a_j$, and the system of three pairwise moments solves directly:

$$a_1 = \sqrt{\frac{\mathbb{E}[v_1 v_2]\,\mathbb{E}[v_1 v_3]}{\mathbb{E}[v_2 v_3]}}$$

and the same permutation for $a_2$ and $a_3$. Note what the method requires: sources that vote on everything. The keyword rules above abstain on most of the corpus, so this block switches to four rules that answer every ticket.

```python
import random, math

REF = [("refund", .55), ("money back", .38), ("return", .45), ("credit", .30),
       ("send it back", .12), ("charged twice", .14), ("invoice", .10),
       ("password", .08), ("delivery", .12), ("late", .10), ("app", .28), ("order", .40)]
OTH = [("invoice", .30), ("password", .32), ("otp", .28), ("delivery", .42),
       ("late", .30), ("charged twice", .18), ("return", .28), ("credit", .18),
       ("order", .35), ("refund", .09), ("money back", .07), ("app", .30)]


def fixture(n=1200, seed=7):
    rng = random.Random(seed)
    rows, truth = [], []
    for _ in range(n):
        r = rng.random() < .40
        rows.append(" ".join(t for t, p in (REF if r else OTH) if rng.random() < p) or "hello")
        truth.append(1 if r else -1)
    return rows, truth


rows, truth = fixture()
SRC = {"A_refund": lambda s: 1 if "refund" in s else -1,
       "B_money_or_return": lambda s: 1 if ("money back" in s or "return" in s) else -1,
       "C_paperwork": lambda s: -1 if ("invoice" in s or "password" in s or "otp" in s) else 1,
       "B2_return_only": lambda s: 1 if "return" in s else -1}
V = {k: [f(s) for s in rows] for k, f in SRC.items()}
n = len(rows)
a = {k: sum(V[k][i] * truth[i] for i in range(n)) / n for k in V}
E = lambda x, y: sum(V[x][i] * V[y][i] for i in range(n)) / n

print("source            E[vY]  raw acc  (v always votes, so no abstain correction)")
for k in V:
    print(f"{k:18s}{a[k]:+6.3f}{(1 + a[k]) / 2:8.3f}")

for trio in [("A_refund", "B_money_or_return", "C_paperwork"),
             ("A_refund", "B_money_or_return", "B2_return_only")]:
    x, y, z = trio
    exy, exz, eyz = E(x, y), E(x, z), E(y, z)
    est = {x: math.sqrt(abs(exy * exz / eyz)),
           y: math.sqrt(abs(exy * eyz / exz)),
           z: math.sqrt(abs(exz * eyz / exy))}
    print(f"\n{x} + {y} + {z}")
    print(f"  E[xy] {exy:+.3f}   E[xz] {exz:+.3f}   E[yz] {eyz:+.3f}")
    print("  source            estimated  true   gap")
    for k in trio:
        print(f"  {k:18s}{est[k]:9.3f}{a[k]:+7.3f}{est[k]-a[k]:+7.3f}")
    print(f"  check E[yz] against a_y*a_z: {eyz/(a[y]*a[z]):.2f}x")
```

```text
source            E[vY]  raw acc  (v always votes, so no abstain correction)
A_refund          +0.562   0.781
B_money_or_return +0.347   0.673
C_paperwork       +0.430   0.715
B2_return_only    +0.248   0.624

A_refund + B_money_or_return + C_paperwork
  E[xy] +0.208   E[xz] +0.205   E[yz] +0.140
  source            estimated  true   gap
  A_refund              0.552 +0.562 -0.009
  B_money_or_return     0.377 +0.347 +0.031
  C_paperwork           0.371 +0.430 -0.059
  check E[yz] against a_y*a_z: 0.94x

A_refund + B_money_or_return + B2_return_only
  E[xy] +0.208   E[xz] +0.247   E[yz] +0.785
  source            estimated  true   gap
  A_refund              0.256 +0.562 -0.306
  B_money_or_return     0.814 +0.347 +0.468
  B2_return_only        0.964 +0.248 +0.716
  check E[yz] against a_y*a_z: 9.12x
```

On a triplet that roughly satisfies the independence assumption, the closed form lands within 0.06 of the truth on all three sources, with no labels used. That is the result the method is sold on, and it is real.

Now put the correlated pair in. B and B2 both key on "return", so $\mathbb{E}[v_B v_{B2}]$ comes out at 0.785, against 0.086 if they failed independently. The estimate for B2, a rule whose true scaled accuracy is 0.248, becomes 0.964. The worst source in the set is handed the highest weight, and the good source A is downgraded from 0.562 to 0.256.

The diagnostic is one line and it uses numbers you already have: divide the observed pairwise moment by the product of the two accuracies. A ratio near 1.0 is consistent with independence. A ratio of 9.12 is a shared keyword, and no closed form repairs that, only a dependency structure that models it. The paper's own contribution is exactly that extension, triplets over a dependency graph rather than a flat set ([arXiv:2002.11955](https://arxiv.org/abs/2002.11955)).

## How to run this on your own corpus

The procedure is short enough to put in a pull request template.

1. **Version the rules.** Each labeling function gets a name, a pattern, and a hash. A rule edited without a version bump silently invalidates every accuracy you measured for it.
2. **Publish the coverage and accuracy table.** Two numbers per rule, on a held-out slice. Rules under 0.05 coverage are usually not worth their maintenance.
3. **Stratify by number of voters.** One global accuracy number will hide the strata that matter, as the 0.910 against 0.746 split shows.
4. **Buy 100 to 600 gold rows, stratified.** Report Wilson intervals alongside point estimates. A rule with fewer than 30 gold votes has no usable precision estimate.
5. **Check the pairs, not just the columns.** Any two rules sharing a token, a lexicon or a knowledge-base entity get their pairwise moment divided by the product of their accuracies. A ratio above roughly 1.5 means the label model needs a dependency term.
6. **Then decide between counting, weighting and dropping.** The count is the baseline to beat, and at small gold budgets it often wins.

The one rule that is not negotiable: keep clean validation data. The 2023 review across eight benchmarks is unambiguous that without it the whole family fails to beat its own weak labels, and that is the failure mode a label-free estimator cannot detect on its own ([arXiv:2305.17442](https://arxiv.org/abs/2305.17442)). The audit is cheap because the gold set doubles as the tool that proves the audit correct.

Two operations help when the audit passes. Bucketing rows by their combination of votes is a supported analysis primitive in Snorkel itself, exposed as `get_label_buckets`, which returns data-point indices grouped by label combination ([Snorkel analysis package](https://snorkel.readthedocs.io/en/latest/packages/analysis.html)). And once rules are versioned and measured, the matrix becomes reproducible: the same eight rules over the same 1,200 rows produce identical accuracy figures on every run, which is what makes it safe to retrain the end model on them.

## Key takeaways

| Question | What to measure | Threshold from this fixture |
|---|---|---|
| Does the matrix cover the corpus? | Share of rows with at least two agreeing votes | 61% coverage, 83% accuracy on covered rows |
| Are individual rules usable? | Accuracy per rule with a Wilson interval | L4 at 0.317 is worse than chance and must be inverted or dropped |
| Is agreement evidence? | Unanimous versus conflicted accuracy | 0.910 against 0.746, a 16-point spread |
| Do the rules fail together? | Joint error over the product of marginal errors | 2.54x for the pair sharing "return" |
| Is a gold sample worth it? | Weighted versus counted accuracy on a shared row set | 120 gold: counting wins 0.842 to 0.812; 600 gold: weighting wins 0.849 |
| Can you skip labels entirely? | Pairwise moment over the product of estimated accuracies | 0.94x is safe, 9.12x means the triplet is broken |

The matrix is not the model. It is a set of claims about your data, made by people who wrote regexes at different times, and the audit is how those claims become measurable enough to train on.

## References

- [Snorkel: Rapid Training Data Creation with Weak Supervision](https://arxiv.org/abs/1711.10160) — Ratner et al., VLDB 2018. Label matrix definition, the tie-vote example, the 2.8x and 45.5% user study, and the 2.16-point optimizer result.
- [Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale](https://arxiv.org/abs/1812.00417) — Bach et al., SIGMOD 2019. Three Google classification tasks, quality comparable to tens of thousands of hand labels, 52% average F1 gain.
- [Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods](https://arxiv.org/abs/2002.11955) — Fu et al., ICML 2020. The closed-form triplet estimator and its dependency-graph extension.
- [A Survey on Programmatic Weak Supervision](https://arxiv.org/abs/2202.05433) — Zhang et al., 2022. The labeling-function, label-model, end-model workflow and the keyword-rule class of LFs.
- [Weaker Than You Think: A Critical Look at Weakly Supervised Learning](https://arxiv.org/abs/2305.17442) — Zhu et al., 2023. The finding that WSL methods fail without clean validation samples, across eight WRENCH datasets.
- [Weak supervision, glossary entry](https://arize.com/glossary/weak-supervision/) — Arize. The correlated-failure formulation quoted above.
- [Snorkel analysis package](https://snorkel.readthedocs.io/en/latest/packages/analysis.html) — `get_label_buckets` and the scoring utilities.

## Related posts

- [How Many Labels Does Your Language Need? Build the Curve Before You Buy](/posts/annotation-budget-language-classifier/) — the active-learning side of the same budget, which asks which rows to pay for.
- [Agreement Before Accuracy: Auditing the Labels You Are About to Train On](/posts/annotator-agreement-before-you-train/) — human inter-annotator agreement, the audit this post runs on machines.
- [Your Corpus Is Duplicating Itself](/posts/minhash-lsh-corpus-dedup/) — near-duplicate detection, the other way a label matrix gets inflated.
- [Your Encoding Saw Its Own Label: A Four-Number Audit for Target Encoding](/posts/target-encoding-leakage-audit/) — the same style of pre-training audit applied to features.
