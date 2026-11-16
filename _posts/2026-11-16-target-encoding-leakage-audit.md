---
title: "Your Encoding Saw Its Own Label: A Four-Number Audit for Target Encoding"
date: 2026-11-16 00:00:00 +0300
categories: [Machine Learning, Data Science]
tags: [target encoding, data leakage, cross-validation, feature engineering, evaluation, fraud]
math: true
image:
  path: /assets/img/cover-target-encoding-leakage-audit.webp
  alt: A grid of merchant tiles where one tile is wired back to its own label, beside an AUC readout falling from 0.96 in-sample to 0.50 on fresh rows
---

> **The short version**
> Target encoding replaces a category with the average label of the rows in that category, and one of those rows is the row being encoded. A column with one row per level reports an AUC of 1.0000 against the label it was built from; at five rows per level, 0.9601; on fresh rows from merchants the encoding never saw, 0.5000. Four numbers tell you which of two different bugs you have, and smoothing fixes neither: at a shrinkage of 500 the noise column still reads 0.6912 while the honest out-of-fold figure stays near 0.49. Every figure below comes from a seeded synthetic book of 30,000 to 50,000 transactions, reproducible and yours to resize.
{: .prompt-info }

A payments book has a column that looks like a gift. Six thousand merchants, thirty thousand transactions, two percent fraud, and a single `merchant_id` that a gradient-boosted model cannot take as a number. Target encoding turns it into one float per merchant: the merchant's historical fraud rate. The feature correlates with the label, the model ranks well, and nothing in the pipeline raises a warning.

The problem is not that the encoding is wrong. It is that the encoding for a row includes that row's own outcome, and the amount of it included is a quantity you can compute before you train anything.

Two published measurements frame the effect. A data scientist writing in August 2026 built 50,000 random merchant-style IDs with no connection to the label, encoded them the way most tutorials do, and pushed a five-fold cross-validated AUC to 0.996; on a fresh batch the same model scored 0.649, a gap cross-validation had reported without hesitation ([Medium, 18 August 2026](https://medium.com/data-and-beyond/when-your-target-encoding-is-leaking-and-cross-validation-dont-catch-it-bfb3666ccdf9)). scikit-learn's documentation ships a worked example with the same shape: a Ridge regressor on target-encoded features scoring 0.8585 on its training set and 0.6338 on test when the encoding is naive, against 0.8000 and 0.7928 when the encoder cross-fits ([scikit-learn example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html)).

## What the encoding computes

For a categorical column with level $c$, the plain encoding of every row in $c$ is

$$S_c = \frac{1}{n_c}\sum_{i \in c} y_i$$

where $n_c$ is the number of rows at that level. Pick any row $j$ inside the category and the same expression turns into two terms:

$$S_c = \frac{1}{n_c}y_j + \frac{1}{n_c}\sum_{i \in c,\, i \neq j} y_i$$

The first term is the row's own label, weighted by $1/n_c$. That weight is the entire leak, and it is not hidden anywhere: it is the reciprocal of the level's row count. A merchant with one transaction contributes its own label to its own feature with a weight of 1.0. A merchant with a hundred transactions contributes it with a weight of 0.01.

Shrinkage, the usual companion of target encoding, mixes the level mean with the global mean:

$$S_c^{(m)} = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m} = \lambda_c \bar{y}_c + (1 - \lambda_c)\bar{y}, \qquad \lambda_c = \frac{n_c}{n_c + m}$$

scikit-learn states the estimator in the same two forms, with `smooth="auto"` setting $m$ to an empirical Bayes estimate $m = \sigma_c^2/\tau^2$, where $\sigma_c^2$ is the variance of the label within the category and $\tau^2$ is its global variance ([user guide, section 8.3](https://scikit-learn.org/stable/modules/preprocessing.html)). The parameter is documented, the formula is documented, and the leakage is documented too, in a sentence most readers skip: the encoder's `fit` method "does not use any cross fitting schemes and learns one encoding on the entire training set. It is discouraged to use this method because it can introduce data leakage".

So the technique is not the problem. The problem is that the plain formula is a *ratio* between the row's own label and the number of rows that share its level, and in payments data both sides of that ratio are small.

## The leak is arithmetic you can run before training

The first block builds the worst case honestly: thirty thousand transactions, a two percent fraud rate, and a level column with *no relationship at all* to the label, arranged so every level holds exactly a fixed number of rows. It encodes each row with its own level's mean and computes the AUC of that single column against the label. It also encodes a second, independent draw of rows from the same ID space using the map learned on the first, which is what happens when a model meets a merchant it has already seen.

{% raw %}
```python
"""Block 1 — how much of a row's own label sits inside its encoding."""
import numpy as np


def auc(y, s):
    """Rank AUC with average ranks for ties. No sklearn needed."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    rs = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        rs[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(s))
    ranks[order] = rs
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def build(n, rows_per_level, seed):
    """Level ids with exactly rows_per_level rows each. Labels carry NO level signal."""
    rng = np.random.default_rng(seed)
    level = (rng.permutation(n) // rows_per_level).astype(int)
    y = (rng.random(n) < 0.02).astype(int)      # 2% base rate, independent of level
    return level, y


print(f"{'rows/level':>10} {'levels':>7} {'1/n_c':>7} {'naive AUC':>10} {'fresh AUC':>10}")
for rpl in (1, 2, 5, 20, 100):
    level, y = build(30000, rpl, seed=1)
    codes, inv = np.unique(level, return_inverse=True)
    n_c = np.bincount(inv, minlength=len(codes)).astype(float)
    s_c = np.bincount(inv, weights=y, minlength=len(codes))
    enc = s_c / n_c
    own_weight = float((1.0 / n_c[inv]).mean())   # mean share of a row's own label

    fresh_level, fresh_y = build(30000, rpl, seed=2)   # new rows, same id space
    lut = dict(zip(codes.tolist(), enc.tolist()))
    prior = float(y.mean())
    fresh_enc = np.array([lut.get(int(c), prior) for c in fresh_level])

    print(f"{rpl:>10} {len(codes):>7} {own_weight:>7.2f} "
          f"{auc(y, enc[inv]):>10.4f} {auc(fresh_y, fresh_enc):>10.4f}")
```
{% endraw %}

```text
rows/level  levels   1/n_c  naive AUC  fresh AUC
         1   30000    1.00     1.0000     0.5004
         2   15000    0.50     0.9898     0.5003
         5    6000    0.20     0.9601     0.5081
        20    1500    0.05     0.8595     0.5031
       100     300    0.01     0.6881     0.5062
```

Every number in the first row is exact. With one row per level the encoding *is* the label, so the column ranks perfectly, and the AUC of 1.0000 is not a coincidence of the seed.

The last row is the one that matters for anyone who thought about this already. At a hundred rows per level, a row's own label contributes one percent of its own feature. One percent sounds negligible, and the naive AUC of 0.6881 says it is not. The encoding ranks levels against each other, and at a two percent base rate the spread it ranks on is tiny: on the same construction, the standard deviation of the level means is 0.01427 against a self-contribution of 0.01. The leak is about seventy percent of the signal the column is supposed to carry, in a setting where the column carries nothing.

The ratio to estimate for your own table is $1/n_c$ against the between-level spread, and both halves are cheap:

- $1/n_c$ is a mean over your category counts. If the median count is 13, the typical row carries 7.7 percent of its own label.
- The between-level spread is the standard deviation of level means, which for a rare event of rate $p$ and count $n$ is roughly $\sqrt{p(1-p)/n}$. At $p = 0.02$ and $n = 100$ that is 0.014, which is where the measured 0.01427 comes from.

The last column is the control. The same column, applied to fresh rows from the same ID space, is a coin toss: 0.5004, 0.5003, 0.5081, 0.5031, 0.5062. The information was never in the feature. It was in the fact that the feature was computed from the label it was being scored against.

## Four numbers, two different bugs

The leak has two distinct causes and they produce the same symptom, which is why teams argue about it. The first is self-inclusion: the encoding for a training row is computed from a set that includes that row. The second is evaluation-side: the *test* rows are encoded with their own labels, so the score you report is computed from data the score is supposed to be measuring.

Both are visible at the feature level if you compute four numbers instead of one. The block below builds a five-rows-per-level column with no signal, then measures the same column four ways: encoded on the full frame and scored in-sample, encoded out-of-fold and scored in-sample, applied to genuinely new IDs from the training map, and applied to new IDs encoded with their own labels.

{% raw %}
```python
"""Block 2 — four numbers, two different bugs."""
import numpy as np


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    rs = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        rs[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(s))
    ranks[order] = rs
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def enc_map(level, y):
    """Level -> mean label. Returns codes, encodings, and the global prior."""
    codes, inv = np.unique(level, return_inverse=True)
    n_c = np.bincount(inv, minlength=len(codes)).astype(float)
    s_c = np.bincount(inv, weights=np.asarray(y, dtype=float), minlength=len(codes))
    return codes, s_c / n_c, float(np.mean(y))


def apply_map(codes, enc, prior, level):
    lut = dict(zip(codes.tolist(), enc.tolist()))
    return np.array([lut.get(int(c), prior) for c in level])


def oof_encode(level, y, k=5, seed=11):
    """Out-of-fold: each fold is encoded from the other k-1 folds only."""
    fold = np.random.default_rng(seed).permutation(len(y)) % k
    out = np.empty(len(y))
    for f in range(k):
        tr, te = fold != f, fold == f
        codes, enc, prior = enc_map(level[tr], y[tr])
        out[te] = apply_map(codes, enc, prior, level[te])
    return out


def build(n, rows_per_level, seed, offset=0):
    rng = np.random.default_rng(seed)
    level = (rng.permutation(n) // rows_per_level).astype(int) + offset
    y = (rng.random(n) < 0.02).astype(int)     # no relationship to level
    return level, y


level, y = build(30000, 5, seed=4)                      # 6,000 merchants, 5 rows each
codes, enc, prior = enc_map(level, y)

naive = apply_map(codes, enc, prior, level)             # (a) full-frame, scored in-sample
oof = oof_encode(level, y)                              # (b) out-of-fold, scored in-sample

fresh_level, fresh_y = build(30000, 5, seed=5, offset=10**6)   # genuinely new merchants
fresh_from_train = apply_map(codes, enc, prior, fresh_level)   # (c) encode from train map
fc, fe, fp = enc_map(fresh_level, fresh_y)
fresh_self = apply_map(fc, fe, fp, fresh_level)               # (d) encode with own labels

print(f"(a) in-sample,  full-frame encoding      AUC = {auc(y, naive):.4f}")
print(f"(b) in-sample,  5-fold out-of-fold       AUC = {auc(y, oof):.4f}")
print(f"(c) fresh rows, encoded from train map   AUC = {auc(fresh_y, fresh_from_train):.4f}")
print(f"(d) fresh rows, encoded with own labels  AUC = {auc(fresh_y, fresh_self):.4f}")
```
{% endraw %}

```text
(a) in-sample,  full-frame encoding      AUC = 0.9639
(b) in-sample,  5-fold out-of-fold       AUC = 0.4969
(c) fresh rows, encoded from train map   AUC = 0.5000
(d) fresh rows, encoded with own labels  AUC = 0.9582
```

Four numbers, four different computations, one column with no information in it. Read them as a diagnosis:

| Number | What produced it | What it tells you |
|---|---|---|
| (a) 0.9639 | Encoding fit and applied on the same rows | You are looking at the leak, not at a feature |
| (b) 0.4969 | Encoding fit without the row's fold | The column has no signal, and this is what that looks like |
| (c) 0.5000 | Training map applied to unseen levels | All rows received the prior, so all rows tie |
| (d) 0.9582 | Test rows encoded with their own labels | Your evaluation number is a fiction |

The pair that decides your action is (a) against (b). A gap of 0.47 with a feature you believed in means the encoding is fitted in the wrong place. The pair that decides whether you can trust the number you are about to report is (d) against (c), and (d) is the easier one to ship by accident, because the leak lives in a notebook cell where the encoder was applied to the frame before `train_test_split`.

The exact 0.5000 in (c) looks like a bug and is not. Every level in the fresh draw is new to the training map, so every row falls back to the prior, every row receives the same number, ties break evenly, and the rank AUC is exactly one half. That is the honest answer to the question asked: nothing is known about these merchants. If the model must score merchants it has never seen, the target encoding is not the feature to reach for; build the feature from each row's own context with a point-in-time cut so it cannot see the outcome (the mechanics are in [Temporal Validation for Fraud Models](/posts/temporal-validation-fraud-models/)).

One more detail in (b): the folds are unstratified. At a two percent base rate that can leave very few positives in a training fold, which makes that fold's level means noisy and the estimate pessimistic. Stratify on the label and fix the seed.

## Smoothing shrinks the leak and does not remove it

The standard advice after "encode out of fold" is "shrink small categories toward the global mean", and it comes from Micci-Barreca's 2001 paper on high-cardinality attributes, which is also the reference scikit-learn cites for its estimator ([SIGKDD Explorations 3(1):27-32](https://dl.acm.org/doi/10.1145/507533.507538)). The recipe is sound. What it is not is a fix for leakage, and the next block measures the difference.

The data changes shape here, because real merchant counts are not equal. The block draws 2,000 merchants from a log-normal popularity distribution, which produces the skew that any payments book has: a median of 13 rows per merchant, a maximum of 938, ten percent of levels holding two rows or fewer, and those small levels holding six tenths of one percent of the rows. Then it sweeps the shrinkage $m$ and reports the naive and out-of-fold AUC for a column with no signal and for a column with a real one.

{% raw %}
```python
"""Block 3 — smoothing shrinks the leak; it does not remove it."""
import numpy as np


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    rs = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        rs[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(s))
    ranks[order] = rs
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def enc_map(level, y, m):
    """Shrunk level means: S_c = (n_c*ybar_c + m*prior) / (n_c + m)."""
    codes, inv = np.unique(level, return_inverse=True)
    prior = float(np.mean(y))
    n_c = np.bincount(inv, minlength=len(codes)).astype(float)
    s_c = np.bincount(inv, weights=np.asarray(y, dtype=float), minlength=len(codes))
    return codes, (s_c + m * prior) / (n_c + m), prior


def apply_map(codes, enc, prior, level):
    lut = dict(zip(codes.tolist(), enc.tolist()))
    return np.array([lut.get(int(c), prior) for c in level])


def zipf_levels(n, n_levels, seed):
    """Merchant popularity is skewed: a few huge levels, a long tail of small ones."""
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 1.2, n_levels)
    return rng.choice(n_levels, size=n, p=w / w.sum())


def build(n, n_levels, signal_sd, seed):
    level = zipf_levels(n, n_levels, seed)
    rng = np.random.default_rng(seed + 100)
    p = np.full(n, 0.04)                        # 4% base rate
    if signal_sd > 0:                           # optional real level effect
        eta = rng.normal(0.0, signal_sd, n_levels)[level]
        p = np.clip(0.04 * np.exp(eta), 1e-5, 0.9)
    y = (rng.random(n) < p).astype(int)
    return level, y


level, y = build(50000, 2000, 0.0, seed=6)
codes, inv = np.unique(level, return_inverse=True)
n_c = np.bincount(inv, minlength=len(codes))
print(f"levels={len(codes)}  rows/level: min={n_c.min()} median={int(np.median(n_c))} max={n_c.max()}")
print(f"levels with <=2 rows: {(n_c <= 2).mean():.1%} of levels, "
      f"{n_c[n_c <= 2].sum() / n_c.sum():.1%} of rows")
print()
print(f"{'smooth m':>9} {'noise naive':>12} {'noise OOF':>10} {'signal naive':>13} {'signal OOF':>11}")

for m in (0.0, 5.0, 20.0, 100.0, 500.0):
    cells = []
    for sd in (0.0, 2.5):                       # noise column, then informative column
        level, y = build(50000, 2000, sd, seed=6)
        codes, enc, prior = enc_map(level, y, m)
        naive = auc(y, apply_map(codes, enc, prior, level))
        fold = np.random.default_rng(12).permutation(len(y)) % 5
        oof = np.empty(len(y))
        for f in range(5):
            tr, te = fold != f, fold == f
            c2, e2, p2 = enc_map(level[tr], y[tr], m)
            oof[te] = apply_map(c2, e2, p2, level[te])
        cells += [naive, auc(y, oof)]
    print(f"{m:>9.0f} {cells[0]:>12.4f} {cells[1]:>10.4f} {cells[2]:>13.4f} {cells[3]:>11.4f}")
```
{% endraw %}

```text
levels=1951  rows/level: min=1 median=13 max=938
levels with <=2 rows: 10.0% of levels, 0.6% of rows

 smooth m  noise naive  noise OOF  signal naive  signal OOF
        0       0.7407     0.4919        0.9449      0.9175
        5       0.7391     0.4935        0.9384      0.9203
       20       0.7312     0.4940        0.9280      0.9122
      100       0.7095     0.4927        0.9145      0.9009
      500       0.6912     0.4882        0.9064      0.8944
```

Read the first pair down the table. Smoothing moves the naive figure from 0.7407 to 0.6912, a real reduction of about five points, and it never gets near 0.5. The out-of-fold column, which is the honest measurement, sits between 0.4882 and 0.4940 for every value of $m$, because a column with no signal is a column with no signal however you shrink it.

Read the second pair across. The same shrinkage costs the informative column 2.3 points of out-of-fold AUC, from 0.9203 at $m = 5$ to 0.8944 at $m = 500$, because it pulls genuinely informative small merchants toward the base rate. The trade is five points of leak reduction for two points of signal loss, and the five points are not a fix: the number you would put on a slide is still wrong by nineteen points.

There is a mechanical reason smoothing cannot do the job, and it explains when the parameter does anything at all. If every level has the same count $n$, then $S_c^{(m)}$ is a strictly increasing function of $\bar{y}_c$ with the same constants for every level, so it preserves the ordering exactly and any rank-based metric is unchanged; on the equal-count construction, the ordering at $m = 0$ and at $m = 500$ is identical. Shrinkage only alters the ranking when counts differ, since that is the only case in which levels are pulled toward the prior by different amounts. On the skewed construction it reorders the tail, which is why the naive number moves at all.

What shrinkage genuinely buys is variance control in that tail. Ten percent of levels hold two rows or fewer, and without shrinkage those levels carry encodings of 0, 0.5 or 1.0, noise dressed as a probability. That is a real defect and $m$ is the right instrument for it. It is a different defect from leakage.

## What the library already does, and the numbers behind it

The reason to fix this rather than abandon the technique is that regularized target encoding is the encoder that wins benchmarks. Pargent, Pfisterer, Thomas and Bischl compared encoding strategies across lasso, random forest, gradient boosting, k-nearest neighbours and support vector machines, and report that "regularized versions of target encoding … consistently provided the best results", while encodings that "make unreasonable assumptions to map levels to integers" did worse ([Comput Stat 37:2671-2692](https://arxiv.org/abs/2104.00629)). Keep the technique. Move the computation.

scikit-learn has shipped `TargetEncoder` since version 1.3, and the important property is documented as an asymmetry that surprises people: `fit(X, y).transform(X)` does not equal `fit_transform(X, y)`. The second form splits the training data into $k$ folds and encodes each fold using encodings learned from the other $k-1$, with $k = 5$ by default, and the documentation is explicit that "training data should always be trained and transformed with `fit_transform(X_train, y_train)`". Levels unseen at fit time are encoded with the global target mean.

Their example puts a number on the difference, on 50,000 rows with an informative column, a shuffled column and a near-unique column, scored by a Ridge regressor. The raw features score 0.0049896314219659565 on the training set and 0.004577621581464908 on test, which is what a permuted feature looks like. With cross-fitting the pipeline scores 0.8000184677460299 and 0.7927845601690924, a gap of 0.007. Without cross-fitting it scores 0.858486250088675 on training and 0.6338211367110548 on test, a gap of 0.225 and a test score 0.159 below the honest one.

The example also settles what to do with the columns that cause this. Its near-unique column is described as the case where high cardinality features "are basically unique identifiers for samples which should generally be removed from machine learning datasets". If the level count is a large fraction of the row count, no encoding will save the column, because there is nothing to learn that generalises.

## The audit, and the case where the column is fine

The last block packages the diagnosis into one function you can point at a level column: it splits the table, computes the naive, out-of-fold and fresh-data AUC, reports the share of held-out rows whose level was never seen, and returns a verdict. It runs on two columns from the same book, one with no level effect and one where merchants genuinely differ in risk.

{% raw %}
```python
"""Block 4 — audit_encoding(): point it at your own level column."""
import numpy as np


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    rs = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        rs[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(s))
    ranks[order] = rs
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def enc_map(level, y, m):
    codes, inv = np.unique(level, return_inverse=True)
    prior = float(np.mean(y))
    n_c = np.bincount(inv, minlength=len(codes)).astype(float)
    s_c = np.bincount(inv, weights=np.asarray(y, dtype=float), minlength=len(codes))
    return codes, (s_c + m * prior) / (n_c + m), prior


def apply_map(codes, enc, prior, level):
    lut = dict(zip(codes.tolist(), enc.tolist()))
    return np.array([lut.get(int(c), prior) for c in level])


def audit_encoding(level, y, m=20.0, k=5, seed=0, holdout=0.2):
    """Report the naive, out-of-fold and fresh-data AUC of one level column."""
    n = len(y)
    cut = int(n * (1 - holdout))
    lv_tr, y_tr, lv_te, y_te = level[:cut], y[:cut], level[cut:], y[cut:]

    codes, enc, prior = enc_map(lv_tr, y_tr, m)
    naive = auc(y_tr, apply_map(codes, enc, prior, lv_tr))
    fresh = auc(y_te, apply_map(codes, enc, prior, lv_te))

    fold = np.random.default_rng(seed).permutation(cut) % k
    oof = np.empty(cut)
    for f in range(k):
        tr, te = fold != f, fold == f
        c2, e2, p2 = enc_map(lv_tr[tr], y_tr[tr], m)
        oof[te] = apply_map(c2, e2, p2, lv_tr[te])
    oof_auc = auc(y_tr, oof)

    unseen = float(np.mean([c not in set(codes.tolist()) for c in lv_te]))
    if naive - oof_auc > 0.05 and fresh < naive - 0.10:
        verdict = "LEAK - fix where the encoding is computed, not the model"
    elif oof_auc - fresh > 0.08:
        verdict = "UNSEEN LEVELS - trustworthy in-sample, blind on new levels"
    else:
        verdict = "ok - in-sample, out-of-fold and fresh agree"
    return {"naive": naive, "oof": oof_auc, "fresh": fresh,
            "unseen_share": unseen, "verdict": verdict}


def zipf_levels(n, n_levels, seed):
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 1.2, n_levels)
    return rng.choice(n_levels, size=n, p=w / w.sum())


def build(n, n_levels, signal_sd, seed):
    level = zipf_levels(n, n_levels, seed)
    rng = np.random.default_rng(seed + 100)
    p = np.full(n, 0.04)
    if signal_sd > 0:
        eta = rng.normal(0.0, signal_sd, n_levels)[level]
        p = np.clip(0.04 * np.exp(eta), 1e-5, 0.9)
    return level, (rng.random(n) < p).astype(int)


for name, sd in (("merchant_id (no signal)", 0.0), ("agent_id (real signal)", 2.5)):
    level, y = build(50000, 2000, sd, seed=7)
    r = audit_encoding(level, y)
    print(f"{name}: naive={r['naive']:.4f} oof={r['oof']:.4f} fresh={r['fresh']:.4f} "
          f"unseen={r['unseen_share']:.1%}")
    print(f"    {r['verdict']}")
```
{% endraw %}

```text
merchant_id (no signal): naive=0.7528 oof=0.5143 fresh=0.5009 unseen=0.2%
    LEAK - fix where the encoding is computed, not the model
agent_id (real signal): naive=0.9347 oof=0.9180 fresh=0.9204 unseen=0.2%
    ok - in-sample, out-of-fold and fresh agree
```

The second line is the point of the whole exercise. Target encoding is not a trap; a column whose levels really do differ in risk audits clean, and the honest number is within 1.5 points of the naive one. When the audit says `ok`, your encoding is fine and you can stop worrying about it.

The unseen share is also worth noticing. Only 0.2 percent of held-out rows have a level the training slice never saw, so the leakage in the first line is not about new merchants at all. It is entirely about a row's own label being used to build that row's feature, which is why a well-populated merchant history does not protect you.

## How to apply this to a column you already have

1. **Compute two numbers before you compute any model.** The mean of $1/n_c$ over the rows, and the standard deviation of the level means. If the first is a large fraction of the second, the encoding cannot be trusted as it stands, and you now have the arithmetic to say so in a design review.
2. **Move the encoding inside the pipeline.** `fit_transform` on training rows, `transform` on test rows, one object, one call site. The commonest version of this bug is an encoder fitted on the frame before the split, and moving the fit is a smaller change than it sounds.
3. **Report the out-of-fold number, not the in-sample one.** If a colleague quotes a validation AUC for a model whose features include a target encoding, ask which fold the encoding came from. "All of them" means the number is not a validation score.
4. **Delete levels with a handful of rows.** Near-unique identifiers cannot be encoded into something that generalises; scikit-learn's own example says to remove them. The audit's `unseen_share` output tells you how much of your evaluation set already lives in that regime.
5. **Set shrinkage from the tail, not from the metric.** $m$ controls how much a two-row merchant is allowed to say. Tune it on out-of-fold data and expect it to cost a point or two on the informative tail; that cost is the price of not shipping noise as a probability.
6. **Decide what happens to a level nobody has seen.** Falling back to the prior is correct and, as the numbers above show, uninformative. If new merchants must be scored, build the feature from the row's own context with a point-in-time join rather than from the level's history.

## Key takeaways

| What looks like the answer | What it is | What to do instead |
|---|---|---|
| The encoding is a mean, so it is safe | Each row's encoding contains $1/n_c$ of its own label | Compute $\mathbb{E}[1/n_c]$ and compare it to the spread of level means |
| A hundred rows per level makes the leak negligible | 1% self-weight against a 0.01427 between-level spread is 70% of the signal | Treat rarity and count together, not the count alone |
| The 0.96 AUC proves the feature works | That is the same column whose fresh-row AUC is 0.5000 | Report the out-of-fold or fresh-data number next to it |
| Cross-validation caught nothing, so it is fine | CV catches neither self-inclusion nor test-side encoding | Compute all four numbers; the pattern names the bug |
| Smoothing fixes leakage | At $m=500$ the noise column still reads 0.6912 and costs 2.3 points of real signal | Fix the computation first, then shrink for the tail |
| Target encoding is too dangerous to use | Regularized versions beat other encoders across five algorithms | Keep it, cross-fit it, audit it |

The demo is synthetic and seeded, and every figure above came from the four blocks in this post, run as published. What transfers is the shape: the leak is a ratio you can compute before training, the diagnosis is four numbers rather than one, and shrinkage is a variance control that has been mistaken for a fix.

## References

- scikit-learn, `TargetEncoder` API reference (cross-fitting, `smooth="auto"`, unseen levels encoded with the target mean) — https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- scikit-learn user guide, section 8.3 Preprocessing data (shrinkage formulas, `fit_transform` cross-fitting, the warning against `fit`) — https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn example, *Target Encoder's Internal Cross fitting* (50,000 rows; 0.8585/0.6338 without cross-fitting against 0.8000/0.7928 with it) — https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
- Micci-Barreca, *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems*, SIGKDD Explorations 3(1):27-32, 2001 — https://dl.acm.org/doi/10.1145/507533.507538
- Pargent, Pfisterer, Thomas & Bischl, *Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features*, Computational Statistics 37:2671-2692, 2022 — https://arxiv.org/abs/2104.00629
- Iakubovskyi, *When Your Target Encoding Is Leaking, and Cross-Validation Don't Catch It*, Data And Beyond, 18 August 2026 (50,000 random IDs, CV AUC 0.996 against 0.649 on fresh data) — https://medium.com/data-and-beyond/when-your-target-encoding-is-leaking-and-cross-validation-dont-catch-it-bfb3666ccdf9
- MetricGate, *Pitfalls of Target Encoding*, June 2026 (the leave-one-out plus m-estimate recipe) — https://metricgate.com/blogs/pitfalls-of-target-encoding/
- MetricGate, *Data Leakage: The Silent Killer of ML Models*, April 2026 (target encoding on the full frame listed among train-test contamination) — https://metricgate.com/blogs/data-leakage-machine-learning/

## Related posts

- [Temporal Validation for Fraud Models](/posts/temporal-validation-fraud-models/) — the split strategy that has to sit underneath any encoding of history
- [Contamination Checks Before You Trust a Fine-Tune Score](/posts/contamination-checks-before-you-trust-a-score/) — the same question asked of a benchmark instead of a feature
- [Clicks Are Not Judgments: Debiasing Retrieval Feedback](/posts/click-log-propensity-debiasing/) — another case where the naive estimate of a rate is biased by how the data was collected
- [Your Model's 0.9 Is Not a 90% Chance](/posts/probability-calibration-risk-scores/) — what to do once the ranking is honest and the number matters
- [How Many Labels Does Your Language Need?](/posts/annotation-budget-language-classifier/) — the label-count budget that sets how much any per-category statistic can carry
