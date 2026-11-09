---
title: "How Many Labels Does Your Language Need? Build the Curve Before You Buy"
date: 2026-11-09 00:00:00 +0300
categories: [Machine Learning, AI in Africa]
tags: [active learning, low-resource NLP, annotation, evaluation, Swahili, data budgets]
image:
  path: /assets/img/cover-annotation-budget-language-classifier.webp
  alt: A learning curve plotting five label budgets against macro-F1, a dashed line at 95 percent of the full-data score and a marker at the fitted n95 budget
---

> **The short version**
> Annotation is usually the largest line item in an African-language NLP project, and the label count is rarely measured before it is spent. A September 2026 study does the measurement for 28 language-task pairs: roughly 400 labels reach 90% of full-data macro-F1 for news topics, while tweet sentiment is still improving at full training size in 11 of 12 languages. This post turns that finding into a procedure you can run on your own corpus in an afternoon, and fills the gap the paper names in its own limitations: it did not test active learning, so we test whether choosing *which* labels to buy beats buying them at random.

## Why the label count decides the project

A character n-gram model with no pretrained weights trains in 0.1 to 6 seconds per language on two CPU cores. The paper that reports that number, by Bhanu Prakash Vangala and colleagues at the University of Missouri, generates every result in it on the same two cores with 7 GB of memory, in about two and a half hours for 28 languages, seven budgets, three seeds and two transfer matrices.

Compute like that is no longer what limits a newsroom or a university lab in Nairobi. Native-speaker annotation is. It is paid per language, again per task, and again when the label scheme changes. So the budget question comes first: how many labels do we need, and can labels already collected in another African language count towards it.

The paper answers both questions on public benchmarks. The answer is more specific than the usual advice.

## What the paper measured

Two benchmarks supply the data: MasakhaNEWS for news topic classification in 16 African languages, and AfriSenti for tweet sentiment in 12. That gives 28 language-task pairs. The model family is deliberately plain, character n-gram TF-IDF with a linear support vector machine, because it is what runs without a GPU.

| Measurement | Result |
|---|---|
| Labels needed for 90% of full-data macro-F1, news topics | about 400 in the median language |
| Labels needed for 72% of full-data score | 100 |
| Raw n95 (first budget reaching 95%), news | median 800, range 200 (chiShona) to the full training set (Tigrinya, Oromo, Rundi, Lingala, Luganda) |
| Fitted n95, news | median 575, most languages between 400 and 900 |
| Sentiment at full training size | still improving in 11 of 12 languages |
| Pooling all other languages, at 25 target labels | +0.20 macro-F1 for news (up to +0.43 for Lingala), +0.08 for sentiment |
| Pooling at full training size | hurts in 9 of 16 news languages and 8 of 12 sentiment languages |
| Zero-shot transfer, no target labels | median 13% of the floor-to-full gap for news, 4% for sentiment |

The paper's own summary of the trade-off is the sharpest line in it. A large pretrained model with pattern-exploiting training reached about 93% of full performance from ten examples per label on MasakhaNEWS. The same 90% mark costs the simple model 400 labels. The choice is between 40 labels and a GPU, or 400 labels and a laptop, and for most African deployments the second is the machine that exists.

## Build the curve before you buy

Plan from a curve measured on your corpus, at your label scheme, with your model, rather than from a number in a paper. The recipe is small: hold out a test set, subsample the labelled pool at several budgets, train and score at each, then fit the curve.

The fixture below is synthetic so that every number is reproducible with the standard library alone. It models five news classes with skewed priors, 16% label noise in the training pool and a clean test set, and documents where 84% of the words come from a shared background vocabulary. That ratio is what makes the task hard: the classes are separable, but only weakly, on the evidence in a single document.

```python
import math, random
from collections import Counter, defaultdict

CLS = ["health", "politics", "economy", "sport", "weather"]
PRI = dict(zip(CLS, [.30, .25, .20, .15, .10]))
STEM = {
    "health": "afya dawa hospitali mgonjwa chanjo kliniki madaktari wagonjwa".split(),
    "politics": "serikali uchaguzi bunge rais kura chama waziri majimbo".split(),
    "economy": "bei soko biashara shilingi madeni benki kodi mauzo".split(),
    "sport": "mpira timu goli mchezo wanariadha kikapu refa ligi".split(),
    "weather": "mvua jua hali hewa baridi joto upepo radar".split(),
}
BG = ("na ya kwa hii sasa pia kama zaidi leo watu habari wiki mwaka nchi "
      "mkutano taarifa kesho jana kwenye hilo huyu baada kabla ndani nje").split()
NOISE = 0.84
AFFIX = ("", "ki", "ma", "wa")


def doc(c, rng, stems=STEM, affix=AFFIX, noise=NOISE):
    out = []
    for _ in range(rng.randint(8, 20)):
        out.append(rng.choice(BG) if rng.random() < noise
                   else rng.choice(affix) + rng.choice(stems[c]))
    return " ".join(out)


def corpus(n, seed, label_noise=0.10, stems=STEM, affix=AFFIX):
    rng, rows = random.Random(seed), []
    for c in CLS:
        for _ in range(int(round(n * PRI[c]))):
            y = rng.choice([x for x in CLS if x != c]) if rng.random() < label_noise else c
            rows.append((doc(c, rng, stems, affix), y))
    rng.shuffle(rows)
    return rows


def grams(text, k=3):
    t = "^" + text.lower().replace(" ", "_") + "$"
    return Counter(t[i:i + k] for i in range(len(t) - k + 1))


class NB:
    def __init__(self, alpha=1.0):
        self.alpha, self.cnt = alpha, defaultdict(Counter)
        self.tot, self.prior, self.vocab = Counter(), Counter(), set()

    def fit(self, rows):
        for text, y in rows:
            self.prior[y] += 1
            for g, c in grams(text).items():
                self.cnt[y][g] += c
                self.tot[y] += c
                self.vocab.add(g)
        self.V = max(1, len(self.vocab))
        self.N = sum(self.prior.values())
        return self

    def scores(self, text):
        gs = grams(text)
        out = {}
        for y in self.prior:
            v = math.log(self.prior[y] / self.N)
            den = self.tot[y] + self.alpha * self.V
            for g, c in gs.items():
                v += c * math.log((self.cnt[y][g] + self.alpha) / den)
            out[y] = v
        return out

    def predict(self, text):
        return max(self.scores(text).items(), key=lambda kv: kv[1])[0]


class Constant:
    """Predicts the most frequent training class: the floor a model must beat."""

    def fit(self, rows):
        self.y = Counter(y for _, y in rows).most_common(1)[0][0]
        return self

    def predict(self, text):
        return self.y


def macro_f1(model, test):
    tp, fp, fn = Counter(), Counter(), Counter()
    for text, y in test:
        p = model.predict(text)
        tp[y] += p == y
        if p != y:
            fp[p] += 1
            fn[y] += 1
    out = []
    for y in CLS:
        pr = tp[y] / (tp[y] + fp[y]) if tp[y] + fp[y] else 0.0
        rc = tp[y] / (tp[y] + fn[y]) if tp[y] + fn[y] else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(out) / len(out)


train = corpus(500, seed=7)
test = corpus(150, seed=99, label_noise=0.0)
full = macro_f1(NB().fit(train), test)
print(f"corpus={len(train)} test={len(test)} full-data macro-F1={full:.3f}")
print(f"{'labels':>7}{'median':>9}{'range':>15}{'% of full':>11}")

curve = {}
for B in [25, 50, 100, 200, 400]:
    run = []
    for s in range(3):
        idx = random.Random(s).sample(range(len(train)), B)
        run.append(macro_f1(NB().fit([train[i] for i in idx]), test))
    run.sort()
    curve[B] = run[1]
    print(f"{B:7d}{run[1]:9.3f}{run[0]:7.3f}-{run[-1]:.3f}{run[1]/full:10.1%}")

best = None
for ai in range(61):                        # asymptote at or above the data
    a = max(curve.values()) + ai * 0.005
    for ci in range(1, 301):
        c, xs = ci / 100, [n ** -(ci / 100) for n in curve]
        b = sum(x * (a - curve[n]) for x, n in zip(xs, curve)) / sum(x * x for x in xs)
        if b <= 0:
            continue
        err = sum((curve[n] - (a - b * x)) ** 2 for x, n in zip(xs, curve))
        if best is None or err < best[0]:
            best = (err, a, b, c)
_, a, b, c = best
target = 0.95 * full
n95 = math.ceil(((a - target) / b) ** (-1 / c))
print(f"fit F(n)=a-b*n^-c: a={a:.3f} b={b:.3f} c={c:.2f} rmse={math.sqrt(best[0]/len(curve)):.3f}")
print(f"n95 (fit reaches 95% of full)={n95}")
print(f"gain per step: 25->100 {curve[100]-curve[25]:+.3f} "
      f"100->200 {curve[200]-curve[100]:+.3f} 200->400 {curve[400]-curve[200]:+.3f}")
```

```
corpus=500 test=150 full-data macro-F1=0.880
 labels   median          range  % of full
     25    0.375  0.375-0.463     42.7%
     50    0.586  0.517-0.619     66.6%
    100    0.690  0.675-0.728     78.4%
    200    0.833  0.813-0.861     94.8%
    400    0.878  0.877-0.883     99.9%
fit F(n)=a-b*n^-c: a=1.058 b=3.293 c=0.49 rmse=0.015
n95 (fit reaches 95% of full)=245
gain per step: 25->100 +0.314 100->200 +0.144 200->400 +0.045
```

Read the middle of that table, not the ends. Going from 25 to 100 labels buys 31 points of macro-F1. The next 100 buys 14, and the 200 after that buys four and a half. Your money stops working somewhere on that curve, and the curve shows you where before you have spent it.

One stop rule worth writing into the plan: if 400 labels in your language do not clear the majority-class floor by a margin you can use, the problem is the label scheme or the task, not the budget. More labels will not fix a scheme that separates nothing.

## Turn the curve into a budget you can defend

The fitted line follows the form F(n) = a - b·n^-c, the inverse power law the paper borrows from Figueroa and colleagues. Fitting it gives one number a manager can act on: n95, the smallest budget at which the fitted curve reaches 95% of the full-data score.

Two caveats about that fit. The estimated asymptote here is 1.058, above the maximum possible macro-F1 of 1.0, because five points on a short curve cannot pin an asymptote down. And the residual error is small (0.015), which makes the fitted n95 of 245 look more precise than it is. Treat n95 as a planning band, not a promise.

The paper's own numbers put n95 at a median of 575 fitted and 800 raw for news, because its 16 languages include several whose curves were still rising at the full training set. My fixture needs 245 because five classes with distinctive stems separate faster than fourteen news topics do. Both numbers are correct for their corpora. That is the argument for measuring yours rather than importing one.

## Which 100 labels: the strategy is not the lever

Active learning is the standard answer to a small budget: start with a seed set, train, then buy the examples the model is least sure about. The paper did not test it and says so in its limitations, so the second experiment fills that gap on the same fixture, at a fixed budget of 100 labels and a 20-label seed.

Four strategies, three seeds each. Least-confidence buys the highest maximum log-probability, margin buys the smallest gap between the top two classes, balanced buys by uncertainty but caps each predicted class at two per batch, and random ignores the model entirely.

```python
import math, random
from collections import Counter, defaultdict

CLS = ["health", "politics", "economy", "sport", "weather"]
PRI = dict(zip(CLS, [.30, .25, .20, .15, .10]))
STEM = {
    "health": "afya dawa hospitali mgonjwa chanjo kliniki madaktari wagonjwa".split(),
    "politics": "serikali uchaguzi bunge rais kura chama waziri majimbo".split(),
    "economy": "bei soko biashara shilingi madeni benki kodi mauzo".split(),
    "sport": "mpira timu goli mchezo wanariadha kikapu refa ligi".split(),
    "weather": "mvua jua hali hewa baridi joto upepo radar".split(),
}
BG = ("na ya kwa hii sasa pia kama zaidi leo watu habari wiki mwaka nchi "
      "mkutano taarifa kesho jana kwenye hilo huyu baada kabla ndani nje").split()
NOISE = 0.84
AFFIX = ("", "ki", "ma", "wa")


def doc(c, rng, stems=STEM, affix=AFFIX, noise=NOISE):
    out = []
    for _ in range(rng.randint(8, 20)):
        out.append(rng.choice(BG) if rng.random() < noise
                   else rng.choice(affix) + rng.choice(stems[c]))
    return " ".join(out)


def corpus(n, seed, label_noise=0.10, stems=STEM, affix=AFFIX):
    rng, rows = random.Random(seed), []
    for c in CLS:
        for _ in range(int(round(n * PRI[c]))):
            y = rng.choice([x for x in CLS if x != c]) if rng.random() < label_noise else c
            rows.append((doc(c, rng, stems, affix), y))
    rng.shuffle(rows)
    return rows


def grams(text, k=3):
    t = "^" + text.lower().replace(" ", "_") + "$"
    return Counter(t[i:i + k] for i in range(len(t) - k + 1))


class NB:
    def __init__(self, alpha=1.0):
        self.alpha, self.cnt = alpha, defaultdict(Counter)
        self.tot, self.prior, self.vocab = Counter(), Counter(), set()

    def fit(self, rows):
        for text, y in rows:
            self.prior[y] += 1
            for g, c in grams(text).items():
                self.cnt[y][g] += c
                self.tot[y] += c
                self.vocab.add(g)
        self.V = max(1, len(self.vocab))
        self.N = sum(self.prior.values())
        return self

    def scores(self, text):
        gs = grams(text)
        out = {}
        for y in self.prior:
            v = math.log(self.prior[y] / self.N)
            den = self.tot[y] + self.alpha * self.V
            for g, c in gs.items():
                v += c * math.log((self.cnt[y][g] + self.alpha) / den)
            out[y] = v
        return out

    def predict(self, text):
        return max(self.scores(text).items(), key=lambda kv: kv[1])[0]


class Constant:
    """Predicts the most frequent training class: the floor a model must beat."""

    def fit(self, rows):
        self.y = Counter(y for _, y in rows).most_common(1)[0][0]
        return self

    def predict(self, text):
        return self.y


def macro_f1(model, test):
    tp, fp, fn = Counter(), Counter(), Counter()
    for text, y in test:
        p = model.predict(text)
        tp[y] += p == y
        if p != y:
            fp[p] += 1
            fn[y] += 1
    out = []
    for y in CLS:
        pr = tp[y] / (tp[y] + fp[y]) if tp[y] + fp[y] else 0.0
        rc = tp[y] / (tp[y] + fn[y]) if tp[y] + fn[y] else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(out) / len(out)


train = corpus(500, seed=7)
test = corpus(150, seed=99, label_noise=0.0)


def acquire(strategy, budget=100, seed_n=20, batch=10, seed=0):
    rng = random.Random(100 + seed)
    pool = list(range(len(train)))
    rng.shuffle(pool)
    chosen, unlab = pool[:seed_n], pool[seed_n:]
    while len(chosen) < budget:
        model = NB().fit([train[i] for i in chosen])
        if strategy == "random":
            pick = rng.sample(unlab, batch)
        else:
            scored = []
            for i in unlab:
                lp = sorted(model.scores(train[i][0]).values(), reverse=True)
                scored.append((lp[0] - lp[1] if strategy == "margin" else lp[0], i))
            scored.sort()
            pick, seen = [], Counter()
            for _, i in scored:
                pred = model.predict(train[i][0])
                if strategy != "balanced" or seen[pred] < 2:
                    pick.append(i)
                    seen[pred] += 1
                if len(pick) == batch:
                    break
        chosen += pick
        unlab = [i for i in unlab if i not in set(pick)]
    return chosen


print(f"{'strategy':<11}{'seed0':>8}{'seed1':>8}{'seed2':>8}{'mean':>8}{'spread':>8}{'min class':>10}")
result = {}
for strategy in ["random", "least_conf", "margin", "balanced"]:
    scores, cover = [], []
    for s in range(3):
        chosen = acquire(strategy, seed=s)
        scores.append(macro_f1(NB().fit([train[i] for i in chosen]), test))
        cover.append(min(Counter(train[i][1] for i in chosen).values()))
    result[strategy] = scores
    print(f"{strategy:<11}" + "".join(f"{v:8.3f}" for v in scores)
          + f"{sum(scores)/3:8.3f}{max(scores)-min(scores):8.3f}{min(cover):10d}")

base = sum(result["random"]) / 3
spread = max(result["random"]) - min(result["random"])
for k in ["least_conf", "margin", "balanced"]:
    delta = sum(result[k]) / 3 - base
    print(f"{k}: mean delta {delta:+.3f} vs random; random seed spread {spread:.3f}; "
          f"clears the spread: {abs(delta) > spread}")
```

```
strategy      seed0   seed1   seed2    mean  spread min class
random        0.771   0.789   0.692   0.751   0.097        12
least_conf    0.793   0.807   0.789   0.797   0.018         6
margin        0.744   0.721   0.750   0.738   0.029        14
balanced      0.803   0.824   0.783   0.803   0.041        12
least_conf: mean delta +0.046 vs random; random seed spread 0.097; clears the spread: False
margin: mean delta -0.012 vs random; random seed spread 0.097; clears the spread: False
balanced: mean delta +0.052 vs random; random seed spread 0.097; clears the spread: False
```

Every active-learning strategy lands between 0.738 and 0.803 on the mean, against 0.751 for random, and none of the gaps clears random's own seed spread of 0.097. Margin sampling is slightly worse than random. I have not seen that result in a vendor deck, and it is the one that should decide whether you build the loop: on a hard, imbalanced five-class task with a plain model, *which* labels you buy is a smaller effect than the draw.

Two differences do hold up. Variance: least-confidence scores 0.789 to 0.807 across seeds while random spans 0.692 to 0.789. Buying the most uncertain examples removes most of the lottery, which is worth something when you get one shot at a budget. Class coverage: least-confidence leaves the smallest class with 6 labelled examples against random's 12, because it keeps buying the hardest documents and those cluster in the confusable classes. The balanced variant holds 12 and posts the best mean, which is the combination to copy if you are going to use active learning at all.

## The cheapest multiplier is another language, if it is the right one

Pooling is the second half of the paper's answer, and the half that gets misquoted. Labels from another African language help most when you have almost none of your own, and the gain decays to zero as your own set grows. The paper reports +0.20 macro-F1 at 25 target labels for news, +0.08 for sentiment, decay to nothing by 800 labels, and active harm at full size in 9 of 16 news languages.

Whether that holds for you depends on how much the two languages actually share. The third experiment runs two sources: one that shares five of eight roots per class and inflects them differently, and one with genuinely different roots in the same script.

```python
import math, random
from collections import Counter, defaultdict

CLS = ["health", "politics", "economy", "sport", "weather"]
PRI = dict(zip(CLS, [.30, .25, .20, .15, .10]))
STEM = {
    "health": "afya dawa hospitali mgonjwa chanjo kliniki madaktari wagonjwa".split(),
    "politics": "serikali uchaguzi bunge rais kura chama waziri majimbo".split(),
    "economy": "bei soko biashara shilingi madeni benki kodi mauzo".split(),
    "sport": "mpira timu goli mchezo wanariadha kikapu refa ligi".split(),
    "weather": "mvua jua hali hewa baridi joto upepo radar".split(),
}
BG = ("na ya kwa hii sasa pia kama zaidi leo watu habari wiki mwaka nchi "
      "mkutano taarifa kesho jana kwenye hilo huyu baada kabla ndani nje").split()
NOISE = 0.84
AFFIX = ("", "ki", "ma", "wa")


def doc(c, rng, stems=STEM, affix=AFFIX, noise=NOISE):
    out = []
    for _ in range(rng.randint(8, 20)):
        out.append(rng.choice(BG) if rng.random() < noise
                   else rng.choice(affix) + rng.choice(stems[c]))
    return " ".join(out)


def corpus(n, seed, label_noise=0.10, stems=STEM, affix=AFFIX):
    rng, rows = random.Random(seed), []
    for c in CLS:
        for _ in range(int(round(n * PRI[c]))):
            y = rng.choice([x for x in CLS if x != c]) if rng.random() < label_noise else c
            rows.append((doc(c, rng, stems, affix), y))
    rng.shuffle(rows)
    return rows


def grams(text, k=3):
    t = "^" + text.lower().replace(" ", "_") + "$"
    return Counter(t[i:i + k] for i in range(len(t) - k + 1))


class NB:
    def __init__(self, alpha=1.0):
        self.alpha, self.cnt = alpha, defaultdict(Counter)
        self.tot, self.prior, self.vocab = Counter(), Counter(), set()

    def fit(self, rows):
        for text, y in rows:
            self.prior[y] += 1
            for g, c in grams(text).items():
                self.cnt[y][g] += c
                self.tot[y] += c
                self.vocab.add(g)
        self.V = max(1, len(self.vocab))
        self.N = sum(self.prior.values())
        return self

    def scores(self, text):
        gs = grams(text)
        out = {}
        for y in self.prior:
            v = math.log(self.prior[y] / self.N)
            den = self.tot[y] + self.alpha * self.V
            for g, c in gs.items():
                v += c * math.log((self.cnt[y][g] + self.alpha) / den)
            out[y] = v
        return out

    def predict(self, text):
        return max(self.scores(text).items(), key=lambda kv: kv[1])[0]


class Constant:
    """Predicts the most frequent training class: the floor a model must beat."""

    def fit(self, rows):
        self.y = Counter(y for _, y in rows).most_common(1)[0][0]
        return self

    def predict(self, text):
        return self.y


def macro_f1(model, test):
    tp, fp, fn = Counter(), Counter(), Counter()
    for text, y in test:
        p = model.predict(text)
        tp[y] += p == y
        if p != y:
            fp[p] += 1
            fn[y] += 1
    out = []
    for y in CLS:
        pr = tp[y] / (tp[y] + fp[y]) if tp[y] + fp[y] else 0.0
        rc = tp[y] / (tp[y] + fn[y]) if tp[y] + fn[y] else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(out) / len(out)


train = corpus(500, seed=7)
test = corpus(150, seed=99, label_noise=0.0)
full = macro_f1(NB().fit(train), test)
floor = macro_f1(Constant().fit(train), test)
print(f"in-language full data {full:.3f}, majority-class floor {floor:.3f}")

STEM_REL = {c: STEM[c][:5] + [s + "ni" for s in STEM[c][5:]] for c in CLS}
STEM_UNREL = {
    "health": "ubuzima umuti ibitaro umurwayi kinga ivuriro abaganga indwara".split(),
    "politics": "leta inteko perezida amatora ishyaka minisitiri intara abadepite".split(),
    "economy": "igiciro isoko ubucuruzi amafaranga imyenda banki imisoro amasoko".split(),
    "sport": "umupira ikipe igitego umukino abakinnyi inkundu umusifuzi shampiyona".split(),
    "weather": "imvura izuba ubururu umuyaga ubushyuhe ikirere rada ibicu".split(),
}
AFFIX_S = ("", "lu", "ba", "chi")

for name, stems in [("shared roots", STEM_REL), ("different roots", STEM_UNREL)]:
    src = corpus(500, seed=11, stems=stems, affix=AFFIX_S)
    zero = macro_f1(NB().fit(src), test)
    print(f"\n{name}: source-only {zero:.3f}, recovers "
          f"{(zero - floor) / (full - floor):.1%} of the floor-to-full gap")
    for B in [25, 100, 400]:
        mono, pool = [], []
        for s in range(3):
            sub = [train[i] for i in random.Random(s).sample(range(len(train)), B)]
            mono.append(macro_f1(NB().fit(sub), test))
            pool.append(macro_f1(NB().fit(sub + src), test))
        print(f"  {B:4d} target labels: monolingual {sum(mono)/3:.3f} "
              f"pooled {sum(pool)/3:.3f} delta {sum(pool)/3 - sum(mono)/3:+.3f}")
```

```
in-language full data 0.880, majority-class floor 0.092

shared roots: source-only 0.908, recovers 103.7% of the floor-to-full gap
    25 target labels: monolingual 0.404 pooled 0.901 delta +0.496
   100 target labels: monolingual 0.698 pooled 0.900 delta +0.203
   400 target labels: monolingual 0.879 pooled 0.886 delta +0.006

different roots: source-only 0.210, recovers 14.9% of the floor-to-full gap
    25 target labels: monolingual 0.404 pooled 0.549 delta +0.145
   100 target labels: monolingual 0.698 pooled 0.827 delta +0.129
   400 target labels: monolingual 0.879 pooled 0.909 delta +0.030
```

The unrelated source recovers 14.9% of the gap on its own, which is close to the paper's median of 13% for news and 4% for sentiment. It adds 14.5 points at 25 target labels and 3 points at 400. That is the shape the paper describes, reproduced on a fixture you can audit.

The shared-roots source tells a different and more useful story. It scores 0.908 alone, above the 0.880 in-language model trained on five times as much data, and it "recovers" 103.7% of the gap. A transfer source that beats your in-language ceiling is a leakage alarm, not a win: the other language is close enough to a paraphrase that you are training on your own data with a different affix table. If you see that number in a pilot, stop and check the roots, the script and the label prior before you count the gain.

That is the paper's warning, made concrete. A source that looks strong on paper may be exploiting the label prior rather than learning the language, because a model that only predicts the majority class already scores 0.092 here. A weak model with the wrong prior can clear that bar and still teach your classifier nothing transferable.

## A protocol you can run this week

| Step | What you do | What it decides |
|---|---|---|
| 1 | Hold out 150 to 300 documents; never label them into the training pool | whether your score means anything |
| 2 | Measure macro-F1 at 25, 50, 100, 200 and 400 labels, three draws each | the marginal return on your next 100 labels |
| 3 | Fit F(n) = a - b·n^-c and report n95 with its residual error | the number you put in the budget request |
| 4 | Run one week of least-confidence or balanced acquisition against random, same budget | whether acquisition is worth the engineering |
| 5 | Score one related and one unrelated source language, with a majority-class floor | whether pooling earns its keep, and when to drop it |

Steps 1 to 3 cost one afternoon and settle most of the argument. Steps 4 and 5 are the ones people skip, and they are the ones that keep a pilot honest.

## What this does not settle

The fixture is synthetic and five classes deep. Your corpus has fourteen topics, code-switching, OCR noise and annotators who disagree, and the paper's own caveat applies: three draws per budget is enough for error bars, not enough to rule out a single unlucky draw at 25 labels that misses a class entirely.

The model matters too. A pretrained encoder would lift every curve and, on the evidence in the MasakhaNEWS paper, flatten the news curves much earlier. The 400-label figure is a laptop-class result, not a universal one.

Cost is the largest omission. All four strategies above spend labels at a uniform price. The EACL 2026 work by Bonaventure Dossou and colleagues shows why that is wrong for African annotation: costs vary with task complexity, annotator expertise and data availability, and their KnapsackBALD method treats acquisition as a 0.1 knapsack problem, beating seven baselines across three cost scenarios with the gap widening as costs become more unequal. If a health label costs three times a sports label in your setting, the strategy table above is answering the wrong question.

## Key takeaways

| Finding | Number |
|---|---|
| Labels for 90% of full-data macro-F1, news topics | about 400 (paper), 245 fitted n95 on this fixture |
| Return on the second 100 labels | +0.144 macro-F1, versus +0.314 for the first 75 |
| Best active-learning mean vs random | +0.052, inside random's own 0.097 seed spread |
| Real gain from acquisition | variance: spread 0.018 versus 0.097 |
| Pooling an unrelated language at 25 labels | +0.145 macro-F1, decaying to +0.030 at 400 |
| Pooling a near-duplicate language | leakage alarm, not a gain |

## References

- Vangala, Guda, Peddi and Vangala, *How Many Labels Does a Language Need? Annotation Budgets and Cross-Lingual Pooling for African-Language Text Classification*, arXiv 2609.37882, 29 September 2026 — https://arxiv.org/abs/2609.37882
- Dossou, Arous, Durand and Cheung, *Active Learning with Non-Uniform Costs for African Natural Language Processing*, Findings of EACL 2026, pages 6644-6656 — https://aclanthology.org/2026.findings-eacl.349/
- Settles, *Active Learning Literature Survey*, University of Wisconsin-Madison — https://burrsettles.com/pub/settles.activelearning.pdf
- modAL documentation, uncertainty sampling strategies (least confident, margin, entropy) — https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html
- EACL 2026 paper PDF, cost scenarios and KnapsackBALD results — https://aclanthology.org/2026.findings-eacl.349.pdf

## Related posts

- [Agreement before accuracy: the ceiling your F1 is measured against](/posts/annotator-agreement-before-you-train/)
- [Low-resource NLP: what works when the corpus is small](/posts/low-resource-nlp/)
- [Fine-tuning African language models on a budget](/posts/fine-tuning-african-language-llms/)
- [Pass@k is not accuracy: reading a reliability number](/posts/passk-eval-reliability/)
