---
title: "Who Pays for the Draft: A Break-Even Harness for Speculative Decoding"
date: 2026-11-23 00:00:00 +0300
categories: [LLM, AI Engineering]
tags: [speculative decoding, inference, latency, throughput, llm serving, evaluation]
math: true
image:
  path: /assets/img/cover-speculative-decoding-draft-cost.webp
  alt: A row of small draft tokens racing ahead of a heavier verification beam, with rejected tokens falling away
---

> **The claim, and the four numbers behind it**
> Every speculative-decoding pitch reduces to one formula: $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$. It has an acceptance rate $\alpha$, a draft length $\gamma$, and a draft-cost ratio $c$. Most blog posts quote the speedup and skip $c$; published benchmark tables usually set $c=0$, which is the same as assuming drafting is free. This post reproduces the published table from the formula, then measures $\alpha$ on two real corpora with a drafter you can run in the standard library, and reports where the arithmetic stops holding.
{: .prompt-info }

The setup that triggers the question is familiar. A team reads that speculative decoding gives 2 to 4 times the tokens per second, turns on a draft model, and measures 1.1x. Nothing is broken. The quoted number was computed for a draft cost and an acceptance rate that their workload does not have, and the same arithmetic that produced the quote also tells you what to measure first.

## What the published table assumes

Leviathan, Kalman and Matias give two results that matter here. Equation 1 is the expected number of tokens produced per verification pass:

$$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

Theorem 3.8 turns that into a wall-clock improvement, where $c$ is the ratio of one draft forward pass to one target forward pass:

$$S = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$$

The paper is explicit that $c$ is a property of your hardware and implementation, not of the models: "unlike $\alpha$ which is an intrinsic property of the models and the task, the value of $c$ depends on the hardware configuration and software implementation details". In their own experiments a draft two orders of magnitude smaller than the target kept $c$ below 0.05.

Their Table 1 reports speedups at $c = \hat{c} = 0$, which is the free-draft case. That is a legitimate bound (n-gram drafts are close to it, since evaluation is a table lookup), but it is also the most optimistic column, and it is the one that gets copied.

The code below computes the formula and checks it against the six published rows, then prices the same pairs at two realistic draft costs.

```python
# Leviathan, Kalman & Matias (2023), Theorem 3.8:
#   speedup = (1 - a**(g+1)) / ((1 - a) * (g*c + 1))
# a = acceptance rate, g = draft length, c = draft cost / target cost
def speedup(a, g, c):
    return (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))

# Rows from Table 1 of arXiv:2211.17192 (c = 0 assumed in that table)
published = [
    (0.6, 2, 1.96), (0.7, 3, 2.53), (0.8, 2, 2.44),
    (0.8, 5, 3.69), (0.9, 2, 2.71), (0.9, 10, 6.86),
]

print(f"{'alpha':>5} {'gamma':>5} {'published':>9} {'formula':>8} {'delta':>6}")
worst = 0.0
for a, g, want in published:
    got = speedup(a, g, 0.0)
    worst = max(worst, abs(got - want))
    print(f"{a:>5.1f} {g:>5d} {want:>8.2f}x {got:>7.2f}x {got-want:>+6.2f}")

print()
print(f"largest gap vs published table: {worst:.2f}x")

# what the same table looks like when the draft is not free
print()
print("same (alpha, gamma) pairs at three draft-cost ratios")
print(f"{'alpha':>5} {'gamma':>5} {'c=0.00':>7} {'c=0.05':>7} {'c=0.20':>7}")
for a, g, _ in published:
    row = " ".join(f"{speedup(a, g, c):>6.2f}x" for c in (0.0, 0.05, 0.20))
    print(f"{a:>5.1f} {g:>5d} {row}")

# the 1/(1-a) ceiling people quote
print()
for a in (0.6, 0.8, 0.9):
    print(f"alpha={a:.1f}: asymptotic ceiling 1/(1-alpha) = {1/(1-a):.2f}x")
```

```
alpha gamma published  formula  delta
  0.6     2     1.96x    1.96x  +0.00
  0.7     3     2.53x    2.53x  +0.00
  0.8     2     2.44x    2.44x  +0.00
  0.8     5     3.69x    3.69x  -0.00
  0.9     2     2.71x    2.71x  -0.00
  0.9    10     6.86x    6.86x  +0.00

largest gap vs published table: 0.00x

same (alpha, gamma) pairs at three draft-cost ratios
alpha gamma  c=0.00  c=0.05  c=0.20
  0.6     2   1.96x   1.78x   1.40x
  0.7     3   2.53x   2.20x   1.58x
  0.8     2   2.44x   2.22x   1.74x
  0.8     5   3.69x   2.95x   1.84x
  0.9     2   2.71x   2.46x   1.94x
  0.9    10   6.86x   4.57x   2.29x

alpha=0.6: asymptotic ceiling 1/(1-alpha) = 2.50x
alpha=0.8: asymptotic ceiling 1/(1-alpha) = 5.00x
alpha=0.9: asymptotic ceiling 1/(1-alpha) = 10.00x
```

The formula reproduces the table to two decimals, which is the useful part: you can now move $c$ without re-deriving anything. The sixth row is the one worth staring at. At a free draft, $\alpha = 0.9$ with a ten-token draft buys 6.86x. At $c = 0.05$ it buys 4.57x, and at $c = 0.20$ it buys 2.29x. Same acceptance, same draft length, one third of the headline. If a vendor quotes the free-draft column while shipping a draft model that costs a fifth of the target, the gap is arithmetic rather than dishonesty.

## The rejection rule you cannot skip

Speculation is safe because it preserves the target distribution exactly, and that guarantee comes from one detail: on rejection, you resample from the residual $\text{norm}(\max(0, p-q))$, not from $p$. The residual is the part of the target's probability that the draft failed to cover, and because $\sum \max(0, p-q) = 1-\alpha$, it carries exactly the mass that rejection throws away.

Dropping the renormalisation, or resampling from $p$ instead, is a small-looking edit that silently changes what your model outputs. The block below runs both rules 200,000 times against the same distribution pair and measures total variation distance from the target.

```python
import random

# target and draft distributions over a 5-token vocabulary
p = [0.40, 0.25, 0.15, 0.12, 0.08]
q = [0.15, 0.15, 0.25, 0.25, 0.20]


def draw(dist, rng):
    """Sample an index in proportion to dist (no normalisation assumed)."""
    total = sum(dist)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(dist):
        acc += w
        if r < acc:
            return i
    return len(dist) - 1


def exact_step(rng):
    """Leviathan Algorithm 1 / Chen Algorithm 2."""
    x = draw(q, rng)
    if rng.random() < min(1.0, p[x] / q[x]):
        return x, True
    residual = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    return draw(residual, rng), False


def naive_step(rng):
    """The tempting shortcut: on rejection, resample from p instead of the residual."""
    x = draw(q, rng)
    if rng.random() < min(1.0, p[x] / q[x]):
        return x, True
    return draw(p, rng), False


def tv(emp, ref):
    return 0.5 * sum(abs(e - r) for e, r in zip(emp, ref))


N = 200_000
for name, step in (("exact rule", exact_step), ("naive resample-from-p", naive_step)):
    rng = random.Random(20261123)
    counts = [0] * len(p)
    accepted = 0
    for _ in range(N):
        tok, ok = step(rng)
        counts[tok] += 1
        accepted += ok
    emp = [c / N for c in counts]
    print(f"{name:24s} acceptance={accepted/N:.3f}  TV vs target={tv(emp, p):.4f}")
    print("   empirical:", " ".join(f"{v:.3f}" for v in emp))
    print("   target   :", " ".join(f"{v:.3f}" for v in p))

print()
print(f"E[min(p,q)] analytic acceptance = {sum(min(a,b) for a,b in zip(p,q)):.3f}")
print(f"rejection mass 1 - alpha        = {1 - sum(min(a,b) for a,b in zip(p,q)):.3f}")
```

```
exact rule               acceptance=0.651  TV vs target=0.0018
   empirical: 0.400 0.248 0.151 0.120 0.080
   target   : 0.400 0.250 0.150 0.120 0.080
naive resample-from-p    acceptance=0.651  TV vs target=0.1230
   empirical: 0.291 0.236 0.203 0.162 0.108
   target   : 0.400 0.250 0.150 0.120 0.080

E[min(p,q)] analytic acceptance = 0.650
rejection mass 1 - alpha        = 0.350
```

The exact rule lands 0.0018 from the target, which is sampling noise at 200,000 draws. The shortcut is 0.1230 away, and the drift is not random: it over-produces the tokens the draft liked (index 2 goes from 0.15 to 0.203) and under-produces the tokens only the target wanted (index 0 falls from 0.40 to 0.291). The measured acceptance rate is identical in both cases, 0.651, matching $E[\min(p,q)] = 0.650$. That is the trap: every throughput metric looks correct while the output distribution has quietly moved. If you implement speculation yourself, test the distribution, not the acceptance rate.

## Break-even is a curve, not a threshold

A rule of thumb circulating in deployment guides puts break-even near 60 to 70 percent acceptance. The formula does not have a single threshold, because break-even depends on $\gamma$ and $c$ as well. For each draft cost, here is the smallest $\alpha$ that clears 1.0x, plus the draft length that maximises the gain at three acceptance rates.

```python
def speedup(a, g, c):
    return (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))


def min_alpha(g, c, target=1.0):
    """Smallest acceptance rate that clears `target` speedup at this gamma and cost."""
    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        if speedup(mid, g, c) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def best_gamma(a, c, gmax=16):
    return max(range(1, gmax + 1), key=lambda g: speedup(a, g, c))


costs = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
print("acceptance needed to break even (S >= 1.00)")
print(f"{'draft cost c':>12} {'gamma=2':>9} {'gamma=4':>9} {'gamma=8':>9}")
for c in costs:
    row = " ".join(f"{min_alpha(g, c):>8.2f}" for g in (2, 4, 8))
    print(f"{c:>12.2f} {row}")

print()
print("best gamma and its speedup at three acceptance rates")
print(f"{'draft cost c':>12} {'alpha=0.5':>16} {'alpha=0.7':>16} {'alpha=0.85':>16}")
for c in costs:
    cells = []
    for a in (0.5, 0.7, 0.85):
        g = best_gamma(a, c)
        cells.append(f"g={g:<2d} {speedup(a,g,c):>4.2f}x")
    print(f"{c:>12.2f} {' '.join(f'{x:>16}' for x in cells)}")

print()
print("the same model, quoted as a single number (gamma=5, c=0.05):")
for a in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"  alpha={a:.2f} -> {speedup(a,5,0.05):.2f}x")
```

```
acceptance needed to break even (S >= 1.00)
draft cost c   gamma=2   gamma=4   gamma=8
        0.00     0.00     0.00     0.00
        0.02     0.04     0.07     0.14
        0.05     0.09     0.17     0.29
        0.10     0.17     0.29     0.44
        0.20     0.31     0.46     0.62
        0.30     0.42     0.57     0.72

best gamma and its speedup at three acceptance rates
draft cost c        alpha=0.5        alpha=0.7       alpha=0.85
        0.00       g=16 2.00x       g=16 3.33x       g=16 6.25x
        0.02       g=4  1.79x       g=8  2.76x       g=14 4.75x
        0.05       g=3  1.63x       g=6  2.35x       g=10 3.70x
        0.10       g=2  1.46x       g=4  1.98x       g=7  2.85x
        0.20       g=1  1.25x       g=3  1.58x       g=5  2.08x
        0.30       g=1  1.15x       g=2  1.37x       g=4  1.69x

the same model, quoted as a single number (gamma=5, c=0.05):
  alpha=0.50 -> 1.57x
  alpha=0.60 -> 1.91x
  alpha=0.70 -> 2.35x
  alpha=0.80 -> 2.95x
  alpha=0.90 -> 3.75x
```

Read the first table across a row: at $c = 0.05$ a four-token draft needs 17 percent acceptance to pay for itself, so the 60-70 percent folklore is off by a factor of four. At $c = 0.30$ an eight-token draft needs 72 percent, and a two-token draft still needs 42 percent. The folklore is the $c \approx 0.15$, $\gamma \approx 4$ corner generalised into a rule.

The second table is where the decision happens. A draft costing 20 percent of a target forward pass does not justify a long draft: at $\alpha = 0.7$ the best $\gamma$ is 3 and the gain is 1.58x, and every token past the third is paid for whether or not it is accepted.

## Two cost channels, and the one the formula hides

Theorem 3.8 is a wall-clock statement, and it rests on an assumption the paper states plainly: the analysis assumes "that we can run $\gamma + 1$ concurrent evaluations of $M_p$ in parallel". Verifying $\gamma+1$ tokens has to cost about what generating one cost, which is true when the target is memory-bandwidth bound, the normal regime for single-stream decoding. It stops being true once the batch is large enough that the target is compute bound instead.

The paper accounts for that separately. Definition 3.10 introduces $\hat{c}$, the ratio of arithmetic operations per token of the draft to the target, and Theorem 3.11 gives the expected increase in total operations:

$$\text{ops per token} = \frac{(\gamma \hat{c} + \gamma + 1)(1-\alpha)}{1-\alpha^{\gamma+1}}$$

At $\hat{c} = 0$ the numerator is $\gamma + 1$ target passes per accepted token, which is more than the one pass a baseline spends. Speculation buys wall-clock by spending arithmetic, and the two move in opposite directions.

The block below reproduces the paper's OPERATIONS column, then prices a ten-token draft across acceptance rates.

```python
# Table 1 of arXiv:2211.17192 reports two columns: SPEED and OPERATIONS.
# Operations per token (Definition 3.10, Theorem 3.11) with draft cost c_hat:
#   (gamma*c_hat + gamma + 1) * (1 - alpha) / (1 - alpha**(gamma+1))
def ops_per_token(a, g, c_hat=0.0):
    return (g * c_hat + g + 1) * (1 - a) / (1 - a ** (g + 1))


published = [(0.6, 2, 1.53), (0.7, 3, 1.58), (0.8, 2, 1.23),
             (0.8, 5, 1.63), (0.9, 2, 1.11), (0.9, 10, 1.60)]

print(f"{'alpha':>5} {'gamma':>5} {'published ops':>13} {'formula':>8}")
for a, g, want in published:
    print(f"{a:>5.1f} {g:>5d} {want:>12.2f}x {ops_per_token(a, g):>7.2f}x")

print()
print("latency gain and compute cost are different numbers (gamma=10)")
print(f"{'alpha':>5} {'wall-clock':>11} {'ops/token':>10} {'compute-bound thrpt':>20}")
for a in (0.5, 0.7, 0.8, 0.9):
    s = (1 - a ** 11) / ((1 - a) * (10 * 0.05 + 1))
    o = ops_per_token(a, 10, 0.05)
    print(f"{a:>5.1f} {s:>10.2f}x {o:>9.2f}x {1/o:>19.2f}x")

print()
print("the draft-cost ratio that cancels the gain entirely (wall-clock S = 1)")
for a, g in ((0.9, 10), (0.8, 5), (0.7, 3)):
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (1 - a ** (g + 1)) / ((1 - a) * (g * mid + 1)) >= 1:
            lo = mid
        else:
            hi = mid
    print(f"  alpha={a:.1f} gamma={g:>2d}: breaks even while c < {hi:.2f}")
```

```
alpha gamma published ops  formula
  0.6     2         1.53x    1.53x
  0.7     3         1.58x    1.58x
  0.8     2         1.23x    1.23x
  0.8     5         1.63x    1.63x
  0.9     2         1.11x    1.11x
  0.9    10         1.60x    1.60x

latency gain and compute cost are different numbers (gamma=10)
alpha  wall-clock  ops/token  compute-bound thrpt
  0.5       1.33x      5.75x                0.17x
  0.7       2.18x      3.52x                0.28x
  0.8       3.05x      2.52x                0.40x
  0.9       4.57x      1.68x                0.60x

the draft-cost ratio that cancels the gain entirely (wall-clock S = 1)
  alpha=0.9 gamma=10: breaks even while c < 0.59
  alpha=0.8 gamma= 5: breaks even while c < 0.54
  alpha=0.7 gamma= 3: breaks even while c < 0.51
```

The last table in that output matters most. At $\alpha = 0.9$ with a ten-token draft, latency improves 4.57x while arithmetic per token rises 1.68x. Both numbers are correct. They measure different things, and which one your service is judged on decides whether the configuration is good. An interactive chat or voice agent is latency bound, so a 4.5x cut in inter-token time is the whole product argument. An offline batch summariser at batch 32 is compute bound, and the same flags spend capital: 1.68x the arithmetic for the same tokens means fewer tokens per GPU-hour, so the throughput move is 0.60x.

The same block corrects a natural misreading of the section above. Corollary 3.9 of the paper says that if $\alpha > c$ there is always a $\gamma$ that improves wall-clock, with the improvement at least $\frac{1+\alpha}{1+c}$. The measured thresholds agree: at $\alpha = 0.9$ a ten-token draft keeps paying until the draft costs 0.59 of a target pass, and at $\alpha = 0.7$ a three-token draft holds until 0.51. Keeping the draft two orders of magnitude smaller is a safe default rather than a requirement. What matters is the pair.

The paper's own Table 2 is the cleanest version of this argument, because it shows a better draft losing. On T5-XXL with English to German at temperature 0, a T5-small draft accepted 0.75 of its tokens and delivered 3.4x. A T5-large draft accepted 0.82, higher, and delivered 1.7x. The stronger draft cost more per pass than its extra acceptance returned. The same table shows acceptance falling from 0.75 to 0.62 at temperature 1, which is the reminder that $\alpha$ is a property of the task and the sampling settings, not a constant you look up.

That temperature effect is worth planning around. A code assistant decoding near-greedy has the friendliest acceptance of any workload, because the target's next-token distribution is sharp and a draft can match a sharp distribution with one confident guess. A chat product sampling at temperature 0.8 to 1.0 widens that distribution, and the same draft and the same prompt will accept less often, so a configuration tuned on a benchmark suite at temperature 0 can underperform on the endpoint it was chosen for. If your product exposes a temperature slider, the honest way to size the draft is to measure at the top of its range and accept that the low end is a bonus.


## Measure acceptance on your own text

The number that decides all of this is $\alpha$, and $\alpha$ is workload-specific. A cheap way to see that without training anything is the lookup drafter that llama.cpp ships as `ngram-simple`: match the last $K$ tokens against history, offer the stored continuation as a draft, and skip drafting when the key has never been seen. The documentation describes it as requiring "no additional model but rely on patterns that have already appeared in the generated text", with source-code rewriting offered as the example use.

Two pinned corpora, both downloaded once and verified by hash:

`curl -sL -o big.txt https://norvig.com/big.txt` (sha256 `fa066c7d40f0f201ac4144e652aa62430e58a6b3805ec70650f678da5804e87b`)

`for f in argparse textwrap difflib random; do curl -sL -o c_$f.py https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/$f.py; done && cat c_argparse.py c_textwrap.py c_difflib.py c_random.py > code_corpus.py` (sha256 `4c903ac4d3a1065caf9cc81bcf26715166a942c6d606e7535eb7e492e136cf5f`)

```python
import re
from collections import defaultdict, Counter

TOKEN = re.compile(r"[a-z']+|[0-9]+|[^\sa-z0-9]", re.I)
GMAX = 8
CAP = 250_000


def tokens(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return TOKEN.findall(fh.read())[:CAP]


def build_index(seq, k, per_key=32):
    index = defaultdict(list)
    for i in range(len(seq) - k - GMAX):
        index[tuple(seq[i:i + k])].append(i)
    return {key: v[-per_key:] for key, v in index.items()}


def draft_from(seq, k, hits):
    """Most frequent continuation after the key; latest occurrence breaks ties."""
    best = Counter(seq[h + k] for h in hits).most_common(1)[0][0]
    start = max(h for h in hits if seq[h + k] == best)
    return seq[start + k:start + k + GMAX]


def runs_for(seq, k):
    split = int(len(seq) * 0.9)
    index = build_index(seq[:split], k)
    runs = []
    for i in range(split, len(seq) - GMAX):
        hits = index.get(tuple(seq[i:i + k]))
        if not hits:
            runs.append(None)
            continue
        draft = draft_from(seq, k, hits)
        run = 0
        while run < len(draft) and seq[i + run] == draft[run]:
            run += 1
        runs.append(run)
    return runs


def speedup(runs, g, c):
    drafted = [r for r in runs if r is not None]
    produced = sum(min(r, g) + 1 for r in drafted) + (len(runs) - len(drafted))
    cost = len(drafted) * (g * c + 1) + (len(runs) - len(drafted))
    return produced / cost


for path, name in (("big.txt", "prose"), ("code_corpus.py", "python source")):
    seq = tokens(path)
    print(f"== {name}: {len(seq):,} tokens")
    print(f"{'K':>3} {'drafts':>7} {'alpha1':>7} {'mean acc':>9} {'S g=4 c=.02':>12} {'S g=4 c=.10':>12}")
    for k in (1, 2, 3, 4):
        runs = runs_for(seq, k)
        drafted = [r for r in runs if r is not None]
        a1 = sum(1 for r in drafted if r >= 1) / len(drafted)
        mean_acc = sum(min(r, 4) for r in drafted) / len(drafted)
        print(f"{k:>3} {len(drafted)/len(runs):>6.1%} {a1:>7.3f} {mean_acc:>9.2f} "
              f"{speedup(runs,4,0.02):>11.2f}x {speedup(runs,4,0.10):>11.2f}x")
    print()
```

```
== prose: 250,000 tokens
  K  drafts  alpha1  mean acc  S g=4 c=.02  S g=4 c=.10
  1  94.8%   0.018      0.02        0.95x        0.74x
  2  59.4%   0.017      0.02        0.96x        0.82x
  3  22.4%   0.046      0.05        0.99x        0.93x
  4   7.3%   0.031      0.03        1.00x        0.97x

== python source: 54,823 tokens
  K  drafts  alpha1  mean acc  S g=4 c=.02  S g=4 c=.10
  1  88.7%   0.192      0.20        1.10x        0.87x
  2  54.3%   0.211      0.21        1.07x        0.92x
  3  28.2%   0.303      0.31        1.06x        0.98x
  4  16.1%   0.531      0.55        1.07x        1.02x
```

Two things stand out. Prose is a poor match for a lookup drafter: the best cell accepts 4.6 percent of first tokens, and the end-to-end ratio is 0.99x at $c = 0.02$ and 0.93x at $c = 0.10$, which is slower than just running the target. Python source is better, reaching 0.531 acceptance with a four-token key, but only 16 percent of positions had any draft at all, so the end-to-end gain tops out at 1.07x.

Most throughput claims leave that last number out. Acceptance measured over drafted positions is not the acceptance your server will see. The quantity that enters Theorem 3.8 is the acceptance averaged over every position, including the ones where the lookup missed and the target decoded alone. Multiply the two and a 53 percent acceptance rate becomes an 8 percent workload gain.

Two caveats about what this measures. The tokenizer here is a regex over words and punctuation rather than a BPE vocabulary, so the absolute numbers will move if you re-run against real token ids; what transfers is the structure, which is coverage times acceptance, with decay by position. Pointing it at your own data is a one-line change: replace the `TOKEN` pattern and the `tokens()` function with `tokenizer.encode(text)` in the loop. The corpora are also capped at 250,000 tokens with a 90/10 split so the whole run finishes in about a second, which means the held-out slice is one author's prose and four standard-library modules, not representative traffic.

## Where the geometric model breaks

Equation 1 also assumes that acceptance is constant across draft positions. That is an assumption about the pair of distributions, and it is testable. The block below measures the conditional acceptance of each draft position separately on the Python corpus, and compares the measured mean accepted length with what a constant-$\alpha$ model predicts.

```python
import re
from collections import defaultdict, Counter

TOKEN = re.compile(r"[a-z']+|[0-9]+|[^\sa-z0-9]", re.I)
GMAX = 8


def tokens(path, cap=250_000):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return TOKEN.findall(fh.read())[:cap]


def build_index(seq, k, per_key=32):
    index = defaultdict(list)
    for i in range(len(seq) - k - GMAX):
        index[tuple(seq[i:i + k])].append(i)
    return {key: v[-per_key:] for key, v in index.items()}


def draft_from(seq, k, hits):
    best = Counter(seq[h + k] for h in hits).most_common(1)[0][0]
    start = max(h for h in hits if seq[h + k] == best)
    return seq[start + k:start + k + GMAX]


seq = tokens("code_corpus.py")
K = 4
split = int(len(seq) * 0.9)
index = build_index(seq[:split], K)

runs = []
for i in range(split, len(seq) - GMAX):
    hits = index.get(tuple(seq[i:i + K]))
    if not hits:
        runs.append(None)
        continue
    draft = draft_from(seq, K, hits)
    run = 0
    while run < len(draft) and seq[i + run] == draft[run]:
        run += 1
    runs.append(run)

drafted = [r for r in runs if r is not None]
a1 = sum(1 for r in drafted if r >= 1) / len(drafted)
print(f"python source, K={K}: {len(drafted):,} drafted positions, alpha1={a1:.3f}")
print(f"{'gamma':>5} {'mean acc':>9} {'geometric':>10} {'S c=.02':>8} {'S c=.10':>8}")
for g in (1, 2, 4, 8):
    mean_acc = sum(min(r, g) for r in drafted) / len(drafted)
    geo = sum(a1 ** j for j in range(1, g + 1))
    produced = sum(min(r, g) + 1 for r in drafted) + (len(runs) - len(drafted))
    costs = {c: produced / (len(drafted) * (g * c + 1) + (len(runs) - len(drafted)))
             for c in (0.02, 0.10)}
    print(f"{g:>5} {mean_acc:>9.2f} {geo:>10.2f} {costs[0.02]:>7.2f}x {costs[0.10]:>7.2f}x")

print()
print("position-by-position, the way the formula assumes it never changes")
for j in (1, 2, 3, 4):
    prev = sum(1 for r in drafted if r >= j - 1)
    cur = sum(1 for r in drafted if r >= j)
    if prev == 0:
        print(f"  P(accept token {j}) = no surviving runs, undefined")
        continue
    print(f"  P(accept token {j} | tokens 1..{j-1} accepted) = {cur/prev:.3f}"
          f"   (formula assumes {a1:.3f})")

print()
print(f"positions with no draft at all: {(len(runs)-len(drafted))/len(runs):.1%}")
```

```
python source, K=4: 884 drafted positions, alpha1=0.531
gamma  mean acc  geometric  S c=.02  S c=.10
    1      0.53       0.53    1.08x    1.07x
    2      0.55       0.81    1.08x    1.05x
    4      0.55       1.04    1.07x    1.02x
    8      0.55       1.12    1.06x    0.96x

position-by-position, the way the formula assumes it never changes
  P(accept token 1 | tokens 1..0 accepted) = 0.531   (formula assumes 0.531)
  P(accept token 2 | tokens 1..1 accepted) = 0.028   (formula assumes 0.531)
  P(accept token 3 | tokens 1..2 accepted) = 0.000   (formula assumes 0.531)
  P(accept token 4) = no surviving runs, undefined

positions with no draft at all: 83.9%
```

The drafter accepts its first token 53 percent of the time and its second 2.8 percent of the time. The constant-$\alpha$ model predicts 0.81 accepted tokens at $\gamma = 2$; the drafter delivers 0.55, and every extra drafted token past the first is dead weight: mean accepted length is flat from $\gamma = 2$ onward, while the cost term climbs. At $c = 0.10$ the model switches from 1.05x at $\gamma = 2$ to 0.96x at $\gamma = 8$, which is the crossing the arithmetic predicts once the real decay is in the numbers instead of the assumption.

One caution about generalising. This decay is measured for a drafter that emits a single candidate sequence from a lookup table, so token two is conditioned on an arbitrary stored context rather than on a distribution. A trained draft behaves differently, because it reads the target's own hidden states: llama.cpp's documentation says EAGLE-3 "uses a small draft model that reads the target model's hidden states to predict the next tokens, so it reaches higher acceptance than a standalone draft model of the same size." The block-diffusion variants in the same document go further, emitting a whole block per pass and adding a semi-autoregressive Markov head to recover some of the left-to-right signal that pure block diffusion loses. The decay is gentler there, not absent, and the point survives: measure acceptance per position, because a single $\alpha$ summarises a curve that is steep at the front.

## How to use this before you buy a draft

The order of operations is cheap and it settles the question before any training run:

1. **Measure coverage and acceptance on your traffic.** Score the draft against 200 to 500 real generations from your own system. Report both numbers: the fraction of positions where a draft exists, and the acceptance rate over those positions.
2. **Estimate $c$ on your hardware.** Time the target alone at your batch size and context length, time the draft alone under the same conditions, and divide. Three things bias that number: a draft running concurrently with the target on the same GPU contends for the same bandwidth, so a draft timed in isolation looks cheaper than it is; the verification pass handles $\gamma+1$ tokens, so its cost sits closer to one target pass than to $\gamma+1$ of them, which is the assumption Theorem 3.8 encodes; and per-pass fixed costs such as sampling and kernel launch get charged once, which flatters long drafts at small batch.
3. **Pick $\gamma$ from the table, not from a blog post.** With $\alpha = 0.6$ and $c = 0.1$, the fourth-token addition is already losing money; `--spec-draft-n-max` in llama.cpp is the flag, and `--spec-draft-p-min` truncates a draft at the first step whose probability falls below a threshold, which is the same idea enforced per position.
4. **Check the end-to-end ratio, not the drafted-position ratio.** Coverage is the multiplier the formula hides. In the measurements above, 0.53 acceptance became 1.07x once the 84 percent of positions without a draft were counted.
5. **Do not expect batching to rescue it.** A 2025 analysis of batched speculation finds that ragged acceptance lengths desynchronise position ids, attention masks and KV state, and that the realignment costs up to 40 percent of computation. Their conclusion is the practical one: "batch speculation amplifies per-sequence gains; it cannot create them."

One more choice sits above the arithmetic: which drafter. A lookup drafter needs no training and no checkpoint, which is why it ships in llama.cpp as a flag, but the measurements above show what that buys on free-running text. A trained draft costs a conversion step and a checkpoint tied to a specific target model, and that cost is only repaid when acceptance is materially higher on your data. The cheap test is to run the coverage and acceptance measurement against both candidates: if the lookup drafter already clears the break-even $\alpha$ at the $\gamma$ your $c$ allows, a trained draft has to beat that number by enough to justify the extra artifact and the extra ops.

Worked end to end on a hypothetical pair, so the arithmetic is easy to transplant. Suppose your timings give $c = 0.08$ and your logs give $\alpha = 0.65$ on 90 percent of positions. The break-even table puts the optimum at $\gamma = 4$, the formula gives 1.91x on drafted positions, and coverage turns that into about 1.82x on the workload. The operations column says the same configuration spends 2.11x the arithmetic per token, so it is a latency win on a bandwidth-bound box and a capacity loss on a saturated one. Now change one assumption, a messier workload where $c = 0.20$ and $\alpha = 0.45$: $\gamma = 2$ gives 1.18x, $\gamma = 3$ gives 1.09x, and $\gamma = 4$ gives 0.99x. Same code, same flags, a configuration that now costs more than it saves. That is the entire reason for measuring before switching it on.

A useful sanity check on any published number: ask which of the three quantities is held fixed. A 2 to 4 times claim with Chinchilla, as in the DeepMind paper, was measured with a draft two orders of magnitude smaller than the target, at an acceptance rate the authors could measure but not publish per workload. That claim is true for their setup. Whether it is true for yours is a measurement, and the first two blocks above are the harness for it.

## Key takeaways

| Question | What the arithmetic says |
|---|---|
| Should I quote the published speedup? | Only with $c$ and $\alpha$ attached. The table assumes a free draft, and the same row can lose a third of its gain at $c = 0.2$ |
| Is there an acceptance threshold? | No. Break-even is 17 percent at $\gamma = 4$, $c = 0.05$, and 72 percent at $\gamma = 8$, $c = 0.30$ |
| Which draft length? | The one your $c$ can pay for. At $c = 0.2$ the best $\gamma$ at $\alpha = 0.7$ is 3, worth 1.58x |
| What breaks the output distribution? | Dropping the residual resample. Measured TV distance rose from 0.0018 to 0.1230 while acceptance stayed at 0.651 |
| What breaks the speedup model? | Constant-$\alpha$ across positions. Measured second-token acceptance was 0.028 against a 0.531 assumption |

Speculation is an arithmetic trade with three inputs, and only one of them ($\gamma$) is a free parameter. $\alpha$ comes from your draft and your data, $c$ from your hardware, and both are measurable in an afternoon with the standard library.

## References

- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) (Equation 1, Theorem 3.8, Table 1)
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) (2 to 2.5x with Chinchilla, rejection-sampling algorithm)
- llama.cpp, *Speculative Decoding* documentation — [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) (`ngram-simple`, `--spec-draft-n-max`, `--spec-draft-p-min`, EAGLE-3 hidden-state detail)
- *Batch Speculative Decoding Done Right* — [arXiv:2510.22876](https://arxiv.org/html/2510.22876v3) (ragged acceptance and the realignment cost)
- Norvig, public domain text corpus — [norvig.com/big.txt](https://norvig.com/big.txt)
- CPython v3.12.3 `argparse.py`, `textwrap.py`, `difflib.py`, `random.py` — [github.com/python/cpython](https://github.com/python/cpython/tree/v3.12.3/Lib)

## Related posts

- [Cascade Routing: Pay for the Big Model Only on the Queries That Need It](/posts/model-cascade-escalation-budget/) — the same cost-ratio arithmetic applied to model choice
- [Report the Horizon You Ship: A Seed Sweep Harness for Rollout Evaluations](/posts/rollout-horizon-seed-sweep-harness/) — measuring the parameter your claim depends on, before you publish
- [Your Encoding Saw Its Own Label: A Four-Number Audit for Target Encoding](/posts/target-encoding-leakage-audit/) — a small harness that makes a hidden assumption visible
- [The Chunk Boundary Is the Bug: Measuring What a Split Costs Your Retrieval](/posts/chunk-boundary-damage-rag/) — measure the damage, then choose the setting
