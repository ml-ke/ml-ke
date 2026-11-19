---
title: "Report the Horizon You Ship: A Seed Sweep Harness for Rollout Evaluations"
date: 2026-11-19 00:00:00 +0300
categories: [ML Ops, Machine Learning]
tags: [evaluation, learned simulators, rollout, bootstrap, statistics, reproducibility]
math: true
image:
  path: /assets/img/cover-rollout-horizon-seed-sweep-harness.webp
  alt: Two error curves crossing above eight faint per-seed curves on a horizon axis
---

> **The short version**
> A learned simulator has one number on its model card and a whole curve behind it. This post builds a small harness that sweeps horizons and seeds, pairs the two models on identical initial states, bootstraps the paired difference, and returns a ship or no-ship verdict instead of a ranking.
{: .prompt-info }

Last week's post on the [unified particle simulator](/posts/unified-particle-simulator-six-materials/) ended with a single run: two fitted step models, one seed, and a ranking that reversed between 200 and 800 rollout steps. That result was real, and it was also the weakest kind of evidence, because one seed cannot tell you whether a reversal is a property of the two models or a property of the two fits you happened to draw.

This post fixes the measurement. The technique is not specific to physics: any model you evaluate by feeding its own predictions back in, from a learned simulator to a multi-step RAG pipeline to a one-week-ahead demand forecast, has the same problem. One-step error describes the map. Your users live on the curve.

## What a one-step metric cannot tell you

The Well benchmark makes the gap concrete in its own evaluation section. Looking at a turbulent radiative layer, the authors report that on one-step performance "it appears that CNextU-net has a sizable advantage, but when we look at longer time horizons, this advantage quickly dissipates and all models apart from the original U-net become largely interchangeable" ([arXiv:2412.00568](https://arxiv.org/abs/2412.00568)). The benchmark also flags a subtlety that catches people comparing papers: long rollouts always start at the beginning of the simulation, while one-step evaluation uses sliding windows sampled from ground truth, so the two numbers describe different tasks.

The same paper normalises its error so that predicting the mean value of a field scores exactly 1.0. That convention is worth copying, because a raw error in model units cannot be compared across systems, and it tells you immediately whether a model beats the mean predictor by 20 percent or by four orders of magnitude.

A 2026 benchmark of physics foundation models goes after the aggregation step directly. Its authors argue that evaluations "often collapse performance into a single average score under a fixed training distribution", which makes it hard to tell whether a model learned transferable dynamics or one setting. Their benchmark spans 8 physical dynamics, 3 training mixtures, 25 test regimes and 5 architectures at 4 variants each, 60,000 measurements in total, and the conclusion is that the models behave as conditional rather than universal generalists ([arXiv:2605.29283](https://arxiv.org/abs/2605.29283)).

There is also a cheap pre-screen. Semigroup error compares a model's direct prediction over an interval $s+t$ with the result of applying it over $s$ and then $t$. Across one-dimensional heat and Burgers dynamics with ConvNet and Fourier neural operator baselines, that diagnostic correlates with rollout degradation at a trajectory-level Spearman $\rho = 0.635$, 95% CI $[0.621, 0.649]$ ([arXiv:2605.26324](https://arxiv.org/abs/2605.26324)). The paper is explicit that the score works better as a diagnostic than as a training objective, since regularising on it had mixed effects.

The statistical half of the problem is well mapped in reinforcement learning. Point estimates drawn from a handful of runs disagreed with a proper statistical analysis on Atari 100k, which is why that work argues for interval estimates, interquartile means and performance profiles over bare means ([arXiv:2108.13264](https://arxiv.org/abs/2108.13264)). Reporting every run rather than the best one flips conclusions too, which is the argument for compute-budget curves ([arXiv:1909.03004](https://arxiv.org/abs/1909.03004)), and seed variance alone was enough to distort comparisons long before that ([arXiv:1709.06560](https://arxiv.org/abs/1709.06560)). In fine-tuning, the same effect shows up on GLUE and SuperGLUE as variance in aggregate scores plus instability in individual predictions across runs ([arXiv:2503.07329](https://arxiv.org/abs/2503.07329)).

Put those together and the evaluation target is not a number. It is a matrix: models down one axis, horizons down another, seeds filling the cells.

## Two tasks that share a name

Before building the matrix, separate two measurements that both get called rollout error. On a sliding window, the model is handed a true state, predicts one step, and is scored. On a rollout from the start, the model is handed a true state once and every subsequent input is its own output. The second task contains the first, and a model can be strong at one and useless at the other, because the rollout never gets corrected.

| Measurement | What it tells you | What it hides |
|---|---|---|
| One-step on ground-truth windows | How accurate the map is where the data lives | Compounding, since the input is always true |
| Rollout from the initial state | What a user sees, after the error feeds back | Which step first went wrong, unless you also store per-step error |
| Per-step error inside a rollout | When the trajectory diverges, and in which direction | Nothing, at the cost of storing a full trajectory per run |

The Well's own numbers illustrate the split: its one-step table and its longer-rollout table do not rank the same way, partly because the two experiments start from different places and are therefore different in difficulty. A harness that mixes the two in one table produces a controversy rather than a measurement.

This is also why the cache in the next section matters for correctness and not just speed. If the reference is recomputed with a different starting state for each model, you have quietly turned a paired comparison into an unpaired one, and the pairing is the part that removes the state-dependent difficulty.


## A system small enough to check by hand

The demonstration uses a damped Duffing oscillator, $x'' = -\delta x' - \alpha x - \beta x^3$, with $\delta = 0.10$, $\alpha = \beta = 1.0$, integrated by RK4 at $\Delta t = 0.005$. It is nonlinear enough that a linear one-step map is genuinely approximate and cheap enough that the reference trajectory costs nothing.

Each candidate model is a linear map from a four-feature state to the one-step increment, fitted by least squares on trajectories sampled from the reference. The second candidate sees the same data with a ridge penalty of 1.0, which is the kind of knob nobody reports and everybody turns.

{% raw %}
```python
import numpy as np

DT = 0.005
DELTA, ALPHA, BETA = 0.10, 1.00, 1.00   # x'' = -DELTA*x' - ALPHA*x - BETA*x**3


def deriv(s):
    x, v = s
    return np.array([v, -DELTA * v - ALPHA * x - BETA * x ** 3])


def truth(s0, steps):
    """RK4 reference at dt=0.005; the data-generating process."""
    s, out = np.array(s0, float), [np.array(s0, float)]
    for _ in range(steps):
        k1 = deriv(s); k2 = deriv(s + 0.5 * DT * k1)
        k3 = deriv(s + 0.5 * DT * k2); k4 = deriv(s + DT * k3)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(s.copy())
    return np.array(out)


def feats(s):
    x, v = s
    return np.array([x, v, x ** 3, 1.0])


def fit_map(seed, ridge=0.0, n_traj=8, steps=400):
    """Least-squares one-step map:  s[t+1] - s[t] = B @ feats(s[t])."""
    rng = np.random.default_rng(seed)
    Phi, D = [], []
    for _ in range(n_traj):
        traj = truth([rng.uniform(-2, 2), rng.uniform(-2, 2)], steps)
        for t in range(steps):
            Phi.append(feats(traj[t])); D.append(traj[t + 1] - traj[t])
    Phi, D = np.array(Phi), np.array(D)
    A = Phi.T @ Phi + ridge * np.eye(4)
    return np.linalg.solve(A, Phi.T @ D)


def rollout(B, s0, steps):
    s, out = np.array(s0, float), [np.array(s0, float)]
    for _ in range(steps):
        s = s + feats(s) @ B
        out.append(s.copy())
    return np.array(out)


def one_step_rmse(B, seed, n=300):
    """Held-out one-step error, the number a model card usually reports."""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        s = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2)])
        gt_next = truth(s, 1)[1]
        errs.append(np.linalg.norm(s + feats(s) @ B - gt_next))
    return float(np.sqrt(np.mean(np.square(errs))))


def rollout_rmse(B, s0, steps):
    return float(np.sqrt(np.mean((rollout(B, s0, steps) - truth(s0, steps)) ** 2)))


B_ls = fit_map(seed=0, ridge=0.0)
B_rg = fit_map(seed=0, ridge=1.0)
ICS = [[1.5, 0.0], [-1.8, 0.6], [0.9, -1.4], [2.1, 1.1], [-0.7, 2.0], [1.2, 1.6]]

print(f"{'model':8s} {'1-step':>10s} {'H=50':>10s} {'H=200':>10s} {'H=800':>10s} {'H=3200':>10s}")
for name, B in (("ls", B_ls), ("ridge", B_rg)):
    row = [one_step_rmse(B, 99)] + [np.mean([rollout_rmse(B, ic, h) for ic in ICS])
                                    for h in (50, 200, 800, 3200)]
    print(f"{name:8s} " + " ".join(f"{v:10.6f}" for v in row))
```
{% endraw %}

```text
model        1-step       H=50      H=200      H=800     H=3200
ls         0.000055   0.000823   0.002938   0.006335   0.030227
ridge      0.000055   0.000841   0.002896   0.006460   0.031186
```

The one-step column is the punchline. Both maps score 0.000055, identical to six decimals, so the metric a model card would print cannot separate them at all. The rollout columns separate them in three different directions: the plain fit wins at 50 steps, the ridge fit wins at 200, and the plain fit wins again at 800 and by 3 percent at 3,200.

That is one seed, and the middle reversal is the reason to distrust it. The next question is where the reversal comes from, because a ranking that moves with horizon is only actionable if you know the mechanism.

## Why rankings move at all

A one-step map is wrong in two distinguishable ways. It can be wrong the same way every time, which is bias: an additive push $b$ per step accumulates into an error that grows like $bH$. Or it can be wrong differently each time, which is scatter: zero-mean per-step noise $\sigma$ accumulates as a random walk, so the error grows like $\sigma\sqrt{H}$.

Those two rates cross when $bH = \sigma\sqrt{H}$, at

$$H^{*} = \left(\frac{\sigma}{b}\right)^{2}$$

which means a model with twenty times more scatter than bias looks worse at short horizons and better at long ones, and the crossover sits at a specific, computable step count. The mechanism is worth measuring once, because it explains reversals that have nothing to do with seed noise:

{% raw %}
```python
import numpy as np

DT = 0.005
DELTA, ALPHA, BETA = 0.10, 1.00, 1.00

def deriv(s):
    x, v = s
    return np.array([v, -DELTA * v - ALPHA * x - BETA * x ** 3])

def truth(s0, steps):
    s, out = np.array(s0, float), [np.array(s0, float)]
    for _ in range(steps):
        k1 = deriv(s); k2 = deriv(s + 0.5 * DT * k1)
        k3 = deriv(s + 0.5 * DT * k2); k4 = deriv(s + DT * k3)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(s.copy())
    return np.array(out)

def biased(s0, steps, b=(1e-5, 0.0)):
    """Exact integrator plus a constant per-step push: error has a mean."""
    gt = truth(s0, steps)
    return gt + np.outer(np.arange(steps + 1), np.array(b))

def noisy(s0, steps, sigma=2e-4, seed=7):
    """Exact integrator plus zero-mean per-step noise: error is a random walk."""
    gt = truth(s0, steps)
    rng = np.random.default_rng(seed)
    return gt + np.cumsum(rng.normal(0.0, sigma, size=(steps + 1, 2)), axis=0)

def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

s0 = [1.5, 0.0]
print(f"{'H':>6s} {'bias':>10s} {'noise':>10s} {'noise/bias':>11s}")
for H in (25, 100, 400, 1600, 3200):
    gt = truth(s0, H)
    eb = rmse(biased(s0, H), gt)
    en = np.mean([rmse(noisy(s0, H, seed=s), gt) for s in range(5)])
    print(f"{H:6d} {eb:10.6f} {en:10.6f} {en/eb:11.3f}")

# growth exponents between H=100 and H=3200 (log-log slope)
gt1, gt2 = truth(s0, 100), truth(s0, 3200)
eb = (rmse(biased(s0, 3200), gt2) / rmse(biased(s0, 100), gt1))
en = (np.mean([rmse(noisy(s0, 3200, seed=s), gt2) for s in range(5)]) /
      np.mean([rmse(noisy(s0, 100, seed=s), gt1) for s in range(5)]))
Hratio = 32.0
print(f"\nslope bias  = {np.log(eb)/np.log(Hratio):.3f}")
print(f"slope noise = {np.log(en)/np.log(Hratio):.3f}")
print(f"predicted crossover H* = (sigma/b)^2 = {(2e-4/1e-5)**2:.0f}")
```
{% endraw %}

```text
     H       bias      noise  noise/bias
    25   0.000103   0.000548       5.316
   100   0.000409   0.001085       2.650
   400   0.001634   0.003015       1.845
  1600   0.006533   0.005395       0.826
  3200   0.013065   0.007203       0.551

slope bias  = 0.999
slope noise = 0.546
predicted crossover H* = (sigma/b)^2 = 400
```

The measured slopes are 0.999 for the biased model and 0.546 for the noisy one, against the predicted 1 and 0.5, and the measured crossover falls between 400 and 1,600 steps where the formula predicted 400. The noisy model is 5.3 times worse at 25 steps and 45 percent better at 3,200.

So before any seed enters the picture, two one-step maps with equal one-step error can rank in either order depending on how far you roll them out. The harness now needs to separate that structural effect from fit-to-fit luck.

## The harness: seeds across, horizons down

Three design choices do most of the work. Every model is evaluated on the same set of initial conditions, so the per-state difficulty cancels in a paired difference. Every fit is repeated across eight data seeds, so the cell variance is visible rather than hidden behind a mean. And the comparison is bootstrapped, because eight seeds is not a sample from which you can read a standard deviation by eye.

The reference trajectory is cached. Without the cache every seed recomputes the same RK4 path for every model, which dominates the runtime and produces a script slow enough that nobody runs it twice.

{% raw %}
```python
import numpy as np

DT = 0.005
DELTA, ALPHA, BETA = 0.10, 1.00, 1.00
HZ = (50, 200, 800, 3200)
SEEDS = tuple(range(8))
ICS = [[1.5, 0.0], [-1.8, 0.6], [0.9, -1.4], [2.1, 1.1], [-0.7, 2.0], [1.2, 1.6]]

def deriv(s):
    x, v = s
    return np.array([v, -DELTA * v - ALPHA * x - BETA * x ** 3])

_CACHE = {}

def truth(s0, steps):
    """RK4 reference. Cached: every model and seed reuses the same trajectory."""
    key = (tuple(np.round(s0, 12)), steps)
    if key in _CACHE:
        return _CACHE[key]
    s, out = np.array(s0, float), [np.array(s0, float)]
    for _ in range(steps):
        k1 = deriv(s); k2 = deriv(s + 0.5 * DT * k1)
        k3 = deriv(s + 0.5 * DT * k2); k4 = deriv(s + DT * k3)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(s.copy())
    _CACHE[key] = np.array(out)
    return _CACHE[key]

def feats(s):
    x, v = s
    return np.array([x, v, x ** 3, 1.0])

def fit_map(seed, ridge=0.0, n_traj=8, steps=400):
    rng = np.random.default_rng(seed)
    Phi, D = [], []
    for _ in range(n_traj):
        traj = truth([rng.uniform(-2, 2), rng.uniform(-2, 2)], steps)
        for t in range(steps):
            Phi.append(feats(traj[t])); D.append(traj[t + 1] - traj[t])
    Phi, D = np.array(Phi), np.array(D)
    return np.linalg.solve(Phi.T @ Phi + ridge * np.eye(4), Phi.T @ D)

def rollout(B, s0, steps):
    s, out = np.array(s0, float), [np.array(s0, float)]
    for _ in range(steps):
        s = s + feats(s) @ B
        out.append(s.copy())
    return np.array(out)

def rmse(B, s0, steps):
    return float(np.sqrt(np.mean((rollout(B, s0, steps) - truth(s0, steps)) ** 2)))

def sweep(make_B):
    """err[seed, horizon] = mean rollout RMSE over the shared eval set."""
    return np.array([[np.mean([rmse(make_B(s), ic, h) for ic in ICS]) for h in HZ]
                     for s in SEEDS])

def bootstrap_ci(d, n=10000, alpha=0.05, seed=20261119):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))

def report(name_a, a, name_b, b):
    print(f"\n{name_a} vs {name_b}  (negative diff = {name_b} lower error)")
    print(f"{'H':>6s} {name_a:>10s} {name_b:>10s} {'spread_a':>9s} {'mean_diff':>11s} "
          f"{'95% CI':>20s} {'wins_b':>7s}")
    verdicts = []
    for hi, h in enumerate(HZ):
        ca, cb = a[:, hi], b[:, hi]
        d = cb - ca
        lo, hi_ci = bootstrap_ci(d)
        wins = int((d < 0).sum())
        excludes_zero = (lo > 0) or (hi_ci < 0)
        verdicts.append(excludes_zero)
        print(f"{h:6d} {ca.mean():10.6f} {cb.mean():10.6f} "
              f"{(ca.max()-ca.min())/ca.mean()*100:8.1f}% {d.mean():+11.6f} "
              f"[{lo:+.6f},{hi_ci:+.6f}] {wins:5d}/8"
              f"{' *' if excludes_zero else ''}")
    return verdicts

a = sweep(lambda s: fit_map(s, ridge=0.0))
b = sweep(lambda s: fit_map(s, ridge=1.0))
c = sweep(lambda s: fit_map(s, ridge=0.0, n_traj=2))
d_ = sweep(lambda s: fit_map(s, ridge=0.0, n_traj=64))

report("ls8", a, "ridge", b)
report("ls2", c, "ls64", d_)


def verdict(a, b, name_a, name_b, ship_h=3200):
    """Decide whether b beats a, per horizon, then across the shipped horizon."""
    rows = []
    for hi, h in enumerate(HZ):
        d = b[:, hi] - a[:, hi]
        lo, hi_ci = bootstrap_ci(d)
        reliable = (lo > 0) or (hi_ci < 0)
        better = name_b if d.mean() < 0 else name_a
        rows.append((h, reliable, better, int((d < 0).sum())))
    reliable_rows = [r for r in rows if r[1]]
    signs = {r[2] for r in reliable_rows}
    ship_row = [r for r in rows if r[0] == ship_h][0]
    if not reliable_rows:
        decision = "NO-DIFFERENCE"
    elif len(signs) > 1:
        decision = "REVERSAL"
    elif not ship_row[1]:
        decision = "NOT-SHOWN-AT-SHIP-HORIZON"
    else:
        decision = f"{ship_row[2]}-WINS"
    print(f"\n{name_a} vs {name_b}: reliable at {sum(r[1] for r in rows)}/4 horizons; "
          f"winners among reliable {sorted(signs) or ['-']}; decision: {decision}")
    return decision


print("\nscorecard")
verdict(a, b, "ls8", "ridge")
verdict(c, d_, "ls2", "ls64")
```
{% endraw %}

```text

ls8 vs ridge  (negative diff = ridge lower error)
     H        ls8      ridge  spread_a   mean_diff               95% CI  wins_b
    50   0.000816   0.000830      8.6%   +0.000014 [+0.000010,+0.000018]     0/8 *
   200   0.002684   0.002673     21.9%   -0.000011 [-0.000025,+0.000008]     7/8
   800   0.006380   0.006478     29.5%   +0.000098 [-0.000036,+0.000223]     3/8
  3200   0.028145   0.029023     44.8%   +0.000879 [+0.000576,+0.001161]     0/8 *

ls2 vs ls64  (negative diff = ls64 lower error)
     H        ls2       ls64  spread_a   mean_diff               95% CI  wins_b
    50   0.000869   0.000803     28.6%   -0.000066 [-0.000120,-0.000026]     7/8 *
   200   0.002881   0.002573     36.1%   -0.000308 [-0.000538,-0.000110]     7/8 *
   800   0.007607   0.005760    103.4%   -0.001847 [-0.003607,-0.000602]     7/8 *
  3200   0.033068   0.025687    114.4%   -0.007380 [-0.016067,-0.000380]     6/8 *

scorecard

ls8 vs ridge: reliable at 2/4 horizons; winners among reliable ['ls8']; decision: ls8-WINS

ls2 vs ls64: reliable at 4/4 horizons; winners among reliable ['ls64']; decision: ls64-WINS
```

Two of those rows deserve a closer read, because they are the two mistakes a table like this invites.

The `spread_a` column is the marginal spread across the eight fits, min to max, as a percentage of the mean. It grows with horizon: 8.6 percent at 50 steps for the plain fit, 21.9 at 200, 29.5 at 800 and 44.8 at 3,200. For the two-trajectory fit it reaches 103 and then 114 percent. Seed noise is not a fixed property of a model. It is a property of a model at a horizon, and it is largest exactly where people tend to report a single number.

Yet at 3,200 the ridge fit loses on all eight seeds while the marginal spread is 44.8 percent. That looks contradictory and is not: the spread is over what different fits achieve, and the paired difference asks what two fits trained on the same seed achieve. Both models see the same nine initial conditions, so the hardest initial condition is hard for both, and subtracting removes it. Pairing is the cheapest variance reduction available, and it costs nothing beyond keeping the evaluation set fixed.

The second mistake is reading the win column alone. At 50 steps the ridge fit loses 0 of 8 seeds, with a confidence interval that excludes zero, and the difference is 0.000014, about 1.7 percent of the error. Statistically detectable, operationally irrelevant. At 200 steps the ridge fit wins on 7 of 8 seeds, and the interval spans zero, because the one seed where it loses loses by more than the seven where it wins. A win count with no magnitude is a coin toss with extra steps.

## Choosing the horizon grid

The grid here is 50, 200, 800 and 3,200 steps, which is geometric with a factor of four. Geometric spacing is the right default for a compounding quantity, because error growth is often closer to exponential early and to a power law later, and linear spacing spends all its cells in the region where every model looks the same. Four horizons times eight seeds is 32 cells and takes under a minute on one core; the same design at twelve horizons and twenty seeds still costs little, but usually answers a question nobody asked.

The upper end should sit above the horizon you ship, not at it. If the last cell is the shipped horizon and the trend is still improving, you cannot tell whether the ranking is about to reverse one step past your last measurement. In this run the shipped horizon is 3,200 steps, the longest cell, and the ridge fit's disadvantage is still growing at that point, which is at least consistent with the reversal risk being low in that range.

The lower end should be where your users start caring. A horizon of 50 steps is a quarter of a second of simulated time at this time step, so it earns its place only as a check on the short-horizon story. For a three-day hourly demand forecast, the ship horizon is 72 steps and the grid should span something like 6, 24, 72 and 216, with the last cell past the deployment window on purpose.

Report steps and physical time together. A rollout length in steps is ambiguous the moment someone changes the time step, and the ambiguity is exactly the kind that makes two papers disagree while both are correct. The Well avoids this by publishing time windows in its rollout table, and the habit is worth copying.

## What the scorecard decided

The `verdict` function encodes the rule the tables imply, and it is deliberately blunt. A difference counts only if the interval excludes zero at that horizon. Among the horizons where it counts, the winner has to be the same model every time. And the winner also has to hold at the horizon you ship, because a model that wins at 50 steps and loses at 3,200 is not a better model for a 3,200-step rollout.

For the regularised pair it returns `ls8-WINS` with evidence at 2 of 4 horizons, both pointing at the plain fit. The ridge variant is not better anywhere it matters, and it is reliably worse at the short horizon where the tiny 1.7 percent penalty sits. The practical outcome is to drop it, which is not the outcome the 200-step row would have suggested.

For the data-volume pair it returns `ls64-WINS` with evidence at all 4 horizons. Fitting on 64 trajectories instead of 2 is a real improvement at every horizon tested, the direction never reverses, and the margin grows from 0.000066 at 50 steps to 0.007380 at 3,200. That is what a difference that survives horizon changes looks like.

One comparison was decided, the other dissolved, and neither answer was available from the one-step column, where both pairs were nearly indistinguishable.

A `REVERSAL` verdict is the third outcome and the one worth planning for. It means two or more horizons are individually reliable and disagree, so the pair is genuinely horizon-dependent and no single ranking is honest. The response is not to pick the horizon that agrees with your prior. It is to report both crossings, say which horizon the deployment uses, and either extend the grid until one model loses everywhere past the ship horizon or accept that the two are interchangeable at the shipped length and choose on cost, latency or licensing instead. The same logic applies to a `NOT-SHOWN-AT-SHIP-HORIZON` verdict: a model that is reliably better at 200 steps and indistinguishable at 3,200 is a research result, not a deployment argument.

There is one more decision the harness deliberately leaves out. With eight seeds, an interquartile mean is a trimmed mean over the middle six values, which is more stable than the mean but still a single number, and the interval around a paired difference is already doing the work that the aggregate metric would do across tasks. If the sweep covered dozens of initial conditions per seed rather than six, aggregating with the interquartile mean and a bootstrap interval, as the reinforcement-learning literature recommends for many-task comparisons, would start to matter. At this size it does not, and adding it would imply a precision the sample cannot support.

## Applying it to your own model

The pieces transfer unchanged to any autoregressive evaluation. Fix the horizon your users actually consume before you run anything, because the horizon is the decision and not an implementation detail. Keep one evaluation set for all candidates, so the paired difference has something to cancel. Run at least five fits, and report the minimum and maximum per horizon rather than a mean with a standard deviation computed from three samples. Bootstrap the paired differences, seeded, so a rerun gives the same interval and a reviewer can reproduce it. Treat win counts as a description of the sample, never as the evidence. Count how many times the sign of the difference changes across horizons, and read that count as a warning about the sensitivity of the claim.

If any of that is expensive, the cheap version still helps: fix the horizon, rerun the comparison on three seeds, and publish the per-seed values next to the mean. An interval from three seeds is wide, and wide is honest.

For learned simulators in particular, the semigroup diagnostic is worth running before a rollout sweep, since it needs no rollouts at all and its published correlation with rollout degradation is strong if not decisive. Treat it as a filter for which models deserve the sweep, not as a substitute for the sweep.

## Key takeaways

| Question | Bad answer | Answer the harness gives |
|---|---|---|
| Which model wins? | The one with lower mean error | The one that wins at the shipped horizon, with an interval excluding zero |
| How accurate is it? | One RMSE | A curve, min to max across seeds, per horizon |
| Is the difference real? | It won 7 of 8 seeds | Paired bootstrap interval on the paired differences |
| Is the difference usable? | It is 1.7 percent better | Significant at 50 steps, irrelevant at 1.7 percent of error |
| Does the ranking hold? | It held at the horizon I tested | Sign reversals counted across every horizon tested |
| How much noise is there? | A standard deviation from 3 runs | Marginal spread per horizon, which grows with rollout length |

## References

- The Well: a large-scale collection of diverse physics simulations for machine learning — [arXiv:2412.00568](https://arxiv.org/abs/2412.00568), NeurIPS 2024 Datasets and Benchmarks; one-step advantages, VRMSE normalisation, per-field metrics
- Do physics foundation models learn generalizable physics? A bias-aware benchmark — [arXiv:2605.29283](https://arxiv.org/abs/2605.29283); 8 dynamics, 25 test regimes, 60,000 measurements
- Semigroup consistency as a diagnostic for learned physics simulators — [arXiv:2605.26324](https://arxiv.org/abs/2605.26324); Spearman 0.635, 95% CI [0.621, 0.649]
- Deep reinforcement learning at the edge of the statistical precipice — [arXiv:2108.13264](https://arxiv.org/abs/2108.13264), NeurIPS 2021; interquartile mean, stratified bootstrap, performance profiles
- Show your work: improved reporting of experimental results — [arXiv:1909.03004](https://arxiv.org/abs/1909.03004); compute-budget curves
- Deep reinforcement learning that matters — [arXiv:1709.06560](https://arxiv.org/abs/1709.06560), AAAI 2018; seed variance and reporting standards
- Assessing the macro and micro effects of random seeds on fine-tuning large language models — [arXiv:2503.07329](https://arxiv.org/abs/2503.07329), IJCNLP 2025; GLUE and SuperGLUE variance

## Related posts

- [Six Materials, One Architecture](/posts/unified-particle-simulator-six-materials/) is the run this harness re-measures, including the single-seed flip that motivated it
- [One Label Holds Ten Numbers](/posts/distillation-target-logit-budget/) is another case where the metric you optimise and the metric you ship diverge
- [The Denominator Is the Metric](/posts/rag-recall-at-k-denominator/) treats the other half of evaluation design, choosing what counts as a hit
- [Evaluating LLMs: Benchmarks That Matter for African Use Cases](/posts/evaluating-llms-african-use-cases/) applies the same insistence on per-condition reporting to a different stack
