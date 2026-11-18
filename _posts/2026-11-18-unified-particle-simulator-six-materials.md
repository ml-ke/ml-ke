---
title: "Six Materials, One Architecture: What a Unified Simulator Actually Unifies"
date: 2026-11-18 00:00:00 +0300
categories: [Machine Learning, AI Engineering]
tags: [simulation, world models, transformer, scientific machine learning, protein dynamics, granular materials]
math: true
image:
  path: /assets/img/cover-unified-particle-simulator-six-materials.webp
  alt: Three particle clusters for sand, cloth and a protein chain merging into a single block of super tokens, with the merged attention cost marked at a third of full resolution
---

On 9 November MIT News described a system that simulates sand, cloth, fluids and proteins with one deep-learning architecture, reporting higher accuracy than numerical simulation methods. The paper behind it, [WorldParticle](https://arxiv.org/abs/2605.15305), had been on arXiv since May and is scheduled for ACM SIGGRAPH Asia. The authors are a wide consortium: Minghao Guo and Peter Yichen Chen co-led it, with Caoliwen Wang and Siyuan Chen, Wojciech Matusik at MIT CSAIL, and researchers from Georgia Tech, Inria, Meta, Penn, Utah, UCLA and Hong Kong.

The result is worth reading closely for two reasons. It is one of the cleaner attempts at replacing the pile of hand-coded solvers that scientific computing runs on, and the way it splits a simulation step is directly reusable even if you never train a particle model.

> **The short version**
> What unifies here is the architecture, the particle representation and the form of the update, not the weights. Six dynamics categories means six trained models. Read the headline as "one code path, one step design, six surrogates" and every number afterwards makes sense.
{: .prompt-info }

## Six regimes, six checkpoints

The paper covers cloth, elastic solids, Newtonian fluids, non-Newtonian fluids, granular materials and molecular dynamics. Training data for each comes from an established solver rather than from experiment: DFSPH for the fluids, GIPC for cloth, material point method (MPM) for sand and non-Newtonian fluids, VBD for elastic shapes, OpenMM for protein dynamics.

For each category the authors train a separate model and randomise material parameters, initial conditions and boundary configurations inside that category. Held-out test scenes then vary the same factors. Table 1 of the paper gives the test sets:

| Regime | Training data from | Test frames | Particles |
|---|---|---|---|
| Newtonian fluid | DFSPH | 600 | 14,040 |
| Cloth | GIPC | 90 | 9,800 |
| Sand | MPM | 100 | 20,000 |
| Non-Newtonian fluid | MPM | 250 | 20,000 |
| Elastic solids | VBD | 100 | 17,514 |
| Protein dynamics | OpenMM | 100 | 20,628 |

Particle counts are the reported degrees of freedom divided by three, so the largest test set here is roughly 20,000 particles. What generalises across the six is the architecture: the abstract claims the same design handles unseen materials, boundary configurations, initial conditions and external forces. The MIT write-up calls it "one unified artificial intelligence architecture", which is the accurate phrasing. A single checkpoint trained on all six is the stated long-term goal, not the delivered artifact. The authors describe wanting a foundation model for physical simulation that works out of the box; the paper ships the architecture that would sit under it.

That distinction is practical, not pedantic. Six models means six data pipelines, six training runs on eight A100s, and six sets of hyperparameters to maintain. It is still a large reduction in engineering compared with writing six solvers, but it is not the thing a benchmark table usually implies.

## The step: apply what you know, learn the residual

The interesting design choice sits in the update. Each particle carries position, velocity and per-particle attributes such as mass, material parameters or atom type. Static boundary particles carry walls, obstacles and contact surfaces, with normals and friction coefficients.

Rather than predict the next state directly, the model advances particles first under the known external forces, producing an intermediate state $\tilde{X}_t$. A learned corrector then predicts the residual:

$$
X_{t+\Delta t} = \tilde{X}_t + \Delta X_t, \qquad V_{t+\Delta t} = \tilde{V}_t + \Delta V_t
$$

The residual is where the paper's argument lives. Contact projections, implicit solves, pressure coupling and damping cannot be expressed as a single explicit acceleration, so they cannot be folded into the predictor. Predicting both position and velocity corrections leaves the network free to absorb exactly those effects.

The corrector runs in three stages. A particle tokenizer encodes local particle-particle, particle-boundary and topology-guided interactions. A super-token encoder then merges particle tokens hierarchically, halving the token count at each level through alternating self-attention and token merging. A super-token decoder lifts the merged representation back to particle resolution by cross-attention and emits per-particle corrections. Progressive merging is what keeps attention affordable: cost is quadratic in token count, so each level costs a quarter of the previous one.

Two details explain why one code path can serve six domains. Inputs a domain does not have are zero-filled rather than branched around, so fluids have no rest-shape topology and proteins have no boundary particles, and the same tensor shapes travel through. And because the tokenizer works on relative displacements rather than absolute coordinates, the model tolerates particle densities it never saw in training: the fluid model was trained on about 7,000 particles and stayed stable at higher counts.

The ablations are worth noting because they show the hierarchy is load-bearing. Removing the super-token encoder and decoder, so the correction is predicted from particle tokens alone, reduces the model to a Deep Lagrangian-style predictor with no multi-level communication. Removing the neighbourhood aggregation branches leaves each token encoding a single particle. For fluids the paper also tests a divergence-free regulariser computed with the same SPH kernel as the reference solver, which trades a little error for physical plausibility.

## The number you publish is the horizon you ship

The reported errors are rollout mean squared errors in particle position and velocity, averaged over particles, frames and test sequences, with throughput and per-step cost on a single 80 GB A100:

| Regime | Position error (x10^-3) | Velocity error (x10^-3) | Frames/s | TFLOP/step |
|---|---|---|---|---|
| Newtonian fluid | 516.585 | 263.149 | 16.176 | 1.323 |
| Cloth | 0.077 | 0.454 | 24.050 | 0.923 |
| Sand | 0.054 | 1.442 | 11.500 | 1.886 |
| Non-Newtonian fluid | 0.706 | 1.760 | 11.168 | 1.886 |
| Elastic solids | 0.158 | 3.530 | 11.708 | 1.651 |
| Protein dynamics | 0.001 | 0.092 | 14.695 | 0.944 |

Read the scale before the ranking. Errors are in units of $10^{-3}$ of whatever length unit each regime uses, and the fluid row is three to four orders of magnitude larger than the cloth row. Averaging those into a single "accuracy" figure would be meaningless, so the six rows have to be read as six comparisons against six different solvers.

The cost column largely tracks particle count. Cloth runs at 24 frames per second for 0.923 TFLOP per step; sand and non-Newtonian fluid, both at 20,000 particles, run at 11.5 and 11.2 frames per second for 1.886 TFLOP per step. Two regimes with different physics and nearly identical cost is what a shared architecture should look like.

The quantity that actually matters is the rollout horizon, and it is the easiest thing to leave out of a table. To see how much it changes a conclusion, here is a deliberately plain experiment on a damped projectile with a floor bounce, a case where the ground truth is cheap to compute and a learned step is easy to fit. Two linear models see the same data and differ only in feature set, then roll out autoregressively from the same starting state:

{% raw %}
```python
import numpy as np

DT, G, REST = 0.01, 9.81, 0.6

def simulate(v0, k, steps):
    """Ground truth: quadratic drag plus an exact floor bounce."""
    v, x = np.array(v0, float), np.zeros(2)
    xs = [x.copy()]
    for _ in range(steps):
        v = v + DT * (np.array([0.0, -G]) - k * np.linalg.norm(v) * v)
        x = x + DT * v
        if x[1] < 0.0:
            x[1] = 0.0
            if v[1] < 0:
                v[1] = -REST * v[1]
        xs.append(x.copy())
    return np.array(xs)

def small(v, x):
    s = np.linalg.norm(v)
    return np.array([v[0], v[1], s * v[0], s * v[1], 1.0])

def big(v, x):
    s = np.linalg.norm(v)
    return np.array([v[0], v[1], s * v[0], s * v[1], x[1],
                     v[0] * v[1], v[0] ** 2, v[1] ** 2, 1.0])

def fit(feat, seed=3):
    rng = np.random.default_rng(seed)
    rows, nxt = [], []
    for k in [0.05, 0.10, 0.20]:
        for _ in range(50):
            v0 = rng.normal(0, 2, 2) + np.array([5.0, 9.0])
            xs = simulate(v0, k, 140)
            for t in range(len(xs) - 2):
                v = (xs[t + 1] - xs[t]) / DT
                rows.append((feat(v, xs[t]), xs[t], v))
                nxt.append((xs[t + 1], (xs[t + 2] - xs[t + 1]) / DT))
    Phi = np.array([r[0] for r in rows])
    Y = np.hstack([np.array([n[0] for n in nxt]) - np.array([r[1] for r in rows]),
                   np.array([n[1] for n in nxt]) - np.array([r[2] for r in rows])])
    W, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    return W

def rollout(feat, W, v0, k, steps):
    v, x = np.array(v0, float), np.zeros(2)
    for _ in range(steps):
        p = feat(v, x) @ W
        x, v = x + p[:2], v + p[2:]
        if x[1] < 0.0:
            x[1] = 0.0
            if v[1] < 0:
                v[1] = -REST * v[1]
    return x

tests = [(np.array([6.0, 10.0]), 0.15), (np.array([3.0, 12.0]), 0.08)]
horizons = [1, 50, 200, 800]
print("mean final position error (m) over two held-out trajectories")
print(f"{'features':>10} " + " ".join(f"{'h=' + str(h):>9}" for h in horizons))
for label, feat in [("5-term", small), ("9-term", big)]:
    W = fit(feat)
    row = [np.mean([np.linalg.norm(rollout(feat, W, v0, k, h) - simulate(v0, k, h)[-1])
                    for v0, k in tests]) for h in horizons]
    print(f"{label:>10} " + " ".join(f"{e:9.4f}" for e in row))
```
{% endraw %}

```text
mean final position error (m) over two held-out trajectories
  features       h=1      h=50     h=200     h=800
    5-term    0.0026    0.3131    1.7315    6.5885
    9-term    0.0026    0.3265    0.7839   10.8525
```

At one step the two models are indistinguishable, both within 3 mm of the truth. At 200 steps the richer model is 2.2 times better. At 800 steps it is 1.6 times worse. Nothing about the models changed between columns; only the horizon did. A paper that reports the first column, or even the third, can end up recommending the model that loses at the step count its readers will actually run.

This is the reason the WorldParticle tables are mostly rollout errors rather than one-step errors, and it is why one of their smaller results is the more convincing one: the cloth model was trained on 200-frame sequences and tested at 250, a 25 percent extrapolation past its training horizon, and it kept winding without collapsing or interpenetrating. Long-horizon stability is the hard part of autoregressive simulation, and it does not show up until the rollout is longer than anything in the training set.

The qualitative comparison points the same way. Against GNS, Deep Lagrangian and a neural operator baseline, the paper reports artefacts in the baselines' long rollouts: sand particles floating in the air during a collapse, cloth blowing up under collision, meshes interpenetrating. A model that produces those artefacts can still post a respectable short-horizon error.

## Three baselines, three kinds of failure

The comparisons run on fluid, cloth and sand, against three different lineages of learned simulator, and it is worth knowing what each one is because the failure modes differ.

Graph network simulators (GNS, Sanchez-Gonzalez et al., 2020) build a graph over nearby particles and pass messages along its edges. They are local by construction, so information travels one hop per step, and long rollouts tend to lose the global structure: a pile collapses correctly at its contact points while particles that should have been dragged along stay where they were.

Deep Lagrangian methods learn the particle step directly from state, which is the approach the paper's own local-only ablation collapses into once the super-token levels are removed. Direct step prediction has no mechanism to communicate beyond the neighbourhood, so it handles contact-rich scenes the way a very good local extrapolator would: it smooths over the discontinuities.

Neural operators learn a mapping between function spaces rather than between particle states, which suits fields on regular grids better than irregular particle sets. Applied to sand and cloth, the mismatch shows up as interpenetration: a hard boundary has to be absorbed into the learned operator rather than applied as a constraint at each step.

The pattern across all three failures is the same: the model is asked to represent something global, or something discontinuous, using machinery that is local and smooth. WorldParticle's answer is to put the global part in a hierarchy of merged tokens and the discontinuous part in a learned residual, while keeping the part with a clean closed form in the explicit predictor. Whether or not the specific architecture survives, that decomposition is the transferable idea.

## Control, inverse design and data from a real robot

Interactive control means steering a rollout toward a goal while it runs, rather than simulating forward and hoping. A learned step that costs tens of milliseconds per frame makes that loop practical, because the controller can try a candidate against the model before committing to it.

Inverse design runs the same machinery backwards. Instead of asking what a system does under given parameters, you ask which parameters produce a target behaviour, and optimise them against the simulator. The paper frames this as a reason to care about a single general architecture: when the same model covers granular materials and elastic solids, a design problem that spans both regimes stops needing two toolchains and a handoff.

The third demonstration is the one that matters most for anyone outside computer graphics. The authors fine-tune on real-world manipulation data, meaning trajectories captured from a physical setup rather than generated by a solver. A model trained only on solver output learns the solver; a model that can absorb real trajectories is being asked to learn the material instead. That is the difference between a fast approximation of MPM and something that could eventually disagree with MPM and be right.

## What the surrogate inherits from its solver

Every model here is trained on solver output, which sets a ceiling that no amount of architecture work moves. DFSPH, GIPC, MPM, VBD and OpenMM each encode assumptions: how incompressible a fluid is, how friction is regularised at a contact, which thermostat governs a protein. A surrogate that matches its solver to three decimal places has matched those assumptions too.

Two practical consequences follow. The first is that the learned step is tied to the timestep it was trained on. Asking it to advance the clock in bigger jumps is extrapolation in time, not just in state, and the stability experiments reported in the paper vary the rollout length and the scene, not the step size, so they say nothing about coarser steps. The second is that the error you are measuring is error against the solver, not against physics. A learned simulator that reproduces MPM's sand exactly inherits MPM's treatment of friction, including whatever it gets wrong.

This is why the real-world manipulation experiment carries more weight than another decimal place in the benchmark table. It is the only check in the paper where the reference is not another simulator.

## Where the compute goes

Progressive token merging is the part you can copy without a cluster, and its effect is arithmetic rather than empirical. Attention cost is quadratic in token count, and halving the token count at each of four levels cuts the pair count to a third of what four full-resolution levels would cost:

{% raw %}
```python
DOF = {"fluid": 42120, "cloth": 29400, "sand": 60000,
       "non-newtonian": 60000, "elastic": 52542, "protein": 61884}

def pairs(n, levels=4):
    """Attention pair counts: every level halving the token count."""
    return sum((n >> i) ** 2 for i in range(levels))

def pairs_full(n, levels=4):
    return levels * n ** 2

print(f"{'category':>14} {'particles':>10} {'merged/full':>12} {'fp16 GB at full':>16}")
for name, dof in DOF.items():
    n = dof // 3
    share = pairs(n) / pairs_full(n)
    print(f"{name:>14} {n:>10,} {share:>11.1%} {2 * n ** 2 / 2**30:>16.2f}")

print()
for n in (20000, 50000, 100000):
    print(f"N={n:,}: 4-level merged cost {pairs(n) / 1e9:.3f}e9 pairs "
          f"({pairs(n) / pairs_full(n):.1%} of four full-resolution levels); "
          f"one fp16 attention matrix {2 * n ** 2 / 2**30:.1f} GB")
```
{% endraw %}

```text
category  particles  merged/full  fp16 GB at full
         fluid     14,040       33.2%             0.37
         cloth      9,800       33.2%             0.18
          sand     20,000       33.2%             0.75
 non-newtonian     20,000       33.2%             0.75
       elastic     17,514       33.2%             0.57
       protein     20,628       33.2%             0.79

N=20,000: 4-level merged cost 0.531e9 pairs (33.2% of four full-resolution levels); one fp16 attention matrix 0.7 GB
N=50,000: 4-level merged cost 3.320e9 pairs (33.2% of four full-resolution levels); one fp16 attention matrix 4.7 GB
N=100,000: 4-level merged cost 13.281e9 pairs (33.2% of four full-resolution levels); one fp16 attention matrix 18.6 GB
```

The last column is the constraint that sets the ceiling. The paper's stated limitation is memory: the first encoder layer attends at full particle resolution, which caps the system at about 50,000 particles on a single 80 GB A100. At that count one attention matrix in fp16 is already 4.7 GB, and a real forward pass holds more than one. The authors point at sparse attention as the route past it. Nothing here is a FLOP problem; the throughput figures in the table above are 11 to 24 frames per second for the whole step, so a 600-frame fluid rollout lands in under a minute of GPU time.

## What the result does and does not support

Supported by the paper's own tables and figures: one architecture trains on six particle regimes; the same weights generalise to unseen material parameters, initial conditions, boundary configurations and forces within a regime; the cloth rollout stays stable 25 percent past its training horizon and the fluid case runs 800 frames; the corrections support interactive control, inverse design and learning from real manipulation data.

Not supported: that one checkpoint covers all six, which the per-category training explicitly contradicts; that a general-purpose simulator is available off the shelf, which the authors describe as future work; and that numerical solvers can be retired. Every model here learns from solver output, so its ceiling is the solver, and its valid timestep is bounded by what appears in training data.

The "about 100 times faster than numerical simulators" figure needs a caveat that the paper's table does not give: the table reports frames per second, not the physical timestep each frame advances. Frames and solver timesteps are not automatically the same quantity, so treat the 100x number as a headline from the reporting rather than a like-for-like comparison. The honest version is that the learned step is cheap per frame, which the 11 to 24 frames per second figure already shows.

## When a surrogate pays for itself, and what to check besides error

A learned step at 11 to 24 frames per second on one GPU is worth its training cost when you need thousands of rollouts and cannot afford a solver for each one: design sweeps over material parameters, control sequences searched inside a loop, planning under uncertainty where a policy has to try twenty candidate actions before it acts. Training on eight A100s amortises against that inner loop. It does not amortise against one careful simulation of one new material, where writing the solver input is cheaper than assembling a training set and validating a network.

The second thing worth checking is invariants rather than error. A particle simulator that scores well on positional MSE can still be unphysical, and the failures the paper reports in its baselines are exactly that: sand particles hanging in the air are a violation of support, not a large number; cloth that explodes is a violation of boundedness; interpenetrating meshes violate non-overlap. Four checks catch most of this and cost nothing at inference: particle count is conserved, no particle ends a step below a boundary it should rest on, the energy of a damped system decreases monotonically, and no coordinate or velocity returns a NaN.

The third is spread. The horizon table above is one seed and two held-out trajectories, which is enough to show that rankings move with horizon and not enough to rank the two models. Any comparison of learned simulators should report the spread across seeds and across trajectory families, because the failure mode of a single flattering run is exactly the failure mode this class of model has.

## How to apply this

Six things transfer even if your problem is not a particle simulator.

1. Split the step. Apply the closed-form physics in the loop and learn only the residual. The residual is where the effects without a clean acceleration live: contact, damping, implicit solves. Training on the residual keeps the network's target scale small.
2. Feed relative displacements. The model tolerates densities it never saw because its tokenizer works on inter-particle distances. Absolute-coordinate features would tie it to the training resolution.
3. Zero-fill missing inputs instead of branching. One code path across domains costs nothing at inference and saves you a pipeline per domain.
4. Benchmark at the horizon you ship, and publish the whole column, not the best cell. The demo above shows a 2.2x ranking advantage turning into a 1.6x deficit between 200 and 800 steps.
5. Treat token merging as a knob before you buy hardware. Four levels at a third of the pair count is arithmetic you can plan against.
6. Check memory before FLOPs. The published limit is 50,000 particles on an 80 GB card, and it comes from a full-resolution attention layer, not from throughput.

## Key takeaways

| Claim | Measured | How to read it |
|---|---|---|
| One architecture, six regimes | Six trained models, one per category | Architecture-level unification, not one checkpoint |
| Higher accuracy than numerical methods | Rollout MSE per regime against that regime's solver | Six separate comparisons; scales are not comparable across rows |
| Stable long rollouts | Cloth tested 25 percent past its training horizon | The extrapolation is the result, not a side note |
| About 100x faster on proteins | Frames/s on one A100, protein row 14.7 | Headline from reporting; frames and solver steps are different units |
| ~50,000 particle ceiling | First layer attends at full resolution | Memory bound, 4.7 GB per fp16 matrix at 50k |

## References

- WorldParticle: Unified World Simulation of Lagrangian Particle Dynamics via Transformer — [arXiv:2605.15305](https://arxiv.org/abs/2605.15305), [HTML v4](https://arxiv.org/html/2605.15305v4)
- A single AI model simulates the behavior of many substances, from sliding sand to folding proteins — [MIT News, 9 November 2026](https://news.mit.edu/2026/single-ai-model-simulates-many-substances-behavior-1109)
- Unified AI model accurately simulates diverse physical phenomena from sand to proteins — [BioEngineer.org, 14 November 2026](https://bioengineer.org/unified-ai-model-accurately-simulates-diverse-physical-phenomena-from-sand-to-proteins/)
- Project page — [worldparticle.github.io](https://worldparticle.github.io/WorldParticle_Web/)

## Related posts

- [Words In, Enzymes Out: What Two Protein-Design Releases Actually Delivered](/posts/words-in-enzymes-out-protein-design/)
- [When the Model Designs the Experiment: AI-Designed Physics Experiments](/posts/ai-designed-physics-experiments/)
