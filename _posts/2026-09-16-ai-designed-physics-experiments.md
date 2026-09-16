---
title: "The Lab Partner That Never Sleeps: How AI Now Designs Physics Experiments"
date: 2026-09-16 00:00:00 +0300
categories: [AI Engineering, Machine Learning]
tags: [ai-for-science, experiment-design, optimization, search, physics-informed-ml]
image:
  path: /assets/img/cover-ai-designed-physics-experiments.webp
  alt: An optical table with laser, mirrors and detectors, a dashed ghost path showing a computer-proposed alternative layout, and a side panel of scored candidate circuits
---

## Introduction

On 2 September 2026, a review article appeared in *Nature* with a title that reads like a modest methods paper and a conclusion that does not: [Designing physics experiments with artificial intelligence](https://www.nature.com/articles/s41586-026-10898-6) (Klimesch, Arlt, Ruiz-Gonzalez et al., *Nature* **657**, 47–58, 2026). The authors' own summary of the state of the field is blunt — AI-driven design methods "have begun to move beyond tuning a handful of parameters to proposing entirely new experimental layouts," and the configurations they discover "often challenge established design conventions while matching or even exceeding the performance of human-designed set-ups."

> **Why this is the story worth reading**
> Most AI headlines are about models that talk, models that fail, or models that get abused. This one is about machines finding experiments that humans did not think of — documented, peer-reviewed, and already running in electron microscopes, fusion research, particle detectors and gravitational-wave labs. It is the clearest case of AI as a *scientific instrument* rather than a chatbot.
{: .prompt-info }

Three write-ups agree on the substance: the [TU Wien press release](https://physik.univie.ac.at/en/news/news-detail/news/artificial-intelligence-suggests-new-physics-experiments/) (3 Sept 2026), [Phys.org's report](https://phys.org/news/2026-09-ai-physics-outperform-human-setups.html), and the [Tübingen AI Center announcement](https://tuebingen.ai/news/ai-could-help-scientists-design-experiments-humans-would-never-think-of) from Mario Krenn's group. The paper is a review — a survey of what has already been achieved, not one new result — which makes the claim stronger, not weaker: this is a decade of evidence, summarised.

## One night in Vienna, ten years ago

The origin story is unusually concrete. As a student in Vienna, Mario Krenn — now professor of machine learning in science at the University of Tübingen — was trying to build a quantum optics experiment. He and his group could not find a configuration that would demonstrate the effect they wanted.

So he described the components on the bench mathematically and let an algorithm search combinations of them.

> "Programming it only took a few hours. Then I went home and left the computer running," Krenn says. "When I came into the office the next day, the program had produced a file containing a proposed solution... unlike all of us, the computer had found an experimental setup that satisfied the necessary criteria."
{: .prompt-tip }

That result was published in 2016 as *Automated Search for new Quantum Experiments* (Krenn, Malik, Fickler, Lapkiewicz and Zeilinger, [Phys. Rev. Lett. **116**, 090405](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.090405); preprint [arXiv:1509.02749](https://arxiv.org/abs/1509.02749)) — the "Melvin" line of work that seeded the field. Ten years later, the same approach has a *Nature* review, co-authored by people from quantum optics, electron microscopy, particle physics and gravitational-wave detection.

## This is not a chatbot — it is a search problem

The clearest part of the review is what it is *not*. Krenn is explicit that this has "little in common with the kind of AI familiar from large language models," because chatbots generate statistically likely text, while experiment design is not a language task at all.

> "It is an enormous optimization problem. There is an unimaginably vast space of possible experiments that can be set up with the available components, and the computer must systematically search this space to find the best possible solution."
{: .prompt-info }

The review organises that search around four questions, and they double as a specification for anyone building a design system:

1. **How do we engineer expressive search spaces?** The components in a lab — lasers, lenses, mirrors, detectors, electronics — can be combined in a fantastically large number of ways. The space must be defined so that good designs *exist* inside it.
2. **How do we build fast and reliable simulators?** Every candidate design has to be scored, which means simulating it. A simulator that is too slow limits how much of the space you can explore; a simulator that is wrong makes the whole search worthless.
3. **How do we translate scientific goals into computable objectives?** "Make the measurement unambiguous" is not a number. The objective function is where a scientific intent becomes something a machine can optimise.
4. **How do we explore both discrete and continuous choices?** Which components are connected at all is a discrete, combinatorial question; the angle of a waveplate is a continuous one. Real benches mix both.

That framing is why the results are useful beyond physics. Any engineering problem where you have a fixed catalogue of parts, a simulator, and a measurable goal is the same shape.

## Where it already works

The review's examples are not hypotheticals. AI-driven design has already been used to improve [fusion reactors](https://phys.org/news/2026-09-ai-physics-outperform-human-setups.html), generate new ideas for particle detectors, and propose ways of making gravitational-wave detector systems even more sensitive.

Electron microscopy is the most vivid case, according to Philipp Haslinger, head of the Center for Electron Microscopy at TU Wien: "In electron microscopy in particular, we are only now beginning to work systematically with entanglement and new quantum-mechanical microscopy concepts. Human intuition in this area is often still very limited. Artificial intelligence can therefore identify microscope designs that a human would probably never have come up with, but which can produce significantly better images or offer entirely new measurement possibilities."

And the tooling is not locked inside a lab. [PyTheus](https://github.com/artificial-scientist-lab/PyTheus), the open-source discovery framework from the same research line, was published with a paper that found [100 diverse quantum experiments](https://quantum-journal.org/papers/q-2023-12-12-1204/) in one run (*Quantum* **7**, 1204, 2023; preprint [arXiv:2210.09980](https://arxiv.org/abs/2210.09980)). PyTheus is a `pip install` away, not a bespoke industrial system.

## A tiny version you can run right now

Here is a deliberately small version of the same idea: five component types, six slots, and a target operation no single available component can produce. A design is a sequence of slots, a simulator turns it into a 2×2 unitary, and an objective function scores how close it lands to the target.

{% raw %}
```python
"""Toy experiment design: search a discrete component space for a circuit that
reaches an operation no single available part can produce."""
import cmath
import itertools
import math


def bs(theta):
    """Beam splitter with mixing angle theta."""
    c, s = math.cos(theta), math.sin(theta)
    return [(c, 1j * s), (1j * s, c)]


def ps(phi):
    """Phase shifter."""
    return [(cmath.exp(1j * phi), 0j), (0j, 1 + 0j)]


def identity():
    return [(1 + 0j, 0j), (0j, 1 + 0j)]


def mm(a, b):
    return [
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    ]


# What the lab owns: 5 part types, addressed by name.
ALPHABET = {
    "BS45": bs(math.pi / 4),   # 50/50 splitter
    "BS22": bs(math.pi / 8),   # 22.5 degree splitter
    "PS90": ps(math.pi / 2),   # 90 degree phase
    "PS45": ps(math.pi / 4),   # 45 degree phase
    "IDLE": identity(),        # empty slot
}
SLOTS = 6

# What we want to build: a 30 degree mixer plus a 60 degree phase.
TARGET = mm(bs(math.pi / 6), ps(math.pi / 3))

# The layout a person writes first: splitter then phase, rest of the bench idle.
HUMAN = ("BS45", "PS90", "IDLE", "IDLE", "IDLE", "IDLE")


def transfer(design):
    u = identity()
    for name in design:
        u = mm(ALPHABET[name], u)
    return u


def fidelity(u):
    tr = sum(u[i][j].conjugate() * TARGET[i][j] for i in range(2) for j in range(2))
    return abs(tr) / 2


space = list(itertools.product(ALPHABET, repeat=SLOTS))
scored = sorted(((fidelity(transfer(d)), d) for d in space), key=lambda t: -t[0])
best_f, best_d = scored[0]
top = [d for f, d in scored if best_f - f < 1e-9]

print(f"designs enumerated            : {len(space):,}")
print(f"human layout fidelity         : {fidelity(transfer(HUMAN)):.4f}")
print(f"best design                   : {' '.join(best_d)}")
print(f"best fidelity                 : {best_f:.6f}")
print(f"designs tied at best (1e-9)   : {len(top)}")
for d in top[:3]:
    print(f"  - {' '.join(d)}")
print(f"designs with fidelity >= 0.95 : {sum(1 for f, _ in scored if f >= 0.95):,}")
print(f"median fidelity of space      : {scored[len(scored) // 2][0]:.4f}")
```
{% endraw %}

Run it, and you get this — verbatim, reproducible, no randomness:

```text
designs enumerated            : 15,625
human layout fidelity         : 0.6830
best design                   : PS90 PS90 PS90 PS90 PS45 BS22
best fidelity                 : 0.982963
designs tied at best (1e-9)   : 27
  - PS90 PS90 PS90 PS90 PS45 BS22
  - BS45 BS45 BS45 BS45 PS45 BS22
  - PS90 PS90 PS90 PS45 PS90 BS22
designs with fidelity >= 0.95 : 115
median fidelity of space      : 0.3536
```

Three things in that output are the whole lesson. The conventional layout — splitter, then phase, rest of the bench idle — lands at **0.6830**, while the best six-slot design reaches **0.982963**, and it is not a neat human pattern: four phase shifters, a 45° phase and a 22.5° splitter. And **27 designs tie at the best score** while only 115 of 15,625 clear 0.95 — the space is mostly bad, the good region is small, and inside it the answer is not unique.

This is a toy: no noise, no cost model, no real quantum mechanics. But it is the same loop the *Nature* review describes, at a scale you can run in a second.

## The honest limits

The review does not oversell, and neither should we. There are three real costs.

**Interpretability.** Sometimes the found design is instantly understandable. Often it is not: "You can calculate that the new experimental setup works better, but you cannot really put into words why." A design nobody can explain is a hard sell to a lab, a reviewer, or a funder — and it is a real risk when the machine is optimising an objective that imperfectly encodes what you actually care about.

**The simulator is the ceiling.** Every result inherits the assumptions of the model that scored it. If the simulator ignores a loss mechanism, the search will happily exploit the omission and hand you a bench that cannot be built or will not work.

**Non-uniqueness.** Our 27-way tie is the toy version of a general problem: many different layouts score identically, and the differences that decide which one survives contact with reality — cost, part availability, alignment tolerance, robustness — are usually not in the objective function.

Krenn's own framing of the human role is more interesting than either hype or doom:

> "Human work is simply shifting to a higher level. In the past, calculations had to be done by hand, and nobody wants to go back to that today."
{: .prompt-tip }

## How to apply this to your own problem

Strip the physics away and the recipe is portable:

1. **Write your components down as a search space.** Enumerate what is fixed (parts you own, space on the bench, budget) and what is free (connections, ordering, parameters). If good designs are not inside the space you defined, no search will find them.
2. **Build the simulator before the AI.** Accuracy first, speed second, and keep it honest about what it ignores. Log the assumptions.
3. **Turn the goal into one number.** Prefer a metric you would accept as evidence in a design review, not a proxy you can score highly while failing the real task — the benchmark-poisoning lesson applies to objective functions too.
4. **Match the search to the space.** Discrete structure (which part connects where) needs combinatorial or evolutionary search; continuous parameters suit gradient-based or Bayesian methods; realistic problems need both, staged.
5. **Assume the answer is not unique.** When several designs tie, add the constraints you left out of the objective — cost, availability, manufacturability, tolerance — and treat interpretation as part of the work, not an afterthought.
6. **Verify by ablation and by experiment.** Re-score the winner with the constraint removed to see what it is exploiting, then build the simplest version and measure.

For a lab with modest hardware — which describes most research groups, including many in Africa — the encouraging property is that this loop is software-first. PyTheus and comparable frameworks run on ordinary machines, so designs can be explored and shortlisted *before* anything is procured. The bottleneck shifts from bench equipment to the quality of your simulator and the sharpness of your objective, which is a much cheaper bottleneck to attack.

## Key takeaways

| Lesson | What to do |
|--------|-----------|
| AI design is search, not conversation | Frame the problem as space + simulator + objective, then choose an exploration method |
| Discovered layouts beat conventional ones | Do not assume the textbook configuration is optimal; let the search try irregular shapes |
| The objective function decides everything | Encode the real goal, then stress-test what the optimum is exploiting |
| Ties are the norm | Expect many equal-scoring designs; choose between them with the constraints you left out |
| Explanation lags performance | Budget human time for interpretation and ablation, not just for compute |
| The loop is software-first | Explore and shortlist designs on a laptop before committing budget to hardware |

Ten years ago a student left a program running overnight and came back to an experiment his group could not design. That is still the shape of the story in 2026 — except now it has a *Nature* review, open-source tooling, and a track record in fusion, microscopy, particle physics and gravitational-wave detection. The interesting question is no longer whether AI can propose a good experiment. It is how many labs will set up the search, the simulator and the objective to let it.

## References

1. [Designing physics experiments with artificial intelligence — *Nature* 657, 47–58 (2026)](https://www.nature.com/articles/s41586-026-10898-6)
2. [Artificial Intelligence Suggests New Physics Experiments — University of Vienna, Faculty of Physics (3 Sept 2026)](https://physik.univie.ac.at/en/news/news-detail/news/artificial-intelligence-suggests-new-physics-experiments/)
3. [AI suggests new physics experiments that could outperform human-designed setups — Phys.org (3 Sept 2026)](https://phys.org/news/2026-09-ai-physics-outperform-human-setups.html)
4. [AI Could Help Scientists Design Experiments Humans Would Never Think Of — Tübingen AI Center](https://tuebingen.ai/news/ai-could-help-scientists-design-experiments-humans-would-never-think-of)
5. [Our Review in Nature just got published — Mario Krenn's blog (7 Sept 2026)](https://mariokrenn.wordpress.com/)
6. [Automated Search for new Quantum Experiments — *Phys. Rev. Lett.* 116, 090405 (2016)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.090405)
7. [Automated Search for new Quantum Experiments — arXiv:1509.02749](https://arxiv.org/abs/1509.02749)
8. [Digital Discovery of 100 diverse Quantum Experiments with PyTheus — *Quantum* 7, 1204 (2023)](https://quantum-journal.org/papers/q-2023-12-12-1204/)
9. [PyTheus — open-source discovery framework (GitHub)](https://github.com/artificial-scientist-lab/PyTheus)
10. [Digital Discovery of 100 diverse Quantum Experiments with PyTheus — arXiv:2210.09980](https://arxiv.org/abs/2210.09980)

## Related posts

- [AI-Enabled Security: Seeing the Anomaly Humans and Static Rules Miss](/posts/ai-enabled-security-anomaly-detection/)
- [Fraud Model Drift Monitoring: Why Your Model Rots Quietly](/posts/fraud-model-drift-monitoring/)
- [Graph ML for Fraud Ring Detection: When Per-Account Scores See Nothing](/posts/graph-fraud-ring-detection/)
- [Tuesday AI Update: Global Roundup](/posts/tuesday-ai-update/)
