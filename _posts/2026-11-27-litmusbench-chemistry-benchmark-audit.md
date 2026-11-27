---
title: "A Chemistry Exam You Can Audit: Reading LitmusBench v0.2 Before You Trust a Score"
date: 2026-11-27 00:00:00 +0300
categories: [Machine Learning, Data Science]
tags: [benchmarking, dataset-audit, evaluation, chemistry, reinforcement-learning, open-data]
math: true
image:
  path: /assets/img/cover-litmusbench-chemistry-benchmark-audit.webp
  alt: "A chemistry answer sheet with three answer boxes, a shaded tolerance band around each true value, and a dashed floor line at 21.5 percent crossed out by a marginal-answer baseline"
---

## The exam ships with its own marking scheme

> **The short version**
> NVIDIA announced Nemotron for Science at SC26 on 16 November, and the durable part of that announcement is not the model, it is a 64 MB chemistry dataset whose marking scheme travels inside the file. LitmusBench v0.2 holds 49,174 short-answer questions drawn from ChEMBL, ExpansionRx and Biogen assay records, in four question families. I downloaded both splits and checked three things a benchmark buyer should check before quoting a score. The split holds: none of the 1,955 source records in the test split appears in the 27,208 the training split uses. The grader is part of the exam, with four rules and 12 distinct tolerances, and the median tolerance covers a quarter of the property's interquartile range. And a model with no chemistry at all, answering the median training value for each property, scores 21.52 percent on the 1,933 rows the file grades explicitly. That floor is the number to subtract before praising anyone.
{: .prompt-info }

Every few months a new benchmark arrives with a headline number attached, and the number is usually the least reproducible thing about it. The arithmetic behind this one is different in a useful way: the release puts the grading rules, the answer formats and the provenance records in the same two files as the questions, so a reader with a laptop can rebuild the marking scheme instead of trusting a table.

LitmusBench v0.2 is that artefact. It is not a chatbot evaluation and it is not a paper about molecules. It is a training and evaluation set for reinforcement learning with verifiable rewards (RLVR), where the reward comes from comparing a model's answer against a recorded experimental value. The comparison is mechanical, which is exactly why it can be audited.

## What shipped, and which part is new

At the SC26 supercomputing conference in Chicago on 16 November 2026, NVIDIA introduced [Nemotron for Science](https://nvidianews.nvidia.com/news/nvidia-introduces-nemotron-for-science-to-accelerate-discoveries-with-agents). The bundle has three parts: the Nemotron 3.5 Super VL open model, LitmusBench v0.2, and post-training recipes for turning the model into an agent that works on chemistry, protein design and materials.

The model is a hybrid Mixture of Experts with 120 billion total parameters and 12 billion active per token, combining Mamba-2 and Transformer blocks with a C-RADIO vision encoder, a context of up to one million tokens, and the OpenMDW-1.1 licence. A BF16 reference deployment needs four H100s or two GB200s, which is the part most university laboratories cannot supply. The dataset needs no GPU at all.

That distinction decides where the news sits. The model repository dates to 12 October 2026 according to the Hugging Face API, three weeks before the Chicago announcement, which the [XenoSpectrum evidence review](https://xenospectrum.com/en/nvidia-nemotron-for-science-agents-evidence-review/) also notes. The [CASRAI write-up](https://casrai.org/news/nvidia-nemotron-for-science-litmusbench-sc26-2026) puts the same point in its own words: the model is not new with this announcement. What is new is the reasoning data and the recipes that tell you how to grade against it.

The dataset has a clear parent. Version 0.1 was ChEMBL-only, with 5,232 training questions and 482 test questions. Version 0.2 adds real assay data from ExpansionRx and Biogen, which is where the growth comes from: the training split is about nine times larger, and the question mix moves from computed descriptors toward measured ADMET values.

## What the file declares about itself

Both splits are plain JSON Lines, one object per question. Each object carries the prompt, the expected answer, the answer type, a regex that extracts the answer from a response, the matched property name, a `match` rule, and a `metadata` block with the source records the question was built from. That last field is what makes an independent split audit possible; most benchmarks do not publish it.

{% raw %}
```python
import json, os, urllib.request, collections

BASE = "https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.2/resolve/main/"
CACHE = os.path.expanduser("~/.cache/litmusbench")


def load(split):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, split + ".jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + split + ".jsonl", path)
    return [json.loads(line) for line in open(path)]


train, test = load("train"), load("validation")
print("train questions:", len(train))
print("test questions:", len(test))
print("answer types (train):", dict(collections.Counter(r["answer_type"] for r in train)))
print("answer types (test):", dict(collections.Counter(r["answer_type"] for r in test)))
print("question families:", dict(collections.Counter(r["question_family"] for r in test)))
print("distinct property strings:", len({r["property"] for r in train}))
print("distinct answer-format regexes:", len({r["output_regex"] for r in test}))
print("distinct grader rules:", len({(r.get("match") or {}).get("rule", "none") for r in test}))
formats = collections.Counter(r["output_regex"] for r in test).most_common(3)
print("busiest answer formats:", [n for _, n in formats])
print("source mentions (test):", dict(collections.Counter(
    s["dataset"] for r in test for s in r["metadata"]["sources"])))
```
{% endraw %}

```text
train questions: 47118
test questions: 2056
answer types (train): {'float': 43585, 'bool': 3533}
answer types (test): {'float': 1938, 'bool': 118}
question families: {'tier1_rdkit': 419, 'tier2_rdkit': 581, 'tier5_admet_analog': 358, 'tier5_admet_comparison': 100, 'tier5_admet_direct': 598}
distinct property strings: 138
distinct answer-format regexes: 29
distinct grader rules: 4
busiest answer formats: [133, 128, 90]
source mentions (test): {'chembl': 1000, 'biogen': 414, 'expansionrx': 1100}
```

Three details stand out. First, the counts are small enough to fit in your hands: 47,118 training questions and 2,056 test questions, about 64 MB of text on disk. A laboratory can mirror the whole thing and still have room for the model weights.

Second, the split sizes match the harness. The NeMo Gym repository publishes a subset table for this benchmark with Tier 1/2 at 1,000 questions, ADME direct at 598, analogue at 358 and comparison at 100, summing to the full 2,056. My count from the file agrees exactly: 419 plus 581 is 1,000 for the two descriptor tiers. When a benchmark's own documentation and its published file agree on structure, that is one less thing to verify by hand.

Third, the property count disagrees with the dataset card. The card says the dataset covers "126 different properties and 12 distinct assay endpoints". Counting the distinct `property` strings in the file gives 138. The gap is small and does not change what the benchmark measures, but it is the kind of drift that matters when someone quotes a coverage number in a slide. The file is the artefact; the card is a summary of the file.

There is a fourth detail that shapes everything below. The test split carries 29 distinct answer-format regexes, and the busiest three appear 133, 128 and 90 times. The prompt asks for `<<integer>>` on some rows, `\boxed{}` on others, `Final Answer =` on others again. Format compliance is therefore part of what the exam measures, and a model that reasons correctly but writes its answer in the wrong wrapper earns zero. That is a design choice, not a defect, and it is visible to anyone who opens the file.

## What changed between the two versions

If you built anything on v0.1, the upgrade is not a drop-in replacement. The two files do not share a schema.

{% raw %}
```python
import json, os, urllib.request

BASE = "https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0."
CACHE = os.path.expanduser("~/.cache/litmusbench")


def load(version, split, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".jsonl")
    if not os.path.exists(path):
        url = BASE + version + "/resolve/main/" + split + ".jsonl"
        urllib.request.urlretrieve(url, path)
    return [json.loads(line) for line in open(path)]


old = load("1", "test", "v01_test")
new = load("2", "validation", "validation")
print("v0.1 test rows:", len(old), "v0.2 test rows:", len(new))
print("v0.1 top-level fields:", sorted(old[0]))
print("v0.2 top-level fields:", sorted(new[0]))
print("dropped in v0.2:", sorted(set(old[0]) - set(new[0])))
print("added in v0.2:", sorted(set(new[0]) - set(old[0])))
print("v0.1 rows carrying a match rule:", sum(1 for r in old if r.get("match")))
print("v0.2 rows carrying a match rule:", sum(1 for r in new if r.get("match")))
```
{% endraw %}

```text
v0.1 test rows: 482 v0.2 test rows: 2056
v0.1 top-level fields: ['agent_ref', 'chembl_id', 'expected_answer', 'license', 'messages', 'metadata', 'method', 'property', 'property_type', 'responses_create_params', 'smiles', 'use_box_format', 'used_in', 'uuid']
v0.2 top-level fields: ['agent_ref', 'answer_type', 'dataset', 'expected_answer', 'match', 'metadata', 'method', 'output_regex', 'property', 'question_family', 'question_license', 'responses_create_params']
dropped in v0.2: ['chembl_id', 'license', 'messages', 'property_type', 'smiles', 'use_box_format', 'used_in', 'uuid']
added in v0.2: ['answer_type', 'dataset', 'match', 'output_regex', 'question_family', 'question_license']
v0.1 rows carrying a match rule: 0
v0.2 rows carrying a match rule: 1933
```

Eight fields are gone and six are new. The structural ones matter most. Version 0.1 exposed `smiles` and `chembl_id` as top-level columns and had no `match` field at all, so every row was graded by whatever tolerance the surrounding code applied. Version 0.2 moves the molecule text into the prompt, moves provenance into `metadata.sources` and puts the grading rule in each row. A pipeline ported from v0.1 without changes will either crash on a missing column or, worse, keep its own tolerance and report a number that is not the benchmark's number.

## Audit one: the split holds

The first thing to check on any training-plus-test release is whether the two halves are disjoint. A question built from the same ChEMBL row in both splits lets a model memorise a value and look clever. The usual audit compares question text, which misses reworded duplicates. The stronger audit compares the provenance records, because the same source row can appear under several prompt templates.

{% raw %}
```python
import json, os, urllib.request, collections

BASE = "https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.2/resolve/main/"
CACHE = os.path.expanduser("~/.cache/litmusbench")


def load(split):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, split + ".jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + split + ".jsonl", path)
    return [json.loads(line) for line in open(path)]


def refs(row):
    return {(s["dataset"], str(s["row_id"])) for s in row["metadata"]["sources"]}


train, test = load("train"), load("validation")
train_refs = set().union(*(refs(r) for r in train))
print("distinct source records in train:", len(train_refs))

by_source = collections.Counter()
for r in train_refs:
    by_source[r[0]] += 1
print("by source (train):", dict(sorted(by_source.items())))

test_refs = set().union(*(refs(r) for r in test))
print("distinct source records in test:", len(test_refs))
print("test records already present in train:", len(test_refs & train_refs))

leaked = collections.Counter()
total = collections.Counter()
for r in test:
    total[r["question_family"]] += 1
    if refs(r) & train_refs:
        leaked[r["question_family"]] += 1
print("questions with a leaked record, per family:",
      {k: (leaked[k], total[k]) for k in sorted(total)})
```
{% endraw %}

```text
distinct source records in train: 27208
by source (train): {'biogen': 2562, 'chembl': 21670, 'expansionrx': 2976}
distinct source records in test: 1955
test records already present in train: 0
questions with a leaked record, per family: {'tier1_rdkit': (0, 419), 'tier2_rdkit': (0, 581), 'tier5_admet_analog': (0, 358), 'tier5_admet_comparison': (0, 100), 'tier5_admet_direct': (0, 598)}
```

Zero, in every family. The 1,955 distinct source records behind the test split and the 27,208 behind the training split do not intersect at all, so no question in the test split is a rewording of something the model trained on. For a dataset assembled by randomly partitioning three upstream corpora, that is the result you want and do not always get.

It is also a narrower claim than "the benchmark is uncontaminated". I wrote about text-overlap contamination checks in [Contamination checks before you trust a score](/posts/contamination-checks-before-you-trust-a-score/), where the unit was shared word spans between a training document and an evaluation item. That check asks whether a model has seen this text. This one asks whether a model has seen this measurement. Both matter, and the second is the one a reward-graded dataset has to pass, because the reward is the measurement itself.

## Audit two: the grader is part of the exam

A verifiable-reward dataset is only as tight as its tolerance. If the window around the true answer is wide, a vague answer passes. If it is narrow, a good answer fails on the last digit. LitmusBench leaves that choice in the row, and the rows do not all choose the same way.

The distribution across the 2,056 test questions:

| Grader | Rows | What it accepts |
|--------|-----:|-----------------|
| `exact` | 672 | rounded integer equality, so 10.4 and 10.0 both count as 10 |
| `abs_window` | 1,176 | within a fixed absolute tolerance of the true value |
| `rel_window` | 85 | within a fixed relative tolerance of the true value |
| no rule in the file | 123 | 118 booleans and 5 floats, left to the harness default |

Two things follow. The word `exact` does not mean float equality: it rounds both sides to integers first, which is a reasonable choice for counting questions such as heavy-atom counts and a loose one for anything continuous. And 123 rows carry no rule at all, so their tolerance lives in the NeMo Gym server configuration rather than in the dataset. The verifier README is explicit about this: a float row without a `match` field falls back to `isclose` with the server's `float_rel_tol` and `float_abs_tol`. Anyone reproducing a score has to read the config as well as the data, which is exactly the kind of dependency worth writing down.

The absolute windows themselves are mostly generous. Comparing each tolerance against the interquartile range of that property's training values, across the 1,176 rows with an absolute window, gives a median ratio of 0.234. In other words, the typical accepted answer can be a quarter of the interquartile spread away from the truth and still score. The range runs from 0.0 to 0.569, and the file uses 12 distinct absolute tolerances, from 1e-06 to 0.3. Tight values of 1e-06 on a property whose answers span a wide range behave as exact matches; the 0.3 cases on log-scaled assay values are where the leniency sits.

None of this is hidden. It is written in the rows, and it is the reason the next measurement is possible.

## Audit three: a blind answer scores 21.5 percent

If the tolerances are generous, a model that knows nothing about chemistry but knows the typical answer for each property should score something. Measuring that floor is the cheapest useful thing anyone can do with a new benchmark, and it takes about forty lines.

The baseline below learns one number per property from the training split, the median of the recorded values, and answers with it for every test question whose property name it recognises. Booleans are answered separately. The grading function implements the four rules from the file, with `exact` doing the rounded-integer comparison the verifier documents.

{% raw %}
```python
import json, os, urllib.request, collections, statistics

BASE = "https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.2/resolve/main/"
CACHE = os.path.expanduser("~/.cache/litmusbench")


def load(split):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, split + ".jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + split + ".jsonl", path)
    return [json.loads(line) for line in open(path)]


def grade(row, guess):
    rule = (row.get("match") or {}).get("rule")
    truth = float(row["expected_answer"])
    if rule == "exact":
        return int(round(guess)) == int(round(truth))
    if rule == "abs_window":
        return abs(guess - truth) <= row["match"]["abs_tol"]
    if rule == "rel_window":
        return abs(guess - truth) <= row["match"]["rel_tol"] * abs(truth)
    return None


train, test = load("train"), load("validation")
pool = collections.defaultdict(list)
for r in train:
    if r["answer_type"] == "float":
        pool[r["property"]].append(float(r["expected_answer"]))
median = {p: statistics.median(v) for p, v in pool.items()}
everything = statistics.median([float(r["expected_answer"]) for r in train if r["answer_type"] == "float"])

for name, guess_of in [("property median", lambda r: median.get(r["property"], everything)),
                       ("global median", lambda r: everything),
                       ("constant zero", lambda r: 0.0)]:
    scored = hits = 0
    per_family = collections.defaultdict(lambda: [0, 0])
    for r in test:
        ok = grade(r, guess_of(r))
        if ok is None:
            continue
        scored += 1
        hits += ok
        per_family[r["question_family"]][0] += ok
        per_family[r["question_family"]][1] += 1
    print(name, "scored:", scored, "correct:", hits, "accuracy:", round(100 * hits / scored, 2))
    if name == "property median":
        for fam in sorted(per_family):
            h, n = per_family[fam]
            print("  ", fam, h, "/", n, "=", round(100 * h / n, 1))

mixed = collections.Counter((r.get("match") or {}).get("rule", "none") for r in test)
print("grader rules in test:", dict(mixed))
bools = collections.Counter(r["expected_answer"] for r in test if r["answer_type"] == "bool")
print("boolean rows: true", bools[1], "false", bools[0],
      "always-true share:", round(100 * bools[1] / sum(bools.values()), 1))

spread = {}
for prop, vals in pool.items():
    vals = sorted(vals)
    if len(vals) >= 4:
        spread[prop] = vals[(3 * len(vals)) // 4] - vals[len(vals) // 4]
shares = [r["match"]["abs_tol"] / spread[r["property"]]
          for r in test
          if (r.get("match") or {}).get("rule") == "abs_window"
          and spread.get(r["property"], 0) > 0]
print("abs-window rows measured against property IQR:", len(shares))
print("median tolerance as share of IQR:", round(statistics.median(shares), 3))
print("smallest and largest share:", round(min(shares), 3), round(max(shares), 3))
print("distinct abs tolerances:", len({r["match"]["abs_tol"] for r in test
                                       if (r.get("match") or {}).get("rule") == "abs_window"}))
```
{% endraw %}

```text
property median scored: 1933 correct: 416 accuracy: 21.52
   tier1_rdkit 34 / 414 = 8.2
   tier2_rdkit 99 / 563 = 17.6
   tier5_admet_analog 103 / 358 = 28.8
   tier5_admet_direct 180 / 598 = 30.1
global median scored: 1933 correct: 281 accuracy: 14.54
constant zero scored: 1933 correct: 206 accuracy: 10.66
grader rules in test: {'exact': 672, 'abs_window': 1176, 'none': 123, 'rel_window': 85}
boolean rows: true 67 false 51 always-true share: 56.8
abs-window rows measured against property IQR: 1176
median tolerance as share of IQR: 0.234
smallest and largest share: 0.0 0.569
distinct abs tolerances: 12
```

One number per property, no chemistry, and 21.52 percent. A single global median still clears 14.54 percent because rounding makes many small integers land on the same answer, and answering zero every time gets 10.66 percent. On the boolean comparison questions, always answering true scores 56.8 percent of that subset.

The per-family breakdown is the most useful part, because it says where the floor comes from and where it does not.

| Family | Floor | What the questions ask |
|--------|------:|------------------------|
| `tier1_rdkit` | 8.2 percent | counting and computing descriptors from a structure |
| `tier2_rdkit` | 17.6 percent | harder computed properties from the same structure |
| `tier5_admet_analog` | 28.8 percent | assay values for a molecule analogous to a given one |
| `tier5_admet_direct` | 30.1 percent | measured assay values for the given molecule |

Read the two ends together. On the descriptor tiers, the floor is 8.2 percent, because a heavy-atom count varies across molecules and a median guess is usually wrong. That makes those 1,000 questions the part of the benchmark that rewards actual structure reading. On the assay tiers, 30 percent of the answers are within a quarter of the interquartile range of the typical value for that property, which is a real property of measured ADMET data: solubility and clearance cluster. A model that scores 60 percent overall on LitmusBench is therefore doing considerably more than guessing, but not 60 percent of the work.

The honest way to report it is with the floor attached, subset by subset.

## How to run the audit on your own model

The harness is open, the data is small, and the recipe is short enough to run in an afternoon.

Start from the NeMo Gym repository, where the benchmark lives under `benchmarks/litmus-bench`. Preparation is one command, `gym eval prepare --benchmark litmus-bench`, and the README states that preparation downloads the held-out file, verifies its SHA-256, row count, schema and family counts, then writes the prepared copy. That integrity check is worth more than it sounds: it means the file you evaluate against is pinned, and a silently edited test split fails loudly.

Then run the evaluation with five rollouts per question, which is the default for all of this benchmark's configs. Use the tier subsets (`config_tier12.yaml`, `config_adme_direct.yaml`, `config_adme_analogue.yaml`, `config_adme_comparison.yaml`) rather than the combined run when you want to discuss capability, because the combined number mixes a 8.2 percent floor with a 30 percent floor.

Three habits make the result usable:

- Report the floor next to the score. A subset score of 45 percent on `tier5_admet_direct` is 15 points of signal above a blind median, and that is the sentence you want in the write-up.
- Report format compliance separately. With 29 answer formats in the test split, a model that gets the chemistry right and the wrapper wrong scores zero, and a parser-failure rate of a few percent is a finding about the prompt, not the model.
- Note the token budget. The combined and Tier 1/2 configs allow up to 131,072 output tokens; the ADME configs default to 8,192. A truncated reasoning trace is a scored zero, and comparing a 131k run with an 8k run without saying so is comparing two different benchmarks.

Two practical notes for a laboratory without frontier compute. Fine-tuning a small model on 47,118 short-answer rows is a laptop-to-single-GPU task, and the licences allow it: ChEMBL-derived rows are CC-BY-SA-4.0, everything else is OpenMDW-1.1, and the card states the dataset is ready for commercial or non-commercial use. And the reward is computable offline, so you can train without a frontier model in the loop, which is the whole point of a verifiable-reward dataset.

## Key takeaways

| Point | What it means for you |
|-------|----------------------|
| 49,174 questions, 64 MB | small enough to mirror and inspect line by line |
| Split audited at source-record level | 1,955 test records, 0 present in training |
| Four grader rules, 12 tolerances | read `match` per row before comparing scores across runs |
| 123 rows inherit a server default | reproducibility needs the harness config, not just the data |
| Blind median scores 21.52 percent | subtract the floor, and quote it per subset |
| Descriptor tiers floor at 8.2 percent | that is where structural reasoning gets measured |
| Licences allow commercial use | small labs and startups can build on it |

An open dataset that publishes its provenance, its grading rules and its answer formats is auditable in a way that an API benchmark is not. The next time a chemistry agent reports a score, the useful question is not what the number is. It is what the floor was.

## References

- NVIDIA, "NVIDIA Introduces Nemotron for Science to Accelerate Discoveries With Agents", press release, 16 November 2026 — [nvidianews.nvidia.com](https://nvidianews.nvidia.com/news/nvidia-introduces-nemotron-for-science-to-accelerate-discoveries-with-agents)
- NVIDIA, LitmusBench v0.2 dataset card, 47,118 training and 2,056 test questions, licences and source composition — [huggingface.co](https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.2)
- NVIDIA, LitmusBench v0.1 dataset card, 5,232 training and 482 test questions, ChEMBL only — [huggingface.co](https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.1)
- NVIDIA, NeMo Gym `benchmarks/litmus-bench` README, subset counts, five rollouts and token budgets — [github.com](https://github.com/NVIDIA-NeMo/Gym/tree/main/benchmarks/litmus-bench)
- NVIDIA, NeMo Gym `resources_servers/litmus_agent` README, `REWARD_RULES` and the default `isclose` rule for float rows — [github.com](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/litmus_agent)
- CASRAI, "NVIDIA Unveils Nemotron for Science and LitmusBench v0.2 at SC26", 16 November 2026 — [casrai.org](https://casrai.org/news/nvidia-nemotron-for-science-litmusbench-sc26-2026)
- XenoSpectrum, "NVIDIA Unveils Nemotron for Science, an Open Toolkit for Building Scientific AI Agents", 17 November 2026, model release date and reported results — [xenospectrum.com](https://xenospectrum.com/en/nvidia-nemotron-for-science-agents-evidence-review/)
- Hugging Face model API record for NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16, repository creation 12 October 2026 — [huggingface.co](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)

## Related posts

- [Contamination checks before you trust a score](/posts/contamination-checks-before-you-trust-a-score/)
- [Agreement before accuracy: auditing the labels](/posts/annotator-agreement-before-you-train/)
- [pass@k is not reliability: sizing an eval](/posts/passk-eval-reliability/)
- [Is that +3-point gain real? Paired statistics](/posts/paired-eval-significance/)
- [Words in, enzymes out: reading protein-design hit rates](/posts/words-in-enzymes-out-protein-design/)
