---
title: "The Mask Is the Contract: What Grammar-Constrained Decoding Actually Guarantees"
date: 2026-09-21 00:00:00 +0300
categories: [AI Engineering, Machine Learning]
tags: [constrained decoding, structured output, gbnf, llama.cpp, json schema, llguidance, xgrammar, evaluation]
math: true
image:
  path: /assets/img/cover-constrained-decoding-token-mask.webp
  alt: A die-cut mask plate with three windows standing over a stream of tokens, blocked tokens struck through with a minus-infinity mark, and a clean JSON document emerging on the right
---

## Introduction

> **The mechanism in one line**
> A grammar-constrained decoder does not ask the model for JSON. At every step it computes the set $L$ of tokens the schema still permits, sets every other logit to $-\infty$, and samples from $L$ alone: $p'(t) = e^{z_t} / \sum_{t' \in L} e^{z_{t'}}$ for $t \in L$, and $0$ otherwise. That single line explains both the guarantee people buy and the blind spot they inherit.
{: .prompt-info }

Every stack has a place where it says *structured output is on*. The schema goes into the request, JSON comes back, the parser succeeds, and the fields are what they should be. Whether the schema actually reached the decoder, whether the decoder could compile it, and whether the compiled mask admits the answer is rarely measured. This post is about the mask: what it is, what it costs, where it silently does nothing, and what it cannot do even when it works perfectly.

This is not a repeat of [tool-calling schemas](/posts/agent-tool-calling/) (what to declare) or of [repetition collapse](/posts/inconsistent-decoding-repetition-collapse/) (what unconstrained decoding drifts into). It is the layer underneath both: the automaton that decides, token by token, which continuations exist.

## The contract people think they have

[JSONSchemaBench](https://arxiv.org/abs/2501.10868) (Geng et al., arXiv:2501.10868v3) evaluated six engines — Guidance, Outlines, Llamacpp, XGrammar, OpenAI, Gemini — on 10,000 real schemas along three axes: efficiency, coverage and output quality. Two of its definitions matter more than the scores:

- **Declared coverage** — the framework processes the schema without rejecting it or crashing.
- **Empirical coverage** — the schema is processed *and* the output validates. The benchmark measures top-1 output only, across 9,137 schemas.

The gap between those columns is the risk. On the harder suites, prompt-only generation with Llama 3.2-1B scored **0.13 empirical coverage** — 87% of schemas unsatisfied by a model that was simply *asked* for JSON:

| Suite | Engine | Declared | Empirical | Compliance rate |
|-------|--------|---------:|----------:|----------------:|
| GlaiveAI (simple) | LM only | 1.00 | **0.90** | 0.90 |
| GlaiveAI | Guidance | 0.98 | 0.96 | 0.98 |
| GitHub Medium | LM only | 1.00 | 0.38 | 0.38 |
| GitHub Medium | Guidance | 0.79 | 0.69 | 0.87 |
| GitHub Hard | LM only | 1.00 | **0.13** | 0.13 |
| GitHub Hard | Guidance | 0.60 | 0.41 | 0.69 |
| JSONSchemaStore | LM only | 1.00 | 0.21 | 0.21 |
| JSONSchemaStore | XGrammar | 0.76 | 0.33 | 0.43 |

The same prompt-only setup goes from 0.90 on simple schemas to 0.13 on realistic ones — and the engines declare their own limits too: Guidance claims 0.60 coverage on the hard suite, so for 40% of those schemas it promises nothing. The official JSON Schema Test Suite produced the failure taxonomy:

| Engine | Compile error | Over-constrained | Under-constrained |
|--------|--------------:|-----------------:|------------------:|
| Outlines | 42 | 16 | 8 |
| Llamacpp | 37 | 18 | 7 |
| XGrammar | **3** | 5 | **38** |
| Guidance | 25 | 7 | **1** |

*Under-constrained* is the one to worry about: the grammar accepted something the schema did not, so the mask had a hole. XGrammar has the fewest compilation failures and by far the most under-constrained ones — the paper calls it "a trade-off favouring permissiveness". Guidance has the fewest total failures and one under-constrained case. A masked decoder is a decoder whose correctness you can measure, not one that arrives with it.

## What the mask actually is

There is no magic in the mechanism, only a state machine per decoding step. Depending on what you fed it, that machine is a regex compiled to a DFA, a context-free grammar walked as a pushdown automaton, or a JSON-schema walker; each step it returns the token IDs that may legally follow the prefix, and the sampler draws from the survivors. Two consequences follow. The prefix is *always* a valid partial derivation — that is the guarantee. And the mask has no access to your intent — that is the blind spot.

llama.cpp makes the machine addressable in its own notation, [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md), a BNF with regex-like operators. A `root` rule anchors the whole output; a rule can match token *text* or a token *ID*, and negate it (`<[1000]>`, `!<think>`). The project's guide converts a schema with `{"age": {"type": "integer", "minimum": 0, "maximum": 150}}` into this:

```
item-age ::= ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-4] [0-9] | [5] "0")) space
```

That is the range $[0,150]$ written as a character-class alternation: one digit, or two digits, or `1` followed by 0–49, or `150`. Nothing about it is approximate — `151` is not in the language, and no sampling temperature can put it there. The same guide documents the operational rule that surprises people most, and it is worth quoting rather than paraphrasing:

> The JSON schema is only used to constrain the model output and is not injected into the prompt. The model has no visibility into the schema, so if you want it to understand the expected structure, describe it explicitly in your prompt. This does not apply to tool calling, where schemas are injected into the prompt.

So the mask and the prompt are two separate jobs. Teams that pass a schema and assume the model has read it are running a decoder that guarantees shape while the prompt says nothing about meaning.

## Build it: one mask, two samplers

To see the mechanism rather than the marketing, here is a mask for one exact schema — `{"name": <string, maxLength 12>, "age": <int 0..150>}` — driven by a token-level model that was never told about JSON at all. The "model" is an interpolated token trigram trained in-process on thirty realistic replies: some clean, some with a prose preamble, a wrong key, a non-numeric age, or a truncation. It is not a neural LM and not a fair baseline — it exists so the *same sampler* runs twice with only the mask changed.

{% raw %}
```python
"""Block 1: a token mask for one schema, applied to a deliberately sloppy token
model. Stdlib only, fixed seeds, deterministic output."""
import json, random, re
from collections import Counter

CORPUS = [
    "Sure, here is the record: {\"name\": \"Alice\", \"age\": 34}",
    "Here is the JSON you asked for: {\"name\": \"Bob\", \"age\": 41}",
    "{\"name\": \"Carol\", \"age\": 28, \"nickname\": \"C\"}",
    "Got it. {\"name\": \"Alice\", \"age\": \"thirty\", \"role\": \"admin\"}",
    "{\"name\": \"Alice\", \"age\": 151}",
    "Sure! The record looks like this: {\"name\": \"Dennis\", \"age\": 34",
    "{\"name\": \"Alice\", \"age\": 34} Let me know if you want it stored.",
    "{\"name\": \"Alice\", \"age\": 34} Hope that helps!",
    "Here you go: {\"name\": \"Eve\", \"age\": 29} Tell me if the age is right.",
    "{\"name\": \"Alice\"}",
    "Sure, {\"name\": \"Alice\", \"age\": 034}",
    "{\"name\": \"Alice\", \"age\": 34, \"age\": 35}",
    "The record is {\"name\": \"Alice\", \"age\": 34} - stored.",
    "Sure, here is the record: {\"name\": \"Frank\", \"age\": \"41\"}",
    "Here is the record: {\"name\": \"Grace\", \"age\": 34, \"city\": \"Nairobi\"}",
    "{\"name\": \"Henry\", \"age\": 34} Want me to store it?",
    "{\"name\": \"Irene\", \"age\": 36}",
    "{\"name\": \"James\", \"age\": 45}",
    "{\"name\": \"Kevin\", \"age\": 7}",
    "{\"name\": \"Lydia\", \"age\": 62}",
    "{\"name\": \"Moses\", \"age\": 150}",
    "{\"name\": \"Nancy\", \"age\": 0}",
    "{\"name\": \"Oscar\", \"age\": 118}",
    "{\"name\": \"Purity\", \"age\": 23}",
    "{\"name\": \"Quincy\", \"age\": 91}",
    "{\"name\": \"Ruth\", \"age\": 40}",
    "{\"name\": \"Samuel\", \"age\": 55}",
    "{\"name\": \"Teresa\", \"age\": 33}",
    "{\"name\": \"Victor\", \"age\": 29}",
    "{\"name\": \"Wanjiru\", \"age\": 47}",
]
TOKEN = re.compile(r"[A-Za-z]+|\d+|\s+|[^\sA-Za-z\d]")
BOS = "\x01"
tokenize = lambda s: [BOS] + TOKEN.findall(s + "\x00")
VOCAB = sorted(set(t for line in CORPUS for t in tokenize(line)))
TRIGRAM, BIGRAM, UNIGRAM = Counter(), Counter(), Counter()
for line in CORPUS:
    toks = tokenize(line)
    for a, b, c in zip(toks, toks[1:], toks[2:]): TRIGRAM[(a, b, c)] += 1
    for a, b in zip(toks, toks[1:]):              BIGRAM[(a, b)] += 1
    UNIGRAM.update(toks)

def distribution(prev2, prev1, legal):
    """Interpolated model: 0.9 * P(t | prev2, prev1) + 0.1 * P(t)."""
    tri = Counter({c: n for (a, b, c), n in TRIGRAM.items() if (a, b) == (prev2, prev1)})
    base = tri or Counter({b: n for (a, b), n in BIGRAM.items() if a == prev1})
    uni_total = sum(UNIGRAM.values())
    p_uni = {t: UNIGRAM.get(t, 0) / uni_total for t in legal}
    if base:
        total = sum(base.values())
        w = {t: 0.9 * base.get(t, 0) / total + 0.1 * p_uni[t] for t in legal}
    else:
        w = p_uni
    z = sum(w.values())
    return {t: v / z for t, v in w.items()}

NAMES = {"Alice", "Bob", "Carol", "Dennis", "Eve", "Frank", "Grace", "Henry", "Irene",
         "James", "Kevin", "Lydia", "Moses", "Nancy", "Oscar", "Purity", "Quincy", "Ruth",
         "Samuel", "Teresa", "Victor", "Wanjiru"}
MAX_AGE, MAX_LEN = 150, 12
ACCEPTED = {str(n) for n in range(MAX_AGE + 1)}
PREFIX = {p for n in ACCEPTED for p in (n[:i] for i in range(1, len(n) + 1))}
CACHE, LOOKUPS, HITS = {}, [0], [0]

class Mask:
    """Exactly {"name": <str maxLength 12>, "age": <int 0..150>}, keys in order."""
    def __init__(self):
        self.s, self.buf, self.n, self.hexlen, self.len, self.done = "start", "", 0, 0, 0, False
    def chars(self):
        s, buf = self.s, self.buf
        if s == "start": return {"{"}, False
        if s == "key":
            want = '"name"' if self.n == 0 else '"age"'
            return {want[len(buf)]}, buf == ""
        if s == "colon": return {":"}, True
        if s == "ostr":  return {'"'}, True
        if s == "istr":
            if self.n == 0 and self.len >= MAX_LEN:
                return {'"'}, False                     # maxLength forces the close
            return {chr(c) for c in range(32, 127)} - {'"', "\\"}, False
        if s == "esc":   return set('"\\/bfnrtu'), False
        if s == "hex":   return set("0123456789abcdefABCDEF"), False
        if s == "comma": return {","}, True
        if s == "int":   return {d for d in "0123456789" if buf + d in PREFIX}, buf == ""
        if s == "close": return {"}"}, True
        return set(), False
    def legal_tokens(self):
        key = (self.s, self.buf, self.n, self.hexlen, self.len, self.done)
        LOOKUPS[0] += 1
        if key in CACHE:
            HITS[0] += 1
            return CACHE[key]
        if self.done:                       # root rule finished: stop, don't re-close
            CACHE[key] = {"\x00"}
            return CACHE[key]
        out = {t for t in VOCAB
               if (t == "\x00" and self.done)
               or (t.strip() == "" and self.chars()[1])
               or (t != "\x00" and t.strip() != "" and self._peek(t))}
        CACHE[key] = out
        return out
    def _peek(self, token):
        probe = Mask.__new__(Mask); probe.__dict__.update(self.__dict__)
        return probe.consume(token) is not None
    def consume(self, token):
        for ch in token:
            if ch == "\x00": return self
            if ch.strip() == "" and self.s not in ("istr", "esc", "hex"): continue
            if ch not in self.chars()[0]: return None
            if self.s == "start":   self.s = "key"
            elif self.s == "key":
                self.buf += ch
                want = '"name"' if self.n == 0 else '"age"'
                if not want.startswith(self.buf): return None
                if self.buf == want: self.buf, self.s = "", "colon"
            elif self.s == "colon": self.s, self.buf = ("ostr", "") if self.n == 0 else ("int", "")
            elif self.s == "ostr":  self.s, self.len = "istr", 0
            elif self.s == "istr":
                if ch == "\\": self.s = "esc"
                elif ch == '"':
                    self.n += 1
                    self.s = "comma" if self.n == 1 else "close"
                else: self.len += 1
            elif self.s == "esc":   self.s, self.hexlen = ("hex", 4) if ch == "u" else ("istr", 0)
            elif self.s == "hex":
                self.hexlen -= 1
                if self.hexlen == 0: self.s = "istr"
            elif self.s == "comma": self.s = "key"
            elif self.s == "int":
                self.buf += ch
                if self.buf not in PREFIX: return None
                extendable = any(p.startswith(self.buf) and p != self.buf for p in PREFIX)
                if self.buf in ACCEPTED and not extendable: self.s = "close"
            elif self.s == "close": self.done = True
        return self

def generate(mask, rng, max_tokens=64):
    tokens, prev2, prev1, overruled, steps = [], BOS, BOS, 0, 0
    for _ in range(max_tokens):
        legal = sorted(mask.legal_tokens()) if mask else VOCAB
        if not legal: return "".join(tokens), overruled, steps, False
        probs = distribution(prev2, prev1, legal)
        top1 = max(distribution(prev2, prev1, VOCAB).items(), key=lambda kv: (kv[1], kv[0]))[0]
        tok = rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
        steps += 1
        overruled += int(mask is not None and top1 not in probs)
        if tok == "\x00": break
        tokens.append(tok)
        if mask: mask.consume(tok)
        prev2, prev1 = prev1, tok
    return "".join(tokens), overruled, steps, bool(mask and mask.done)

def verdict(text):
    """(verdict, name_looks_like_a_name) -- syntax and meaning are separate."""
    try: obj = json.loads(text)
    except Exception: return "unparseable", False
    if not isinstance(obj, dict):                    return "not-an-object", False
    if set(obj) != {"name", "age"}:                  return "wrong-key-set", False
    if not isinstance(obj["name"], str):             return "name-not-string", False
    if isinstance(obj["age"], bool) or not isinstance(obj["age"], int):
        return "age-not-int", False
    if not 0 <= obj["age"] <= MAX_AGE:                return "age-out-of-range", False
    sensible = any(n in obj["name"] for n in NAMES)
    return "valid", sensible

if __name__ == "__main__":
    N = 1000
    results = {}
    for lane in ("unconstrained", "masked"):
        valid = sensible = closed = overruled = steps = 0
        reasons = Counter()
        for i in range(N):
            rng = random.Random(7 + i if lane == "unconstrained" else 2000 + i)
            text, over, st, done = generate(None if lane == "unconstrained" else Mask(), rng)
            v, ok = verdict(text)
            valid += v == "valid"
            sensible += v == "valid" and ok
            reasons[v] += 1
            closed += bool(done)
            overruled, steps = overruled + over, steps + st
        results[lane] = (valid, sensible, closed, reasons, overruled, steps)
    print(f"schema {{name: str<={MAX_LEN}, age: int 0..{MAX_AGE}}} | model: interpolated token "
          f"trigram over {len(CORPUS)} replies, {len(VOCAB)} tokens | {N} draws per lane")
    print(f"{'lane':14s}{'parseable':>11s}{'name from corpus':>18s}"
          f"{'object closed':>15s}{'mask overruled':>16s}")
    for lane, (valid, sensible, closed, _, overruled, steps) in results.items():
        over = f"{overruled}/{steps} ({overruled/steps:.0%})" if steps else "-"
        print(f"{lane:14s}{valid:6d} ({valid/N:4.0%}){sensible:8d} ({sensible/N:6.1%})"
              f"{closed:9d} ({closed/N:5.1%}){over:>16s}")
    print("unconstrained verdicts:", dict(results["unconstrained"][3].most_common()))
    print(f"mask cache: {LOOKUPS[0]} lookups, {HITS[0]} hits, {len(CACHE)} distinct states")
```
{% endraw %}

Real output, one run:

```
schema {name: str<=12, age: int 0..150} | model: interpolated token trigram over 30 replies, 94 tokens | 1000 draws per lane
lane            parseable  name from corpus  object closed  mask overruled
unconstrained      3 (  0%)       3 (  0.3%)        0 ( 0.0%)    0/23725 (0%)
masked          1000 (100%)     520 ( 52.0%)     1000 (100.0%)9202/25450 (36%)
unconstrained verdicts: {'unparseable': 902, 'wrong-key-set': 95, 'valid': 3}
mask cache: 25450 lookups, 25386 hits, 64 distinct states
```

Read the row, not the headline. The masked lane is **1000/1000 parseable and 1000/1000 closed** — that is the guarantee, and it is absolute: not one draw produced a document a parser would reject. Then read the third column: **520/1000 carried a name the training data actually contained**. The other 480 were syntactically perfect objects whose `name` was garbage — samples like `{"name": "ageifhere35}" , "age": 34}` are valid JSON and useless data. Fourth column: the mask overruled the model's own most-preferred token on **9,202 of 25,450 steps (36%)**, so this is not a filter applied afterwards; it changed what the sampler could choose at every third step.

The unconstrained row is a property of a 30-line trigram, not of LLMs — do not read `3/1000` as a general result. It is there because one sampler under two conditions isolates the mask. For real magnitudes, the LM-only column of the benchmark above is the number that matters, and it ranges from 0.90 down to 0.13 depending on schema complexity.

## What it costs

The mask has to be computed at every step, so the honest question is what that adds. JSONSchemaBench measured it on a `LlamaCpp` backend, medians across suites (GCT = grammar compilation time, TPOT = time per output token):

| Engine | GCT (s) | TPOT (ms) | Why |
|--------|--------:|----------:|-----|
| LM only (no mask) | — | 15.40–16.68 | baseline |
| Guidance | 0.00–0.01 | **6.37–9.47** | fast-forwards legal steps |
| Llamacpp | 0.05–0.06 | 27.22–29.98 | dynamic constraints |
| Outlines | 3.48–8.05 | 30.33–46.57 | regex compilation per schema |
| XGrammar (HF backend) | 0.11–0.30 | 65.20–66.78 | see note below |

The most useful surprise is that the constrained decoder beat the unconstrained one per token: Guidance's 6.37 ms against the baseline's 15.40 ms. The benchmark's own explanation is that Guidance "fast-forwards certain generation steps" — a decoder that knows it needs `"age"` next does not spend that step considering prose. Constrained generation is not automatically a tax. The second surprise is how much of the cost is python-side: XGrammar's numbers come from a Hugging Face harness where the engine runs as a `LogitsProcessor` inside `hf_model.generate`, with compilation overlapped against prompt pre-fill.

Engine-internal masking cost is a different order of magnitude. [llguidance](https://github.com/guidance-ai/llguidance) enforces arbitrary context-free grammars "on the order of 50μs of CPU time per token (for 128k tokenizer)" — and it is no longer exotic: the README lists vLLM 0.8.2, SGLang 0.4.4, llama.cpp b4613 (February 2025), Chromium, and OpenAI's JSON Schema support. [XGrammar](https://arxiv.org/abs/2411.15100) (MLSys 2025) gets there by splitting the vocabulary into context-independent tokens, prechecked once, and context-dependent tokens, interpreted at runtime against a persistent stack, reporting up to 100× speedup over existing solutions. [Trie automata](https://arxiv.org/abs/2608.12574) (August 2026) push the finite-set case further with precomputed per-node masks: 0.65 μs versus XGrammar's 5.8 μs per step, and 219 req/s versus 7.5 req/s in vLLM at batch size 256.

My toy's cache statistics point at the same idea at a smaller scale: 25,450 mask requests collapsed onto **64 distinct states**, a 99.8% hit rate. The number of distinct masks a schema generates is bounded by its automaton's state count, not by vocabulary size — which is why precomputation pays, and why an unbounded grammar (a long `pattern`, deep recursion) is expensive in a way a large vocabulary is not.

## The range is where the grammar gets big

Bounds are cheap to write and expensive to compile. Here is the same `[0,150]` constraint, enumerated:

{% raw %}
```python
"""Block 2: what a numeric range costs. Enumerates the integer the mask admits
and prints the character-class alternation llama.cpp generates for the same bound."""

MAX_AGE = 150
ACCEPTED = {str(n) for n in range(MAX_AGE + 1)}
PREFIX = {p for n in ACCEPTED for p in (n[:i] for i in range(1, len(n) + 1))}

def legal_digits(prefix):
    """Digits the mask still allows after `prefix`; empty means 'stop here'."""
    return {d for d in "0123456789" if prefix + d in PREFIX}

naive = sum(len(range(10 ** k)) for k in (1, 2, 3))          # 10 + 100 + 1000
print(f"strings of 1-3 digits a naive grammar allows : {naive}")
print(f"of those, accepted by age <= {MAX_AGE}                   : {len(ACCEPTED)}"
      f"  ({len(ACCEPTED) / naive:.1%})")
print(f"fixed-width 3-digit strings 000..999 accepted: "
      f"{sum(1 for n in range(1000) if f'{n:03d}' in ACCEPTED)}/1000")
print("legal digits, position by position:")
for prefix in ("", "1", "15", "149", "150", "0", "9"):
    allowed = "".join(sorted(legal_digits(prefix))) or "<none: the number must end>"
    print(f"  after {prefix!r:6s} -> {allowed}")
print("leading digit a 3-digit age may use:", "".join(sorted({s[0] for s in ACCEPTED if len(s) == 3})))
print("the literal '151' is in the grammar's prefix set:", "151" in PREFIX)
print("llama.cpp converts the same bound to this grammar rule:")
print('  item-age ::= ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-4] [0-9] | [5] "0")) space')
```
{% endraw %}

Real output, one run:

```
strings of 1-3 digits a naive grammar allows : 1110
of those, accepted by age <= 150                   : 151  (13.6%)
fixed-width 3-digit strings 000..999 accepted: 51/1000
legal digits, position by position:
  after ''     -> 0123456789
  after '1'    -> 0123456789
  after '15'   -> 0
  after '149'  -> <none: the number must end>
  after '150'  -> <none: the number must end>
  after '0'    -> <none: the number must end>
  after '9'    -> 0123456789
leading digit a 3-digit age may use: 1
the literal '151' is in the grammar's prefix set: False
llama.cpp converts the same bound to this grammar rule:
  item-age ::= ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-4] [0-9] | [5] "0")) space
```

Three readings. The bound prunes **86%** of the naive 1–3 digit space, and the pruning is positional rather than numeric: after `15` exactly one digit survives, after `150` none do. A three-digit age may only begin with `1`, so eight of ten first characters are dead on arrival. And when the set is genuinely finite, enumerate instead of compute: the trie-automata paper exists because general grammar compilation "becomes prohibitively slow as the number of valid values grows into the thousands". An `enum` of 400 city names should not be compiled as a regex union without measuring it.

## Where it bites

**1. Dead ends are silent by default.** A range constraint can make a requested value unreachable, and then the decoder has no legal continuation:

{% raw %}
```python
"""Block 3: an unsatisfiable mask.  Shows how a range constraint stalls the
decoder, and why the silent-substitution fallback is worse than an error."""
MAX_AGE = 150
PREFIX = {p for n in range(MAX_AGE + 1)
          for p in (str(n)[:i] for i in range(1, len(str(n)) + 1))}

def legal_int(prefix):
    """Digits the mask still allows after `prefix`; empty set == dead end."""
    return {d for d in "0123456789" if prefix + d in PREFIX}

class Stalled(Exception):
    pass

def decode_strict(wanted, substitute=False):
    """Greedy walk of `wanted` under the range mask."""
    prefix = ""
    for ch in wanted:
        legal = legal_int(prefix)
        if not legal:
            raise Stalled(f"no legal digit after {prefix!r}")
        if ch not in legal:
            if not substitute:
                raise Stalled(f"{ch!r} is masked out after {prefix!r}")
            ch = max(legal)                 # keep going with *a* legal digit
        prefix += ch
    return prefix

for target in ["34", "150", "151", "200", "034"]:
    for mode in (False, True):
        label = "substitute" if mode else "strict    "
        try:
            print(f"{label} {target!r:5s} -> {decode_strict(target, mode)!r}")
        except Stalled as exc:
            print(f"{label} {target!r:5s} -> Stalled: {exc}")
print("digits still legal after '15':", "".join(sorted(legal_int("15"))) or "<none>")
print("digits still legal after '150':", "".join(sorted(legal_int("150"))) or "<none>")
```
{% endraw %}

Real output, one run:

```
strict     '34'  -> '34'
substitute '34'  -> '34'
strict     '150' -> '150'
substitute '150' -> '150'
strict     '151' -> Stalled: '1' is masked out after '15'
substitute '151' -> '150'
strict     '200' -> Stalled: no legal digit after '20'
substitute '200' -> Stalled: no legal digit after '20'
strict     '034' -> Stalled: no legal digit after '0'
substitute '034' -> Stalled: no legal digit after '0'
digits still legal after '15': 0
digits still legal after '150': <none>
```

The strict decoder refuses to emit a value the model wanted. The substitution fallback emits `150` where the model said `151` — valid JSON, wrong number, and nothing in the output tells the caller. That is the failure mode to design against: a constrained decoder must raise on an empty legal set, not improvise. The third case is worse than either: after `20` there is no legal digit *and* no substitution available, so a decoder that assumed "there is always a legal continuation" simply stops. Check the empty-mask condition explicitly and log the grammar state and prefix — it usually means the schema and the data disagree.

**2. A mask that fails to compile can fail open.** llama.cpp issue [#19051](https://github.com/ggml-org/llama.cpp/issues/19051) reported exactly this: with `response_format: json_schema`, when the schema converts but the *grammar* fails to parse — an invalid regex in a `pattern`, for instance — the server logs `llama_grammar_init_impl: failed to parse grammar` and then "continues generation unconstrained, returning 200 OK. This silently drops structured output enforcement unless logs are inspected." The reporter's expected behaviour is the correct one: "Any grammar parse failure should fail closed and return an HTTP error (e.g. 400)." The issue was closed as stale in March 2026 without a fix attached, which makes the point rather than weakening it — the guarantee you get is the one you test for. Assert on a canary schema at startup, and assert that a deliberately broken `pattern` produces an error rather than prose.

**3. The mask guarantees syntax, not meaning.** 520 of 1,000 masked draws carried a real name because the model's distribution was coherent there; the other 480 were parseable and semantically empty. Since the schema is not injected into the prompt, nothing ever told the model what `name` was for. Structure is a precondition for automation, not evidence of correctness: validate ranges, cross-field invariants and referential integrity after parsing.

**4. Your vocabulary has to be able to express the grammar.** The first version of my mask dead-ended on `\u` escapes because no single hex character existed as a token — a grammar can be valid and still unmaskable against a given tokenizer. Engines handle this by compiling the grammar's character classes into the target vocabulary, which is why the same schema behaves differently across models.

## How to apply this

| What to pin | Why | How to check |
|-------------|-----|--------------|
| Fail-closed behaviour | A grammar that fails to compile must error, not fall back to prose | Send a deliberately invalid `pattern`; expect 4xx |
| Actual coverage of your schemas | Declared ≠ empirical; 0.13–0.90 depending on complexity | Validate 100 real outputs per schema, not one |
| Under-constrained holes | An over-permissive grammar passes invalid objects | Fuzz values outside the schema and assert rejection |
| Empty legal set | Dead ends stop generation with no error | Log state + prefix when the legal set is empty |
| Mask cost at your tokenizer size | Precompute beats per-step recomputation | Compare TPOT with and without the mask, medians |
| Bounded strings and enums | Unbounded `pattern`s and large enums are what compile slowly | Add `maxLength`; measure `enum` compilation |
| Schema in the prompt | The mask does not tell the model what fields mean | Include field descriptions; check value plausibility |

The documented invocations, from the project's own [GBNF guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md), are the `grammar` body field (or `--grammar-file` in the CLI) for raw grammars:

```bash
llama-cli -m model.gguf --grammar-file grammars/json.gbnf -p 'Some prompt'
```

and `json_schema` (or `response_format` on `/chat/completions`, `-j` in the CLI) for schema-driven masking:

```bash
llama-cli -hf bartowski/Phi-3-medium-128k-instruct-GGUF \
  -hff Phi-3-medium-128k-instruct-Q8_0.gguf \
  -j '{"type": "object", "properties": {"name": {"type": "string", "maxLength": 12},
       "age": {"type": "integer", "minimum": 0, "maximum": 150}},
       "required": ["name", "age"], "additionalProperties": false}' \
  -p 'Return the record as JSON.'
```

One last GBNF detail worth carrying into schema design: the guide warns that `x? x? x? ...` repeated N times "may result in extremely slow sampling" and should be written `x{0,N}`. Optional repetition is where hand-written grammars quietly become quadratic.

## Key takeaways

| Claim | Evidence |
|-------|----------|
| The mask is a per-step legal-token set with everything else at $-\infty$ | Definition of constrained decoding; reproduced in block 1: 1,000/1,000 parseable |
| Prompt-only JSON fails more than people expect | JSONSchemaBench LM-only empirical coverage 0.90 (simple) → 0.13 (GitHub Hard) |
| "Declared coverage" is not "it works" | Guidance declares 0.60 / achieves 0.41 on GitHub Hard; XGrammar declares 0.69 / achieves 0.28 |
| Under-constrained grammars are a real failure class | 38 under-constrained failures for XGrammar vs 1 for Guidance on the JSON Schema Test Suite |
| Constrained decoding is not automatically slower | Guidance 6.37 ms TPOT vs 15.40 ms unconstrained (LlamaCpp backend) |
| Mask cost is state-bound, not vocabulary-bound | 25,450 requests → 64 distinct masks here; llguidance ~50 μs/token, XGrammar up to 100× |
| Syntax is guaranteed; meaning is not | 480/1,000 masked draws parsed and were semantically empty |
| Guarantees must be tested, not assumed | Issue #19051: grammar parse failure → unconstrained 200 OK until fixed |

## References

- Geng, S., Cooper, H., Moskal, M., et al. *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models.* arXiv:2501.10868v3 — [arxiv.org/abs/2501.10868](https://arxiv.org/abs/2501.10868)
- Dong, Y., Ruan, C. F., Cai, Y., et al. *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models.* MLSys 2025 — [arxiv.org/abs/2411.15100](https://arxiv.org/abs/2411.15100)
- Xu, X., Bouyarmane, K. *Trie Automata for Constrained Decoding over Large Finite Sets.* arXiv:2608.12574 — [arxiv.org/abs/2608.12574](https://arxiv.org/abs/2608.12574)
- Guidance AI. *llguidance: Super-fast Structured Outputs* — [github.com/guidance-ai/llguidance](https://github.com/guidance-ai/llguidance)
- llama.cpp contributors. *GBNF Guide* — [grammars/README.md](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- llama.cpp contributors. *json.gbnf*, the 601-byte JSON grammar — [grammars/json.gbnf](https://github.com/ggml-org/llama.cpp/blob/master/grammars/json.gbnf)
- llama.cpp issue #19051, *llama-server fails open when JSON schema grammar parsing fails* — [github.com/ggml-org/llama.cpp/issues/19051](https://github.com/ggml-org/llama.cpp/issues/19051)
- Zero Entropy. *Constrained decoding: forcing LLM output to a grammar* — [zeroentropy.dev](https://zeroentropy.dev/concepts/constrained-decoding/)

## Related posts

- [Inconsistent by Construction: Reproducing and Catching a Repetition Collapse](/posts/inconsistent-decoding-repetition-collapse/)
- [The Denominator Is the Metric: Auditing a RAG Retriever Before You Blame the Model](/posts/rag-recall-at-k-denominator/)
- [Agent Tool Calling: Schemas, Validation and Failure Modes](/posts/agent-tool-calling/)
- [Serving LLMs with vLLM: PagedAttention, KV Cache and Continuous Batching](/posts/vllm-llm-serving/)
