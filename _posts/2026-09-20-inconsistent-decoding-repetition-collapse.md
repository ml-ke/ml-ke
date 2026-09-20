---
title: "Inconsistent by Construction: Reproducing and Catching a Repetition Collapse"
date: 2026-09-20 00:00:00 +0300
categories: [AI Engineering, Machine Learning]
tags: [llm, decoding, greedy decoding, repetition, sampling, llama.cpp, agents, observability]
math: true
image:
  path: /assets/img/cover-inconsistent-decoding-repetition-collapse.webp
  alt: A record disc whose needle is stuck in one groove beside a token lane reading k azi ya k uf anya repeated 41 times, a probability meter climbing from 0.29 to 0.99 and an entropy meter falling from 3.13 to 0.05 bits
---

## Introduction

At 21:11:36 UTC on 19 September 2026, a coding agent stopped working. Over the next 41 seconds it emitted one token — `" shame"` — 2,537 consecutive times, 15,222 bytes, and nothing else: no tool call, no error, and `thinking: null`. Only the turn's output-token limit ended the generation.

The operator then respawned the turn and switched the model from its fast non-reasoning configuration to a reasoning one. The new turn opened with a 1,400-character deliberation, diagnosed the real problem (the Ahem test font's wider character metrics inside a `Row`), emitted a `replace_file_content` tool call, and the pipeline reached production. The word "shame" appears nowhere in the repository, the tests, the stack traces, or the prompt.

That sequence is usually read as a vendor bug or bad luck. It is neither. Repetition collapse is a *proven* property of the decoding algorithms in production, it is reproducible on a CPU-only laptop in minutes with an open model, and its shape — an arbitrary first token, a deterministic everything-after — is what determines what you can catch.

> **The framing**
> This is the decoding layer: what greedy search is guaranteed to allow, how to reproduce a collapse with a pinned artifact, what the sampler's probabilities look like on the way in, and a streaming guard for a serving path. Not a model comparison and not a prompt trick — the incident is the motivating trace, not the subject.
{: .prompt-info }

## What the log pins down

| Step | Time (UTC) | Event |
|---|---|---|
| 346 | 21:10:47 | Test fails: `RenderFlex overflowed by 34 pixels`, `landing_page.dart:641:12` |
| 348–356 | 21:10:51–21:11:14 | Model edits the test to print widget trees, then hooks `FlutterError.onError`; same overflow |
| 364 | 21:11:32 | `view_file` on `landing_page.dart` lines 630–655 (`_HeroTrustBadge`) |
| 366 | 21:11:36→21:12:17 | **`" shame"` × 2,537 = 15,222 bytes. Zero tool calls. Ended by the token cap** |
| 367–368 | 21:12:17–21:12:22 | Operator: "respawn and fix thought process", model switched to reasoning; 1,400-char trace, correct diagnosis, tool call, fix |

Three facts do the heavy lifting. **One distinct token:** the step contained `{'shame'}` and nothing else, so the model was not failing to decide — its distribution had collapsed onto a single continuation. **Zero structured output:** nothing for the agent loop to act on, so it could not self-correct. **A hard cap ended it:** not the model, not a heuristic, not an error. The recovery is equally informative — respawning changed the *state*, not the task.

## The theory already said this would happen

**Greedy search is provably inconsistent.** [Welleck and colleagues](https://arxiv.org/abs/2002.02492) define the discrepancy between a model and the decoder on top of it: "inconsistency of a decoding algorithm, meaning that the algorithm can yield an infinite-length sequence that has zero probability under the model." They prove that "commonly used incomplete decoding algorithms – greedy search, beam search, top-k sampling, and nucleus sampling – are inconsistent," and their Theorem 3.4 constructs a case where an incomplete decoder "finds an infinite-length sequence with probability 1." *Incomplete* means any decoder that stops early — a token budget, a top-k cutoff, a probability threshold. That is every decoder in production, so `max_tokens` is not a safety net you added to a working system; in this failure mode it is the only thing that terminates the generation.

**Maximum likelihood is the wrong objective for open-ended generation.** [Holtzman and colleagues](https://arxiv.org/abs/1904.09751) showed that "using likelihood as a decoding objective leads to text that is bland and strangely repetitive," and that maximisation-based decoding "gets stuck in repetitive loops" — from the same models that score well on understanding tasks. Human text is not the most likely text.

**Why the last token keeps winning.** [Su and colleagues](https://arxiv.org/abs/2202.06417) locate a structural cause: "an underlying reason for model degeneration is the anisotropic distribution of token representations." Token representations crowd into a narrow cone of the embedding space, so the model's own recently emitted tokens stay unusually close to the current hidden state and keep taking the argmax. The repetition is representation geometry showing through, not a glitch in the sampler.

**Copying is a mechanism the model is trained to have.** [Olsson and colleagues](https://arxiv.org/abs/2209.11895) describe induction heads: "attention heads that implement a simple algorithm to complete token sequences like [A][B] ... [A] -> [B]." That circuit gives in-context learning much of its power, and it is a copying machine. When the context already ends in a repeated span, the copy answer *is* that span.

### The popular explanation I'd drop

The folk account says identical adjacent tokens "create a sharp attention sink where each newly generated token attends almost 100% to the identical tokens preceding it." The first half is right; the conclusion is wrong. [Gu and colleagues](https://arxiv.org/abs/2410.10781) (ICLR 2025) show the attention sink sits on the *first* tokens and "acts more like key biases, storing extra attention scores, which could be non-informative and not contribute to the value computation" — largely an artefact of softmax normalisation, since sigmoid attention without normalisation removes sinks up to 1B parameters. A sink on the first token is not a copy loop over the last six; the load-bearing mechanisms are the copy circuit and anisotropic representations. That distinction predicts what I found: **the loop needs no memorable token.** The incident's "shame" has a rich pop-culture lineage, and the same model collapses identically on a Swahili verb phrase with no meme history at all.

## Reproduce it on your own machine

Everything below ran CPU-only (4 threads, no GPU), and the artifact is pinned so your bytes match mine.

```bash
# 1. engine: any recent llama.cpp build (this is the one verified: b11062)
# 2. model: 0.8B, Apache-2.0, 672,329,792 bytes
#    sha256 4af8ee1df3ec9008f763ebe95e6f21df3acd8d42c541feeb13314ca22e560afc
curl -L -o afrislm-q4.gguf \
  https://huggingface.co/qvac/TranslatePsy-AfriSLM-0.8B-Q4-GGUF/resolve/main/TranslatePsy-AfriSLM-0.8B-Q4_K_M-imat.gguf

# 3. serve it
llama-server -m afrislm-q4.gguf -c 2048 -t 4 --port 8099
```

Then ask for a translation — greedy, with no repetition defence at all (`temperature 0`, `top_k 1`, repetition penalty at its disabled value of 1.0):

```bash
curl -s http://127.0.0.1:8099/completion -H 'Content-Type: application/json' -d '{"prompt":"You are a professional English to Swahili translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Swahili grammar, vocabulary, and cultural sensitivities. Produce only the Swahili translation, without any additional explanations or commentary.\nPlease translate the following English text into Swahili: Wash your hands with soap before you prepare food. Translation:","n_predict":256,"temperature":0,"top_k":1}' | python3 -c "import sys,json;print(json.load(sys.stdin)['content'])"
```

Complete output, verbatim:

```
 Kwa ajili ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi ya kufanya kazi
```

Forty-one copies of the six-token span `[" k","azi"," ya"," k","uf","anya"]`, no end-of-sequence token: **256 tokens, 9 distinct ids (0.035), longest periodic run 246 tokens (6 × 41), no EOS.** The sampler's probability for the token it chose rose from **0.291** to **0.994**; truncated top-20 entropy fell from **3.13 to 0.05 bits**. The identical request rerun produced a byte-identical sha256, so this is a reproduction rather than a lucky demo.

That entropy floor is the story in one figure: at 0.05 bits there is no probability mass left anywhere else in the vocabulary, so there is no room for `{"tool": "replace_file_content", ...}`. The agent in the incident could not self-correct not from stubbornness but from arithmetic.

### The same token, the same trap

Prompting with a context that already repeats a span makes the copying explicit — the chat template built by hand (system, user, then an assistant turn with the reasoning block pre-closed):

| Run (greedy, cap 256) | Tokens | Distinct | Longest loop | EOS | p(top-1) | Entropy |
|---|---|---|---|---|---|---|
| Translation prompt (above) | 256 | 0.035 | 246 = 6 × 41 | no | 0.291 → 0.994 | 3.13 → 0.05 bits |
| `" shame"` prompt | 256 | 0.023 | 247 = 2 × 123 | no | 0.192 → 0.968 | 3.18 → 0.22 bits |
| Repeat-penalty 1.10 | 93 | 0.796 | none | yes | 0.291 → 0.119 | 3.13 → 3.79 bits |
| Repeat-penalty 1.30 | 37 | 1.000 | none | yes | 0.291 → 0.166 | 3.13 → 3.16 bits |
| DRY 0.8 / 1.75 / len 2 | 54 | 0.537 | 5 tokens | yes | 0.291 → 0.244 | 3.13 → 3.18 bits |

Three runs of the `shame` prompt at cap 48 were byte-identical (sha256 prefix `df95d9fcc088bde4`): the incident's exact token reproduces in seconds from a cold start.

## Watch the distribution collapse, step by step

The interesting part is the *entry*. Asking the server for top-20 log-probabilities at each step shows the model's beliefs rather than its adjectives:

{% raw %}
```python
import json
import math
import urllib.request

PROMPT = ("You are a professional English to Swahili translator. Your goal is to "
          "accurately convey the meaning and nuances of the original English text "
          "while adhering to Swahili grammar, vocabulary, and cultural sensitivities. "
          "Produce only the Swahili translation, without any additional explanations "
          "or commentary.\n"
          "Please translate the following English text into Swahili: "
          "Wash your hands with soap before you prepare food. Translation:")


def post(prompt, n_predict=24):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "n_probs": 20, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:8099/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def entropy_bits(top):
    ps = [math.exp(t["logprob"]) for t in top]
    s = sum(ps)
    ps = [p / s for p in ps] if s > 0 else ps
    return -sum(p * math.log2(p) for p in ps if p > 0)


d = post(PROMPT)
probs = d["completion_probabilities"]
print(f"generated {len(probs)} tokens, last token is EOS: {probs[-1]['token'] == ''!s}")
print(f"{'step':>4} {'token':>10} {'p(top-1)':>9} {'H(top-20) bits':>15}")
for i, p in enumerate(probs[:6]):
    print(f"{i:>4} {p['token']!r:>10} "
          f"{math.exp(p['top_logprobs'][0]['logprob']):>9.4f} "
          f"{entropy_bits(p['top_logprobs']):>15.2f}")
print("  ...")
for i in (len(probs) // 2, len(probs) - 1):
    p = probs[i]
    print(f"{i:>4} {p['token']!r:>10} "
          f"{math.exp(p['top_logprobs'][0]['logprob']):>9.4f} "
          f"{entropy_bits(p['top_logprobs']):>15.2f}")
```
{% endraw %}

Output:

```
generated 24 tokens, last token is EOS: False
step      token  p(top-1)  H(top-20) bits
   0       ' K'    0.2909            3.13
   1       'wa'    0.1415            3.64
   2      ' aj'    0.0767            4.01
   3      'ili'    0.9692            0.27
   4      ' ya'    0.9901            0.08
   5       ' k'    0.1992            3.51
  ...
  12       'uf'    0.5459            2.19
  23       ' k'    0.6573            1.80
```

Step 2 is the moment before the trap closes: the model is genuinely unsure (7.7% on its top choice, 4.01 bits). Then it picks `ili` and the entropy floor drops out. The consequence for detection is direct: **there is no "confident bad decision" to catch at the entry point.** A guard watching for over-confident output is blind for the first three tokens, and by the time confidence arrives the loop is six tokens deep. What is detectable from step two is not confidence but *periodicity*.

## The same collapse in the shape your agent emits

A repeated token is the dramatic version; the unit of collapse can also be a whole record. Same protocol, different model — Llama-3.2-1B-Instruct (Q4_K_M, 807,694,464 bytes, sha256 `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`) — asked in its own chat template for twelve JSON tool calls on the `RenderFlex` investigation:

| Sampler (greedy, cap 384) | Tokens | Objects | Unique | Distinct ids | Duplicate 12-grams | EOS |
|---|---|---|---|---|---|---|
| No defence | 379 | 13 | **1** | 0.066 | 335 of 367 | after the array closed |
| Repeat-penalty 1.30 | 70 | 3 | 3 | 0.729 | 0 of 58 | yes |
| DRY 0.8 (then 2.0 / len 1) | 379 | 13 | **1** (then 13) | 0.066 (then 0.151) | 335/367 (then 99/372) | same as greedy (then cap) |

Thirteen tool calls, all identical: the same `dart:html` inspection of the same file at the same line, twelve past the request. This is the version that hurts an agent loop — not gibberish, but a *plausible* payload the harness will execute forever.

## Why your coding agent is not already protected

The first defence was a **repetition penalty**: divide the logits of tokens already in the context before sampling,

$$p_i = \frac{\exp\!\left(x_i / (T \cdot I(i \in g))\right)}{\sum_j \exp\!\left(x_j / (T \cdot I(j \in g))\right)}$$

where $g$ is the set of recently generated tokens and $I$ equals $\theta$ for those tokens and 1 otherwise. [Keskar and colleagues](https://arxiv.org/abs/1909.05858), who introduced it in CTRL, report that "using a greedy sampling and θ ≈ 1.2 yields a good balance between truthful generation and lack of repetition," and that $\theta = 1$ is the same as no penalty. That last clause is the incident: at the disabled setting, nothing in the sampler resists a loop.

Agent stacks leave it weak or off because penalising token recurrence damages what code and structured output are made of — indentation, braces, repeated JSON keys, template delimiters. The [DRY paper](https://arxiv.org/abs/2608.22761) (August 2026) states the trade-off: standard defences "act on token recurrence rather than the sequential structure of a loop, and often suppress looping only at strengths that also degrade formatting or fluency." My 1B run shows the damage: under repeat-penalty 1.30 the loop stopped but the structure spoiled — the path became `/lib/features/public/LandingPage.dart`, keys appeared that the schema never asked for (`startLine`, `functionName`), and one of three objects failed to parse.

DRY takes the other route, penalising "a candidate token only when generating it would extend the current suffix into an exact continuation of a span seen earlier in the context," with sequence breakers protecting template and formatting tokens. The paper reports a **47% reduction in suffix-extension rate** across models from 1.5B to 120B, nine prompt families and a 600-pair human study, with an intervention-matched placebo producing no comparable reduction; it ships in llama.cpp, ExLlamaV2 and text-generation-webui. My 1B results are more modest, and worth stating as they came out: DRY at multiplier 0.8 changed nothing in that run, and at multiplier 2.0 with allowed length 1 it removed the verbatim duplicates (13 unique objects, duplicated 12-grams down from 335/367 to 99/372) while the model still never emitted an end-of-sequence token inside 384 steps. A 1B model is not the paper's evaluation setting, and a penalty that leaves output unchanged is itself useful information: DRY is not a switch but a strength you set and measure.

## A guard that costs one pass over the token stream

You do not have to choose between no defence and damaged formatting. A collapse is *observable* — an ending that has become periodic — and periodicity is exact, cheap, and needs no reference text.

{% raw %}
```python
MIN_TOKENS = 20          # ignore short periodic endings, e.g. '---' rules
MIN_REPS = 5
MAX_PERIOD = 24          # loop span of up to 24 tokens


def periodic_ending(tokens, min_tokens=MIN_TOKENS, min_reps=MIN_REPS,
                    max_period=MAX_PERIOD):
    """Return (period, reps) if the stream ends in a loop, else None."""
    n = len(tokens)
    for p in range(1, max_period + 1):
        run = p * min_reps
        if run < min_tokens or n < run:
            continue
        window = tokens[-run:]
        if all(window[i] == window[i % p] for i in range(run)):
            return p, run // p
    return None


class LoopWatch:
    """Feed generated tokens in; returns the token count when the alarm fires."""

    def __init__(self, **kw):
        self.cfg = kw
        self.tokens = []
        self.fired_at = None

    def step(self, token):
        self.tokens.append(token)
        if self.fired_at is None and periodic_ending(self.tokens, **self.cfg):
            self.fired_at = len(self.tokens)
        return self.fired_at


# 1. the shape of the incident: one token, repeated until the cap
incident = [" shame"] * 2537
w = LoopWatch()
fired = None
for t in incident:
    fired = fired or w.step(t)
print(f"incident trace : {len(incident):>5} tokens, alarm at token "
      f"{fired:>3} ({fired / len(incident):.1%} of the released budget), "
      f"{len(incident) - fired} tokens recoverable")

# 2. the local collapse: one 6-token span, repeated 41 times
collapse = [" k", "azi", " ya", " k", "uf", "anya"] * 41
w = LoopWatch()
fired = None
for t in collapse:
    fired = fired or w.step(t)
print(f"local collapse : {len(collapse):>5} tokens, alarm at token "
      f"{fired:>3} ({fired / len(collapse):.1%}), {len(collapse) - fired} "
      f"tokens recoverable")

# 3. healthy output must not fire (a real translation, word-level tokens)
healthy = ("Osha mikono yako kwa sabuni kabla ya kutayarisha chakula . "
           "Kliniki itafunguliwa saa mbili asubuhi . Matokeo ya kipimo "
           "yalikuwa hasi , kwa hivyo hahitaji kurudi hadi Juni 2027 .").split()
w = LoopWatch()
print(f"healthy output : {len(healthy):>5} tokens, alarm at token "
      f"{[w.step(t) for t in healthy][-1]}")
```
{% endraw %}

Output:

```
incident trace :  2537 tokens, alarm at token  20 (0.8% of the released budget), 2517 tokens recoverable
local collapse :   246 tokens, alarm at token  30 (12.2%), 216 tokens recoverable
healthy output :    30 tokens, alarm at token None
```

False alarms matter more than detection speed, so I measured the guard over 26,478 word-tokens of English technical writing (seven windows of 4,000):

| `min_reps` / `min_tokens` | Windows fired | What fired |
|---|---|---|
| 3 / 12, 4 / 16, 4 / 24 | 3, 2, 2 of 7 | Periodicity inside *quoted program output*: `+ - + -`, `PS90 PS90 PS90 PS90` |
| **5 / 20** (default) | **0 of 7** | — |

Those were not noise but genuinely periodic spans inside quoted output — the honest cost of exact-match detection: a document containing a repeated table row looks locally like a loop. Word-level tokenisation is also a pessimistic proxy, since a real subword tokeniser produces far less exact repetition in prose. Treat `5 / 20` as a starting point: in the incident it fires at token 20 of 2,537 and recovers 2,517 — about the 41 seconds the machine spent on a turn that could do nothing.

## What to do when it happens to you

| Signal | Action |
|---|---|
| Periodic ending detected | Abort the turn; keep the raw token stream as evidence |
| One distinct token id in a generation | Treat as collapse before the periodicity window fills |
| Tool call missing after a tool intent | Do not retry the same context — respawn with a new prefix |
| Recovery | Respawn with reasoning on, change one sampler parameter, or drop the degenerate tail |
| Not a fix | Raising the cap, waiting for the model to "notice", shipping a strong repeat penalty to protect code |

The recovery row is where the second half of the incident pays off. [Li and colleagues](https://arxiv.org/abs/2402.12875) show a chain of thought gives a transformer *serial* computation it otherwise lacks, so a reasoning configuration does not merely try harder — it produces a long run of intermediate tokens before the answer, which is a different decoding trajectory from a different prefix. That matches both the log and the local reproduction: the same model escapes when the context stops being periodic. You do not need a new model; you need a different prefix, and a guard that tells you when to spend it.

## Key takeaways

| Takeaway | Evidence |
|---|---|
| Greedy search is provably inconsistent; the cap is what ends a collapse | Welleck et al., Theorem 3.4 — infinite-length sequence with probability 1 |
| Likelihood maximisation degenerates on open-ended generation | Holtzman et al. — "bland and strangely repetitive" |
| The last token keeps winning because representations are anisotropic | Su et al. — "anisotropic distribution of token representations" |
| Copying is a trained mechanism (induction heads), not an accident | Olsson et al. — `[A][B] … [A] -> [B]` |
| Attention sinks are first-token softmax artefacts, not the repetition mechanism | Gu et al., ICLR 2025 — "key biases... non-informative" |
| The loop needs no memorable token, and can repeat a whole record | Reproduced on `shame` and `kufanya kazi`; 13 identical tool calls |

## References

1. Sean Welleck, Ilia Kulikov, Jaedeok Kim, Richard Yuanzhe Pang, Kyunghyun Cho. *Consistency of a Recurrent Language Model With Respect to Incomplete Decoding.* EMNLP 2020. [arXiv:2002.02492](https://arxiv.org/abs/2002.02492)
2. Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi. *The Curious Case of Neural Text Degeneration.* ICLR 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
3. Olsson et al. *In-context Learning and Induction Heads.* 2022. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895), [transformer-circuits.pub](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
4. Su et al. *A Contrastive Framework for Neural Text Generation.* NeurIPS 2022. [arXiv:2202.06417](https://arxiv.org/abs/2202.06417)
5. Gu et al. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR 2025 (Spotlight). [arXiv:2410.10781](https://arxiv.org/abs/2410.10781)
6. Keskar et al. *CTRL: A Conditional Transformer Language Model for Controllable Generation.* 2019 — §4 penalized sampling. [arXiv:1909.05858](https://arxiv.org/abs/1909.05858)
7. Weidmann et al. *Don't Repeat Yourself: Stopping Verbatim Loops at Sampling Time.* August 2026. [arXiv:2608.22761](https://arxiv.org/abs/2608.22761)
8. Li et al. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR 2024. [arXiv:2402.12875](https://arxiv.org/abs/2402.12875)
9. llama.cpp server and sampling documentation (DRY, repeat penalty, `n_probs`). [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
10. Model artifacts used: `qvac/TranslatePsy-AfriSLM-0.8B-Q4-GGUF` (sha256 `4af8ee1d…e560afc`) and `bartowski/Llama-3.2-1B-Instruct-GGUF` (sha256 `6f85a640…5611df83`).

## Related posts

- [The Backdoor You Approved: MCP Servers as a Payments Attack Surface](/posts/mcp-payments-attack-surface/) — the tool layer an agent loop drives
- [LLM Data Exfiltration via Indirect Prompt Injection](/posts/llm-data-exfiltration-prompt-injection/) — other ways a generation leaves its lane
- [Split Before You Believe: Why Offline Fraud Models Score Better Than They Perform](/posts/temporal-validation-fraud-models/) — measurement hygiene for the numbers you ship
- [The Denominator Is the Metric: Auditing a RAG Retriever Before You Blame the Model](/posts/rag-recall-at-k-denominator/) — the same instinct, applied to retrieval
