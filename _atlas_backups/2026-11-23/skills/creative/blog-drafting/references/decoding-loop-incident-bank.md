# Decoding-loop / repetition-collapse bank (verified 2026-09-20)

Reusable anchors for posts about LLM decoding failure modes, agent loops, sampler settings and
generation-time observability. Every figure below was produced locally on CPU (4 threads, no GPU)
with llama.cpp build **b11062** unless marked "paper claim".

## Papers (all read at body level; quotes verified against the PDF/abstract)

| Paper | Anchor | Key line |
|---|---|---|
| Welleck et al., EMNLP 2020 — arXiv:2002.02492 | inconsistency of a decoding algorithm | "the algorithm can yield an infinite-length sequence that has zero probability under the model"; greedy/beam/top-k/nucleus are all inconsistent; Theorem 3.4 gives an infinite-length sequence "with probability 1" |
| Holtzman et al., ICLR 2020 — arXiv:1904.09751 | likelihood maximisation degenerates | likelihood as a *decoding* objective "leads to text that is bland and strangely repetitive"; maximisation decoding "gets stuck in repetitive loops" |
| Olsson et al., 2022 — arXiv:2209.11895 | induction heads | "attention heads that implement a simple algorithm to complete token sequences like [A][B] ... [A] -> [B]" |
| Su et al., NeurIPS 2022 — arXiv:2202.06417 | representation anisotropy | "an underlying reason for model degeneration is the anisotropic distribution of token representations" |
| Gu et al., ICLR 2025 — arXiv:2410.10781 | attention sink (softmax artefact) | sink "acts more like key biases, storing extra attention scores, which could be non-informative"; sigmoid attention → sinks do not emerge up to 1B params |
| Keskar et al., 2019 — arXiv:1909.05858 §4 | repetition penalty origin | "using a greedy sampling and θ ≈ 1.2 yields a good balance... θ = 1 is equivalent to" no penalty |
| Weidmann et al., Aug 2026 — arXiv:2608.22761 | DRY sampler | "penalizes a candidate token only when generating it would extend the current suffix into an exact continuation of a span seen earlier"; 47% suffix-extension reduction, placebo control; standard defences "often suppress looping only at strengths that also degrade formatting or fluency" |
| Li et al., ICLR 2024 — arXiv:2402.12875 | CoT = serial computation | constant-depth transformers without CoT ⊆ AC⁰; T CoT steps solve size-T circuits |

**Terminology trap:** the popular name "sink state" does **not** appear in the Welleck PDF
(`pdftotext | grep -c sink` → 0). Attribute the formal concept as *inconsistency of an incomplete
decoding algorithm*; reserve "attention sink" for the softmax first-token artefact.

## Pinned artifacts (download-and-match)

| Model | Repo / file | Bytes | sha256 |
|---|---|---|---|
| TranslatePsy-AfriSLM-0.8B Q4_K_M (imat) | `qvac/TranslatePsy-AfriSLM-0.8B-Q4-GGUF` → `TranslatePsy-AfriSLM-0.8B-Q4_K_M-imat.gguf` | 672,329,792 | `4af8ee1df3ec9008f763ebe95e6f21df3acd8d42c541feeb13314ca22e560afc` |
| Llama-3.2-1B-Instruct Q4_K_M | `bartowski/Llama-3.2-1B-Instruct-GGUF` → `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807,694,464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |

## Measured collapses (greedy: temperature 0, top_k 1, repeat penalty 1.0)

| Configuration | Tokens | Distinct ids | Longest periodic run | EOS | p(top-1) | H(top-20) |
|---|---|---|---|---|---|---|
| afrislm + raw translation prompt (cap 256) | 256 | 0.035 | 246 = 6 × 41 | no | 0.291→0.994 | 3.13→0.05 bits |
| afrislm + templated `shame` prompt (cap 256) | 256 | 0.023 | 247 = 2 × 123 | no | 0.192→0.968 | 3.18→0.22 bits |
| afrislm + repeat-penalty 1.10 | 93 | 0.796 | none | yes | 0.291→0.119 | 3.13→3.79 |
| afrislm + repeat-penalty 1.30 | 37 | 1.000 | none | yes | 0.291→0.166 | 3.13→3.16 |
| afrislm + DRY 0.8/1.75/len2/last256 | 54 | 0.537 | 5 | yes | 0.291→0.244 | 3.13→3.18 |
| llama-3.2-1B + 12-JSON-tool-call task (cap 384) | 379 | 0.066 | 13 objects, 1 unique, 335/367 dup 12-grams | after array closed | — | — |
| llama-3.2-1B + repeat-penalty 1.30 | 70 | 0.729 | 3 objects, 3 unique, 1 unparseable | yes | — | — |
| llama-3.2-1B + DRY 2.0/1.75/len1 | 384 (cap) | 0.151 | 13 unique, 99/372 dup 12-grams | no | — | — |

Determinism: byte-identical sha256 on rerun — `fe022903217efd0b` (translation collapse, twice),
`df95d9fcc088bde4` (`shame`, three runs at cap 48).

## Operational recipes that work

```bash
# serve (CPU, 4 threads)
llama-server -m model.gguf -c 2048 -t 4 --port 8099

# raw greedy completion with per-token top-k logprobs
curl -s http://127.0.0.1:8099/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"...","n_predict":256,"temperature":0,"top_k":1,"n_probs":20}'
```

* `completion_probabilities[].top_logprobs` is the cheapest way to make quantitative
  "the model believed X" claims; probabilities are reported **after** penalties when penalties are on.
* Raw `/completion` prompts are **not** chat-templated, and the same request routed through
  `/v1/chat/completions` can behave differently (in this session the templated version echoed the
  user's repeated span and stopped at EOS, while the raw prompt collapsed). State which path you used.
* Loop guard that worked (exact periodicity, no reference text): alarm when the stream ends in a
  period-p run of ≥5 reps over ≥20 tokens; `min_reps=5 / min_tokens=20` gave 0 false alarms over
  26,478 word-tokens of prose (3/12, 4/16, 4/24 gave 3, 2, 2 — all on genuinely periodic quoted
  program output). Word-level tokenisation is a pessimistic proxy for subword tokenisers.
* Fires at token 20 of 2,537 on the incident shape; 12.2% into a 246-token local collapse.
