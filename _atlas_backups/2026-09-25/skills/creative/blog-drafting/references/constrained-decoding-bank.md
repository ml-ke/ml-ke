# Constrained decoding / structured-output anchor bank

Verified while writing `_posts/2026-09-21-constrained-decoding-token-mask.md`. Every figure below was read
from the source body (not a snippet). Reuse instead of re-deriving; re-verify only if a newer version of a
source exists.

## JSONSchemaBench — arXiv:2501.10868v3 (27 Feb 2025)

Six engines (Guidance, Outlines, Llamacpp, XGrammar, OpenAI, Gemini) on 10K real schemas; 9,137 used for
the output-validation runs; top-1 output only, validated with the `jsonschema` library (Draft-2020-12,
`format` checks on).

- **Definitions.** *Declared coverage* = the framework processes the schema without rejecting it or
  crashing. *Empirical coverage* = processed **and** the generated output validates.
- **Coverage table (Declared / Empirical / Compliance rate).** GlaiveAI: LM only 1.00/0.90/0.90, Guidance
  0.98/0.96/0.98, XGrammar 1.00/0.93/0.93. GitHub Easy: LM only 1.00/0.65/0.65. GitHub Medium: LM only
  1.00/0.38/0.38, Guidance 0.79/0.69/0.87, XGrammar 0.79/0.52/0.66. GitHub Hard: LM only 1.00/0.13/0.13,
  Guidance 0.60/0.41/0.69, Llamacpp 0.61/0.39/0.63, XGrammar 0.69/0.28/0.41. JSONSchemaStore: LM only
  1.00/0.21/0.21, Guidance 0.35/0.30/0.88, XGrammar 0.76/0.33/0.43, OpenAI 0.06/0.06/1.00.
- **Failure taxonomy (official JSON Schema Test Suite).** Compile error / over-constrained /
  under-constrained: Outlines 42/16/8, Llamacpp 37/18/7, XGrammar **3/5/38**, Guidance 25/7/**1**. The paper
  reads XGrammar's profile as "a trade-off favouring permissiveness"; Guidance has the fewest total
  failures. *Under-constrained* = the grammar accepted something the schema did not (a hole in the mask).
- **Efficiency (Table 2, LlamaCpp backend, medians).** GCT(s)/TTFT(s)/TPOT(ms): LM only —/0.10/15.40
  (GlaiveAI), 15.83, 16.23, 16.68, 15.32; Guidance 0.00–0.01/0.24–0.54/**6.37–9.47**; Llamacpp
  0.05–0.06/0.18–0.30/27.22–29.98; Outlines 3.48–8.05/3.65–8.38/30.33–46.57. The constrained decoder beat
  the unconstrained one per token because Guidance "fast-forwards certain generation steps".
- **Table 3 (HF Transformers backend, because XGrammar does not support llama.cpp):** Guidance TPOT
  35.88–44.21 ms vs XGrammar 65.20–66.78 ms — i.e. the "masking is free" claim is engine-internal, not
  end-to-end: in that harness the engine runs as a `LogitsProcessor` inside `hf_model.generate`.
- Outlines converts schemas to regex constraints with high compilation time; its handling of `minItems`,
  `maxItems`, `enum`, `Array` took "40 seconds to 10 minutes" on some schemas.

## Engine internals (verified quotes)

- **llguidance** (github.com/guidance-ai/llguidance): arbitrary context-free grammar, "on the order of
  50μs of CPU time per token (for 128k tokenizer)", negligible startup; v1.0.0 Jun 2025; shipped in
  OpenAI JSON Schema (May 2025), Chromium, vLLM 0.8.2, SGLang 0.4.4, llama.cpp b4613 (Feb 2025).
- **XGrammar** — arXiv:2411.15100, MLSys 2025: splits the vocabulary into context-independent tokens
  (prechecked) and context-dependent tokens (interpreted at runtime) against a persistent stack; reports up
  to 100x speedup and near-zero end-to-end overhead. XGrammar-2 = ACM AI and Agentic Systems 2026,
  DOI 10.1145/3786335.3813124.
- **Trie automata** — arXiv:2608.12574 (12 Aug 2026): Aho-Corasick precomputed per-node masks for finite
  sets; **0.65 µs vs 5.8 µs** per step (7x) vs XGrammar; 2–6.5x faster compilation at K >= 300; sub-100 ms
  compilation to K = 10,000; vLLM **219 req/s vs 7.5 req/s** at batch size 256 (29x). Names the "cardinality
  wall" — general grammar compilation becomes prohibitively slow as the valid-value set grows.
- Parser stack classification — arXiv:2608.03065: mask cost independent of vocabulary size.

## llama.cpp GBNF (from grammars/README.md, master)

- GBNF = GGML BNF; `root` anchors the whole output; terminals are Unicode code points; `#` comments.
- **Token-level matching:** `<[1000]>` (token id) and `<think>` (token text, only if it is a single token);
  negation with `!<[1000]>` / `!<think>`.
- **Performance gotcha:** `x? x? x? ...` (N repetitions) "may result in extremely slow sampling" — write
  `x{0,N}` instead (upstream issue 4218).
- **Conversion targets:** `grammar` body field / `--grammar-file` / `--grammar` for raw grammars;
  `json_schema` body field for completion endpoints, `response_format` on `/chat/completions`, `-j` in the
  CLI for schema-driven masking. Conversion is of a **subset** of JSON Schema.
- **Quote worth reusing:** "The JSON schema is only used to constrain the model output and is not injected
  into the prompt. The model has no visibility into the schema... This does not apply to tool calling,
  where schemas are injected into the prompt."
- Generated rule for `minimum 0, maximum 150` (their example, name/age array):
  `item-age ::= ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-4] [0-9] | [5] "0")) space`.
- `grammars/json.gbnf` is the 601-byte generic JSON grammar.

## Fail-open anchor (design point, not a breach story)

llama.cpp issue **#19051** (opened 2026-01-23, closed 2026-03-09, labels `bug-unconfirmed`, `stale`): with
`response_format: json_schema`, when the schema converts but the *grammar* fails to parse (e.g. an invalid
regex in a `pattern`), the server logs `llama_grammar_init_impl: failed to parse grammar` and then
"continues generation unconstrained, returning 200 OK. This silently drops structured output enforcement
unless logs are inspected." Expected: "Any grammar parse failure should fail closed and return an HTTP error
(e.g. 400)." Reuse as the argument for canary-schema startup assertions — never as a crime/incident hook.

## Demo recipe (stdlib only, deterministic, no GPU needed)

Run the mask mechanics without a model server: train an interpolated token trigram in-process on ~30
realistic replies (mix clean replies, prose preambles, wrong keys, string-typed numbers, out-of-range
values, truncations), then sample the same model twice — once with `legal = VOCAB`, once with
`legal = mask.legal_tokens()`. Design notes that made it work:

- Model tokens as multi-character units (regex `[A-Za-z]+|\d+|\s+|[^\sA-Za-z\d]`) — masks act on token
  IDs, not characters, and character-level n-grams produce unsalvageable noise.
- Prepend a BOS token to every training line and start the sampler from (BOS, BOS), or unigram backoff
  makes the model open with whitespace.
- Interpolate as `0.9 * P(t | prev2, prev1) + 0.1 * P(t)`; count-based smoothing (`count + k*unigram`) lets
  a common whitespace token outrank real evidence.
- **Bound string lengths (`maxLength`) in the schema.** Inside a string every character is legal, so a
  degenerate sampler loops forever and the mask cannot help; a length bound is what forces the close.
- Make the root rule stop: once `done`, the legal set must be exactly the end-marker — otherwise trailing
  whitespace plus another `}` is re-emitted and the output is invalid.
- Numeric completion: transition to the next state when the buffer is in the accepted set **and cannot be
  extended** (`any(p.startswith(buf) and p != buf for p in PREFIX)`) — the naive `value * 10 > max` test
  dead-ends on single-digit `0`.
- Cache masks on the automaton state tuple; on the reference run 25,450 lookups hit **64 distinct states**
  (99.75%) — the demo number that motivates precomputation.
- Report a semantic metric separately from parseability (e.g. "the value appears in the training data"),
  and label the no-mask lane as a toy baseline, not an LLM baseline.
