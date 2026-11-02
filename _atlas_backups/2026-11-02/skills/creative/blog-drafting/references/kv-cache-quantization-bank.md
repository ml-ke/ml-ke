# KV cache / inference-memory anchor bank

Verified anchors for posts about KV cache size, dtype choices, long-context retrieval and inference
memory. Every figure below was checked at body level (fetched page or arXiv abstract, not a snippet)
unless marked. Reuse instead of re-deriving; add new verified entries rather than starting a new file.

## The core incident (best anchor in this class)

**vLLM, "The State of FP8 KV-Cache and Attention Quantization in vLLM" (2026-04-22)** —
https://vllm.ai/blog/2026-04-22-fp8-kvcache (also mirrored at vllm-project.github.io; fetch the mirror,
`extract-web-text.py` works on both).

- 128k needle-in-a-haystack on **Hopper**: FP8 accuracy **91% (BF16) -> 13% (FP8)**; after the fix, **89%**.
- Root cause, in their words: Hopper's FP8 tensor cores "are documented as accumulating into FP32
  registers, but in practice the intermediate accumulation loses precision when the contraction dimension
  is large"; **past ~100K** it "causes drastic numerical errors". Same issue hit **DeepSeek-V3 training**
  (their Fig 7(b)).
- Where it lands: `Softmax(AttnScore) * V` — the contraction dimension **is the context length**.
- Fix: **two-level accumulation**, partials written to a real FP32 register
  (`flash-attention#104`, per SageAttention2); costs prefill speed, mitigated by tiling
  (`flash-attention#125`); head_dim > 128 prefill still behind BF16.
- Reasoning cost: 1-2 pts average change on Qwen3-30B-A3B-Thinking-2507 (lowest recovery 97%,
  GPQA:Diamond, BF16 model); 0.7 pts on Qwen3.5-27B (lowest 99%, AIME25).
- Long context: Llama-3.3-70B-Instruct MRCR tracks baseline closely to 128k; valid up to 1M-token prompts.
- Per-token KV cost: **54% of BF16** in the best case. Sliding-window models (gpt-oss-20b): FP8 ITL slope
  **96% of BF16** -> break-even beyond **700k tokens**.
- Flags: `--kv-cache-dtype fp8`; `--kv-cache-dtype-skip-layers sliding_window` (recommended for
  hybrid-attention models; a 128-token sliding window cannot amortise FP8 overhead).

**vLLM TurboQuant study (2026-05-11)** — https://vllm.ai/blog/2026-05-11-turboquant. FP8 remains the
recommended default: 2x capacity, negligible accuracy loss. Llama-3.3-70B on 4xH100: FP8 = 2.6x higher
burst throughput than BF16; Qwen3-30B-A3B matches BF16 throughput at 2x capacity. `k8v4` only 2.4x capacity
and not worth the throughput loss; `4bit-nc` the most practical variant; `k3v4-nc` / `3bit-nc` show
"meaningful accuracy drops" on reasoning and very long context. Use this to show the April and May posts
are consistent (the April fix is what made FP8 the safe default).

## Third-party measurements

**SOTAAZ, llama.cpp KV cache quantization on one A100 (2026-09-15)** —
https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en. Rig: A100 80GB PCIe, Qwen3-8B Q4_K_M
(4.68 GiB), mainline `69320fe`, `-fa 1 -ngl 99`.
- `-ctk q8_0 -ctv q8_0` matches f16 perplexity; cuts the **32K cache by 2.1 GiB**.
- Prefill -3%/-4%, generation -6% (8K/32K vs 128-token generation).
- **Decode at 64K depth falls to 55% of f16** (q4_0: 50%) — the memory win is not a free speed win.
- `q5_1` and a `K q8_0 / V q4_0` mix **silently ran prefill on the CPU** at **43** and **62.7 tok/s**
  (q5_1 sat at 2-4% GPU utilisation) — a ~100x prefill regression with no warning. Bank this as the
  "quiet fallback" caution: quantized V caches need flash attention in llama.cpp, and unsupported
  combinations fall back instead of erroring.

**llama.cpp discussion #20969 (DGX Spark GB10, 128K ctx, Nemotron-3-Nano-30B-A3B)** —
https://github.com/ggml-org/llama.cpp/discussions/20969. The comment was **edited to correct earlier
numbers** (the original 92.5% prompt collapse / memory paradox claims were retracted), so quote only the
corrected table: KV buffer 768 MiB -> 408 MiB (q8_0, **-47%**) -> 216 MiB (q4_0, **-72%**); prompt
throughput unaffected by cache type; generation at ~110K **38.0 -> 25.0 -> 24.0 tok/s** (q4_0 -36.8%).
Lesson: prefer a source's own correction over its first version.

**particula.tech summary (2026-08-03)** — https://particula.tech/blog/kv-cache-quantization-accuracy-loss-benchmarks.
Useful for *model-dependence* and *build-date* framing (attribute it; it is a vendor blog): independent
q8_0 KL-divergence 0.108 (Gemma 4 31B dense) vs 0.377 (26B A4B MoE), <0.04 for both Qwen 3.6 models; a
Hadamard rotation merged into llama.cpp on 2026-04-01 reportedly moved a 0.6B model's q5_1 KV perplexity
from 61.70 to 14.15 and lifted AIME25 at q8_0 from 31.7% to 37.1%. Do **not** attribute the B200 claims
or the "no change in the latency histogram" phrasing to vLLM without re-checking the vLLM post itself.

## Papers

- **KIVI** (ICML 2024), arXiv:2402.02750 — keys should be quantized **per-channel**, values **per-token**;
  tuning-free 2-bit; **2.6x** less peak memory (incl. weights), up to **4x** larger batch,
  **2.35-3.47x** throughput. Cite numbers from the v2 abstract (v1 said 8x batch / 2.05x throughput).
- **KVQuant** (NeurIPS 2024), arXiv:2401.18079 — per-channel key quantization, **pre-RoPE** keys,
  non-uniform per-layer datatypes, per-vector dense-and-sparse; **<0.1 perplexity degradation at 3-bit**
  (Wikitext-2 and C4); LLaMA-7B at **1M context on one A100-80GB**, 10M on 8 GPUs; **~1.7x** speedup.
- **H2O** (NeurIPS 2023), arXiv:2306.14048 — 20% heavy hitters; up to **29x / 29x / 3x** throughput over
  DeepSpeed ZeRO-Inference / HF Accelerate / FlexGen on OPT-6.7B and OPT-30B; latency -1.9x at equal batch.

## Model geometry (read from Hub `config.json`, Sept 2026)

| model | layers | KV heads | head_dim | bf16 KiB/token |
|---|---|---|---|---|
| Llama-3.1-8B (unsloth mirror; meta-llama is gated) | 32 | 8 | 128 | 128 |
| Qwen3-8B | 36 | 8 | 128 | 144 |
| Mistral-7B-v0.1 | 32 | 8 | 128 | 128 |
| gpt-oss-20b (sliding_window 128, hybrid layer_types) | 24 | 8 | 64 | 48 |
| Llama-2-7B (MHA) | 32 | 32 | 128 | 512 |

`bytes/token = 2 * layers * kv_heads * head_dim * bytes_per_value`. Cross-check: Qwen3-8B 32K = 4.500 GiB,
47% of that is **2.115 GiB** — the SOTAAZ measurement says 2.1 GiB. Validating your own table against a
third party's measurement is cheap and makes the post's arithmetic credible.

## Demo recipe (runs with no GPU, ~2s)

Three self-contained blocks (each imports its own deps — the verifier runs them in isolation):
1. **stdlib** sizing calculator over the geometry table above, printing KiB/token, GiB at 128k at 2 and 1
   bytes, and concurrent 8k requests inside a KV budget.
2. **numpy** storage audit: synthetic K with persistent channel outliers (`K[:, ::16] *= 4.0`) and V with
   outlier tokens (`V[::64, :] *= 6.0`); compare per-token vs per-channel reconstruction error (reproduces
   KIVI's axis result, and shows the K advantage flipping at 2 bits), then count top-1 needle flips at a
   fixed logit margin. Measured: **1/200 at 8-bit, 155/200 at 4-bit, 198/200 at 3-bit, 200/200 at 2-bit**.
3. **numpy** accumulator floor: sum `n` *equal* terms (softmax normalises to 1) in a running accumulator
   rounded to fp32/fp16/bf16, with an optional two-level variant. Measured at n=131,072:
   **fp32 1.000000, fp16 0.015625, bf16 0.001953, bf16 two-level 1.000000**. ULP floors: fp32 8,388,608,
   fp16 1,024, bf16 256 tokens. Emulate bf16 with `((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000)` on the
   float32 bit pattern.

## Pitfalls specific to this class

- **Test the mechanism, not a plausible proxy.** Summing a *weighted average* (terms shrinking as 1/n)
  measured **no error growth with context** (bf16 rel-L2 5.98e-03 at 2k vs 4.20e-03 at 256k), i.e. the
  opposite of the published failure. The accumulator effect only appears when the terms stay **equal** so
  the increment eventually drops below one ULP. A null result on a synthetic model means the model is
  wrong, not that the vendor's measurement is.
- **Label synthetic results as synthetic** in the intro and next to the output ("a model of the
  accumulator, not a reproduction of Hopper silicon") and keep the vendor's measured number as the anchor.
- **Beware needle constructions that distort the axis comparison.** Setting a needle key from the query
  (`K[i] = q * (bgmax + margin) / (q @ q)`) makes one row huge in every channel and inverts the
  per-token/per-channel ranking. Keep the needle's magnitude in the same range as background keys.
- **`uv run --with=numpy`** — system python3 (3.14) has no numpy here; a bare `python3 demo.py` fails.
- **Quoted stdout must be re-run, not remembered.** Pair every ```python fence with its quoted ```text block
  and assert byte-for-byte equality on a fresh run; `verify-post-code.py` proves the code executes but says
  nothing about whether the numbers quoted in prose match stdout.
