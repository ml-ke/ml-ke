---
title: "Half the Cache, Twice the Context — measuring what KV quantization costs at long range"
date: 2026-09-27 00:00:00 +0300
categories: [LLM, ML Ops]
tags: [kv-cache, quantization, long-context, inference, memory, vllm, llama-cpp, serving]
math: true
image:
  path: /assets/img/cover-kv-cache-quantization-long-context.webp
  alt: A token tape coarsening into blocks, a halving bar chart and a gauge pinned near zero
---

## Introduction

> **The short version**
> The KV cache decides how many requests your card can hold, and halving it with one byte per value is arithmetic. Whether the half you keep still answers the question is not arithmetic — it depends on a register most teams never configure. vLLM's own stress test measured the failure: at 128k context, needle-in-a-haystack accuracy fell from 91% (BF16) to 13% (FP8), surfacing no error anywhere in the serving stack while that same FP8 path was winning on decode speed.
{: .prompt-info }

Two things get called "quantizing the KV cache", and they fail differently. **Storage quantization** changes the bytes each key and value occupies; **accumulation precision** changes the register that attention's dot products add into. The first one is where all the marketing lives. The second one is where a retrieval task silently degrades.

This post separates them, sizes the cache with arithmetic you can check against a published measurement, and then measures both failure modes on a laptop — no GPU, no model download. What follows is a *model* of the accumulator, not a reproduction of Hopper silicon; the authoritative numbers for real accelerators are vLLM's, and I quote them in full below.

**What this is not:** [the embedding compression post](/posts/embedding-compression-audit/) measured static vectors at rest — do compressed embeddings still retrieve the same neighbours? This is the runtime cache, and the loss here is a numerical one. [The vLLM serving post](/posts/vllm-llm-serving/) covered PagedAttention and block sharing — the architecture. This is the dtype decision and what it costs.

## Size the cache before you quantize it

One formula covers every model, and it is exact enough to plan capacity with:

$$\text{bytes per token} = 2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes per value}$$

The `2` is keys and values. The `KV heads` term is where the modern savings already happened: grouped-query attention [Llama-2-7B](/posts/self-hosting-open-weight-llms/) has 32 KV heads, its Llama-3 successors have 8 with the same head dimension. Here is the geometry of five models people actually self-host, read from each model's `config.json`:

```python
"""Block 1 — KV cache sizing. Stdlib only."""
# Geometry pulled from each model's config.json on the Hub (Sept 2026).
CONFIGS = {
    "Llama-3.1-8B":    (32, 8, 128),
    "Qwen3-8B":        (36, 8, 128),
    "Mistral-7B-v0.1": (32, 8, 128),
    "gpt-oss-20b":     (24, 8, 64),
    "Llama-2-7B":      (32, 32, 128),   # multi-head: no GQA saving
}
CTX = 131_072          # 128k
BUDGET_GIB = 24.0      # KV budget you would give one model


def bytes_per_token(layers, kv_heads, head_dim, item_bytes=2):
    # K and V, one entry per KV head per layer per token
    return 2 * layers * kv_heads * head_dim * item_bytes


print(f"{'model':17s} {'bf16 KiB/tok':>12s} {'128k GiB':>9s} {'fp8 128k GiB':>13s} {'8k reqs @24GiB':>15s}")
for name, (L, H, D) in CONFIGS.items():
    b = bytes_per_token(L, H, D)
    b8 = bytes_per_token(L, H, D, 1)
    ctx_gib = b * CTX / 1024**3
    ctx_gib8 = b8 * CTX / 1024**3
    req_8k = int(BUDGET_GIB * 1024**3 / (b * 8192))
    print(f"{name:17s} {b/1024:12.0f} {ctx_gib:9.1f} {ctx_gib8:13.1f} {req_8k:15d}")

print()
print("Same model size class, different attention geometry (x = bytes/token vs Llama-3.1-8B):")
for name, (L, H, D) in CONFIGS.items():
    bpt = bytes_per_token(L, H, D)
    print(f"  {name:17s} KV heads {H:2d} -> {bpt/1024:5.0f} KiB/token ({bpt/131_072:.1f}x)")
```

```text
model             bf16 KiB/tok  128k GiB  fp8 128k GiB  8k reqs @24GiB
Llama-3.1-8B               128      16.0           8.0              24
Qwen3-8B                   144      18.0           9.0              21
Mistral-7B-v0.1            128      16.0           8.0              24
gpt-oss-20b                 48       6.0           3.0              64
Llama-2-7B                 512      64.0          32.0               6

Same model size class, different attention geometry (x = bytes/token vs Llama-3.1-8B):
  Llama-3.1-8B      KV heads  8 ->   128 KiB/token (1.0x)
  Qwen3-8B          KV heads  8 ->   144 KiB/token (1.1x)
  Mistral-7B-v0.1   KV heads  8 ->   128 KiB/token (1.0x)
  gpt-oss-20b       KV heads  8 ->    48 KiB/token (0.4x)
  Llama-2-7B        KV heads 32 ->   512 KiB/token (4.0x)
```

Two things fall out of that table. First, "7B" and "7B" are not the same cache: Llama-2-7B costs 4x the KV bytes of an 8B Llama-3 model because GQA is the 4x, and it was already collected. Second, the cache is not a rounding error next to the weights — 16 GiB of KV at 128k is more than three times the 4.68 GiB Q4_K_M weight file the A100 test below ran an 8B model from.

**The arithmetic checks against a real measurement.** A September 2026 third-party test ran Qwen3-8B with `-ctk q8_0 -ctv q8_0` on one A100 and reported the 32K cache shrinking by 2.1 GiB against f16. My calculator puts Qwen3-8B at 144 KiB/token, so 32K is 4.500 GiB, and a 47% saving on that is 2.115 GiB. The formula and the benchmark agree to the second decimal.

## Two losses with one name

| | storage quantization | accumulation precision |
|---|---|---|
| What changes | the bytes K and V are stored in | the register the dot products add into |
| Where it bites | every context length, worst on outlier-heavy tensors | contraction dimension of ~100k or more |
| Symptom | slight output drift, higher perplexity | the needle stops being attended to |
| Log line, alert, latency change | none | none |
| Literature fix | scale K per channel, V per token (KIVI); pre-RoPE keys, per-vector dense-and-sparse (KVQuant) | two-level accumulation into a real FP32 register |
| Published result | KVQuant: under 0.1 perplexity degradation at 3-bit; 1M-token LLaMA-7B on one A100-80GB | vLLM: 91% to 13% on 128k NIAH, recovered to 89% |

The storage side is well understood, and the fix is an axis choice. KIVI's contribution was showing why: **key** caches carry persistent outlier channels, so scaling each channel across tokens keeps the informative channels in range, while **value** caches need per-token scaling. Both papers report the work in that direction, and both are worth reading before you pick a dtype.

## What the cache loses when you round it

Here is that axis question, plus the consequence that matters operationally: at what bit width does a needle stop being the top-1 attention target? The test builds K with persistent channel outliers (`K[:, ::16] *= 4.0`) and V with occasional outlier tokens (`V[::64, :] *= 6.0`) — the structure both papers describe — and puts a needle key at a fixed 0.5-logit margin above the haystack.

```python
"""Block 2 — what the cache loses: axis of scaling, and the bits-vs-margin cliff."""
import numpy as np

rng = np.random.default_rng(7)
N, D, TRIALS, MARGIN = 4096, 128, 200, 0.5


def quant(x, bits, axis):
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(x).max(axis=axis, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(x / scale), -qmax, qmax) * scale


# --- Which axis to scale? Reconstruct a cache built the way real caches look:
# K has persistent channel outliers (same channels, every token); V has token outliers.
K = rng.normal(0, 1, (N, D)); K[:, ::16] *= 4.0          # K: 8 outlier channels
V = rng.normal(0, 1, (N, D)); V[::64, :] *= 6.0           # V: 64 outlier tokens
print("reconstruction error of the cache itself (relative L2, lower is better)")
print(f"{'cache':6s} {'bits':>4s} {'per-token':>11s} {'per-channel':>12s}  winner")
for name, X in (("K", K), ("V", V)):
    for bits in (8, 4, 2):
        pt = np.linalg.norm(quant(X, bits, 1) - X) / np.linalg.norm(X)
        pc = np.linalg.norm(quant(X, bits, 0) - X) / np.linalg.norm(X)
        print(f"{name:6s} {bits:4d} {pt:11.3e} {pc:12.3e}  "
              f"{'per-token' if pt < pc else 'per-channel'}")

# --- The consequence: a needle that scores 0.5 logits above everything else.
print()
print(f"needle recovery, {TRIALS} trials, margin {MARGIN} logits, K per-channel / V per-token")
print(f"{'K / V bits':>12s} {'top-1 flips':>12s} {'max w err':>10s} {'out rel-L2':>11s}")
for kb, vb in ((8, 8), (4, 4), (3, 3), (2, 2)):
    flips, werr, oerr = 0, [], []
    for _ in range(TRIALS):
        q = rng.normal(0, 1, D)
        Kt = rng.normal(0, 1, (N, D)); Kt[:, ::16] *= 4.0
        Vt = rng.normal(0, 1, (N, D))
        needle = N // 2
        Kt[needle] = q * ((Kt @ q).max() + MARGIN) / (q @ q)
        s = Kt @ q
        w = np.exp(s - s.max()); w /= w.sum()
        out = w @ Vt
        Kq, Vq = quant(Kt, kb, 0), quant(Vt, vb, 1)
        sq = Kq @ q
        wq = np.exp(sq - sq.max()); wq /= wq.sum()
        flips += int(np.argmax(sq) != needle)
        werr.append(np.abs(wq - w).max())
        oerr.append(np.linalg.norm(wq @ Vq - out) / np.linalg.norm(out))
    print(f"{f'{kb}/{vb}':>12s} {f'{flips}/{TRIALS}':>12s} {np.mean(werr):10.2e} {np.mean(oerr):11.2e}")
```

```text
reconstruction error of the cache itself (relative L2, lower is better)
cache  bits   per-token  per-channel  winner
K         8   1.212e-02    8.506e-03  per-channel
K         4   2.199e-01    1.542e-01  per-channel
K         2   7.749e-01    9.020e-01  per-token
V         8   6.458e-03    2.876e-02  per-token
V         4   1.175e-01    5.104e-01  per-token
V         2   7.642e-01    9.078e-01  per-token

needle recovery, 200 trials, margin 0.5 logits, K per-channel / V per-token
  K / V bits  top-1 flips  max w err  out rel-L2
         8/8        1/200   3.39e-02    7.07e-02
         4/4      155/200   4.90e-01    9.82e-01
         3/3      198/200   7.12e-01    1.46e+00
         2/2      200/200   8.48e-01    1.77e+00
```

Three readings:

1. **The axis choice is real and it is not universal.** Per-channel wins for K at 8 and 4 bits, and per-token for V at every width — which is the KIVI result, reproduced on synthetic tensors with the same outlier structure. At 2 bits the K advantage flips to per-token (7.7e-01 against 9.0e-01), so "per-channel keys" is a rule about 4-8 bit budgets, not a law.
2. **The margin, not the bit count, decides whether you lose the needle.** At 0.5 logits of separation, 8-bit caching lost the needle in 1 of 200 trials and 4-bit lost it in 155. If your retrieval target is comfortable — a distinctive key, a large margin — 4 bits may be fine. If you are asking for one fact among 128k tokens, the margin is small and the same setting is a coin flip.
3. **Output error is the number to watch, not weight error.** At 4 bits the maximum attention-weight error is 0.49 while the output vector's relative error is 0.98 — the drift compounds through the value mix rather than staying local.

## The cliff is in the accumulator, not the storage

Now the other loss. Softmax guarantees the attention weights sum to 1.0. On real hardware that sum is computed by accumulating `n` products into a running register, where `n` is the contraction dimension — for the `softmax(attn) @ V` matmul, that is your context length. A register has a mantissa, and once an individual increment is smaller than one unit in the last place of the running sum, it cannot change the sum at all.

```python
"""Block 3 — why a long contraction dimension eats your attention weights."""
import numpy as np


def round_acc(x, kind):
    """Round a running accumulator to the precision the hardware actually keeps."""
    if kind == "fp32":
        return np.float32(x)
    if kind == "bf16":
        u = np.asarray(x, np.float32).view(np.uint32)
        return np.uint32((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    return np.float16(x)


def flat_sum(n, kind, two_level=False):
    """Sum n equal terms (softmax weights normalise to 1) in a running accumulator."""
    term, acc = 1.0 / n, np.float32(0.0)
    for _ in range(n):
        acc = np.float32(acc + term) if two_level else round_acc(acc + term, kind)
    return float(acc)


print("Summing n equal weights that must total 1.0 (what softmax guarantees):")
print(f"{'n tokens':>9s} {'fp32 flat':>11s} {'fp16 flat':>11s} {'bf16 flat':>11s} {'bf16 2-level':>13s}")
for n in (64, 256, 1024, 4096, 16384, 65536, 131072):
    row = [flat_sum(n, k) for k in ("fp32", "fp16", "bf16")]
    row.append(flat_sum(n, "bf16", two_level=True))
    print(f"{n:9d} " + " ".join(f"{v:11.6f}" if i < 3 else f"{v:13.6f}" for i, v in enumerate(row)))

print()
print("Accumulator resolution: an increment is lost once it falls below one ULP of the running sum")
for kind, mantissa in (("fp32", 23), ("fp16", 10), ("bf16", 8)):
    print(f"  {kind}: 1 ULP at a sum of 1.0 = 2^-{mantissa} = {2.0**-mantissa:.6f} "
          f"-> a flat distribution longer than {int(2**mantissa):,} tokens stops accumulating")
```

```text
Summing n equal weights that must total 1.0 (what softmax guarantees):
 n tokens   fp32 flat   fp16 flat   bf16 flat  bf16 2-level
       64    1.000000    1.000000    1.000000      1.000000
      256    1.000000    1.000000    1.000000      1.000000
     1024    1.000000    1.000000    0.250000      1.000000
     4096    1.000000    0.500000    0.062500      1.000000
    16384    1.000000    0.125000    0.015625      1.000000
    65536    1.000000    0.031250    0.003906      1.000000
   131072    1.000000    0.015625    0.001953      1.000000

Accumulator resolution: an increment is lost once it falls below one ULP of the running sum
  fp32: 1 ULP at a sum of 1.0 = 2^-23 = 0.000000 -> a flat distribution longer than 8,388,608 tokens stops accumulating
  fp16: 1 ULP at a sum of 1.0 = 2^-10 = 0.000977 -> a flat distribution longer than 1,024 tokens stops accumulating
  bf16: 1 ULP at a sum of 1.0 = 2^-8 = 0.003906 -> a flat distribution longer than 256 tokens stops accumulating
```

Read the bf16 column against the number it must produce. A flat distribution over 131,072 tokens — every weight 1/131072 — sums to **0.001953** in a bf16 accumulator instead of 1.0. The weights lost 99.8% of their mass, and the accumulator stops growing at a power of two because that is where the increment crosses one ULP. The same sum in fp32 returns 1.000000, and the two-level variant — keep the partial sums in a real FP32 register and round once at the end — returns 1.000000 even with bf16 as the storage type.

Three caveats, because this is a model rather than a die-shot:

- This rounds after **every** addition, which is the worst case. Real tensor cores accumulate into FP32 registers with limited effective precision, so the truth sits between the flat and two-level columns — but the sensitivity to mantissa width is exactly as tabulated: 8,388,608 vs 1,024 vs 256 tokens.
- Real attention is rarely flat. Peaked distributions (the needle case) are much more forgiving, which is why short-context benchmarks look fine.
- The direction is confirmed on hardware. vLLM's April 2026 investigation found that on Hopper, once the contraction dimension passes ~100K, FP8 attention loses enough accumulator precision that the softmax numerator and denominator drift, the model attends to the wrong span, and 128k needle-in-a-haystack falls from 91% to 13%. They shipped two-level accumulation — partials written to an FP32 register instead of trusting the tensor core accumulator — which restored it to **89%**.

That is the whole shape of the problem. The config change halves memory, the dashboard stays green, and the retrieval task fails.

## What the vendors actually measured

| Source | Setting | Measured |
|---|---|---|
| [vLLM, Apr 2026](https://vllm.ai/blog/2026-04-22-fp8-kvcache) | FP8 KV cache, Hopper, 128k NIAH | 13% vs 91% BF16; 89% after two-level accumulation |
| vLLM, Apr 2026 | FP8 KV cache, reasoning | 1-2 points of average accuracy change (Qwen3-30B-A3B-Thinking-2507), lowest recovery 97%; 0.7 points on Qwen3.5-27B |
| vLLM, Apr 2026 | FP8 KV cache, per-token cost | 54% of the BF16 cost in the best case; slopes equal to BF16 on sliding-window models (break-even beyond 700k tokens) |
| [vLLM TurboQuant study, May 2026](https://vllm.ai/blog/2026-05-11-turboquant) | FP8 vs 3-4 bit variants, 30B-200B | FP8: 2x capacity, negligible loss, 2.6x burst throughput on 4xH100 (Llama-3.3-70B); k3v4 and 3-bit "meaningful accuracy drops" |
| [llama.cpp on one A100, Sep 2026](https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en) | Qwen3-8B, `-ctk q8_0 -ctv q8_0` | Matches f16 perplexity, cuts the 32K cache by 2.1 GiB, prefill -3%/-4%, generation -6%; decode at 64K depth falls to 55% of f16 (q4_0: 50%) |
| [llama.cpp community, DGX Spark](https://github.com/ggml-org/llama.cpp/discussions/20969) | 128K context, 30B-A3B | KV buffer 768 to 408 MiB (q8_0) to 216 MiB (q4_0); generation at ~110K falls 38.0 to 25.0 tok/s |
| [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750) | 2-bit, asymmetric per-channel K | 2.6x less peak memory, up to 4x batch, 2.35x-3.47x throughput |
| [KVQuant (NeurIPS 2024)](https://arxiv.org/abs/2401.18079) | 3-bit, pre-RoPE keys | Under 0.1 perplexity degradation; 1M-token LLaMA-7B on one A100-80GB, 10M on eight; ~1.7x speedup |
| [H2O (NeurIPS 2023)](https://arxiv.org/abs/2306.14048) | 20% heavy hitters, eviction | Up to 29x throughput over DeepSpeed ZeRO-Inference on OPT-6.7B/30B; latency cut up to 1.9x |

Two rows in that table contradict the folk wisdom, and both are worth sitting with. First, q8_0 is not merely "almost free": on one A100 it matched f16 perplexity and still cost 45% of decode throughput at 64K depth, because dequantization is paid per token. Second, the same setting measured in April and in August gave different accuracy — a Hadamard rotation merged into llama.cpp on 2026-04-01 reportedly took a 0.6B model's q5_1 KV perplexity from 61.70 to 14.15. KV quantization results are **properties of a build**, not constants of a format.

## How to apply this

1. **Size the cache before you tune it.** Run the first block with your model's geometry. If a 128k context needs 16 GiB of KV, no dtype decision rescues a plan that assumed 8.
2. **Pick the axis deliberately.** Scale K per channel and V per token at 4-8 bits; re-check the axis if you go to 2 bits, where my measurements flipped the winner for K.
3. **Test at your real maximum context.** A 4K benchmark cannot see the accumulator effect. The published failure appears only past roughly 100k contraction steps.
4. **Skip layers that cannot amortize.** vLLM added `--kv-cache-dtype-skip-layers sliding_window` because a hybrid-attention model like gpt-oss-20b keeps 128-token sliding windows in BF16 with a better result than quantizing them.
5. **Budget the accumulator, not just the bytes.** If a two-level accumulation path exists in your kernel, that is the flag that decides long-context recall. On vLLM's measurements, it was worth 76 points of needle accuracy.
6. **Watch for the quiet fallback.** In the A100 test, `q5_1` and a `K q8_0 / V q4_0` mix ran prefill on the CPU at 43 and 63 tokens per second with no warning, because the GPU kernel does not support that combination. A 100x prefill regression that reports nothing is worse than a config error that fails loudly.
7. **Re-measure after every kernel upgrade.** The 13% to 89% recovery and the llama.cpp rotation change were both kernel work. A dtype that was unsafe at one version can be the default at the next.
8. **Know where it still costs.** Large head dimensions (`head_dim = 256`) still show slower prefill than BF16 even with the fixes in.

## Key takeaways

| Takeaway | Detail |
|---|---|
| The cache is a formula, not a mystery | `2 x layers x KV heads x head dim x bytes per value`; validated against a measured 2.1 GiB saving on Qwen3-8B |
| GQA already collected the biggest win | Llama-2-7B costs 512 KiB/token against 128 KiB/token for Llama-3.1-8B at the same head dimension |
| Two losses share one name | Storage quantization: choose the axis, loses 1/200 needles at 8 bits. Accumulation precision: fix the register, or lose 155/200 at 4 bits |
| Halving bytes never shows up in latency monitoring | The FP8 path that produced vLLM's 91% to 13% regression was also the one winning on decode speed — the accuracy loss and the speed gain arrive in the same measurement |
| The floor is `2^-mantissa` | 8,388,608 tokens for fp32, 1,024 for fp16, 256 for bf16 — a flat distribution longer than that stops accumulating |
| Two-level accumulation is the fix | Rounding the running sum once, into a real FP32 register, returns the sum to 1.000000 at every context length tested |

## References

- [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm.ai/blog/2026-04-22-fp8-kvcache) — the 91% to 13% NIAH regression, the two-level accumulation fix, reasoning and long-context evaluations
- [A First Comprehensive Study of TurboQuant](https://vllm.ai/blog/2026-05-11-turboquant) — FP8 as the recommended default across 30B-200B models, and where 3-4 bit variants cost accuracy
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) — per-channel keys, per-token values, 2.6x peak memory reduction (ICML 2024)
- [KVQuant: Towards 10 Million Context Length LLM Inference](https://arxiv.org/abs/2401.18079) — pre-RoPE keys, per-vector dense-and-sparse, under 0.1 perplexity degradation at 3-bit (NeurIPS 2024)
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference](https://arxiv.org/abs/2306.14048) — 20% heavy hitters and the eviction case (NeurIPS 2023)
- [llama.cpp KV cache quantization, measured on one A100](https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en) — q8_0 matching f16 perplexity, the 2.1 GiB saving, and the decode cost at 64K depth
- [llama.cpp discussion: KV cache quantization on a DGX Spark](https://github.com/ggml-org/llama.cpp/discussions/20969) — corrected KV buffer measurements across f16, q8_0 and q4_0 at 128K
- [KV cache quantization: what FP8 and q8_0 cost in accuracy](https://particula.tech/blog/kv-cache-quantization-accuracy-loss-benchmarks) — an independent 2026 summary of the model-dependence and build-date variables
- Model geometry read from [Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json), [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json) and [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json) `config.json`

## Related posts

- [vLLM and LLM Serving: PagedAttention, KV Cache and Continuous Batching](/posts/vllm-llm-serving/) — the architecture this dtype decision sits inside
- [Thirty Times Smaller: auditing embedding compression](/posts/embedding-compression-audit/) — compression at rest, and why recall, not ratio, is the claim to test
- [Self-Hosting Open-Weight LLMs](/posts/self-hosting-open-weight-llms/) — where GQA, quantization and memory budgets first meet
- [Model Serving 101](/posts/model-serving-101/) — the serving stack this optimisation belongs to
- [GPU Optimization for ML Workloads](/posts/gpu-optimization/) — the memory-bandwidth view of the same bottleneck
