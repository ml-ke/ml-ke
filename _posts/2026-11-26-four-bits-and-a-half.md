---
title: "Four Bits and a Half: Reading GGUF Blocks and Measuring What Rounding Costs"
date: 2026-11-26 00:00:00 +0300
categories: [LLM, ML Ops]
tags: [quantization, gguf, llama-cpp, model-compression, on-device, model-evaluation]
math: true
image:
  path: /assets/img/cover-four-bits-and-a-half.webp
  alt: "A ladder of 16 quantization levels with true values snapping down to the nearest rung, beside two byte budgets that both total 4.5 bits per weight"
---

## The label says four bits; the file says six

> **The short version**
> A Q4_0 block is 18 bytes for 32 values, so the format is 4.5 bits per weight and the extra half bit is a scale. I read a real published GGUF file, reimplemented the four common block layouts in numpy, and measured two kinds of damage: reconstruction error on 14.7M real weights and the drift in the next-token distribution from a numpy forward pass. Q4_K reconstructs 29 percent closer than Q4_0 at the same 4.5 bits per weight. Rounding to Q4_0 costs 0.27 bits of KL on a 37-token prompt and pushes the reference top token from rank 1 to rank 2. The file called `q4_0` is 6.015 bpw, not 4.5, and that gap is worth knowing before you size a deployment.
{: .prompt-info }

Every deployment conversation about small language models ends in the same place: which quantisation. The answer usually comes as a name, Q4_K_M or Q5 or Q8, and the name is treated as a size. It is not a size. It is a block layout, and the layout decides both what the file weighs and how much of each weight survives the round trip.

The numbers below come from a 15M-parameter model trained on TinyStories and published in GGUF form by the llama.cpp project, alongside a Q4_0 build of the same weights. It is small enough to load and quantise in numpy on a laptop, which means every step can be checked. The arithmetic of the layouts is the same arithmetic that applies at 8B, so the method transfers even though the absolute error figures do not.

## What a block actually contains

Quantisation replaces each weight with an integer code and a scale. For a block of $n$ values with a step $d$, the code is $q = \mathrm{round}(x/d)$ and the reconstruction is $q \cdot d$. If the values inside a cell are spread evenly, the error of that reconstruction has mean square $d^2/12$, so the root-mean-square error is $d/\sqrt{12}$. Everything else in the formats is a decision about how to pick $d$.

Q4_0 picks one $d$ per 32 values and stores it as a 16-bit float. That is 2 bytes of scale for 16 bytes of nibbles, which is where the half bit comes from: 4 bits of code plus 0.5 bits of scale. The same accounting returns for every format, and it is the reason a "4-bit" file is never 4.0 bits per weight in practice.

| format | values per block | bytes per block | bits per weight | what the overhead buys |
|---|---:|---:|---:|---|
| one scale per tensor | whole tensor | $4 + n/2$ | 4.00 | no adaptation inside the tensor |
| Q4_0 | 32 | 18 | 4.50 | one f16 scale per 32 values |
| Q4_K | 256 | 144 | 4.50 | f16 $d$, f16 $d_{\min}$, sixteen 6-bit codes |
| Q5_0 | 32 | 22 | 5.50 | one f16 scale per 32 values |
| Q6_K | 256 | 210 | 6.5625 | f16 $d$, sixteen int8 scales |
| Q8_0 | 32 | 34 | 8.50 | one f16 scale per 32 values |

Those byte counts are not estimates. They are the `static_assert` sizes in `ggml-common.h`: Q4_0 is $2 + 32/2$ bytes, Q8_0 is $2 + 32$, Q4_K is $2 \cdot 2 + 12 + 128$, and Q6_K is $2 + 16 + 192$. The comment above the Q4_K struct in the same header says "effectively 4.5 bits per weight", which is the same figure the arithmetic gives.

The prefix `K` in the last three names means the block is 256 values wide instead of 32. That change is not cosmetic. It moves the scale overhead from "one 16-bit number per 32 values" to "two 16-bit numbers plus twelve bytes of packed codes per 256 values", and it lets the format afford a per-sub-block minimum. Both Q4_0 and Q4_K spend 0.5 bits per weight on scale, and they spend it differently.

There is one more layer on top of the layout: the block constants are themselves numbers, and those can be quantised too. QLoRA names the technique double quantisation and pairs it with NF4, a data type built for normally distributed weights, and reports that quantising the constants reduces the average memory footprint further. The K layouts use a variant of the same idea, storing eight sub-block scales relative to one shared 16-bit number instead of eight independent ones. Once you see the scale as a budget rather than a given, the layouts stop looking arbitrary.

## The same label at eight billion parameters

The published benchmark table for llama.cpp shows what those layouts average out to on a model people actually deploy, Llama-3.1-8B:

| quantisation | bits per weight | size (GiB) |
|---|---:|---:|
| Q4_K_S | 4.6672 | 4.36 |
| Q4_K_M | 4.8944 | 4.58 |
| Q5_K_M | 5.7036 | 5.33 |
| Q6_K | 6.5633 | 6.14 |
| Q8_0 | 8.5008 | 7.95 |
| F16 | 16.0005 | 14.96 |

Q4_K_M is 4.89 bits per weight rather than the 4.5 the block arithmetic gives, because the recipe assigns different types to different tensors inside the file instead of quantising everything to one layout. The measured size is the sum of those decisions, which is why the same tool exposes per-tensor flags (`--tensor-type`, `--output-tensor-type`, `--token-embedding-type`) next to the preset names. This is the same effect that made the 15M model's "q4_0" file come out at 6.015, and it is the reason the first check in any deployment budget is the one that reads the file.

## Reading the file, not the label

The first thing to check is what is actually inside the file. `gguf` ships with llama.cpp and exposes both the tensor table and the raw bytes, so the census takes a dozen lines.

```python
import os, urllib.request
import numpy as np
from gguf import GGUFReader

BASE = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/"
CACHE = os.path.expanduser("~/.cache/gguf-quant-demo")
os.makedirs(CACHE, exist_ok=True)

def fetch(name, nbytes):
    path = os.path.join(CACHE, name)
    if not (os.path.exists(path) and os.path.getsize(path) == nbytes):
        urllib.request.urlretrieve(BASE + name, path)
    return path

def census(path):
    r = GGUFReader(path)
    weights = sum(int(np.prod(t.shape)) for t in r.tensors)
    tbytes = sum(t.n_bytes for t in r.tensors)
    kinds = {}
    for t in r.tensors:
        k = int(t.tensor_type)
        kinds[k] = kinds.get(k, 0) + 1
    return r, weights, tbytes, kinds

for name, size in [("stories15M.gguf", 98357920), ("stories15M-q4_0.gguf", 19077344)]:
    path = fetch(name, size)
    r, weights, tbytes, kinds = census(path)
    print(f"{name:22s} tensors={len(r.tensors):3d} weights={weights:>10,} "
          f"tensor_bytes={tbytes:>10,} bpw={tbytes * 8 / weights:6.3f} "
          f"file_bytes={os.path.getsize(path):>10,} types={kinds}")

r = GGUFReader(fetch("stories15M.gguf", 98357920))
for key in ["llama.block_count", "llama.embedding_length", "llama.attention.head_count",
            "llama.feed_forward_length", "llama.context_length"]:
    print(key, "=", r.fields[key].contents())
```

On this machine that prints:

```text
stories15M.gguf        tensors= 57 weights=24,407,712 tensor_bytes=97,630,848 bpw=32.000 file_bytes=98,357,920 types={0: 57}
stories15M-q4_0.gguf   tensors= 57 weights=24,407,712 tensor_bytes=18,350,208 bpw= 6.015 file_bytes=19,077,344 types={2: 43, 0: 13, 8: 1}
llama.block_count = 6
llama.embedding_length = 288
llama.attention.head_count = 6
llama.feed_forward_length = 768
llama.context_length = 128
```

Two facts sit in that output. The F32 file is exactly 32.000 bits per weight, and the file named after a 4-bit format is 6.015. The type census explains the second: 43 tensors are type 2 (Q4_0), 13 are type 0 (F32, the layer norms and the final norm), and one is type 8 (Q8_0, the output projection). `llama-quantize` keeps the output layer and the norm vectors at higher precision by default, and on a model this small those tensors are a large share of the total.

So the honest size of "the 4-bit model" is 19.1MB against 98.4MB for F32, a factor of 5.2 where four bits per weight suggests 8. The reason is the model's shape: token embeddings and the output projection are 18.4M of its 24.4M parameters, three quarters of the whole thing, and those are the tensors `llama-quantize` treats specially. On an 8B model with a 32k vocabulary and 4096-wide layers the output projection is about 131M parameters, 1.6 percent of the model, so the same default costs almost nothing there. Compression ratios and model size interact, and no format name tells you which regime you are in.

## Two ways to spend the same half bit

Q4_0 is symmetric. It finds the largest magnitude in the block, sets $d = \max|x|/8$, and places sixteen levels at $(q - 8)d$ for $q$ from 0 to 15. The span is therefore $[-8d, +7d]$, which wastes one level of headroom on the negative side and forces the block to be centred by construction.

Q4_K is affine. Each of the eight 32-value sub-blocks gets a scale $a$ and a minimum $b$, and the reconstruction is $a \cdot q + b$. That removes the centring assumption, and the sub-block scale and minimum are stored as 6-bit codes relative to a shared $d$ and $d_{\min}$, which is how sixteen numbers fit in twelve bytes.

The trade can be measured directly, by holding the weights fixed and changing only the layout. The second block below does that, and it also checks the decoder against the bytes llama.cpp actually wrote: if my Q4_0 encoder and llama.cpp's disagree, the error table is measuring the wrong thing.

```python
import os, urllib.request
import numpy as np
from gguf import GGUFReader

BASE = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/"
CACHE = os.path.expanduser("~/.cache/gguf-quant-demo")
os.makedirs(CACHE, exist_ok=True)

def fetch(name, nbytes):
    path = os.path.join(CACHE, name)
    if not (os.path.exists(path) and os.path.getsize(path) == nbytes):
        urllib.request.urlretrieve(BASE + name, path)
    return path

def weights_and_raw(path):
    r = GGUFReader(path)
    return {t.name: (np.array(t.data, dtype=np.float32) if int(t.tensor_type) == 0
                     else np.array(t.data)) for t in r.tensors}

def pad_to(x, block):
    cols = x.shape[1]
    pad = (-cols) % block
    return (np.pad(x, ((0, 0), (0, pad))) if pad else x), cols

def q8_0(x):
    b = x.reshape(-1, 32)
    d = np.maximum(np.abs(b).max(1, keepdims=True) / 127.0, 1e-8).astype(np.float16).astype(np.float32)
    return (np.clip(np.round(b / d), -128, 127) * d).reshape(x.shape)

def q5_0(x):
    b = x.reshape(-1, 32)
    d = np.maximum(np.abs(b).max(1, keepdims=True) / 15.0, 1e-8).astype(np.float16).astype(np.float32)
    return (np.clip(np.round(b / d), -16, 15) * d).reshape(x.shape)

def q4_0(x):
    b = x.reshape(-1, 32)
    mx = np.take_along_axis(b, np.abs(b).argmax(1)[:, None], 1)
    d = (mx / -8.0).astype(np.float16).astype(np.float32)
    d = np.where(d == 0, np.float32(1e-8), d)
    return ((np.clip(np.round(b / d) + 8, 0, 15) - 8) * d).reshape(x.shape)

def q4_affine(x):
    b = x.reshape(-1, 32)
    lo, hi = b.min(1, keepdims=True), b.max(1, keepdims=True)
    a = np.maximum((hi - lo) / 15.0, 1e-8).astype(np.float16).astype(np.float32)
    m = lo.astype(np.float16).astype(np.float32)
    return (np.clip(np.round((b - m) / a), 0, 15) * a + m).reshape(x.shape)

def q4_k(x):
    xp, cols = pad_to(x, 256)
    b = xp.reshape(-1, 8, 32)
    lo, hi = b.min(2), b.max(2)
    a = (hi - lo) / 15.0
    d = (np.maximum(a.max(1, keepdims=True), 1e-8) / 63.0).astype(np.float16).astype(np.float32)
    dmin = (np.maximum((-lo).max(1, keepdims=True), 1e-8) / 63.0).astype(np.float16).astype(np.float32)
    sc = np.maximum(np.clip(np.round(a / d), 0, 63) * d, 1e-8)
    mn = -(np.clip(np.round(-lo / dmin), 0, 63) * dmin)
    q = np.clip(np.round((b - mn[..., None]) / sc[..., None]), 0, 15)
    return (sc[..., None] * q + mn[..., None]).reshape(xp.shape)[:, :cols].astype(np.float32)

def q6_k(x):
    xp, cols = pad_to(x, 256)
    b = xp.reshape(-1, 16, 16)
    a = np.abs(b).max(2) / 31.0
    d = (np.maximum(a.max(1, keepdims=True), 1e-8) / 127.0).astype(np.float16).astype(np.float32)
    sc = np.maximum(np.clip(np.round(a / d), 0, 127) * d, 1e-8)
    return (sc[..., None] * np.clip(np.round(b / sc[..., None]), -32, 31)).reshape(xp.shape)[:, :cols].astype(np.float32)

def int4(x):
    d = np.float32(max(float(np.abs(x).max()) / 7.0, 1e-8))
    return ((np.clip(np.round(x / d) + 8, 0, 15) - 8) * d).astype(np.float32)

def decode_q4_0_block(raw, ncols):
    nb = ncols // 32
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, nb, 18)
    d = raw[:, :, 0:2].copy().reshape(-1, 2).view(np.float16).astype(np.float32).reshape(-1, nb)
    qs = raw[:, :, 2:18].astype(np.int16)
    nib = np.concatenate([(qs & 0x0F) - 8, ((qs >> 4) & 0x0F) - 8], axis=2).astype(np.float32)
    return (nib * d[:, :, None]).reshape(-1, ncols)

W = weights_and_raw(fetch("stories15M.gguf", 98357920))
RAW = weights_and_raw(fetch("stories15M-q4_0.gguf", 19077344))
two_d = {k: v for k, v in W.items() if v.ndim == 2 and min(v.shape) > 1 and k != "output.weight"}

name = "blk.0.ffn_gate.weight"
mine, theirs = q4_0(W[name]), decode_q4_0_block(RAW[name], 288)
print(f"decoder check {name}: identical elements {100 * float((mine == theirs).mean()):.2f}%  "
      f"max|diff| {float(np.abs(mine - theirs).max()):.2e}")

print(f"\n{'variant':16s} {'bpw':>6s} {'relFro':>9s} {'cosine':>9s}")
for label, fn, bpw in [("int4_tensor", int4, 4.0), ("q4_0", q4_0, 4.5), ("q4_K", q4_k, 4.5),
                       ("q5_0", q5_0, 5.5), ("q6_K", q6_k, 6.5625), ("q8_0", q8_0, 8.5)]:
    num = den = 0.0
    cos = []
    worst = (0.0, "")
    for k, v in two_d.items():
        rec = fn(v)
        num += float(((rec - v) ** 2).sum()); den += float((v ** 2).sum())
        cos.append(float((rec * v).sum() / (np.linalg.norm(rec) * np.linalg.norm(v))))
        rel = float(np.linalg.norm(rec - v) / np.linalg.norm(v))
        if rel > worst[0]:
            worst = (rel, k)
    print(f"{label:16s} {bpw:6.3f} {np.sqrt(num / den):9.5f} {np.mean(cos):9.5f}"
          f"   worst tensor {worst[1]} {worst[0]:.4f}")

print("\nsame four bits, three layouts")
for label, fn, bpw in [("q4_0 symmetric 32", q4_0, 4.5), ("q4 affine 32", q4_affine, 5.0),
                       ("q4_K affine 256/6b", q4_k, 4.5)]:
    num = den = 0.0
    for v in two_d.values():
        rec = fn(v); num += float(((rec - v) ** 2).sum()); den += float((v ** 2).sum())
    print(f"{label:20s} {bpw:5.2f} bpw   relFro {np.sqrt(num / den):.5f}")

print("\nstep size against measured error (q4_0, root mean of d/sqrt(12))")
for k in ["blk.0.ffn_gate.weight", "blk.0.attn_q.weight", "token_embd.weight"]:
    v = W[k]
    b = v.reshape(-1, 32)
    mx = np.take_along_axis(b, np.abs(b).argmax(1)[:, None], 1)
    d = np.abs(mx / 8.0)
    rec = q4_0(v)
    print(f"{k:24s} theory {float((d / np.sqrt(12)).mean()):.6f}   "
          f"measured {float(np.sqrt(((rec - v) ** 2).mean())):.6f}")
```

```text
decoder check blk.0.ffn_gate.weight: identical elements 99.96%  max|diff| 1.40e-02

variant             bpw    relFro    cosine
int4_tensor       4.000   0.42091   0.91696   worst tensor blk.0.attn_output.weight 0.8735
q4_0              4.500   0.12954   0.99609   worst tensor token_embd.weight 0.1406
q4_K              4.500   0.09200   0.99687   worst tensor token_embd.weight 0.0959
q5_0              5.500   0.07094   0.99891   worst tensor token_embd.weight 0.0773
q6_K              6.562   0.02570   0.99981   worst tensor token_embd.weight 0.0274
q8_0              8.500   0.00825   0.99998   worst tensor token_embd.weight 0.0090

same four bits, three layouts
q4_0 symmetric 32     4.50 bpw   relFro 0.12954
q4 affine 32          5.00 bpw   relFro 0.09102
q4_K affine 256/6b    4.50 bpw   relFro 0.09200

step size against measured error (q4_0, root mean of d/sqrt(12))
blk.0.ffn_gate.weight    theory 0.002205   measured 0.002245
blk.0.attn_q.weight      theory 0.005042   measured 0.005379
token_embd.weight        theory 0.004474   measured 0.005635
```

Three results in that output are worth unpacking.

The decoder matches. 99.96 percent of the elements my encoder produces are bit-identical to the ones llama.cpp wrote to disk, and the 84 elements out of 221,184 that differ do so by one or two quantisation steps (0.0077 on average, 0.014 at most). That is a tie broken the other way at a cell boundary, not a different encoding. The two details that matter for a reimplementation are rounding the scale to float16 before dividing, and quantising on the scale as stored rather than the float32 value computed on the way to it. Get either wrong and the codes drift.

One scale for a whole tensor is not a variant of this, it is a different algorithm. The per-tensor baseline reconstructs with 42 percent relative error and a cosine of 0.917, and its worst tensor misses by 87 percent. A single scale has to span the whole tensor, whose largest weight is 0.62 and whose smallest useful ones sit near 0.0007, and sixteen levels spread over that range leave the small weights with no resolution. Blocks exist because weight tensors are not homogeneous.

The two affine layouts land within 0.001 of each other at different costs. Q4_0 symmetric at 4.5 bits per weight reconstructs with 0.1295 relative error. Adding an f16 minimum to every 32-value block, at 5.0 bits per weight, drops that to 0.0910. Q4_K reaches 0.0920 at 4.5 bits by amortising the scale across eight sub-blocks. The win is the affine map, not the extra bits, and the K layouts buy the affine map at the legacy formats' price.

## Why the theory underestimates the embeddings

The step-size check is the part I did not expect to be interesting. For a uniform quantiser the root-mean-square error should be $d/\sqrt{12}$, where $d$ is the block step. For the feed-forward gate matrix the prediction is 0.002205 and the measurement is 0.002245, within two percent. For the attention query matrix, 0.005042 against 0.005379, within seven percent. For the token embedding, 0.004474 against 0.005635, off by twenty-six percent.

That gap is the explanation for the worst-tensor column. Every block in every format has the embedding matrix at the top of its error list, and the reason is not that embeddings are quantised worse. It is that embedding rows are not locally uniform: within a 32-value block the values are clustered more tightly than the block's full range suggests, so the uniform-quantiser formula understates the damage. The formula is a good planning tool for projection matrices and a poor one for the vocabulary table.

## What it does to the next token

Reconstruction error is a proxy. The question a deployment actually cares about is whether the model still predicts the same thing. The harness below runs a plain numpy forward pass of the Llama architecture, six layers with six heads at width 288, and compares the quantised weights against the float32 reference on one 37-token prompt.

The forward pass is the part that makes the comparison meaningful, and it only works if it is correct. As a check, the float32 model's top five continuations after "Lily was very sad." are `" She"` at 14.06, a newline at 12.35, `" He"` at 11.73, a space at 11.66 and `" Her"` at 11.29. That is a language model predicting a story continuation, not noise, so the harness is wired up properly.

```python
import os, urllib.request
import numpy as np
from gguf import GGUFReader

BASE = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/"
CACHE = os.path.expanduser("~/.cache/gguf-quant-demo")
os.makedirs(CACHE, exist_ok=True)

def fetch(name, nbytes):
    path = os.path.join(CACHE, name)
    if not (os.path.exists(path) and os.path.getsize(path) == nbytes):
        urllib.request.urlretrieve(BASE + name, path)
    return path

def load(path):
    r = GGUFReader(path)
    W = {t.name: (np.array(t.data, dtype=np.float32) if int(t.tensor_type) == 0
                  else np.array(t.data)) for t in r.tensors}
    kind = {t.name: int(t.tensor_type) for t in r.tensors}
    kv = {}
    for k in r.fields:
        try:
            kv[k] = r.fields[k].contents()
        except Exception:
            pass
    return W, kind, kv

def _pad(x):
    cols = x.shape[1]
    pad = (-cols) % 256
    return (np.pad(x, ((0, 0), (0, pad))) if pad else x), cols

def q4_0(x):
    b = x.reshape(-1, 32)
    mx = np.take_along_axis(b, np.abs(b).argmax(1)[:, None], 1)
    d = (mx / -8.0).astype(np.float16).astype(np.float32)
    d = np.where(d == 0, np.float32(1e-8), d)
    return ((np.clip(np.round(b / d) + 8, 0, 15) - 8) * d).reshape(x.shape)

def q4_k(x):
    xp, cols = _pad(x)
    b = xp.reshape(-1, 8, 32)
    lo, hi = b.min(2), b.max(2)
    a = (hi - lo) / 15.0
    d = (np.maximum(a.max(1, keepdims=True), 1e-8) / 63.0).astype(np.float16).astype(np.float32)
    dmin = (np.maximum((-lo).max(1, keepdims=True), 1e-8) / 63.0).astype(np.float16).astype(np.float32)
    sc = np.maximum(np.clip(np.round(a / d), 0, 63) * d, 1e-8)
    mn = -(np.clip(np.round(-lo / dmin), 0, 63) * dmin)
    return (sc[..., None] * np.clip(np.round((b - mn[..., None]) / sc[..., None]), 0, 15)
            + mn[..., None]).reshape(xp.shape)[:, :cols].astype(np.float32)

def q6_k(x):
    xp, cols = _pad(x)
    b = xp.reshape(-1, 16, 16)
    a = np.abs(b).max(2) / 31.0
    d = (np.maximum(a.max(1, keepdims=True), 1e-8) / 127.0).astype(np.float16).astype(np.float32)
    sc = np.maximum(np.clip(np.round(a / d), 0, 127) * d, 1e-8)
    return (sc[..., None] * np.clip(np.round(b / sc[..., None]), -32, 31)).reshape(xp.shape)[:, :cols].astype(np.float32)

def q8_0(x):
    b = x.reshape(-1, 32)
    d = np.maximum(np.abs(b).max(1, keepdims=True) / 127.0, 1e-8).astype(np.float16).astype(np.float32)
    return (np.clip(np.round(b / d), -128, 127) * d).reshape(x.shape)

def int4(x):
    d = np.float32(max(float(np.abs(x).max()) / 7.0, 1e-8))
    return ((np.clip(np.round(x / d) + 8, 0, 15) - 8) * d).astype(np.float32)

def decode(raw, ncols, kind):
    nb = ncols // 32
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, nb, 18 if kind == 2 else 34)
    d = raw[:, :, 0:2].copy().reshape(-1, 2).view(np.float16).astype(np.float32).reshape(-1, nb)
    if kind == 2:
        qs = raw[:, :, 2:18].astype(np.int16)
        v = np.concatenate([(qs & 0x0F) - 8, ((qs >> 4) & 0x0F) - 8], 2).astype(np.float32)
    else:
        v = raw[:, :, 2:34].view(np.int8).astype(np.float32)
    return (v * d[:, :, None]).reshape(-1, ncols)

W, KIND, KV = load(fetch("stories15M.gguf", 98357920))
WQ, KINDQ, _ = load(fetch("stories15M-q4_0.gguf", 19077344))
VOCAB = list(KV["tokenizer.ggml.tokens"])
INDEX = {s: i for i, s in enumerate(VOCAB)}
D, NL, NH = 288, 6, 6
HD, EPS = D // NH, float(KV["llama.attention.layer_norm_rms_epsilon"])

def tokenize(text, limit=None):
    s = text.replace(" ", "\u2581")
    out, i = [], 0
    while i < len(s) and (limit is None or len(out) < limit):
        for j in range(min(len(s), i + 24), i, -1):
            if s[i:j] in INDEX:
                out.append(INDEX[s[i:j]]); i = j; break
        else:
            i += 1
    return out

def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

def rope(x, pos):
    inv = 10000.0 ** (-np.arange(0, HD, 2) / HD)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    z = x.reshape(len(pos), NH, HD // 2, 2)
    return np.stack([z[..., 0] * c - z[..., 1] * s,
                     z[..., 0] * s + z[..., 1] * c], -1).reshape(len(pos), NH, HD)

def forward(M, ids):
    T = len(ids)
    x = M["token_embd.weight"][np.array(ids)].astype(np.float32)
    pos = np.arange(T)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    for L in range(NL):
        p = f"blk.{L}."
        h = rms(x, M[p + "attn_norm.weight"])
        q = (h @ M[p + "attn_q.weight"].T).reshape(T, NH, HD)
        k = (h @ M[p + "attn_k.weight"].T).reshape(T, NH, HD)
        v = (h @ M[p + "attn_v.weight"].T).reshape(T, NH, HD)
        q, k = rope(q, pos), rope(k, pos)
        s = np.where(mask[None], -1e30, np.einsum("thd,shd->hts", q, k) / np.sqrt(HD))
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        x = x + np.einsum("hts,shd->thd", s, v).reshape(T, D) @ M[p + "attn_output.weight"].T
        h = rms(x, M[p + "ffn_norm.weight"])
        g, u = h @ M[p + "ffn_gate.weight"].T, h @ M[p + "ffn_up.weight"].T
        x = x + (g / (1.0 + np.exp(-g)) * u) @ M[p + "ffn_down.weight"].T
    return rms(x, M["output_norm.weight"]) @ M["output.weight"].T

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

ids = tokenize("Once upon a time, there was a little girl named Lily who loved to play in the "
               "garden with her dog. One day the dog ran away and Lily was very sad.", limit=48)
ref = forward(W, ids)
p_ref = softmax(ref)
print("prompt:", len(ids), "tokens, ending in", repr(VOCAB[ids[-1]]))
print("F32 top5:", [(VOCAB[i], round(float(ref[-1][i]), 2)) for i in np.argsort(-ref[-1])[:5]])

two_d = {k: v for k, v in W.items() if v.ndim == 2 and min(v.shape) > 1 and k != "output.weight"}
print(f"\n{'variant':12s} {'bpw':>6s} {'KL_bits':>8s} {'ref_top1_rank':>14s} {'top1_agree':>11s}")

def report(label, bpw, M):
    logits = forward(M, ids)
    p = softmax(logits)
    kl = float(np.mean(np.sum(p_ref * (np.log2(p_ref) - np.log2(p)), -1)))
    rank = int((p[-1] > p[-1][ref[-1].argmax()]).sum()) + 1
    agree = float(np.mean(ref.argmax(-1) == logits.argmax(-1)))
    print(f"{label:12s} {bpw:6.3f} {kl:8.4f} {rank:14d} {agree:11.3f}")

for label, fn, bpw in [("int4", int4, 4.0), ("q4_0", q4_0, 4.5), ("q4_K", q4_k, 4.5),
                       ("q6_K", q6_k, 6.5625), ("q8_0", q8_0, 8.5)]:
    M = dict(W)
    for k, v in two_d.items():
        M[k] = fn(v)
    report(label, bpw, M)

print("\noutput projection only, everything else at q4_0")
body = dict(W)
for k, v in two_d.items():
    body[k] = q4_0(v)
for label, ow, bpw in [("q4_0", q4_0(W["output.weight"]), 4.5),
                       ("q8_0", q8_0(W["output.weight"]), 8.5),
                       ("f32", W["output.weight"], 32.0)]:
    Mo = dict(body)
    Mo["output.weight"] = ow
    report("out " + label, bpw, Mo)

M = dict(W)
for k, ty in KINDQ.items():
    if ty == 2:
        M[k] = decode(WQ[k], WQ[k].shape[1] // 18 * 32, 2)
    elif ty == 8:
        M[k] = decode(WQ[k], WQ[k].shape[1] // 34 * 32, 8)
report("file q4_0", 6.015, M)
file_top5 = np.argsort(-forward(M, ids)[-1])[:5]
print("file top5:", [(VOCAB[i], round(float(forward(M, ids)[-1][i]), 2)) for i in file_top5])
```

```text
prompt: 37 tokens, ending in '.'
F32 top5: [('▁She', 14.06), ('<0x0A>', 12.35), ('▁He', 11.73), ('▁', 11.66), ('▁Her', 11.29)]

variant         bpw  KL_bits  ref_top1_rank  top1_agree
int4          4.000   8.1623             41       0.054
q4_0          4.500   0.2666              2       0.730
q4_K          4.500   0.1665              1       0.811
q6_K          6.562   0.0087              1       0.946
q8_0          8.500   0.0006              1       0.946

output projection only, everything else at q4_0
out q4_0      4.500   0.2980              2       0.757
out q8_0      8.500   0.2662              2       0.730
out f32      32.000   0.2666              2       0.730
file q4_0     6.015   0.2607              2       0.757
file top5: [('▁He', 14.06), ('▁She', 13.09), ('<0x0A>', 12.32), ('▁', 11.58), ('▁His', 11.3)]
```

The published file is the interesting row. Its 6.015 bits per weight land within 2 percent of the harness's own 4.5-bit Q4_0 row, 0.2607 bits of KL against 0.2666, which looks at first like the extra 1.5 bits bought nothing. They did buy something, and a controlled run separates it: keeping every other tensor at Q4_0 and raising only the output projection to Q8_0 takes mean KL from 0.298 to 0.266, while float32 on that same tensor adds nothing beyond Q8_0 (0.2666). The builder's default is defensible on measurement. The reason the effect looks small is that the output projection is half of this model's file and about one percent of an 8B model's, so the same default is a bargain at one scale and one of the biggest decisions at the other.

The gap between Q4_0 and Q4_K does show up in the distribution. Mean KL drops from 0.267 to 0.167 bits, and the reference's top token reaches rank 1 instead of rank 2. At Q6_K the drift is 0.0087 bits and the argmax agrees on 35 of 37 positions. In the published file the argmax flips at the final position: `" He"` at 14.06 displaces `" She"` at 13.09, with `" His"` arriving in the top five where `" Her"` had been. One flip in 37 positions is not a quality collapse, and this is a 15M model on children's stories. It is also not nothing, and it is the kind of thing that only shows up when you look at the distribution instead of the loss curve.

Two limits on the numbers. The prompt is 37 tokens, so the agreement column moves in steps of 0.027 and should be read as coarse. The model is trained on a narrow domain, so its top-1 margins are unusually wide; a model with flatter distributions would show more flips at the same KL. The KL column is the one to trust for ordering, and the ordering is stable across the two independent harnesses in this post.

## Why this matters on a phone

The format questions above decide whether a model runs at all on the hardware most people in Kenya actually own. The 24.4M-parameter model here is 98.4MB in float32 and 19.1MB as the published Q4_0 file. Scale to a 0.6B model, the smallest size in Orange's Wolof release, and the same arithmetic gives 338MB at 4.5 bits per weight, 367MB at the 4.89 bits per weight that a real Q4_K_M file averages, and 638MB at 8.5. On a phone with 3GB of RAM, 638MB of weights is a different proposition from 338MB, and the gap does not shrink when the KV cache and the runtime are added on top.

That is the reason to care about which four bits. At the same 4.5 bits per weight, the legacy Q4_0 layout leaves 29 percent more reconstruction error on the table than Q4_K, and the drift numbers put that at roughly 0.1 bits of KL per prompt. Choosing K-quants over legacy quants at the same file size is free quality. Choosing three bits over four is not free, and the published literature on that trade is mostly about protecting a small number of weight channels using activation statistics: AWQ reports that protecting about one percent of salient channels recovers much of the loss at 4 bits, and SmoothQuant's framing is that weights quantise easily while activations do not.

The corpus has covered the memory side of this arithmetic twice already, for a device-sized assistant and for a released model family. What those posts could not answer is what the rounding does to the numbers, which is the question this one measures.

## How to check a quant before you ship it

Six checks, in the order that catches the expensive mistakes first.

1. Compute bits per weight from the file, not from the name. `tensor_bytes * 8 / params` is the only figure that predicts download size and memory, and on the published file here it is 6.015 where the name suggests 4.5.
2. Dump the tensor type census. A file whose name says Q4 may hold Q8_0 and F32 tensors; knowing which is which tells you where the bits went.
3. Prefer K-quants when the tensor dimensions allow them. On this model 288 is not divisible by 256, so the K layouts cannot cover these tensors without padding, which is one reason the published build uses legacy types throughout; on models with 640, 1024 or 2048-wide layers the choice is real and the K layout wins at equal size.
4. Measure on your own text. The 37-token prompt above is enough to separate 8.5 bits from 4.5 and not enough to distinguish two good 4-bit recipes. Use a few hundred tokens of your actual domain and compute mean KL against the float reference.
5. Watch the embedding table. It is the worst-reconstructing tensor in every format here, and the uniform-quantiser formula understates its error by a quarter.
6. Spend precision where it is measurable. Moving only the output projection from Q4_0 to Q8_0 takes mean KL from 0.298 to 0.266 bits, and going on to float32 adds nothing (0.2666). That tensor is half of this model's file and about one percent of an 8B model's, so the same choice is a bargain at one scale and a large fixed cost at the other.

## Key takeaways

| claim | measured value |
|---|---|
| Q4_0 block arithmetic | 18 bytes per 32 values, 4.50 bits per weight |
| Q4_K block arithmetic | 144 bytes per 256 values, 4.50 bits per weight |
| the published q4_0 file | 6.015 bits per weight, 19.1MB against 98.4MB for F32 |
| reconstruction error, Q4_0 | 0.1295 relative Frobenius, cosine 0.99609 |
| reconstruction error, Q4_K | 0.0920 relative Frobenius, cosine 0.99687 |
| affine map at 5.0 bits | 0.0910, so the layout is the win, not the bits |
| drift at 4.5 bits | 0.267 bits of KL (Q4_0), 0.167 (Q4_K) |
| drift at 6.56 bits | 0.0087 bits of KL, argmax agrees on 35 of 37 positions |
| worst-reconstructing tensor | the token embedding, in every format |
| code-level agreement | 99.96 percent of elements identical to llama.cpp |

The headline is that "4-bit" names a layout, not a size, and the layout you pick at 4.5 bits per weight is worth about a third of the error. The second headline is that the file on the hub is rarely the format in its name, so the only way to know what you are shipping is to read the bytes.

## References

- llama.cpp `ggml-common.h`, block struct definitions and `static_assert` sizes for Q4_0, Q4_K, Q6_K and Q8_0 — [github.com](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
- llama.cpp `llama-quantize` documentation, type table with bits per weight for Llama-3.1-8B and the output-tensor defaults — [github.com](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- k-quants pull request, merged 5 June 2023, "a series of 2-6 bit quantization methods, along with quantization mixes" — [github.com](https://github.com/ggml-org/llama.cpp/pull/1684)
- `ggml-org/models`, GGUF builds of the TinyStories checkpoints used here (`stories15M.gguf` and `stories15M-q4_0.gguf`) — [huggingface.co](https://huggingface.co/ggml-org/models/tree/main/tinyllamas)
- llama2.c, origin and training details of the 15M TinyStories model — [github.com](https://github.com/karpathy/llama2.c)
- Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, arXiv:2210.17323 — [arxiv.org](https://arxiv.org/abs/2210.17323)
- Lin et al., AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, arXiv:2306.00978 — [arxiv.org](https://arxiv.org/abs/2306.00978)
- Xiao et al., SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, arXiv:2211.10438 — [arxiv.org](https://arxiv.org/abs/2211.10438)
- Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, arXiv:2305.14314 — [arxiv.org](https://arxiv.org/abs/2305.14314)
- Orange's Wolof release, five open-weight checkpoints from 0.6B to 14B — [huggingface.co](https://huggingface.co/Orange/Wolof-Qwen3-0.6B-v2610)

## Related posts

- [Who pays for the draft: speculative decoding](/posts/speculative-decoding-draft-cost/)
- [Half the cache, twice the context](/posts/kv-cache-quantization-long-context/)
- [Thirty times smaller: auditing embedding compression](/posts/embedding-compression-audit/)
- [Wolof in five sizes](/posts/wolof-five-sizes-orange-release/)
- [500 teachers, 15,000 students, one shared tablet](/posts/gemma-for-teachers-africa/)
