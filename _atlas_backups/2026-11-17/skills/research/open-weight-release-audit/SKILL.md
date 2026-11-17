---
name: open-weight-release-audit
description: Use when auditing a released open-weight model artifact.
version: 1.0.1
author: hermes-agent
license: MIT
metadata:
  hermes:
    category: research
    tags: [open weights, hugging face, model release, licence, hardware fit]
    related_skills: [blog-drafting, grounded-citations, topic-scouting]
---

# Auditing an open-weight release

When a lab publishes weights, the event is the download, not the capability. Capability was announced
earlier and was probably already covered. What is new is that the artifact can be fetched, audited and
run, so the deliverable is measured facts (sizes, architecture, licence, fit) with capability claims
attributed and labelled.

## When to Use

- A lab publishes model weights and you need to establish what is actually downloadable and runnable.
- You are writing or reviewing a claim about a model's parameter count, memory footprint, licence terms
  or hardware requirements.
- Someone asks "can we self-host this?" about a newly released model.

## 1. Pull primary sources in this order

```bash
# what the org published lately, with metadata
curl -s "https://huggingface.co/api/models?author=<org>&sort=createdAt&direction=-1&limit=12" | python3 -c "
import sys,json
for m in json.load(sys.stdin):
    print(m['modelId'], m.get('createdAt'), m.get('downloads'), m.get('likes'))"

curl -sL https://huggingface.co/<org>/<model>/raw/main/README.md      # card, incl. licence key
curl -sL https://huggingface.co/<org>/<model>/raw/main/config.json    # architecture
curl -sL https://huggingface.co/<org>/<model>/raw/main/LICENSE | head -20   # the text, not the name
```

Press coverage is for the announcement date, funding context and quotes. It is never the source of an
architecture figure, a parameter count or a size.

## 2. Measure the artifact

Sum the reported file sizes per directory, for the vendor checkpoint, any quantised sibling, and every
community GGUF mirror:

```bash
curl -s "https://huggingface.co/api/models/<org>/<model>?blobs=true" | python3 -c "
import sys,json,collections
d=json.load(sys.stdin); agg=collections.defaultdict(float)
for s in d.get('siblings',[]):
    agg[s['rfilename'].split('/')[0] if '/' in s['rfilename'] else '(root)'] += s.get('size') or 0
for k,v in sorted(agg.items(), key=lambda x:-x[1]): print(f'{k:16s}{v/1e9:8.1f} GB')"
```

Record per repository: checkpoint GB and shard count, quantised sibling GB, each GGUF quant, any bundled
draft head, and `downloads` / `likes` / `createdAt` as adoption evidence.

**Sanity-check the parameter count against the bytes.** An FP8 checkpoint is one byte per parameter, so a
"1T" model must land within a few percent of 1,000 GB. If it does not, the quantisation is different or
the headline parameter count is wrong, and either is worth a sentence.

**Check the shipped precision.** `config.json` → `quantization_config` (for example `quant_method: fp8`,
`weight_block_size: [128,128]`) tells you whether full precision was released at all. If the reference
checkpoint is already quantised, FP16 exists only as a community conversion at roughly double the size;
say so rather than implying readers can fetch it.

## 3. Read the architecture

Keys to quote: `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `n_routed_experts`,
`num_experts_per_tok`, `n_shared_experts`, `moe_intermediate_size`, `vocab_size`,
`max_position_embeddings`, `layer_types`, and for latent attention `kv_lora_rank`, `qk_rope_head_dim`,
`index_topk`, `index_n_heads`.

- **Total versus active parameters.** Cards often give routed parameters (what a token's compute touches)
and a larger figure including embeddings and the output layer; press coverage quotes the larger one. State
both and say which predicts decode speed.
- **Cache arithmetic, labelled as arithmetic.** Latent attention stores
`(kv_lora_rank + qk_rope_head_dim) × num_hidden_layers` values per token; at one byte per value that is
the per-token cost, and long contexts multiply it. State the assumption. A sparse indexer cuts attention
compute, not cache memory, and conflating those two savings is the common error.

## 4. Derive hardware fit

| Accelerator | HBM each |
|---|---|
| H100 | 80 GB |
| H200 | 141 GB |
| B200 | 192 GB |
| GB200 | 192 GB |

For each rig, `total = count × capacity`, then check two things against the vendor's own serving command:

1. Does the checkpoint fit at all, i.e. `total − checkpoint`?
2. Does it fit at the utilisation the vendor recommends? Quants typically land near 0.85 usable, so
   `0.85 × total < checkpoint` means the documented example assumes bigger cards than the reader owns.

A table of rig → total → what fits, with leftovers, is the most useful artifact a deployment decision can
carry. It is arithmetic, not a benchmark, and should be labelled as such.

## 5. Read the licence as a clause, not a label

Modified-MIT variants usually grant use, modification, redistribution and derivative training, then carve
out **Model-as-a-Service**: reselling inference where the customer controls inputs or parameters needs a
separate commercial licence, while internal use and capabilities embedded in an end-user product stay
inside the grant. Name the practical line ("build and embed yes, resell inference no") and compare with
Apache-2.0 releases, including the same vendor's smaller models, because the restriction usually tracks
size rather than company policy.

## 6. Separate claims from facts

- **Primary beats press on architecture.** Cards separate routed from total active parameters and print
the exact GPU count where coverage rounds it. Quote the card and note the difference instead of silently
picking one.
- **Index scores are vendor-run.** An "Intelligence Index" or "Coding Agent Index" figure on a card was
measured by the vendor on a third-party harness; attribute it, and say the cross-vendor comparison is not
independent.
- **Token-efficiency claims are per-task economic claims.** The honest test is the reader's own traces:
same tasks, same harness, tokens per **successful** task, compared with a paired method.
- **Download counters are download events**, not deployment. Report them with that caveat, noting that a
multi-hundred-GB repository is not casually fetched.

## 7. Pitfalls

- **Do not tell readers to download BF16 and convert themselves** when the vendor ships a quantised
checkpoint and mirrors already publish every quant. That spends a day of bandwidth on a format that
exists upstream.
- **Transfer arithmetic needs its assumption in the sentence** ("on a link that sustains 2.5 MB/s").
Shard counts and per-file hashes are the practical detail that makes a large pull resumable, so quote
them.
- **Do not compare a quantised copy's quality to the reference** unless the vendor published that
comparison; most releases ship a quantised sibling precisely so nobody has to guess.
- **A size is not a verdict.** "635 GB" means nothing without the companion fact, here that the smallest
published quant is 233 GB, which is what changes what the model can actually answer for.

## Calibration example (measured 2026-11-01 from the HF API)

Mistral Large 4 (`mistralai/Mistral-Large-4-1T-A52B`), 1T total / 49B routed / 52B including embeddings,
72 layers, 128 routed experts with 4 active, 512-value latent KV, 1,048,576 context, "Mistral Ecosystem
Open-Weight License" (modified MIT plus a Model-as-a-Service clause):

| Artifact | GB |
|---|---|
| Reference checkpoint (FP8, 54 tensor shards) | 1,060.6 |
| NVFP4 sibling | 610.1 |
| GGUF BF16 | 2,098.9 |
| GGUF Q8_0 | 1,115.2 |
| GGUF Q4_K_M (13 shards) | 634.8 |
| GGUF Q2_K | 380.7 |
| GGUF IQ2_XXS | 297.7 |
| Unsloth UD-Q4_K_XL / UD-IQ1_S | 604.8 / 233.0 |
| DSpark draft head | 4.2 |

Fit arithmetic: 8×B200 = 1,536 GB (fits, ~475 GB spare); 8×H200 = 1,128 GB (fits with ~67 GB spare,
which is under a 0.85 utilisation target, so the vendor's `--tensor-parallel-size 8` example assumes the
bigger cards); 4×B200 = 768 GB (NVFP4 fits); 2×H200 = 282 GB (only the smallest quant fits, with no room
for context).
