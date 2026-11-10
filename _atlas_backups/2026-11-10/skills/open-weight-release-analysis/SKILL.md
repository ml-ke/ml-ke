---
name: open-weight-release-analysis
description: "Use when a lab ships model weights: measure, licence, fit."
version: 1.0.0
author: Hermes curator
license: MIT
metadata:
  hermes:
    tags: [open weights, licensing, self-hosting, quantization, hardware sizing, model releases]
---

# Open-weight release analysis

A weights release is a logistics event disguised as a capability event. Capability was announced
weeks earlier and is now syndicated everywhere; what actually changed is that a team can hold the
artifact, read its licence, and decide whether it fits on hardware. Those three things are
measurable, and measuring them is what makes the writeup worth reading or the procurement memo
worth signing.

## When to Use

Use this skill for two jobs, which share one procedure:

- writing about a release (blog post, newsletter, brief) where the differentiator must be measured
  facts rather than benchmark paraphrase;
- answering "can we self-host this?" with numbers instead of vibes.

If the output is a post for the ML Kenya blog, load the `blog-drafting` skill afterwards for house
format, cover images and the prose-style gate. This skill supplies the numbers and the framing.

## Procedure

### 1. Locate the repositories, including the quantised siblings

```bash
curl -s "https://huggingface.co/api/models?author=<org>&sort=createdAt&direction=-1&limit=12" \
  | python3 -c "import sys,json; [print(m['modelId'], m.get('createdAt'), m.get('downloads')) for m in json.load(sys.stdin)]"
```

Look for a base checkpoint, a vendor-quantised sibling (FP8/NVFP4/AWQ), and community GGUF
conversions (`ggml-org`, `unsloth`). The GGUF repos are what local teams can actually run; the
vendor checkpoint is what defines the licence and the reference quality.

### 2. Measure the artifacts

```bash
python3 ~/.hermes/skills/mlops/open-weight-release-analysis/scripts/hf-artifact-size.py \
  mistralai/Mistral-Large-4-1T-A52B \
  mistralai/Mistral-Large-4-1T-A52B-NVFP4 \
  ggml-org/Mistral-Large-4-1T-A52B-GGUF \
  unsloth/Mistral-Large-4-1T-A52B-GGUF --config
```

The script sums published file sizes per subdirectory (one quantisation per line), and prints the
download counter, creation date, licence and base model. Two uses:

- **Parameter count, checked arithmetically.** One trillion parameters at one byte per parameter
  must land near 1,000 GB. A measured 1,060.6 GB across 54 tensor shards confirms the announced
  count; a total half that size means the repo is quantised, sharded differently, or the count is
  marketing.
- **The hardware conversation, with real inputs.** Sizes feed the fit table in step 5.

### 3. Read `config.json` for geometry, then compute the cache

The same script with `--config` prints layers, hidden size, heads, expert counts, vocabulary,
context length, `quantization_config`, and, for latent/MLA attention, bytes per token:

```
kv bytes/token = (kv_lora_rank + qk_rope_head_dim) x num_hidden_layers   # at 1 byte per value
```

State the assumption out loud (1 byte per value, full context stored) and label the result an
estimate. It matters because at 1M tokens a 72-layer, 512-latent model costs roughly 41 GB on top
of the weights, per concurrent request.

### 4. Read the licence, not the adjective

```bash
curl -sL "https://huggingface.co/<repo>/raw/main/LICENSE" | head -40
```

Classify it in one of three buckets and say which:

| Bucket | What it permits | What to check |
|---|---|---|
| Permissive (Apache 2.0, MIT) | commercial use, resale of inference | attribution only |
| Open-weight with a service clause ("modified MIT") | use, modify, fine-tune, embed in products, internal use | a Model-as-a-Service carve-out: reselling raw inference needs a separate commercial licence |
| Research / non-commercial | research and evaluation only | whether "commercial" is defined anywhere |

Also record whether the restrictive licence tracks model size: an organisation often ships 3B and
100B models under Apache 2.0 and applies the serving clause only to the flagship. That pattern is a
fact worth a sentence, and it changes which artifact a startup should build on.

### 5. Compute hardware fit against the vendor's own flags

Build the table from published accelerator capacities, then compare with the vendor's recommended
`--tensor-parallel-size` and `--gpu-memory-utilization`:

```
left_over = n_gpus x HBM_per_gpu x utilisation - checkpoint_bytes
```

Worked example (FP8 checkpoint 1,060.6 GB):

| Rig | Total HBM | At 0.85 utilisation | Verdict |
|---|---|---|---|
| 8x B200 (192 GB) | 1,536 GB | 1,306 GB | fits, ~245 GB left for cache |
| 8x H200 (141 GB) | 1,128 GB | 959 GB | does not fit at the recommended utilisation |

That last row is the finding that makes the analysis useful: the naive "1,128 > 1,060, so it fits"
comparison is wrong once the vendor's own utilisation target is applied.

### 6. Check adoption, carefully

Download counters are events, not deployments. A few thousand downloads for a 600 GB repository in
its first days means a small audience of engineers and evaluators pulling a quantised copy, and
saying so is more honest than quoting the launch's press coverage. Say what the counter measures.

### 7. Reconcile vendor and press numbers

When launch coverage and the model card disagree (cluster size, active-parameter count, context),
quote the card and note the press figure. The card is the primary source, and the discrepancy is
itself a paragraph a reader values.

### 8. Shape the output as a ladder

End with the decision, ordered by capital:

1. call the API (no download, reversible);
2. rent a machine and serve a quantised copy (state the quality trade and the disk figure);
3. buy the metal (state GPUs, HBM, checkpoint size);
4. run something smaller that fits hardware you already own, which is often correct and is almost
   never discussed.

## Pitfalls

- **Re-telling the launch is the failure mode.** If a weekly roundup already covered the model, the
  follow-on post must be the download, licence, size or hardware angle. Grep the roundups for the
  model name before drafting; a release-week post that repeats the announcement adds nothing.
- **Do not compute fit from active parameters.** A 1T MoE with 49B active costs 49B of compute per
  token and ~1TB of memory. Every expert must be resident; the router can pick any of them per token.
- **Sparse or indexed attention cuts compute, not memory.** An indexer that selects top-2048 keys per
  query removes the quadratic attention arithmetic; the KV cache is still stored in full.
- **A community BF16 conversion is not the release.** If the vendor checkpoint ships pre-quantised
  (FP8 with 128x128 blocks is common), the "full precision" download is a third-party artifact at
  roughly double the bytes, with no vendor quality claim.
- **Quantised GGUF tiers are not equivalent products.** 233 GB and 635 GB copies of the same model
  differ by more than download time; name the tier you are recommending and why.
- **Re-measure before quoting.** Repository contents change (new quants, re-shards, deleted files).
  Cite the day you measured, and rerun the probe rather than reusing a number from a draft.

## Supporting files

- `scripts/hf-artifact-size.py` — the probe (stdlib only, deterministic): per-directory sizes,
  counters, licence, base model, and `--config` geometry plus KV estimate.
- `references/verified-release-anchors.md` — measured numbers for releases already analysed here, so
  a later session reuses verified figures instead of re-deriving them.
