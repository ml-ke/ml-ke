# Verified release anchors

Measured numbers for releases already analysed here. Reuse them, and re-measure before quoting
any figure you are about to publish in a new piece (repository contents change; the probe is
cheap).

---

## Mistral Large 4 ("Le Chonk"), weights released 29 October 2026 — measured 1 November 2026

Probed with `scripts/hf-artifact-size.py ... --config`.

**Model facts (model card):** 1T total, 49B active per token routed (52B including embeddings and
output), 128 routed experts with 4 active plus 1 shared, 72 layers, hidden 6,144, 64 heads,
`kv_lora_rank` 512, `qk_rope_head_dim` 64, `index_topk` 2048 per query on every layer, vocab
204,800, `max_position_embeddings` 1,048,576, Pixtral vision encoder (30 layers, 1,540-pixel
images), trained from scratch on 3,800 NVIDIA Grace Blackwell GPUs in two months in European
datacenters, embedded DSpark speculative draft head (4.2 GB BF16).

**Measured sizes (GB, summed from the HF API):**

| Artifact | Repository | Size |
|---|---|---|
| Reference checkpoint, FP8 with 128x128 blocks | mistralai/Mistral-Large-4-1T-A52B | 1,060.6 (54 tensor shards) |
| NVFP4 checkpoint | mistralai/Mistral-Large-4-1T-A52B-NVFP4 | 610.1 |
| GGUF BF16 / Q8_0 / Q4_K_M / Q2_K / IQ2_XXS | ggml-org | 2,098.9 / 1,115.2 / 634.8 / 380.7 / 297.7 |
| Unsloth dynamic UD-Q4_K_XL / UD-IQ1_S | unsloth | 604.8 / 233.0 |

**Licence:** Mistral Ecosystem Open-Weight License, a modified MIT. Commercial and non-commercial
use, modification, fine-tuning and embedding in products are permitted; operating a
Model-as-a-Service business requires a separate commercial licence. Internal use and end-user
products with the model embedded in features are explicitly exempt. Mistral's Shieldstral 3B and
Leanstral 1.5 119B-A6B ship under Apache 2.0, so the serving clause tracks the flagship, not the
company.

**Hardware fit (computed):** 8x B200 at 0.85 utilisation = 1,306 GB usable, so the 1,060.6 GB
checkpoint leaves about 245 GB for cache. 8x H200 at 0.85 = 959 GB, less than the checkpoint, so
the vendor's recommended `--tensor-parallel-size 8 --gpu-memory-utilization 0.85` implies
Blackwell-class memory, not H200. NVFP4 at 610.1 GB fits 4x B200 (768 GB total).

**Cache:** (512 + 64) x 72 = 41,472 B/token at one byte per value, so roughly 41 GB per 1M-token
context per request, before indexer state. The 2,048-key indexer reduces attention compute, not the
stored cache.

**Adoption at capture (three days after release):** 4,576 downloads on the FP8 checkpoint, 631 on
NVFP4, 4,494 on the ggml-org GGUF repo, 10,269 on Unsloth's. Download events, not deployments.

**Discrepancies worth keeping:** launch coverage said 4,000 GPUs and "52 billion active"; the card
says 3,800 GPUs and 49B routed active (52B including embeddings). Quote the card, note the press
figure.

**Claims to attribute, never restate as fact:** Artificial Analysis Coding Agent Index 48%
"effectively tied" with GLM-5.3 while using 47% fewer tokens per task; Intelligence Index v5 = 26,
described as highest among western open-weight models. All vendor-reported from Mistral's own card
and harness.
