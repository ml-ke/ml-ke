---
title: "635 gigabytes of free: what a trillion-parameter open release costs to own"
date: 2026-10-30 00:00:00 +0300
categories: [Machine Learning, LLM]
tags: [open weights, mistral, mixture of experts, quantization, self-hosting, licensing, inference cost]
image:
  path: /assets/img/cover-open-weights-download-cost.webp
  alt: A download bar filling beside a stack of drive platters, with the labels 1,060 GB, 635 GB and 233 GB marking three quantization tiers of the same model
---

> **The short version**
> On 29 October Mistral published the weights of Mistral Large 4, a trillion-parameter mixture-of-experts model. The download is free. The smallest usable copy of the model is 233 GB and the copy Mistral ships as the reference checkpoint is 1,060.6 GB. That gap, between free and ownable, is the whole story of open-weight releases in 2026.

A weights release is usually reported as a capability event. This one is better read as a logistics event. The model's abilities were announced on 6 October; what changed on 29 October is that a team with no contract with Mistral can now hold the artifact, audit it, and decide for itself what to do with it. That decision has a price list attached, and the price list is measurable in a way that benchmark charts are not.

This post reads the release the way an engineering team in Nairobi or Lagos would read it: what shipped, what the licence actually permits, what the files weigh, what hardware they fit on, and which of Mistral's claims survive contact with the artifact.

## The release, and what came with it

Mistral released the weights of Mistral Large 4, nicknamed Le Chonk, on Hugging Face and through its own API as `mistral-large-4`, [TechCrunch reported on the day](https://techcrunch.com/2026/10/29/mistral-releases-weights-for-its-trillion-parameter-ai-model-targeting-cybersecurity/). The model had been in preview since 6 October, when the company [announced it as the strongest open model built outside China](https://www.cnbc.com/2026/10/06/mistral-ai-model-le-chonk.html).

Three things came with the weights beyond the tensors.

The first is tooling. The model card lists support in vLLM, SGLang, transformers and llama.cpp, with fine-tuning paths through Axolotl, Unsloth and TRL. The text and vision checkpoints are converted, and the GGUF conversions sit in two community repositories, `ggml-org` and `unsloth`. A release that lands in llama.cpp on day one is a release a 16 GB laptop can at least attempt.

The second is a quantised sibling. Alongside the reference checkpoint there is an [NVFP4 build](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4) aimed at Blackwell GPUs, which Mistral says matches the FP8 checkpoint's behaviour across CyberGym, SWE-bench Pro, Terminal-Bench 4, DeepSWE, AutomationBench and Graphwalks 512k while using less memory.

The third is a speculative decoding head called DSpark, embedded in the checkpoint as a 4.2 GB BF16 draft model. Speculative decoding needs a small draft model that guesses the next tokens and a big model that verifies them. Shipping the draft inside the release removes the usual step of finding a compatible one.

## The model in numbers

The card describes a natively multimodal mixture of experts. The routing is granular: 128 routed experts, four active per token, plus one shared expert that every token passes through.

| Property | Value | Source |
|---|---|---|
| Total parameters | ~1 trillion | model card |
| Active per token | 49B routed, 52B including embeddings and output layers | model card |
| Layers | 72 | config.json |
| Hidden size | 6,144 | config.json |
| Attention heads | 64, with a 512-dimension latent KV cache | config.json |
| Routed experts | 128 (4 active) plus 1 shared | config.json |
| Vocabulary | 204,800 tokens | config.json |
| Context window | 1,048,576 tokens | config.json |
| Training hardware | 3,800 NVIDIA Grace Blackwell GPUs, two months, European datacenters | model card |
| Vision encoder | Pixtral, 30 layers, 1,540-pixel images | config.json |

The 49B and 52B figures are a refinement worth noticing. The October launch coverage, including [CNBC's](https://www.cnbc.com/2026/10/06/mistral-ai-model-le-chonk.html), reported 52 billion active parameters. The card separates the two: 49B participate in the compute of a token, and 52B is the count once the embedding table and output layer are included. Both numbers are correct. Only one of them predicts how fast the model generates.

The architectural detail that matters for anyone planning to run it is the attention. Every one of the 72 layers is an indexed attention layer with an indexer that selects the top 2,048 keys for each query, and the key-value state is stored as a 512-value latent per token rather than as separate key and value tensors per head. This is the DeepSeek-style latent cache, and it is the reason a million-token context is plausible at all: the cache grows at roughly 40 KB per token rather than the 512 KB a conventional 64-head model would need.

## Open weight is a licence, and this one has a clause

The release uses the Mistral Ecosystem Open-Weight License, a modified MIT licence. Modified how is the part people skip.

Read plainly, the grant is broad. You may use, copy, modify, merge, publish, distribute, sublicense and sell copies of the model, and you may run, deploy, further train and build derivative works from it. Attribution is required, in the ordinary MIT sense.

Then comes section 2. If you or an affiliate operate a Model-as-a-Service business, you need a separate commercial licence from Mistral before using the model or a derivative for any commercial purpose. The licence defines Model as a Service as giving a third party access to the model for inference or post-training in a way that lets that third party exercise meaningful control over inputs, parameters or training data.

Two carve-outs follow, and they decide most real projects. Internal use is exempt, defined as any use that does not make the model available to third parties. End-user products where the model's capabilities are embedded inside specific features are also exempt, as is merely relaying requests to somebody else's hosted model.

So the practical line runs between building and reselling. A Kenyan fintech that embeds the model behind its own document-review feature, with its own users and its own prompts, is inside the grant. A Nairobi developer who rents out raw inference on the same weights, with customers choosing their own prompts and parameters, is outside it and needs to talk to Mistral's commercial team. Anyone comparing this licence to Apache 2.0 should be clear that the difference is not cosmetic: it is the right to compete with the licensor on serving.

For a team weighing a frontier model against a Chinese open release, this clause is a real line item, not a footnote. Apache-licensed Chinese models carry no equivalent restriction. The trade being offered is capability plus a European legal jurisdiction, against a narrower right to operate it commercially.

The restriction also tracks size rather than company policy. Mistral published Shieldstral, a 3B safety classifier, under Apache 2.0 in July, and Leanstral, a 119B proof-agent model, under Apache 2.0 as well. The one-trillion-parameter flagship is the model that comes with the serving clause.

## The download is the price

Here is where the post stops paraphrasing and starts measuring. Every figure in the next two tables was read from the Hugging Face repositories on 1 November 2026 by summing the file sizes the API reports, not from a press release.

| Artifact | Repository | Size |
|---|---|---|
| Reference checkpoint (FP8) | mistralai/Mistral-Large-4-1T-A52B | 1,060.6 GB |
| NVFP4 checkpoint | mistralai/Mistral-Large-4-1T-A52B-NVFP4 | 610.1 GB |
| GGUF, BF16 | ggml-org | 2,098.9 GB |
| GGUF, Q8_0 | ggml-org | 1,115.2 GB |
| GGUF, Q4_K_M | ggml-org | 634.8 GB |
| GGUF, Q2_K | ggml-org | 380.7 GB |
| GGUF, IQ2_XXS | ggml-org | 297.7 GB |
| Unsloth dynamic, UD-Q4_K_XL | unsloth | 604.8 GB |
| Unsloth dynamic, UD-IQ1_S | unsloth | 233.0 GB |
| DSpark draft head | unsloth | 4.2 GB |

The headline finding is the first row. One trillion parameters at one byte per parameter should be about 1,000 GB, and the reference checkpoint comes in at 1,060.6 GB across 54 tensor shards. The arithmetic confirms the parameter count in a way that no announcement can. It also tells you something less convenient: the "open" checkpoint ships pre-quantised to FP8 with 128 by 128 blocks, per the `quantization_config` block in its own `config.json`. If you want the full-precision weights you are relying on a community BF16 conversion at 2,098.9 GB, roughly double the disk for a fidelity difference the vendor decided you did not need.

Now the hardware. These are the standard accelerator memory capacities, and the point of the table is that only some of the rows close.

| Rig | Total HBM | What fits |
|---|---|---|
| 8x B200 (192 GB each) | 1,536 GB | FP8 checkpoint, with about 475 GB left for cache and activations |
| 8x H200 (141 GB each) | 1,128 GB | FP8 checkpoint, with 67 GB left, which is under a typical 0.85 memory-utilisation target |
| 4x B200 | 768 GB | NVFP4 checkpoint at 610.1 GB, with about 158 GB left |
| 2x H200 | 282 GB | UD-IQ1_S at 233.0 GB, with 49 GB left and no room for a serious context |
| 1x H100 (80 GB) | 80 GB | nothing in this release |

That third column contains the decision. The vLLM command on the model card asks for a tensor-parallel size of eight and a GPU memory utilisation of 0.85. On eight B200s that is 1,306 GB of usable memory against a 1,060.6 GB checkpoint, which leaves room for a real KV cache. On eight H200s the same 0.85 target gives 959 GB, less than the checkpoint alone. Eight H200s can run this model, but not at the utilisation the vendor recommends and not with a large context, unless you drop to the NVFP4 build or add another node.

None of that is a defect. It is the honest shape of a trillion-parameter model. It does mean the sentence "you can download and deploy it yourself" has a hardware clause that announcements rarely spell out.

## The transfer is not free either

Bandwidth is a line item that never appears on a model card. 635 GB is 634,800 megabytes. On a link that sustains 2.5 MB/s, roughly a 20 Mbit/s connection, that is about 70 hours of uninterrupted transfer. At 12.5 MB/s it is about 14 hours. Neither is impossible, and both need a plan that survives a dropped connection.

The file layout helps. The Q4_K_M conversion arrives as 13 shards of 43 to 50 GB, and the FP8 checkpoint as 54 tensor shards, so a pull can resume file by file and each shard can be checked against the hash the repository publishes. The failure mode to avoid is pulling the 2,098.9 GB BF16 GGUF in order to do your own conversion: the vendor checkpoint is already quantised to FP8, and the community repositories already publish Q8, Q4, Q2, IQ2 and the dynamic Unsloth variants. Spending four times the bytes to reach a format that exists upstream is a day of bandwidth for nothing.

The route most teams end up taking is to rent an instance in the same region as the object storage, pull and convert there, and move only the artifact they intend to serve. That keeps the 635 GB transfer inside a datacenter and shrinks the part crossing your own link to the size of the final file.

## The siblings most teams will actually run

A trillion-parameter release is a flag planted at the top of a range, and the rest of the range shipped through 2026. Mistral's Hugging Face organisation lists, within a few months of each other, [Leanstral 1.5](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B), a 119B total and 6B active proof agent for Lean 4; [Shieldstral 1.0](https://huggingface.co/mistralai/Shieldstral-1.0-3B), a 3B third-party-policy safety classifier that returns a continuous score in one forward pass; a [time-series family](https://huggingface.co/mistralai/Timestral-209M) from 5M to 209M parameters; and Arabic speech and language-identification models.

Those numbers describe hardware a team here already owns. A 3B classifier at FP8 is about 3 GB and fits a single consumer GPU; a 209M forecaster runs on a CPU; the 119B model at 6B active is the interesting middle, because mixture-of-experts at that scale fits a workstation-class machine with enough system memory while reaching reasoning quality no 8B dense model matches.

The honest reading of the family is that the frontier artifact sets the ceiling and the small artifacts do the work. A team that downloads 635 GB to show a board has bought a benchmark. A team that deploys a 3B guard model on the same day it reads about the 1T model has bought a capability.

## How to test the claim on your own workload

The 47 percent figure is a prompt-level economics claim, and the only version that matters is the one measured against your own logs. The protocol that fits into a week looks like this.

Pick 40 to 60 tasks from production traces that have a checkable outcome, not tasks invented for the exercise. Run both models on the same tasks through the same tool harness, and record three things per task: whether it succeeded, tokens consumed, and wall-clock time. Compare success rates with a paired method, because the same tasks are used on both sides, and compare tokens per successful task rather than tokens per task, because a model that fails cheaply is not cheap. The [paired significance test](/posts/paired-eval-significance/) post works through the statistics; the token accounting is the part that is new here.

Expect the gap to shrink. Vendor token-efficiency numbers are measured on harnesses the vendor tuned, and an agent calling your tools in your order will spend tokens differently. A claim that survives at 30 percent on your traces is still a strong reason to switch. A claim that vanishes tells you the comparison was measuring the harness.

## Why the memory is larger than the compute

A mixture of experts separates two costs that dense models tie together. The compute per token is set by the active parameters, 49 billion here. The memory is set by all of them, roughly a trillion.

That is why the throughput story and the procurement story diverge. Generating a token costs about what a 49B model costs, so a well-provisioned instance can be fast. Holding the weights costs about 1,060 GB at FP8, so the machine has to be large and the model has to be fully resident. Mixture-of-experts does not let you leave 90 percent of the weights on disk; every token can route to any expert, and the router decides per token.

To that fixed cost you add the cache. With a 512-value latent per token across 72 layers, one byte per value gives about 40 KB per token, so a million-token prompt adds roughly 41 GB before the indexer's own state. Halve the context and you halve that. Two concurrent million-token requests and the cache alone is 83 GB, which is why long-context serving quotes are usually dominated by cache, not by weights.

There is a second reason the cache is survivable at all. The indexer picks 2,048 keys per query, so attention compute at a million tokens scales with 2,048 rather than with a million. The KV cache is still stored in full; what the indexer removes is the quadratic blow-up in attention arithmetic. Teams that assume sparse indexing means cheap long context get the compute saving without the memory one.

## What the 47 percent claim actually means

Mistral's economic argument is not about raw capability. It is about tokens per task.

On the Artificial Analysis Coding Agent Index the card reports a score of 48 percent, described as effectively tied with GLM-5.3, the leading Chinese open-weight model, while using 47 percent fewer tokens per task on average. On the Artificial Analysis Intelligence Index v5 the model scores 26, the highest among western open-weight models, with the card conceding that Chinese open models score higher on that aggregate.

Read the pairing together. A coding agent that reaches the same index score while spending 47 percent fewer tokens is cheaper per completed task, and agentic coding is a workload where token spend is the dominant cost. That is a more useful claim than a benchmark delta, because it is denominated in the currency the buyer actually spends.

It is also a vendor-run comparison. The indices belong to Artificial Analysis, but the numbers and the "47 percent fewer tokens" figure come from Mistral's own card, measured on Mistral's own harness. Independent replication of the token-efficiency claim across a team's real repositories is the only version that matters for procurement, and it has not happened yet in public.

## Where the numbers disagree

Three discrepancies surfaced while checking this release, and each is a small lesson in reading announcements.

The GPU count is one. The model card says the model was trained from scratch on 3,800 NVIDIA Grace Blackwell GPUs. The October launch coverage said 4,000. Both are plausible roundings of a cluster, but the card is the primary source and 3,800 is the number to quote.

The active-parameter count is another, covered above: 52 billion in launch coverage, 49 billion routed plus 52 billion including embeddings on the card.

The third is adoption. Three days after the release, the Hugging Face counters showed 4,576 downloads for the FP8 checkpoint, 631 for the NVFP4 build, 4,494 for the ggml-org GGUF repository and 10,269 for Unsloth's. Those are download events, not organisations running the model, and a 635 GB repository is not casually fetched. Compare that with any small model's counters and the gap tells you the real audience for a trillion-parameter open release in its first week: a few thousand engineers and evaluators, most of them pulling a quantised copy to see what breaks.

## What a team here can actually do with 635 gigabytes

The honest options, in ascending order of capital.

Call the API. Mistral serves the same weights as `mistral-large-4`, with no download, no hardware and no operations. For most product work this is the correct first move, and the open release is what makes it a reversible one: if the price or the terms change, you have an exit that does not require rewriting your application.

Run a quantised copy on rented GPUs. A 233 GB to 635 GB GGUF is a different product from the reference checkpoint. At IQ1 or Q2 the model has been compressed by a factor of four or more, and the quality loss on long-context reasoning is real, which is exactly why the release of NVFP4 matters: it is the vendor's own answer to the compression question, at 610.1 GB, on Blackwell only.

Run it on your own metal. Eight B200s is a capital project, not a line item. For the institutions that genuinely need inference inside a national border, that is the point of the release, and it is now a budget question with a 1,060.6 GB answer instead of a negotiation with a vendor.

Run something smaller. The most common correct decision for a team in Nairobi is the least discussed. A 30B or 8B model that fits on hardware you already own, tuned on your own data, will beat a trillion-parameter model you cannot afford to serve on most narrow tasks. The open release expands the ceiling. It does not move the floor.

What the release changes for a builder is bargaining position, not capability. The capability was available through an API since 6 October. What arrived on 29 October is the ability to verify the artifact, keep it if the vendor changes direction, and answer the question "what happens if this company disappears or is acquired" with a file instead of a promise. That is worth 635 GB, and it is worth knowing that 635 GB is what it costs.

## What to watch

The next data point is Kimi K3.1, whose weights were scheduled for 4 November, which will test whether the Chinese open tier keeps its aggregate-index lead while Mistral holds the token-efficiency argument. The one after that is independent measurement: somebody reproducing the 47 percent on a real repository, with their own harness, and publishing the receipts.

Until then, treat the release as what it verifiably is. A trillion-parameter model, pre-quantised to FP8, licensed for commercial use with a serving carve-out, downloadable today at a size you can measure, and servable on a machine most teams will have to rent.

> **What this post measured, and what it quoted**
> Artifact sizes, layer counts, vocabulary, context length and attention geometry were read from the Hugging Face API and the repository's `config.json` on 1 November 2026. Training hardware, parameter counts, licence terms and benchmark scores come from Mistral's model card, its licence file and press coverage, and are attributed as such. The hardware-fit table is arithmetic on published accelerator capacities, not a benchmark.

## References

- Mistral Large 4 model card — [huggingface.co/mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
- Mistral Large 4 NVFP4 checkpoint — [huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4)
- Mistral Ecosystem Open-Weight License — [LICENSE](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/main/LICENSE)
- Model configuration — [config.json](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/main/config.json)
- Weights release coverage, 29 October 2026 — [TechCrunch](https://techcrunch.com/2026/10/29/mistral-releases-weights-for-its-trillion-parameter-ai-model-targeting-cybersecurity/)
- Launch coverage, 6 October 2026 — [CNBC](https://www.cnbc.com/2026/10/06/mistral-ai-model-le-chonk.html)
- Mistral release post — [mistral.ai/news/mistral-large-4-oss](https://mistral.ai/news/mistral-large-4-oss/)
- GGUF conversions — [ggml-org](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF) and [unsloth](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF)
- API model page — [docs.mistral.ai/models/mistral-large-4-0](https://docs.mistral.ai/models/mistral-large-4-0)

## Related posts

- [A mine, forty clinics, ten million downloads: what open weights did for TB screening](/posts/open-weights-tb-screening-front-line/)
- [How many lists, how many probes: sizing an IVF index against a recall budget](/posts/ivf-index-sizing-recall-budget/)
- [Small AI by design: ten Kenyan startups and the offline-first constraint](/posts/kenya-ai-academy-offline-first/)
