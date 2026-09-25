---
layout: post
title: "One Licence, Three Checkpoints: What the MiMo-V2.6 Open Release Hands Builders"
date: 2026-09-25 00:00:00 +0300
categories: [AI Engineering, LLM]
tags: [open-weights, reinforcement-learning, grpo, model-release, quantization, agentic-ai, mit-license, xiaomi-mimo]
image:
  path: /assets/img/cover-mimo-v26-open-release-builders.webp
  alt: Three nested checkpoint boxes labelled 1.02T, 309B and 9.4B beside a GRPO ring split into rollout 43.8 percent, training 43.5 percent and grader 12.7 percent, with a green MIT weights badge
---

## Introduction

> **The short version**
> On 22 September 2026 Xiaomi's MiMo team released the **MiMo-V2.6 series**: a 1.02T-parameter flagship, a 309B sibling, and a 9.4B starter model distilled from the flagship — all three under the **MIT licence**, all three with weights on Hugging Face. The unusual part is not the benchmark table. It is that the reinforcement-learning loop that produced them — the task environments, the verifiers, the training framework and even the bill — was published alongside the weights.
{: .prompt-info }

Most "open-weight" releases hand you a checkpoint and a model card. Someone else's training pipeline stays dark, and you inherit whatever biases, reward hacks and data composition choices were baked in. MiMo-V2.6 is a different shape of release: three checkpoints covering three hardware budgets, plus a documented RL recipe and roughly 7,000 verified training tasks you can run yourself.

It is also the most useful open release of the month for anyone building outside a data centre. The smallest checkpoint in the family is 9.4B parameters, and community quantisations of it start at **3.54 GB**. That is a frontier-lab agent model you can plausibly hold on a laptop disk.

This post is about the size ladder and the published loop, not the leaderboard. We covered the scale race when [Kimi K3 landed](/posts/kimi-k3-china-open-source/) and on-device translation when the [TranslatePsy-AfriSLM models shipped](/posts/translatepsy-afrislm-offline-translation/). Here the question is narrower and more practical: **what can you actually download, run and reproduce on 25 September 2026?**

## What actually shipped

Xiaomi's own [model update log](https://mimo.mi.com/docs/en-US/updates/model) dates the series release to 22 September 2026 (the company teased it on X a day earlier). Three checkpoints were published on Hugging Face the same day:

| Checkpoint | Parameters | Context | Licence | Hugging Face (25 Sep) |
| --- | --- | --- | --- | --- |
| [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | 1.02T total / 42B activated | 1M tokens | MIT | 42,062 downloads · 485 likes |
| [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 309B total / 15B activated | 1M tokens | MIT | 20,473 downloads · 450 likes |
| [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) | 9.41B (SFT of Qwen3.5-9B) | card does not state | MIT | 6,652 downloads · 449 likes |

The parameter counts are worth reading twice, because the two published numbers tell different stories. The model card says 1.02T total with **42B activated** — it is a sparse mixture-of-experts backbone, so the compute per token is closer to a mid-size dense model than the headline count suggests. The Hugging Face safetensors index sums to 1,024.22B for Pro and 310.76B for Flash, which includes the modality encoders.

Both large checkpoints are omnimodal — text, image, video and audio in one model. The published architecture:

| Component | What the card specifies |
| --- | --- |
| Backbone | 70 layers, 60 sliding-window + 10 global attention; sparse MoE, 384 routed experts, 8 active (Pro) |
| Vision encoder | 681M-parameter MiMo ViT, 28 layers |
| Audio | 308M AudioTokenizer (20 RVQ codebooks) + 127M audio patch encoder |
| Speculative decoding | 5-layer multi-token-prediction drafter, 7 tokens per forward pass |
| Serving recipes | SGLang and vLLM commands published on the model card |

The Flash variant halves the expert count to 256 routed experts (still 8 active) and drops activation to 15B. The 1M-token context is the headline feature for agent work: an entire repository, or a multi-session tool trace, fits in one window. [SiliconANGLE's write-up](https://siliconangle.com/2026/09/22/xiaomi-introduces-mimo-v2-6-series-open-source-ai-model-family/) notes the same family is served through OpenRouter with a 1.05M-token context.

## The nine-gigabyte door

Here is the part that matters if you pay for your own bandwidth.

Full bf16 weights for the flagship are about 2.05 TB at two bytes per parameter (1,024.22B × 2); Flash is roughly 0.62 TB. Neither is a download, it is a procurement. The 9.4B starter checkpoint is the one that fits a normal life, and the community has already turned it into a full quantisation ladder. Real blob sizes from the Hugging Face file index:

| Build (GGUF) | File size | Rough RAM to serve |
| --- | ---: | --- |
| IQ2_M | 3.54 GB | ~4–5 GB |
| Q3_K_M | 4.48 GB | ~5–6 GB |
| **Q4_K_M** | **5.84 GB** | **~7–8 GB** |
| Q5_K_M | 6.88 GB | ~8–9 GB |
| Q6_K | 7.79 GB | ~9–10 GB |
| Q8_0 | 9.55 GB | ~11 GB |
| bf16 | 17.92 GB | ~20 GB |
| mmproj (vision projector) | 0.92 GB | add to any row above |

Bandwidth arithmetic, because this is where deployments die: at a steady 10 Mbps, the Q4_K_M file is **78 minutes**; the IQ2_M build is 47 minutes; the 17.92 GB bf16 GGUF is four hours; the flagship's 2.05 TB is **19 days** of continuous transfer. The RAM column is planning guidance for a quantised weight file plus context and runtime overhead — not a measured figure, and we say so below.

> **What we did not do**
> We could not pull the 5.84 GB checkpoint from this host inside the publishing window, so this post contains **no local inference benchmark and no throughput claim**. Every size above comes from the Hugging Face API and a real HTTP HEAD request; the runnable block below checks licence, parameter count and file size rather than tokens per second. Treat any "runs at N tokens/s" number for these quants as unverified until you measure it on your own hardware.
{: .prompt-warning }

The 9.4B is not the flagship in miniature — it is a supervised fine-tune of Qwen3.5-9B on MiMo-generated data across coding, cyber, general and visual tasks. The training mixture is published to one decimal place: 77.4B total tokens, of which 27.2B are loss-bearing, split Code 23.2B / Cyber 11.0B / General 22.0B / Visual 21.2B. Hugging Face lists it as an image-text-to-text model, and the community GGUF repo ships that 0.92 GB mmproj projector alongside the text quants, so vision travels with the small model too.

## The part that is genuinely unusual: the loop got published

The [technical report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf) (published on the model repo) states the goal plainly: *"We open-source the training dynamics, RL environments, and RL framework to facilitate reproduction and further research on scaled RL and model self-improvement."* That sentence is the release. The numbers behind it:

| Published RL fact | Value |
| --- | --- |
| Batch shape | 1,568 prompts × group size 16 = ~25K sequences per step |
| Tokens per step | 2.7B–3.7B (roughly 110K–150K tokens per sequence) |
| Cost to run RL | $2.6M for Pro, $0.9M for Flash |
| Where the money went (Pro) | rollout 43.8%, training 43.5%, grader 12.7% |
| DeepSWE v1.1 avg@3 over the run | Pro 58.4 → 72.6, Flash 48.7 → 65.7 |
| Stability trick | MoE router frozen during RL |
| Anti-reward-hacking | adversarial screening, verifier cross-checks, offline trajectory audits |

Two design decisions are worth stealing even if you never train a trillion-parameter model. First, **the grader gets real compute** — 12.7% of the Pro bill goes to agentic grading, because binary pass/fail cannot rank two solutions that both pass, and the groupwise grading scheme (rubrics synthesised offline from contrasting rollouts, advantage redistributed online toward higher-quality passing trajectories) is aimed at shorter, cheaper solutions rather than merely correct ones.

Second, **the environments are the product**. Xiaomi released roughly 7,000 verified training tasks with their verifiers:

| Domain | Released tasks | Verifier |
| --- | ---: | --- |
| Software engineering (Code) | ~3k | Executable tests |
| Vulnerability reproduction (Cyber) | ~1k | Rule checks |
| Knowledge work (General) | ~1k | Rubric-based judging |
| Visual / web development | ~2k | Visual grading |

Plus about 1,000 music-generation tasks, an end-to-end RL framework and a composable mini-harness for agentic interaction. That is a materially different gift from a checkpoint dump: environments with executable verifiers are the expensive, unglamorous part of any agentic RL pipeline, and they are usually the thing nobody publishes.

## Can the released stack actually move a small model?

Yes, and the report shows the receipts — which is the strongest part of the release. Starting from the *same* released 9.4B SFT checkpoint, Xiaomi ran domain-specific GRPO using the released environments. All eleven reported evaluations improved:

| Benchmark | Qwen3.5-9B | MiMo 9.4B (SFT) | MiMo 9.4B (+ RL) |
| --- | ---: | ---: | ---: |
| SWE-bench Verified (avg@3) | 60.0 | 61.1 | 66.2 |
| SWE-bench Pro (avg@3) | 32.0 | 44.6 | 47.6 |
| MiMo Code Bench mini (avg@3) † | 19.5 | 51.6 | 59.9 |
| MiMo Cyber Bench mini (avg@3) † | 5.7 | 31.3 | 47.0 |
| AutomationBench v1.0.6 (avg@1) | 5.0 | 30.3 | 33.1 |
| Terminal Bench 2.1 (avg@1) | 27.0 | 37.1 | 52.8 |
| Toolathlon-Verified (avg@1) | 25.9 | 35.2 | 38.0 |
| OfficeQA Pro (avg@1) | 9.0 | 19.5 | 24.8 |
| JobBench (avg@1) | 2.6 | 18.3 | 25.2 |
| MiMo General Bench mini (avg@1) † | 28.5 | 62.2 | 70.6 |
| MiMo Visual Coding mini (avg@1) † | 61.7 | 64.0 | 72.4 |

† internal evaluation sets.

The multi-harness experiment is the one to copy: jointly training across four mini-harnesses improved **all 21** dataset–harness pairs, including three held-out harnesses the model never trained on, with gains of 1.8 to 9.3 percentage points on the internal code bench. Generalisation across harnesses — not just across tasks — is the claim worth testing on your own stack, because it is the difference between a model that works in your eval script and one that works in your agent's runtime.

## Reading the claims honestly

Three caveats, in order of how much they should change your decisions.

**The SFT step is uneven, and the card shows it.** Against plain Qwen3.5-9B, the distil gains +1.1 on SWE-bench Verified (60.0 → 61.1) — inside the noise band for avg@3 on that benchmark — while AutomationBench jumps 5.0 → 30.3 and JobBench 2.6 → 18.3. Read that as "agentic behaviour transferred, generic coding barely moved". If your task is ordinary code completion, the distil is not obviously better than its base model; if your task is tool use and computer control, it is.

**Everything is self-reported, and the internal sets are marked as such.** Four of the eleven rows above are internal benches; the Pro-versus-closed comparison is Xiaomi's own testing. On the card, Pro leads Claude Opus 5 on AutomationBench (53.1 vs 50.3), ties it on Agents' Last Exam (31.6) and edges it on Terminal Bench 2.1 (89.9 vs 89.1) — and trails it clearly on ProgramBench (26.5 vs 37.0), Terminal Bench 4.0 (34.9 vs 49.0) and GDPval-AA 2.1 (1673 vs 1708). The "highest-ranked open-weight model at launch" line rests on a 46.32 score on the Artificial Analysis Intelligence Index v4.3, and it is a launch-day snapshot. We have not seen independent replication of any of these numbers yet.

**Vendor economics are vendor economics.** The claim that Pro costs about one-twentieth to one-sixtieth of comparable overseas models holds only with cached-token pricing in the mix. The list prices themselves are checkable and unchanged from the previous generation: Flash at $0.14 per million input and $0.28 per million output tokens, Pro at $0.435 and $0.87, and the latency-tuned Pro-UltraSpeed at roughly ten times Pro. Those are the numbers to budget with; the ratio claim is marketing.

## How to apply this

1. **Verify before you commit bandwidth.** Check the licence and the parameter count yourself rather than trusting a blog post — including this one. The block below is stdlib Python, no dependencies, and it prints the licence, parameter count and a HEAD-verified file size straight from the Hugging Face API:

{% raw %}
```python
import json, urllib.request

API = "https://huggingface.co/api/models/"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ml-ke-post-check/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


repos = [
    "XiaomiMiMo/MiMo-V2.6-Pro-RL",
    "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
]
for repo in repos:
    m = get(API + repo)
    params = (m.get("safetensors") or {}).get("total") or 0
    print(
        f"{repo:38s} license={m['cardData'].get('license', '?'):5s} "
        f"params={params / 1e9:7.2f}B downloads={m.get('downloads')}"
    )

q = get(API + "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF?blobs=true")
for f in q["siblings"]:
    if f["rfilename"].endswith("Q4_K_M.gguf"):
        url = "https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF/resolve/main/" + f["rfilename"]
        print(f"\n{f['rfilename']} listed at {f['size'] / 1e9:.2f} GB")
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ml-ke-post-check/1.0"})
        with urllib.request.urlopen(req, timeout=40) as r:
            print(
                f"HEAD {r.status} | content-length={r.headers.get('content-length')} "
                f"| x-linked-etag={r.headers.get('x-linked-etag')}"
            )
```
{% endraw %}

Output on 25 September 2026:

```text
XiaomiMiMo/MiMo-V2.6-Pro-RL            license=mit   params=1024.22B downloads=42062
XiaomiMiMo/MiMo-V2.6-Flash-RL          license=mit   params= 310.76B downloads=20473
XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B   license=mit   params=   9.41B downloads=6652

MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf listed at 5.84 GB
HEAD 200 | content-length=5841049120 | x-linked-etag=None
```

Download counts move daily; the licence and the byte counts are what you are checking.

2. **Pick the checkpoint by hardware, not by benchmark.** One 80 GB accelerator: Flash quantised, expect to work for it. A workstation with 24–48 GB of VRAM: Flash at reduced precision, or Pro through an inference provider. A laptop or a single consumer GPU: the 9.4B starter, and budget the mmproj projector only if you need vision. A phone or an offline field device: the IQ2 class build, with the honest expectation that reasoning quality degrades before size becomes comfortable.

3. **If you do RL research, start from the 9.4B and the released environments.** The report's baselines are reproducible in principle precisely because the tasks, verifiers and framework are published. Table 6 gives you eleven numbers to beat; the multi-harness result gives you a generalisation test rather than a single-benchmark target.

4. **If you ship a product, the useful part is the price tier.** Flash at $0.14/$0.28 per million tokens with a 1M context is a genuinely different economics for document-heavy agents in markets where per-request cost decides whether a feature survives. Prototype on the local 9.4B to keep iteration free, then move to the hosted Flash tier for production traffic.

5. **Watch for the two things that would change the picture:** third-party replication of the benchmark rows, and evidence that the router-freezing plus groupwise-grading recipe transfers to models small enough to fine-tune in a university lab. If it does, the barrier to doing real agentic RL research drops from "thousands of GPUs" to "one node and a well-verified environment".

## Key takeaways

| Point | Detail |
| --- | --- |
| What shipped | Three MIT-licensed checkpoints (1.02T/42B, 309B/15B, 9.4B) plus RL environments, framework and mini-harness, 22 September 2026 |
| Why it is different | Training dynamics, ~7k verifier-backed RL tasks and the RL cost breakdown were published, not just weights |
| Smallest useful build | 9.41B SFT checkpoint; IQ2_M quant is 3.54 GB, Q4_K_M is 5.84 GB (78 min at 10 Mbps) |
| Best evidence | All 11 reported evals improved from the released 9.4B via GRPO; 21/21 dataset–harness pairs improved with multi-harness training |
| Main caveat | Self-reported; five rows use internal benches; the SFT gain on SWE-bench Verified is only +1.1 |
| Verified here | Licences, parameter counts and file sizes via the Hugging Face API and an HTTP HEAD request — not a throughput benchmark |

## References

- [MiMo-V2.6-Pro-RL model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) and [technical report (PDF)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf) — architecture, RL batch shape, $2.6M/$0.9M cost, cost split, DeepSWE progression, released environments.
- [MiMo-V2.6-Flash-RL model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) — 309B total / 15B activated, 1M context, 256 routed experts.
- [MiMo-V2.6-Distill-Qwen-9B model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) — SFT data composition, Table 6 baselines.
- [Xiaomi MiMo model update log](https://mimo.mi.com/docs/en-US/updates/model) — 22 September 2026 release entry and the three SKUs.
- [Hugging Face models API](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Pro-RL) — licence and safetensors parameter totals used in the verification block.
- [bartowski GGUF builds](https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF) and [ggml-org GGUF builds](https://huggingface.co/ggml-org/MiMo-V2.6-Distill-Qwen-9B-GGUF) — quantisation ladder sizes and mmproj projector.
- [SiliconANGLE: Xiaomi introduces Mimo-V2.6 series](https://siliconangle.com/2026/09/22/xiaomi-introduces-mimo-v2-6-series-open-source-ai-model-family/) — omnimodal detail, benchmark comparisons, API pricing, OpenRouter context.
- [TestingCatalog: Xiaomi open-sources MiMo-V2.6 Pro and Flash](https://www.testingcatalog.com/xiaomi-open-sources-mimo-v2-6-pro-and-flash-models/) — RL run summary and pricing cross-check.

## Related posts

- [Kimi K3: China's 2.8 Trillion Parameter Challenge to Silicon Valley](/posts/kimi-k3-china-open-source/)
- [Nineteen Languages, One Download: Reading the TranslatePsy-AfriSLM Release Past the Headline](/posts/translatepsy-afrislm-offline-translation/)
- [Self-Hosting Open-Weight LLMs](/posts/self-hosting-open-weight-llms/)
- [Serving LLMs with vLLM](/posts/vllm-llm-serving/)
- [MLOps in Constrained Environments](/posts/mlops-constrained-environments/)
- [RAG Recall@k: The Denominator Problem](/posts/rag-recall-at-k-denominator/)
