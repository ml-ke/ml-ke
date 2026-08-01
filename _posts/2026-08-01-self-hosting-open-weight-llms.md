---
layout: post
title: "How to Self-Host Open-Weight LLMs: A Practical Guide"
date: 2026-08-01 00:00:00 +0300
image:
  path: /assets/img/cover-series-practical-playbook.webp
  alt: cover series practical playbook
categories: [AI Engineering, LLM]
tags: [self-hosting, open-source-llm, vllm, ollama, inference, cost-optimization]
---

The balance of power in large language models shifted decisively in 2026. With Kimi K3 released as open weights, alongside GLM-5.2, Llama 4, Qwen 3, and DeepSeek V3, the era of proprietary-only frontier AI is over. The question is no longer whether you *can* run these models yourself — it's whether you *should*.

For consistent, production-grade workloads, the answer is increasingly yes. Here's why, and how.

## Why Self-Host?

API pricing from the major providers has settled into a narrow band: GPT-5.6, Claude 4.5 Opus, and Grok 4.5 all hover around **$2 per million tokens** for input. That's affordable for prototyping. For a customer-facing chat application serving 100,000 conversations a day? The bill climbs past **$20,000 per month** — fast.

Self-hosting flips the economics. If you're running a 70B-parameter model in 4-bit quantisation on two RTX 4090s (roughly **$6,000 one-time hardware cost**), your marginal cost per token is electricity and cooling — typically **60–80% cheaper** than API pricing at scale. The breakeven point for most mid-size workloads is two to three months.

Beyond cost, self-hosting gives you data control (your prompts never leave your infrastructure), latency customisation (no shared inference queues), and model flexibility (swap between architectures as new ones drop — and new ones are dropping every week).

## The Tooling Landscape

Three main inference engines dominate self-hosting today:

### Ollama — Easiest for Development

[Ollama](https://ollama.com) is the fastest path from zero to running model. It wraps llama.cpp underneath, handles GPU detection, model downloading, and exposes a clean OpenAI-compatible API — all with a single command:

```bash
ollama run llama3.2:70b
```

For experimentation and single-GPU setups, Ollama is the obvious choice. It supports GGUF quantised models, which means you can run a 13B model on a single consumer GPU with minimal performance loss. The trade-off: it's not built for high-throughput production serving. No continuous batching, limited request queuing, and no multi-node support.

**Best for:** Prototyping, local development, personal use, teams sharing a single GPU workstation.

### vLLM — Production Serving at Scale

[vLLM](https://github.com/vllm-project/vllm) is the production workhorse. Its core innovation — **PagedAttention** — eliminates the memory fragmentation that plagues naive KV-cache management, enabling near-100% GPU utilisation. Combined with continuous batching, it dynamically packs incoming requests into the same forward pass, driving throughput far higher than batch-size-limited alternatives.

```python
# serve a model with vLLM's OpenAI-compatible server
vllm serve Qwen/Qwen3-72B-GPTQ-Int4 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
```

vLLM also supports speculative decoding, prefix caching, and multi-LoRA serving — features that matter when you're optimising for latency and cost per token under real traffic.

**Best for:** Production APIs, high-traffic chatbots, any workload where throughput and latency matter.

### TGI — HuggingFace's Offering

[Text Generation Inference](https://huggingface.co/docs/text-generation-inference) (TGI) is HuggingFace's own serving stack, tightly integrated with their model hub and the `transformers` ecosystem. It offers continuous batching, tensor parallelism, and watermarking, with first-class support for safetensors and custom kernels.

TGI shines when you're already deep in the HuggingFace workflow — fine-tuning with PEFT, logging with `transformers`, or deploying models that use custom architectures not yet supported by vLLM.

**Best for:** HuggingFace-native workflows, models with custom architectures, teams already on the HF platform.

## What Hardware Do You Actually Need?

The rise of 4-bit and 8-bit quantisation has made self-hosting surprisingly attainable. Here's what fits with today's hardware:

- **7B models** (Llama 3.2 8B, Gemma 3 7B): 14 GB VRAM full precision, 8 GB at 8-bit, 5 GB at 4-bit
- **13B models** (Llama 3.3 13B, Qwen 3 14B): 26 GB VRAM full, 14 GB at 8-bit, 8 GB at 4-bit
- **70B models** (Llama 4 70B, DeepSeek V3): 140 GB VRAM full, 72 GB at 8-bit, 40 GB at 4-bit
- **2.8T Kimi K3 MoE**: ~560 GB VRAM full, ~280 GB at 8-bit, ~160 GB at 4-bit

A **70B model in 4-bit** fits comfortably on **two RTX 4090s** (48 GB total VRAM). That's a $6,000 workstation — or a $0.80/hour rental on Vast.ai — that replaces $20,000+/month in API costs for a comparable workload.

Kimi K3 at 2.8T parameters is a different beast. Its Mixture-of-Experts architecture means only a fraction of parameters activate per token (~450B active), but the full model still requires significant hardware. Early deployments use 4× A100-80GB or 2× H100 nodes with aggressive quantisation.

### For African Engineers and Teams

Full self-hosting isn't always practical in African markets, where hardware import costs and electricity reliability are real constraints. The good news: there are excellent intermediate options:

- **[Vast.ai](https://vast.ai)** and **[RunPod](https://runpod.io)** — rent fractional GPUs by the hour. A single RTX 4090 costs ~$0.30–0.50/hour. Great for serving a 7B or 13B model to small user bases.
- **[Together AI](https://together.ai)** — managed inference at prices well below the frontier API tier, with support for open-weight models. A middle ground between raw API and full self-hosting.
- **[Lambda Labs](https://lambdalabs.com)** — bare-metal GPU rentals with predictable pricing, increasingly available through African cloud partners.

The playbook: start on Together AI or RunPod to validate demand, move to dedicated GPU instances once usage justifies it, then invest in on-premise hardware when you need data residency or hit the cost crossover point — typically at 1–2 million tokens per day.

## Getting Started

Your first self-hosted model can be running in ten minutes. Pick your tool:

```bash
# Ollama — two commands
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.2:8b

# vLLM via pip
pip install vllm
vllm serve Qwen/Qwen3-14B-GPTQ-Int4 --port 8000

# TGI with Docker
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:3.0 \
    --model-id Qwen/Qwen3-14B-GPTQ-Int4
```

Start with the 8B class, validate your latency and quality requirements, then scale up to 70B quantised models. By the time you outgrow consumer hardware, the breakeven math on datacenter GPUs is already in your favour.

The open-weight revolution is here. The only question left is whether you're renting the intelligence or running it yourself.
