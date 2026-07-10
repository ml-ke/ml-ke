---
title: "vLLM and High-Throughput LLM Serving: PagedAttention and Continuous Batching"
date: 2026-07-10 00:00:00 +0300
categories: [ML Ops, AI Engineering]
tags: [mlops, vllm, llm-serving, inference-optimization, openai-api]
image:
  path: /assets/img/cover-vllm-llm-serving.webp
  alt: Illustration showing the vLLM architecture with PagedAttention managing KV cache blocks
---

Serving large language models (LLMs) at scale is fundamentally different from serving classical ML models. An LLM inference request doesn't just produce a single output — it runs an autoregressive loop that generates tokens one at a time, with each step consuming GPU memory for the key-value (KV) cache. This memory overhead is the primary bottleneck.

Enter **vLLM**: the most popular open-source LLM serving engine, used by companies like OpenAI, Databricks, and Anthropic in production. It achieves state-of-the-art throughput through two key innovations: **PagedAttention** and **continuous batching**.

## The KV Cache Problem

When an LLM generates text, every attention layer computes keys and values for each token in the sequence. These are cached for reuse across generation steps — known as the **KV cache**. For a 70B parameter model, a single 4096-token sequence can consume over 40 GB of GPU memory just for the KV cache.

Traditional serving systems pre-allocate contiguous memory blocks for each request's KV cache. This leads to severe **memory fragmentation** — like a file system that allocates files in fixed-size blocks without paging. When a request finishes or a new one arrives, the freed memory is often in small, non-contiguous chunks that can't be reused.

## PagedAttention: Solving KV Cache Fragmentation

PagedAttention, introduced in the [vLLM paper](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023), takes inspiration from virtual memory in operating systems. Instead of allocating contiguous memory for each request's KV cache, it divides the cache into fixed-size **blocks** (pages) and maintains a **page table** mapping logical blocks to physical memory locations.

Here's how it works:

{% raw %}
```
Without PagedAttention (Contiguous):
|  Request A (full block)  |    Free    |  Request B (full block)  |
|    20 GB contiguous      |  10 GB     |     15 GB contiguous     |

With PagedAttention (Paged):
| A0 | B0 | C0 | A1 | Free | B1 | C1 | A2 | Free | Free |
| 4KB blocks of KV cache, non-contiguous per request
| Page Table: A → [0, 3, 7], B → [1, 5], C → [2, 6]
```
{% endraw %}

The benefits are dramatic:

- **Near-zero memory waste** from fragmentation
- **Shared prefixes**: when multiple requests share a prompt prefix (e.g., system prompt), PagedAttention can share those KV cache blocks across requests
- **Copy-on-write**: efficient handling of beam search and parallel sampling
- **Memory overselling**: you can accept more concurrent requests than fit in GPU memory, swapping cold blocks to CPU

## Continuous Batching: Never Leave the GPU Idle

Traditional request batching works like a bus — you wait for a full batch to arrive, run them all together, and return results. But LLM generation has highly variable completion times. A request generating 10 tokens finishes much faster than one generating 1000 tokens.

**Continuous batching** (also called in-flight batching or iteration-level scheduling) solves this by re-evaluating the batch at every decoding step. When a request finishes, it's immediately removed and a new request is added to the batch, keeping the GPU fully utilized.

{% raw %}
```python
# Simplified illustration of continuous batching
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size=64):
        self.running = []      # Requests currently being processed
        self.waiting = deque()  # Requests waiting to start
        self.max_batch_size = max_batch_size

    def step(self):
        """One iteration of the decoding loop."""
        # Remove finished requests
        self.running = [r for r in self.running if not r.finished]

        # Add waiting requests up to max batch size
        while len(self.running) < self.max_batch_size and self.waiting:
            self.running.append(self.waiting.popleft())

        if not self.running:
            return

        # Batch decode — run one step for all active requests
        inputs = torch.cat([r.last_token for r in self.running])
        outputs = self.model(inputs)

        # Distribute outputs to individual requests
        for i, req in enumerate(self.running):
            next_token = outputs[i].argmax(dim=-1)
            req.append_token(next_token)
```
{% endraw %}

## Setting Up vLLM

vLLM provides an OpenAI-compatible API server, making it drop-in replaceable with OpenAI's API for compatible clients.

### Installation

{% raw %}
```bash
pip install vllm
```
{% endraw %}

### Serving a Model

{% raw %}
```bash
# Serve Meta Llama 3.1 8B
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype auto
```

```bash
# With quantization (AWQ quantized model)
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-Chat-AWQ \
    --quantization awq \
    --max-model-len 4096
```
{% endraw %}

Key parameters explained:

| Parameter | Purpose | Recommended |
|-----------|---------|-------------|
| `--tensor-parallel-size` | Split model across N GPUs | Set to number of GPUs |
| `--max-model-len` | Maximum context window | Match model's max, or lower to save memory |
| `--gpu-memory-utilization` | Fraction of GPU memory for KV cache | 0.85–0.95 |
| `--dtype` | Model precision (auto/half/float16/bfloat16) | `auto` or `bfloat16` |
| `--quantization` | Quantization method (awq/gptq/squeezellm) | Use quantized models for large models |
| `--max-num-seqs` | Max concurrent sequences | Start at 256, tune up |
| `--enable-prefix-caching` | Cache shared prompt prefixes | Enable for chatbots with system prompts |

## Quantization: Fitting Big Models on Fewer GPUs

LLMs are typically trained in 16-bit (float16 or bfloat16), requiring 2 bytes per parameter. A 70B model at 16-bit takes ~140 GB of GPU memory — too large for a single A100 (80 GB). Quantization reduces precision while preserving quality.

### AWQ (Activation-aware Weight Quantization)

AWQ analyzes the model weights and identifies the 1% most "important" weights that disproportionately affect output quality. It keeps those in higher precision while aggressively quantizing the rest.

{% raw %}
```bash
# Serve an AWQ-quantized model
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/mixtral-instruct-awq \
    --quantization awq \
    --max-model-len 8192
```
{% endraw %}

### GPTQ (Post-Training Quantization)

GPTQ uses approximate second-order information to find optimal quantization points. It's slightly slower at serving than AWQ but achieves comparable quality.

### Memory Comparison

| Model | FP16 | AWQ 4-bit | GPTQ 4-bit | GGUF (CPU) |
|-------|------|-----------|------------|-------------|
| Llama 3.1 8B | 16 GB | 5 GB | 5 GB | 4.5 GB |
| Llama 3.1 70B | 140 GB | 40 GB | 40 GB | 35 GB |
| Mixtral 8x22B | 280 GB | 80 GB | 80 GB | 70 GB |

## OpenAI-Compatible API

Once vLLM is running, you can use the standard OpenAI Python client:

{% raw %}
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't auth by default
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention in three sentences."},
    ],
    temperature=0.7,
    max_tokens=512,
)

print(response.choices[0].message.content)
```
{% endraw %}

This API compatibility means any tool that works with OpenAI (LangChain, LlamaIndex, etc.) also works with vLLM — just change the `base_url`.

## Benchmarking vLLM Performance

Here's a quick benchmark setup using the built-in benchmarking scripts:

{% raw %}
```bash
# Clone vLLM for benchmark scripts
git clone https://github.com/vllm-project/vllm
cd vllm/benchmarks

# Benchmark throughput
python benchmark_throughput.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --backend vllm \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 1000 \
    --num-scheduler-steps 10
```
{% endraw %}

Typical results (on 1x A100 80 GB with Llama 3.1 8B):

| Config | Throughput (tokens/s) | Latency P50 | Latency P99 |
|--------|-----------------------|-------------|-------------|
| Batch=1, no continuous batching | ~120 | 120ms | 180ms |
| Batch=64, continuous batching | ~2,800 | 320ms | 850ms |
| Batch=256, continuous batching | ~3,500 | 950ms | 2.1s |

## Performance Tuning Tips

1. **Set `--max-model-len` lower than the model's maximum** if your use case doesn't need long contexts. Shorter context = smaller KV cache = more concurrent requests.

2. **Use `--gpu-memory-utilization 0.90`** leaving 10% for model weights, CUDA overhead, and scratch memory.

3. **Enable prefix caching** (`--enable-prefix-caching`) if your requests share a system prompt — vLLM will share KV cache blocks across all requests.

4. **Tune `--max-num-seqs`** — too low underutilizes the GPU, too high causes OOM. Start at 256 and increase until you hit memory limits.

5. **Use `bfloat16`** on Ampere and newer GPUs — it's more numerically stable than float16 and avoids the overflow issues that can cause NaN outputs.

6. **Consider tensor parallelism** for models that don't fit on one GPU — `--tensor-parallel-size 2` splits the model across 2 GPUs with minimal overhead (<5%).

## When vLLM Is and Isn't the Right Choice

**Use vLLM when:**
- You're serving decoder-only transformer models (Llama, Mistral, Qwen, etc.)
- You need high throughput with many concurrent users
- You want OpenAI API compatibility
- You have GPU resources (A100, H100, or high-end consumer GPUs)

**Consider alternatives when:**
- Your model isn't a standard transformer (use TensorRT-LLM or TGI)
- You need CPU inference (use llama.cpp with GGUF)
- You're running on very limited hardware (use Ollama with quantization)
- You need custom model parallelism beyond tensor parallelism (use DeepSpeed)

## Next Steps

vLLM handles the serving layer, but production requires orchestration, monitoring, and scaling. In the next post, we'll dive into **GPU optimization** — understanding CUDA memory management, parallelism strategies, and profiling tools to get the most out of your hardware.
