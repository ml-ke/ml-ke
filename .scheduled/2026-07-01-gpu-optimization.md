---
title: "GPU Optimization for ML Workloads: CUDA, Memory Management, and Parallelism"
date: 2026-07-01 00:00:00 +0300
categories: [ML Ops, AI Engineering]
tags: [mlops, gpu, cuda, inference-optimization, model-parallelism]
image:
  path: /assets/img/cover-gpu-optimization.webp
  alt: Diagram of GPU architecture showing streaming multiprocessors, memory hierarchy, and parallel execution
---

You've deployed your model and it works. But is it using your GPU efficiently? Most ML engineers treat the GPU as a black box — feed it tensors and hope for the best. The difference between naive usage and optimized usage can be **5–10x throughput** on the same hardware.

This post covers the CUDA fundamentals every ML engineer should know, practical memory management techniques, parallelism strategies for big models, and profiling tools to identify bottlenecks.

## CUDA Basics Every ML Engineer Should Know

### The Execution Model

A GPU has thousands of cores organized into **Streaming Multiprocessors (SMs)**. When you launch a CUDA kernel, groups of 32 threads called **warps** execute together in lockstep on a single SM. All threads in a warp must execute the same instruction — if your code has a branch that diverges threads, some threads are masked out and the warp executes both branches serially.

This matters for ML because:

- **Batched operations are naturally parallel** — matrix multiplications map perfectly to this execution model.
- **Small operations waste GPU power** — launching a kernel for a tiny tensor means the overhead of scheduling dominates the actual computation.
- **Branch divergence in custom kernels kills performance** — try to write warp-uniform code.

### The Memory Hierarchy

{% raw %}
```
Register File (per SM)       ~256 KB     ~0.5 cycle
    ↓
L1 Cache / Shared Memory    ~128 KB      ~5 cycles
    ↓
L2 Cache                    ~40 MB       ~50 cycles
    ↓
HBM (Global Memory)         ~80 GB       ~400 cycles
```
{% endraw %}

The key insight: **global memory access is 800x slower than register access**. Every time your model loads a weight or an activation from HBM, it stalls for hundreds of cycles. The entire optimization game in GPU computing is about minimizing global memory traffic.

### CUDA Streams and Concurrency

CUDA streams are sequences of operations that execute in order on the GPU. Operations from different streams can **overlap**:

{% raw %}
```python
import torch

# Create two streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# Enqueue operations on different streams
with torch.cuda.stream(stream1):
    # This matmul runs on stream1
    output1 = torch.matmul(A, B)

with torch.cuda.stream(stream2):
    # This matmul runs concurrently with the first
    output2 = torch.matmul(C, D)

# Synchronize
torch.cuda.synchronize()
```
{% endraw %}

This is useful when you have independent computations — for example, running two different model heads on the same encoder output.

## GPU Memory Management

### Understanding Where Memory Goes

When you load an LLM on a GPU, memory is consumed by:

1. **Model weights**: 2 bytes × number of parameters (in FP16)
2. **Optimizer states**: 8–16 bytes per parameter during training (Adam uses 2x states + momentum)
3. **Activations**: proportional to batch size × sequence length × hidden size
4. **KV cache**: 2 × layers × hidden_size × sequence_length × batch_size × 2 bytes
5. **CUDA context / miscellaneous overhead**: ~1–3 GB

For a 70B model serving a batch of 32 sequences with 4096 context:

{% raw %}
```
| Component       | Memory     | Running Total |
|-----------------|------------|---------------|
| Weights (FP16)  | ~140 GB    | 140 GB        |
| KV cache        | ~20 GB     | 160 GB        |
| CUDA context    | ~2 GB      | 162 GB        |
| Activations     | ~0.5 GB    | 162.5 GB      |
```
{% endraw %}

No single GPU has this much memory. That's where parallelism comes in.

### Unified Memory and Memory Pooling

PyTorch manages GPU memory through a **caching memory allocator** — it caches freed memory blocks for reuse instead of returning them to the OS. This is why `torch.cuda.empty_cache()` doesn't reduce `nvidia-smi` reported memory; the memory is cached, not freed.

{% raw %}
```python
# Check memory usage
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Clear cache (returns cache to pool, not to OS)
torch.cuda.empty_cache()

# Enable memory stats tracking
torch.cuda.memory_stats()
```
{% endraw %}

### Memory-Efficient Inference Techniques

**Flash Attention** is the most impactful optimization for transformer inference. It computes attention in tiles that fit in shared memory, never materializing the full N×N attention matrix:

{% raw %}
```python
# Without Flash Attention — O(n²) memory
attn = torch.matmul(Q, K.transpose(-2, -1))  # This is huge for long sequences
attn = torch.softmax(attn / sqrt_d, dim=-1)
output = torch.matmul(attn, V)

# With Flash Attention — O(n) memory
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, dropout_p=0.0, causal=True)
```
{% endraw %}

Flash Attention 2 and 3 have further improved this, making it possible to train and serve with context windows of 128K+ tokens.

**PagedAttention** (covered in the previous post) is another memory optimization specifically for the KV cache during serving.

## Parallelism Strategies

When a model is too large for one GPU — or when you want higher throughput — you need to distribute work across multiple GPUs. There are three fundamental strategies.

### Data Parallelism

Each GPU holds a full copy of the model but processes different data batches. Gradients are averaged across GPUs after each step.

{% raw %}
```text
GPU 0: [Model Copy] ← Batch A → [Gradient ↕]
GPU 1: [Model Copy] ← Batch B → [Gradient ↕] → AllReduce → Average → Update
GPU 2: [Model Copy] ← Batch C → [Gradient ↕]
GPU 3: [Model Copy] ← Batch D → [Gradient ↕]
```
{% endraw %}

**Best for**: Training when model fits on one GPU but you need larger effective batch sizes.

**Limitation**: Every GPU needs enough memory for the full model. Doesn't help if the model exceeds single-GPU memory.

### Tensor Parallelism

Each GPU holds a **slice of each layer** (e.g., half the attention heads). Operations use collective communication (all-reduce) after each layer.

{% raw %}
```python
# vLLM uses tensor parallelism internally
# Launch command
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```
{% endraw %}

{% raw %}
```text
GPU 0: Heads 0-15 | GPU 1: Heads 16-31 | GPU 2: Heads 32-47 | GPU 3: Heads 48-63
         ↑                          ↑
         └──────── AllReduce ────────┘
                  (after each layer)
```
{% endraw %}

**Best for**: Models that don't fit on a single GPU (70B+). Nearly linear scaling for large models.

**Communication**: Heavy — each layer requires an all-reduce. InfiniBand or NVLink strongly recommended.

### Pipeline Parallelism

Different layers of the model are placed on different GPUs. Data flows through the pipeline sequentially.

{% raw %}
```text
GPU 0: Layers 1-10 → GPU 1: Layers 11-20 → GPU 2: Layers 21-30 → GPU 3: Layers 31-40
         |                     |                      |                      |
       Batch 1              Batch 1                 Batch 1               Batch 1
                          Batch 2                  Batch 2                Batch 2
```
{% endraw %}

**Best for**: Very deep models where tensor parallelism alone isn't enough. Combined with tensor parallelism in practice.

**Limitation**: Pipeline bubbles (idle time between batches) reduce efficiency. Use many micro-batches to fill the pipeline.

### Choosing a Strategy

| Strategy | Inter-GPU Communication | Per-GPU Memory | Scaling Efficiency | Setup Complexity |
|----------|------------------------|----------------|--------------------|------------------|
| Data Parallel | Low (gradients only) | Full model copy | Near-linear | Low |
| Tensor Parallel | High (every layer) | 1/N of model | ~80-90% | Moderate |
| Pipeline Parallel | Low (between stages) | 1/N of layers | ~70-90% | Moderate |
| Combined (3D) | Very High | Minimal | ~70-80% | High |

For **inference**, tensor parallelism is the standard choice — vLLM, TensorRT-LLM, and TGI all use it. For **training large models**, you typically combine all three (3D parallelism) as done in training Llama 70B+.

## Profiling: Finding the Bottlenecks

Don't guess — profile. The two most important tools:

### NVIDIA SMI (Quick Health Check)

{% raw %}
```bash
# Watch GPU stats in real-time
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

# Show running processes
nvidia-smi pmon -s u
```
{% endraw %}

What to look for:
- **GPU-Util < 80%**: The GPU is idle too much — likely CPU bottleneck, data loading issue, or small batch size.
- **Memory-Util high, GPU-Util low**: Memory bandwidth bound — often caused by attention operations on long sequences.
- **Volatile GPU-Util fluctuating wildly**: Uneven workload distribution or I/O waits.

### PyTorch Profiler (Detailed Analysis)

{% raw %}
```python
import torch.profiler as profiler

def run_training_step():
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    with_stack=True,
) as p:
    for step in range(10):
        run_training_step()
        p.step()

# Export to Chrome Trace Viewer
p.export_chrome_trace("trace.json")
# View in browser: chrome://tracing → Load trace.json
```
{% endraw %}

Common patterns visible in traces:
- **Gaps between CUDA kernels**: Kernel launch overhead — consider kernel fusion or `torch.compile`.
- **Long data transfer**: Move data to GPU earlier, use pinned memory.
- **Sequential small kernels**: Use `torch.jit.script` or `torch.compile` to fuse operations.
- **CUDA kernels waiting on CPU**: Pre-fetch data with DataLoader's `prefetch_factor`.

### Using torch.compile for Free Speedups

PyTorch 2.0+ includes `torch.compile`, which uses TorchInductor to generate optimized CUDA kernels:

{% raw %}
```python
import torch

model = MyTransformerModel().cuda()
model = torch.compile(model, mode="reduce-overhead")

# First few calls compile the model
output = model(input_tensor)
# Subsequent calls use the compiled graph
```
{% endraw %}

With `mode="max-autotune"`, torch.compile tries hundreds of kernel configurations to find the fastest one for your specific model and GPU. Real-world speedups of 1.3–2× are common.

## Practical Recipe: Optimizing an Inference Pipeline

Here's a step-by-step recipe for optimizing any ML inference workload:

{% raw %}
```
1. [Baseline] Measure current throughput and latency
   → python -c "benchmark_model(model, input)"

2. [Dtype] Switch to FP16/BF16
   → model.half() or model.bfloat16()

3. [Fusion] Enable torch.compile with reduce-overhead
   → compiled_model = torch.compile(model)

4. [Batching] Tune batch size for your GPU
   → Double batch size until latency degrades >20%

5. [Quantization] Apply AWQ/GPTQ if memory-bound
   → Use vLLM or AutoAWQ

6. [Flash Attention] Replace attention implementation
   → pip install flash-attn

7. [Profile] Run the PyTorch profiler
   → Find remaining bottlenecks

8. [Parallelism] If still not enough, add GPUs
   → Tensor parallelism for inference
```
{% endraw %}

Each step typically gives 1.5–2× improvement. Combining them all can yield **10–50×** over the naive implementation.

## Summary

GPU optimization for ML isn't magic — it's understanding the memory hierarchy, choosing the right parallelism strategy, and using profiling tools to find bottlenecks. The biggest wins come from:

1. **Minimizing memory traffic** — Flash Attention, PagedAttention, kernel fusion
2. **Maximizing GPU utilization** — continuous batching, right-sized batches
3. **Distributing efficiently** — tensor parallel for inference, 3D parallelism for training
4. **Profiling before optimizing** — NVIDIA SMI for quick checks, PyTorch Profiler for deep dives

In the next post, we'll take this hardware knowledge and apply it to **deploying models on Kubernetes** with Kserve for auto-scaling and production orchestration.
