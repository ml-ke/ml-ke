---
layout: post
title: "MLOps for Constrained Environments: Deploying ML Where Resources Are Tight"
date: 2026-08-08
image:
  path: /assets/img/cover-series-practical-playbook.webp
  alt: cover series practical playbook
categories: [AI Engineering, ML Ops]
tags: [mlops, constrained-environments, model-serving, monitoring, deployment, africa-ml]
---

Most MLOps content assumes abundant infrastructure: unlimited GPU time, reliable 5G, uninterrupted power, and a team that can spin up Kubernetes clusters on demand. But what if your deployment target is a clinic in rural Kano, a mobile money agent in Kampala, or a logistics hub in Kinshasa?

This post covers the strategies, tools, and architectural patterns that keep ML systems running when resources are genuinely tight.

## The Constrained Environment Reality

Before choosing tools, let's be honest about what we're working with:

- **Unreliable power**: Grid outages lasting 4–12 hours are routine across much of sub-Saharan Africa. Cloud-dependent inference simply stops when the power does.
- **Expensive bandwidth**: At $0.50–$3.00/GB relative to a median daily income of ~$3–5, every megabyte of model download or prediction upload carries real cost.
- **Limited GPU access**: A single RTX 4090 costs 2–3 months' median salary in many African markets. Most teams share fractional cloud GPUs or run on CPU-only hardware.
- **High latency**: The round trip from Lagos to a US-based GPU cloud adds 200–400 ms before any inference even starts.

These are not edge cases to optimise for later — they are the primary design constraints. MLOps for these environments must be lean, fault-tolerant, and bandwidth-aware.

## Model Optimisation: Compress Before Deploy

A bloated model is debt you carry through every downstream pipeline. Optimisation must happen *before* deployment, baked into your CI/CD:

**Quantisation (INT8, INT4)** reduces model weights from 32-bit floats to 8- or 4-bit integers. A 7B parameter model at INT4 drops from ~14 GB to ~3.5 GB — the difference between requiring a cloud GPU cluster and fitting on a single edge device. Tools like `llama.cpp` and the `transformers` `quantization_config` make this a config flag during export.

**Pruning** removes redundant weights or attention heads. Magnitude pruning (zeroing weights below a threshold) is simple and effective for vision models. Structured pruning (removing entire layers or heads) delivers real speedups on CPU.

**Distillation** trains a compact student model to replicate a larger teacher. In our earlier post on [constrained environment AI](/posts/constrained-environment-ai/), we showed how a distilled 600M NLLB translation model retains 95%+ quality while running 2.2× faster on CPU.

The rule: if a model can't run inference in under 500 ms on target hardware at INT8, don't deploy it — go back to the optimisation stage.

## Batch Inference: Maximise Every GPU Cycle

When GPU time costs $0.40–$0.80/hour on [RunPod](https://runpod.io) or [Vast.ai](https://vast.ai), you want every second to count. Batch inference accumulates requests and processes them together, amortising the overhead of model loading and kernel launches.

```python
class BatchInferenceProcessor:
    def __init__(self, model, max_batch_size=32, max_wait=1.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait
        self.queue = []

    async def predict(self, inputs):
        self.queue.append(inputs)
        if len(self.queue) >= self.max_batch_size:
            return await self._flush()
        await asyncio.sleep(self.max_wait)
        return await self._flush()

    async def _flush(self):
        batch = self.queue[:]
        self.queue = []
        return self.model(batch)
```

For production, tools like [BentoML](https://www.bentoml.com/) and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) handle adaptive batching out of the box — configurable by `max_batch_size` and `batch_timeout_ms`.

## Async Serving: Queue-Based Architecture

Throughput spikes are inevitable — payday in mobile money, harvest season in agri-tech, election night in news. A synchronous API that scales vertically will either blow your GPU budget or crash.

The pattern: **Redis + Celery** (or [Ray](https://www.ray.io/) for Python-heavy stacks). Incoming requests go into a Redis queue; workers consume them at a steady rate, and results are retrieved via a polling endpoint or webhook.

```python
# tasks.py
from celery import Celery

app = Celery('ml_tasks', broker='redis://localhost:6379/0')

@app.task(rate_limit='10/m')
def predict_task(input_data: dict) -> dict:
    model = load_model()
    return model.predict(input_data).tolist()
```

This architecture handles 10× traffic spikes gracefully — requests queue up, inference stays predictable, and you only pay for the workers you need.

## Edge-First: Inference on Device, Sync When Connected

For many African use cases — crop disease diagnosis, offline health screening, point-of-sale fraud detection — the best latency is no network call at all.

The edge-first pattern:
1. A quantised model (TFLite or ONNX Runtime) runs entirely on-device
2. Predictions are stored locally in SQLite or Room
3. When connectivity returns (Wi-Fi at a hub, overnight), results sync to the cloud
4. Model updates are downloaded opportunistically

[We covered this architecture in detail](/posts/model-serving-101/) — but the MLOps implication is critical: your CI/CD must produce deployable edge artifacts (`.tflite`, `.onnx`) *and* cloud-serving artifacts from the same training pipeline. Tools like [MLflow](https://mlflow.org/) and [DVC](https://dvc.org/) for data versioning make this tractable — tag both artifacts with the same experiment ID for traceability.

## Monitoring Without Dashboards

A Grafana dashboard is useless when the power is out and your phone has the only screen that's on. For constrained environments, monitoring must be **alert-driven, not dashboard-driven**:

- **Lightweight logging**: Structured JSON logs shipped to [Axiom](https://axiom.co/) or a self-hosted Loki instance. Keep payloads small — log prediction IDs, not full inputs.
- **SMS/WhatsApp alerts**: Tools like [Twilio](https://www.twilio.com/) or [Africa's Talking](https://africastalking.com/) can send drift detection alerts or crash notifications as SMS. We showed how to build these alerting pipelines in [our monitoring post](/posts/ml-monitoring/).
- **Usage-based billing tracking**: Log every prediction with a user/tenant ID and model version. Export to a simple billing CSV — essential when you're charging per-prediction to cover GPU costs.

```python
# Minimal drift alert via WhatsApp
def check_and_alert(psi_data: float, threshold: float = 0.05):
    if psi_data > threshold:
        send_whatsapp(
            to="+254****5678",
            message=f"Data drift detected: PSI={psi_data:.3f} "
                    f"(threshold={threshold}). Model version {CURRENT_VERSION} "
                    f"may need retraining."
        )
```

## CI/CD for Constrained Envs

Standard CI/CD pipelines (GitHub Actions, GitLab CI) work, but with constrained-environment additions:

- **Model size checks**: Fail the pipeline if an artifact exceeds a threshold (e.g., 50 MB for edge models).
- **Automated quantisation**: For every merged PR, produce `model_int8.onnx` and `model_float32.onnx` as build artifacts.
- **Benchmark gates**: Run inference benchmarks on a CPU runner and fail if latency > 500 ms.

```yaml
# .github/workflows/model-ci.yml (excerpt)
- name: Check model size
  run: |
    SIZE=$(stat -c%s "models/classifier.onnx")
    if [ $SIZE -gt $((50 * 1024 * 1024)) ]; then
      echo "Model too large for edge deployment: $(($SIZE/1024/1024)) MB"
      exit 1
    fi
- name: Quantize model
  run: |
    python -c "
    from optimum.onnxruntime import ORTQuantizer
    quantizer = ORTQuantizer.from_pretrained('models/')
    quantizer.quantize()
    "
```

## Tying It Together: The Stack

For a team deploying ML in a constrained environment, here's the recommended stack:

- **Model serving**: [BentoML](https://www.bentoml.com/) / [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) — adaptive batching, Python-native, GPU-aware
- **Data versioning**: [DVC](https://dvc.org/) — works with S3/MinIO, no DB required
- **Experiment tracking**: [MLflow](https://mlflow.org/) — tracks model + artifact lineage
- **Queue**: Redis + Celery — handles spikes without over-provisioning
- **Edge inference**: ONNX Runtime / TFLite — cross-platform, quantised, tiny footprint
- **Alerting**: Africa's Talking / Twilio — SMS/WhatsApp, works without internet
- **Compute**: Vast.ai / RunPod / Together AI — fractional GPUs, pay-per-hour

This stack runs on a single $40/month VPS with a Redis instance, defers heavy compute to rented GPUs, and keeps the critical path (inference) running on-device. It won't win any cloud architecture awards — but it *will* keep working when the grid goes down.

## Building on What We've Covered

This post draws from three earlier pieces in the series. If you haven't read them:

- **[Model Serving 101](/posts/model-serving-101/)** (Jun 29) — from notebook to production API with FastAPI and Docker
- **[Monitoring ML Systems](/posts/ml-monitoring/)** (Jul 11) — drift detection, performance tracking, and alerting
- **[Constrained Environment AI](/posts/constrained-environment-ai/)** (Jun 28) — model compression and offline-first patterns

The constrained environment is not a limitation to work around — it's the design brief. Build for the worst connectivity, the cheapest hardware, and the most intermittent power, and your system will work everywhere.
