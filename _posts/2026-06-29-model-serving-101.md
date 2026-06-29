---
title: "Model Serving 101: From Jupyter Notebook to Production API"
date: 2026-06-29 00:00:00 +0300
categories: [ML Ops, AI Engineering]
tags: [mlops, model-serving, fastapi, docker, production]
image:
  path: /assets/img/cover-model-serving-101.webp
  alt: Diagram showing the journey from a Jupyter notebook to a production model serving API
---

## The Gap Between Training and Production

You've just finished training a model in Jupyter. The loss curves look great, validation accuracy is solid, and you're ready to "put it in production." But what does that actually mean?

A Jupyter notebook is not a service. It runs on your machine, holds the model in memory as a Python object, and offers no API, no concurrency handling, no health checks, and no isolation. To turn that trained model into something other systems can talk to, you need **model serving**.

This post walks through the complete journey: exporting a trained model, building a FastAPI serving endpoint, containerizing it with Docker, adding production essentials like health checks and metrics, and handling concurrent requests properly.

## Step 1: Exporting the Model

Before you can serve anything, you need to save the model in a format that can be loaded without the original training code. Training frameworks save models in framework-specific formats, but for serving you typically want an **intermediate representation** that decouples the model from the framework.

### PyTorch → TorchScript

{% raw %}
```python
import torch
import torchvision.models as models

# Load a trained model
model = models.resnet18(pretrained=True)
model.eval()

# Trace with a dummy input
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)

# Save the TorchScript model
traced_model.save("model.pt")
```
{% endraw %}

### TensorFlow/Keras → SavedModel

{% raw %}
```python
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("my_model.h5")

# Save as SavedModel (the standard TF serving format)
model.save("model/1/", save_format="tf")
```
{% endraw %}

### Framework-Agnostic Export with ONNX

ONNX lets you export from almost any framework and serve with a runtime like ONNX Runtime:

{% raw %}
```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
```
{% endraw %}

ONNX with dynamic batching (as shown above) allows the serving layer to batch inference requests for higher throughput.

## Step 2: Building the Serving API with FastAPI

FastAPI is the go-to choice for model serving in Python. It's async-first, fast, has automatic OpenAPI docs, and handles request validation via Pydantic.

Here's a complete FastAPI serving app for our exported model:

{% raw %}
```python
# app.py
import torch
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Model Serving API", version="1.0.0")

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("model.pt", map_location=device)
model.eval()

# ImageNet class labels (simplified)
LABELS = ["cat", "dog", "bird", "fish", ...]  # load from file

# Preprocessing pipeline
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    """Basic metrics endpoint."""
    return {
        "model": "resnet18",
        "device": str(device),
        "ready": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run inference on an uploaded image."""
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    return {
        "prediction": LABELS[pred_idx] if pred_idx < len(LABELS) else "unknown",
        "confidence": round(confidence, 4),
        "class_id": pred_idx,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
{% endraw %}

Key design decisions here:
- **Load the model at startup**, not on every request — this avoids paying the loading cost repeatedly.
- **Health and metrics endpoints** are separate from the prediction endpoint so monitoring systems can check the service without triggering inference.
- **Async handlers** let FastAPI handle many concurrent connections while the model runs in a thread pool (PyTorch ops release the GIL).

## Step 3: Containerizing with Docker

A Docker image makes your serving stack reproducible and deployable anywhere. Here's a multi-stage Dockerfile that keeps the image small:

{% raw %}
```dockerfile
# Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

# Create a non-root user
RUN useradd --uid 1000 --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
{% endraw %}

With a requirements.txt like:

{% raw %}
```
fastapi==0.110.0
uvicorn[standard]==0.27.0
torch==2.3.0
torchvision==0.18.0
pillow>=10.0.0
python-multipart>=0.0.6
```
{% endraw %}

Build and run:

{% raw %}
```bash
docker build -t model-server:latest .
docker run -p 8000:8000 --gpus all model-server:latest
```
{% endraw %}

## Step 4: Adding Production Readiness

### Concurrent Request Handling

By default, Uvicorn runs with one worker process. For production, use multiple workers:

{% raw %}
```bash
# Using Gunicorn with Uvicorn workers for process-level parallelism
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 app:app
```
{% endraw %}

The number of workers should generally be `2 × num_cores` for CPU inference. For GPU inference, one or two workers often suffice since the GPU serializes kernel execution anyway.

### Request Batching

For higher throughput, implement dynamic batching — collect requests over a short window and run them as a batch:

{% raw %}
```python
import asyncio
from collections import deque


class BatchInferenceHandler:
    def __init__(self, model, max_batch_size=32, max_wait=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait
        self.queue = deque()
        self._lock = asyncio.Lock()

    async def infer(self, input_tensor):
        future = asyncio.Future()
        async with self._lock:
            self.queue.append((input_tensor, future))
            if len(self.queue) >= self.max_batch_size:
                asyncio.create_task(self._process_batch())
        # Wait for the batch to be processed
        return await future

    async def _process_batch(self):
        async with self._lock:
            batch = list(self.queue)
            self.queue.clear()

        if not batch:
            return

        inputs = torch.cat([x[0] for x in batch])
        with torch.no_grad():
            outputs = self.model(inputs)

        for (_, future), output in zip(batch, outputs):
            future.set_result(output)
```
{% endraw %}

### Graceful Shutdown and Readiness Probes

For orchestrated environments (Kubernetes), your API should handle graceful shutdowns:

{% raw %}
```python
import signal
import sys

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down gracefully...")
    # Release GPU memory, close connections, etc.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```
{% endraw %}

## Sync vs Async Serving Patterns

| Pattern | When to Use | Throughput | Complexity |
|---------|------------|------------|------------|
| **Sync** | Low concurrency, CPU models | Low | Simple |
| **Async+ThreadPool** | GPU models, moderate concurrency | Medium | Moderate |
| **Async+Batching** | GPU models, high concurrency | High | Complex |
| **Async+Streaming** | LLMs, long-running inference | High | Advanced |

For most serving scenarios, **async with batching** hits the sweet spot — you get high GPU utilization and reasonable latency.

## Putting It All Together

The path from notebook to production API looks like this:

1. **Export** your model to TorchScript, ONNX, or SavedModel
2. **Build a FastAPI app** with `/health`, `/metrics`, and `/predict` endpoints
3. **Containerize** with a multi-stage Docker build
4. **Add concurrent handling** — gunicorn workers + optional dynamic batching
5. **Deploy** behind a reverse proxy (nginx, Traefik) or onto Kubernetes

In the next post, we'll take this foundation and scale it up with vLLM for serving large language models at high throughput.

## Quick Reference: Commands and Tools

```bash
# Build and run the Docker container
docker build -t model-server:latest .
docker run -p 8000:8000 --gpus all model-server:latest

# Test the API
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Run with multiple workers
gunicorn -k uvicorn.workers.UvicornWorker -w 4 app:app
```
