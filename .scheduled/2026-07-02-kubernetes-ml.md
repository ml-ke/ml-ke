---
title: "Kubernetes for ML: Deploying Models at Scale with Kserve"
date: 2026-07-02 00:00:00 +0300
categories: [ML Ops, AI Engineering]
tags: [mlops, kubernetes, kserve, model-deployment, auto-scaling]
image:
  path: /assets/img/cover-kubernetes-ml.webp
  alt: Kubernetes architecture diagram showing model inference pods being managed by Kserve with auto-scaling and GPU scheduling
---

You've built a FastAPI server. It's containerized. You can run it on a single machine. But what happens when you need to serve 1000 concurrent users? What about zero-downtime model updates? Or rolling back a bad model? Or sharing GPU resources across multiple teams?

This is where **Kubernetes** comes in — and specifically **Kserve**, the open-source model serving platform built on Kubernetes that powers production ML at companies like Bloomberg, Uber, and Spotify.

## Why Kubernetes for ML?

Kubernetes provides the orchestration layer that turns a collection of GPU machines into a unified serving platform:

- **Auto-scaling**: Scale from 0 to 100 replicas based on request load
- **GPU scheduling**: Assign GPUs to pods that need them, share GPUs between lightweight models
- **Rolling updates**: Update models without dropping requests
- **Resource isolation**: Separate models into namespaces with guaranteed resources
- **Observability**: Prometheus metrics, structured logging, distributed tracing

### The Minimal Service Architecture

{% raw %}
```text
                     ┌─────────────┐
                     │  Ingress    │  (Traefik / nginx / Istio)
                     │  Controller │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  Kserve     │  (Inference CRDs + routing)
                     │  Controller │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼────┐  ┌────▼────┐  ┌────▼────┐
        │ Model A  │  │ Model B │  │ Model C │  (Inference pods)
        │ v1.2.3   │  │ v2.0.0  │  │ v1.0.0  │
        │ GPU pod  │  │ GPU pod │  │ CPU pod │
        └──────────┘  └─────────┘  └─────────┘
```
{% endraw %}

## Setting up Kserve

### Prerequisites

A Kubernetes cluster with GPU nodes. For local development, we use kind (Kubernetes in Docker):

{% raw %}
```bash
# Create a GPU-capable kind cluster
cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
    extraMounts:
      - hostPath: /var/run/nvidia
        containerPath: /var/run/nvidia
EOF

# Install NVIDIA GPU operator
helm install gpu-operator nvidia/gpu-operator \
    --set driver.enabled=false  # if drivers already installed on host
```
{% endraw %}

### Install Kserve

{% raw %}
```bash
# Add Kserve helm repo
helm repo add kserve https://kserve.github.io/helm-charts
helm repo update

# Install Kserve with Istio ingress
helm install kserve kserve/kserve \
    --namespace kserve --create-namespace \
    --set ingress.enabled=true \
    --set ingress.ingressGateway=istio-ingressgateway/istio-ingressgateway

# Verify installation
kubectl get pods -n kserve
# Should see: kserve-controller-manager-xxx   Running
```
{% endraw %}

## Deploying Your First Model

Kserve introduces the `InferenceService` custom resource — you define your model in YAML, and Kserve handles everything else.

### Serving a Scikit-learn Model

{% raw %}
```yaml
# sklearn-model.yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "iris-classifier"
spec:
  predictor:
    sklearn:
      storageUri: "gs://my-bucket/iris-model/"
      resources:
        requests:
          cpu: "500m"
          memory: "512Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
```
{% endraw %}

### Serving a PyTorch Model with GPU

{% raw %}
```yaml
# pytorch-gpu-model.yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "resnet-classifier"
spec:
  predictor:
    pytorch:
      storageUri: "s3://ml-models/resnet18/model.pt"
      resources:
        requests:
          cpu: "2"
          memory: "8Gi"
          nvidia.com/gpu: "1"
        limits:
          cpu: "4"
          memory: "16Gi"
          nvidia.com/gpu: "1"
    minReplicas: 0
    maxReplicas: 5
```
{% endraw %}

Key fields:

- **`storageUri`**: Where the model artifact lives (GCS, S3, Azure Blob, PVC, or local)
- **`resources.requests.nvidia.com/gpu`**: Request 1 GPU — Kserve schedules this pod on a GPU node
- **`minReplicas` / `maxReplicas`**: Horizontal Pod Autoscaler configuration
- **`minReplicas: 0`**: Scale to zero when idle (saves GPU costs)

### Deploying a vLLM-based LLM

For LLMs, you typically use a custom container image with vLLM:

{% raw %}
```yaml
# llm-inference.yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "llama3-8b"
spec:
  predictor:
    containers:
      - name: kserve-container
        image: vllm/vllm-openai:latest
        args:
          - "--model"
          - "meta-llama/Meta-Llama-3.1-8B-Instruct"
          - "--tensor-parallel-size"
          - "1"
          - "--max-model-len"
          - "8192"
          - "--gpu-memory-utilization"
          - "0.90"
        ports:
          - containerPort: 8000
            protocol: TCP
        env:
          - name: HUGGING_FACE_HUB_TOKEN
            valueFrom:
              secretKeyRef:
                name: hf-token
                key: token
        resources:
          requests:
            nvidia.com/gpu: "1"
            memory: "32Gi"
          limits:
            nvidia.com/gpu: "1"
            memory: "64Gi"
    minReplicas: 0
    maxReplicas: 3
```
{% endraw %}

Apply the configuration:

{% raw %}
```bash
kubectl apply -f llm-inference.yaml

# Check the status
kubectl get inferenceservice llama3-8b

# NAME         URL                                                  READY
# llama3-8b    http://llama3-8b.kserve.example.com/v1/chat/completions   True
```
{% endraw %}

## Auto-Scaling: From 0 to 100 and Back

Kserve integrates with Kubernetes' **Horizontal Pod Autoscaler (HPA)** and can also use **Keda** for event-driven scaling.

### Concurrency-Based Scaling

The default scaling metric is CPU, but for inference you typically want **concurrency-based scaling**:

{% raw %}
```yaml
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 10
    scaleMetric: concurrency
    scaleTarget: 5  # Target 5 concurrent requests per pod
```
{% endraw %}

When requests spike beyond 5 per pod, Kserve spins up new pods. When traffic drops, it scales back down — even to zero, freeing GPU resources.

### Scale-from-Zero Latency

Scaling from zero takes ~30–60 seconds (pulling the container image and loading the model). For latency-sensitive applications:

- **Keep a minimum running replica** (`minReplicas: 1`)
- **Pre-pull images** on all nodes using DaemonSets
- **Use model caching** with a shared PVC or sidecar that pre-warms

## GPU Scheduling

Kubernetes manages GPUs through the **device plugin framework**. The NVIDIA device plugin exposes GPUs as schedulable resources:

{% raw %}
```bash
# Check GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu

# Check which pods are using GPUs
kubectl get pods -o custom-columns=NAME:.metadata.name,GPU_REQUEST:.spec.containers[*].resources.limits.nvidia\\.com/gpu
```
{% endraw %}

### GPU Sharing Strategies

| Strategy | Method | Isolation | Use Case |
|----------|--------|-----------|----------|
| **Exclusive GPU** | `nvidia.com/gpu: 1` | Full | Large models, critical services |
| **MIG (A100/H100)** | `nvidia.com/mig-*.slices` | Hardware | Multi-tenant, guaranteed perf |
| **Time-slicing** | Config in device plugin | Software | Dev/test, non-critical workloads |
| **MPS** | CUDA MPS daemon | Memory-based | Small models, best-effort |

For most production use, **exclusive GPU assignment** is simplest and safest. MIG is worth the complexity for truly multi-tenant clusters.

## Rolling Updates and Canary Deployments

Kserve supports **progressive delivery** natively. To deploy a new model version safely:

{% raw %}
```yaml
# canary-deploy.yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "my-model"
spec:
  predictor:
    canary:
      storageUri: "s3://models/my-model-v2/"
      trafficPercent: 10  # Send 10% of requests to new version
    resources:
      nvidia.com/gpu: 1
```
{% endraw %}

This creates a canary deployment receiving 10% of traffic. Once you confirm it's working:

{% raw %}
```bash
# Promote canary to 100%
kubectl patch inferenceservice my-model \
    --type='merge' \
    -p='{"spec":{"predictor":{"canaryTrafficPercent":100}}}'

# Or rollback by deleting the canary spec
kubectl patch inferenceservice my-model \
    --type='json' \
    -p='[{"op": "remove", "path": "/spec/predictor/canary"}]'
```
{% endraw %}

## Monitoring with Prometheus

Kserve exposes Prometheus metrics automatically. Each inference pod emits:

- `request_count`: Total inference requests
- `request_latency`: Latency buckets (50ms, 100ms, 250ms, 500ms, 1s, 5s)
- `response_count`: Responses by status code
- `inference_service_running`: Whether the service is healthy

### Sample Prometheus Query

{% raw %}
```promql
# P99 latency over the last 5 minutes
histogram_quantile(0.99,
  sum(rate(request_latency_bucket{service_name="llama3-8b"}[5m])) by (le)
)

# Request rate (per second)
rate(request_count{service_name="llama3-8b"}[5m])

# GPU utilization across all model pods
avg(nvidia_gpu_utilization{kubernetes_pod_name=~"llama3-8b-.*"})
```
{% endraw %}

### Setting up a Dashboard

{% raw %}
```yaml
# grafana-dashboard.yaml (excerpt)
panels:
  - title: "Request Rate"
    type: timeseries
    targets:
      - expr: rate(request_count{namespace="kserve"}[5m])
  - title: "P99 Latency"
    type: gauge
    targets:
      - expr: histogram_quantile(0.99, rate(request_latency_bucket[5m]))
  - title: "GPU Memory"
    type: timeseries
    targets:
      - expr: nvidia_memory_used_bytes / nvidia_memory_total_bytes
```
{% endraw %}

## Production Checklist

Deploying ML on Kubernetes is more than writing YAML. Here's what you need for production:

### Security
- [ ] Pull model images from a private registry (ECR, GCR, Harbor)
- [ ] Store secrets (HF tokens, S3 keys) in Kubernetes Secrets or Vault
- [ ] Run containers as non-root with read-only root filesystem
- [ ] Network policies restricting pod-to-pod communication

### Reliability
- [ ] Pod disruption budgets (PDB) to prevent all replicas from being evicted
- [ ] Resource requests = resource limits (to avoid noisy neighbors)
- [ ] Liveness and readiness probes configured
- [ ] Graceful shutdown handling (SIGTERM → finish current requests → exit)

### Cost
- [ ] Scale-to-zero for non-production environments
- [ ] Spot instances for batch inference and training
- [ ] GPU node auto-scaling (Cluster Autoscaler or Karpenter)

### Observability
- [ ] Prometheus metrics exported from every inference pod
- [ ] Grafana dashboards for latency, throughput, and GPU utilization
- [ ] Structured JSON logging (not ad-hoc print statements)
- [ ] Distributed tracing for multi-model pipelines

## Common Pitfalls

### Cold Starts Are Real

When scaling from zero, loading a large model (70B parameters) can take 2–5 minutes. Mitigations:
- Pre-warm models on dedicated nodes
- Use PVC-backed model caches
- Accept that scale-from-zero adds latency and plan SLAs accordingly

### GPU Fragmentation

If you have 4 models each requesting 1 GPU but you have 4 GPUs on 1 node, a single node failure takes down all models. Use **pod anti-affinity** to spread replicas across nodes:

{% raw %}
```yaml
spec:
  predictor:
    affinity:
      podAntiAffinity:
        preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: my-model
              topologyKey: "kubernetes.io/hostname"
```
{% endraw %}

### OOM Kills Without Graceful Degradation

If a model's memory grows beyond its limits (e.g., KV cache with very long sequences), Kubernetes will **kill the pod** without finishing requests. Set memory limits slightly above your expected usage and add admission control for request sizes.

## Summary

Kubernetes + Kserve gives you a production-grade model serving platform that scales, self-heals, and handles multi-model deployments. The key concepts are:

1. **InferenceService** CRD defines the model, resources, and scaling policy
2. **Scale-to-zero** saves GPU costs when models aren't in use
3. **Canary deployments** enable safe rollouts of new model versions
4. **Prometheus metrics** give visibility into every model's performance
5. **GPU scheduling** ensures GPUs are used efficiently

In the next post, we'll look at **ML CI/CD** — automating the pipeline from training → registration → deployment so you never manually deploy a model again.
