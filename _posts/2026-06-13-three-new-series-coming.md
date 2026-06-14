---
title: "Three New Series: Building AI Agents, African AI, and AI Infrastructure"
date: 2026-06-13 00:00:00 +0300
categories: [AI Engineering, Machine Learning, AI in Africa]
tags: [series, ai-agents, african-ai, mlops, infrastructure, roadmap]
image:
  path: /assets/img/cover-three-new-series.webp
  alt: Three interconnected pillars representing the three new series
---

We've spent the last two weeks diving deep into AI security — from {% post_url 2026-06-01-mlsecops-pipeline-security %} to {% post_url 2026-06-12-ml-secrets-management %} — and the response has been incredible. The MLsecOps series showed us how fragile our pipelines really are, and that knowledge matters.

But security isn't the only story. It's time to **build.**

Starting **June 15**, ml.co.ke launches three concurrent series that span the full AI stack — from writing your first agent loop to deploying at scale in African cloud environments. These series will run in parallel, cross-referencing each other as they unfold.

Here's what's coming.

---

## Series 1: Building AI Agents from Scratch 🧠

**Starts: June 15 | Frequency: Weekly (Mon/Wed/Fri)**

Agents are this year's defining paradigm shift. Every major AI lab is betting on agentic systems — and for good reason. LLMs alone can't act. Agents can.

This series covers:

- **Agent fundamentals** — ReAct loops, tool calling, memory systems
- **Multi-agent architectures** — Orchestration, delegation, supervisor patterns
- **Observability & debugging** — Tracing agent decisions, handling failure modes
- **Agent security** — Prompt injection in agent loops, tool access control
- **Production deployment** — State management, rate limiting, fallback chains

Whether you're building a simple chatbot with function calling or a multi-agent research system, this series gives you the architecture and code.

**Prerequisites:** Python, basic LLM API experience. We'll build everything from first principles.

---

## Series 2: AI in the African Ecosystem 🌍

**Starts: June 17 | Frequency: Bi-weekly (Tue/Thu)**

AI doesn't exist in a vacuum. The tools, datasets, and infrastructure that work in San Francisco or London don't always translate. Africa has unique realities — mobile-first users, intermittent connectivity, low-resource languages, local regulatory frameworks — that demand different approaches.

This series covers:

- **Swahili & low-resource NLP** — Tokenization, dataset building, fine-tuning for African languages
- **Mobile-first AI** — On-device inference, model compression, offline-capable agents
- **African AI communities** — The labs, startups, and research groups shaping the continent
- **AI for agriculture & healthcare** — Real-world deployment case studies from East Africa
- **Local data sovereignty** — Regulatory landscapes, data localization, compliant infrastructure

The goal isn't just commentary — it's actionable guidance for building AI that works in African contexts.

---

## Series 3: AI Infrastructure & MLOps ⚙️

**Starts: June 19 | Frequency: Weekly (Tue/Thu/Sat)**

Agents need infrastructure. Models need deployment pipelines. Data needs versioning. This series tackles the engineering backbone:

- **Model serving 101** — vLLM, TGI, Triton Inference Server, ONNX Runtime
- **GPU optimization** — CUDA graphs, tensor parallelism, quantization, PagedAttention
- **Kubernetes for ML** — Kubeflow, K8s operators for model serving, autoscaling
- **CI/CD for ML pipelines** — DVC, MLflow, automated retraining, A/B testing in production
- **Monitoring & observability** — Drift detection, alerting, LLM eval pipelines
- **Secrets & security** — Building on our earlier {% post_url 2026-06-12-ml-secrets-management %} post into production-grade vault patterns

This series pairs tightly with the Agents series — you'll see the infrastructure patterns behind the agent architectures.

---

## How the Three Series Connect

Think of them as a **stack:**

| Layer | Series | Focus |
|:------|:--------|:------|
| Applications | Building AI Agents | What you build — agent logic, tool use, multi-agent systems |
| Context | AI in the African Ecosystem | Where and for whom you build — local languages, mobile, regulation |
| Foundation | AI Infrastructure & MLOps | How you run it — serving, scaling, monitoring, security |

You don't need all three to benefit — jump in at any layer. But if you follow all three, you'll see the same patterns appear at every level. An agent's tool-use loop (Series 1) mirrors the request routing patterns you'll learn in infrastructure (Series 3). The mobile-first constraints from Series 2 inform the quantization strategies in Series 3.

**New content will drop every weekday**, alternating between series so you always have fresh material.

---

## The Roadmap

| Date | Series | Topic |
|:-----|:-------|:------|
| June 15 (Mon) | Agents | Setting Up Your AI Dev Environment |
| June 16 (Tue) | Infrastructure | Model Serving 101: From Notebook to API |
| June 17 (Wed) | Agents | The ReAct Loop: Building Your First Tool-Calling Agent |
| June 18 (Thu) | Africa | Swahili NLP: Tokenization for Low-Resource Languages |
| June 19 (Fri) | Infrastructure | GPU Optimization for Inference |
| June 22 (Mon) | Agents | Memory Systems: Short-Term, Long-Term, and Episodic Memory |
| June 23 (Tue) | Africa | Mobile-First AI: On-Device Inference with TFLite & ONNX |
| June 24 (Wed) | Infrastructure | Kubernetes for ML: Deploying with Kubeflow |
| June 25 (Thu) | Agents | Multi-Agent Architectures: Orchestration Patterns |
| June 26 (Fri) | Africa | AI for Agriculture: Crop Disease Detection in East Africa |
| June 29 (Mon) | Agents | Agent Observability: Tracing, Logging, Debugging |
| June 30 (Tue) | Infrastructure | CI/CD for ML: Automating the Pipeline |
| July 1 (Wed) | Africa | Data Sovereignty: Building Compliant AI Infrastructure |
| July 2 (Thu) | Agents | Agent Security: Defending Against Prompt Injection at Runtime |
| July 3 (Fri) | Infrastructure | Monitoring & Drift Detection in Production LLM Systems |

We'll update this roadmap as we go, and reader feedback shapes the course. **If there's a topic you want covered, comment or reach out.**

---

## What About MLsecOps?

The MLsecOps series isn't ending — it's evolving. Security remains a thread woven through all three new series. You'll see:

- Agent-specific security patterns in the Agents series
- Data sovereignty and compliance in the African AI series
- Secrets management, pipeline hardening, and runtime security in Infrastructure

Our earlier posts — from {% post_url 2026-06-01-prompt-injection-llm-security %} to {% post_url 2026-06-11-multimodal-attacks %} — still serve as the foundation. We'll reference them, build on them, and continue publishing dedicated security deep-dives alongside the new series.

---

## Next Up

**Setting Up Your AI Dev Environment** — The Agents series kicks off June 15 with a complete walkthrough: Python environment setup, LLM provider configuration, tool-calling scaffolding, and a minimal end-to-end agent you can run in under 30 minutes.

See you Monday. 🚀
