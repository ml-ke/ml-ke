---
title: "Building and Deploying AI in Constrained Environments: Low-Bandwidth and Edge Solutions"
date: 2026-06-28 00:00:00 +0300
categories: [Machine Learning, AI in Africa]
tags: [africa, constrained-environments, edge-ai, model-compression, offline-ml]
image:
  path: /assets/img/cover-constrained-environment-ai.webp
  alt: Edge device running AI inference with offline-first architecture diagram
---

## The Reality of Constrained Environments

Throughout this series, we've touched on the constraints that define AI deployment in Africa: intermittent electricity, slow internet, expensive data, and budget hardware. This final post brings it all together with a practical framework for building AI systems that work *despite* these constraints — not in a hypothetical "we'll handle edge cases later" way, but as a first-class design requirement.

Let's be concrete about the environment we're designing for:

| Constraint | Typical Value | Impact on AI |
|------------|---------------|--------------|
| Internet bandwidth | 0.5-8 Mbps (shared) | Large model downloads take minutes to hours |
| Data cost | $0.50-$3.00/GB relative to $0.15 median hourly wage | Every MB counts |
| Internet reliability | 60-85% uptime | Offline operation is mandatory |
| Electricity | 12-20 hours/day with unannounced outages | Cloud-dependent pipelines fail unpredictably |
| Device RAM | 2-4 GB | Large models cause out-of-memory crashes |
| Device storage | 16-64 GB total | Model files compete with photos, apps, and OS |

The mindset shift: **constrained environments are not a degradation scenario**. They are the *primary* design target. If your system works perfectly on a Pixel 9 with 5G and fails on a Tecno Spark with 3G, it simply doesn't work for most of your users.

## Model Compression Techniques

### 1. Pruning: Removing What You Don't Need

Pruning removes redundant weights from a neural network. After training, many weights are near zero and contribute little to the output. Removing them reduces model size with minimal accuracy loss.

```python
import tensorflow_model_optimization as tfmot

# Apply pruning during training
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=0.80,
        begin_step=0,
        end_step=1000,
    )
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model, **pruning_params
)

pruned_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train with pruning callback
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir="./logs"),
]

pruned_model.fit(
    train_dataset,
    epochs=10,
    callbacks=callbacks,
)

# Strip pruning wrappers for deployment
stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_pruned = converter.convert()

# Result: Model size reduced from 12.3 MB to 4.1 MB (67% reduction)
# Accuracy drop: 1.2% (from 93.4% to 92.2%)
```

**Real-world tip**: Magnitude pruning works well for computer vision models. For transformer-based NLP models, consider movement pruning (faster inference) or structured pruning (removes entire attention heads for better hardware utilization).

### 2. Quantization: Going from Float to Integer

We covered this in the mobile AI post, but it's worth revisiting for the broader constrained-environment context. Quantization converts model weights from 32-bit floats to lower-precision formats:

| Format | Bits per Weight | Size Factor | Speedup | Hardware Required |
|--------|----------------|-------------|---------|-------------------|
| Float32 | 32 | 1x | 1x | Any CPU |
| Float16 | 16 | 0.5x | 1.5-2x | GPU with FP16 support |
| Int8 | 8 | 0.25x | 2-4x | DSP, NPU, or NNAPI |
| Int4 | 4 | 0.125x | 3-6x | Specialized hardware |
| Binary | 1 | 0.031x | 8-16x | Specialized hardware |

For most African deployments, **Int8 full-integer quantization** is the sweet spot: 4x smaller, 2-4x faster, and minimal accuracy loss with proper calibration.

### 3. Knowledge Distillation: Training a Student from a Teacher

Distillation trains a small "student" model to mimic a large "teacher" model. The teacher can be a massive model trained in the cloud; the student runs on-device.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        # Soft loss: match teacher's soft probabilities
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # Hard loss: match ground truth labels
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# Example: Distill a 1.5B parameter NLLB model to a 250M student
teacher = load_teacher_model("facebook/nllb-200-distilled-1.3B")
student = load_student_model("facebook/nllb-200-distilled-600M")

# Train with distillation
criterion = DistillationLoss(temperature=4.0, alpha=0.7)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)

for batch in dataloader:
    with torch.no_grad():
        teacher_logits = teacher(batch["input_ids"])
    student_logits = student(batch["input_ids"])
    loss = criterion(student_logits, teacher_logits, batch["labels"])
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Result**: A distilled 600M NLLB model retains 95%+ of the 1.3B teacher's translation quality while being 2.2x faster on CPU and fitting in 1/4 the RAM.

## ONNX Runtime: Cross-Platform Inference

TensorFlow Lite is great for Android, but what if your deployment targets are mixed? ONNX Runtime provides a unified inference engine across platforms:

```python
import onnxruntime as ort

# Load an ONNX model
session = ort.InferenceSession("model.onnx", providers=[
    "CPUExecutionProvider",
    # Fall back to different providers depending on hardware
])

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run(
    [output_name],
    {input_name: input_data}
)
```

ONNX Runtime works on:
- **Android** (via Java/Kotlin bindings)
- **iOS** (Swift/Objective-C)
- **Linux ARM** (Raspberry Pi, BeagleBone)
- **Windows/Linux x86**
- **WebAssembly** (in-browser inference)

For African deployments, the key advantage is that you train once and deploy everywhere — no need to maintain separate code paths for Android, iOS, and web.

## Offline-First Architecture Patterns

### Pattern 1: Sync-When-Available

This is the most common pattern for African mobile apps with ML:

```
[User Action] → [On-device ML] → [Store result locally]
                                    ↓
                     [When internet available]
                                    ↓
                     [Sync results to cloud]
                     [Download new model version]
```

```kotlin
class OfflineFirstMLService(context: Context) {
    private val localDB = Room.databaseBuilder(
        context, AppDatabase::class.java, "ml-results"
    ).build()
    
    private val modelManager = ModelManager(context)
    
    fun processAndSync(image: Bitmap) {
        // 1. Run inference locally immediately
        val result = runOnDeviceML(image)
        
        // 2. Store locally
        localDB.resultDao().insert(
            PredictionResult(image_hash, result.label, result.confidence)
        )
        
        // 3. Try to sync (non-blocking)
        CoroutineScope(Dispatchers.IO).launch {
            if (isOnline()) {
                syncResults()
                checkForModelUpdate()
            }
        }
    }
    
    private suspend fun syncResults() {
        val pending = localDB.resultDao().getUnsynced()
        if (pending.isNotEmpty()) {
            apiService.syncPredictions(pending)
            localDB.resultDao().markSynced(pending.map { it.id })
        }
    }
}
```

### Pattern 2: Progressive Degradation

When constraints become extreme, the system degrades gracefully rather than failing entirely:

```python
class AdaptiveInferenceEngine:
    def __init__(self):
        self.models = {
            "full": self.load_model("high_quality_model.tflite"),
            "medium": self.load_model("compressed_model.tflite"),
            "light": self.load_model("lightweight_model.tflite"),
            "rule_based": RuleBasedFallback(),
        }
    
    def predict(self, input_data, battery_level, available_ram, priority):
        # Determine which model to use based on device state
        if battery_level < 15 and priority == "background":
            return self.models["rule_based"].predict(input_data)
        
        if available_ram < 200 * 1024 * 1024:  # < 200 MB
            return self.models["light"].predict(input_data)
        
        if battery_level < 30:
            return self.models["medium"].predict(input_data)
        
        return self.models["full"].predict(input_data)
```

### Pattern 3: Progressive Web Apps (PWAs) with ML

For users who can't or won't install an app, PWAs provide app-like experiences through the browser. With WebAssembly-based ML runtimes (TensorFlow.js, ONNX Runtime Web), inference runs in-browser:

```javascript
import * as ort from 'onnxruntime-web';

async function runCropDiseaseInference(imageElement) {
    // Create session
    const session = await ort.InferenceSession.create(
        './models/maize_disease_int8.onnx'
    );
    
    // Preprocess image
    const tensor = preprocessImage(imageElement);
    
    // Run inference
    const results = await session.run({ 'input': tensor });
    
    // Display result
    const probs = results['output'].data;
    const classes = ['Healthy', 'MLND', 'Fall Armyworm', 'Deficiency'];
    const maxIdx = probs.indexOf(Math.max(...probs));
    
    return {
        disease: classes[maxIdx],
        confidence: probs[maxIdx],
    };
}
```

The PWA can be installed on the home screen, works offline after the first load via a service worker, and auto-updates when connectivity returns.

## Resilient Data Pipelines

### Batched Uploads

Don't upload one prediction at a time. Batch them:

```python
class BatchUploader:
    def __init__(self, max_batch_size=50, max_wait_seconds=300):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self.timer = None
    
    def add_prediction(self, prediction):
        self.queue.append(prediction)
        if len(self.queue) >= self.max_batch_size:
            self.flush()
        elif self.timer is None:
            self.timer = threading.Timer(self.max_wait_seconds, self.flush)
            self.timer.start()
    
    def flush(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        if not self.queue:
            return
        
        batch = self.queue[:]
        self.queue = []
        
        try:
            api_client.upload_batch(batch)
        except (ConnectionError, TimeoutError):
            # Re-queue with exponential backoff
            self.queue.extend(batch)
            time.sleep(min(2 ** self.retry_count, 60))
            self.retry_count += 1
```

### Compression Before Upload

If you must upload data (e.g., field photos for model improvement), compress aggressively:

```python
def compress_for_upload(image, max_size_kb=100):
    """Compress image to under max_size_kb"""
    quality = 85
    while quality > 10:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        if buffer.tell() / 1024 <= max_size_kb:
            return buffer.getvalue()
        quality -= 10
    
    # Last resort: resize
    scale = 0.5
    while scale > 0.1:
        w, h = int(image.width * scale), int(image.height * scale)
        resized = image.resize((w, h), Image.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=60)
        if buffer.tell() / 1024 <= max_size_kb:
            return buffer.getvalue()
        scale -= 0.1
    
    return None  # Cannot compress sufficiently
```

## Real-World Case Study: mDROID

[mDROID](https://mdroid.eye/) is a smartphone-based AI system for diabetic retinopathy screening deployed in rural Kenya. It's one of the best examples of constrained-environment AI done right.

### The Constraints
- **Connectivity**: Many screening locations have no internet at all
- **Power**: Screening often happens at community health centers with unreliable electricity
- **Hardware**: The system runs on a Tecno Phantom 9 (4 GB RAM, Helio P70 chipset)
- **Data cost**: Uploading fundus images (5-10 MB each) would cost prohibitive amounts over mobile data

### The Solution

1. **All inference on-device**: A quantized EfficientNet-Lite model (3.2 MB) runs entirely on the phone
2. **Offline-first**: The screening app works 100% offline
3. **Compressed sync**: Only inference results (a few bytes) are uploaded when connectivity is available; raw fundus images are uploaded only for positive cases requiring specialist review
4. **Battery-aware**: The ML pipeline pauses if battery drops below 20%
5. **Voice feedback**: Results are read aloud in Kiswahili and Dholuo (no reading required)

### Results
- **15,000+** screenings completed
- **87%** sensitivity and **91%** specificity (comparable to specialist examination)
- **45%** of screenings occur in locations with no internet
- **< 1%** of sessions fail due to technical issues

## Decision Framework: Choose Your Stack

When starting a new project for constrained environments, use this decision tree:

```
Do users have reliable internet?
├── YES → Use cloud inference (simplest, most powerful models)
└── NO → Continue ↓

Do users have intermittent internet (some access per day)?
├── YES → Use on-device inference with sync-when-available
├──     → TFLite or ONNX Runtime
├──     → Model compression: Int8 quantization + pruning
└── NO (fully offline) → Continue ↓

Does the device have a GPU/NPU?
├── YES → Use hardware acceleration if available
├──     → Consider GPU delegates (NNAPI, OpenCL)
└── NO → Use CPU-only inference
├──     → Model compression: pruning + distillation + Int8
├──     → Inference time target: < 200ms per prediction
└──     → Consider XGBoost/random forest as simpler alternatives

Is storage below 64 GB?
├── YES → Bundle models in APK (not downloaded post-install)
├──     → Model target: < 5 MB per model
└── NO → Can download models on first Wi-Fi connection
```

## The Final Word

Building AI for constrained environments isn't about "making do" with inferior technology. It's about disciplined engineering: understanding your resource budget, measuring everything, and making deliberate trade-offs. A 3 MB model that works offline on a budget phone and achieves 88% accuracy is more valuable than a 300 MB model that achieves 94% accuracy but only works in Nairobi with 4G and a full battery.

The techniques in this post — pruning, quantization, distillation, offline-first architecture, adaptive inference — are not niceties. They are the minimum viable requirements for AI systems that actually serve the billion-plus people living in constrained environments across Africa and beyond.

*This concludes the "African AI / Real-World ML" series. Thank you for reading. The conversation continues on the [Masakhane](https://www.masakhane.io/) Slack, at the next [Deep Learning Indaba](https://deeplearningindaba.com/), and in the fields and labs across Africa where this work is happening every day.*
