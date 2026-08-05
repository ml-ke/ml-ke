---
layout: post
title: "Edge AI: Running Models on Mobile in African Markets"
date: 2026-08-05
image:
  path: /assets/img/cover-series-practical-playbook.webp
  alt: cover series practical playbook
categories: [AI Engineering, AI in Africa]
tags: [edge-ai, mobile-ml, tflite, on-device, quantisation, offline-ai]
---

## The Mobile-First Context

Africa leapfrogged the desktop era — the internet arrived via SIM cards, not broadband cables. Smartphone penetration is now climbing past 50% across the continent, but the devices powering that growth are overwhelmingly mid-range and budget Android phones: 3–4 GB of RAM, 32–64 GB of storage, and chipsets optimised for power efficiency rather than raw compute.

These devices run AI inference differently than the cloud-connected flagship phones of Silicon Valley. The difference is that inference often happens **without internet** — because data costs in Africa are among the highest in the world (average 4–8% of monthly income for 1 GB), and large swathes of the continent have intermittent or absent connectivity.

This is where **edge AI** — running machine learning models directly on the device, not phoning home to a server — shifts from a nice-to-have to a hard requirement.

## What Edge AI Unlocks

Edge AI eliminates the three bottlenecks of cloud-based AI in African markets:

- **Latency**: No round-trip to a distant server. Inference in milliseconds, regardless of network quality.
- **Cost**: Zero data charges per inference. The model runs on hardware already in the user's pocket.
- **Reliability**: Works in flight mode. Works in rural areas with no signal. Works during network outages.

When you're building for users who top up mobile data in small increments — 50 MB or 100 MB at a time — every kilobyte counts. Edge AI burns none.

## The On-Device Toolkit

Multiple frameworks now support edge inference, and the right choice depends on your target platform and model architecture:

- **TensorFlow Lite (TFLite)** — The dominant choice on Android, which accounts for roughly 85% of the African smartphone market. Google's ML Kit provides pre-built APIs on top of TFLite for common tasks like text recognition, barcode scanning, and image labelling, cutting development time significantly.

- **Core ML** — Apple's on-device framework is essential if you're targeting the growing (though still smaller) iOS markets in South Africa, Nigeria, and Kenya.

- **ONNX Runtime** — The best cross-platform option when your model was trained in PyTorch or exported from HuggingFace. ONNX Runtime provides a consistent inference interface across Android, iOS, and Linux, simplifying multi-platform deployments.

- **Qualcomm SNPE / MediaTek NeuroPilot** — Hardware-specific SDKs that unlock the DSP, GPU, and NPU on Qualcomm and MediaTek chipsets respectively. These are the processors inside most mid-range African phones (e.g., the MediaTek Helio G-series found in Tecno, Infinix, and Itel devices). Using them can yield 3–5x faster inference than CPU-only execution.

## Quantisation: The Secret Sauce

The single most impactful optimisation for edge AI is **quantisation** — reducing the numerical precision of model weights and activations. The trade-off is small; the payoff is enormous.

- **FP32** (full precision): The training standard. A 100 MB model is too large to bundle in an APK for a device with 32 GB of storage.
- **FP16** (half precision): Cuts size by 50% with negligible accuracy loss. Provides a 1.5–2x speedup when GPU delegates are available.
- **INT8** (integer quantisation): Reduces model size by 75% — that 100 MB model becomes 25 MB. Accuracy drops by 0.5–2%, which is acceptable for most classification tasks. Hardware delegates (DSP, NPU) can run INT8 with 3–4x speedup.
- **INT4** (extreme quantisation): A relatively new technique that achieves 4–8x compression. Still experimental for many architectures, but promising for very constrained deployments.

Here is a practical quantisation pipeline using TensorFlow Lite:

```python
import tensorflow as tf

# Load a trained model
model = tf.keras.models.load_model("maize_disease_model.keras")

# Configure the converter for int8 full-integer quantisation
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide a representative dataset for calibration
def representative_dataset():
    for _ in range(200):
        data = tf.random.normal([1, 224, 224, 3])
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Convert and save
tflite_quantised = converter.convert()
with open("model_int8.tflite", "wb") as f:
    f.write(tflite_quantised)

print(f"Original size:   {model.count_params() * 4 / 1e6:.1f} MB (FP32)")
print(f"Quantised size:  {len(tflite_quantised) / 1e6:.1f} MB (INT8)")
```

The result: a model that was 20 MB in FP32 drops to roughly 5 MB in INT8, loads in under a second on a budget phone, and runs inference in 100–200 ms — all without a network connection.

## Edge AI in the Wild: Running in Africa Today

These aren't theoretical deployments. Several edge AI applications are already serving users across the continent:

- **PlantVillage Nuru** — A crop disease detection app built by Penn State and CIMMYT. Nuru runs a TensorFlow Lite model on-device to identify cassava, maize, and wheat diseases from smartphone photos. A farmer in rural Kenya can diagnose a diseased plant in seconds, with zero data costs and no internet signal. The app has been downloaded over 1 million times across East and West Africa.

- **Ada Health** — AI-powered symptom assessment that runs its core inference on-device. Users answer questions about their symptoms and receive a triage recommendation, all without sending health data to a server. This is critical in markets where privacy concerns and intermittent connectivity make cloud-dependent health apps unusable.

- **Google Flood Alerts** — While the flood prediction models run on Google's servers, the alert delivery and local risk assessment now leverage on-device ML to work offline. Users receive actionable warnings even when cell towers are down — a common scenario during flood events.

These examples share a common architecture: a quantised model bundled inside the app, a lightweight inference runtime, and a smart sync layer that updates the model when Wi-Fi is available.

## What's Coming: On-Device LLMs

The next frontier is running large language models on phones. Google's Gemini Nano is already integrated into Android's AICore, available on Pixel and select Samsung devices. For the broader ecosystem, two developments matter:

- **llama.cpp** has been ported to Android via projects like LLMInference and MLCCL (MLC Chat Client Library). A 2–3 billion parameter quantised model (Q4_K_M) fits comfortably within 2 GB of RAM and delivers token-by-token generation on a current mid-range chipset at 5–15 tokens per second — usable for translation, SMS composition, and basic Q&A.

- **MediaTek's NeuroPilot** stack and Qualcomm's AI Engine both now include dedicated LLM acceleration paths, with INT4 quantisation support that brings memory requirements down to under 1 GB for a 1.5B-parameter model.

For African markets, the killer LLM use case isn't a general-purpose chatbot. It's language translation (Swahili, Hausa, Yoruba to and from English), SMS autocomplete in African languages, and offline local-language FAQ bots for agriculture, healthcare, and financial services.

## Where This Fits

This post builds directly on our earlier guide, [Mobile-First AI with TensorFlow Lite](/posts/mobile-first-ai/), which covered the fundamentals of on-device ML for African smartphone markets. That post walks through model conversion, Android deployment patterns, and the specific constraints of budget devices (3 GB RAM, thermal throttling, no GPU). If you haven't read it, start there — then come back for the edge AI perspective.

## The Bottom Line

Edge AI is not a niche optimisation for African markets. It is the deployment model that makes AI actually accessible to the next billion users. The combination of quantisation, hardware delegates, and purpose-built frameworks means that a $100 Tecno phone today can run models that required a cloud server five years ago. Building for the edge means building for reality — where the network is patchy, data is expensive, and the phone in someone's hand is the only computer they own.
