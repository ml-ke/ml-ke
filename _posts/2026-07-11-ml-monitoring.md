---
title: "Monitoring ML Systems in Production: Drift Detection, Performance, and Alerting"
date: 2026-07-11 00:00:00 +0300
categories: [ML Ops, AI Engineering]
tags: [mlops, ml-monitoring, drift-detection, alerting, model-performance]
image:
  path: /assets/img/cover-ml-monitoring.webp
  alt: Dashboard showing ML monitoring metrics including data drift, model performance, and system health
---

Your model is deployed. Traffic is flowing. Users are getting predictions. Everything looks good — until it doesn't.

Unlike traditional software, ML systems can degrade silently. The code doesn't crash; the model just starts producing worse predictions. A recommendation model that was 90% accurate in March might be 60% by July, not because the code changed, but because user behavior shifted. This is **model drift**, and detecting it is the core challenge of ML monitoring.

This post covers the four types of drift, how to detect each one, infrastructure monitoring, and how to build alerting that actually wakes someone up when something goes wrong.

## The Four Types of Model Drift

### 1. Data Drift (Input Distribution Changes)

The statistical properties of your input features change over time. For example, if you trained a fraud detection model in 2023, but by 2025 transaction patterns have shifted (more mobile payments, fewer card swipes), the model sees inputs it wasn't trained on.

{% raw %}
```text
Training Distribution:     Production Distribution (drifted):
  age: μ=35, σ=10           age: μ=42, σ=12
  income: μ=65K, σ=30K      income: μ=72K, σ=35K
  mobile_ratio: 0.3         mobile_ratio: 0.7  ← significant shift
```
{% endraw %}

### 2. Concept Drift (Label Relationship Changes)

The relationship between inputs and the target variable changes. A model that predicts house prices might suddenly become inaccurate not because the features changed, but because interest rates rose and the fundamental price/sq.ft relationship shifted.

{% raw %}
```text
Training: price = 250 * sqft + 50K * bedrooms  (r²=0.85)
After rate hike: price = 200 * sqft + 40K * bedrooms  (r²=0.60)
The relationship changed, not the inputs.
```
{% endraw %}

### 3. Prediction Drift (Output Distribution Changes)

The model's predictions themselves shift over time. Even if inputs look normal, the outputs might start clustering differently. This is often a symptom of data or concept drift manifesting.

### 4. Model Performance Degradation

The true metric you care about — accuracy, precision, recall, CTR — drops. This is the **most important signal** but also the hardest to measure because you need ground truth labels, which often arrive with a delay (e.g., did the user actually click? Did the transaction turn out to be fraudulent?).

{% raw %}
```text
Drift Type     | Detection Method          | Label Required? | Response Time
---------------|---------------------------|-----------------|--------------
Data Drift     | Statistical tests        | No              | Immediate
Concept Drift  | Performance monitoring   | Yes             | Delayed
Prediction Drift| Output statistics       | No              | Immediate
Performance    | Ground-truth comparison  | Yes             | Delayed
```
{% endraw %}

## Detecting Data Drift with Statistical Tests

### Feature-Level Drift

For numerical features, use the **Kolmogorov-Smirnov (KS) test** or **Wasserstein distance**. For categorical features, use the **Chi-squared test**.

{% raw %}
```python
# drift_detection.py
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def detect_numerical_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str,
    threshold: float = 0.05,
) -> dict:
    """
    Detect drift in a numerical feature using Kolmogorov-Smirnov test.

    Reference: training data or early production data.
    Current: recent production sliding window.
    """
    statistic, p_value = ks_2samp(reference, current)
    drifted = p_value < threshold

    if drifted:
        logger.warning(
            f"Data drift detected in '{feature_name}': "
            f"KS-stat={statistic:.4f}, p-value={p_value:.6f}"
        )

    return {
        "feature": feature_name,
        "type": "numerical",
        "drift_detected": drifted,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "reference_mean": float(np.mean(reference)),
        "current_mean": float(np.mean(current)),
    }


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
    threshold: float = 0.05,
) -> dict:
    """
    Detect drift in a categorical feature using Chi-squared test.
    """
    # Create contingency table
    cats = list(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts().reindex(cats, fill_value=0)
    cur_counts = current.value_counts().reindex(cats, fill_value=0)

    table = pd.DataFrame({"reference": ref_counts, "current": cur_counts})
    chi2, p_value, _, _ = chi2_contingency(table)
    drifted = p_value < threshold

    return {
        "feature": feature_name,
        "type": "categorical",
        "drift_detected": drifted,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
    }
```
{% endraw %}

### Multivariate Drift Detection

Feature-level tests miss interactions. Two features might each look fine individually but shift in combination. Use **PCA-based drift detection** for multivariate coverage:

{% raw %}
```python
from sklearn.decomposition import PCA


def detect_multivariate_drift(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    Detect drift in the joint distribution using PCA reconstruction error.

    Fit PCA on reference data, then compare reconstruction error on
    reference vs current as a proxy for distribution shift.
    """
    n_components = min(10, reference.shape[1], current.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(reference)

    # Reconstruction error on reference
    ref_reconstructed = pca.inverse_transform(pca.transform(reference))
    ref_error = np.mean((reference - ref_reconstructed) ** 2)

    # Reconstruction error on current
    cur_reconstructed = pca.inverse_transform(pca.transform(current))
    cur_error = np.mean((current - cur_reconstructed) ** 2)

    # Relative error ratio
    error_ratio = cur_error / ref_error
    drifted = error_ratio > (1 + threshold)

    return {
        "method": "pca_reconstruction",
        "drift_detected": drifted,
        "reference_reconstruction_error": float(ref_error),
        "current_reconstruction_error": float(cur_error),
        "error_ratio": float(error_ratio),
    }
```
{% endraw %}

## Using Evidently AI for Production Monitoring

[Evidently AI](https://www.evidentlyai.com/) is the go-to open-source library for ML monitoring. It computes drift metrics and generates interactive reports.

### Setting Up a Monitoring Dashboard

{% raw %}
```python
# monitoring_dashboard.py
import pandas as pd
from datetime import datetime, timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import *
from evidently import ColumnMapping
import joblib

# Load reference data (training or early production)
reference = pd.read_csv("data/training_data.csv")

# Column mapping
column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=["age", "income", "tenure"],
    categorical_features=["region", "plan_type"],
)

# Build monitoring report
report = Report(metrics=[
    DataDriftPreset(),
    RegressionPreset(),
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="income"),
    TargetByFeaturesTable(),
])

# Run on current production data (last hour's predictions)
current = pd.read_parquet("data/production_logs_latest.parquet")
report.run(
    reference_data=reference,
    current_data=current,
    column_mapping=column_mapping,
)

# Save as HTML dashboard
report.save_html("monitoring_reports/drift_report_latest.html")

# Extract drift summary
drift_summary = report.as_dict()
drifted_features = [
    metric["column_name"]
    for metric in drift_summary["metrics"]
    if metric["metric"] == "ColumnDriftMetric"
    and metric["result"]["drift_detected"]
]

if drifted_features:
    print(f"⚠ Drift detected in: {', '.join(drifted_features)}")
```
{% endraw %}

### Scheduling Periodic Checks

{% raw %}
```python
# schedule_monitoring.py — run every hour via cron or Airflow
import schedule
import time

def run_monitoring_check():
    """Hourly monitoring check — logs drift status."""
    current_data = fetch_production_snapshot()  # last hour's predictions
    drift_results = check_all_drift(current_data)

    if drift_results["drift_detected"]:
        send_alert(drift_results)
        log_to_prometheus(drift_results)

    log_to_mlflow(drift_results)
    print(f"[{datetime.now()}] Monitoring check complete. "
          f"Drift: {drift_results['drift_detected']}")

schedule.every(1).hour.do(run_monitoring_check)

while True:
    schedule.run_pending()
    time.sleep(60)
```
{% endraw %}

## Infrastructure Monitoring: Latency, Throughput, and Errors

Model drift is critical, but don't ignore infrastructure. A model that's 99% accurate but takes 10 seconds per request is useless.

### Key Metrics to Track

{% raw %}
```text
| Metric                | Good      | Warning    | Critical    |
|-----------------------|-----------|------------|-------------|
| P50 Latency           | < 100ms   | 100-500ms  | > 500ms     |
| P99 Latency           | < 500ms   | 500ms-2s   | > 2s        |
| Requests/sec          | Baseline  | ↓ 30%      | ↓ 50%       |
| Error Rate (4xx/5xx)  | < 0.1%    | 0.1-1%     | > 1%        |
| GPU Utilization       | 70-95%    | < 50%      | < 30%       |
| GPU Memory            | < 80%     | 80-90%     | > 90%       |
| Model Drift Score     | < 0.05    | 0.05-0.15  | > 0.15      |
```
{% endraw %}

### Prometheus Metrics from Your Inference Server

{% raw %}
```python
# metrics.py — add to your FastAPI server
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time
import psutil

# Define metrics
PREDICTIONS = Counter(
    "model_predictions_total",
    "Total number of predictions",
    ["model_name", "model_version"],
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_name", "model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)
ERRORS = Counter(
    "model_errors_total",
    "Total number of prediction errors",
    ["model_name", "error_type"],
)
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used",
    ["gpu_index"],
)
INPUT_SCHEMA_HASH = Gauge(
    "model_input_schema_hash",
    "Hash of input feature schema for drift detection",
    ["feature"],
)

# Example: wrap your prediction endpoint
@app.post("/predict")
async def predict_with_monitoring(request: PredictionRequest):
    start = time.time()
    try:
        result = model.predict(request.features)
        PREDICTIONS.labels(model_name="churn-model", model_version="v4").inc()
        return result
    except Exception as e:
        ERRORS.labels(model_name="churn-model", error_type=type(e).__name__).inc()
        raise
    finally:
        elapsed = time.time() - start
        PREDICTION_LATENCY.labels(
            model_name="churn-model", model_version="v4"
        ).observe(elapsed)


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")
```
{% endraw %}

## Building an Alerting System

Drift detection is only useful if it triggers action. Build a tiered alerting system:

### Alert Tiers

{% raw %}
```text
Tier 1 — INFO (Slack message, no escalation):
    • P50 latency increase < 20%
    • Minor data drift (p > 0.01)
    • Quick recovery after transient spike

Tier 2 — WARNING (Slack alert + email):
    • P99 latency > 1s
    • Data drift detected (p < 0.01)
    • Error rate 0.1-1%
    • GPU memory > 85%

Tier 3 — CRITICAL (PagerDuty or phone call):
    • Service down (100% errors)
    • P99 latency > 5s
    • Model accuracy drop > 5%
    • Severe concept drift detected
    • No recovery after 5 minutes
```
{% endraw %}

### Alert Manager Configuration (Prometheus + Alertmanager)

{% raw %}
```yaml
# alertmanager.yml
route:
  group_by: ['model_name', 'alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  routes:
    - match:
        severity: critical
      receiver: pagerduty
      repeat_interval: 15m
    - match:
        severity: warning
      receiver: slack-alerts
    - match:
        severity: info
      receiver: slack-info

receivers:
  - name: pagerduty
    pagerduty_configs:
      - routing_key: '<pagerduty-key>'
        severity: critical

  - name: slack-alerts
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/TXXX/BXXX/XXX'
        channel: '#ml-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: slack-info
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/TXXX/BXXX/XXX'
        channel: '#ml-logs'
```
{% endraw %}

### Prometheus Alert Rules

{% raw %}
```yaml
# prometheus-rules.yml
groups:
  - name: ml_monitoring
    rules:
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            rate(model_prediction_latency_seconds_bucket[5m])
          ) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency above 1s for {{ $labels.model_name }}"
          description: "P99 latency is {{ $value }}s"

      - alert: HighErrorRate
        expr: |
          rate(model_errors_total[5m]) / rate(model_predictions_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 1% for {{ $labels.model_name }}"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: DataDrift
        expr: model_drift_score > 0.15
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Model drift detected for {{ $labels.model_name }}"
          description: "Drift score is {{ $value }}"

      - alert: GPUMemoryPressure
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory > 90% on {{ $labels.instance }}"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
```
{% endraw %}

## Custom Monitoring Dashboard with Grafana

Here's a complete Grafana dashboard JSON (key panels) for ML monitoring:

{% raw %}
```json
{
  "title": "ML Model Monitoring",
  "panels": [
    {
      "title": "Request Rate & Errors",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate(model_predictions_total[5m])",
          "legendFormat": "{{model_name}} - Requests"
        },
        {
          "expr": "rate(model_errors_total[5m])",
          "legendFormat": "{{model_name}} - Errors"
        }
      ]
    },
    {
      "title": "P50 / P95 / P99 Latency",
      "type": "timeseries",
      "targets": [
        {"expr": "histogram_quantile(0.50, rate(model_prediction_latency_seconds_bucket[5m]))"},
        {"expr": "histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))"},
        {"expr": "histogram_quantile(0.99, rate(model_prediction_latency_seconds_bucket[5m]))"}
      ]
    },
    {
      "title": "Feature Drift Scores",
      "type": "heatmap",
      "targets": [
        {"expr": "model_drift_score"}
      ]
    },
    {
      "title": "GPU Utilization",
      "type": "stat",
      "targets": [
        {"expr": "avg(nvidia_gpu_utilization)"}
      ]
    }
  ]
}
```
{% endraw %}

## What to Do When Drift Is Detected

Detecting drift isn't enough — you need a runbook. Here's a template:

{% raw %}
```text
## DRIFT RESPONSE RUNBOOK

### Step 1: Acknowledge
- Log the drift event in the monitoring dashboard
- Determine severity (INFO / WARNING / CRITICAL)
- Notify the team via Slack (#ml-alerts)

### Step 2: Investigate
- Which features drifted? (check Evidently report)
- When did the drift start? (check timeline in Grafana)
- Did we deploy a new model version recently? (check MLflow)
- Did upstream data sources change? (check data pipelines)

### Step 3: Decide
- Can we retrain with recent data? → Trigger retraining pipeline
- Is a rollback appropriate? → Rollback to previous production model
- Is this a temporary seasonal shift? → Monitor for 24h before acting

### Step 4: Act
- If retraining: trigger CI/CD pipeline with recent data
- If rollback: promote previous model version to Production in MLflow
- If monitoring: set increased alert frequency, document expected duration

### Step 5: Document
- Log the incident: cause, action taken, outcome
- Update drift thresholds if the "drift" was actually a permanent shift
- Schedule a post-mortem if this was critical
```
{% endraw %}

## Tools Comparison

| Tool | Best For | Strengths | Limitations |
|------|----------|-----------|-------------|
| **Evidently AI** | Data & prediction drift reports | Rich visual reports, open-source | Not real-time; batch analysis |
| **WhyLabs** | End-to-end monitoring | Managed platform, good UI | SaaS; pricing at scale |
| **Prometheus + Grafana** | Infrastructure metrics | Fast, real-time, industry standard | No ML-specific drift detection |
| **MLflow** | Model registry & performance | Integrates with training pipeline | Limited monitoring capabilities |
| **Great Expectations** | Data validation | Great for batch data quality | Not designed for real-time serving |
| **NannyML** | Performance estimation without labels | Estimates accuracy without ground truth | Newer project, smaller community |

## Summary: The Monitoring Maturity Model

{% raw %}
```text
Level 1 — "It Works"          : nvidia-smi + curl health checks
Level 2 — "Dashboard"         : Prometheus + Grafana for latency/throughput
Level 3 — "Drift Detection"   : Evidently/WhyLabs for data drift
Level 4 — "Alerting"          : PagerDuty for critical failures
Level 5 — "Auto-Remediation"  : Automated retraining/rollback
Level 6 — "Predictive"        : Predict drift before it happens
```
{% endraw %}

Most teams start at Level 1–2. Getting to Level 4 (alerting on both infrastructure and ML-specific drift) is the minimum for production-grade ML. Level 5–6 is where you save significant engineering time through automation.

## Final Thoughts

Monitoring ML systems is harder than monitoring traditional software because the failure modes are more subtle. A model doesn't crash — it just gets slightly worse every day until someone notices. By combining:

- **Statistical drift detection** (Evidently + custom tests)
- **Infrastructure monitoring** (Prometheus + Grafana)
- **Tiered alerting** (Slack → PagerDuty)
- **Clear runbooks** (documented response procedures)

...you build a system that catches degradation early, alerts the right people, and makes it easy to roll back or retrain before users notice anything wrong.

This concludes the **AI Infrastructure & Serving** series. We've covered:
1. Model Serving 101 — From notebook to API
2. vLLM — High-throughput LLM serving
3. GPU Optimization — Getting the most from hardware
4. Kubernetes for ML — Scaling with Kserve
5. ML CI/CD — Automating model pipelines
6. ML Monitoring — Drift, performance, alerting

The common thread: production ML is a systems engineering challenge as much as a data science one. Build it right, and your models will serve reliably for years.
