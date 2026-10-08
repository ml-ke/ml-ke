# PSI / Fraud-Model Drift Monitoring Bank (verified Sep 3, 2026)

Reusable fact-checked material for ML-theme (Thu) drift/monitoring posts on
ml.co.ke. Consumed by `_posts/2026-09-03-fraud-model-drift-monitoring.md`.
Re-verify before reuse — sources can change.

## PSI threshold convention (0.10 / 0.25)

The Population Stability Index is the credit-scorecard industry's standard
population-drift meter. Score range cut into k=10 fixed bins; compare expected
(training) share E_i vs actual (live) share A_i per bin:

$$\text{PSI} = \sum_{i=1}^{k} (A_i - E_i) \times \ln(A_i / E_i)$$

Rule of thumb (multi-source verified):

| PSI | Interpretation | Action |
|-----|----------------|--------|
| < 0.10 | Little change | No action |
| 0.10 – 0.25 | Moderate shift | Investigate drivers, watch next period |
| > 0.25 | Significant shift | Retrain trigger — backtest challenger |

Sources: Fiddler AI blog (formula + binning), Coralogix AI blog (PSI >= 0.25
"significant change — model unstable, needs update"), ResearchGate paper
"Statistical Properties of the Population Stability Index" (Yildirim et al.,
0.10/0.25 rule of thumb), CRAN `scorecard::perf_psi` docs (same bands).

## Differentiated angle map (ML theme collisions to avoid)

| Post | Angle | NOT to duplicate |
|------|-------|------------------|
| Sep 3 fraud-model-drift-monitoring | **Population-level** monitoring of a fraud model: PSI on the score distribution, monthly score-creep demo, feature fill-rate gates, retrain triggers | — |
| Aug 31 anomaly-detection-reconciliation | Per-line / entity-day **detection** of anomalous transactions (z-score, CUSUM, Isolation Forest) | re-telling NCBA/Flutterwave as detection stories |
| Jul 11 ml-monitoring | Generic drift taxonomy (4 types), KS/chi-squared tests, Evidently dashboards, Prometheus/Grafana infra | re-covering KS tests or dashboards |
| Aug 25 mlops-regtech-model-governance | CBK/ODPC/EU AI Act regulatory **governance stack** (drift mentioned as a compliance obligation) | regulatory framing |
| Aug 20 fraud-ml-mobile-money | Feature design for mobile-money fraud (velocity, graph) | features themselves |

**Differentiation formula used:** incidents stay the same (Flutterwave ₦11B
sub-trigger adaptation, NCBA ghost accounts) but the LENS changes — they are
*adversarial population drift* (the model's world moved), not per-line
anomalies and not a governance checklist. State the differentiation in the
intro in one sentence ("The Aug 31 post asked how to detect the anomalous
line; this post asks how you know the model itself has gone stale").

## Reusable demo recipe (seed 42, outputs verified Sep 3)

- `score_population(n, shift)`: gaussian scores mean 38 + shift*14, sigma 9,
  clipped to [0,100], plus an adversarial bump (`random.random() < 0.18*shift`
  → uniform 55–70, i.e. below the alert band) to model sub-threshold splitting.
- `psi(expected, actual, bins=10)`: fixed bin grid over [0,100], guard empty
  bins with `max(share, 1e-4)`.
- Shift ladder 0.00/0.00/0.05/0.18/0.32/0.65 across Jan–Jun produces the clean
  zone progression (verified output, quote verbatim):
  ```
  period     PSI vs train   zone
  Jan             0.000   OK (<0.10)
  Feb             0.000   OK (<0.10)
  Mar             0.012   OK (<0.10)
  Apr             0.121   WATCH (0.10-0.25)
  May             0.344   RETRAIN (>0.25)
  Jun             1.223   RETRAIN (>0.25)
  ```
- Jun band table shows the 50–70 score band growing from ~9% of population at
  training to ~44% live ("risky but not alert-worthy" — the fraudsters'
  adaptation footprint).
- Data-quality gate demo: device_id fill-rate 99.4% → 93.1% by month with a
  `<95% → FAIL` gate — the fill-rate degrades while the score stays in-band
  (model compensates via fallback features). This is the failure class PSI on
  scores cannot see.
- Code contains f-string braces → wrap in `{% raw %}...{% endraw %}` (Pitfall
  #3). Verify with `scripts/verify-post-code.py` before quoting output.

## Incident anchors (adversarial-drift framing, 2+ sources each)

- **Flutterwave Apr 2024**: ₦11B (~$7M; insider says ≥₦20B/$13.5M) to accounts
  across 5 financial institutions over 4 days, undetected because deposits
  were kept **below fraud-check trigger limits** — the canonical engineered
  distribution shift. Sources: TechCabal exclusive (2024-05-16), Techpoint
  Africa, Business Insider Africa (all three re-verified in search Sep 3).
- **NCBA Bank Rwanda Jun 2025**: 70 ghost accounts, 260 txs, Ksh 57.5M /
  USD 446k over 8 days, caught by EOW reconciliation — concept drift: a *new
  pattern* (insider faking the success side of the ledger) the training
  population had no label for. Sources: event library (kenyainsights,
  254news, businessdailyafrica).
- **Danske Bank + Teradata**: 1,200 FPs/day, 99.5% of investigated cases not
  fraud → ML cut FPs ~50%, raised detection ~60%. FP rate trend is concept
  drift in disguise (decision boundary no longer matches reality). Sources:
  Teradata case study + Fintech Futures.
- **Cover metaphor used** ("population creep"): cyan dashed EXPECTED curve +
  red ghost curves creeping right month-over-month (opacity ladder
  0.13/0.26/0.42) into a solid LIVE curve, PSI zone ruler with needle, right
  panel FILL-RATE GATE. Distinct from #30 dashboard, #33 radar, #29 pipeline.

## Production controls list (the "how we can do better" payload)

1. Score PSI weekly, feature PSI monthly; alert at 0.10, escalate at 0.25.
2. Publish the retrain trigger decision rule BEFORE the alert fires.
3. Gate every batch on feature fill-rate (one threshold per critical feature).
4. Watch FP-rate trend as an operational canary (Danske lesson).
5. Log raw scores + features per period so PSI is retroactively computable.
6. Treat monitoring as the adversary's clock — drift visibility during the
   weeks-long adaptation phase is the whole point.
