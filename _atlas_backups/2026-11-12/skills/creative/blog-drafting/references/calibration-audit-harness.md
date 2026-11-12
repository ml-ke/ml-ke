# Probability calibration / reliability audit — verified harness and banks

Added Oct 7 2026 (post `probability-calibration-risk-scores`). Reuse this instead of
re-deriving: the numbers below are the shipped, byte-verified ones.

## Why a post lands in this class

Any Lane B tutorial about *probabilities* rather than *rankings*: reliability diagrams, ECE/MCE,
Brier score, Platt/temperature/isotonic scaling, calibration-in-the-large, threshold selection,
score-to-loss arithmetic. Differentiated from the retriever/drift/split siblings by
`rag-recall-at-k-denominator` (retrieval), `fraud-model-drift-monitoring` (PSI drift),
`temporal-validation-fraud-models` (split discipline), `embedding-compression-audit` (vector audit).

## Verified sources (fetched at body level, not snippets)

- Guo, Pleiss, Sun, Weinberger (2017), *On Calibration of Modern Neural Networks*, ICML —
  `https://arxiv.org/abs/1706.04599` (abs page works with `extract-web-text.py`; PDF via
  `curl -sL -o guo.pdf` + `pdftotext -layout`). Quotes verified verbatim from the PDF:
  "modern neural networks, unlike those from a decade ago, are poorly calibrated"; "most datasets and
  models experience some degree of miscalibration, with ECE typically between 4 to 10%";
  "temperature scaling -- a single-parameter variant of Platt Scaling -- is surprisingly effective";
  ECE is eq. (3), partitioning predictions into M equally-spaced bins; "temperature scaling does not
  affect the model's accuracy" (argmax unchanged); Table 1 uses M = 15 bins.
- Kumar, Liang, Ma (2019), *Verified Uncertainty Calibration*, NeurIPS spotlight —
  `https://arxiv.org/abs/1909.10155`. Platt/temperature scaling are "less calibrated than reported";
  histogram binning needs `O(B/eps^2)` samples vs `O(1/eps^2)` for scaling methods; an estimator
  from the meteorological community measures calibration error with `O(sqrt(B))` instead of `O(B)`;
  35% lower calibration error than histogram binning on CIFAR-10/ImageNet; ships a Python library.
- Van Calster et al. (2019), *Calibration: the Achilles heel of predictive analytics*, BMC Medicine
  17:230 — `https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7`. **Open access
  and curl-friendly** (unlike PubMed). Four levels: mean (calibration-in-the-large), weak (intercept
  and slope), moderate (flexible curve), strong (every covariate pattern). Calibration-slope target
  1, slope < 1 = risks too extreme; intercept target 0, negative = overestimation. Suggests a
  minimum of 200 events and 200 non-events for a precise curve.
- Van Calster et al. (2016), *A calibration hierarchy for risk models was defined*,
  J. Clin. Epidemiol. 74:167-176 — PubMed `26772608` is JS-gated; cite it but quote the hierarchy
  from the 2019 BMC article above.
- van den Goorbergh, van Smeden, Timmerman, Van Calster (2022), *The harm of class imbalance
  corrections for risk prediction models*, JAMIA 29(9):1525-1531 — arXiv PDF `2202.09101`
  (curl-friendly). Verified figures: imbalance correction did **not** improve AUROC; median
  calibration intercepts of −4.5 or lower at a 1% event fraction, −2.1 at 10%, −0.7 at 30%;
  intercept-only recalibration repaired them to between −0.07 and 0.03; SMOTE/ROS produced median
  slopes below 1.
- Murphy (1973), *A New Vector Partition of the Probability Score*, J. Appl. Meteorol. 12(4):595-600,
  DOI `10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2` — Brier = reliability − resolution +
  uncertainty. The AMetSoc record page is fetchable.
- scikit-learn calibration docs — `https://scikit-learn.org/stable/modules/calibration.html`.
  Sigmoid (Platt) works best with symmetric errors and small samples; isotonic is more powerful and
  more prone to overfitting, performing as well or better above roughly 1,000 samples; isotonic
  introduces ties and can move ROC-AUC, while sigmoid is strictly monotone and preserves ranking;
  temperature scaling is `softmax(z/T)` fitted by `log_loss` and does not change accuracy.

## TRAP: Guo Table 1 column order

```
Uncalibrated | Hist. Binning | Isotonic | BBQ | Temp. Scaling | Vector Scaling | Matrix Scaling
```

CIFAR-100 ResNet-110 row: 16.53% | **2.66%** | 4.99% | 5.46% | **1.26%** | 1.32% | 25.49%.
The 2.66% is **histogram binning**; temperature scaling is **1.26%**. Quoting the wrong column is
an easy error — print the header row and count columns before writing the number into prose.

## Harness (numpy only; deterministic; seed 11; block-isolated)

`risk_data()` = true process `logit = -2.9 + 1.4*z`, `y ~ Bernoulli(sigmoid(logit))`, shipped score
`logit_model = 1.6*logit_true + 0.35*N(0,1) + 0.9` (overconfident spread, level approximately
right). Split halves: fit on `[:n//2]`, evaluate on `[n//2:]`. Helpers used: `reliability()`
(equal-width and equal-mass bins, returns rows/ECE/MCE), `fit_logistic()` (Newton/IRLS — needs
`np.eye(k)` for k = number of columns; a hardcoded `np.eye(2)` crashes the intercept-only fit),
intercept-only recalibration by 1-D Newton on `sum(y-p)/sum(p(1-p))` with the slope pinned at 1,
`fit_temperature()` (golden-section on NLL with `phi = (5**0.5-1)/2`), `murphy()` (group by 2-dp
score: the identity is exact only when every forecast inside a group is identical). Full published
code lives in `_posts/2026-10-07-probability-calibration-risk-scores.md`.

## Verified fixture numbers (n = 60,000; reuse directly, do not re-quote from memory)

- Full set: base rate 9.52%, mean score 9.62%; at the 0.90 cut 269 flagged (0.4%), precision 72.1%;
  0.80 → 670 flagged, 64.5%; 0.50 → 2,887, 48.2%.
- Held-out half (30,000): base 9.50%, mean score 9.62%; ECE (15 equal-width) 3.12%, MCE 28.87%,
  ECE (equal-mass) 3.13%, Brier 0.0749; top bin 0.93-1.00 predicts 96.1% and observes 77.9%
  (n = 68); 20,538 rows in the lowest bin.
- Small-sample trap: ECE recomputed on a 300-row sample = 6.35% (vs 3.12%).
- Weak calibration: intercept b = −0.626, slope a = 0.613, level shift c = −0.015, T = 1.259.
- Variants (ECE / Brier / level in points / flagged at 0.90): raw 3.12% / 0.0749 / +0.12 / 135;
  intercept shift 3.11% / 0.0748 / +0.03 / 134; temperature T = 1.259 → 2.69% / 0.0747 / +2.28 / 58;
  Platt b = −0.63, a = 0.613 → 0.34% / 0.0720 / +0.01 / 4.
- ROC-AUC 0.8179 raw and after Platt (monotone fits cannot change it).
- The 135 rows above the raw 0.90 cut carry calibrated probabilities 0.67-0.95 (mean 0.75) and
  defaulted 69.6% of the time.
- Murphy: reliability 0.0005 − resolution 0.0145 + uncertainty 0.0860 = 0.0720 (direct 0.0720).

If a future post needs a *pure overconfidence* story (temperature scaling visibly improving ECE),
use a smaller offset; if it needs a *level* error for the imbalance-correction angle, use a larger
positive offset so mean predicted risk exceeds the event rate. Both variants were explored; the
shipped one trades temperature-scaling credit for the level-versus-spread lesson.
