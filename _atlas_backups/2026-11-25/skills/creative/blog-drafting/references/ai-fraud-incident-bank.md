# AI Fraud / Deepfake Incident Bank (verified Aug 2026)

Reusable, fact-checked anchors for fintech / AI-security posts on ml.co.ke. Every
figure below was verified against **2+ independent sources** at capture time
(search snippet → fetch article body → confirm exact number). Re-verify before
reuse — sources can be pulled, corrected, or superseded.

Mirror new anchors into `editorial-calendar.md`'s events library at the repo root
when a post consumes them.

## Incidents

| # | Incident | Date | Verified figures | Sources |
|---|----------|------|------------------|---------|
| 1 | Arup Hong Kong deepfake CFO scam | Feb 2024 (HK police disclosure; Arup confirmed May 2024) | ~HK$200M (~US$25M); 15 transfers across 5 HK accounts; real-time deepfake video call of CFO + "colleagues" built from public meeting footage; victim the only real person | CNN 2024-02-04, CNN Business 2024-05-16, Fortune 2024-05-17 |
| 2 | UK energy firm voice-clone CEO fraud | 2019 (reported Sep 2019, WSJ) | €220K (~US$243K) wire to "Hungarian supplier"; CEO's voice cloned; approved on voice recognition alone | Forbes 2019-09-03 (WSJ original), Bitdefender |
| 3 | KYC biometric injection volume | Group-IB "Weaponized AI", Jan 2026 | **8,065** injection attempts against ONE financial institution's KYC liveness for digital loan onboarding, Jan–Aug 2025; 300+ dark-web posts referencing "deepfake"+"KYC" 2022–Sep 2025; **+52% unique sellers in 2025**; deepfake-as-a-service from ~$5 | Biometric Update (Jan 2026), Signzy, Group-IB press release |
| 4 | Deepfake tool economy | Sensity 2024 report | **2,000+** deepfake creation tools online, dozens built to bypass KYC; fake ID photos / verification videos sell for a few dollars on underground markets | identity.org, sensity.ai/reports |
| 5 | Africa 2025 fraud data | Sumsub Identity Fraud Report 2026 (2025 data) | Kenya: deepfake ≈ **10% of fraud attempts**; South Africa: deepfake incidents **+269% YoY** while total fraud fell 31%; Côte d'Ivoire fraud +51% YoY to 4.5%; Tanzania 5.0% (highest in Africa); Uganda 4.7% | Businessday NG (Aug 2026) |
| 6 | Interpol global scale | The Register, Mar 16 2026 | 2025 global financial fraud losses ≈ **$442B**, expected to rise on AI; SG Valdecy Urquiza: "we are witnessing the **industrialization of fraud**"; Feb 2026 op: 651 arrests, 16 African countries, 1,200+ victims | The Register, Interpol |
| 7 | Operation Jackal III (West Africa) | announced Aug 25 2026 | 8-month op: **58 arrests, 263 suspects, 22 countries**; 39 arrests in South Africa (romance/investment scams) | Sahara Reporters 2026-08-25, BBC |
| 8 | Flutterwave unauthorized transfers (Nigeria) | Feb 2023 – Apr 2024 | 4 incidents / 14 months. **Feb 2023: ₦2.9B (~$4.2M)** in 63 txs across 28 accounts, spread to **107 accounts in 27 banks** (court petition Feb 20 2023); hundreds frozen; merchant-key/social-engineering theory; hack denied. **Mar 2023: ₦550M** to ~107 accounts in 27 banks. **Oct 2023: ₦19B (~$24M)** via unauthorized POS-merchant txs, ~6,000 holders across 35 banks; $24M recovery court order ~Mar 2024. **Apr 2024: ₦11B ($7M)** (insider ≥₦20B/$13.5M) to 5 institutions over 4 days; undetected because deposits kept **below fraud-check trigger limits**; Mareva injunction Feb 2024. Kenya: Jul 2022 ARA froze KSh 6.2B (~$52.5M) across 62 accounts; withdrawn Feb 2023 | TechCrunch 2023-03-05, TechCabal 2023-03-10, TechCabal 2024-05-16, Techpoint Africa 2023-03-05, Techpoint Africa 2023-02-06, TechCabal 2022-08-30, Techloy |

## Verification recipe

- **web_extract may be unavailable** (search-only backend). Fallback: fetch the
  article with curl and strip tags in python, then grep the exact figure:
  ```bash
  curl -sL --max-time 25 -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36" URL -o page.html
  python3 -c "
  import re, html
  t = html.unescape(re.sub(r'<[^>]+>', ' ', open('page.html', encoding='utf-8', errors='ignore').read()))
  t = re.sub(r'\s+', ' ', t)
  i = t.find('KEYWORD'); print(t[max(0,i-350):i+350])"
  ```
- Search snippets are NOT enough for quoted numbers — confirm the figure in the
  article body before publishing.
- Prefer primary/industry sources (CNN, The Register, Group-IB, Biometric Update,
  Businessday NG) over aggregators; drop any claim that can't be double-sourced.

## Post formula used (2026-08-26 deepfake post, "Deepfake Fraud in Financial Services")

Incident anatomy (Arup) → voice cloning (UK energy) → KYC front line (Group-IB,
Sensity) → Africa data table (Sumsub) → enforcement (Interpol, Jackal III) →
7-point "how we can do better" playbook → takeaways table → 13 references.

**House word count:** ~900–1100 body words including tables/callouts is normal
for main-agent editorial posts (the 700–900 figure is only the batch-subagent
target). 2 tables + 13 refs + 7-point playbook ≈ 1080 body words is fine.

## Ring / graph-detection anchors (verified 2026-09-11 — used by `graph-fraud-ring-detection`)

Body-level verified with `scripts/extract-web-text.py`; reuse instead of re-researching.

- **Europol EMMA 9** (Dec 5 2023): 1,013 arrests in 26 countries, 10,759 mules + 474
  recruiters, 2,800+ banks, >€100M losses exposed, €32M more prevented, 10,736 fraudulent
  transactions. OCCRP `occrp.org/en/news/global-operation-targets-money-mules`.
- **Operation Jackal IV** (sweep Nov 2025–Jun 2026; published Aug 25 2026): 58 arrested,
  263 suspects, 22 countries; South Africa 39 arrests, $2.67m seized, 257 accounts blocked.
  BBC `bbc.com/news/articles/cq5xdnxppl4o` + Sahara Reporters (Aug 25 2026).
- **Cifas Fraudscape 2026** (Mar 12 2026): 444,000+ cases to the UK National Fraud Database
  in 2025 (record, +6% YoY), >1,200/day, £2.4bn prevented, 242,000 identity-fraud cases (54%),
  78,000 facility-takeover cases (18%, +6%), **unauthorised SIM swaps +38%**. Verbatim: crime
  "increasingly digital, organised and international", OCGs "operating across borders and
  targeting multiple sectors simultaneously" `cifas.org.uk/newsroom/fraudscape2026`.
- **Hard vs soft links** — arXiv:2512.19061 (Dec 2025): hard links = high-confidence identity
  (phone/card/national ID), soft links = behavioural (device/cookie/IP); collapse hard-link
  components to super-nodes, then cluster → 25M → 7.7M nodes, doubled detection coverage.
- **GNN review** — arXiv:2411.05815 (Cheng et al., 100+ studies): relational models beat
  per-record baselines on financial fraud.
- **Elliptic Bitcoin benchmark**: 203,769 nodes / 234,355 edges, 4,545 labelled illicit (~2%)
  — `kumo.ai/pyg/datasets/elliptic-bitcoin/`, Nature Sci Rep s41598-025-95672-w.
- **Precision caution**: Aite-Novarica — ~90% of declined transactions are legitimate
  (reported by Analytics Vidhya, Nov 2025). Pair with any ring-detection post's FP discussion.
- **Neo4j** financial-fraud graph architecture (PageRank on shared devices, cycle detection)
  `neo4j.com/developer/industry-use-cases/finserv/retail-banking/ieee-cis-fraud-graphs/`.

**Reusable demo recipe:** synthetic transaction graph with `networkx` — soft-link device
clusters, hard-link phone components, directed cycle detection for layering loops, then a
weighted corroboration score (device + phone + loop + pass-through + fan-out). Rings surface
through *different* doors (one by shared device, one by loop) and shared-device-only clusters
must land at REVIEW, not FLAG. Verbatim output quoted in the post; recipe lives in the
`graph-fraud-ring-detection` post body.

