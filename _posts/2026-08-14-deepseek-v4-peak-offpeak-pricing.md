---
title: "DeepSeek V4 Peak/Off-Peak Pricing: The End of Flat-Rate LLM APIs"
date: 2026-08-14 00:00:00 +0300
categories: [LLM, AI Engineering]
tags: [deepseek, llm-pricing, cost-optimization, off-peak, api-economics, benchmarks]
image:
  path: /assets/img/cover-deepseek-v4-peak-offpeak-pricing.webp
  alt: Clock face with red peak-hour arcs and a yellow price tag, symbolizing time-of-day LLM billing
---

## DeepSeek introduces time-of-day billing

On **August 16, 2026 at 16:00 UTC (19:00 EAT)**, DeepSeek switches its V4 API from flat-rate pricing to **peak / off-peak billing** — a first for a major Chinese model provider at this scale. Off-peak rates are exactly half of peak rates, and the peak windows are narrow: **01:00–04:00 and 06:00–10:00 UTC**, which translates to **04:00–07:00 and 09:00–13:00 EAT**. Everything else — roughly 75% of the day for Kenyan users — is off-peak.

> **The headline:** even the *new off-peak* rates are more expensive than today's prices, and peak rates are 2–4× current levels depending on token type. The era of flat, ever-declining API prices is over for DeepSeek.
{: .prompt-warning }

## What actually changes

| Model | Period | Input (cache hit) | Input (cache miss) | Output |
|-------|--------|-------------------|--------------------|--------|
| **v4-flash** | Today | $0.0028 | $0.14 | $0.28 |
| | Off-peak | $0.007 (↑150%) | $0.22 (↑57%) | $0.66 (↑136%) |
| | Peak | $0.014 (↑400%) | $0.44 (↑214%) | $1.32 (↑371%) |
| **v4-pro** | Today | $0.003625 | $0.435 | $0.87 |
| | Off-peak | $0.022 (↑507%) | $0.66 (↑52%) | $1.98 (↑128%) |
| | Peak | $0.044 (↑1114%) | $1.32 (↑203%) | $3.96 (↑355%) |

*All prices per 1M tokens. Source: [DeepSeek pricing docs](https://api-docs.deepseek.com/quick_start/pricing/).*

Two patterns stand out. First, **cache-hit input is the biggest jump** — 150–507% off-peak, up to 1,114% at peak for v4-pro. DeepSeek is pricing the convenience of hot KV-cache reads much closer to their real serving cost. Second, **output tokens are the real money**: at peak, a v4-pro output token costs $3.96/1M, over 4.5× today's price. Anyone running agent loops or long generation pipelines just saw their bill's dominant line item inflate.

## Why now: inference is a capacity business

This is textbook demand-shaping, straight out of the electricity market playbook. When a model is as popular as V4 — released April 24, 2026 with **87.5 MMLU-Pro, 80.6% SWE-bench Verified and 93.5 LiveCodeBench** for v4-pro (official numbers) — the marginal cost of serving a token at 3 a.m. is far lower than at 9 a.m. Peak/off-peak pricing nudges batch jobs and cron workloads into the cheap hours, smoothing the load curve without building idle capacity.

DeepSeek's own efficiency claims make the economics visible: the V4 technical report says **v4-pro needs only 27% of the single-token inference FLOPs and 10% of the KV cache of V3.2 in the 1M-token setting**. That efficiency is why V4 could undercut Western labs for months — but capacity is still finite, and time-of-day pricing is the lever that keeps utilization high without rationing.

## DeepSeek is not alone — the whole market is repricing

The flat-rate-to-dynamic shift is happening while **frontier labs are raising prices across the board**:

- **Anthropic**: Claude Sonnet 5 launched at intro pricing of **$2/$10** per 1M input/output tokens — and that intro pricing ends **August 31, 2026**, with an increase effective **September 1**.
- **OpenAI**: GPT-5.5 launched at **roughly 2× GPT-5.4's per-token price** (GPT-5.4 was $2.50/$15), with the flagship tier now at **$30/1M input**. A **10% regional-processing uplift** applies to models released on or after March 5, 2026.
- **Google is the counter-trend**: Gemini 3.7 and 3.6 Flash are offered at **$0.75/$3.75** introductory pricing through December 31, 2026 — a deliberate price cut against the hiking pack.

The market is bifurcating: **frontier labs raise prices while challengers (and open weights) compress them**. DeepSeek's move is best read not as a pure hike but as *capacity management* — they keep the absolute floor low (cache-hit at $0.007 off-peak is still cheaper than any Western frontier model) while making the premium tiers behave like a real utility.

## What this means for how we use LLMs

**1. Time-shift everything that can wait.** Batch inference, nightly indexing, evals, report generation, training-data synthesis — anything without a hard latency requirement should run in off-peak hours. For Kenya, that means scheduling heavy workloads **after 13:00 EAT** (and avoiding the 09:00–13:00 morning peak entirely).

**2. Cache hits are now the single biggest lever.** A cache-hit input token at off-peak is **~30× cheaper than a cache-miss** ($0.007 vs $0.22 for v4-flash). Designing prompts with stable system prefixes, shared context blocks, and conversation-level caching isn't an optimization anymore — it's the difference between a $10/month bill and a $300/month bill on high-volume workloads.

**3. Judge models by price-per-benchmark-point, not sticker price.** The old DeepSeek story was "near-frontier quality at 1/10th the price." The new story is "near-frontier quality at 1/10th the price *if you schedule smart*." At off-peak, v4-flash output at **$0.66/1M** with **79.0% SWE-bench Verified** is still a phenomenal deal vs GPT-5.5 Pro at $30/1M input. At peak, the gap narrows fast.

**4. Agents change the calculus.** Agent loops are output-heavy and latency-sensitive — the two things peak pricing punishes hardest. Expect agent platform costs to be the first thing renegotiated, and expect providers to start offering agent-specific off-peak tiers.

## Practical takeaway for Kenyan teams

With EAT peak hours at **04:00–07:00 and 09:00–13:00**, the cheapest compute day looks like: schedule cron jobs and batch pipelines **after 13:00**, keep interactive work on cache-heavy prompts, and push any 24/7 agent workloads to the 13:00–04:00 window where possible. This is the same demand-response thinking that keeps Kenyan electricity tariffs lower at night — applied to tokens.

> **Bottom line:** DeepSeek's peak/off-peak model is the first visible sign that LLM pricing is maturing from "race to the bottom" to "capacity market." The cheap era isn't over — it's just moved to specific hours, and the teams that schedule around it will pay a fraction of what the peak-hour crowd pays.
{: .prompt-tip }

## References

- [DeepSeek API pricing docs](https://api-docs.deepseek.com/quick_start/pricing/) — official peak/off-peak announcement
- [Macaron: DeepSeek V4 Benchmarks](https://macaron.im/blog/deepseek-v4-benchmarks) — official V4 benchmark summary
- [Claude pricing docs](https://platform.claude.com/docs/en/about-claude/pricing) — Sonnet 5 intro pricing ending Aug 31
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) — GPT-5.5 tiers and regional uplift
- [Google Cloud Agent Platform pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing) — Gemini 3.7/3.6 Flash intro pricing

## Related

- [DeepSeek's 52B Chip: Efficiency at the Silicon Level](/posts/deepseek-52b-chip/)
- [Self-Hosting Open-Weight LLMs: When It Finally Pays Off](/posts/self-hosting-open-weight-llms/)
- [Evaluating LLMs for African Use Cases](/posts/evaluating-llms-african-use-cases/)
