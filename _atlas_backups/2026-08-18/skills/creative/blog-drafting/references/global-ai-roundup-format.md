# Global AI Roundup Format (Tuesday Recurring)

## Purpose
A weekly post that surveys AI developments worldwide, organised by region. Published every Tuesday on ml.co.ke.

## Coverage Mandate
The user explicitly requires coverage BEYOND Western companies (Claude, Gemini, OpenAI). Every roundup MUST include developments from:

| Region | Key Players to Monitor |
|--------|----------------------|
| Asia/China | Moonshot/Kimi, Z.AI/Zhipu/GLM, DeepSeek, Alibaba/Qwen, Tencent, Baidu/Ernie, Kling AI, ByteDance/Doubao |
| Europe | Mistral, Aleph Alpha, DeepL, Synthesia, ElevenLabs, Lovable, Poolside, EU AI Act |
| MENA | G42/Core42, e& UAE, Inception42, Saudi funds, Egyptian startups, Israel AI labs |
| Africa | Google Africa AI Lab, Cue, Lelapa AI, ALX, Prosus, startup funding, AI policy |
| South America | Brazil AI Plan (PBIA), AI Summit Brazil, Brazilian startups, DP World data |
| Russia | Yandex/YandexGPT, Sber/GigaChat, sanctions workarounds, Chinese chip procurement |
| Western | Include but do NOT dominate — Grok, OpenAI GPT, Anthropic, Google, Microsoft |

## Post Structure

Front matter:
```yaml
title: "Global AI Roundup: Week of YYYY-MM-DD"
date: YYYY-MM-DD 00:00:00 +0300
categories: [AI Engineering, AI in Africa]
tags: [global-ai, weekly-roundup]
image:
  path: /assets/img/cover-global-ai-roundup-july-2026.webp
  alt: Global AI connectivity across seven regions
```

Body: H2 intro, region H3 sections with 2-4 items each, "What This Means for Africa" section, Next Week forward look. 800-1200 words. Every fact backed by real source URL.

Sources to check: Reuters AI, Bloomberg, TechCrunch, VentureBeat, Launch Base Africa, BNamericas, Middle East AI News, Tech Buzz China, RFE/RL, regional Google News searches.

Cron: fires Tue 12:00 EAT, writes directly to _posts/, git-pushes with blog-drafting skill loaded.
