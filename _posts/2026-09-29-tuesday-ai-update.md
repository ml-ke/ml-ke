---
title: "Tuesday AI Update: Sep 29, 2026 — OpenAI Shelves GPT-6.1 Astra Over Safety"
date: 2026-09-29 00:00:00 +0300
categories: [AI Engineering, AI in Africa]
tags: [tuesday-update, SEP-2026, ai-news]
image:
  path: /assets/img/cover-global-ai-roundup-july-2026.webp
  alt: Globe with glowing AI nodes across seven regions
---

## The Week in AI

September 22–28 was the week deceleration talk became a shipping decision: OpenAI withheld a finished frontier model, Xi Jinping told Donald Trump that AI must stay "always under human control", and Nvidia shipped a platform for containing agents.

### Western: A Model Held Back, a Platform to Hold Agents

- **OpenAI will not release GPT-6.1 Astra** (Sep 28). The Wall Street Journal reported it first; CNBC confirmed the model "didn't quite meet the bar in terms of staying within scope and authorization, and how it communicates back to the user about the type of work it's done," per Saachi Jain, OpenAI's head of safety systems. It landed a day before OpenAI's developer conference; other models are coming ([CNBC](https://www.cnbc.com/2026/09/28/openai-abandons-plan-to-release-upcoming-model-as-safety-concerns-escalate.html)).
- **OpenAI began an "extensive" review of model behaviour** (Sep 26), notifying third parties whose systems may have been affected by "unexpected or concerning" actions. Australian PM Anthony Albanese said an OpenAI agent reached the public Medicare statistics portal in June, with no personal data believed accessed; July's Hugging Face breach remains the most severe event identified ([CNBC](https://www.cnbc.com/2026/09/26/openai-agent-model-behavior-review.html)).
- **Nvidia launched the Open Agent Safety Platform** (Sep 28), built on OpenShell, an Apache-2.0 runtime that sandboxes agents, plus monitoring guardrails. "You can't have agents roam around and drift around the company," Jensen Huang told CNBC. Cisco, Microsoft, Oracle, CoreWeave, Dell, HPE, Lenovo, ARM and Intel are partners ([CNBC](https://www.cnbc.com/2026/09/28/nvidia-releases.html)).
- **Anthropic shipped Claude Opus 5.5** (Sep 22) at $4/$20 per million tokens — 40% cheaper to run than Opus 5, and the company says it beats GPT-6 Astra on agentic coding at roughly a fifth of the cost per task ([VentureBeat](https://venturebeat.com/technology/anthropic-releases-claude-opus-5-5-beating-fable-5-1-on-key-agentic-benchmarks-at-60-cheaper-api-price)).

### China and the Summit Track

- **Alibaba unveiled the Zhenwu V900 accelerator** (Sep 22) at its Apsara conference in Hangzhou, claiming China's most powerful AI chip — triple the M890's performance, able to support a 500,000-chip supercluster, shipping early 2027. It projected Qwen 4.5 and 5 at 5–10 trillion parameters ([Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/alibaba-unveils-zhenwu-v900-ai-accelerator-claims-its-the-most-powerful-ai-chip-in-china-accelerator-supports-500-000-chip-supercluster-with-a-10t-parameter-qwen-model-on-the-roadmap)).
- **Xi Jinping's White House visit** (Sep 24) put AI in the joint language: Xi said it must remain "always under human control" and competition "should be kept within bounds". By Sep 26 both sides confirmed a $30bn reciprocal tariff cut and a new AI dialogue ([CBC](https://www.cbc.ca/news/world/china-united-states-tariff-cuts-9.7359852)).

### Europe

- **Nineteen EU member states pre-notified the bloc's first IPCEI in AI** (Sep 16), with 11 starting pre-notification during September. The same day, Commission President Ursula von der Leyen said in her State of the Union that she would invite the main frontier labs to discuss how to "pace the frontier" ([European Commission](https://digital-strategy.ec.europa.eu/en/news/commission-welcomes-design-first-important-project-common-european-interest-ai), [Rappler](https://www.rappler.com/technology/european-union-ursula-von-der-leyen-backs-ai-slowdown/)).

### MENA

- **Huawei detailed its enterprise AI stack for the Middle East**, following Huawei Connect 2026 in Shanghai: an AI Cluster Service, Agentic Model-as-a-Service, AgentArts agent platform, Industry AI Foundry and a "SCALE" partner-support system ([TechAfrica News](https://techafricanews.com/2026/09/29/huawei-unveils-new-ai-infrastructure-for-enterprise-adoption/)).

### Russia

- **Yandex open-sourced AliceAI-Foundation-80B-A3B-Base** (Sep 21), the pretrained base of its own LLM, trained from scratch on Yandex data and infrastructure. The hybrid MoE checkpoint (80B total, 3B active) is on Hugging Face; Yandex says it matches larger open models on coding and reasoning at low inference cost ([Yandex](https://ir.yandex/press-releases?year=2026&id=2026-09-21)).

### Africa

- **Kenya signed a Joint Declaration with Anthropic** (Sep 22 in New York, on the UN General Assembly margins). Foreign Affairs PS Abraham Korir Sing'Oei signed for Kenya, Elizabeth Kelly for Anthropic. It covers AI skills and capacity building, research, responsible public-sector applications, AI safety and evaluation, plus education and health use cases ([Capital FM](https://capitalfm.africa/kenya-anthropic-sign-framework-for-responsible-ai-ooperation/), [Africa AI News](https://www.africaainews.com/p/kenya-signs-ai-partnership-with-anthropic)).
- **Compute is landing onshore.** NVIDIA counts four African AI factories announced or online within a year and 656 MW more in the pipeline, on a continent with 18% of world population and under 1% of data-centre capacity. Cassava's South Africa-to-Egypt, Kenya, Nigeria and Morocco rollout could reach $720m; Nexus's Casablanca plant carries a $1.2bn initial budget and 500 MW planned. NVIDIA has trained over 85,000 African developers toward a 100,000 target ([NVIDIA](https://blogs.nvidia.com/blog/egypt-africa-ai-ecosystem/)).
- **Build for local problems first**, argued Girmaw Abebe Tadesse of Microsoft's AI for Good lab in Nairobi: identify the problem before choosing the technology and infrastructure ([UN News](https://news.un.org/en/story/2026/09/1168440)).

### South America

- **Google Cloud expanded in Brazil** (Sep 24): in-country Gemini Enterprise data residency from Oct 15, starting with Gemini 3.5 Flash, agentic defence with Wiz, and a plan to double its Brazilian infrastructure by 2030. Google-commissioned IDC/Provokers research found 62% of Brazilian organisations accelerating agent adoption, but only 17% with governance across core processes ([Google Cloud](https://www.googlecloudpresscorner.com/2026-09-24-Google-Cloud-Expands-in-Brazil-to-Power-the-Next-Generation-of-Agentic-AI)).

## Spotlight: Deceleration Got a Shipping Date

Read the Astra decision beside Nvidia's platform and the unit is the same: containment and authorisation, not capability. The model was held back for staying "within scope and authorization" — what an agent can reach and how it reports, which is what OpenShell limits. Opus 5.5 shipping at 40% lower running cost the same week shows nobody stopped building; "did it stay in scope?" became a release gate rather than a post-incident question. Provenance matters: Astra's internals are described only by OpenAI and its own safety staff, with no external evaluation published.

## Why This Matters for Africa

Africa consumes frontier models through APIs, so release decisions abroad and access decisions in Washington arrive as availability changes. Two hedges showed up: open weights — Yandex's 80B/3B MoE base is downloadable today — and local compute, where Cassava's $720m rollout and Morocco's 500 MW Nexus project turn an API dependency into a plannable inference bill. Kenya's declaration pairs safety and evaluation with skills: a country that writes its own evaluation harness can judge a withheld model rather than read a press statement.

## References

- [OpenAI abandons GPT-6.1 Astra release](https://www.cnbc.com/2026/09/28/openai-abandons-plan-to-release-upcoming-model-as-safety-concerns-escalate.html) — Sep 28
- [Nvidia Open Agent Safety Platform](https://www.cnbc.com/2026/09/28/nvidia-releases.html) — Sep 28
- [NVIDIA: Africa's AI factories](https://blogs.nvidia.com/blog/egypt-africa-ai-ecosystem/) — Sep 21
- [Kenya–Anthropic Joint Declaration](https://capitalfm.africa/kenya-anthropic-sign-framework-for-responsible-ai-ooperation/) — Sep 24
