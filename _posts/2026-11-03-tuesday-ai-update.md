---
title: "Tuesday AI Update: 3 November 2026 — A frontier model you cannot buy"
date: 2026-11-03 00:00:00 +0300
categories: [AI Engineering, AI in Africa]
tags: [tuesday-update, NOVEMBER-2026, ai-news]
image:
  path: /assets/img/cover-global-ai-roundup-july-2026.webp
  alt: Global AI roundup cover with regional markers
---

Three model releases in eight days set the tone for this week. Google shipped Gemini 4 Carbon on 28 October and handed it to vetted cyber defenders rather than to developers. Mistral published the weights of its trillion-parameter Large 4 the next day. Meituan unveiled a 1.6-trillion-parameter model on Monday that was trained entirely on Chinese accelerators.

Meanwhile the money kept moving: Canada bought equity in Cohere, and an infrastructure investor picked Lagos and Delta State for two AI-ready data centres.

## The week in AI

### Western: cheaper computer use and a state buying equity

- **OpenAI released GPT-6.1 Terra on 26 October** for ChatGPT Work, Codex and the API, aimed at visual reasoning and screen-based computer use, at half the price of the GPT-5.6 Terra it replaces ([OpenAI](https://openai.com/index/introducing-gpt-6-1-terra/), [9to5Mac](https://9to5mac.com/2026/10/26/openai-upgrades-chatgpt-and-codex-with-new-gpt-6-1-terra-model/)).
- **Canada committed up to C$1.6bn of equity to Cohere** as part of a C$7bn round valuing the Toronto lab at C$35.7bn. The terms bind Cohere to 1,000 new Canadian jobs and C$1.7bn of domestic compute by 2030, and keep incorporation and headquarters in Canada ([Prime Minister of Canada](https://www.pm.gc.ca/en/news/news-releases/2026/10/29/canada-announces-landmark-investment-cohere-scale-secure-sovereign-ai), [Globe and Mail](https://www.theglobeandmail.com/business/article-ottawa-buys-stake-in-cohere-357-billion-in-new-financing/)).

### China: frontier-adjacent on domestic silicon

- **Meituan unveiled LongCat-2.5 on 2 November**, a natively multimodal mixture-of-experts model with 1.6 trillion total parameters that activates roughly 48bn per token, built for long-horizon coding agents and trained end to end on domestic accelerators for the second consecutive run ([Caixin](https://www.caixinglobal.com/2026-11-03/meituan-unveils-new-ai-model-trained-on-homegrown-chips-102500305.html), [TechTimes](https://www.techtimes.com/articles/329925/20261103/meituan-releases-longcat-25-chinese-chips-benchmark-claims-await-independent-tests.htm)).
- **The benchmark claim is Meituan's own.** It places LongCat-2.5 between Claude Opus 4.8 and Opus 5 on mainstream evaluations, and no independent evaluator has published a result yet. Meituan spent ¥7.7bn ($1.1bn) on R&D in the second quarter, up 22.5% year on year, and runs the model inside its merchant workbench rather than selling raw compute.

### Europe: open weights, restricted resale

- **Mistral released Large 4 weights on 29 October** in FP8 and NVFP4 under a modified MIT licence: self-hosting and fine-tuning are permitted, model-as-a-service resale needs a separate commercial licence ([Mistral](https://mistral.ai/news/mistral-large-4-oss/), [TNW](https://thenextweb.com/news/mistral-large-4-open-weights-release)).
- **Three more weeks of reinforcement learning moved Terminal-Bench 4.0 from 28% to 41%**, and the model scores 79% on CyberGym-E2E. A quantised build needs four B200 or B300 GPUs, the number to price before planning to self-host.

### MENA: sovereign work keeps pulling in vendors

- **Cisco and Mannai are positioning for Qatar's sovereign AI market** as a $20bn investment push accelerates, with the state's Qai vehicle and its Brookfield joint venture anchoring demand ([Middle East Observer](https://meobserver.news/science-tech/2026/10/29/cisco-mannai-target-qatars-sovereign-ai-market-as-20bn-investment-push-accelerates/)).

### Africa: data centres, resale rights and a film school

- **INFRAGORA Global Capital plans two AI-ready data centres in Lagos and Delta State**, with construction starting in the second quarter of 2027 and $650m to $1bn committed for 60 to 70MW, expandable to 100MW ([Africa AI News](https://www.africaainews.com/p/1b-nigeria-data-centre-project-starts)).
- **Honeywell launched its Forge Plant Cognition platform at Dangote's Lekki refinery**, with preliminary estimates of $60m a year in gross operational benefit at full capacity, about 29 cents a barrel (Reuters, via the same digest).
- **Cassava Technologies became an authorised reseller of Claude through Amazon Bedrock**, giving African enterprises a procurement path to a frontier model without building compute (Cassava Technologies).
- **Google.org put $1m behind a free AI filmmaking programme** run by Idris Elba's Akuna Group: a 12-week academy followed by mentorship and production funding over 50 weeks, open to creators in Kenya, Ghana, Nigeria, Sierra Leone and South Africa until 15 November ([TechTrendsKE](https://techtrendske.co.ke/2026/10/30/google-backed-ai-filmmaking-programme-africa/)).

### South America: courting the compute it wants to keep

- **Brazil's re-elected president wants data centres sited at home**, tying tax breaks and renewable energy supply to capacity built in the country ([pv magazine](https://www.pv-magazine.com/2026/10/27/lula-targets-data-center-investment-using-brazils-renewable-energy-potential/)).
- **The scale is real.** Brazilians send 215 million messages to ChatGPT a day, making the country OpenAI's third-largest market for weekly users after the United States and India ([Fortune](https://fortune.com/2026/11/01/data-centers-brazil-lula-redata-amazon-google-microsoft-openai/)).

## Spotlight: staged access becomes the default

Gemini 4 Carbon tops the Artificial Analysis Intelligence Index at 59, ahead of Claude Opus 5.5 at 58, GPT-6 Astra at 57 and GPT-6.1 Sol at 55. On the security benchmarks it leads with, it scores 88.2% on SEC-bench Pro for finding vulnerabilities, 54.8% on ExploitGym for exploitation and 71.0% on CWE-bench for patching, and it sits fourth on WebDev Arena at 1768 ([Google](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/), [The Hacker News](https://thehackernews.com/2026/10/googles-gemini-4-carbon-leads.html)).

The interesting part is the distribution. Carbon goes first to a subset of cyber defenders and then to trusted partners through Google's Fairwind programme, with no date for developers, businesses or consumers. A paid Gemini subscription does not buy access. Pricing is set at the same introductory $2 per million input and $10 per million output tokens as Argon, with cached input at 95% off, and Google says it is working through the US government's pre-release testing process.

A model that both finds and patches real vulnerabilities is dual-use in the strictest sense, so the lab sells it the way defence contractors sell capability: named customers, vetting, and a public tier that arrives later, if at all.

## Why this matters for Africa

The practical consequence is that frontier capability is now sold in tiers, and the tier an African team can reach is usually the open-weight one. Mistral's licence is genuinely permissive, but four B200s is a capital commitment most local firms cannot carry, so quantisation and sparse activation are the levers that matter. LongCat-2.5 activating 48bn of 1.6 trillion parameters is the same arithmetic from the other direction: it lets a weaker chip set serve a frontier-sized model.

Procurement is the other access path. Cassava reselling Claude through Bedrock, and Core42-style sovereign tenancy in the Gulf, get enterprises to a frontier model without owning silicon. That is a legitimate answer to the access question, and it is cheaper than building.

The binding constraint on hosting is the grid. Nigeria's 60 to 70MW is realistic because it is financeable; Brazil's pitch is explicitly about renewable supply. Any African data centre strategy that quotes capacity without a power contract is a press release, not infrastructure.

**Next in this series:** the roundup returns next Tuesday. For background on how open-weight economics and licence terms shape deployment decisions, see [Open weights: what a download actually costs](/posts/open-weights-download-cost/).

## References

- Google — Introducing Gemini 4 Carbon — https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/
- The Hacker News — Gemini 4 Carbon leads benchmarks for finding, exploiting and patching vulnerabilities — https://thehackernews.com/2026/10/googles-gemini-4-carbon-leads.html
- Caixin Global — Meituan unveils new AI model trained on homegrown chips — https://www.caixinglobal.com/2026-11-03/meituan-unveils-new-ai-model-trained-on-homegrown-chips-102500305.html
- Mistral — Mistral Large 4 is now open weight — https://mistral.ai/news/mistral-large-4-oss/
- Prime Minister of Canada — Canada announces landmark investment in Cohere — https://www.pm.gc.ca/en/news/news-releases/2026/10/29/canada-announces-landmark-investment-cohere-scale-secure-sovereign-ai
- Africa AI News — $1B Nigeria data centre project starts 2027 — https://www.africaainews.com/p/1b-nigeria-data-centre-project-starts
