---
title: "Tuesday AI Update: Sep 22, 2026 — Claude Now Leads 26% of Anthropic's Own R&D"
date: 2026-09-22 00:00:00 +0300
categories: [AI Engineering, AI in Africa]
tags: [tuesday-update, SEP-2026, ai-news]
image:
  path: /assets/img/cover-global-ai-roundup-july-2026.webp
  alt: Globe with glowing AI nodes across seven regions
---

## The Week in AI

September 15–21 answered a question the industry has circled all year: how much of AI research can AI do? Anthropic said a quarter of its own. The same week, Z.ai showed an agent building the production stack for its own model on 100,000-plus Chinese accelerators, and Alibaba open-sourced a CT model that outscored 23 of 26 radiologists in testing.

### Western: An Index, an Agent Ad, a Cheaper Frontier

- **Anthropic's prototype R&D Automation Index** (Sep 17) says Claude "leads" 26% of its model research and development as of August, up from under 1% in February, and works at or above "collaborates" on over 90% of it. About 30,000 agents ran concurrently; of a billion-plus decisions that month, one in 47,000 was blocked. In a sampled July week, 6% of research compute went to safety — 12% when AI led ([Technology Org](https://www.technology.org/2026/09/18/anthropic-claude-leads-26-percent-ai-research/)).
- **OpenAI began piloting Sponsored Agents** (Sep 16): click an ad in ChatGPT and you enter a conversation with an advertiser-funded bot, launched with HubSpot and Shopify in the US ([The Register](https://www.theregister.com/ai-and-ml/2026/09/16/openais-new-sponsored-agents-are-happy-to-chat-about-selling-you-things/5296946)).
- **SpaceXAI shipped Grok 4.7** (Sep 21) on a larger base model and a longer RL run at the same $2/$6 per-million pricing as 4.6 ([MarkTechPost](https://www.marktechpost.com/2026/09/21/spacexai-releases-grok-4-7/)).
- **Google, Nvidia and Emerald AI launched the AI Energy Management Alliance** (Sep 16) so data centres shed load on demand, Google committing 1GW of reducible demand ([DCD](https://www.datacenterdynamics.com/en/news/google-nvidia-and-emerald-ai-found-the-ai-energy-management-alliance-to-support-demand-response-capabilities-within-the-data-center-sector/)).

### China: An Agent Builds the Stack That Serves It

Z.ai's Sep 17 account is the week's most under-read engineering post: an **Infra Agent powered by GLM-5.3** did most of the work building production inference for **GLM-5.3-Flash** (320B total / 18B active, 1M context) on 100,000-plus Chinese accelerators, tripling throughput in under two weeks. It confirmed the anonymous *ox-alpha* test model was GLM-5.3-Flash, which topped OpenCode and OpenRouter within a week. The method was **dense feedback**: folding correctness tests, traces, logs and microbenchmarks into local, cheap, verifiable loops, so the agent kills a hypothesis without a full deployment. Company claims, not independent benchmarks ([Unite.AI](https://www.unite.ai/z-ai-details-glm-5-3-flash-inference-build-on-100-000-chinese-chips/)).

Alibaba's DAMO Academy then **open-sourced RADAR** (Apache 2.0, Sep 18, a day after the paper ran in *Science*): a vision-language model reading contrast-enhanced abdominal CT across 18 organs and flagging 146 findings, trained on 424,911 exams, with a mean AUC of 0.913 ([RuntimeWire](https://runtimewire.com/article/alibaba-damo-radar-open-source-abdominal-ct-ai), [SCMP](https://www.scmp.com/tech/big-tech/article/3368055/alibaba-open-sources-medical-ai-model-can-detect-cancer-and-nearly-150-conditions)).

### Europe, MENA & Russia

Europe: about 190 organisations have signed the Commission's **Code of Practice on Transparency of AI-generated Content** — 82–83 providers and 152 deployers as of Sep 9 — with two task forces launching this month ([European Commission](https://digital-strategy.ec.europa.eu/en/news/strong-backing-code-practice-transparency-ai-generated-content)).

MENA: the US Commerce Department **cleared export of up to 70,000 advanced AI chips** — 35,000 Nvidia GB300-equivalents each — for Abu Dhabi's G42 and Saudi's HUMAIN ([Middle East AI News](https://www.middleeastainews.com/p/us-approves-up-to-70000-advanced)). Both court outside capital: HUMAIN is recruiting an IPO preparation team and raising $2.5bn; G42 weighs external investors ([AGBI](https://www.agbi.com/ai/2026/09/gulf-ai-giants-humain-and-g42-look-to-raise-outside-capital/)).

Russia: its first AI law took effect Sep 1, making "technological independence" and overseas promotion national goals ([Atlantic Council](https://www.atlanticcouncil.org/dispatches/russia-wants-to-sell-sovereign-ai-abroad-heres-what-its-really-selling/)). Bloomberg's Sep 16 read: Russia lags on frontier models but co-opts them well — one publisher trained an agent to flag book passages that risk breaking the law ([Japan Times](https://www.japantimes.co.jp/news/2026/09/16/world/politics/ai-putin-goals-ambitions/)).

### Africa: Fund, Train, Translate, Build

- **Egypt and Intel signed an MoU** (Sep 16) to train a million citizens a year in AI for three years, including 1,500 certified trainers ([Africa AI News](https://www.africaainews.com/p/egypt-intel-to-train-1m-in-ai-annually)).
- **22 On Sloane launched KUMii**, an AI founder-matching platform, alongside Sloane Capital's R1bn (~$63m) fund ([TechCabal](https://techcabal.com/2026/09/16/22-on-sloane-launches-kumii/)). **Askya** takes 10 AI startups through a six-week zero-equity programme worth up to $200,000 each ([Disrupt Africa](https://disruptafrica.com/2026/09/02/askya-ai-growth-platform-offers-200k-zero-equity-funding-to-african-startups/)).
- Funding: **Janguru raised $25m** for African-language translation, **Synapse Analytics $13m**, and Nigeria's **Aeon** $1m pre-seed led by Terra Industries (Africa AI News).
- Infrastructure: the US DFC approved a record **$155m equity investment in WIOCC** (Sep 16) ([CNBC Africa](https://www.cnbcafrica.com/2026/us-agency-to-invest-155-million-in-african-digital-infrastructure-firm-wiocc)); **Ethiopia** is pitching its ~96% hydropower mix at data-centre investors, while Cape Town rights groups push back on proposed Equinix hyperscale sites as Africa's compute demand heads for 2.2GW by 2030.

### South America

The **IDB** (Sep 21) estimates broad AI adoption could leave Latin America and the Caribbean 5.1% larger after a decade — but wages could fall by up to 20.9% if workers cannot move into expanding roles, versus +5.3% with mobility ([Reuters](https://www.investing.com/news/economic-indicators/ai-could-boost-latam-caribbean-economy-51-but-wages-may-fall-idb-says-4909514)).

## Spotlight: The Bottleneck Moved to Verification

Read Anthropic's index beside Z.ai's write-up and the same constraint appears. Anthropic's 26% only means something because the control surface came with it — one in 47,000 decisions blocked, safety compute roughly doubling where AI led. Z.ai's agent worked because the team built local, cheap, verifiable feedback first: a correlation is not a root cause. Neither result is a smarter model; both are better checks — and Anthropic calls its index a prototype.

## Why This Matters for Africa

Both stories attack the cost of engineering labour — Africa's binding constraint. But this week's African news puts the harder constraint downstream: WIOCC's $155m, Ethiopia's hydropower pitch and Cape Town's protests are arguments about power, water and land. The leverage sits in what shipped open: GLM-5.3-Flash, and an Apache-2.0 CT model that flags 146 findings from one scan where radiologist density is low — how a small Nairobi or Kigali team competes without a 100,000-chip cluster.

## References

- [Anthropic: Claude leads 26% of AI R&D](https://www.technology.org/2026/09/18/anthropic-claude-leads-26-percent-ai-research/) — Sep 17
- [Z.ai: inference on 100,000+ Chinese accelerators](https://www.unite.ai/z-ai-details-glm-5-3-flash-inference-build-on-100-000-chinese-chips/) — Sep 17
- [Africa AI News weekly digest](https://www.africaainews.com/p/egypt-intel-to-train-1m-in-ai-annually) — Sep 18

**Related:** [Tuesday AI Update, Sep 15](/posts/tuesday-ai-update/) · [RAG Recall@K: The Denominator Nobody Checks](/posts/rag-recall-at-k-denominator/) · [Constrained Decoding: Token Masks in Practice](/posts/constrained-decoding-token-mask/)
