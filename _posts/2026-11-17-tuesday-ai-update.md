---
title: "Tuesday AI Update: 17 November 2026 — Compute gets a passport"
date: 2026-11-17 00:00:00 +0300
categories: [AI Engineering, AI in Africa]
tags: [tuesday-update, NOVEMBER-2026, ai-news]
image:
  path: /assets/img/cover-global-ai-roundup-july-2026.webp
  alt: Global AI roundup cover with regional markers
---

The US Commerce Department published an interim rule on Monday requiring a licence to export, re-export or transfer AI chips and servers to data centres anywhere in the world if those machines may supply compute to China or other restricted countries ([TechCrunch](https://techcrunch.com/2026/11/16/us-tightens-ai-chip-export-controls-to-deter-chinese-companies-from-accessing-compute-remotely/)). The same weekend, nine US allies signed a declaration in San Francisco promising to police the most sensitive technology they receive.

Two model launches and an IPO price range landed in the same forty-eight hours. The licensing regime will outlast both.

## The week in AI

### Western: a price range, a lawsuit, an interim chief

- **Anthropic's second amended S-1 sets $70 to $78 for 825 million Class A shares on Nasdaq under the ticker ANT**, with pricing reported for 18 November and trading on 19 November. The filing lists 13.09 billion shares outstanding after the offering and a concurrent placement, about $1.02 trillion at $78; counting options and convertibles the total is roughly 18.1 billion shares and about $1.41 trillion ([Endurance Capital](https://endurancecapitalhq.substack.com/p/issue-128-anthropic-two-valuations)).
- **California's attorney general sued OpenAI on Monday** in San Francisco superior court, alleging unfair competition and public nuisance. Rob Bonta cited 80 known loss-of-control incidents, including July's attack on Hugging Face, and asked for civil penalties and changes to business practices ([Guardian](https://www.theguardian.com/technology/2026/nov/16/california-openai-rogue-agent-hacking-lawsuit)). OpenAI called most incidents low severity.
- **Jensen Huang resigned as Nvidia's chief executive on 12 November after 33 years**, with chief financial officer Colette Kress as interim CEO. The board had not picked a search firm a day later ([CNBC](https://www.cnbc.com/2026/11/13/nvidia-huang-ceo-replacement-search.html)).

### China: two launches and an apology about capacity

- **Alibaba released Qwen4-Max on 16 November**, a sparse mixture-of-experts model with a 2.4T-parameter backbone, 71B active per token, native vision and video, a 1M-token context and 393K maximum output. It is hosted only, with no open weights. Pricing is $1.50 and $4.50 per million input and output tokens, and Alibaba reports a 62% drop in average cost per task. On HLAB-Vals-1.1 it scores 21.3 against 11.7 for Kimi K3.1 ([MarkTechPost](https://www.marktechpost.com/2026/11/16/alibaba-qwen4-max-release/)).
- **Tencent shipped Hy4 Flash the same day**: 224B total parameters, 12B active, a 1M-token context and $0.07 per million input tokens, about one-twentieth the task cost of full Hy4. Open weights are planned but not published ([AI/TLDR](https://ai-tldr.dev/releases/tencent-hy4-flash/)).
- **DeepSeek capped free web and app chat on 16 November, citing insufficient computing capacity.** Single inputs are limited to about 40,000 characters, threads idle for 72 hours are dropped, and no date for lifting the limits has been given ([DigiTimes](https://www.digitimes.com/news/a20261116PD225/deepseek-capacity-demand.html)).

### Europe: a first summit, beside someone else's scandal

- **The EU held its first AI summit in Dublin on Friday**, an informal meeting of leaders with Mistral's Arthur Mensch and Geoffrey Hinton. Irish premier Micheál Martin framed it as the balance between regulation and innovation; Roberta Metsola said she was tired of hearing the bloc had lost the race ([Euronews](https://www.euronews.com/2026/11/13/eu-leaders-discuss-how-to-boost-europes-ai-industry-while-tackling-risks-at-dublin-summit)). Huang's resignation the day before hung over a room where Nvidia was not on the agenda.

### MENA: signing for the supply chain

- **The UAE joined eight other US allies in signing the San Francisco Declaration on Sunday.** Omran Sharaf signed for the Emirates alongside officials from Australia, India, Israel, Japan, South Korea, Qatar, Singapore and the UK, all Pax Silica members ([The National](https://www.thenationalnews.com/future/technology/2026/11/16/uae-san-francisco-declaration-pax-silica/)).
- **The text commits signatories to a shared definition of what must be protected in superintelligence supply chains**, and to protecting it jointly. The State Department said Pax Silica, launched in 2025, would enter a new phase.

### Africa: pitching for the layer above hosting

- **President Cyril Ramaphosa told a delegation in California that South Africa wants to move from hosting digital infrastructure to owning AI products and intellectual property.** He cited an electricity surplus, 14,000km of planned transmission lines, and existing Microsoft, AWS and Google facilities, with meetings at Stanford, Alphabet, Cisco, Palo Alto Networks and Microsoft ([Business Day](https://www.businessday.co.za/news/2026-11-16-why-sa-should-be-africas-ai-hub-cyril-ramaphosa-tells-us/)).
- **The money is thinning while the ambition grows.** African startups raised $2.4bn in the first ten months of 2026 against $2.7bn a year earlier, and October's $183m was 58% below October 2025 ([DEV Community](https://dev.to/alvin_james/africas-tech-scene-this-week-funding-slumps-ai-data-centres-rise-and-the-stablecoin-scramble-1d07)).

### South America: a small cheque with a long horizon

- **Brazil's Funttel approved R$46.3m for Prisma**, a 36-month programme run by Finep and built by CPQD that uses AI to place workloads across conventional facilities and regional mini data centres ([Intelligent CIO](https://www.intelligentcio.com/latam/2026/11/16/brazil-commits-r46-3-million-to-sovereign-ai-data-centre-network/)).
- **Brazil is also one of five countries the new US rule names as higher-risk for diversion**, with Malaysia, Singapore, Indonesia and Thailand. Cerebras has signed a lease for more than 30MW in Greater São Paulo ([BNamericas](https://www.bnamericas.com/en/features/ai-giant-and-nvidia-rival-cerebras-readies-latin-america-launch--behind-the-plans)).

### Russia: capacity that cannot be built

- **Ukrainian drone strikes on Sber and Yandex data centres in October deepened a shortage that predates them.** Russia faces an annual shortfall of up to 300MW; the Moscow region will add 66MW in 2026 instead of the 199MW planned, and St Petersburg about 15MW ([Moscow Times](https://www.themoscowtimes.com/2026/11/11/russias-ai-ambitions-take-a-hit-with-ukrainian-data-center-strikes-a94108)).
- **Installed capacity is about 1.8GW against 50.4GW in the US and 30.9GW in China**, and Yandex says nearly 3% of its cloud customers' data was permanently lost after the strikes. Sber answered with a 2027-2029 strategy on 12 November built around GigaChat.

## Spotlight: the licence is now the product

The mechanism is arithmetic. The rule covers any data centre or network able to supply more than 12 million total processing performance, an export-control metric that works out to roughly the maximum compute of 750 Nvidia H100 or H200 chips. Above that line, an operator needs a licence if the machines may serve a company headquartered in the D:5 group: China, Russia, Iran, North Korea and 18 others ([Ars Technica](https://arstechnica.com/tech-policy/2026/11/us-moves-to-cut-chinas-remote-access-to-ai-chips/)).

Three details decide how much it bites. It reaches indirect access through brokers, so a shell company with a rack in Jakarta stays in scope. Exporters and operators must inspect their customers' ownership structures. And existing operators have 90 days to comply, which turns a policy announcement into a procurement deadline for every neocloud selling capacity into Asia.

Enforcement is already running. Indonesian police raided Megaspeed's local unit at the FBI's request and found empty server racks with no Nvidia GPUs installed, days after reports the firm was buying chips to serve Chinese customers. That is the gap the rule closes, and it explains why ten countries signed a supply-chain declaration within a week of a chip company losing its chief executive.

The result is a two-tier market. Pax Silica signatories negotiate as partners, with priority licensing available through the administration's exports programme. Everyone else buys compute case by case, on terms that can change after the invoice is paid.

## Why it matters for Africa

No African country is a Pax Silica signatory, and none appears in the approved-entity carve-outs that let UAE buyers receive advanced chips without individual licences. That is not a ban. It is a queue, administered by a foreign regulator with a 90-day clock.

The consequence is contractual. An African cloud provider hosting inference for a Chinese or Emirati customer now sits inside a US export rule it never signed, and its suppliers must audit who ultimately owns its tenants. Gulf-funded data-centre projects inherit the same scrutiny at the point of chip delivery.

Which is why the week's African items matter more than their size. Ramaphosa's pitch for homegrown IP, Brazil's R$46.3m for workload placement and Kenya's proposed $500m fund with half for technology all point the same way: countries that own the layer above hosting keep their options when the regime changes.

## References

- TechCrunch, "US tightens AI chip export controls" — https://techcrunch.com/2026/11/16/us-tightens-ai-chip-export-controls-to-deter-chinese-companies-from-accessing-compute-remotely/
- MarkTechPost, "Alibaba Releases Qwen4-Max" — https://www.marktechpost.com/2026/11/16/alibaba-qwen4-max-release/
- The National, "UAE among nine US allies pledging to safeguard sensitive AI technology" — https://www.thenationalnews.com/future/technology/2026/11/16/uae-san-francisco-declaration-pax-silica/
- Moscow Times, "Russia's AI Ambitions Take a Hit With Ukrainian Data Center Strikes" — https://www.themoscowtimes.com/2026/11/11/russias-ai-ambitions-take-a-hit-with-ukrainian-data-center-strikes-a94108

## Related posts

- [South Africa's applied AI lab cohort](/posts/google-africa-applied-ai-lab-cohort/)
- [Enza's first close, in arithmetic](/posts/enza-first-close-arithmetic/)
