---
title: "Ten African Startups, One Constraint Map"
date: 2026-11-13 00:00:00 +0300
categories: [AI in Africa, LLM]
tags: [africa, startups, google, on-device, gemma, applied-ai]
image:
  path: /assets/img/cover-google-africa-applied-ai-lab-cohort.webp
  alt: Ten startup tiles pinned to a map grid, grouped into four columns by the constraint each company removes
---

On 11 November Google named the ten companies in the first cohort of its Google Africa Applied AI Lab. The programme was announced in July, took applications from 1 July to 31 August, and by its own account drew thousands of entries from startups in more than 45 African countries before cutting to ten. [Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/applied-ai-lab-cohort-2026/)

A cohort is the closest thing the sector has to a public sample. Everybody who applied wanted the same thing, everybody was filtered by the same criteria, and the list that survived shows what the filter rewarded. Read that way, the ten names are less a league table than a map of where applied AI in Africa currently has customers.

> **The short version**
> The ten selected companies sell voice interfaces, identity checks, payment rails, insurance infrastructure, offline inference and orbital compute. Almost none of them sells a model. The thing they have in common is a distribution answer, which is also the thing a benchmark score cannot give you.
{: .prompt-info }

## What the lab offers, and what it does not

The lab sits at the Accra AI Community Centre in Ghana, the space Google opened in July 2025 with a $37 million commitment attached to it. ([AmCham Ghana](https://amchamghana.org/2025/07/31/google-launches-ai-community-center-in-ghana-with-37m-investment-deepening-u-s-africa-tech-collaboration/)) It is managed by the Google AI Futures Fund alongside Google Research, and the programme page describes it as a zero-to-one commercialisation platform, pairing a cohort of founders directly with Google engineers. ([Lab FAQ](https://labs.google/aifuturesfund/africaailab))

What a team receives is specific and mostly non-cash: early access to Google DeepMind models, named on the FAQ as Gemini, Gemma and Veo, before general release; hands-on technical mentorship; and go-to-market guidance from Google teams plus four venture firms, 4DX Ventures, Norrsken22, Novastar Ventures and Ventures Platform. Some participants may also be eligible for funding from the AI Futures Fund or those VCs.

What a team gives is narrower than most programmes ask. It has to use Google's AI in some material way, and it has to show up in person for a demo day in Accra in early December. The FAQ is explicit that this is not an exclusivity clause and that teams can use other models alongside Google's. It is equally explicit that teams "are not giving up equity or ownership stake by simply participating", though they may be approached with investment offers.

| Item | What the sources say |
|---|---|
| Cohort size | 10 startups |
| Applications | Thousands, from more than 45 African countries |
| Co-development window | Mid-September to early December 2026 |
| Model access | Gemini, Gemma, Veo before general release |
| Mentorship | Google engineers, research scientists, product leaders |
| Capital partners | AI Futures Fund, 4DX Ventures, Norrsken22, Novastar Ventures, Ventures Platform |
| Equity taken | None for participating; funding offers are separate |
| Funding amount | Not disclosed for any team |
| Second cohort | Not announced |

Two details are worth separating from the announcement language. The first is that no cheque size has been published, for anyone. Tech-ish read the programme page and concluded that what a startup walks away with is "mostly access", with money a maybe. ([tech-ish](https://tech-ish.com/2026/11/11/google-picks-10-startups-for-its-first-africa-applied-ai-lab/)) TechBooky put the same point more bluntly: the cohort "should not be described as ten funded deals of a known size". ([TechBooky](https://www.techbooky.com/google-picks-10-startups-for-its-first-africa-applied-ai-lab/))

The second is a correction to earlier coverage, including ours. When the lab opened in July, the roundup on this blog described a cohort of fifteen startups receiving equity-free funding. The FAQ now describes ten teams and conditional funding, with terms that "will differ from team to team". Announcement copy written on day one is not a term sheet, and the FAQ is the better document to quote.

## The ten, grouped by the constraint they remove

Google's own list is alphabetical, which hides the structure. Sorted by what each company takes off a customer's plate, four groups appear.

| Startup | Listed base | Product | Constraint it removes |
|---|---|---|---|
| Smile ID | Pan-African | Computer vision for identity verification, onboarding customers for financial services | Knowing who the customer is |
| Termii | Nigeria | Customer verification, messaging and operational alerts for financial services | Reaching the customer at the right moment |
| Vendy | Nigeria | Storefronts and payments inside WhatsApp | Owning a shop without a website |
| AethexAI | Pan-African | AI voice agents for banks, telecoms providers and support centres | Staffing a call centre |
| Intron | Nigeria | Clinical and enterprise speech recognition for African accents and code-switched speech | Being understood when you switch languages mid-sentence |
| Viamo | Pan-African | Generative AI over ordinary phone calls in local languages | Using AI without a smartphone or data |
| Agrails | Pan-African | AI-powered insurance infrastructure for farmers, gig workers and small borrowers | Getting cover priced and paid out |
| Tolbi | Senegal | Satellite imagery plus voice guidance for smallholder farmers | Knowing what the field needs this week |
| Crane AI Labs | Uganda | Offline, on-device AI for health workers and schools on affordable phones | Working with no connection |
| Satlyt | Kenya | Satellite software that processes data in orbit | Sending raw data to the ground at all |

Count the rows by group and the shape is clear. Three companies sit on the rails that money and identity move along. Three remove an interface cost, because their users speak a language mix that off-the-shelf speech systems handle badly or hold a handset that cannot run an app. Two remove a cost in the physical economy, where the output has to be a payout or a farming decision. Two remove a compute-location cost by pushing inference to a phone or to orbit.

### The rails: identity, verification, messaging, payments

Smile ID, Termii and Vendy all sell to businesses that already have customers and already pay for infrastructure. Smile ID's Kenyan page, cited by tech-ish, says its checks can run against IPRS, the government population register, as well as eCitizen and the Business Registration Service. Termii sells verification and operational messaging into financial services. Vendy turns WhatsApp into a shopfront with payments attached.

None of these is an AI research problem at this point. They are integrations, uptime and fraud economics. The value a model adds here is incremental and cumulative: a better face match or a better delivery window improves a service whose customers renew monthly. That is the least glamorous row in the table and probably the most durable business in it.

### The interfaces: voice in, voice out

Intron transcribes clinical and enterprise speech in African accents and across language switches. Viamo reaches people who have no smartphone at all, using ordinary voice calls in local languages. AethexAI runs voice agents for banks, telecoms providers and support centres.

Google's announcement opens with three scenes drawn from the cohort: a farmer in Senegal receiving a voice message about watering and harvest timing, a nurse in a rural Ugandan clinic checking symptoms on a cheap phone with no internet, and a woman in Nigeria dialling a number, asking a question in Hausa and hearing the answer read back. ([Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/applied-ai-lab-cohort-2026/))

Voice is a different evaluation regime from text. Accented speech, telephone-band audio, code-switching mid-sentence and background noise in a clinic are the operating conditions, and a word error rate measured on clean read speech says very little about them. The interesting engineering in this group is not the model choice, it is the data and the fallback path when confidence is low.

### The physical economy: insurance and agronomy

Agrails builds insurance infrastructure for farmers, gig workers and small borrowers. Tolbi combines satellite imagery with voice guidance for smallholder farmers.

Both depend on something outside the model: a measurable trigger that pays out, or an agronomic recommendation specific enough to act on. Index insurance lives or dies on basis risk, the gap between what the index says happened and what happened to the farmer's field. A model that predicts yield well can still produce a product nobody trusts if the payout trigger is wrong. That is a data and product-design problem, and it is why these two companies look more like underwriters than like app developers.

### The edge: a phone with no connection, a satellite with no downlink

Crane AI Labs ships offline on-device tools for health workers and schools on affordable phones. Satlyt runs software on satellites so that data can be processed in orbit and only findings come down.

These are the two rows that invert the cloud default. Everyone else in the cohort assumes a network hop somewhere; these two assume the hop is the expensive part. Satlyt owns no satellites, per tech-ish: its software runs on hardware other operators have already launched, and the same article reports that the company said in October it had not yet demonstrated multi-satellite compute sharing in orbit. The company raised an $8 million seed in October, which this blog covered at the time.

Read the four groups together and one line runs through them. Eight of the ten are constraints on reach rather than on reasoning: who the user is, which channel the customer already uses, what language they switch into, whether the phone has a connection, whether the data can leave the ground. The two exceptions, Agrails and Tolbi, are constraints on deciding well, where the output has to survive a payout trigger or a planting date. The lab's FAQ lists five thematic futures it wants to prototype for: the future of work, knowledge, software development, creativity and entertainment. The cohort that arrived is about identity, insurance, agronomy, clinics and orbit. What the filter rewarded, in the end, was a named distribution channel attached to a real constraint.

## The money question nobody disclosed

The capital part of the announcement is the part with the least information in it. Four venture firms are named as partners, some participants may be eligible for funding, and nothing else is quantified.

Set that against what African AI fundraising looks like this year. Launch Base Africa's deal-level dataset, published on 3 November, counts roughly 30 AI-labelled transactions out of more than 200 tracked for 2026, with combined disclosed value of about $93 million. The median disclosed AI round in that dataset is about $1.9 million and the average about $4.5 million, and there is no pure-AI Series B or Series C anywhere in it. One company, South Africa's Healthleap, accounts for about 41 percent of the disclosed AI total. ([Launch Base Africa](https://launchbaseafrica.com/2026/11/03/south-africa-and-egypt-contest-africas-ai-funding-crown/))

That context reframes what "venture capital backing" means in a cohort announcement. The same dataset lists Ventures Platform and Norrsken22 among the backers of AethexAI, and TLcom and Enza in Satlyt's round. Two of the four VC partners in the lab are therefore already on the cap table of at least one selected company. That is not a criticism, since programmes naturally recruit investors who know the companies, but it does mean the partner list is partly a description of existing relationships.

The practical consequence for a founder reading this announcement: treat the lab as a technical and procurement event rather than a funding event. The scarce inputs it offers are engineering time and pre-release model access, and those have real value in a market where the typical disclosed cheque is under $2 million and the number of AI rounds is small enough to count. If you are choosing between programmes, the question is not which one has the biggest logo on the page. It is which one shortens the distance between your prototype and a paying customer, and by how much.

## What an offline model costs on a phone

Crane AI Labs' one-line description is a hardware requirement in disguise: "offline, on-device AI tools for health workers and schools on affordable phones". Offline and affordable together define a memory budget, and the budget decides which models are even candidates.

The arithmetic is simple, and worth doing because "runs on a phone" is a claim that often survives a demo and dies in a clinic. Weights at 4-bit cost half a byte per parameter plus a scale factor, about 0.5625 bytes per parameter with one fp16 scale per 32 weights. The KV cache costs two tensors per layer per token, and it grows with context rather than with the model.

```python
"""Memory budget for an on-device model on a low-cost phone.

Assumptions, all explicit:
  - 4-bit weights cost 0.5 bytes/parameter plus an fp16 scale per 32 weights
    (2 bytes / 32 = 0.0625), so 0.5625 bytes/parameter.
  - 8-bit weights cost 1.0 + 0.0625 = 1.0625 bytes/parameter.
  - KV cache = 2 (K and V) * layers * kv_heads * head_dim * bytes per element.
  - Usable memory is what is left for the app: 1.5 GiB on a 2 GB phone,
    2.5 GiB on a 3 GB phone.
"""
GiB = 1024 ** 3
MiB = 1024 ** 2

MODELS = [
    # name, parameters (billions), layers, kv heads, head dim
    ("0.27B", 0.27, 18, 4, 128),
    ("1B", 1.0, 26, 8, 128),
    ("2B", 2.0, 34, 8, 128),
    ("4B", 4.0, 40, 8, 128),
]
QUANT = [("4-bit", 0.5 + 0.0625), ("8-bit", 1.0 + 0.0625)]
BUDGETS = [("2 GB phone", 1.5), ("3 GB phone", 2.5)]


def kv_bytes_per_token(layers, kv_heads, head_dim, elem_bytes=2):
    return 2 * layers * kv_heads * head_dim * elem_bytes


def max_context(params_b, bytes_per_param, layers, kv_heads, head_dim, budget_gib):
    weights = params_b * 1e9 * bytes_per_param
    headroom = budget_gib * GiB - weights
    if headroom <= 0:
        return 0
    return int(headroom // kv_bytes_per_token(layers, kv_heads, head_dim))


print("weights and KV cache for four small open models, fp16 KV cache")
print(f"{'model':>6} {'quant':>6} {'weights':>10} {'KV/1k tok':>10} "
      f"{'ctx@1.5GiB':>11} {'ctx@2.5GiB':>11}")
for name, params, layers, kv_heads, head_dim in MODELS:
    for quant, bpp in QUANT:
        w = params * 1e9 * bpp
        kv1k = kv_bytes_per_token(layers, kv_heads, head_dim) * 1000
        c15 = max_context(params, bpp, layers, kv_heads, head_dim, 1.5)
        c25 = max_context(params, bpp, layers, kv_heads, head_dim, 2.5)
        print(f"{name:>6} {quant:>6} {w/MiB:9.0f}M {kv1k/MiB:9.1f}M {c15:>11,} {c25:>11,}")

# Where does the ceiling actually sit? Compare KV cost per token against the
# weight file for the 2B model, the size class a 4-bit phone deployment targets.
params, layers, kv_heads, head_dim = 2.0, 34, 8, 128
weights = params * 1e9 * (0.5 + 0.0625)
kv16 = kv_bytes_per_token(layers, kv_heads, head_dim)
kv8 = kv_bytes_per_token(layers, kv_heads, head_dim, elem_bytes=1)
print()
print(f"2B at 4-bit: weights {weights/GiB:.2f} GiB")
print(f"2B KV per token: fp16 {kv16/1024:.1f} KiB, int8 {kv8/1024:.1f} KiB")
print(f"tokens of context bought by 100 MiB of RAM: fp16 {100*MiB//kv16:,}, int8 {100*MiB//kv8:,}")
print(f"a 4k-token context in fp16 costs {4096*kv16/MiB:.0f} MiB, int8 {4096*kv8/MiB:.0f} MiB")
```

Run on this machine, the script prints the following. The layer and KV-head counts are stand-ins for four size classes, not a claim about any specific released model.

```text
weights and KV cache for four small open models, fp16 KV cache
 model  quant    weights  KV/1k tok  ctx@1.5GiB  ctx@2.5GiB
 0.27B  4-bit       145M      35.2M      39,570      68,697
 0.27B  8-bit       274M      35.2M      35,908      65,035
    1B  4-bit       536M     101.6M       9,841      19,924
    1B  8-bit      1013M     101.6M       5,146      15,229
    2B  4-bit      1073M     132.8M       3,486      11,197
    2B  8-bit      2027M     132.8M           0       4,016
    4B  4-bit      2146M     156.2M           0       2,651
    4B  8-bit      4053M     156.2M           0           0

2B at 4-bit: weights 1.05 GiB
2B KV per token: fp16 136.0 KiB, int8 68.0 KiB
tokens of context bought by 100 MiB of RAM: fp16 752, int8 1,505
a 4k-token context in fp16 costs 544 MiB, int8 272 MiB
```

Three ceilings show up in that table, and they bind in a specific order.

The first is the weight file. A 4B model at 8-bit needs about 4 GiB before it has read a single token, which is more than a 3 GB phone has available to any application. A 4B model at 4-bit fits on a 3 GB phone and leaves about 410 MiB for everything else, which buys about 2,651 tokens of context. That is a consultation summary, not a patient record.

The second is the KV cache, and it is the one teams underestimate. At 2B and 4-bit, weights take 1,073 MiB of the 1,536 MiB budget on a 2 GB phone. The remaining 463 MiB holds 3,486 tokens at fp16. Moving the cache to int8 doubles that to about 7,000, which is the cheapest single win available for a fixed weight file. No amount of prompt engineering buys back the difference between 3,486 and 7,000 tokens.

The third is the operating system. The 1.5 and 2.5 GiB figures in the script are assumptions about headroom, not specifications, and the honest move is to measure them on the actual device. Android will kill an app that overshoots, and a health worker does not get to see an out-of-memory stack trace.

For a model running at 0.27B parameters, memory stops being the binding constraint at all and the model's own context limit takes over. That is the config that makes an offline symptom checker plausible on the cheapest hardware, at the cost of reasoning ability. Whether that trade is acceptable is an evaluation question, and it has to be answered on the device, at the context length the product ships with.

This is the part of the cohort that a demo day cannot show. A model that answers well from a laptop with a 32k context window may be serving a product that ships with 3,500 tokens available and a queue of patients behind it.

## How to use this

For founders, the useful lesson from the ten is about framing, not technology. Every selected company can state the constraint it removes in one clause and can name who pays to have it removed. That is the sentence a selection committee can act on, and it is missing from most applications, which describe model capability instead of customer pain.

If the lab runs a second cohort, which Google has not announced, a few questions are worth asking before accepting a place:

| Question | Why it matters |
|---|---|
| What exactly do we get in engineering hours, and from whom? | Access to researchers is the scarce input; a Slack channel is not |
| What does "material use of Google AI" cost us at production volume? | Model access is early, not free; unit economics survive or die here |
| Which model versions are we allowed to ship, and for how long? | A pre-release model can be withdrawn; a product cannot |
| Who owns what we build jointly? | The FAQ settles equity for participation, not for co-development |
| What is the demo day for, a fundraise or a showcase? | The answer sets the preparation work for November and December |

For everyone else, the cohort is a set of things to watch rather than a result to celebrate. Three measurable ones: how many of the ten disclose a priced round in the next twelve months and who leads it; whether any of them ships the product into a second country with a different language mix; and whether any publishes a before-and-after on accuracy, cost or reach that can be attributed to the programme. TechBooky's framing is right, that the evidence which matters comes from product results and not from the selection announcement.

Two limits belong in the same paragraph as the optimism. Ten companies chosen by one company is not a sample of a sector, and the thousands that were not selected are invisible in this data. And Google's stated aim, building "Africa's first generation of AI-native unicorn startups", is a goal on a programme page, not an outcome anyone can verify in November.

What is verifiable is narrower and still worth noting. Ten teams that were already selling into clinics, banks, farms and telcos now have engineering time with the people who build the models they depend on, and a December deadline to show something. In a market where the median disclosed AI round is under $2 million, that is not a small thing to hand a company.

## Key takeaways

| Point | Detail |
|---|---|
| What was announced | Ten startups in the first Google Africa Applied AI Lab cohort, chosen from thousands of applications across 45+ countries |
| What they get | Early access to Gemini, Gemma and Veo, technical mentorship, go-to-market guidance, no equity given up |
| What was not disclosed | Any funding amount, how many teams will be funded, whether a second cohort follows |
| The cohort's shape | Three rails in financial services, three voice or speech interfaces, two in the physical economy, two at the edge |
| The common thread | A distribution answer attached to a constraint, not a model claim |
| The money context | About $93 million disclosed across roughly 30 AI deals in Africa this year, median round about $1.9 million |
| The engineering constraint | On a 2 GB phone, a 2B model at 4-bit leaves about 3,500 tokens of context; int8 KV roughly doubles it |

## References

- Google Africa Blog, "Meet the 10 startups joining the first Google Africa Applied AI Lab cohort", 11 Nov 2026 — [blog.google](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/applied-ai-lab-cohort-2026/)
- Google AI Futures Fund, "Google Africa Applied AI Lab" programme FAQ — [labs.google](https://labs.google/aifuturesfund/africaailab)
- tech-ish, "Google picks 10 startups for its first Africa Applied AI Lab", 11 Nov 2026 — [tech-ish.com](https://tech-ish.com/2026/11/11/google-picks-10-startups-for-its-first-africa-applied-ai-lab/)
- TechBooky, "Google Picks 10 Startups for Its First Africa Applied AI Lab", 12 Nov 2026 — [techbooky.com](https://www.techbooky.com/google-picks-10-startups-for-its-first-africa-applied-ai-lab/)
- American Chamber of Commerce Ghana, "Google Launches AI Community Center in Ghana With $37M Investment", 31 Jul 2025 — [amchamghana.org](https://amchamghana.org/2025/07/31/google-launches-ai-community-center-in-ghana-with-37m-investment-deepening-u-s-africa-tech-collaboration/)
- Launch Base Africa, "South Africa and Egypt Contest Africa's AI Funding Crown", 3 Nov 2026 — [launchbaseafrica.com](https://launchbaseafrica.com/2026/11/03/south-africa-and-egypt-contest-africas-ai-funding-crown/)
- Business Tech Africa, "Google Selects 10 African Startups for Its First Applied AI Lab" — [businesstechafrica.co.za](https://www.businesstechafrica.co.za/article/google-selects-10-african-startups-for-its-first-applied-ai-lab)

## Related posts

- [Africa's AI Spring: Labs, Capital and the Applied Turn](/posts/africa-ai-spring/)
- [Enterprise AI Decisioning in Africa](/posts/africa-enterprise-ai-decisioning/)
- [Offline-First AI for Kenyan Schools](/posts/kenya-ai-academy-offline-first/)
- [Open Weights at the TB Screening Front Line](/posts/open-weights-tb-screening-front-line/)
- [A Crop Model Across Seven Countries](/posts/crop-model-seven-countries/)
