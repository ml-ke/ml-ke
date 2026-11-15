---
title: "A First Close Is Not a Cheque: The Arithmetic of a $21M African AI Fund"
date: 2026-11-15 00:00:00 +0300
categories: [AI in Africa, AI Engineering]
tags: [venture capital, africa, fund mechanics, ai startups, funding, enza]
image:
  path: /assets/img/cover-enza-first-close-arithmetic.webp
  alt: A committed-capital bar split into fee, opex, reserve and primary slices above a grid of cheque slots with eleven lit
---

> **The short version**
> On 11 November Enza Capital, a Nairobi-founded pan-African fund, reached a first close of about $21 million on a $60 million AI Africa Fund, 35% of target. That is a commitment, not a deployment: the firm expects a final close in 2027 and first investments in the same year. A decade of management fees and operating costs removes about 23% of it before a single cheque clears, and a follow-on reserve halves what is left. At the fund's own sweet spot of $750,000 per first cheque, the first close buys roughly eleven companies, not the eighty-four that the $250,000 floor implies. At the full $60 million target it buys about thirty-one.
{: .prompt-info }

Funding announcements are written as if the money had already been spent. A first close of $21 million reads like $21 million in portfolio companies. It is not even $21 million in the fund's bank account. It is a set of signed commitments from limited partners, callable over the next several years, minus a fee that starts accruing immediately.

That gap between the headline and the cheque is arithmetic, and the arithmetic is public enough to run. The terms below come from the two announcement write-ups at body level and from the firm's own investor FAQ. The fee schedules are market conventions, labelled as such, because Enza did not disclose its terms. The point of the exercise is not to second-guess one firm. It is to give readers a way to price the next announcement they see, and this one is worth pricing because the fund was raised in a year when disclosed AI funding on the continent fell by more than 90%.

## What was announced

Enza Capital reached the first close of its third fund, Enza Capital III LP, at about $21 million, or 35% of a $60 million target, with a hard cap of $80 million. The firm says it expects further commitments before the end of 2026, a final close in 2027, and first investments in 2027. ([Disrupt Africa](https://disruptafrica.com/2026/11/13/pan-african-vc-firm-enza-capital-reaches-21m-first-close-of-ai-africa-fund/), [Launch Base Africa](https://launchbaseafrica.com/2026/11/11/enza-capital-ai-fund/), [TechBuild Africa](https://techbuild.africa/enza-capital-closes-21m-african-ai-fund-startup/))

The new fund writes initial cheques of $250,000 to $3 million from pre-seed through Series B and reserves capital for follow-ons. Target sectors are financial services, healthcare, enterprise technology, education, agriculture, climate and emerging infrastructure. The firm, founded in 2019 and now with offices in Nairobi, Johannesburg and Cairo, says it has backed 49 companies across ten African countries through its first two funds, and it recorded its first realised exit in 2025 when South African hospital group Netcare bought into Quro Medical.

Three numbers in that paragraph deserve a second look.

The first is the shape of the raise. Enza closed its previous fund at $58 million in September 2023, having made 48 investments by then. ([Disrupt Africa, 2023](https://disruptafrica.com/2023/09/29/pan-african-vc-firm-enza-capital-closes-58m-fund-ii/)) A $60 million target is therefore a 3% increase on the last fund, not a step change. In a market where the median AI round is under $2 million, a manager holding its fund size roughly flat in a downturn is a modest, checkable signal: the firm is not claiming a bigger platform than it built.

The second is that neither write-up names a single limited partner. For a fund that has already closed $21 million, that is the largest undisclosed fact in the story. Who commits to an Africa-focused AI vehicle at first close, and on what terms, is exactly what the next fundraise depends on.

The third is the timeline. Money committed in November 2026 is not expected to reach a company until 2027. That is normal, and it is the reason a first close belongs in a different category from a funding round.

## Commitment, capital and equity are three different things

Venture money passes through states, and announcements tend to report the first one while readers hear the last one.

| State | What it means | When it shows up |
|-------|--------------|------------------|
| Committed | An LP has signed a legal obligation to fund up to an amount | At the first close |
| Called | The GP has requested a slice and the cash is in the fund's account | Over the investment period, on notice |
| Deployed | The fund has wired money into a company in exchange for shares | After a signed term sheet and closing |
| Reserved | Committed capital earmarked for the existing portfolio, not for new companies | Committed at the same time, spent later |

A first close is the point at which a fund can legally start charging fees and, usually, start investing. It is not the point at which capital has been called. Enza's stated plan, first investments in 2027, puts roughly a year between the announcement and the first wire.

The fee clock does not wait for the wire. Management fees are typically charged on committed capital during the investment period, which is why a first close matters to the manager's revenue before it matters to any founder. Carta's fund data puts the median management fee at 2% of committed capital during the investment period across every vintage from 2018 to 2025, with a 75th percentile at 2.5%, and a median investment period of five years. Operating expenses are a separate line: the median fund between $1 million and $10 million spends about 3.4% of committed capital on audit, tax, legal and administration. ([1752.vc on Carta's fund economics](https://www.1752.vc/learn/venture-capital-management-fees))

Run those conventions against $21 million and the headline shrinks in a way nobody's press release shows.

## The decade of fees before the first cheque

Below is the whole model, in about fifty lines of standard library Python. It takes the published terms of the fund, applies three plausible fee schedules to a ten-year life, then divides what is left between first cheques and a follow-on reserve. The follow-on reserve is set at 50%, which is a common starting point for a fund that expects to defend its ownership in later rounds.

{% raw %}
```python
"""First-close arithmetic for Enza Capital's AI Africa Fund (announced Nov 2026).

Published terms: USD 21.0m first close, USD 60.0m target, USD 80.0m hard cap,
first cheques USD 0.25m-3.0m. Fee schedules and the opex rate are market
conventions, not the firm's disclosed terms.
"""

COMMITTED   = 21.0     # USD m, first close
TARGET      = 60.0     # USD m
CHEQUES     = (0.25, 0.75, 1.50)   # USD m
OPEX_RATE   = 0.034    # Carta median opex, funds of USD 1m-10m
RESERVE     = 0.50
FUND_I_N    = 23       # companies backed from Fund I
FUND_I_FO   = 12       # of those, at least one follow-on
DECADE      = 10

SCHEDULES = {
    "flat 2.0% all 10y":  [(10, 0.020)],
    "2.0% 5y then 1.0%":  [(5, 0.020), (5, 0.010)],
    "2.5% 5y then 1.25%": [(5, 0.025), (5, 0.0125)],
}


def investable(committed, schedule, opex_rate=OPEX_RATE):
    """Return (capital left after fees and opex, fee total)."""
    assert sum(y for y, _ in schedule) == DECADE
    fee = sum(y * r for y, r in schedule)
    return committed * (1.0 - fee - opex_rate), committed * fee


print("first close: USD 21.0m committed")
print(f"{'fee schedule':21s}{'fees':>6s}{'opex':>6s}{'investable':>11s}{'primary':>9s}")
for label, sched in SCHEDULES.items():
    cap, fee = investable(COMMITTED, sched)
    print(f"{label:21s}{fee:6.2f}{COMMITTED * OPEX_RATE:6.2f}{cap:11.2f}{cap * (1 - RESERVE):9.2f}")

flat_cap, _ = investable(COMMITTED, SCHEDULES["flat 2.0% all 10y"])
flat_primary = flat_cap * (1 - RESERVE)
print()
print("first cheques the first close buys (flat-fee case, 50% held back)")
print(f"{'cheque':>8s}{'companies':>11s}")
for c in CHEQUES:
    print(f"{c:7.2f}m{flat_primary / c:11.1f}")

target_cap, _ = investable(TARGET, SCHEDULES["flat 2.0% all 10y"])
target_primary = target_cap * (1 - RESERVE)
print()
print(f"same at the USD 60.0m target: investable {target_cap:.2f}m, primary {target_primary:.2f}m")
for c in CHEQUES:
    print(f"{c:7.2f}m{target_primary / c:11.1f}")

p = FUND_I_FO / FUND_I_N
print()
print(f"Fund I follow-on rate: {FUND_I_FO}/{FUND_I_N} = {p:.3f}")
print("reserve ratio a fund needs, by follow-on size relative to the first cheque")
print(f"{'follow-on size':>15s}{'reserve needed':>16s}")
for m in (1.0, 1.5, 2.0):
    print(f"{m:14.1f}x{p / (1 + p * m):16.1%}")

ai_disclosed, deals, median_round = 93.0, 30, 1.9
africa_total, ai_2025, ai_h1_2026 = 2400.0, 170.0, 16.0
print()
print(f"AI share of 2026 African startup funding: {ai_disclosed / africa_total:.1%}")
print(f"fund target / 2026 disclosed AI funding: {TARGET / ai_disclosed:.2f}x")
print(f"first close / median AI round: {COMMITTED / median_round:.1f}x")
print(f"first close / all H1 2026 disclosed AI funding: {COMMITTED / ai_h1_2026:.2f}x")
print(f"disclosed AI funding, 2025 to H1 2026: {ai_2025:.0f}m to {ai_h1_2026:.0f}m, {1 - ai_h1_2026 / ai_2025:.1%} fall")

price, stake, seed = 7.1, 0.5588, 1.1
implied = price / stake
print(f"Quro implied equity value: {implied:.1f}m, {implied / seed:.1f}x its 2021 seed round")
```
{% endraw %}

Run it and the first table is the announcement, restated:

```text
first close: USD 21.0m committed
fee schedule           fees  opex investable  primary
flat 2.0% all 10y      4.20  0.71      16.09     8.04
2.0% 5y then 1.0%      3.15  0.71      17.14     8.57
2.5% 5y then 1.25%     3.94  0.71      16.35     8.17
```

On the standard 2% for ten years, fees take $4.20 million and operating costs another $0.71 million, which leaves $16.09 million of investable capital out of $21 million. A step-down to 1% after the investment period returns about $1 million to the fund. That is the first honest number in the story, and it is still not the number of companies.

## Eleven first cheques, not eighty-four

The fund's stated range is $250,000 to $3 million. Divide the announced $21 million by the $250,000 floor and you get 84 companies, a figure no venture fund of this size would attempt, because a manager cannot sit on 84 boards or reserve follow-on capacity across 84 positions. The useful number is the sweet spot, not the endpoints.

Enza's FAQ gives a first check of $250,000 to $5 million with a sweet spot between $500,000 and $2 million, and says it reserves capital to follow on into portfolio companies. ([Enza Capital FAQ](https://enza.capital/)) The new fund is reported at a narrower $250,000 to $3 million. Take $750,000, inside both descriptions, and the count falls out:

```text
first cheques the first close buys (flat-fee case, 50% held back)
  cheque  companies
   0.25m       32.2
   0.75m       10.7
   1.50m        5.4
```

Eleven companies at $750,000, or five at $1.5 million. For comparison, the firm says it followed on into 12 of the 23 companies it backed from Fund I. Eleven first cheques from a first close is about half of one prior fund's company count, which is a reasonable place for a fund that is 35% raised to be.

The same arithmetic at the full target:

```text
same at the USD 60.0m target: investable 45.96m, primary 22.98m
   0.25m       91.9
   0.75m       30.6
   1.50m       15.3
```

About thirty companies at the sweet spot, more than Fund I's 23 and fewer than the 49 the firm has backed across two funds. That is the portfolio this fund can plausibly hold, and it is worth remembering when a later announcement talks about the size of the AI opportunity on the continent. A $60 million fund writing $250,000 cheques is a seed machine. The same fund at $3 million cheques holds about eight positions. The cheque range is a strategy statement, and the arithmetic turns it into a portfolio count.

> **Why the reserve matters more than the cheque size**
> Every dollar reserved for follow-on is a dollar that cannot open a new position. On the same $21 million, a 50% reserve and $750,000 cheques hold a portfolio of eleven. A 25% reserve holds sixteen, and accepts that most of its companies will not be defended in later rounds. Neither is wrong. They are different bets on whether the next round happens.
{: .prompt-tip }

## The reserve is a forecast, not a rule

Fund I's follow-on record is the most useful number in this post, because it is measured rather than declared: at least one follow-on into 12 of 23 companies, or 52%. Plug that rate back in and you can compute the reserve ratio a fund actually needs, given how large its follow-ons are relative to the first cheque.

```text
Fund I follow-on rate: 12/23 = 0.522
reserve ratio a fund needs, by follow-on size relative to the first cheque
 follow-on size  reserve needed
           1.0x           34.3%
           1.5x           29.3%
           2.0x           25.5%
```

The result is counter-intuitive if you have only heard the 50% rule of thumb. At a 52% follow-on rate where follow-ons match the first cheque, a 34% reserve is self-funding. Even at twice the first cheque, 26% covers the expected calls. A 50% reserve is therefore not a defensive default. It is an explicit forecast that the fund will either follow on more often than it has historically, or write follow-ons large enough to defend ownership through a priced round, or hold back because the next round will need a bigger cheque than the first.

For a founder reading the announcement, this is the difference between a fund that can lead a Series A and one that cannot. The $21 million first close, after fees and a 50% reserve, has about $8 million on each side of the split. If it puts $750,000 into eleven companies and follows on into the five or six that historically would raise again, the reserve funds about four of them at $2 million a round and leaves the rest undefended. That allocation is decided at the first close, before a single founder has pitched.

## What the one exit says about the stage

The firm's first realised exit is a useful calibration, because it is the only completed price in the story. Netcare agreed to buy a controlling stake in Quro Medical for R121 million, about $7.1 million, in a deal that closed in June 2025. Netcare's initial 55.88% stake dilutes to 46.74% once a reserved employee share plan is allocated, so "controlling" is accurate at closing and imprecise afterwards. Quro had raised $1.1 million in 2021 led by Enza and Mohau Equity Partners, and by late 2023 it had treated more than 1,000 patients at home with reimbursement contracts covering over 90% of South Africa's private medical scheme market. ([Launch Base Africa on the deal](https://launchbaseafrica.com/2025/12/16/the-acqui-hire-playbook-inside-netcares-strategic-acquisition-of-south-africas-quro-medical/))

Two arithmetic facts follow. The price paid for 55.88% implies an equity value of about $12.7 million for the whole company, or 11.6 times the size of Quro's seed round. That is a good outcome for a 2021 African health seed. It is also, at fund scale, small: a fund the size of Fund II returning three times its capital needs $174 million of gross proceeds, which is about fourteen exits at this price. The company's metrics are the more interesting half of the comparison, because a home-hospital service with 90% scheme coverage is exactly the kind of business a fund would want to defend in later rounds.

The useful takeaway from the deal is not the multiple. It is that the exit was a strategic acquisition by a domestic hospital group, not a venture round. In a market where disclosure is thin, the buy-side is the local incumbent, and the price is set by what the incumbent saves on beds.

## The market the fund is fishing in

Set the fund against the flow it is trying to invest in. Africa: The Big Deal's tracker has African startups raising $2.4 billion in the first ten months of 2026, down 12% on $2.7 billion, with equity up 5% to $1.6 billion and debt down 33% to $740 million. Fewer companies raised: 343 against 438, a 22% fall and the lowest January to October count since the tracker began counting $100,000-plus deals in 2021. ([Scommerce summary of the tracker](https://www.scommerce.com/african-startup-funding/))

Against that total, AI is still a small line. Launch Base Africa's deal-level count puts about 30 AI-labelled transactions in 2026 at roughly $93 million disclosed, a median round of about $1.9 million, and disclosed AI funding down from at least $170 million in 2025 to about $16 million in the first half of 2026.

```text
AI share of 2026 African startup funding: 3.9%
fund target / 2026 disclosed AI funding: 0.65x
first close / median AI round: 11.1x
first close / all H1 2026 disclosed AI funding: 1.31x
disclosed AI funding, 2025 to H1 2026: 170m to 16m, 90.6% fall
```

Read those five lines together and the announcement changes character. AI was about 60% of global venture investment in 2025 on Launch Base Africa's framing, and 3.9% of African startup funding in 2026 on the same trackers. The windows do not align exactly, since the $2.4 billion covers January to October and the AI figure is a year-to-date count published in November, but the order of magnitude is not in doubt: AI is a low single-digit share of the money raised on the continent. The fund's $60 million target is 65% of everything disclosed as African AI funding this year. Its first close alone is 1.31 times the entire disclosed AI total for the first half of 2026.

That is not a complaint about the fund. It is the reason the fund exists, and it is the strongest positive reading of the announcement: a specialist vehicle is being capitalised into a gap rather than out of one, at the moment when generalist capital has withdrawn. The 90.6% fall in disclosed AI funding is a statement about the supply of cheques, not about the supply of companies.

## How to read the next fund announcement

Seven questions turn a press release into a portfolio estimate. None of them require a data room.

1. First close or final close? A first close is a commitment; a final close is the total the fund will ever have. Announcements rarely put the distinction in the headline.
2. Target or hard cap? The target is what the manager is aiming to raise and can be cut. The hard cap is the ceiling on commitments. The difference between $60 million and $80 million is a story about demand.
3. What are the fee terms? Two percent for ten years costs 20% of commitments. A step-down after the investment period returns several points. Small funds also carry a higher opex ratio, which compounds the drag.
4. What is the first-cheque range and where is the sweet spot? The range sets the floor, the sweet spot sets the portfolio. Divide primary capital by the sweet spot to get a company count.
5. What reserve policy is stated? A percentage is a forecast. Cross-check it against the firm's own follow-on record if one exists.
6. When does deployment start? A first close followed by a year before the first investment is normal, and it means the announcement is about capacity, not activity.
7. Who are the LPs? If the write-up does not say, the fund's investor base is the part of the story you cannot check.

Applied to this announcement, the answers describe a $21 million-capacity fund that will hold about eleven positions at its sweet spot, with a reserve that covers about $4 million of expected follow-on calls at historical rates, deploying from 2027 into a market that disclosed $93 million of AI funding in 2026. Whether that is enough depends entirely on the quality of eleven companies, which is the part no arithmetic can price.

## Key takeaways

| Point | Number |
|-------|--------|
| First close, as announced | $21.0m, 35% of a $60m target, $80m hard cap |
| Fee and opex drag on a ten-year life, flat 2% | $4.91m, about 23% of commitments |
| Investable capital at the first close | $16.09m before the reserve |
| Primary capital after a 50% follow-on reserve | $8.04m |
| First cheques at the $750,000 sweet spot | 10.7 from the first close, 30.6 at full target |
| Follow-on rate the firm has run | 12 of 23 companies from Fund I, 52% |
| Reserve a 52% rate needs | 34% at equal-size follow-ons, 26% at double |
| Implied value of the one realised exit | $12.7m for Quro Medical, 11.6x its seed round |
| AI share of 2026 African startup funding | 3.9%, about $93m of $2.4bn |
| Distance from announcement to first investment | about a year, by the firm's own timeline |

The announcement is good news, and the arithmetic does not change that. A pan-African manager held its fund size steady and raised a third of it in a year when disclosed AI funding on the continent fell by more than 90%, on a thesis that AI changes the cost of serving fragmented markets. What the arithmetic does change is what the number means. Twenty-one million is not eighty-four companies, or even twenty-one. It is about eleven, plus the option to defend a few of them.

## References

- Disrupt Africa, "Pan-African VC firm Enza Capital reaches $21m first close of AI Africa Fund", 13 November 2026 — https://disruptafrica.com/2026/11/13/pan-african-vc-firm-enza-capital-reaches-21m-first-close-of-ai-africa-fund/
- Launch Base Africa, "Enza Capital Raises $21M to Back African AI as Venture Funding Retreats", 11 November 2026 — https://launchbaseafrica.com/2026/11/11/enza-capital-ai-fund/
- TechBuild Africa, "Enza Capital Closes $21M for African AI Fund as Startup Investment Slows" — https://techbuild.africa/enza-capital-closes-21m-african-ai-fund-startup/
- Disrupt Africa, "Pan-African VC firm Enza Capital closes $58m Fund II", 29 September 2023 — https://disruptafrica.com/2023/09/29/pan-african-vc-firm-enza-capital-closes-58m-fund-ii/
- Enza Capital, investor FAQ (first-check range, growth capital vehicle, Fund I follow-on record) — https://enza.capital/
- 1752.vc, "Venture Capital Management Fees: The 2% Explained With Math", citing Carta's fund economics data — https://www.1752.vc/learn/venture-capital-management-fees
- Launch Base Africa, "The Acqui-Hire Playbook: Inside Netcare's Strategic Acquisition of South Africa's Quro Medical", 16 December 2025 — https://launchbaseafrica.com/2025/12/16/the-acqui-hire-playbook-inside-netcares-strategic-acquisition-of-south-africas-quro-medical/
- Scommerce, "African startup funding falls 12% to $2.4 billion in 10 months", reporting Africa: The Big Deal data — https://www.scommerce.com/african-startup-funding/

## Related posts

- [Ten African Startups, One Constraint Map](/posts/google-africa-applied-ai-lab-cohort/) — the cohort the same venture money is chasing, and what the selection rewarded.
- [One Label Holds Ten Numbers](/posts/distillation-target-logit-budget/) — the same method applied to training data: price the artifact, not the headline.
- [Cascade Routing: Pay for the Big Model Only on the Queries That Need It](/posts/model-cascade-escalation-budget/) — capacity arithmetic on the inference side, worked the same way.
- [Rank Is a Budget: What LoRA at r=8 Costs and Buys on an 8B Model](/posts/lora-rank-budget/) — the parameter-budget question, priced before the training run.
