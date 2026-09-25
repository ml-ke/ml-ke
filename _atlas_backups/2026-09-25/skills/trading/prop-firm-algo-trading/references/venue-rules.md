# Venue rules for automated prop trading

Snapshot of constraints gathered from firm help centres and independent rule
trackers. **Re-verify at the source before recommending or buying anything** —
these policies change monthly, and several are the difference between a payout
and a voided account.

## How to verify (method, not just sources)

Render the firm's own help centre and rules documents in a **real browser**:
prop-firm and platform sites are frequently JS-only or behind bot walls, and a
plain HTTP fetch returns 200 with only nav/cookie-banner chrome, which reads
like success and is not. The general fetch ladder lives in the bundled
`blocked-page-recovery` skill; the pattern that works for policy pages:

```python
# one browser call, many policy pages
first = True
for name, url in pages:
    new_tab(url) if first else goto_url(url)
    first = False
    wait_for_load(); time.sleep(2)
    text = js('document.body.innerText') or ""
    store[name] = text
    low = text.lower()
    hits = [low.find(k) for k in ["funded", "vps", "payout", "automated"] if low.find(k) >= 0]
    s = max(0, min(hits) - 400)          # keyword-anchored slice, not the page head
    print(name, len(text)); print(text[s:s + 4500])
```

Print keyword-anchored windows — page heads are menus, the rules sit further
down. Persist every body to the workspace in the same call so a timeout doesn't
force a refetch.

Then cross-check against an **independent** rule tracker that publishes per-firm
bot policy with an update date. A comparison page published by a prop firm is
marketing; treat every "we are best for algo traders" claim as unverified until
an independent source and the firm's own rules document agree. Read the
funded-phase rules first, then the broker/platform terms underneath them —
restrictions hide in both layers.

## Futures venues

**This entire tier is jurisdiction-gated, so check eligibility before reading the
table.** US-regulated futures firms publish long lists of ineligible citizenships
and residencies — no trading, no funded accounts, no payouts — and none of that
appears in "best for algo" listicles. Expect the tier to be closed to much of the
world while crypto prop and FX/CFD prop stay open, and re-derive the shortlist
from providers that pass the eligibility gate instead of caveating a blocked one.

| Venue | API path | Bots on funded phase | Hosting | Notes |
|---|---|---|---|---|
| Topstep (TopstepX / ProjectX) | REST + real-time hubs on its own API host; billed separately from the challenge, with a trader discount code | Evaluation and first-stage funded only; **not** the live funded account | **Banned** — VPS/VPN/remote servers prohibited, order flow must originate from the user's own device and be actively monitored | Best-documented futures API: per-minute rate limits, 24h session tokens, **no sandbox** (test on a practice account; API orders are final). Read-only server work (data, backtest, logging, dashboards) is allowed as long as order transmission is not |
| Apex Trader Funding | Tradovate / Rithmic, usually via a bridge service | Semi-automated and DCA-style management allowed; fully autonomous entry+exit restricted | Allowed | No HFT, no arbitrage |
| Other futures firms on Tradovate / Rithmic / ProjectX | Via bridges or platform-native strategies | Generally yes; per firm | Generally allowed | Several publish **no consistency rule**, which matters to systematic systems; verify per firm |

**Broker-API trap:** a retail futures broker's own API can require a funded live
account with a minimum balance plus a monthly add-on, and exclude prop accounts
entirely. When that holds, prop automation is bridge-based (charting alert →
bridge service → broker) and the bridge becomes a dependency in the order path.

**Platform-exclusivity trap:** automation platforms get re-licensed to a single
firm, so older tutorials naming that platform for other firms point at the wrong
endpoints. Match the API host against the firm's current help centre before
writing an adapter.

## Crypto venues (24/7, hosting usually tolerated)

| Venue class | API path | Constraints to encode | Trust |
|---|---|---|---|
| Prop firm on a real exchange account (the trader's own exchange API keys) | Exchange REST/WS API via keys bound to that account | Realised loss per position capped (order of a few % of initial balance); a stop-loss must be attached within minutes of any automated entry; stablecoin payouts | Operator is unregulated — counterparty risk is real; withdraw promptly, diversify firms |
| Firms advertising a first-party REST/WS API with no fee or approval step | Firm's own API | Often claim no per-trade cap, no consistency rule, hosting allowed, news trading allowed | All self-published; require independent verification before funding |

Crypto prop is the only venue class where a genuinely unattended, hosted agent
is both permitted and technically clean. Price it against the rule friction:
per-position caps and stop-loss deadlines are sizing constraints the model must
encode, not paperwork.

## FX / indices on cTrader

- cTrader's Open API is a documented broker-side REST/streaming interface — the
  most mature retail algo path, and many firms now offer cTrader accounts.
- Expect EAs permitted on MT5/cTrader, and typically a **consistency rule on the
  evaluation** (no single day may exceed a fraction of total profit) —
  structurally hostile to systems whose edge lands in bursts, because a bot
  cannot choose which session its profit appears in. Firms also police
  automation with checker tools and restrict news-window trading per instrument.
- Best graduation venue once a system is proven: bigger capital, stronger payout
  record, more rule friction.

## Banned at essentially every firm (design nowhere near these)

- HFT / sub-second order flow, tick scalping, latency arbitrage, co-location,
  exploiting simulated fills or the firm's own pricing/execution lag.
- Copy trading from someone else's signals, or letting another party manage the
  account for compensation.
- Undisclosed EAs where disclosure is required — non-disclosure can void a
  payout even at firms that permit bots.
- Trading restricted news windows.

## Decision table — pick the venue from the model's profile

| Model profile | Venue class | Why |
|---|---|---|
| Intraday futures, flat by the close, few trades/day | Eligibility-verified futures firm with a documented API; agent on the user's own machine, monitored | Only if the user's citizenship/residency passes that firm's own eligibility page — several bar whole regions outright, and bridge-based automation is the usual fallback when they don't |
| 24/7 crypto, holds minutes to hours, must run unattended | Crypto prop on a real exchange API | Native API on the funded phase, hosting allowed, fast settlement |
| Crypto, few large concentrated trades | Crypto firm **without** a consistency rule | Consistency rules kill concentrated profit distribution |
| FX/indices, multi-day holds, wants trust and scale | cTrader Open API at a long-established firm | Documented API, overnight holding, strong payout record |
| High order frequency across many symbols | Nowhere | Redesign the strategy; HFT bans are universal |

## Expected-value framing to include with any recommendation

- EV per unit of fee = pass probability × realistic first payout − fee.
  Break-even payout = fee ÷ pass probability.
- Retail baseline: the large majority of challenge buyers fail and only a small
  single-digit share ever receive a payout, with a similar small share of bots
  passing. At baseline rates the fee is close to a coin flip, so the pass-rate
  edge must come from rule compliance and sizing off distance-to-breach — and
  the fee budget must be small enough to lose.
- Drawdown arithmetic follows: risk budget = balance × drawdown, and
  risk budget ÷ per-trade risk = consecutive losses to breach. Size from that
  number, never from conviction.
