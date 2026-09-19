---
name: prop-firm-algo-trading
description: "Use when an agent will trade a prop-firm account."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Trading, Prop-Firm, Automation, Risk, Algo]
    related_skills: [watchers, atlas-lesson-bank, blocked-page-recovery]
---

# Prop-Firm Algo Trading

For any task where the capital is a funded prop-firm account and the trader is
an automated system (bot, EA, agent harness). Prop money does not come from a
clever model — it comes from staying inside the firm's rules while the edge
plays out. Build, judge and report everything in that order.

Companion references: `references/venue-rules.md` — venue-by-venue constraints,
the decision table, the universal ban list, the eligibility/source-verification
method. `references/kenya-money-rails.md` — funding rails in and out, KYC
name-matching, the crypto-tax position, and the pre-flight test that must pass
before any fee is paid. Load both before recommending a firm or designing an
adapter.

## Always-on rules

1. **The user owns the account.** KYC, identity, tax residency and rule
   compliance are theirs; the agent builds, runs and monitors the system under
   their identity. Never imply the agent can open, own or verify an account.
2. **Never spend an evaluation fee before the spend gates pass** (§Spend gates).
   Fees are the recurring cost of this class of work and the easiest money lost.
3. **Eligibility first, then rules — both verified at the source, dated,
   immediately before recommending or buying.** A firm can have the
   best-documented API on the market and still be legally closed to this user's
   citizenship or residency: the failure is fatal, silent, and invisible in every
   "best prop firm for algo" listicle. Prop-firm policies move monthly; listicles,
   aggregator "firms that accept <country>" pages and vendor comparisons go stale
   within weeks, and a firm's own "we are best for algo traders" page is
   marketing, not evidence. When sources disagree, the firm's own domain wins,
   and the check carries a date so a stale one is visible (see §Eligibility gate).
4. **Do the arithmetic with a tool, never mentally** — EV per fee, break-even
   payout, distance-to-breach, position sizing. The numbers decide the plan, so
   they must come from a computation the user can check.
5. **The risk layer has one-way veto power over the model.** If the agent
   believes the model is wrong, the correct action is to stop the run and open a
   research task — never to hand-trade around it.
6. **Deliver as a document set plus a chat push.** Write the plan to
   `~/Dev/<project>/` as files, then push a plain-language summary to Telegram:
   append `MEDIA:<absolute-path>` lines to the body file and send with
   `hermes send --to telegram --file <body>` — a file body avoids shell quoting
   entirely, and keeping it under 4096 chars avoids Telegram clipping. When the
   body itself runs long, send the attachments as a **separate short message**: a
   split or clipped body can drop the trailing `MEDIA:` lines and deliver a
   report with no files attached. Record durable lessons in
   `~/Dev/ATLAS-LEARNINGS/LESSONS.md`.
7. **Channel reality check.** Ranked by expected value per hour for this user:
   bug bounty > productized agent services > content > trading desk > bandwidth
   apps. Bandwidth/proxyware measures ~$2–3 per device per month (platforms keep
   ~91–99% of the $1.75–15/GB they resell your IP for; earnings are
   demand-limited, not bandwidth-limited), and it turns the user's IP into a
   stranger's exit node — quarantine it on a dedicated device and separate
   SIM/WAN, never the line carrying M-PESA, banking, KYC or email. State this
   plainly instead of letting it look like an income channel.

## Eligibility gate — jurisdiction and money rails before the API

**Run this gate before anything else in this skill.** It is the one that gets
skipped, because API capability is the interesting question and eligibility looks
like paperwork — and it is the one whose failure cannot be worked around.

Eligibility is decided **per product, not per brand**: one firm can be open for
its FX/CFD accounts and closed for its futures product, and another can decide on
citizenship rather than residence (so relocating does not fix it). Check in this
order:

1. The firm's **own** eligibility/FAQ page, rendered in a real browser, checked
   against the table for the *specific product* being bought. Record the date.
2. Payment rail **in** from the user's jurisdiction — card rails from some
   countries fail routinely on offshore processors, and mobile money is
   effectively never accepted directly by a prop firm.
3. Payout rail **out**, including the per-request caps firms put on their cheap
   route (see `references/kenya-money-rails.md`).
4. Only then the four automation axes below.

Store `eligibility_checked_on` in the venue config and make the adapter refuse to
start when that date is older than ~90 days (alongside `api_on_funded_phase`,
`hosting_allowed`, `monitor_required`) — these lists change without notice.

Expect a structural consequence: the US-regulated futures tier bars long lists
of citizenships outright, so for many users the tradable universe collapses to
crypto prop plus FX/CFD prop. Say that plainly and rebuild the shortlist from
eligible venues rather than presenting an ineligible firm with caveats.

## The automation gate — four axes

Once eligibility passes, a firm is unusable for unattended automation if it fails
any of these, regardless of its marketing:

1. **Is a real API permitted on the FUNDED phase**, not just the evaluation?
2. **Where may the code run** — own device only, or hosting allowed? A scheduled
   agent dies at any firm that bans VPS/VPN/remote servers.
3. **Rule geometry** — trailing vs static drawdown, daily loss cap and its reset
   timezone, per-position loss cap, mandatory stop-loss deadline, consistency
   rule.
4. **Payout trust** — history, split, settlement method and speed, jurisdiction
   and regulation of the operator.

Then pick the venue **from the model's measured profile**, not up front: the
strategy is an input, the firm is an output. Mapping table in
`references/venue-rules.md`. If a strategy needs order frequency near the HFT
line, redesign the strategy — HFT bans are universal.

**The live-phase swap is the most expensive trap.** Firms routinely allow bots
in the evaluation and restrict them on a real funded account (the firm now
carries broker risk). Read the funded-phase rules first and design for those;
never plan a business on the evaluation's freedoms.

## Rule engine as code — build this before the model

`risk/firm_rules.py` encodes the firm's contract as executable constraints,
enforced on every prospective order, with the decision logged next to the
outcome (approved / sized-down / vetoed + reason):

- Drawdown: type, size, whether it follows realised or unrealised P&L, and
  whether it trails the peak (an early winner shrinks the remaining room).
- Daily loss limit **plus the venue's reset timezone** — get the timezone wrong
  and you breach without breaking any rule you understood.
- Per-position max loss and any mandatory stop-loss deadline on automated
  entries.
- Consistency-rule tracking (share of profit from the best day): size down or
  stop when one day gets too concentrated, because a burst-earning system
  cannot choose which session its profit lands in.
- Blackout windows from the news/event plane, rollover and session edges.
- HFT margin: max orders/minute and minimum hold time enforced locally, far
  below the firm's threshold.
- Max trades/day, concurrency, allowed instruments, max size/leverage.
- `emergency_flatten` on: drawdown proximity, repeated venue rejects, heartbeat
  loss, or the kill-switch flag.

Execution invariants that go with it: every order **bracketed at the venue** (a
crash must not leave an unbounded position); deterministic idempotent client
order IDs so a restart cannot double-fire; reconciliation against the venue's
own positions/equity on every start and at every session close (never trust
locally computed P&L); an append-only order/decision log, which is both the
audit trail and the payout-dispute evidence; one kill switch, tested on a
schedule.

**Hosting: satisfy a "no VPS" posture without a VPS.** cTrader's own cloud runs
cBots 24/7 with the user's device switched off and is not third-party
infrastructure — a thin cBot polling a small authenticated endpoint owned by the
agent's Python side keeps the model local while execution stays continuous. Where
the venue is MT5-only, the local equivalents are a Wine-hosted terminal driven
over RPC or a file-bridge EA; budget for that fragility deliberately instead of
discovering it after the evaluation is bought.

Split the news plane's two consumers: a **risk** consumer publishing
blackouts and event flags the engine enforces, and an **alpha** consumer whose
LLM tags/classifies events into features. LLMs are decent at parsing news and
poor at predicting price — statistics decides whether a tag is tradeable.

## Spend gates — no fee until all pass

- **Money-rail pre-flight, before any fee:** the funding account opened and KYC'd,
  a trade-only API key created with withdrawals disabled, a small round trip of
  the payout asset through the local cash-out route, and one order placed through
  the intended execution path on a demo/practice account. If the rail fails, the
  venue choice changes before money moves.
- Reproducible backtest from a config, with real fees + spread + slippage +
  funding applied per trade, and no lookahead.
- Walk-forward on unseen windows, purged/embargoed folds, performance across
  ≥3 sub-periods, parameter perturbation (±20%) that doesn't destroy the edge,
  and multiple-testing correction — report the deflated figure, not the best
  variant's.
- Baseline first: simple rule-based systems (session breakout, reversion with a
  regime filter, carry) are the bar any model must beat out-of-sample after
  costs. If it can't beat them, ship the baseline.
- Rule-simulator replay against the venue's exact rule set with **zero** would-be
  breaches across the horizon, blackouts included.
- 30 days on the venue's own demo/practice path exercising real order plumbing
  (partial fills, rejects, cancels, reconnect) — then one evaluation at a time.

Decide the stop rules while calm, not mid-drawdown: a hard monthly budget; N
rule-clean failures → back to research instead of buying another fee; any live
breach or near-miss → stop, postmortem the engine, resume only when the
simulator is clean again; live slippage/hit-rate outside the validated band →
demote the model to paper. Treat sizing, labels and gates as hypotheses until
live money proves them; never present an ungated system as validated.

## Funding loop

Per realized payout: ~30% LLM/API credits, ~50% reinvest (accounts, data,
infra), ~20% untouchable reserve never used for trading. Until the first payout,
credits come from an agreed budget — there is no version of this where the desk
self-funds in month one, and any plan implying otherwise is fiction. Withdraw on
schedule and cap exposure per firm: unregulated operators offer no recourse.

## Pitfalls

- **API availability on the live funded phase** decides the venue. A firm
  permitting bots only pre-funded, or only when monitored from the user's own
  machine, cannot host an unattended 24/7 system.
- **Check the broker's own API terms, not just the firm's.** A retail futures
  broker API can require a funded live account plus a monthly add-on and
  explicitly exclude prop accounts, which silently forces automation through
  third-party bridges; some firms also restrict fully autonomous
  entry-and-exit.
- **Stale documentation trap:** automation platforms get re-licensed to a single
  firm, so older tutorials naming that platform for other firms are dead. Match
  the API host and account type against the firm's current help centre before
  writing an adapter.
- **Identical strategies across several accounts** can be flagged as copy
  trading and consolidated or disqualified. Copy-trading bans target other
  people's signals and managed accounts, not the user's own system — but
  EA-disclosure requirements must be honoured or a payout can be voided.
- **Retail baselines are brutal; say them out loud:** the large majority of
  challenge buyers fail and only a small single-digit share ever get paid, with
  a similar share of bots passing. Frame automation as a hypothesis — that rule
  compliance lifts the pass rate — not as a proven edge, and never quote a
  provider's own benchmark page as validation of the user's system.
- **"Bots allowed" is platform- and size-scoped inside a single firm.** One firm
  can permit MT5 EAs while banning them on cTrader/Match-Trader, or require manual
  trading above an account size, or treat an auto-executing indicator as banned
  automation and disqualify on it. Read the clause for the exact platform and
  account size being bought, never the firm-level summary.
- **Search snippets invert eligibility lists.** A snippet can render a firm's
  *supported*-markets list as though it were the restricted one, which reads as a
  hard block and sends the shortlist sideways. Open the page, read the section
  around the country name, and prefer the firm's own domain over any tracker —
  "firms that accept <country>" pages routinely list firms whose own help centre
  bars that country.
- **Tax and jurisdiction facts go stale faster than trading rules.** Worldwide
  income is taxed and declared on the annual return, *and* a separate regime
  applies to the payout asset itself — check whether a transaction-level
  digital-asset tax is still in force or has been replaced by a duty on provider
  fees before quoting either (see `references/kenya-money-rails.md`). Get an
  accountant's ruling on classification, including whether evaluation fees are
  deductible, before the first payout, and keep every statement, payout
  transaction and order log as evidence.
