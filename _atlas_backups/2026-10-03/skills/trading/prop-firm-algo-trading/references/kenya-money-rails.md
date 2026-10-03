# Money rails and jurisdiction facts (Kenya-based user)

Eligibility decides *whether* a venue can be used; these rails decide whether it
can actually be funded and paid. Both fail silently, and both cost money to
discover late. **Re-verify tax and caps before relying on them** — this file is a
starting position, not a ruling.

## Paying an evaluation fee from Kenya

- Card payments issued in Kenya fail often enough on offshore prop processors
  that a card must never be the primary rail.
- **Primary rail: stablecoin.** Buy USDT with M-PESA on a P2P desk (the large
  exchanges run KES/M-PESA markets), then pay the fee in USDT wherever the firm
  accepts crypto. Crypto-accepting gateways used by prop firms include wallet
  payment processors (Confirmo-style) and exchange-pay options; some firms also
  take PayPal, Skrill or bank wire.
- P2P hygiene: high-reputation merchants only, small amounts, no off-platform
  contact, and treat every counterparty dispute as a frozen balance.

## Getting paid

| Route | How it lands in Kenya | Notes |
|---|---|---|
| Crypto to the user's wallet, then P2P | Firm pays USDT/USDC/BTC → sell for KES on P2P → M-PESA | Best speed and cost; P2P merchant history is what makes it repeatable |
| Payout processor (Rise-style) | Email must match the prop account | A percentage fee applies; typically used above the crypto per-request cap |
| Wise | Supports KES to Kenyan bank accounts and M-PESA wallets | Depends on whether the firm pays via Wise |
| Bank wire / SWIFT | Bank account | Slow, fee-heavy, poor FX — last resort |
| Direct M-PESA from the firm | Does not exist in practice | Ignore any blog claiming otherwise |
| Push-to-card | Card | Kenyan inbound push-to-card acceptance is inconsistent; test small first |

Plan around M-PESA limits (order of KES 250,000 per transaction, 500,000 per
day), so a large payout arrives in tranches. Also check the firm's own caps on
the cheap route — several cap crypto payouts per request (order of \$1.5k) and
push anything larger to a percentage-fee processor.

## KYC

- National ID or passport (passport is more universally accepted) plus a
  liveness/selfie check; proof of address (bank statement or utility bill) under
  about three months old. Mobile-money statements are accepted by some firms, not
  all — have a bank statement ready.
- **Name-match is enforced across the whole chain:** prop account, KYC identity,
  funding-account holder and payout destination must be the same person.
  Third-party wallets and relatives' accounts get held.
- Use a dedicated email alias for prop accounts, never the main address.

## Tax layers (two distinct regimes — do not conflate them)

1. **Income:** residents are taxed on worldwide income, progressive up to the top
   individual band, declared on the annual return. Foreign prop-firm payouts are
   therefore taxable; convert each payout at the prevailing rate and keep the FX
   record.
2. **Asset/transaction layer:** the payout medium carries its own regime. Kenya's
   transaction-level digital-asset tax was repealed and replaced by an excise duty
   on the **fees charged by virtual-asset service providers** (exchanges, wallets)
   rather than on transfer value — so guides quoting the old percentage on
   transaction value are out of date. Virtual assets are lawful and provider
   licensing has moved forward under the VASP framework.

Both regimes are moving targets. Get an accountant's ruling on classification
(business vs other income) and on whether evaluation fees are deductible
**before the first payout**, and keep firm statements, payout transactions, P2P
records, FX rates and the harness's order/rule log as evidence.

## Pre-flight test — run before spending any evaluation fee

1. Buy a small amount of the payout asset with M-PESA on P2P; note the effective
   rate and total fees (this is the real cost of the round trip).
2. Open the funding/trading account, complete KYC with the user's documents, and
   create an API key scoped to trade + read with **withdrawals disabled**; confirm
   it authenticates against the venue's demo/practice endpoint.
3. Prove one small outbound leg end-to-end to the local cash-out route.
4. For a cTrader venue, confirm a cloud cBot instance and/or an Open API
   connection actually attaches to the account — if neither works, the venue is
   out before any fee.
5. Only then: dated eligibility screenshot on file, KYC done, rail proven → buy
   **one** evaluation.

## Ops notes

- EAT is UTC+3 year-round (no DST), so London/NY sessions land in the local
  afternoon and evening — convenient for a locally-run harness. Crypto is 24/7 and
  indifferent to it.
- Assume power and connectivity outages. Bracket every order at the venue so an
  outage cannot leave an unbounded position, and reconcile on every restart; for
  anything that must be continuous, prefer venue-side or cloud execution.
- Budget the round-trip friction (P2P spread plus any payout-processor fee) into
  every profit-split calculation — it is a real percentage of the payout.
