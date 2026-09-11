---
title: "The Number Is the Password: SIM Swaps, OTP Interception and Why Bank MFA Fails at Scale"
date: 2026-09-11 00:00:00 +0300
categories: [Cybersecurity, Fintech]
tags: [sim-swap, otp-interception, mfa, account-takeover, mobile-banking, mpesa, fintech-security]
image:
  path: /assets/img/cover-sim-swap-otp-interception-mobile-banking.webp
  alt: A phone number held on a SIM card detached from a handset, with one-time-password envelopes rerouted into an attacker's phone while a bank account drains
---

## The second factor is a third party

Every mobile-banking MFA story ends the same way: the bank sends a one-time password to a phone number it does not control.

On **February 8, 2022**, Mercy Wairimu Kariuki woke to alerts showing **KES 4,418,601** leaving her Diamond Trust Bank account ([TechCabal](https://techcabal.com/2026/07/13/kenyan-court-holds-banks-telcos-liable-over-sim-swap-fraud/)). Her line had been swapped two days earlier — and reported to Safaricom the same day.

> **The key concept**
> An SMS one-time password is not really a second factor. It is a bearer token held by whoever controls a phone number — and the number is administered by a third party, through a help-desk process, outside your bank's perimeter. MFA built on it fails whenever the number moves, the message is relayed, or the handset is compromised.
{: .prompt-info }

This is the identity layer above both [Daraja's open joints](/posts/mpesa-daraja-api-pitfalls/) and [the middleman problem](/posts/aggregator-baas-security-playbook/).

## Kenya: the court that stopped accepting "the PIN was correct"

The swap happened through an M-PESA agent on **February 6, 2022**; the line was reinstated on February 7. From about **05:23 the next morning**, money left through DTB mobile banking and Pesalink in transfers over three days, each under DTB's **KES 2 million daily limit**, straddling a weekend reset ([Tech-ish](https://tech-ish.com/2026/07/13/safaricom-dtb-sim-swap-ruling/), [Techweez](https://techweez.com/2026/07/13/safaricom-dtb-to-pay-sim-swap-victim/)).

DTB argued its systems worked as designed (every transaction followed a correct PIN) and that the swap was a *novus actus interveniens* severing liability. On **June 18, 2026**, the High Court at Machakos dismissed both appeals, upholding a **60:40 split: Safaricom KES 2,630,000, DTB KES 1,788,601** ([TechCabal](https://techcabal.com/2026/07/13/kenyan-court-holds-banks-telcos-liable-over-sim-swap-fraud/)):

> "A bank cannot hide behind a customer's PIN when it is presented with a series of transactions that are so glaringly out of the ordinary that a reasonable banker would have been put on inquiry."
{: .prompt-warning }

Two findings matter: a correct PIN is no longer a defence — the burst of transfers to unrelated accounts was the signal the bank should have caught — and the telco and the bank each owe an **independent duty of care**.

## Kenya: what the carrier's own numbers say

Safaricom's chief corporate security officer Nick Mulila, in November 2024: *"In a month we get about 40 fraudulent swaps out of about 750K swaps"* — against roughly **28,000 swap requests a day** — and confirmed the telco **blocks M-PESA wallet access after agent-channel swaps** ([The Star](https://www.the-star.co.ke/business/kenya/2024-11-06-safaricom-cuts-sim-swap-fraud-to-40-in-every-750000-swaps)) — at 28,000 legitimate swaps a day, a permanent population of freshly re-bound numbers.

In October 2022, the High Court in Milimani let businessman **Abdi Zeila** invite other subscribers into a class action after **KES 495,651** was taken from his NCBA account and M-PESA wallet, including a **KES 66,640 mobile loan** in his name. He was roaming abroad — a swap he could not have requested ([Nairobi Wire](https://nairobiwire.com/2022/10/man-sues-safaricom-after-losing-over-ksh490k-to-sim-swap-fraud.html), [Daily Nation](https://nation.africa/kenya/news/safaricom-hit-by-sim-swap-fraud-class-action-suit-3991258)).

The trend is worse: INTERPOL reports **SIM-swap fraud up 327% during 2025**, with **more than 123,000 fraudulent SIM cards** and an estimated **US$3.8 million** lifted from mobile wallets ([Dawan](https://www.dawan.africa/news/interpol-kenya-emerges-as-east-africas-cybercrime-hotspot-as-mobile-money-fraud-surges)). SIM-swap losses in Kenyan reporting stand at **Sh491.6 million**, against **Sh810.68 million** lost through mobile banking in 2024, over four times 2023's Sh182.41 million ([Nairobi Law Monthly](https://nairobilawmonthly.com/authorities-crack-down-on-cyber-cafes-as-sim-swap-fraud-surges/); see [Fraud Trends, Read as a Dataset](/posts/cbk-fraud-trend-analytics/)).

## Global: the regulator's own MFA got swapped

On **January 9, 2024**, the U.S. SEC's @SECGov account on X posted a fake Bitcoin ETF approval. MFA had not failed — the number had moved:

> "Two days after the incident, in consultation with the SEC's telecom carrier, the SEC determined that the unauthorized party obtained control of the SEC cell phone number associated with the account in an apparent 'SIM swap' attack."
{: .prompt-danger }

The attackers reached no SEC systems or data; they needed only the number, and the number was administratively reassignable ([BleepingComputer](https://www.bleepingcomputer.com/news/security/sec-confirms-x-account-was-hacked-in-sim-swapping-attack/)).

## Global: MFA bypass as a business model

Scattered Spider industrialised the technique. CISA/FBI advisory AA23-320A (2023, updated 2025) describes it: *"After identifying usernames, passwords, PII, and conducting SIM swaps, the threat actors then use layered social engineering techniques"* — calls to discover help-desk reset requirements, then to *"convince IT help desk personnel to reset passwords and/or transfer MFA tokens"* and register their own ([CISA](https://www.cisa.gov/news-events/cybersecurity-advisories/aa23-320a)). MFA was not defeated cryptographically; enrolment was social-engineered.

MGM Resorts' September 2023 incident cost **about $100 million** ([NBC News](https://www.nbcnews.com/business/business-news/cyberattack-cost-mgm-resorts-100-million-las-vegas-company-says-rcna119138)). Five alleged members were indicted in November 2024; Noah Michael Urban, who conspired to steal at least **$800,000 from five victims by SIM-swapping**, got **10 years** and **$13 million** in restitution ([Krebs](https://krebsonsecurity.com/2025/08/sim-swapper-scattered-spider-hacker-gets-10-years/)).

## Two more ways the code leaks

**Real-time relay.** The 2022 "0ktapus" campaign used 169 phishing domains impersonating Okta sign-on pages and a kit that forwarded credentials and codes live to a Telegram bot: **9,931 credentials and 5,441 records containing MFA codes** across **136 organisations** ([Help Net Security, reporting Group-IB](https://www.helpnetsecurity.com/2022/08/25/0ktapus-twilio-cloudflare-phishers-targets/)). Nothing on the victim's phone registered a compromise.

**On-device capture.** In August 2026, Group-IB documented "WindRelay": a **13-minute call impersonating a bank** gets a victim to install an app labelled with the bank's name, the SpyNote trojan takes the handset, a **loan is arranged in the victim's name**, and the card's contactless data is relayed live to a criminal device ([Malwarebytes](https://www.malwarebytes.com/blog/mobile/2026/08/new-android-malware-lets-criminals-use-your-bank-card-in-real-time)). When the phone is the second factor, owning the phone is the bypass.

## Why SMS OTP fails at scale

| Property of SMS OTP | Why it breaks | Documented in |
|---|---|---|
| Possession of a *number*, not a person | A help-desk process can rebind it | Kariuki; SEC |
| Delivered over a channel you don't run | No visibility of a swap or port | Kariuki |
| Relayable in real time | A phishing kit proxies the page | 0ktapus |
| Readable on-device | RATs and notification listeners | SpyNote / WindRelay |
| Re-bindable via recovery | Recovery is the attack surface | FCC rule; Safaricom's block |
| Not phishing-resistant | Only origin-bound cryptography is | [CISA](https://www.cisa.gov/MFA) |

Standards bodies have downgraded it. NIST calls PSTN out-of-band verification *"restricted"* and tells verifiers to weigh *"device swap, SIM change, number porting"* before sending a code ([NIST SP 800-63B](https://pages.nist.gov/800-63-4/sp800-63b/authenticators/)). CISA's advice is the same: move to FIDO/WebAuthn, and use **number matching** where SMS must remain ([CISA](https://www.cisa.gov/MFA)). The FCC's rules, in force since July 8, 2024, force carriers to authenticate customers securely *before* a SIM change or port, and to let customers lock their accounts ([Federal Register](https://www.federalregister.gov/documents/2023/12/08/2023-26338/protecting-consumers-from-sim-swap-and-port-out-fraud)).

## A 30-line step-up gate

Authentication that survives a swap starts by treating a number re-bind as a credit event. The gate below models the Kariuki pattern.

{% raw %}
```python
from datetime import date

DAILY_LIMIT = 2_000_000        # KES -- DTB's per-day cap cited in the ruling
REBIND_AT   = date(2022, 2, 6) # SIM swapped; customer reported it the same day

# (date, amount KES, channel, beneficiary is new)
TXNS = [
    (date(2022, 2, 6), 1_950_000, "mobile",   True),
    (date(2022, 2, 7), 1_950_000, "pesalink", True),
    (date(2022, 2, 8),   518_601, "mobile",   True),
]

def signals(txn_date, amount, new_benef):
    flags = []
    if amount >= 0.90 * DAILY_LIMIT:        # dressed to sit just under the cap
        flags.append("crowds daily limit")
    if new_benef:
        flags.append("new beneficiary")
    age = (txn_date - REBIND_AT).days
    if age <= 7:                            # number still freshly re-bound
        flags.append("number re-bound %dd ago" % age)
    return flags

total = 0
for d, amt, ch, nb in TXNS:
    flags = signals(d, amt, nb)
    decision = "STEP-UP: hold + contact customer" if len(flags) >= 2 else "auto-approve"
    total += amt
    print("%s KES %9s  %-9s %d signals %s -> %s" % (d, format(amt, ","), ch, len(flags), flags, decision))

print()
print("PIN-only rule : every transaction had a valid PIN -> APPROVED, KES %s out" % format(total, ","))
print("Signal gate   : every transaction flagged -> KES 0 out pending confirmation")
print("cooling-off   : 7 days after a re-bind, >KES 500,000 needs a linked-device confirm")
```
{% endraw %}

```text
2022-02-06 KES 1,950,000  mobile    3 signals ['crowds daily limit', 'new beneficiary', 'number re-bound 0d ago'] -> STEP-UP: hold + contact customer
2022-02-07 KES 1,950,000  pesalink  3 signals ['crowds daily limit', 'new beneficiary', 'number re-bound 1d ago'] -> STEP-UP: hold + contact customer
2022-02-08 KES   518,601  mobile    2 signals ['new beneficiary', 'number re-bound 2d ago'] -> STEP-UP: hold + contact customer

PIN-only rule : every transaction had a valid PIN -> APPROVED, KES 4,418,601 out
Signal gate   : every transaction flagged -> KES 0 out pending confirmation
cooling-off   : 7 days after a re-bind, >KES 500,000 needs a linked-device confirm
```

The three amounts sum to exactly the KES 4,418,601 the court awarded. The PIN-only rule clears all three; the signal gate clears none — signal *count*, not any single rule, earns the step-up.

## Closing the window: a checklist

| Layer | Control | Failure it removes |
|---|---|---|
| Carrier | SIM-swap lock / port freeze, notifications, cooling-off after a re-bind | Kariuki, SEC, Zeila |
| Factor | FIDO2/WebAuthn or passkeys; number-matching push; never SMS alone | CISA; 0ktapus relay |
| Binding | Trust a registered device, not a phone number; re-enrol after a re-bind | SpyNote / WindRelay |
| Logic | Score re-bind age, new beneficiary, limit-crowding and weekend straddle together | The Kariuki pattern |
| Recovery | Treat number-change recovery as high risk: human review and a delay | Scattered Spider help-desk abuse |
| Shared | Alert the customer on both sides of a swap; log every number change | FCC rule |

## The bottom line

SIM swapping is not exotic: it is a procurement problem wrapped in a phone call, turning the most common MFA in mobile banking into a single point of failure nobody in the chain fully owns. The Kariuki ruling settles accountability in Kenya — the PIN was correct, and the bank still pays. The engineering answer predates the problem: stop treating a phone number as proof of a person, treat a re-bind as a fraud signal, and hold high-risk movements behind a factor the carrier cannot reassign.

The regulator's own account got swapped. Yours is one help-desk call away.

## References

- Court ruling (Jun 18, 2026) — [TechCabal](https://techcabal.com/2026/07/13/kenyan-court-holds-banks-telcos-liable-over-sim-swap-fraud/) · [Tech-ish](https://tech-ish.com/2026/07/13/safaricom-dtb-sim-swap-ruling/) · [Techweez](https://techweez.com/2026/07/13/safaricom-dtb-to-pay-sim-swap-victim/)
- [The Star — Safaricom SIM swap fraud rate](https://www.the-star.co.ke/business/kenya/2024-11-06-safaricom-cuts-sim-swap-fraud-to-40-in-every-750000-swaps)
- Zeila class action — [Nairobi Wire](https://nairobiwire.com/2022/10/man-sues-safaricom-after-losing-over-ksh490k-to-sim-swap-fraud.html) · [Daily Nation](https://nation.africa/kenya/news/safaricom-hit-by-sim-swap-fraud-class-action-suit-3991258)
- [Dawan (INTERPOL) — Kenya mobile money fraud](https://www.dawan.africa/news/interpol-kenya-emerges-as-east-africas-cybercrime-hotspot-as-mobile-money-fraud-surges)
- [Nairobi Law Monthly — SIM-swap fraud](https://nairobilawmonthly.com/authorities-crack-down-on-cyber-cafes-as-sim-swap-fraud-surges/)
- [BleepingComputer — SEC X account SIM-swap](https://www.bleepingcomputer.com/news/security/sec-confirms-x-account-was-hacked-in-sim-swapping-attack/)
- [CISA — Scattered Spider advisory AA23-320A](https://www.cisa.gov/news-events/cybersecurity-advisories/aa23-320a)
- [NBC News — MGM cyberattack cost $100M](https://www.nbcnews.com/business/business-news/cyberattack-cost-mgm-resorts-100-million-las-vegas-company-says-rcna119138)
- [Krebs — Scattered Spider SIM-swapper sentenced](https://krebsonsecurity.com/2025/08/sim-swapper-scattered-spider-hacker-gets-10-years/)
- [Help Net Security — 0ktapus](https://www.helpnetsecurity.com/2022/08/25/0ktapus-twilio-cloudflare-phishers-targets/)
- [Malwarebytes — Android NFC relay malware](https://www.malwarebytes.com/blog/mobile/2026/08/new-android-malware-lets-criminals-use-your-bank-card-in-real-time)
- [CISA — More than a Password](https://www.cisa.gov/MFA)
- [NIST SP 800-63B — Authenticators](https://pages.nist.gov/800-63-4/sp800-63b/authenticators/)
- [Federal Register — SIM-swap and port-out rules](https://www.federalregister.gov/documents/2023/12/08/2023-26338/protecting-consumers-from-sim-swap-and-port-out-fraud)

## Related posts

- [Mobile Money API Security: Daraja Integration Pitfalls](/posts/mpesa-daraja-api-pitfalls/)
- [The Middleman Problem: Aggregator and BaaS Integration Security](/posts/aggregator-baas-security-playbook/)
- [Fraud Trends, Read as a Dataset: What CBK's Numbers Actually Show](/posts/cbk-fraud-trend-analytics/)
- [The Three-Minute Insider: Privileged Access in Fintech](/posts/insider-threat-privileged-access-fintech/)
- [Fraud ML in Mobile Money: the 70-Account Loophole](/posts/fraud-ml-mobile-money/)
