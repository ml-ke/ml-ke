# Chime Bug Bounty — Program Intelligence

## Program Overview

- **Platform**: Bugcrowd
- **URL**: https://bugcrowd.com/engagements/chime
- **Started**: May 06, 2025
- **Scope rating**: 4/4 (highest)
- **Payouts**: P1 $10K-$20K, P2 $4.5K-$5K, P3 $250-$500, P4 $50-$100
- **Vulnerabilities rewarded**: 18
- **Validation**: ~4 days triage
- **Average payout**: $125 (last 3 months — mostly P3/P4)

## Focus Areas (specifically encouraged)

- GraphQL
- Mobile (iOS/Android)
- Member data
- Money movement between accounts and/or members

## Primary Targets (P1: $10K-$20K)

| Target | Type | Access |
|--------|------|--------|
| `*.chime.com` | Web | Cloudflare JS challenge |
| `*.chimepayments.com` | API | Cloudflare |
| `*.1debit.com` | API | Cloudflare |
| `*.chimecard.com` | Web | Cloudflare (302 → member.chime.com) |
| `*.chmfin.com` | API | Cloudflare |
| `*.chimebank.com` | API | Cloudflare |
| `www.chime.com` | Web | Cloudflare 403 |
| `app.chime.com` | Web | Cloudflare 307 → /login |
| Android App (Prod + Beta) | Mobile | Play Store |
| iOS App (Prod + Beta) | Mobile | App Store |
| `member-qa.chime.com/enroll` | QA (HTTP!) | **Accessible** — Next.js app, no Cloudflare |
| `app-qa.chime.com/users/sign_in` | QA (HTTP!) | Cloudflare blocked |

## Secondary Targets (P1: $4.5K-$7K)

| Target | Type | Stack |
|--------|------|-------|
| `*.saltlabs.com` | Web | Cloudflare, jQuery |
| `app.saltlabs.com` | App | Cloudflare, Lodash, Ruby |
| Salt Labs Android App | Mobile | Play Store |
| Salt Labs iOS App | Mobile | App Store |
| `app.staging.saltlabs.com` | Staging | Cloudflare, Lodash, Ruby |

## Out of Scope

- `chime.financial`
- `chimescholars.org`
- Non-Chime owned assets

## Access Status (as of June 2026)

- **All primary web targets** are behind Cloudflare JS challenge. Browser + curl blocked without residential proxies.
- **QA enrollment** (`member-qa.chime.com/enroll`) is accessible — Next.js app with assets from `enrollment-web-app-assets.prod-ext.chmfin.com`. Serves a React enrollment form.
- **No public GitHub repos** found for Chime's core banking app (closed-source).
- **API endpoints** `api.chime.com`, `api.chimepayments.com` — DNS likely unresolvable or Cloudflare-gated.

## Recon Notes

- Chime is a US fintech (neobank) — requires US residence for real account creation
- @bugcrowdninja.com email works for Bugcrowd signup
- The TechCrunch / open banking tracker shows Chime has a GraphQL API but endpoint is internal
- Next.js (from QA HTML): `_next/static/chunks/` pattern confirmed
- CDN domain: `enrollment-web-app-assets.prod-ext.chmfin.com`
- Stack traces reveal: Next.js, React, `Source Sans Pro` font
- No Cloudflare bypass found — all meaningful surfaces require residential proxies
