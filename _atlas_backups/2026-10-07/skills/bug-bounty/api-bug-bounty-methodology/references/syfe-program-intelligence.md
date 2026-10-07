# Syfe BBP — Program Intelligence (Worked Example)

Applied the Step 0 Program Intelligence + 0F/0G/0H recon workflow to `hackerone.com/syfe_bbp`.

## Program Meta

- **Launched**: March 2025
- **Response efficiency**: 88%
- **Reports received (90 days)**: 863 | **Resolved**: 21 (2.4% hit rate)
- **Total bounties paid**: <$10,000
- **Average response**: 1.4d | **Triage**: 2.3d | **Bounty**: 2w1d
- **Bounties**: Low $50-75, Med $150-250, High $500-750, Crit $1,000-1,500
- **Top hackers**: mrrhacker (57 rep), venum163 (57), hamzaydia (57), lady_block (57), 0d_samii (44)

## Scope — 9 Assets (all Critical, bounty-eligible)

| Asset | Type | Reports | Notes |
|-------|------|---------|-------|
| `www.syfe.com` | Web | 5 (24%) | Main app — Next.js SPA on Webflow hosting |
| `uat-bugbounty.nonprod.syfe.com` | UAT Web | 3 (14%) | Test env — needs credentials |
| `mark8.syfe.com` | API | 1 (5%) | Market data API — returns 401 "Unauthorized" JSON |
| `api.syfe.com` | API | — | Main API — 403 Cloudflare WAF on everything |
| `api-uat-bugbounty.nonprod.syfe.com` | API UAT | — | UAT API — 404 on root paths |
| `alfred.syfe.com` | API | — | "Alfred" service — returns 401 like mark8 |
| `alfred-uat-31.nonprod.syfe.com` | API UAT | — | Alfred test env |
| iOS App (`id1497156434`) | Mobile | 0 — untapped | App Store link |
| Android (`com.syfe`) | Mobile | 0 — untapped | Play Store |

**Untapped gold**: iOS + Android have 0 resolved reports. If user can provide APK/IPA, significant opportunity.

## Tech Stack

All production domains behind **Cloudflare** (104.18.x.x).

- **CDN/Proxy**: Cloudflare (cf-ray, server: cloudflare, x-frame-options, strict-transport-security)
- **Frontend**: Next.js SPA on Webflow. Scripts served from `/_next/static/chunks/*.js`
- **Webflow hosting**: surrogate-key: webflow.*, x-wf-region: us-east-1, _cfuvid cookie with Domain=webflow.io
- **Backend proxy**: OpenResty (seen in 405 responses: `openresty` server header)
- **Storage**: AWS S3 (`stable-production-v1-*-bucket.s3.amazonaws.com` in CSP)
- **Email**: Google Workspace (MX → Google, SPF includes _spf.google.com, sendgrid.net, amazonses.com, mail.zendesk.com)
- **SSO**: Google SAML (`DirectFedAuthUrl → accounts.google.com/o/saml2`)
- **Support**: Zendesk (`help.syfe.com → syfe.zendesk.com` — OUT OF SCOPE, third-party)
- **Chat**: Intercom (widget.intercom.io, js.intercomcdn.com)
- **Monitoring**: Datadog RUM, Sentry
- **Analytics**: Hotjar, Google Analytics, TikTok, Facebook, LinkedIn, Taboola, Outbrain, Bing, Reddit, Twitter
- **Product Analytics**: Pendo, Optibase
- **Scheduling**: Calendly, Hubspot
- **Auth Providers**: Apple Sign-in, Google Sign-in, Passkeys, TOTP MFA
- **Deep linking**: Firebase Dynamic Links (`app.syfe.com` → Fastly)
- **Sentry DSN**: `https://d0412411061642b0b440187091f15bf9@o216759.ingest.us.sentry.io/1358243`
- **Pendo API key**: `a1c8b92f-44f8-442c-8752-458ca2e07835`
- **Google Tag Manager**: GTM-PK3J8HX
- **Zipkin trace endpoint**: `http://localhost:9411/api/v2/spans` (leaked in JS bundle)
- **Recaptcha**: Google reCAPTCHA Enterprise on login

### Config Leak (from Webflow page HTML source)

Found embedded in the `<script>` tags of the Webflow page at `/api/auth/logout` (which returned the page HTML instead of JSON):

```javascript
var DOMAIN = 'https://api.syfe.com';
var BASEAPI_URL = DOMAIN;
var GTM_ID = 'GTM-PK3J8HX';
var DATADOG_APPLICATION_ID = '62056480-273a-4b69-80ca-b6be526a67b8';
var DATADOG_CLIENT_TOKEN = 'pubfd5...85ff';
var ENVIRONMENT_NAME = 'staging';  // Even on production!
```

Notable: `ENVIRONMENT_NAME = 'staging'` on the production site — potential misconfiguration signal.

### Notable CSP Leaks

- `connect-src 'self' data: *` — ALL origins allowed for XHR/fetch (potential SSRF or exfiltration vector)
- `frame-ancestors: 'self'` — clickjacking protection present
- CSP includes dozens of marketing/ad/tracking domains in `script-src`, `img-src`, `frame-src`, `connect-src` — every one of these is an integration that could be an attack surface

## Next.js App Architecture

### Build Info
- Build ID: `ryrZCxN16DLngGKWdW70x` (from `_buildManifest.js`)
- Key chunks (27 total): `5912-906a1b2e830dc814.js` (20KB — API service layer), `pages/_app-*.js` (4MB — all app logic), `pages/login-*.js` (2KB — thin wrapper)

### SSG Data Endpoints (Next.js getStaticProps)
```
/_next/data/ryrZCxN16DLngGKWdW70x/login.json  → 200
/_next/data/ryrZCxN16DLngGKWdW70x/dashboard.json → 200
/_next/data/ryrZCxN16DLngGKWdW70x/portfolio.json → 200
/_next/data/ryrZCxN16DLngGKWdW70x/account.json → 200
```

## API Endpoint Map (from JS Bundle Analysis)

All extracted from `_app.js` (4MB bundle) — paths are relative, served through Next.js proxy at `www.syfe.com/api/*`.

### Auth Endpoints
| Path | Method | Notes |
|------|--------|-------|
| `/auth/login` | POST | 405 on GET (endpoint exists) |
| `/auth/signup` | POST | Registration |
| `/auth/logout` | GET | Session termination |
| `/auth/forgot-password` | POST | Password reset |
| `/auth/reset-password` | POST | |
| `/auth/reset-password/generate-otp` | POST | |
| `/auth/pre-signup` | POST | Pre-registration check |
| `/auth/resend-verification-link` | POST | |
| `/auth/subscribe-user` | POST | |
| `/auth/register-for-feature` | POST | |
| `/auth/passkey/register/options` | GET | WebAuthn registration |
| `/auth/passkey/register/verify` | POST | |
| `/auth/passkey/authenticate/options` | GET | WebAuthn auth |
| `/auth/passkey/authenticate/verify` | POST | |
| `/auth/passkey/credentials` | GET | List registered passkeys |
| `/auth/mfa/available-otp-modes` | GET | MFA discovery |
| `/auth/mfa/generate-otp` | POST | OTP generation |
| `/auth/mfa/validate-otp` | POST | OTP validation |
| `/auth/totp/qrcode` | GET | TOTP QR |
| `/auth/totp/setup` | POST | Setup |
| `/auth/totp/status` | GET | Status check |
| `/auth/totp/verify` | POST | Verify TOTP |
| `/auth/email-verification/generate-otp` | POST | |
| `/auth/email-verification/validate-otp` | POST | |
| `/auth/mobile-verification/generate-otp` | POST | |
| `/auth/mobile-verification/validate-otp` | POST | |

### Account Endpoints
| Path | Notes |
|------|-------|
| `/account/user-info` | User profile data |
| `/account/portfolio/` | Portfolio info |
| `/account/portfolio-details/` | Detailed portfolio |
| `/account/dashboard/kyc-details` | KYC status |
| `/account/virtual-account` | Virtual bank account |
| `/account/withdrawal-config` | Withdrawal settings |
| `/account/transfer-plan` | Transfer plan |
| `/account/deposit-bank-details` | Deposit info |
| `/account/funds-in-transit` | Pending funds |
| `/account/update-details/generate-otp` | Update OTP |
| `/account/update-details/validate-otp` | Update validation |
| `/account/meta-data` | Account metadata |
| `/account/user-rebalance-consent` | Rebalance consent |
| `/account/updateDocuments` | Document upload |
| `/account-closure-notice` | Account closure |

### Portfolio & Trade Endpoints
| Path | Notes |
|------|-------|
| `/portfolio` | Single portfolio |
| `/portfolios` | All portfolios |
| `/portfolios/data/custom` | Custom portfolio data |
| `/portfolios/historical-navs/` | Historical NAV |
| `/portfolios/envision-meta-data` | Envision metadata |
| `/portfolios/lock-in` | Lock-in info |
| `/portfolios/carousels` | Portfolio carousel |
| `/portfolios/landing-cards` | Landing page data |
| `/trade/pricing` | Trade pricing |
| `/trade/pricing/pricing-details` | Pricing details |
| `/composition/edit` | Portfolio composition |
| `/acknowledge-rate` | Rate confirmation |

### Promo & Referral Endpoints
| Path | Notes |
|------|-------|
| `/promo/validate/{code}` | Promo code validation — **PUBLIC on api-au** |
| `/promo/history` | Promo history (needs auth) |
| `/promo/referral/` | Referral info |
| `/promo/avail/` | Available promos |
| `/invite/trade` | Trade invite |
| `/invite/wealth` | Wealth invite |

### Admin/Privileged Endpoints
| Path | Notes |
|------|-------|
| `/advisor/fetch-user-token` | **HIGH VALUE** — fetches user token as advisor |
| `/advisor/login` | Advisor login |
| `/advisor/user-list` | List users |

### Other Endpoints
| Path | Notes |
|------|-------|
| `/myinfo/get-data` | Singapore MyInfo (government data API) |
| `/events/subscribe` | Event subscription |
| `/mandates/banks` | Bank mandates |
| `/poll/get-questions/` | Polling |
| `/poll/post-answers` | Poll answers |
| `/setDepositReminder` | Deposit reminders |
| `/api/transfer-plans` | Transfer plans API route |
| `/v1/logs` | Logging |
| `/v1/traces` | Tracing (Zipkin? Jaeger?) |
| `/status` | Status endpoint |

## Region-Specific API Subdomain Analysis

Three API regions discovered from the JS bundle:
```
api.syfe.com      → Main (SG/Singapore)
api-au.syfe.com   → Australia
api-hk.syfe.com   → Hong Kong
```

### Behaviour Comparison

| Endpoint | api.syfe.com | api-au.syfe.com | api-hk.syfe.com |
|----------|-------------|-----------------|-----------------|
| `/auth/login` | 403 (text/html — CF block) | 405 (JSON — wrong method) | 403 (text/html — CF block) |
| `/auth/signup` | 403 (text/html) | 404 (JSON — not found) | 403 (text/html) |
| `/promo/validate/TEST123` | 403 (text/html) | 200 (JSON — unauthenticated!) | 403 (text/html — CF challenge) |
| `/promo/history` | 403 (text/html) | 401 (JSON — needs auth) | 403 (text/html) |

**Key finding**: `api-au.syfe.com` has substantially weaker Cloudflare WAF rules than the main and HK regions. The promo validation endpoint is fully public and unauthenticated on AU. The AU region returns real JSON error responses (code 6000 format) while SG and HK return Cloudflare HTML blocks.

### Promo Validation Endpoint (AU Region — Unauthenticated)
```
GET https://api-au.syfe.com/promo/validate/{code}
→ 200 {"message":"Invalid code. Check that you are using an active one."}
```
- No authentication required
- Returns descriptive error messages
- Accepts GET only (POST returns 405)
- All test codes returned "Invalid code" — need a real valid promo code to test further

### mark8.syfe.com — Market Data API
```
GET https://mark8.syfe.com/v1
→ 401 {"success":false, "message": "Unauthorized"}
```
- Endpoint exists and responds
- Needs valid API key / authentication token
- 401 (not 403) means the Cloudflare WAF allows the request through to the backend

### alfred.syfe.com — "Alfred" Service
```
GET https://alfred.syfe.com/v1
→ 401 (same pattern as mark8)
```
- Endpoint exists, requires auth

## Other Subdomains

| Subdomain | Infrastructure | Notes |
|-----------|---------------|-------|
| `app.syfe.com` | Fastly (151.101.1.195) | Firebase Dynamic Links — returns "Invalid Dynamic Link" |
| `assets.syfe.com` | AWS S3 (Cloudflare fronted) | Delete marker on index.html — bucket versioned, contents may be enumerable |
| `help.syfe.com` | Zendesk | Out of scope per program rules (third-party hosted) |
| `blog.syfe.com` | Cloudflare | Blog subdomain |
| `help.au.syfe.com` | Zendesk | AU help center |
| `help.hk.syfe.com` | Zendesk | HK help center |

## Auth Mechanism

- **Token storage**: JWT in `localStorage['token']` with expiry in `localStorage['validTill']`
- **Redux state**: Auth reducer stores `access_token`, `valid_till`, and user object
- **MFA**: TOTP and OTP-based MFA available alongside passkey (WebAuthn)
- **Recaptcha**: Google reCAPTCHA Enterprise on login form (`6LdhG2QlAAAAAE6v5yXTxJCdWWmcAqOpxb1asNZg`)

## Key Rules

1. **Header required**: `X-HackerOne-Research: [H1 username]` on all production requests
2. **Signup**: Must use @wearehackerone.com email on production
3. **UAT findings**: Must be reproducible on production to qualify
4. **UAT credentials**: May not be available to new researchers per HackerOne guidelines

## Non-Qualifying Bugs (Do NOT waste time on)

- DoS on Syfe infrastructure
- Missing SSL/TLS best practices
- Missing SPF/DKIM/DMARC
- Rate limiting (unless leads to ATO or financial damage)
- Open redirect (unless chained with additional impact)
- Self-XSS / File Upload XSS
- CORS (unless you can exfiltrate data)
- Exposed API keys without working PoC
- UAT-only information disclosure
- Known vulnerable libraries without working PoC
- Root detection / jailbreak / SSL pinning bypass
- Debug view / login to any account on UAT
- Promo code enumeration
- Whitelisted response manipulation (UI-only changes)
- Pasting auth token from another account
- Promo code URLs on web archive

## Disqualifiers

- Brute forcing or automated attacks on production
- Modification or destruction of user data
- Social engineering of any kind
- Denial of service
- Overwhelming support team

## Attack Surface Prioritization

| Priority | Surface | Rationale | Effort |
|----------|---------|-----------|--------|
| 1 | **API IDOR/BOLA** — api.syfe.com, mark8.syfe.com, alfred.syfe.com | Financial app = user separation. 55% of reports are Medium. IDOR is the classic Medium. | Low (2 accounts, curl) |
| 2 | **OAuth ATO** — Apple/Google Sign-in flow | Common in apps that added social login as an afterthought. Critical payout ($1k-1.5k). | Medium (intercept + replay) |
| 3 | **Business logic** — investment/portfolio workflows | Fintech = complex state machines. Multi-step abuse, race conditions, withdrawal manipulation. | Medium (browser + scripts) |
| 4 | **Advisor endpoint abuse** — `/advisor/fetch-user-token` | Could allow token impersonation of any user. Very high impact if found accessible. | High (needs advisor account) |
| 5 | **MyInfo integration abuse** — `/myinfo/get-data` | Singapore government data. If accessible without proper auth, GDPR/PDPA implications. | Low (check endpoint) |
| 6 | **Promo/referral abuse** | Unauthenticated on AU region. Chain with referral creation. | Medium (needs valid code) |
| 7 | **SSRF** — any URL import/upload feature | Check for Webflow media upload, profile picture, document upload that fetches external URLs. | Low (probe first) |
| 8 | **S3 bucket** — `assets.syfe.com` | Check for public listing, object enumeration, misconfigured ACLs | Low (bucket scan) |

## What User Needs to Set Up

1. Register at `www.syfe.com/create-account?utm_source=bug_bounty` with @wearehackerone.com email
2. Provide the JWT token from `localStorage.getItem('token')` for authenticated API testing
3. UAT credentials (if available — send invite link or share)
4. Optional: APK from Android phone to test mobile apps (0 reports = untapped)

## Verdict & Recommendation (Session Outcome)

After full recon, **GitLab was recommended over Syfe** for the following reasons:

| Factor | Syfe | GitLab |
|--------|------|--------|
| **Bounty range** | $50–$1,500 | **$100–$35,000** |
| **Code access** | Closed (no source) | **Open source (full Ruby on Rails codebase)** |
| **Free account** | ❌ Requires KYC + funding | **✅ Free gitlab.com account, no payment** |
| **Self-host option** | ❌ | **✅ "Your Own Instance" in scope** |
| **First response** | 1.4 days | **6 hours** |
| **Resolved reports** | 21 (2.4% hit rate) | **1,000+ (mature, well-funded program)** |
| **Barrier to entry** | **HIGH** (KYC wall) | **LOW** (immediate access) |
| **SSRF hunting** | Cloudflare WAF blocked CLI | Code available for static analysis |

**Key insight**: Even with a valid session token, Syfe's account was in pre-KYC state, blocking access to portfolio/trade endpoints where the real IDOR and business logic bugs live. The program's 2.4% resolution rate (21/863) also suggests most submissions are duplicates or out-of-scope — a high-effort, low-reward ratio.

**Takeaway for future evaluations**: When a fintech program requires KYC + funding to reach the interesting attack surface, and GitLab (LOW barrier, open code, 10x bounties) is available, the choice is clear. Use Section 0I criteria to produce this comparison proactively.

**Counterpoint — if the user wants to return to Syfe later**: Complete KYC to unlock portfolio management, trade execution, withdrawals, and referral features. Those features are where IDOR and business logic bugs live. Without funded accounts to test against, the vulnerability surface is essentially zero.
