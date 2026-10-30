# Auth0 by Okta — Bugcrowd Recon (Jun 2026)

## Program Details
- **Platform**: Bugcrowd
- **URL**: bugcrowd.com/engagements/auth0-okta
- **Max Payout**: P1 $10,000-$50,000 (Tier 1), P1 $5,000-$15,000 (SDK/Tier 2)
- **Average Payout**: $1,052.77 (last 3 months)
- **Validation**: 8 days (75% accepted/rejected within 8 days)
- **Status**: Ongoing since Apr 2024
- **Disclosure**: Non-disclosure (no public disclosure allowed)

## In-Scope Targets (Tier 1)
- **config.cic-bug-bounty.auth0app.com** (Website) — Researcher testing environment
- **manage.cic-bug-bounty.auth0app.com** (Management Dashboard) — ReactJS
- ***.cic-bug-bounty.auth0app.com** (Website)
- **Auth0 Guardian Android/iOS** (Mobile apps)
- **marketplace.auth0.com** (Website)
- **MFA Integrations**: dashboard.fga.dev, api.us1.fga.dev, customers.us1.fga.dev, play.fga.dev

## SDK Targets (P1 up to $15,000)
Auth0.js, Lock, auth0-spa-js, Auth0.Net, nextjs-auth0, auth0-java, react-native-auth0, auth0-php

## Tier 2 Targets
auth0.com, samltool.io, webauthn.me, openidconnect.net, jwt.io, auth0.net

## Key Out-of-Scope (won't bounty)
- auth0.auth0.com, manage.auth0.com, accounts.auth0.com — immediately OOS
- Customize Login Page XSS, Race conditions for limits, CSRF on anonymous forms
- Clickjacking, Host Header Redirect without impact, Missing security headers
- SPF/DMARC/DKIM, SSL issues, Rate limiting, Username enumeration
- GitHub Actions vulns, Double-dipping (reusing private program reports)

## Credentials & Testing Environment
- **Get Credentials** button at bottom of program page → 3 sets of credentials (3 users, 3 tenants)
- Test ONLY on `manage.cic-bug-bounty.auth0app.com`
- No automated scanning > 5 req/s
- Researcher tenants may be deleted at any time

## April Bonus Program (Apr 9 - May 9 2026)
**Double-check if still active when hunting.**

1. **Brand Customization — Emails**: Cross-tenant email template access, reading/modifying server-side files via templates, privilege escalation, bypassing Liquid escaping functions (SSTI). Requires external SMTP server to test.
   - Docs: auth0.com/docs/customize/email, auth0.com/docs/api/management/v2/email-templates
   - Scope: `sfcc.shopper-myaccount.paymentinstruments.rw`
   
2. **Enterprise Connections Authentication Bypass**: Must involve one of Auth0's Enterprise connector IdPs (SAML, OIDC, Azure AD, Google Workspace).
   - Key research: PortSwigger "The Fragile Lock: Novel Bypasses For SAML Authentication" (Dec 2025)
   - Known Auth0 bypass: Sentor Security (Laban Sköllermark) authentication bypass allowing user impersonation

### Bonus Multipliers
P1: 3x, P2: 2x, P3: 1.5x, P4: 1x (no multiplier)

## Auth0 Tenant Endpoints Found
- auth0.us.auth0.com — OIDC (standard scopes, registration endpoint disabled)
- auth0.eu.auth0.com — EU tenant
- auth0.au.auth0.com — AU tenant
- bugcrowd.auth0.com — Bugcrowd's own Auth0 tenant
- /co/authenticate — Cross-Origin Authentication (needs valid web origin)
- /oauth/token — Token endpoint (needs valid credentials)
- /oidc/register — Dynamic client registration (disabled)
- /dbconnections/signup — Database signup (connection-specific)
- Management API v2: /api/v2/clients (401), /api/v2/email-templates

## Focus Areas (from program brief)
- Identity protocol vulnerabilities (OAuth 2.0, OIDC, SAML)
- Authentication/authorization bypass
- PII exfiltration
- Cross-tenant escalation of privilege

## Attack Vectors to Test
1. **Liquid SSTI in email templates** — Bypass Liquid escaping, execute server-side code
2. **Cross-tenant email template access** — Read/modify another tenant's templates via Management API
3. **Enterprise connection bypass** — SAML XML signature wrapping, OIDC parser differentials
4. **Management API privilege escalation** — Limited admin to full admin via API endpoints
5. **Session metadata manipulation** — PATCH /api/v2/sessions/{id}
6. **Rules/Hooks deprecation** — Being removed Nov 2026, legacy code may have bugs

## Documentation Links
- Authentication API: auth0.com/docs/api/authentication
- Management API v2: auth0.com/docs/api/management/v2
- Management Dashboard: auth0.com/docs/dashboard
- FGA Documentation: docs.fga.dev
