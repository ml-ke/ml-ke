# Coolblue Recon Notes (Jun 2026)

## Program
- Intigriti: Registered/Open, Tier 1 (€50-€2,500), Tier 2 (€25-€1,250)
- 1,218 submissions, 153 accepted (12.5%), €330 avg payout
- Rate limit: 2 req/s NL/BE/DE, 0.3 req/s others (VPN to EU IP helps)
- @intigriti.me required, self-register at coolblue.nl/registreren

## Tech Stack
- Next.js SPA on AWS CloudFront
- OIDC via IdentityServer at accounts.coolblue.nl
- Session: Coolblue-Session cookie (HttpOnly, Secure)
- CSRF: _csrfSecret cookie + x-coolblue-csrf-token header
- CDN: image.coolblue.nl, assets.coolblue.nl
- WebSocket gateway: mimir.coolblue.nl

## OIDC Discovery
- Issuer: https://accounts.coolblue.nl
- Auth endpoint: /connect/authorize (PKCE S256, response_type=code)
- Token endpoint: /oauth/token
- JWKS: RS256 signing
- Grant types: authorization_code, refresh_token, token-exchange
- Custom scopes: openid:customerid, openid:identityroleid, ucp:scopes:checkout_session
- JWT claims: sub, email, email_verified, customer_id, identity_role_id, given_name

## WebSocket (mimir.coolblue.nl)
- Root (/) - connects but needs correct route message format (400 otherwise)
- /ws, /api, /events, etc. - all 403 without auth (need session cookie/token)
- Responds with JSON: {"statusCode":400,"message":"Could not determine route"}
- Route format not yet determined

## Auth Flow
- Two-step: email first, then password page
- Hidden forms for password reset and passwordless login
- reCAPTCHA on login page (2 iframes detected)
- Browser automation needed for CAPTCHA

## Wanted Findings
1. Order flow exploits → free/discounted products
2. Customer PII exposure (emails, addresses, order history)
3. Unauthorized infrastructure/database access

## OOS
- API key disclosure without proven impact
- Leaked/found credentials
- Blind SSRF without proven impact
- Subdomain takeover without actually taking over
- Missing cookie flags / security headers
- Account enumeration
