# SAML Identity Provider Reconnaissance

## Overview

Many healthcare and enterprise platforms use SAML 2.0 for authentication, with either **Shibboleth** or **SimpleSAMLphp** as the identity provider (IdP). Both expose metadata endpoints and have well-known CVEs.

## Detection

### Shibboleth SP
```
Cookie: _opensaml_req_...        (OpenSAML request cookie)
Redirect to: /Shibboleth.sso/Login
```

### SimpleSAMLphp IdP
```
Cookie: SimpleSAMLSessionID=...
Path: /simplesaml/saml2/idp/SSOService.php
```

## Step 1: Find SAML Endpoints

```bash
# Shibboleth endpoints
/Shibboleth.sso/Metadata         # SP metadata (XML) — often PUBLIC
/Shibboleth.sso/Login            # SSO login (redirects to IdP)
/Shibboleth.sso/Logout           # Local logout — often PUBLIC
/Shibboleth.sso/Session          # Session info — often PUBLIC
/Shibboleth.sso/Status           # Status — may be restricted
/Shibboleth.sso/DiscoFeed        # Discovery service feed — JSON list of IdPs

# SimpleSAMLphp endpoints
/simplesaml/saml2/idp/metadata.php              # IdP metadata (XML) — often PUBLIC
/simplesaml/saml2/idp/SSOService.php            # SSO login
/simplesaml/saml2/idp/SingleLogoutService.php   # SLO
/simplesaml/module.php                          # Module interface
/simplesaml/admin                               # Admin (if enabled)
```

## Step 2: Extract Metadata

SAML metadata XML contains:
- **SP/IdP entity ID** — unique identifier
- **X.509 certificates** — for signature verification and encryption
- **SSO/SLO endpoints** — binding URLs
- **NameID formats** — how users are identified
- **Contact information** — admin emails, often real names
- **Organization info** — often internal organizational structure

```bash
# Shibboleth SP metadata
curl -sk "https://target.com/Shibboleth.sso/Metadata"

# SimpleSAMLphp metadata
curl -sk "https://idp.target.com/simplesaml/saml2/idp/metadata.php"
```

## Step 3: Analyze Certificate Info

SAML certificates reveal internal infrastructure:

```bash
# Extract certificate from metadata and decode
openssl x509 -in cert.pem -text -noout | grep -E 'Subject:|Issuer:|Not Before|Not After'
```

**What to look for:**
- **Subject/Issuer names** — internal org names, departments, locations
- **Email addresses** — admin/staff contacts (name@internal-org.com)
- **Certificate age** — old certs (5+ years) indicate poor security hygiene
- **Key size** — RSA-2048 is standard; smaller keys are weak

## Step 4: Discovery Service Feed

Shibboleth's `DiscoFeed` endpoint returns a JSON array of configured identity providers:

```json
[
  {
    "entityID": "...",
    "DisplayNames": [{"value": "IdP Name", "lang": "nl"}],
    "Descriptions": [{"value": "...", "lang": "nl"}],
    "Logos": [{"value": "https://media.internal.be/logo.jpg", ...}]
  }
]
```

This reveals:
- **All linked identity providers** (internal + external/federated)
- **Internal URLs** from logo/image paths (e.g., `media-acc.internal.be`)
- **Federation partners** (e.g., Belgian FAS eID `https://idp.iamfas.belgium.be/fas`)
- **Authentication methods** described (code cards, eID, etc.)

## Step 5: Check for Accessible SAML Endpoints

```bash
/Shibboleth.sso/Session     # 200 = reveals who is logged in
/Shibboleth.sso/Logout      # 200 = logout page accessible (confirms SAML SP)
/Shibboleth.sso/DiscoFeed   # 200 = IdP list accessible (huge info disclosure)
```

## Step 6: SimpleSAMLphp-Specific Checks

```bash
# Common SimpleSAMLphp admin paths
/simplesaml                          # 403 = exists but restricted
/simplesaml/module.php               # 502 = behind proxy
/simplesaml/module.php/admin/federation  # Federation admin

# CVEs to check:
# CVE-2023-37282 - XXE in signature verification
# CVE-2020-5301 - Timing attack on signatures
# Multiple auth bypass CVEs
```

## Pitfalls

- **Metadata is example data**: Some Shibboleth SPs serve placeholder metadata with `<-- This is example metadata only -->` — check if the cert is real or a sample
- **Session IP-binding**: SAML sessions are often bound to IP. Sessions can't be reused from a different IP.
- **Certificate expiry**: Old certs (>5 years) that haven't been rotated suggest poor maintenance — but a valid cert that's about to expire doesn't make the system vulnerable by itself
- **SAML endpoints may be proxied**: SimpleSAMLphp often sits behind nginx/Apache, so `/simplesaml` may respond differently than expected
