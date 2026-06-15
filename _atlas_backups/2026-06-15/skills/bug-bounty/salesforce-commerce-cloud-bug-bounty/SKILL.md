---
name: salesforce-commerce-cloud-bug-bounty
description: "Salesforce Commerce Cloud (SFCC / Demandware) bug bounty methodology. Covers instance identification, OCAPI/SCAPI probing, SLAS JWT analysis, WebDAV discovery, storefront pipeline testing, and common attack vectors (IDOR, promo abuse, gift certs, payment instrument manipulation)."
version: 1.0.0
author: ATLAS
category: bug-bounty
---

# Salesforce Commerce Cloud (SFCC) Bug Bounty Methodology

## Overview

Salesforce Commerce Cloud (formerly Demandware) is a cloud-based e-commerce platform. Many European retailers use it (Torfs, Coolblue, etc.). SFCC instances have a characteristic fingerprint and several API surfaces: the older OCAPI (Open Commerce API), the newer SCAPI (Shopper Commerce API), and the SLAS (Shopper Login & API Access Service) JWT-based auth.

## 1. Fingerprinting SFCC

### Cookies
- `dwsid` — Base64 session ID (always present)
- `sid` — Session ID
- `cc-at_{Site-ID}` — SLAS JWT (guest or user token). Format: ES256-signed JWT
- `cc-nx-g_{Site-ID}` — Session nonce
- `usid_{Site-ID}` — User session ID (UUID)
- `dwanonymous_{hash}` — Anonymous user tracking (15552000s = 180 day expiry)

### Headers
- `x-dw-request-base-id` — SFCC request tracing header
- `x-dw-*` — Custom Demandware headers

### Paths
- `/on/demandware.store/` — Storefront pipeline URLs
- `/on/demandware.static/` — Static assets
- `/on/demandware.servlet/webdav/` — WebDAV (file management)
- `/dw/shop/{version}/` — OCAPI Shop API
- `/dw/` — OCAPI root
- `/-/dw/` — Alternative API base
- `/webref` — Developer reference (sometimes enabled)
- `/waroot/` — WAR root assets

### Server Header
- Often behind Cloudflare. The `set-cookie` domain reveals the actual backend.

## 2. OCAPI (Open Commerce API) Probing

### Version Discovery
SFCC versions follow the pattern `v{major}_{minor}` (e.g., `v19_1`, `v25_5`). Probe them:

```
GET /dw/shop/{version}/products?client_id={id}
```

- **404** = Version not enabled
- **400** with `MissingClientIdException` = Version EXISTS, needs a client_id
- **400** with `UnknownClientIdException` = Client_id is wrong but version works
- **400** with `ResourcePathNotFoundException` = Client_id valid, wrong resource path
- **405** (MethodNotAllowed) = Resource exists, wrong HTTP method

### Client ID Discovery
SFCC uses different client_ids for different resource groups. Common patterns:
- `{Site-ID}` (e.g., `Torfs-Webshop-BE`)
- `{Site-ID}` with underscores (e.g., `Torfs_Webshop_BE`)
- Short hash strings visible in JS bundles
- JS bundles at `/on/demandware.static/Sites-{Site-ID}/.../js/`

### Resource Paths (v25_5 tested)
| Path | Method | Notes |
|------|--------|-------|
| `/baskets` | POST | Create basket (needs auth) |
| `/customers/self` | GET | Current customer (needs auth) |
| `/orders` | GET | List orders |
| `/product_search` | GET/POST | Product search |
| `/categories` | GET | Category listing |
| `/promotions` | GET | Promotions (may need different client_id) |
| `/sessions` | POST | Create session |
| `/customers` | POST | Register customer (needs auth) |
| `/gift-certificates` | GET | Gift certificates |

### Auth for OCAPI
OCAPI requires `Authorization: Bearer *** header with a JWT. Guest JWTs from the `cc-at` cookie DO NOT work with OCAPI — they need a different token type (OCAPI-specific JWT or SLAS token exchanged through the proper flow).

## 3. SCAPI / SLAS (Shopper Commerce API)

SCAPI is the newer API, accessed through Salesforce's CDN infrastructure (not directly on the storefront domain).

### SLAS JWT Format
The `cc-at_{Site-ID}` cookie contains an ES256-signed JWT with:

**Header:**
```json
{
  "ver": "1.0",
  "jku": "slas/prod/{org}",
  "kid": "{uuid}",
  "typ": "jwt",
  "clv": "J2.3.4",
  "alg": "ES256"
}
```

**Payload key claims:**
- `scp` — Space-separated list of scopes (e.g., `sfcc.shopper-baskets-orders.rw`)
- `sub` — Subject: `cc-slas::{org}::scid:{slas_client_id}::usid:{user_id}`
- `isb` — User info bundle: `uido:ecom::upn:{email}::uidn:{name}::gcid:{global_customer_id}::rcid:{registered_customer_id}::sesb:session_bridge::chid:{site_id}`
- `sty` — Session type: `Guest` or `User`
- `aud` — Audience: `commercecloud/prod/{org}`
- `iss` — Issuer: `slas/prod/{org}`
- `exp` / `iat` — Expiry / issued at (30 min typical, 1800s Max-Age)

### SCAPI Scopes (from Torfs)
```
sfcc.shopper-myaccount.baskets
sfcc.shopper-discovery-search
sfcc.shopper-myaccount.paymentinstruments
sfcc.shopper-customers.login
sfcc.shopper-myaccount.orders
sfcc.shopper-productlists
sfcc.shopper-promotions
sfcc.shopper.stores
sfcc.orders.rw
sfcc.session_bridge
sfcc.shopper-myaccount.paymentinstruments.rw
sfcc.shopper-myaccount.productlists
sfcc.shopper-categories
sfcc.shopper-myaccount
sfcc.shopper-myaccount.addresses
sfcc.shopper-products
sfcc.shopper-myaccount.rw
sfcc.shopper-baskets-orders
sfcc.shopper-customers.register
sfcc.shopper-myaccount.addresses.rw
sfcc.shopper-myaccount.productlists.rw
sfcc.shopper-product
sfcc.shopper-baskets-orders.rw
sfcc.shopper-gift-certificates
sfcc.shopper-product-search
```

### SCAPI Gateway
SCAPI endpoints follow the pattern:
```
/api/shopper/{resource}/v1/organizations/{org}/{path}
```

But the API gateway is on Salesforce's CDN (`.api.commercecloud.salesforce.com`), not on the storefront domain. The storefront domain serves the SPA shell HTML for any `/api/*` path. To find the real gateway, check:
- The `jku` claim in the JWT header (JWKS URL)
- The `aud` claim (audience)
- JS bundles for API base URLs

## 4. WebDAV

SFCC exposes WebDAV for file management:

| Path | Typical Response | Meaning |
|------|-----------------|---------|
| `/on/demandware.servlet/webdav/Sites/Cartridges` | 401 (BASIC auth) | Exists, needs credentials |
| `/on/demandware.servlet/webdav/Sites/Impex` | 401 (BASIC auth) | Import/export access |
| `/on/demandware.servlet/webdav/Sites/Code` | 403 | Exists but explicitly blocked |
| `/on/demandware.servlet/webdav/Sites/Static` | 401 | Static files |

The 401 response includes `www-authenticate: BASIC realm="{host}/on/demandware.servlet"` — standard HTTP Basic auth.

## 5. Storefront Pipeline URLs

SFCC uses pipeline-based URLs for server-side rendering:
```
/on/demandware.store/Sites-{Site-ID}/{locale}/{Pipeline-Name}
```

Common pipelines:
| Pipeline | Purpose |
|----------|---------|
| `Login-Form` | Login page |
| `Register-Form` | Registration form |
| `Login-Show` | Login display |
| `Account-Show` | Account page |
| `Cart-Show` | Shopping cart |
| `Product-Show` | Product detail |
| `Search-Show` | Search results |
| `Customer-Show` | Customer info |
| `Logout` | Logout |
| `Page-Show` | Content page |
| `ConsentTracking-SetSession` | Cookie consent |

Direct access to these pipelines without proper session/CSRF context often returns 500 errors (which may leak info).

## 6. Common Attack Vectors

### IDOR on Orders/Baskets
- Test if you can access another user's basket/order by changing IDs
- Baskets are typically associated with the session, not the user ID
- Check if order confirmation pages expose other customer PII

### Promotions / Voucher Abuse
- Check for stackable discounts (multiple promo codes)
- Check if promo codes can be reused
- Check if promo codes are enumerable
- SFCC has `sfcc.shopper-promotions` scope for this

### Gift Certificate Manipulation
- Test if gift certificate balance can be manipulated
- Check if gift certificates are enumerable
- Test gift certificate application to another user's basket

### Payment Instrument Access
- With `sfcc.shopper-myaccount.paymentinstruments.rw` scope, test:
  - Can you read another user's saved payment methods?
  - Can you add/remove payment methods to another account?

### Pricing Manipulation
- Check if product prices are validated server-side or just sent from the client
- SFCC typically validates prices server-side but edge cases exist (bundles, tiered pricing)

### Cross-tenant / Multi-site
- All domains on the same SFCC instance share the same codebase
- A vulnerability found on one domain works on all (but is a duplicate)
- Test if sessions/users from one domain can access data on another domain

### Account Registration / Login
- Registration requires `@intigriti.me` email for Intigriti programs
- Check for auto-login after registration (session fixation)
- Check for email verification bypass

### Storefront AJAX Endpoints
- Look for AJAX endpoints in JS files that return JSON data
- Test for missing auth on AJAX endpoints that return customer data
- Test for CSRF on state-changing AJAX calls

### CSRF Token Discovery
SFCC storefront pages embed CSRF tokens in the HTML. Look for:
```html
<input type="hidden" id="csrf" name="_csrf" value="{token}"/>
```
And in URLs like:
```
Address-SetDefault?addressId=homeaddress&csrf_token={token}
Address-DeleteAddress?addressId=homeaddress&csrf_token={token}
```
CSRF tokens are used for state-changing operations on addresses, baskets, and account settings.

### Address IDOR Testing
SFCC addresses are identified by user-chosen names (not sequential IDs):
- `homeaddress`, `Neighbor`, `billing`, `shipping` — common patterns
- Test `Address-EditAddress?addressId={name}` — returns 200 for valid IDs, 500 for invalid
- Test `Address-DeleteAddress?addressId={name}&csrf_token={token}` — state change
- Test `Address-SetDefault?addressId={name}&csrf_token={token}` — state change
- Since IDs are user-named, direct enumeration isn't feasible, but check for:
  - IDOR on shared/guessable names like "default" or "main"
  - CSRF token reuse across sessions
  - Missing CSRF validation on address operations

### Authentication State Data Attributes
The HTML contains data attributes indicating auth state:
```html
data-authenticated="true|false"
data-registered="true|false"
data-user="{&quot;name&quot;:&quot;Hodari&quot;,&quot;email&quot;:&quot;user@email.com&quot;}"
data-remainingorders="0"
data-pagename="my-addresses|my-orders"
data-action="Address-List|Order-History"
```

### Pipeline Discovery from HTML
Pipeline names are embedded in data attributes and JS:
```html
data-action="Address-List|Order-History|Customer-Show"
```

JS bundles reveal additional pipelines:
- `CQRecomm-Start` — Recommendations engine (Certona/CQLive)
- `ConsentTracking-SetSession` — Cookie consent
- `Product-Ratings` — Product ratings
- `SearchServices-GetSuggestions` — Search suggestions (with query parameter)
- `Page-SetLocale` — Locale switching
- `Page-HeaderFavoriteStore` — Favorite store header

### Cookie Lifecycle

| Cookie | Pattern | Max-Age | Notes |
|--------|---------|---------|-------|
| `cc-at_{Site-ID}` | SLAS JWT (ES256) | 1800s (30 min) | Auth token, refreshed on each request |
| `cc-nx-g_{Site-ID}` | Nonce | 2592000s (30 days) | Session nonce |
| `cc-nx_{Site-ID}` | Nonce | varies | Secondary nonce |
| `usid_{Site-ID}` | UUID | 2592000s (30 days) | User session identifier |
| `dwsid` | Base64 | Session | Demandware session ID |
| `sid` | Alphanumeric | Session | Session ID |
| `dwanonymous_{hash}` | Base64 | 15552000s (180 days) | Anonymous user tracking |
| `__cq_dnt` | "1" | Session | Do not track |
| `dw_dnt` | "1" | Session | Do not track |

### SLAS JWT `isb` Claim Decode
The `isb` (user info bundle) claim is a colon-delimited string:
```
uido:ecom::upn:{email}::uidn:{display_name}::gcid:{global_customer_id}::rcid:{registered_customer_id}::sesb:session_bridge::chid:{site_id}
```

Key fields:
- `upn` — User Principal Name (email)
- `uidn` — User Display Name
- `gcid` — Global Customer ID
- `rcid` — Registered Customer ID
- `chid` — Channel/Site ID

### Basket Data Extraction from HTML
Basket contents are rendered as `data-` attributes:
```html
data-pid="37564105"
data-name="Sneakers wit"
```
Search for these in the page HTML with:
```python
import re
re.findall(r'data-pid="([^"]*)"[^>]*data-name="([^"]*)"', html)
```

## 7. References

- `references/torfs-recon-session.md` — Full recon session on Torfs (Intigriti)
- `references/sfcc-ocapi-endpoints.md` — Full OCAPI endpoint reference
