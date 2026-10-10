# Salesforce Commerce Cloud (SFCC / Demandware) Reconnaissance

## Overview

Salesforce Commerce Cloud (formerly Demandware) is a B2C e-commerce platform. It has two API generations: **OCAPI** (Open Commerce API, deprecated but still common) and **SCAPI** (Shopper API, newer). Most SFCC instances also expose **WebDAV** for cartridge management, and **SLAS** (Shopper Login & API Access Service) for authentication.

## Architecture

```
Storefront:     https://www.target.com (Cloudflare)
OCAPI:          https://www.target.com/dw/shop/{version}/{resource}
SCAPI:          https://www.target.com/api/shopper/{service}/v1/organizations/{org}/{resource}
SLAS JWT:       cc-at_{site} cookie (encrypted JWT)
Sessions:       dwsid cookie, sid cookie
Anonymous:      dwanonymous_{hash} cookie
WebDAV:         /on/demandware.servlet/webdav/Sites/{Cartridges|Impex|Code|Libraries}
Storefront:     /on/demandware.store/Sites-{site}/-/{Pipeline}-{Action}
Static assets:  /on/demandware.static/Sites-{site}/-/...
```

## Step 1: Identify SFCC Instance

Look for these signatures:

```
Cookie: dwsid=...              (Session ID)
Cookie: dwanonymous_...        (Anonymous tracking)
Cookie: cc-at_{site}=...       (SLAS JWT — starts with eyJ...)
X-DW-Request-Base-ID: ...      (Request tracing)
/dw/shop/v25_5/                 (OCAPI path)
/on/demandware.store/           (Storefront pipeline)
```

## Step 2: Find OCAPI Version

```bash
# SFCC OCAPI versions use underscore notation (v19_1 through v25_5+)
# A 400 response (not 404) means the version is active
for ver in v19_1 v19_5 v20_1 v21_1 v22_1 v23_1 v24_1 v25_1 v25_5; do
  code=$(curl -sk -o /dev/null -w "%{http_code}" "https://www.target.com/dw/shop/${ver}/product_search")
  echo "${ver} -> ${code}"  # 400 = exists, 404 = doesn't exist
done
```

**Response code meanings:**
- **400** `MissingClientIdException` — Version EXISTS, needs client_id parameter
- **404** — Version does not exist
- **405** — Version exists, wrong HTTP method

## Step 3: Find Valid Client IDs

OCAPI requires a `client_id` parameter. The error message reveals the auth model:

```bash
# Try common client IDs
curl -sk "https://www.target.com/dw/shop/v25_5/products?client_id=SITE_NAME"
```

**Response code meanings:**
- `UnknownClientIdException` — Client ID not recognized
- `ResourcePathNotFoundException` — Client ID IS VALID but wrong resource path
- `MethodNotAllowedException` — Client ID IS VALID, resource exists, wrong HTTP method

**Where to find client IDs:**
- Site ID in the HTML: `Sites-{SiteID}-Site`
- JS bundles (search for `client_id` or API config)
- SFCC Business Manager config (if accessible)

## Step 4: SLAS JWT Authentication

Authenticated users get a `cc-at_{site}` cookie containing a JSON Web Token:

```json
{
  "scp": "sfcc.shopper-products sfcc.shopper-baskets-orders.rw ...",
  "sty": "User",  // "Guest" or "User"
  "sub": "cc-slas::org::scid:...::usid:...",
  "aud": "commercecloud/prod/{org}",
  "iss": "slas/prod/{org}",
  "isb": "uido:ecom::upn:user@email.com::..."
}
```

**Key JWT claims:**
- `sty`: Token type (`Guest` = anonymous, `User` = registered)
- `scp`: Space-separated list of authorized scopes
- `aud`: Audience (contains org short code like `bcqr_prd`)
- `isb`: User info (contains email, name, customer ID)
- `chid`: Site/channel ID within `isb`

**Common SCAPI scopes:**
```
sfcc.shopper-products              sfcc.shopper-product-search
sfcc.shopper-baskets-orders        sfcc.shopper-baskets-orders.rw
sfcc.shopper-myaccount             sfcc.shopper-myaccount.rw
sfcc.shopper-myaccount.addresses   sfcc.shopper-myaccount.addresses.rw
sfcc.shopper-myaccount.orders      sfcc.shopper-myaccount.paymentinstruments.rw
sfcc.shopper-customers.login       sfcc.shopper-customers.register
sfcc.shopper-promotions            sfcc.shopper-gift-certificates
sfcc.orders.rw                     sfcc.session_bridge
```

## Step 5: Test OCAPI with Valid Client ID

```bash
CID="YourSiteID"
BASE="https://www.target.com/dw/shop/v25_5"

# Resources that exist with this client ID
curl -sk "${BASE}/baskets?client_id=${CID}"        # 405 = exists
curl -sk "${BASE}/orders?client_id=${CID}"          # 405 = exists
curl -sk "${BASE}/sessions?client_id=${CID}"        # 405 = exists
curl -sk "${BASE}/customers/self?client_id=${CID}"  # 401/405 = exists
curl -sk "${BASE}/product_search?client_id=${CID}"  # 400 = might need POST
```

**Auth errors when making API calls:**
- `AuthorizationHeaderMissingException` — Need `Authorization: Bearer <token>` header
- `InvalidAccessTokenException` — Token format is wrong or expired
- `MissingClientIdException` — No client_id parameter

## Step 6: WebDAV Probe

SFCC exposes WebDAV for cartridge management (code deployment):

```bash
# Common WebDAV paths
/on/demandware.servlet/webdav/Sites/Cartridges   # 401 = exists (Basic auth)
/on/demandware.servlet/webdav/Sites/Code          # 403 = exists but blocked
/on/demandware.servlet/webdav/Sites/Impex         # 401 = exists (import/export)
/on/demandware.servlet/webdav/Sites/Libraries     # May exist
```

**Response meaning:**
- **401** with `www-authenticate: BASIC realm="..."` — Exists but needs credentials
- **403** — Exists but explicitly forbidden (different access level from 401)
- **404** — Does not exist

## Step 7: Storefront Pipeline URLs

The SFCC storefront uses pipeline endpoints:

```
/on/demandware.store/Sites-{SiteID}-Site/{locale}/{Pipeline}-{Action}
```

**Common pipelines:**
```
Customer-Show              Account-Show
Order-History              Address-List
PaymentInstruments-List    Basket-Show
Wishlist-Show              Search-Show
Product-Show               Login-Form
Register-Form              Coupon-List
GiftCertificate-List       CQRecomm-Start (recommendations)
```

**CSRF tokens** are embedded in pipeline URLs:
```
Address-DeleteAddress?addressId=xxx&csrf_token=TOKEN
```

## Step 8: Information from Robots.txt

SFCC robots.txt often reveals the pipeline pattern:

```
Disallow: /on/demandware.store/*
Disallow: *Login-*
Disallow: *Register-*
Disallow: *Cart-*
Disallow: *Checkout-*
Disallow: *Account-*
Disallow: *Order-*
```

## Pitfalls

- **SLAS JWT ≠ OCAPI token**: The `cc-at` cookie JWT authenticates the storefront session. OCAPI requires a different token format. The SLAS JWT works with SCAPI endpoints, not OCAPI.
- **Session IP-binding**: `dwsid` and `sid` cookies are typically bound to the client IP. Sessons can't be reused from a different IP.
- **Rate limiting**: SFCC instances often rate-limit at 1-5 req/sec. Check `x-rate-limit-*` headers.
- **SPA routing**: Modern SFCC storefronts are SPAs. Account/basket data loads via AJAX after page render, not in the initial HTML. You need a browser or the actual XHR requests.
- **Client ID per resource**: Different OCAPI resources (products vs customers vs baskets) may use different client IDs.
- **Shared codebase**: Multiple domains sharing the same SFCC instance treat issues as duplicates across all domains.
