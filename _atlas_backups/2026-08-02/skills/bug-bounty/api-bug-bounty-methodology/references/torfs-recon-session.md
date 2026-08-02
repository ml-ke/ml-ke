# Torfs Recon Session (Jun 2026)

## Program Details
- **Platform**: Intigriti (Public)
- **Payout**: Tier 2 (€100-€6,500), Tier 3 (€25-€1,500)
- **Assets**: www.torfs.be, www.torfs.nl, www.schoenentorfs.be, www.schoenentorfs.nl, winkels.torfs.be (T2), samenfittorfs.be (T3)
- **Response time**: <16 hours first response
- **Rules**: Must use @intigriti.me email, 1 req/sec rate limit, no brute force, no directory enumeration
- **Out of scope**: Ordering articles, subdomain takeover, OAuth misconfig, blind SSRF without impact

## Target Details
- **Platform**: Salesforce Commerce Cloud v25.5
- **Site ID**: Torfs-Webshop-BE
- **Org**: bcqr_prd
- **OCAPI client_id**: Torfs-Webshop-BE (works for baskets, orders, sessions)
- **SLAS JWT**: cc-at_Torfs-Webshop-BE cookie, ES256, 30min expiry
- **Auth**: SLAS OIDC, user registered as h0d4r1254@intigriti.me / Hodari Tester
- **WebDAV**: Cartridges (401), Code (403), Impex (401)
- **CDN**: Cloudflare

## JWT Scopes Available
sfcc.shopper-myaccount.baskets, sfcc.shopper-discovery-search, sfcc.shopper-myaccount.paymentinstruments, sfcc.shopper-customers.login, sfcc.shopper-myaccount.orders, sfcc.shopper-productlists, sfcc.shopper-promotions, sfcc.shopper.stores, sfcc.orders.rw, sfcc.session_bridge, sfcc.shopper-myaccount.paymentinstruments.rw, sfcc.shopper-myaccount.productlists, sfcc.shopper-categories, sfcc.shopper-myaccount, sfcc.shopper-myaccount.addresses, sfcc.shopper-products, sfcc.shopper-myaccount.rw, sfcc.shopper-baskets-orders, sfcc.shopper-customers.register, sfcc.shopper-myaccount.addresses.rw, sfcc.shopper-myaccount.productlists.rw, sfcc.shopper-product, sfcc.shopper-baskets-orders.rw, sfcc.shopper-gift-certificates, sfcc.shopper-product-search

## Key Findings
1. OCAPI v25_5 accessible with client_id=Torfs-Webshop-BE
2. WebDAV exposed (Cartridges 401, Code 403)
3. SLAS JWT format confirmed with ES256 algorithm
4. User JWT obtained (sty: User, not Guest)
5. SCAPI gateway not on storefront domain (behind Salesforce CDN)
6. Storefront pipelines return 500 when called without proper context
7. Account registration works with @intigriti.me email
8. Address IDs: `homeaddress` and `Neighbor` (user-named, not sequential)
9. CSRF token discovered: `x0fIZzYIDlkfHkPiQM2Sp3nwV...`
10. Product in basket: PID `37564105` (Sneakers wit), PID `00500400` (€5 coupon)
11. Price extracted: €16.65 for sneakers
12. JWT isb claim decoded: upn=h0d4r1254@intigriti.me, uidn=Hodari Tester
13. Session auth confirmed via data-authenticated=true, data-registered=true
14. CQRecomm-Start pipeline discovered (recommendation engine)
15. Forms: POST /login (form action), /register redirects to /
16. OCAPI returns MethodNotAllowedException for GET on baskets, orders, sessions
17. OCAPI returns AuthorizationHeaderMissingException for POST without valid auth
