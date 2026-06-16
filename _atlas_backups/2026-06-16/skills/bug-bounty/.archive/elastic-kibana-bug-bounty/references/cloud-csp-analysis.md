# CSP Analysis — Elastic Cloud vs Kibana Source

## Confirmed CSP on cloud.elastic.co (Kibana 9.4.2)

### Enforced CSP (Content-Security-Policy header)
```
script-src 'report-sample' 'self';
worker-src 'report-sample' 'self' blob:;
style-src 'report-sample' 'self' 'unsafe-inline';
object-src 'report-sample' 'none';
report-to violations-endpoint
```

### Report-Only CSP
```
form-action 'report-sample' 'self';
default-src 'report-sample' 'none';
font-src 'report-sample' 'self';
img-src 'report-sample' 'self' data: tiles.maps.elastic.co;
connect-src 'report-sample' 'self' telemetry.elastic.co telemetry-staging.elastic.co feeds.elastic.co tiles.maps.elastic.co vector.maps.elastic.co;
script-src 'report-sample' 'self';
worker-src 'report-sample' 'self' blob:;
style-src 'report-sample' 'self' 'unsafe-inline';
object-src 'report-sample' 'none';
report-to violations-endpoint
```

## Key Differences from Source Code Default

### Source Default (`csp_directives.ts`)
```typescript
// Script-src gets 'unsafe-eval' added at line 229 unless disabled
script-src: ['report-sample', 'self']  
// PLUS if disableUnsafeEval !== true:
script-src: ['unsafe-eval', 'report-sample', 'self']
```

### Cloud Actual
```
script-src: 'report-sample' 'self'
```
`'unsafe-eval'` is **NOT present** on cloud.elastic.co. This means:
- `disableUnsafeEval: true` is set in the cloud deployment config
- Any XSS cannot use `eval()` or `Function()` constructor
- `vega-interpreter` AST mode provides the ONLY expression evaluation path

### CSP Weaknesses on Cloud
1. **No nonces** — no `nonce-*` in `script-src`. `script-src 'self'` means any same-origin JS executes.
2. **No `strict-dynamic`** — not present in either enforced or report-only
3. **`'unsafe-inline'` on `style-src`** — CSS injection possible, could be used for data exfiltration
4. **`connect-src` includes `'self'`** — any same-origin XHR/fetch works
5. **`img-src` includes `data:`** — data URI images allowed
6. **Frame ancestors set by `disableEmbedding`** — likely `frame-ancestors 'self'` (prevents clickjacking)

### What CSP Blocks
- Inline scripts (`<script>alert(1)</script>`) — needs nonce or hash
- `eval()` and `Function()` constructor — blocked (unless `unsafe-eval` is present)
- External script loads — only `'self'` allowed

## CSP Bypass Opportunities

1. **Same-origin script injection** — If an attacker can upload/control any JS file on the same origin, `'self'` won't block it. Look for file upload endpoints that accept `.js` files or JSONP endpoints.

2. **Vega-interpreter sandbox escape** — The Vega expression interpreter runs via AST interpretation (not eval). A sandbox escape yields full XSS that is invisible to CSP because the interpreter is a trusted script.

3. **CSS injection** — `style-src 'unsafe-inline'` allows inline styles. Can be used for data exfiltration via CSS selectors + background-image URLs.

4. **Connect-src `'self'`** — Any fetch/XHR to same-origin is allowed. Useful for SSRF chaining.

## Report-Only Observations
The report-only CSP has `default-src 'none'` which is VERY strict — but it's only report-only. The enforced CSP is much more permissive. This suggests Elastic is testing stricter policies but hasn't enabled them yet.

## Console Errors Observed (cloud.elastic.co)
```
"Evaluating a string as JavaScript violates the following CSP directive because 'unsafe-eval' 
is not an allowed source of script: 'script-src 'report-sample' 'self''. The policy is 
report-only, so the violation has been logged but no further action has been taken."
```
This is a **report-only** violation — Kibana's own code uses eval internally but the enforced CSP blocks it. Some feature may break silently.

```
"Executing inline script violates the following CSP directive 'script-src 'report-sample' 'self''. 
Either the 'unsafe-inline' keyword, a hash, or a nonce is required."
```
Also report-only — inline scripts in the bootstrap process produce warnings but don't actually break.
