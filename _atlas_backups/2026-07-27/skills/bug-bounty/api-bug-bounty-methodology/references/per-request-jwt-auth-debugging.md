# Per-Request Signed JWT Auth — Debugging Guide

Some APIs (Fireblocks, potentially others) use per-request signed JWTs for authentication. Every HTTP request carries a freshly-signed JWT. This pattern is common in financial/crypto APIs that need request integrity.

## Failure Modes and How To Fix Them

### 1. "Token signed for incorrect url" (code -4)

**Cause:** The `uri` field in the JWT payload doesn't match the actual request URL path.

**Fix:** Extract URI using the same method as the SDK:
```javascript
const url = new URL(fullRequestUrl);
const uri = url.pathname + url.search;
// Sign this uri, NOT the full URL with host
```

### 2. "Error getting User certificate" / "Error creating public key" (code -7) — ALL requests

**Cause:** The API key's public key couldn't be found by the server. Usually means the API key was:
- Created but not authorized by a console user
- Deleted/revoked
- From a different environment (production key used on sandbox)

**Fix:** Check the Fireblocks Console → API Keys → verify the key exists and is authorized.

### 3. "Error getting User certificate" (code -7) — POST only, GET works

**Two possible causes:**

**A) Missing Content-Length header:** Node.js `https.request()` does not auto-set Content-Length when you write the body with `req.write(str)`. Without it, the server can't parse the body and the auth check fails.
```javascript
// FIX: Always set Content-Length explicitly
'Content-Length': Buffer.byteLength(JSON.stringify(body))
```

**B) Wrong bodyHash:** The bodyHash must be SHA256 of JSON.stringify(body), NOT of the raw JS object.
```javascript
// CORRECT:
const bodyHash = crypto.createHash('sha256')
  .update(JSON.stringify(body))
  .digest().toString('hex');

// WRONG (old Node.js silently hashes "[object Object]"):
const bodyHash = crypto.createHash('sha256')
  .update(body)  // Don't do this!
  .digest().toString('hex');
```

### 4. "Error getting User certificate" (code -7) — ALL requests after a burst

**Cause:** Rate limiting. The sandbox throttles after ~10 rapid requests.

**Fix:** Wait 10-30 seconds, then retry. The key recovers automatically.

### 5. Python JWT rejects but Node.js accepts

**Observation:** The same private key + payload produces different RS256 signatures between Python's `jwt.encode()` (PyJWT library) and Node's `jsonwebtoken.sign()`. The Fireblocks server accepts Node.js signatures but rejects Python's.

**Fix:** Use Node.js for JWT signing. Python's `PyJWT.encode()` produces byte-level differences in RS256 output even with the same inputs.

## Quick Diagnostic Checklist

When hitting JWT auth errors:

1. [ ] Can you decode the JWT and verify it locally?
2. [ ] Does the `uri` match the request path exactly?
3. [ ] Is `Content-Length` set for POST requests?
4. [ ] Is the `bodyHash` SHA256 of `JSON.stringify(body)`, not the raw object?
5. [ ] Is the JWT within the 55-second expiry window?
6. [ ] Try a GET request — does it work? (isolates auth from body issues)
7. [ ] Wait 30s and try again — was it rate limiting?
