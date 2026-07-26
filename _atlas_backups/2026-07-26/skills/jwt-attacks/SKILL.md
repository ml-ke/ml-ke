---
name: jwt-attacks
description: "JWT attack testing — algorithm confusion, kid injection, jku, none algorithm, claim manipulation, secret cracking."
version: 1.0.0
---

# JWT Attack Testing

## Decode JWT
```bash
# Header
echo -n "<header>" | base64 -d 2>/dev/null | python3 -m json.tool
# Payload
echo -n "<payload>" | base64 -d 2>/dev/null | python3 -m json.tool
```

## Attack Vectors

### 1. Algorithm Confusion
If server uses RS256 (asymmetric), test with HS256 (symmetric):
```
alg: RS256 → change to alg: HS256, sign with the public key as HMAC secret
```
The public key can be obtained from the JWKS endpoint (`/v1/jwks`, `/.well-known/jwks.json`).

### 2. None Algorithm
```
alg: RS256 → alg: none
alg: RS256 → alg: None
alg: RS256 → alg: NONE
alg: RS256 → alg: nOnE
```
Remove the signature part (keep only header.payload.)

### 3. Kid Injection
If kid is used for key lookup, try:
- Path traversal: `"kid": "../../../../etc/passwd"`
- SQL injection: `"kid": "foo' UNION SELECT..."`
- OS command injection
- NoSQL injection

### 4. JKU/JWK Injection
- Change jku to point to an attacker-controlled JWKS endpoint
- Inject a jwk header with an attacker's public key
- Change x5u to point to an attacker certificate

### 5. Claim Manipulation
- Change `sub` (subject) to another user's ID
- Add `"admin": true`
- Change `"role": "user"` to `"role": "admin"`
- Remove `exp` (expiration)
- Set `exp` to far-future timestamp
- Remove `nbf` (not before)

### 6. Weak Secret Cracking
If HS256 is used:
```bash
hashcat -a 0 -m 16500 jwt.txt rockyou.txt
```

### 7. Timing Attack
Measure response times for signature validation — valid vs invalid tokens may have different timing.

## Testing Script
```python
import jwt, requests, json

# Test none algorithm
for none in ['none', 'None', 'NONE', 'nOnE']:
    headers = json.dumps({"alg": none, "typ": "JWT"})
    # Encode and test
```
