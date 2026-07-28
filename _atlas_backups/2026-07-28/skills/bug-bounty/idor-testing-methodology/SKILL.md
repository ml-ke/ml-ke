---
name: idor-testing-methodology
description: "IDOR/BOLA testing methodology — UUID/GUID amplification, method tampering, mass assignment, chaining. Updated from rejected submissions and 2025-2026 research."
version: 1.0.0
---

# IDOR/BOLA Testing Methodology

## Core Principle
UUIDs don't prevent IDOR — they make enumeration harder but DON'T enforce authorization. Find an endpoint that LEAKS the UUIDs (GUID amplification).

## What I Was Missing

### 1. HTTP Method Tampering
Test ALL methods on every endpoint, not just GET:
```bash
curl -X GET /api/resource/123   # Read
curl -X POST /api/resource/123  # Create/update
curl -X PUT /api/resource/123   # Update  
curl -X PATCH /api/resource/123 # Partial update
curl -X DELETE /api/resource/123 # Delete
```
Different methods often have different auth levels. A GET might be protected but a DELETE might not be.

### 2. GUID/UUID Amplification
If IDs are UUIDs (not sequential):
- Find an endpoint that LISTS or LEAKS UUIDs
- The list endpoint + the IDOR = complete exploit chain
- Check: list endpoints, search endpoints, export endpoints, analytics, logs
- The leak improves severity from Low to High

### 3. Sequential ID Testing
Even with UUIDs, check for:
- Sequential numbers in some contexts (invoice numbers, order numbers)
- Timestamp-based IDs
- Incrementing integers alongside UUIDs
- Predictable patterns in UUID generation (v1 UUIDs include timestamp)

### 4. Parameter Pollution
Send the same parameter twice:
```
GET /api/resource?id=123&id=456
POST /api/resource with body: {"id": 123, "id": 456}
```
Server may process the second value while auth checks the first.

### 5. Mass Assignment
Add extra parameters to requests:
```json
{"name": "test", "admin": true, "role": "admin", "isAdmin": 1}
```
Check if the API accepts unauthorized fields.

### 6. Chaining IDOR
- IDOR that reveals internal IDs + IDOR that uses those IDs = High severity
- IDOR on non-sensitive data + IDOR on sensitive operation = High severity
- GUID leak endpoint + IDOR on write endpoint = Complete exploit

### 7. Two-Account Requirement
- Can't prove IDOR with one account
- Must verify User A's data is accessible from User B's session
- Register two accounts if only one set of creds available

### 8. GraphQL Batch Queries
```graphql
query {
  a: resource(id: 1) { sensitive }
  b: resource(id: 2) { sensitive }
  c: resource(id: 3) { sensitive }
}
```
Bypasses rate limits for IDOR enumeration.

## Testing Checklist
- [ ] Test GET/POST/PUT/PATCH/DELETE on every endpoint
- [ ] Test adjacent IDs (id=1, id=2, id=3)
- [ ] Test UUIDs if found (check list endpoints for leaks)
- [ ] Test parameter pollution
- [ ] Test mass assignment (extra fields)
- [ ] Register two accounts for cross-account testing
- [ ] Check GraphQL for batch IDOR
- [ ] Look for GUID amplification (endpoints that leak IDs)
- [ ] Test with cookies AND without cookies
- [ ] Test with different user roles if available
