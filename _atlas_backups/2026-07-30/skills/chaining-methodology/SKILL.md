---
name: chaining-methodology
description: "Vulnerability chaining methodology — combine low-severity issues into critical exploit chains. How to chain IDOR, XSS, SSRF, info leaks, and auth flaws."
version: 1.0.0
---

# Vulnerability Chaining Methodology

## Core Principle
A single low-severity bug is often rejected. Chained with another bug, it becomes critical.

## Common Chain Patterns

### Pattern 1: Info Leak → IDOR → Data Access
```
Public endpoint leaks user IDs/GUIDs
  → IDOR on data endpoint using leaked IDs
  → Access another user's private data
```

### Pattern 2: IDOR (read) → IDOR (write) → Full Compromise
```
IDOR to read non-sensitive data (e.g., order numbers)
  → IDOR on write endpoint using same IDs
  → Modify/delete another user's data
```

### Pattern 3: XSS → CSRF → Account Takeover
```
XSS on profile page
  → Steals CSRF token
  → Changes victim's email via CSRF
  → Password reset to attacker email → ATO
```

### Pattern 4: SSRF → Cloud Metadata → Credentials
```
SSRF via webhook URL
  → Access cloud metadata service (169.254.169.254)
  → Steal cloud provider credentials
  → Access cloud resources
```

### Pattern 5: Auth Bypass → Privilege Escalation → RCE
```
Auth bypass on admin endpoint
  → Access admin-only functionality
  → Feature has command injection/RCE
  → Full server compromise
```

### Pattern 6: GUID Leak → IDOR Amplification
```
List endpoint leaks UUIDs/GUIDs
  → Use leaked IDs in IDOR attack
  → Access ANY user's data (not just adjacent IDs)
```

## How to Find Chains
1. After finding ANY vulnerability, ask: "What else can I reach from here?"
2. Map all endpoints accessible after the first finding
3. Test if the first finding unlocks additional attack surface
4. Look for: session tokens, API keys, internal IDs, CSRF tokens
5. Chain = first finding reveals data → second finding uses that data

## Report Structure for Chains
```
Title: [Chain] [Entry Bug] → [Escalation] → [Impact]

Description:
Bug 1: Low-severity XSS on profile page
Bug 2: CSRF on email change (no token validation)
Chain: XSS steals CSRF token → CSRF changes email → password reset → ATO

Impact: Full account takeover without user interaction
```
