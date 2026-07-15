---
name: hackerone-submission-template
category: bug-bounty
description: HackerOne vulnerability report submission template. Use this when filing reports on HackerOne programs.
---

# HackerOne Submission Template

## HackerOne Fields (in order on the submit form)

1. **Asset** — Select the vulnerable target from the dropdown
2. **Report template** — Optional, select if program provides one
3. **Weakness type** — CWE-based dropdown; pick the closest match
4. **Severity** — Optional CVSS calculator; use only if you're confident in the score
5. **Summary** — Brief one-line description
6. **Description** — Full writeup including vulnerability, impact, and proof of concept
7. **Steps to reproduce** — Numbered list of exact steps
8. **Impact** — Business/security impact
9. **Supporting material** — PoC code, screenshots, curl commands
10. **Attachments** — Up to program limit
11. **Custom fields** — Program-specific fields (if any)
12. **Review & submit**

---

## Report Structure Template

Copy this structure for every HackerOne submission:

```
## Summary

[One sentence: what the vulnerability is and what it allows]

## Description

[Detailed description of the vulnerability, including:
- Where it exists (file path, endpoint, component)
- Why it's a vulnerability
- The security boundary being crossed (if access control)]

## Steps To Reproduce

1. [Step one]
2. [Step two]
3. [Step three]
...

## Supporting Material / References

```
[PoC code, curl commands, or other evidence]
```

[Screenshots if applicable]

## Impact

[What an attacker can actually do with this — be concrete, not theoretical]
```

---

## HackerOne-Specific Tips

- **Weakness type** uses CWE taxonomy — search by CWE number or name
- **Severity** is optional but including a CVSS score shows professionalism. Use CVSS 3.1 calculator.
- **Custom fields** vary per program — fill them in or the report may be rejected
- **Attachments** can be added after submission too
- **Program disclosure policy** varies — some auto-close after 30 days of inactivity
- **Bounty negotiation** happens after triage, not in the initial report

---

## Common Weakness Types (CWE) by Vulnerability Class

| Vulnerability | CWE |
|--------------|-----|
| SQL Injection | CWE-89 |
| Cross-Site Scripting (XSS) | CWE-79 |
| Insecure Direct Object Reference (IDOR) | CWE-639 |
| Server-Side Request Forgery (SSRF) | CWE-918 |
| Authentication Bypass | CWE-287 |
| Privilege Escalation | CWE-269 |
| Path Traversal | CWE-22 |
| Information Disclosure | CWE-200 |
| XML External Entity (XXE) | CWE-611 |
| Template Injection (SSTI) | CWE-1336 |
| Open Redirect | CWE-601 |
| Race Condition | CWE-362 |
| Broken Cryptography | CWE-327 |
| Insufficient Entropy | CWE-331 |
| Padding Oracle | CWE-209 |
| Use of Hard-coded Credentials | CWE-798 |
| Command Injection | CWE-77 |
| Cross-Site Request Forgery (CSRF) | CWE-352 |
| Insecure Deserialization | CWE-502 |
| Server-Side Template Injection | CWE-1336 |
