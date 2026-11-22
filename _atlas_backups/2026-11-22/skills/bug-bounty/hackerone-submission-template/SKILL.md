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

**⚠️ CRITICAL — Anti-AI Detection**: As of 2026, HackerOne actively flags reports that look AI-generated. The sectioned template below is for YOUR notes only. The FINAL report must be written as narrative paragraphs, not as a rigid structure.

**The rule**: Write the PoC first. Run it. Capture output. Then write 2-3 paragraphs. No "## Summary", "## Description", "## Steps To Reproduce", "## Impact" headers in sequence — that structure pattern is the #1 AI-detection trigger. Instead, write it as a natural description of what you found and how to reproduce it.

**Before (flagged as AI):**
```
## Summary
Stored XSS in user profile field

## Description
The user profile field does not sanitize HTML input...

## Steps To Reproduce
1. Create a user account
2. Set profile field to <script>alert(1)</script>
3. View profile page
...

## Impact
An attacker could execute arbitrary JavaScript...
```

**After (accepted):**
```
The "bio" field on user profiles does not sanitize HTML. Setting it to:

<img src=x onerror=alert(document.cookie)>

...renders on profile view without encoding. I tested with a self-XSS payload and confirmed the alert fires on Chrome 125. The payload survives page reload and is stored in the database.

This means any user who views the profile executes the attacker's script. Session cookie theft via document.cookie is the immediate concern.
```

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
