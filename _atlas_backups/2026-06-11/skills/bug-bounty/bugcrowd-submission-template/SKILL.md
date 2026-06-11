---
name: bugcrowd-submission-template
category: bug-bounty
description: Bugcrowd vulnerability report submission template. Use when filing reports on Bugcrowd programs.
---

# Bugcrowd Submission Template

## Bugcrowd Fields (in order on the submit form)

1. **Summary title** — Brief one-line description of the vulnerability
2. **Target** — Select from the program's target dropdown (must be explicitly in scope)
3. **Technical severity** — VRT baseline priority (P1-P5) from the Bugcrowd Vulnerability Rating Taxonomy. Note: this is a suggested baseline; program rules override it.
4. **VRT Category** — Select from the VRT dropdown (e.g., "Broken Access Control (BAC) > Insecure Direct Object References (IDOR) > Modify/View Sensitive Information(Iterable Object Identifiers)")
5. **URL / Location of vulnerability** — Optional, e.g., https://target.com/path
6. **Description** — Vulnerability details, impact, and proof of concept / replication steps (max 25,000 characters)
7. **Attachments** — PoC scripts, screenshots, recordings (max 20 files, <400MB each)

---

## Report Structure Template

```
### Summary

[One sentence: what the vulnerability is and what it allows]

### Vulnerability Details

**Target:** [selected from dropdown]
**VRT Category:** [category path]
**Technical Severity:** P[1-5] — [VRT variant name]
**URL:** [optional endpoint URL]

### Description

[Detailed description covering:
- What the vulnerability is
- Where it exists
- Why it's a security issue
- The impact to the program]

### Proof of Concept / Replication Steps

1. [Step one — concrete, copy-pasteable]
2. [Step two]
3. [Step three]

#### Request

```
[Example request — curl, HTTP request, GraphQL query]
```

#### Response

```
[Example response showing the vulnerability]
```

### Impact

[Concrete answer to "As an attacker I could..." — what can an attacker ACTUALLY do with this?
NOT theoretical — must demonstrate crossing a real security boundary]
```

---

## Bugcrowd-Specific Tips

- **Targets not in scope may be rejected** — always verify the target is in the program's scope list
- **VRT category ≠ severity guarantee** — programs can override severity. The VRT is a baseline classification, not a payout guarantee.
- **Description max 25K chars** — keep it concise; use attachments for supplementary material
- **Attachments** can be embedded as Markdown images (.jpg/.gif/.png < 50MB using "Copy as Markdown")
- **Chaining is endorsed** by Bugcrowd Usage Guide — chain low-severity bugs into higher impact
- **One folder per finding**: ~/Dev/REPORTS/<Program>/<Number>/REPORT.md + poc/ subfolder
- **Before submitting**, run the `pre-submission-verification` skill

## Common VRT Categories by Finding Type

| Finding Type | VRT Category Path |
|-------------|------------------|
| IDOR (sequential IDs, sensitive) | BAC > IDOR > Modify/View Sensitive Information(Iterable Object Identifiers) |
| IDOR (sequential IDs, non-sensitive) | BAC > IDOR > View Non-Sensitive Information |
| IDOR (UUID/GUID) | BAC > IDOR > Modify/View Sensitive Information(Complex Object Identifiers GUID/UUID) |
| SSRF (internal, high impact) | Server Security Misconfiguration > SSRF > Internal High Impact |
| SSRF (internal scan/medium) | Server Security Misconfiguration > SSRF > Internal Scan and/or Medium Impact |
| SSRF (external DNS only) | Server Security Misconfiguration > SSRF > External - DNS Query Only |
| Stored XSS | XSS > Stored (pick privilege variant) |
| Reflected XSS | XSS > Reflected (Non-Self or Self) |
| Auth Bypass | Broken Authentication > Authentication Bypass |
| Broken Crypto Primitive | Cryptographic Weakness > Broken Cryptography > Use of Broken Cryptographic Primitive |
| Padding Oracle | Cryptographic Weakness > Side-Channel Attack > Padding Oracle Attack |
| Timing Attack | Cryptographic Weakness > Side-Channel Attack > Timing Attack |
| Key Reuse (Inter-Env) | Cryptographic Weakness > Key Reuse > Inter-Environment |
| Insufficient Entropy (RNG) | Cryptographic Weakness > Insufficient Entropy > Limited RNG Entropy Source |
| Predictable PRNG Seed | Cryptographic Weakness > Insufficient Entropy > Predictable PRNG Seed |
| Open Redirect (GET) | Unvalidated Redirects and Forwards > Open Redirect > GET-Based |
| Admin Portal Exposure | Server Security Misconfiguration > Exposed Portal > Admin Portal |
| Hardcoded Password (Privileged) | Insecure OS/Firmware > Hardcoded Password > Privileged User |
| Disclosure of Secrets (Public) | Sensitive Data Exposure > Disclosure of Secrets > For Publicly Accessible Asset |
| Info Disclosure (Internal) | Sensitive Data Exposure > Disclosure of Secrets > For Internal Asset |
| Privilege Escalation via IDOR | BAC > IDOR > Modify/View Sensitive Information (pick variant by ID type) |
| Template Injection (SSTI) | Server-Side Injection > SSTI > Basic |
| Content Spoofing (iframe) | Server-Side Injection > Content Spoofing > iframe Injection |
| CRLF Injection | Server-Side Injection > HTTP Response Manipulation > Response Splitting (CRLF) |
| Local File Inclusion | Server-Side Injection > File Inclusion > Local |
| Race Condition | Application-Level DoS or pick program-specific |
| OAuth Account Takeover | Server Security Misconfiguration > OAuth Misconfiguration > Account Takeover |
