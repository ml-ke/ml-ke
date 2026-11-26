---
name: bugcrowd-vrt
category: bug-bounty
description: Bugcrowd Vulnerability Rating Taxonomy (VRT) v1.18 — complete reference table extracted from live JSON taxonomy. Always use this to find the CORRECT category path for your finding before submitting.
---

# Bugcrowd VRT v1.18 (March 2026) — Complete Taxonomy

**Source**: https://bugcrowd.com/vulnerability-rating-taxonomy
**JSON**: https://github.com/bugcrowd/vulnerability-rating-taxonomy

## Core Rule

**VRT category = CLASSIFICATION only.** The baseline priority (P1-P5) is a suggested starting point. The target program's own rules override it for actual payout. Pick the category that describes the ROOT CAUSE, not the impact.

## Full Taxonomy Table

All 287 entries organized by severity. To find your category: search by vulnerability type, then trace the path upward.

### P1 — Critical (24 entries)

| Category Path | Vulnerability |
|--------------|--------------|
| AI Application Security > Model Extraction | API Query-Based Model Reconstruction |
| AI Application Security > Remote Code Execution | Full System Compromise |
| AI Application Security > Sensitive Information Disclosure | Cross-Tenant PII Leakage/Exposure |
| AI Application Security > Sensitive Information Disclosure | Key Leak |
| AI Application Security > Training Data Poisoning | Backdoor Injection / Bias Manipulation |
| Automotive Security Misconfiguration > Infotainment, Radio Head Unit | Sensitive data Leakage/Exposure |
| Automotive Security Misconfiguration > RF Hub | Key Fob Cloning |
| Broken Access Control (BAC) > IDOR | Modify/View Sensitive Information(Iterable Object Identifiers) |
| Broken Authentication and Session Management | Authentication Bypass |
| Cloud Security > IAM Misconfigurations | Publicly Accessible IAM Credentials |
| Decentralized Application Misconfiguration > Insecure Data Storage | Plaintext Private Key |
| Decentralized Application Misconfiguration > Marketplace Security | Orderbook Manipulation |
| Decentralized Application Misconfiguration > Marketplace Security | Signer Account Takeover |
| Decentralized Application Misconfiguration > Marketplace Security | Unauthorized Asset Transfer |
| Decentralized Application Misconfiguration > Protocol Security Misconfiguration | Node-level Denial of Service |
| Insecure OS/Firmware > Command Injection | (no variant) |
| Insecure OS/Firmware > Hardcoded Password | Privileged User |
| Sensitive Data Exposure > Disclosure of Secrets | For Publicly Accessible Asset |
| Server Security Misconfiguration > Exposed Portal | Admin Portal |
| Server-Side Injection > File Inclusion | Local |

### P2 — High (24 entries)

| Severity | Category Path | Vulnerability |
|----------|--------------|--------------|
| P2 | AI Application Security > Denial-of-Service (DoS) | Application-Wide |
| P2 | AI Application Security > Prompt Injection | System Prompt Leakage |
| P2 | AI Application Security > Remote Code Execution | Sandboxed Container Code Execution |
| P2 | AI Application Security > Vector and Embedding Weaknesses | Embedding Exfiltration / Model Extraction |
| P2 | Automotive Security Misconfiguration > Infotainment, Radio Head Unit | Code Execution (CAN Bus Pivot) |
| P2 | Automotive Security Misconfiguration > Infotainment, Radio Head Unit | OTA Firmware Manipulation |
| P2 | Automotive Security Misconfiguration > RF Hub | CAN Injection / Interaction |
| P2 | Broken Access Control (BAC) > IDOR | Modify Sensitive Information(Iterable Object Identifiers) |
| P2 | Cloud Security > IAM Misconfigurations | Overly Permissive IAM Roles |
| P2 | Cloud Security > Storage Misconfigurations | Unencrypted Sensitive Data at Rest |
| P2 | Cryptographic Weakness > Key Reuse | Inter-Environment |
| P2 | Decentralized Application Misconfiguration > Marketplace Security | Malicious Order Offer |
| P2 | Decentralized Application Misconfiguration > Marketplace Security | Price or Fee Manipulation |
| P2 | Insecure OS/Firmware > Hardcoded Password | Non-Privileged User |
| P2 | Physical Security Issues > Weakness in physical access control | Commonly Keyed System |
| P2 | Sensitive Data Exposure > Weak Password Reset Implementation | Token Leakage via Host Header Poisoning |
| P2 | Server Security Misconfiguration > OAuth Misconfiguration | Account Takeover |
| P2 | Server Security Misconfiguration > SSRF | Internal High Impact |
| P2 | XSS > Stored | Non-Privileged User to Anyone |

### P3 — Medium (32 entries)

Includes: AI Application Security > Improper Output Handling > XSS (P3), BAC > IDOR > View Sensitive Information(Iterable Object Identifiers) (P3), Cloud Security > Network Configuration Issues (P3), Cryptographic Weakness > Broken Cryptography > Use of Broken Cryptographic Primitive (P3), Cryptographic Weakness > Insecure Key Generation > Insufficient Key Space (P3), Insecure OS/Firmware > Weakness in Firmware Updates > Firmware does not validate update integrity (P3), Server Security Misconfiguration > Misconfigured DNS > Subdomain Takeover (P3), Server Security Misconfiguration > SSRF > Internal Scan and/or Medium Impact (P3), Server-Side Injection > Content Spoofing > iframe Injection (P3), Server-Side Injection > HTTP Response Manipulation > Response Splitting (CRLF) (P3), Server-Side Injection > SSTI > Basic (P3), XSS > Reflected > Non-Self (P3), XSS > Stored > Privileged User to Privilege Elevation (P3), XSS > Stored > CSRF/URL-Based (P3), Sensitive Data Exposure > Disclosure of Secrets > For Internal Asset (P3), and more.

### P4 — Low (57 entries)

Includes: AI Application Security > AI Safety > Misinformation/Wrong Factual Data (P4), AI Application Security > Improper Output Handling > Markdown/HTML Injection (P4), BAC > IDOR > Modify/View Sensitive Information(Complex Object Identifiers GUID/UUID) (P4), Cryptographic Weakness > Broken Cryptography > Use of Vulnerable Cryptographic Library (P4), Cryptographic Weakness > Insufficient Entropy > Limited RNG Entropy Source (P4), Cryptographic Weakness > Insufficient Entropy > Predictable PRNG Seed (P4), Cryptographic Weakness > Key Reuse > Lack of Perfect Forward Secrecy (P4), Cryptographic Weakness > Side-Channel Attack > Padding Oracle Attack (P4), Cryptographic Weakness > Side-Channel Attack > Timing Attack (P4), Server Security Misconfiguration > SSRF > External - Low impact (P4 - note: P5 for DNS only), Unvalidated Redirects and Forwards > Open Redirect > GET-Based (P4), and more.

### P5 — Informational (74 entries)

Includes: AI Application Security > Improper Input Handling > ANSI Escape Codes (P5), Cryptographic Weakness > Insufficient Entropy > IV Reuse (P5), Cryptographic Weakness > Side-Channel Attack > Power Analysis Attack (P5), Cryptographic Weakness > Weak Hash > Use of Predictable Salt (P5), Server Security Misconfiguration > SSRF > External - DNS Query Only (P5), Unvalidated Redirects and Forwards > Open Redirect > POST-Based (P5), and more.

## Finding Your Category (Quick Reference by Vulnerability Class)

### Access Control
| Variant | VRT Path | Severity |
|---------|----------|----------|
| IDOR with sequential IDs, modify sensitive data | BAC > IDOR > Modify Sensitive Information(Iterable Object Identifiers) | P2 |
| IDOR with sequential IDs, view sensitive data | BAC > IDOR > View Sensitive Information(Iterable Object Identifiers) | P3 |
| IDOR with GUID/UUID, modify/view sensitive | BAC > IDOR > Modify/View Sensitive Information(Complex Object Identifiers GUID/UUID) | P4 |
| IDOR, view non-sensitive data | BAC > IDOR > View Non-Sensitive Information | P5 |
| Auth bypass | Broken Authentication > Authentication Bypass | P1 |
| Username/email enum (non-brute) | BAC > Username/Email Enumeration > Non-Brute Force | P4 |

### Cryptographic
| Variant | VRT Path | Severity |
|---------|----------|----------|
| Broken crypto primitive (MD5, RC4, SHA-1 for sigs) | Cryptographic Weakness > Broken Cryptography > Use of Broken Cryptographic Primitive | P3 |
| Vulnerable crypto library | Cryptographic Weakness > Broken Cryptography > Use of Vulnerable Cryptographic Library | P4 |
| Padding oracle | Cryptographic Weakness > Side-Channel Attack > Padding Oracle Attack | P4 |
| Timing attack | Cryptographic Weakness > Side-Channel Attack > Timing Attack | P4 |
| Key reuse across environments | Cryptographic Weakness > Key Reuse > Inter-Environment | P2 |
| Key reuse within environment | Cryptographic Weakness > Key Reuse > Intra-Environment | P5 |
| No PFS | Cryptographic Weakness > Key Reuse > Lack of Perfect Forward Secrecy | P4 |
| Limited RNG entropy | Cryptographic Weakness > Insufficient Entropy > Limited RNG Entropy Source | P4 |
| Predictable PRNG seed | Cryptographic Weakness > Insufficient Entropy > Predictable PRNG Seed | P4 |
| PRNG seed reuse | Cryptographic Weakness > Insufficient Entropy > PRNG Seed Reuse | P5 |
| IV reuse | Cryptographic Weakness > Insufficient Entropy > Initialization Vector (IV) Reuse | P5 |
| Insufficient key space | Cryptographic Weakness > Insecure Key Generation > Insufficient Key Space | P3 |

### SSRF
| Variant | VRT Path | Severity |
|---------|----------|----------|
| Internal, high impact (metadata creds, internal service RCE) | Server Security Misconfiguration > SSRF > Internal High Impact | P2 |
| Internal scan / medium impact | Server Security Misconfiguration > SSRF > Internal Scan and/or Medium Impact | P3 |
| External, low impact | Server Security Misconfiguration > SSRF > External - Low impact | P4 |
| External, DNS query only | Server Security Misconfiguration > SSRF > External - DNS Query Only | P5 |

### XSS
| Variant | VRT Path | Severity |
|---------|----------|----------|
| Stored, non-privileged to anyone | XSS > Stored > Non-Privileged User to Anyone | P2 |
| Stored, privileged to priv escalation | XSS > Stored > Privileged User to Privilege Elevation | P3 |
| Stored, privileged to no priv elevation | XSS > Stored > Privileged User to No Privilege Elevation | P4 |
| Stored, self | XSS > Stored > Self | P5 |
| Reflected, non-self | XSS > Reflected > Non-Self | P3 |
| Reflected, self | XSS > Reflected > Self | P5 |
| Stored, CSRF/URL-based | XSS > Stored > CSRF/URL-Based | P3 |
| Off-domain (data URI) | XSS > Off-Domain > Data URI | P4 |

### Information Disclosure
| Variant | VRT Path | Severity |
|---------|----------|----------|
| Secrets for publicly accessible asset | Sensitive Data Exposure > Disclosure of Secrets > For Publicly Accessible Asset | P1 |
| Secrets for internal asset | Sensitive Data Exposure > Disclosure of Secrets > For Internal Asset | P3 |
| Non-corporate user data | Sensitive Data Exposure > Disclosure of Secrets > Non-Corporate User | P5 |
| PII leakage/exposure | Sensitive Data Exposure > Disclosure of Secrets > PII Leakage/Exposure | PNone |
| Full path disclosure | Sensitive Data Exposure > Visible Detailed Error/Debug Page > Full Path Disclosure | P5 |
| Debug page with stack trace | Sensitive Data Exposure > Visible Detailed Error/Debug Page > Descriptive Stack Trace | P5 |

### Server-Side Injection
| Variant | VRT Path | Severity |
|---------|----------|----------|
| SSTI (basic) | Server-Side Injection > Server-Side Template Injection (SSTI) > Basic | P4 |
| Local file inclusion | Server-Side Injection > File Inclusion > Local | P1 |
| CRLF / response splitting | Server-Side Injection > HTTP Response Manipulation > Response Splitting (CRLF) | P3 |
| iframe injection | Server-Side Injection > Content Spoofing > iframe Injection | P3 |
| HTML content injection | Server-Side Injection > Content Spoofing > HTML Content Injection | P5 |

### Open Redirect
| Variant | VRT Path | Severity |
|---------|----------|----------|
| GET-based | Unvalidated Redirects and Forwards > Open Redirect > GET-Based | P4 |
| POST-based | Unvalidated Redirects and Forwards > Open Redirect > POST-Based | P5 |
| Header-based | Unvalidated Redirects and Forwards > Open Redirect > Header-Based | P5 |
| Flash-based | Unvalidated Redirects and Forwards > Open Redirect > Flash-Based | P5 |

## Common Mistakes

| What You Found | Don't Submit As | Submit As |
|---------------|----------------|-----------|
| Sequential IDs in own scope | IDOR P1 | Not a vulnerability (no boundary crossed) |
| Hidden UI field accessible via API | IDOR | Not a vulnerability (client-side only) |
| JWT has extra claims that server ignores | Auth bypass | Not a vulnerability (server validates server-side) |
| Nonce reuse possible in theory | Crypto weakness | Need demonstrated key recovery |
| SDK hashes wrong data | Auth bypass | Use of Broken Cryptographic Primitive (P3) |
| DNS resolution to internal IP | SSRF P2 | SSRF External - DNS Query Only (P5) unless you get data back |
| Open redirect | SSRF | Unvalidated Redirects and Forwards > Open Redirect |
