# Bugcrowd Vulnerability Rating Taxonomy (VRT) v1.18 — Quick Reference

Last updated: 9 Mar 2026
Source: https://bugcrowd.com/vulnerability-rating-taxonomy

## Severity Levels (P1-P5) — Impact Demonstration Required

| Level | Meaning | Typical Bounty | Key Requirement |
|-------|---------|---------------|-----------------|
| P1 | Critical | Highest | Demonstrated severe impact |
| P2 | High | Significant | Clear exploit path |
| P3 | Medium | Moderate | Information disclosure with harm |
| P4 | Low | Minimal | Low-risk info disclosure |
| P5 | Informational | None/No reward | **No demonstrated risk** |

**CRITICAL LESSON from Bugcrowd triage feedback (June 2026):**
Per the VRT, an endpoint returning data without auth is P5 unless you can answer "as an attacker I could..." with a meaningful action. The triage specifically stated: *"this issue is considered to be a P5 (Informational) finding as per Bugcrowd's VRT. This is typically the case when an issue lacks a demonstrated risk and is considered security best practice."*

Before rating any finding P3 or above, apply the Impact Test:
1. What SPECIFIC action can the attacker perform with this data?
2. Does the leaked data enable a SECOND unauthorized action (not just reveal info)?
3. Can you DEMONSTRATE the full attack chain?
4. Is the exposed data SENSITIVE or trivial?

Unauthenticated access to form names, field IDs, locale lists, or published articles = P5 unless the data enables a SECOND attack step that gives the attacker access or control they shouldn't have.

## Key VRT Categories for Our Findings

### Broken Access Control (BAC) → IDOR
- **P1**: Modify/View Sensitive Information (Iterable Object Identifiers)
- **P2**: Modify Sensitive Information (Iterable Object Identifiers)
- **P3**: View Sensitive Information (Iterable Object Identifiers)
- **P4**: Modify/View Sensitive Information (Complex Object Identifiers — GUID/UUID)
- **P5**: View Non-Sensitive Information

### Broken Authentication and Session Management
- **P1**: Authentication Bypass
- **P3**: Session Fixation, 2FA Bypass
- **P4**: Cleartext Session Token, Failure to Invalidate Session

### Server Security Misconfiguration
- **P2**: SSRF Internal High Impact
- **P3**: SSRF Internal Scan / Medium Impact
- **Varies**: Race Condition
- **P5**: SSRF External DNS Only / Low Impact

### Sensitive Data Exposure
- **P1**: Disclosure of Secrets (Publicly Accessible Asset)
- **P1**: Disclosure of Secrets (Publicly Accessible Asset)
- **P2-P5**: Varies by context

### Cryptographic Weakness
- **P4**: Insecure Key Generation

### Hardcoded Credentials
- **P1**: Hardcoded Password — Privileged User
- **P2**: Hardcoded Password — Non-Privileged User
- **P5**: Sensitive Data Hardcoded (File Paths, OAuth Secret)

## How to Find the Right Category

The VRT is a searchable table. Use the search box on the Bugcrowd page to filter by keyword. The search covers vulnerability name and variant. Keywords that work well: "IDOR", "SSRF", "authentication", "race condition", "hardcoded".

## Bugcrowd's Stance on Chaining (from Usage Guide)

> *"many hackers have used such vulnerabilities within 'exploit chains' consisting of two or three vulns resulting in creative, valid, and high-impact submissions"*

Key quote from the Usage Guide tab:
> *"Low priority does not imply insignificance... Your internal teams or engineers might assess certain vulnerabilities – especially those designated P4 or P5 within the VRT – differently. As a hacker, it's important to not discount lower priority bugs, as many hackers have used such vulnerabilities within 'exploit chains' consisting of two or three vulns resulting in creative, valid, and high-impact submissions"*

**This means Bugcrowd explicitly endorses chaining lower-severity findings into higher-impact chains.**

## Chain Report Structure for Bugcrowd

When submitting a chained vulnerability, structure the report as one submission with:

1. **Title**: Describes the chained impact (e.g., "Complete Webhook Takeover via Authentication Bypass + HMAC Failure + Logic Error")
2. **VRT Category**: Use the HIGHEST severity category from the chain (e.g., "Authentication Bypass" P1 for a chain that starts with auth bypass)
3. **Chain Breakdown**: List each individual finding in the chain with source code references
4. **Attack Flow Diagram**: ASCII/text flow showing how each step composes
5. **VRT Classification Table**: Per-finding priority within the chain
6. **PoC**: A single script or set of scripts that demonstrates the full chain end-to-end
7. **Impact**: What a real attacker achieves, not just the individual bug

### When to use (and not to use) chaining

- **DO chain**: When individual findings are low/medium but together create a high/critical exploit path. Bugcrowd explicitly endorses this.
- **DO chain**: When a single prerequisite bug (e.g., auth bypass) enables exploitation of other bugs.
- **DON'T chain**: When findings are independent with different root causes and fixes. Submit separately.
- **DON'T chain**: When one finding is strong enough on its own (P1 standalone). Adding weaker findings dilutes signal.
