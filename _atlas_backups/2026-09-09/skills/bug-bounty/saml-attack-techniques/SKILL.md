---
name: saml-attack-techniques
category: bug-bounty
description: SAML 2.0 authentication bypass techniques from PortSwigger research (The Fragile Lock) and prior work. Directly applicable to Rapyd SAML promotion and any SAML-integrated target.
---

# SAML 2.0 Attack Techniques

Based on "The Fragile Lock: Novel Bypasses For SAML Authentication" — Zakhar Fedotkin, PortSwigger Research (Jan 2026)
Also incorporates "SAML Roulette: The Hacker Always Wins" and prior XSW research.

## Core Concept

SAML security depends on **XML Signature validation**. The signature verification and assertion processing are often handled by separate modules/parsers (e.g., REXML vs Nokogiri in Ruby-SAML). **Parser discrepancies** between these components are the root cause of most SAML bypasses.

## Attack Techniques

### 1. XML Signature Wrapping (XSW)
Inject a malicious unsigned Assertion alongside the legitimate signed one. The signature verifier validates the legitimate portion; the business logic consumes the attacker's injected Assertion.

**Requirement**: A valid signed SAML document from the target IdP.

### 2. Attribute Pollution
Multiple ID-type attributes (e.g., `ID` and `samlp:ID`) cause different parsers to select different values. Libxml2's `xmlGetProp` ignores namespaces and returns undefined attribute when duplicates exist by simple name. Attribute ORDER determines which wins.

```
<samlp:Response ID="attack" samlp:ID="real_id">
```
REXML may select "attack" while Nokogiri selects "real_id" — or vice versa depending on order.

### 3. REXML Namespace Confusion (No DTDs)
REXML treats `xmlns` as a regular attribute. By redefining namespace declarations, an attacker can split signature detection between parsers:
```
<Parent xmlns='http://www.w3.org/2000/09/xmldsig#'>
  <Child xml:xmlns='#anything'>
    <Signature/>  <!-- REXML sees this, Nokogiri doesn't -->
  </Child>
</Parent>
```

### 4. Void Canonicalization
Relative namespace URIs (e.g., `xmlns:ns="1"`) cause libxml2 canonicalization to ERROR. Instead of failing securely, Nokogiri returns an **empty string**. The digest is then computed over empty input → `SHA256("") = 47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=`

**Golden SAML Response**: A precomputed signature for empty string can be reused indefinitely across any message.

### 5. Extensions/StatusDetail Injection
The SAML XML Schema defines extension points (Extensions, StatusDetail) that appear **before** the Signature element. Malicious Assertions injected here will be processed before the legitimate one:
```
<samlp:Response>
  <samlp:Extensions>
    <!-- Attacker's hidden Signature with real IdP signature goes here -->
  </samlp:Extensions>
  <Assertion>
    <Signature>
      <!-- Fake signature that gets bypassed via Void Canonicalization -->
    </Signature>
  </Assertion>
</samlp:Response>
```

### 6. Signed Error Responses
When a SAML request is malformed (e.g., invalid `IssueInstant`), the IdP may still return a **signed error response**. This signed error document can be used as the source of a valid signature for wrapping attacks.

### 7. WS-Fed / IdP Metadata as Signature Source
Microsoft Entra ID and Okta (~85% of GitHub users) publish **signed metadata by default**. These documents provide convenient, publicly-available signed XML that can be repurposed.

## Testing Methodology

### Step 1: Determine SAML Library
Check the HTTP response for clues:
- `X-SAML-*` headers
- Error messages revealing library/version
- Known library patterns (Ruby-SAML, python3-saml, saml2-js, etc.)

### Step 2: Capture a Valid SAML Response
1. Set up a SAML IdP (Okta, Entra ID, Google Workspace, or SimpleSAMLphp)
2. Configure it for the target SP using the ACS URL and Entity ID
3. Authenticate and capture the SAMLResponse POST
4. Base64-decode and inspect the XML

### Step 3: Test for Signature Wrapping
For each XSW technique (Attribute Pollution, Namespace Confusion, Void Canonicalization):
1. Create a modified SAML Response with a forged Assertion
2. Inject it using schema-allowed extension points (Extensions, StatusDetail)
3. Base64-encode and POST to the ACS URL
4. Check if authentication is granted with the forged identity

### Step 4: Test for Replay
1. Capture a valid SAMLResponse
2. Re-submit it without modification
3. If accepted → Assertion Replay vulnerability

### Step 5: Test InResponseTo
1. Capture a valid AuthnRequest ID from the SP
2. Submit a SAMLResponse with matching InResponseTo
3. Submit the SAME response again → check if InResponseTo reuse is rejected

### Step 6: Test RelayState
1. Modify the RelayState parameter to an external URL
2. Check if redirect follows the modified value
3. If yes → Open Redirect

### Step 7: Test Email/NameID Spoofing
1. Create a SAML Assertion with a different email in NameID
2. If the SP accepts it → Account Takeover via Email Spoofing

## Tools
- **Burp SAML Raider** — SAML message manipulation in Burp
- **XSW Toolkit** (github.com/d0ge/XSW) — Automates XSW exploit generation (from The Fragile Lock paper)
- **SAMLStorm** — Node.js xml-crypto bypass toolkit
- **Custom Python** — Use `signxml` library for XML signature manipulation

## Key References
- "The Fragile Lock" — PortSwigger Research, Jan 2026 (PDF in books folder)
- "SAML Roulette: The Hacker Always Wins" — PortSwigger Research
- "SAMLStorm" — WorkOS research on xml-crypto bypasses
- "Abusing libxml2 quirks to bypass SAML on GitHub Enterprise (CVE-2025-23369)"
- OWASP SAML Security Cheat Sheet
