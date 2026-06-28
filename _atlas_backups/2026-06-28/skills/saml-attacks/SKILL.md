---
name: saml-attacks
description: "SAML authentication bypass testing — XML Signature Wrapping, NameID injection, assertion manipulation. Based on 2025-2026 CVE research."
version: 1.0.0
---

# SAML Attack Testing

## Prerequisites
Access to a SAML IdP metadata endpoint:
```bash
# Common SAML metadata paths
curl -s {base}/Shibboleth.sso/Metadata
curl -s {base}/simplesaml/saml2/idp/metadata.php
curl -s {base}/FederationMetadata/2007-06/FederationMetadata.xml
curl -s {base}/adfs/ls/metadata
```

## What to Check in Metadata
- `AuthnRequestsSigned` — are requests signed?
- `WantAssertionsSigned` — are assertions signed?
- Certificate strength (RSA 2048? Expiry date?)
- Protocol support (SAML 2.0 only or also 1.x?)

## Attack Vectors

### 1. XML Signature Wrapping (XSW)
Core technique: Insert a malicious Assertion into a valid Response while keeping the signature valid.

Workflow:
1. Capture a legitimate SAMLResponse
2. Decode: URL decode → Base64 decode → inflate (if compressed)
3. Clone the signed Assertion, modify NameID/attributes
4. Insert the modified Assertion before the signed one
5. The signature validator finds the original (still valid) while the consumer uses the modified one

### 2. NameID Manipulation
- Change the NameID to impersonate another user
- Test XML comment injection in NameID: `<saml:NameID>admin<!----></saml:NameID>`
- Test with different NameID formats (email, persistent, transient)

### 3. Assertion Injection
If you have ANY IdP-signed SAMLResponse, you can potentially:
1. Decode it
2. Replace the Assertion content (NameID, attributes, conditions)
3. Fix the digest references
4. Re-encode and send

### 4. Certificate Manipulation
- Check if the SP validates the certificate or just the signature
- Test with a self-signed certificate
- Test if the certificate is embedded in the response vs fetched from metadata

### 5. Response Processing
Test these variations:
- Send the response without the Signature element
- Send the response with an empty Signature
- Modify the `Destination` attribute
- Change `NotBefore`/`NotOnOrAfter` conditions
- Remove `AudienceRestriction`

## Tools Needed (not installed yet)
- SAML Raider (Burp extension) — for XSW attack generation
- Python scripts for XML manipulation
- xsw.py scripts from HackTricks/GitHub
