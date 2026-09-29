# Architecture-Aware Hunting (detail)

Detail moved out of SKILL.md for progressive disclosure (loaded on demand).

## Architecture-Aware Hunting (Added Jun 2026 from meta-analysis)

### The Two-Layer Auth Trap

Many modern APIs use a two-layer architecture:
- Layer 1 — Session/Gateway: Accepts connections, manages sessions, lenient auth
- Layer 2 — Data/Service: Enforces actual auth, validates tokens, serves data

Pattern: Layer 1 accepting any token is NOT an auth bypass if Layer 2 independently validates. Compare with CVE-2024-8954 (Composio) where the same broken check was the ONLY gate.

To distinguish a real auth bypass from architecture:
1. Does the token-accepting component ALSO serve protected data? Yes = finding
2. Does the token get forwarded to a downstream service that independently validates? Yes = architecture, not finding
3. Can you find a method/endpoint on Layer 1 that DOESNT forward to Layer 2? Maybe = check for direct data access

### The Title Test

Every finding must pass this test: The title must end with an action an attacker can take.

| Pass | Fail |
|------|------|
| Auth bypass leading to read customer PII from /api/users | Authentication Bypass in MCP Server |
| IDOR on /api/invoices leading to view any user billing data | Insecure Direct Object Reference |
| SSRF via webhook URL leading to read cloud metadata | Server-Side Request Forgery |
| CORS misconfig leading to exfiltrate user data cross-origin | CORS allows any origin |

If you can't write a title that ends with "leads to [action]", the finding isn't developed.

### The Data Classification First Principle

Reverse the order of hunting:
1. First: What PII, credentials, or actionable data does this system handle?
2. Second: Where does that data flow? Which endpoints serve it?
3. Third: Are those endpoints properly protected?

Not:
1. Find interesting endpoint > probe > find behavior > theorize impact > submit

### VDP-Specific Considerations

VDPs accept a wider range of findings than paid bounties but still require:
- Demonstrable security impact (not best-practice violations)
- Clear reproduction steps
- No prohibited testing (DoS, social engineering, physical)
- For no-payout VDPs: focus on high confidence findings with CLEAR impact
  - Exposed admin panels with default creds
  - Confirmed SSRF to metadata endpoints
  - PII leakage via IDOR
  - Subdomain takeover with active service
  - Publicly accessible internal tools/dashboards
- Avoid for VDPs: missing headers, version disclosure, theoretical issues, business metadata

### Information Sources We Underutilized

| Source | What It Finds | Why We Missed It |
|--------|--------------|------------------|
| Shodan/Censys | Exposed instances, dev servers, admin panels | We probed live domains directly instead of discovering shadow IT |
| GitHub dorking | Leaked creds, internal docs, config files | Limited use; didn't search for target in code comments/commits |
| Source code analysis | Auth annotations, middleware chains, config-guarded features | Only done for open-source targets; should check npm/PyPI for target SDKs |
| Nuclei templates | Known CVEs and misconfigs | Rarely ran; could find standing vulns faster |
| Wayback Machine diffs | Removed endpoints, historical JS with API routes | Used wayback URLs but didn't diff versions for security-relevant changes |
| C99 subdomain scans | 50+ data sources for subdomain discovery | Didn't use on most targets |
| OIDC well-known configs | Custom scopes, grant types, JWKS URLs | Only checked on Coolblue; should be standard recon step |

