# Multi-Target Parallel Recon

## Overview

When the user says "find a target and recon it", the optimal workflow is:
1. **Load relevant skills first** (recon-to-exploitation, api-hacking-methodology, atlas-continuous-learning)
2. **Check multiple platforms simultaneously** (Intigriti, Bugcrowd, HackerOne)
3. **Iterate 10+ times** on each target, using continuous learning between iterations
4. **Report findings clearly** with what needs user help (credentials, app install, browser session)

## Program Selection Heuristics

When picking programs from Intigriti (sorted by "Last update: Newest first"):

| Signal | Meaning |
|--------|---------|
| **Few researchers** (< 25) | Undersubscribed — good chance to find unclaimed bugs |
| **Public** | Can start immediately, no application needed |
| **Has test credentials** | Fastest path to exploitation (e.g., Web PACS provided codes) |
| **Suspended** | Don't bother — submissions not accepted |
| **VDP** | No bounties — move on unless the user specifically wants Hall of Fame |
| **< 24h response time** | Fast feedback loop (e.g., Torfs at <16h) |
| **Recently updated** | May have new scope or fresh assets |

## Before Starting Deep Recon

1. Check if the **program is suspended** (page header) — skip if so
2. Check if **credentials are available** ("Get Credentials" button or FAQ)
3. Check **out of scope** list first — don't waste time on excluded vuln types
4. Note the **rate limit** — SFCC typically 1 req/sec

## When Hitting Auth Walls

1. **Check for test credentials** in FAQ/program page
2. **Check developer portals** for API keys or sandbox accounts
3. **Search for bug bounty getting-started PDFs** on Azure blob storage: `<company>bugbountyprod.z16.web.core.windows.net/*.pdf`
4. **Ask the user to register** with their @intigriti.me email
5. **Ask for browser DevTools session** (cookie copy, specific XHR cURL)

## Common Auth Patterns Found This Session

| Target | Auth Type | How to Get In |
|--------|-----------|---------------|
| Visma | Visma Connect OIDC | Student signup with training code + OAuth client creds |
| Skoda Auto | VW IDK OIDC PKCE | Download app, register via identity.vwgroup.io |
| Torfs/SFCC | SLAS JWT (cc-at cookie) | Register on storefront, cookie carries auth |
| Nexuzhealth/Web PACS | SimpleSAMLphp + code card | Test codes in FAQ (code + DOB) |
| Auth0 by Okta | Auth0 tenant with Management API | Bugcrowd researcher environment credentials |
