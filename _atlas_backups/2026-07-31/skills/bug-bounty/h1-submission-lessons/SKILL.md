---
title: HackerOne Submission Lessons & Duplicate Avoidance
name: h1-submission-lessons
description: Lessons learned from 6 HackerOne submissions (5 duplicates, 1 informative). Strategies to avoid duplicates and maximize signal.
---

# HackerOne Submission Lessons

## Track Record (as of June 2026)
| # | Report | Program | Status | Lesson |
|---|--------|---------|--------|--------|
| 1 | SSRF in Notion via Block URL Injection | Vercel OSS | Duplicate | Notion's URL fetch was already reported |
| 2 | SSRF in AI SDK `download()` | Vercel OSS | Duplicate | DNS bypass in AI SDK was known |
| 3 | SSRF Bypass in @ai-sdk/pr | Vercel OSS | Duplicate | Private IP range validation issue was known |
| 4 | Next.js Path Traversal | Vercel OSS | Duplicate | launch-editor path traversal was known |
| 5 | Kibana SSRF via Webhook Connectors | Elastic | Duplicate | Webhook allowlist bypass was known |
| 6 | SSRF in MCP Fetch + Claude Code WebFetch | Anthropic | Informative | Anthropic closed as informative (not a bug in their view) |

## Root Causes of Duplicates

### 0. Not Verifying That Found Keys Are Actually Usable

Before submitting "exposed API key" findings, verify each key against its partner's LIVE API:

- **SFOX key (Blockchain.com case)**: SFOX was acquired by Zero Hash (2023). `api.sfox.com` responds but partner endpoints return 404. Key is defunct.
- **Plaid key**: Requires BOTH `client_id` AND `secret`. Found only the `client_id` — worthless without the secret.
- **ShapeShift key**: Company converted to DEX; centralized exchange API no longer exists.
- **SiftScience key**: Client-side-only, designed to be public, origin-restricted.

Always run: `curl -sL "https://api.partner.com/v1/endpoint" -H "Bearer <key>"` and check if it returns usable data. A 200 with valid data = finding. Anything else = not a finding.

### 1. Not Checking Hacktivity / Resolved Reports Thoroughly
Before submitting, ALWAYS:
- Check the program's **Hacktivity** page for recent resolved reports with similar titles
- Search for related keywords in the program's public disclosures
- Check for **CVEs** matching the vulnerability class on the specific project
- Check GitHub Issues / Security Advisories for the project repo

### 2. Assuming SSRF Is Always Accepted
Not all programs accept SSRF:
- **Vercel OSS**: SSRF is a common class — many SSRF reports already exist, high duplicate risk
- **Anthropic**: Rejected AI agent SSRF as "informative" — their stance is UI:R and PR:L reduce severity below reward threshold
- **Elastic**: Webhook SSRF was already known

**Pre-submission check for SSRF:**
1. Search hacktivity for `SSRF` + program name
2. Check GitHub advisories for SSRF in the target
3. Check CVE database for SSRF in the specific project/version
4. If the program has 10+ SSRF reports already, assume high duplicate risk

### 3. Not Verifying Unique Bypass Techniques
When finding a bypass of an existing protection:
- Check if the SAME bypass technique was already reported
- A new bypass is NOT a new vulnerability if the underlying issue was already fixed
- Document which techniques are truly novel vs. variations of known bypasses

### 4. Informative Classification Patterns

Reports that get marked "informative" (not "duplicate" or "valid"):
- The program disagrees the issue is a vulnerability
- The program considers it a design choice ("by design")
- The program requires additional factors (UI:R, PR:L) that reduce severity below threshold
- The asset is out of scope but the report was reviewed anyway

**Case study — Anthropic SSRF (#6)**: Two independent failures:
1. MCP Fetch server: `github.com/modelcontextprotocol` is **explicitly out of scope**
2. WebFetchTool: Working as designed. The permission prompt IS the security boundary.

Anthropic's stance:
- **PR:L**: Requires authenticated Claude Code user (not unauthenticated remote)
- **UI:R**: User must approve each tool invocation
- **CVSS ~6.1 (Medium)**: PR:L + UI:R reduces exploitability below reward threshold

**Correct Anthropic vulnerability types** (from scope CSV):
- Bypassing permission prompts for unauthorized command execution
- Misrepresenting parameters or tools in permission prompts
- Hidden tool invocation / invisible command execution
- File writes outside working directory without prompt

## Duplicate Avoidance Checklist

Before any submission:

- [ ] Search HackerOne hacktivity for similar findings on the same program
- [ ] Search GitHub Security Advisories for the project
- [ ] Search CVE database for the vulnerability class + project
- [ ] Check if the program recently fixed a related issue (changelog, release notes)
- [ ] **Check for unmerged fix branches**: `git branch -a | grep -i "fix\\|auth\\|security\\|middleware"`
- [ ] For bypass findings: verify no one has published the SAME bypass technique
- [ ] For SSRF: especially high duplicate risk — check resolved report count
- [ ] Check the program's security model: for AI agents, the permission prompt may be the intended boundary

## Reference Files

- `references/h1-specific-workflow.md` — HackerOne-specific pre-submission workflow (scope CSV, hacktivity, decision table)
- `references/cross-program-escalation-research-june2026.md` — Cross-program escalation research

> **Generic pre-submission verification** (applies to all programs including Bugcrowd): Use the standalone skill `bug-bounty/pre-submission-verification` — covers universal gates + class-specific gates for 6 vulnerability types.

## Scoring Guidance — Impact Over Existence

**CRITICAL LESSON (June 2026, Bugcrowd triage):** Severity requires DEMONSTRATED IMPACT. An endpoint returning data without auth is P5 unless you can answer "as an attacker I could..." with a MEANINGFUL action.

The impact test:
1. What SPECIFIC action can the attacker perform with this data/access?
2. Does the leaked data enable a SECOND unauthorized action?
3. Can you DEMONSTRATE the full attack chain?
4. Is the exposed data SENSITIVE or trivial?

If you can't get past question 1, the finding is P5. Don't submit expecting a reward.

## Pre-Submission Fact-Check

- [ ] Every technical claim backed by verifiable source code or tool output
- [ ] No false causality claims
- [ ] CVSS matches actual exploitability (don't inflate)
- [ ] Asset is confirmed in scope
- [ ] No existing CVE for same bug class
- [ ] For bypass findings: technique is novel, not a variant of known issue
- [ ] Reproduction steps are concrete and reproducible

## Best Program Types for Our Skillset

1. **Newer programs** (fewer reports → fewer duplicates)
2. **TypeScript/Node.js open-source** (source code available)
3. **MCP / AI-related** (newer programs have fewer reports)
4. **Self-hostable software** (self-hosted bugs separate from cloud)
