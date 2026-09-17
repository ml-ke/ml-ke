# Anthropic H1 Program Scope (from CSV, May 2026)

## Core Assets (Bounty-Eligible)

| Asset | Type | Instructions |
|-------|------|-------------|
| claude.ai | URL | Claude chat interface — XSS, auth bypass, conversation IDOR |
| API & SDKs | AI_MODEL | api.anthropic.com, Python SDK, TypeScript SDK. Test on api-staging.anthropic.com |
| Official Clients | OTHER | iOS, Android, Desktop apps — MCP integrations, local storage, IPC |
| Claude Code | OTHER | **Hot target.** See vulnerability types below |
| console.anthropic.com | URL | Developer console — API key mgmt, workspace mgmt, billing |
| api.anthropic.com | URL | API endpoint testing |
| Leaked Employee API Keys | OTHER | Publicly leaked keys — authenticate then immediately deauth |

## Non-Core Assets (May still get bounty)

| Asset | Notes |
|-------|-------|
| github.com/anthropics | Source code — archived repos and forks excluded unless impact on Anthropic assets is shown |
| docs.anthropic.com | Documentation |
| support.anthropic.com | Support portal |
| anthropic.atlassian.com | Jira — 4 queues intentionally public. URL asset for external access only |
| Infrastructure & Internal Apps | Auth bypass, privilege escalation, exposed internal services |
| Claude Desktop Extensions & Claude.ai MCP servers | Anthropic-developed connectors only (not third-party) |

## EXPLICITLY OUT OF SCOPE

| Asset | Why |
|-------|-----|
| **github.com/modelcontextprotocol** | OSS MCP code — report to MCP maintainers' GitHub Security page. Issues affecting Anthropic's IMPLEMENTATION of MCP (in Claude.ai, Desktop, etc.) ARE in scope under the respective asset |
| | **This was our mistake in Report #6** |

## Claude Code — Valid Vulnerability Types

From the H1 scope CSV (May 2026 version):

### In Scope
- **Bypassing permission prompts for unauthorized command execution** — making commands run WITHOUT the user seeing a prompt
- **Bypassing permission prompts for file write operations outside working directory** — writing outside allowed paths without prompt
- **Misrepresenting parameters or tools in permission prompts** — prompt shows different information than what executes
- **Executing commands or tools invisibly to users** — zero visibility execution
- Hidden tool invocation — tools execute without appearing in conversation history

### Out of Scope
- Abusing intended functionality of Claude CLI
- Using aliased commands, symlinks, or environment-specific settings to bypass permission prompts
- Local storage of Claude Code credentials, configuration, and logs
- **WebFetch reaching internal hosts** — the permission prompt IS the security boundary (this was our mistake)
- **MCP servers accessing any URL** — documented behavior (this was our mistake)

## Source Code Assets

| Repo | In Scope? | Notes |
|------|-----------|-------|
| github.com/anthropics/claude-code | YES | 129k stars, 5k+ issues, 27 advisories (well-researched) |
| github.com/anthropics/claude-code-action | YES | GitHub Action for Claude Code |
| github.com/anthropics/anthropic-sdk-python | YES | Python SDK |
| github.com/anthropics/anthropic-sdk-typescript | YES | TypeScript SDK |
| github.com/anthropics/claude-agent-sdk-python | YES | Agent SDK — 7.1k stars, 138 issues |
| github.com/anthropics/claude-agent-sdk-typescript | YES | TypeScript Agent SDK — 1.5k stars |
| github.com/anthropics/claude-plugins-official | YES | Plugin directory |
| github.com/anthropics/skills | YES | Skills/prompts — 145k stars, markdown only |
| **github.com/modelcontextprotocol** | **NO** | OSS MCP code |
