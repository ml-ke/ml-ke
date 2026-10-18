# GitHub Advisory Landscape — anthropics/claude-code

Audited: 2026-05-31
Repo: github.com/anthropics/claude-code
Total advisories: 27

## Researchers by Findings Count

### ddworken (Anthropic security engineer — highest volume)
- Sandbox Escape via Persistent Configuration Injection in settings.json (High)
- Command Injection via Directory Change Bypasses Write Protection (High)
- Command Injection via Piped sed Command Bypasses File Write Restrictions (High)
- Permission Deny Bypass Through Symbolic Links (High)
- Claude Code vulnerable to arbitrary code execution caused by maliciously configured git email (High) — GHSA-j4h9-wv2m-wrf7
- Claude Code Vulnerable to Arbitrary Code Execution Due to Insufficient Startup Warning (High) — GHSA-ph6w-f82w-28w6
- Command Injection in Claude Code rg command allowed bypass of user approval prompt (High)

### OctavianGuzu (External researcher)
- Local Privilege Escalation via Directory Junction in CoworkVMService (High)
- SSH Host Key Verification Bypass Allows Man-in-the-Middle Attack on Remote Sessions (High)
- Insecure System-Wide Configuration Loading Enables Local Privilege Escalation on Windows (Moderate)
- Trust Dialog Bypass via Git Worktree Spoofing Allows Arbitrary Code Execution (High)
- Sandbox Escape via Symlink Following Allows Arbitrary File Write Outside Workspace (High)

### jenn-newton
- Permissive Default Allowlist Enables Unauthorized File Read and Network Exfiltration in Claude Code (High)
- Command Injection in Claude Code echo command allowed bypass of user approval prompt for command execution (High) — GHSA-x56v-x2h6-7j34
- Path Restriction Bypass in Claude Code Research Preview could allow unauthorized file access when path prefixes collide (High)

### dmckennirey-ant
- Workspace Trust Dialog Bypass via Repo-Controlled Settings File (High) — CVE-2026-33068

### petery-ant
- Claude Code IDE extensions allow websocket connections from arbitrary origins

## Attack Surface Summary (Exhausted)

| Surface | Advisories | Key Researchers | Remaining Potential |
|---------|-----------|-----------------|---------------------|
| Sandbox escape | 3+ | ddworken, OctavianGuzu | Low — thorough coverage |
| Trust dialog bypass | 3+ | OctavianGuzu, dmckennirey-ant | Low — multiple variants found |
| Command injection | 5+ | ddworken, jenn-newton | Medium — new tools/commands may have issues |
| Privilege escalation | 2 | OctavianGuzu | Medium — Cowork feature is new |
| Network/SSH | 2 | OctavianGuzu, petery-ant | Low |
| Default config | 1 | jenn-newton | Low |
| Path restriction | 1 | jenn-newton | Low |

## Related SDK Advisories (Other Repos)

- **anthropic-sdk-typescript** — GHSA-p7fg-763f-g4gf: BetaLocalFilesystemMemoryTool world-readable files (patched)
- **claude-agent-sdk-python** — Issue #361: `allowed_tools` whitelist bypass (closed w/o fix)
- **claude-agent-sdk-python** — Issue #448: Plugin MCP server permissions unclear (open 6+ months)
- **claude-agent-sdk-typescript** — CHANGELOG mentions bumping dependencies for GHSA-5474-4w2j-mq4c and transitive hono advisories
