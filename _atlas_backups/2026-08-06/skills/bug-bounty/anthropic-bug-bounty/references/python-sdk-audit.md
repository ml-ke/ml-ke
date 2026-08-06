# Claude Agent SDK Python — Source Code Audit

Date: 2026-05-31
Repo: `github.com/anthropics/claude-agent-sdk-python`
Version: current main (cloned to ~/Dev/claude-agent-sdk-python/)
Stars: 7.1k, Forks: 1.1k, Issues: 138

## Architecture

```
src/claude_agent_sdk/
├── __init__.py          # Public API — exports, SdkMcpTool, tool() decorator, create_sdk_mcp_server()
├── client.py            # ClaudeSDKClient — bidirectional interactive sessions
├── query.py             # query() — one-shot unidirectional API
├── types.py             # ClaudeAgentOptions, SandboxSettings, McpServerConfig, etc.
├── _cli_version.py      # CLI version detection
├── _errors.py           # CLINotFoundError, CLIConnectionError, ProcessError
├── _version.py          # SDK version
├── _internal/
│   ├── client.py        # InternalClient — communication with CLI subprocess
│   ├── query.py         # Query — control protocol message routing
│   ├── message_parser.py
│   ├── sessions.py      # Session management
│   ├── session_store.py, session_mutations.py, session_import.py, etc.
│   └── transport/
│       ├── __init__.py          # Transport ABC
│       └── subprocess_cli.py    # SubprocessCLITransport — CLI subprocess management (762 lines)
```

## Transport Layer Security (subprocess_cli.py)

### Command Construction
The SDK constructs the CLI command as a `list[str]` and spawns via `anyio.open_process(cmd, ...)` (line 474). Each option is a **separate array element**, preventing shell injection:

```python
cmd = [self._cli_path, "--output-format", "stream-json", "--verbose"]
# ... extensions for each option as separate list elements
cmd.extend(["--system-prompt", self._options.system_prompt])  # ✅ safe
cmd.extend(["--allowedTools", ",".join(effective_allowed_tools)])  # ✅ safe
```

**Verdict**: No shell injection possible through option value manipulation.

### `_build_command()` — Full CLI Flag Mapping (lines 221-410)

Each `ClaudeAgentOptions` field maps to a CLI flag:

| Options Field | CLI Flag | Notes |
|--------------|----------|-------|
| `tools` | `--tools` | Base set of available tools (allowlist) |
| `allowed_tools` | `--allowedTools` | Auto-approve list (NOT a whitelist) |
| `disallowed_tools` | `--disallowedTools` | Deny list |
| `system_prompt` | `--system-prompt` or `--system-prompt-file` | Supports string, file, or preset |
| `settings` | `--settings` | JSON string or file path; merged with sandbox |
| `sandbox` | (merged into `--settings`) | Injected into settings JSON |
| `mcp_servers` | `--mcp-config` | JSON string of MCP server configs |
| `extra_args` | `--{flag} {value}` | Arbitrary CLI flags (by design) |
| `plugins` | `--plugin-dir` | Local plugin directories only |
| `permission_mode` | `--permission-mode` | bypassPermissions, default, etc. |

### Environment Injection (lines 420-469)

The SDK merges env vars:
```python
process_env = {
    **inherited_env,
    "CLAUDE_CODE_ENTRYPOINT": "sdk-py",  # Signals non-interactive mode
    **self._options.env,
    "CLAUDE_AGENT_SDK_VERSION": __version__,
}
```

Key: `CLAUDE_CODE_ENTRYPOINT=sdk-py` sets non-interactive mode, which **bypasses the trust dialog** for hooks and MCP server approvals.

### `--settings` JSON Processing (lines 129-181)

The `_build_settings_value()` method processes `settings` + `sandbox` fields. If `settings` is a JSON string starting with `{`, it's parsed and potentially merged with `sandbox`:
```python
if sandbox is provided:
    settings_obj["sandbox"] = self._options.sandbox
return json.dumps(settings_obj)
```

This JSON is passed directly to the CLI's `--settings` flag, which loads as `flagSettings` — the highest-priority user-controlled settings layer. `flagSettings` hooks are loaded into merged settings and, in non-interactive mode, execute **without** trust dialog approval.

## Key Security Findings

### 1. `allowed_tools` Misunderstanding (Issue #361 — CLOSED)

**Finding**: `allowed_tools` is an auto-approve list, NOT a tool whitelist. Documentation describes it as:
> "Tool names that are auto-allowed without prompting for permission."

The actual tool restriction mechanism uses separate fields:
- `tools: list[str]` — Base set of available tools (allowlist)
- `allowed_tools: list[str]` — Auto-approves without permission prompt
- `disallowed_tools: list[str]` — Removes tools from available set

The SDK passes `--allowedTools` as a CLI flag (line 256-257). Whether the CLI respects this depends on the CLI version.

### 2. SDK Mode Trust Bypass (Architectural)

**Finding**: In SDK mode (`CLAUDE_CODE_ENTRYPOINT=sdk-py`), the CLI identifies as non-interactive. From `hooks.ts:286-296`:
```typescript
function shouldSkipHookDueToTrust(): boolean {
    const isInteractive = !getIsNonInteractiveSession()
    if (!isInteractive) { return false }  // SDK mode — NEVER skip hooks
    // ...
}
```

And from `mcp/utils.ts:393-403`, MCP servers are auto-approved in non-interactive mode with `projectSettings` enabled.

**Impact**: If an application wraps the SDK and passes user-controlled content to `settings` (hooks with `command` type), the hooks execute without any trust dialog. This is NOT remotely exploitable from the SDK alone — it requires the host application to pass dangerous content.

### 3. `extra_args` — Arbitrary Flag Injection (By Design)

**Finding**: The `extra_args` option (types.py:1729, subprocess_cli.py:364-370) intentionally allows passing arbitrary CLI flags:
```python
for flag, value in self._options.extra_args.items():
    if value is None:
        cmd.append(f"--{flag}")
    else:
        cmd.extend([f"--{flag}", str(value)])
```

**Impact**: SDK consumers could inject `--dangerously-skip-permissions`, `--settings`, `--mcp-config` with malicious content, etc. This is a deliberate flexibility feature — the SDK trusts its consumer.

### 4. MCP Server Command Injection (By Design)

**Finding**: The `mcp_servers` option accepts a dict with MCP server configs. Each config has `command` and `args` fields that specify an executable to spawn. The SDK passes this as:
```python
"--mcp-config", json.dumps({"mcpServers": servers_for_cli})
```

The CLI parses this and spawns the specified command as an MCP server subprocess. No validation on the command path beyond the McpServerConfigSchema.

### 5. Open Issue #448 — Plugin MCP Permissions (OPEN)

**Status**: Open since Dec 31, 2025 — 6+ months unresolved.

The issue asks whether plugins that include MCP servers automatically get permission to use those tools, or whether they go through the standard `canUseTool` callback / permission system. The documentation doesn't clarify:
1. If plugin MCP tools bypass `canUseTool` callbacks
2. If plugin MCP tools are automatically approved when plugin is enabled
3. How to require explicit user approval for plugin MCP tools

**If** plugins bypass the permission system, this is a privilege escalation vector: a plugin could include an MCP server that performs dangerous operations without the host application being able to intercept.

### 6. No Credential Leakage in Transport

The transport doesn't log credentials or API keys. The subprocess passes environment variables via the `env` parameter to `anyio.open_process()`, not through shell expansion. No credential leakage found.

## Test Vectors for Future Sessions

1. **Verify `--allowedTools` CLI support**: Create a script that passes `--allowedTools Read,Grep,Glob` to `claude` and check if other tools are actually restricted.
2. **Verify SDK mode trust bypass**: Pass a `--settings` JSON with hooks to the CLI in SDK mode and check if hooks execute without trust dialog.
3. **Check Issue #448 resolution**: Monitor if Anthropic clarifies plugin MCP permission behavior. If plugins bypass permissions, this is reportable.
4. **Test `sandbox` settings injection**: Pass `sandbox={"hooks":{...}}` via the SDK and check if hooks from nested sandbox settings execute.
5. **Audit `disallowed_tools`**: Check if `disallowed_tools` properly removes tools from the model's context or if they're still accessible through alternative names/paths.
