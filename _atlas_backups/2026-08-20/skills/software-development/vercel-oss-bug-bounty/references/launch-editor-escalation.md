# Next.js launch-editor — Impact Escalation Analysis

## Overview

The `__nextjs_launch-editor` endpoint (exposed in dev mode by `middleware-webpack.ts:634` and `middleware-turbopack.ts:398`) allows an attacker to control the `file` GET parameter which flows through `openFileInEditor()` → `launchEditor()` → `child_process.spawn()`. Absolute paths bypass the project root restriction due to `path.isAbsolute(file)` check at `launch-editor.ts:452`.

## CVE-2025-11953 Precedent (Metro4Shell)

**This vulnerability class was weaponized in the wild.** React Native's Metro development server had the identical vulnerability — an `/open-stack-frame` endpoint that opens files in the developer's editor (CVE-2025-11953, CVSS 9.8). Attackers:
- Scanned the internet for exposed Metro servers (~3,500 found)
- Sent POST requests with PowerShell payloads via the `url` parameter
- Deployed Rust malware on developer machines
- Used compromised machines for cryptomining, credential theft, and supply chain attacks

Key difference: Metro bound to ALL network interfaces by default. Next.js binds to localhost by default, but `--host 0.0.0.0` creates identical exposure.

## 6 Escalation Paths

### 1. Network Exposure via `--host`
- Default: localhost only (safe from network)
- With `--host 0.0.0.0`: Anyone on the LAN can reach the endpoint
- Common in Docker setups, mobile testing, CI ephemeral environments
- Attack: `GET http://victim:3000/__nextjs_launch-editor?file=/etc/passwd`

### 2. CSRF → RCE Auto-Trigger
The endpoint is a simple GET with no CSRF protection. An `<img>` tag on any page the developer visits triggers it automatically:
```html
<img src="http://localhost:3000/__nextjs_launch-editor?file=test.js" width="0" height="0">
```
On Windows, if REACT_EDITOR contains shell metacharacters, cmd.exe evaluates them:
```bash
REACT_EDITOR=code & calc.exe
# cmd.exe /C code & calc.exe test.js
# The & is a command separator → calc.exe runs
```

### 3. Multiple Env Var Entry Points
`launch-editor.ts:225-287` — `guessEditor()` checks in order:
1. `REACT_EDITOR` — passed through `shellQuote.parse()` (line 228)
2. Running process detection via `ps x` (macOS/Linux) or `wmic` (Windows)
3. `VISUAL` — used DIRECTLY as single-element array (line 280-281)
4. `EDITOR` — used DIRECTLY as single-element array (line 282-283)

All four paths lead to `spawn()` with the editor value.

### 4. Supply Chain via .env.local
Next.js auto-loads `.env.local` from the project root. Attack vectors:
- Malicious npm package with `.env.local` in starter template
- Postinstall script appending to `.env.local`: `"postinstall": "echo REACT_EDITOR=code_\\&_malicious >> .env.local"`
- Git repo with committed `.env.local`

### 5. Linux Editor-to-RCE
On Linux, `spawn(editor, args)` at line 419 doesn't use a shell, but if `$EDITOR` points to a binary (e.g., `/tmp/malware.sh`), it executes directly:
```bash
# .env.local
EDITOR=/tmp/revshell.sh
```

### 6. File Existence Oracle
The endpoint returns different HTTP status codes:
- 204 No Content → file exists and editor launched
- 500 Internal Server Error → file exists but editor failed (leaks error details in body)
- 404 Not Found → file does not exist

This maps the filesystem without authentication.

## References
- CVE-2025-11953 (Metro4Shell) — identical vulnerability in React Native Metro server
- Source: `packages/next/src/next-devtools/server/launch-editor.ts`
- Source: `packages/next/src/server/dev/middleware-turbopack.ts` line 398
- Source: `packages/next/src/server/dev/middleware-webpack.ts` line 634
