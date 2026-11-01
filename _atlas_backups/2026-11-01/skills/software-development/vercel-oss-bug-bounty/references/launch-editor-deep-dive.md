# Next.js launch-editor: Path Traversal + Windows RCE (Dev-Only) — FACT-CHECKED

## File Locations
- **Core logic**: `packages/next/src/next-devtools/server/launch-editor.ts` (475 lines)
- **Endpoint handler (webpack)**: `packages/next/src/server/dev/middleware-webpack.ts:634-677`
- **Endpoint handler (turbopack)**: `packages/next/src/server/dev/middleware-turbopack.ts:398-415`
- **Response handler (file existence oracle)**: `packages/next/src/next-devtools/server/middleware-response.ts`

## Endpoint
```
GET /__nextjs_launch-editor?file=PATH&line1=1&column1=1&isAppRelativePath=0
```

## Path Traversal (Linux & macOS)

The `file` query parameter flows to `openFileInEditor()` without path restriction:

```typescript
// launch-editor.ts:452-455
if (path.isAbsolute(file)) {
  filePath = file;           // ← ABSOLUTE PATH USED DIRECTLY — no chroot
} else {
  filePath = path.join(nextRootDirectory, file);
}
```

**Key finding — Linux has NO filename validation at all**: Unlike Windows (which has a regex allowlist at line 395), Linux and macOS accept ANY absolute path. The `canAccess` check at line 324 tests existence via `fs.promises.access` with no path restriction. On Linux, `/etc/passwd` is reachable and would be opened in the editor.

**Attack**: `GET /__nextjs_launch-editor?file=/etc/passwd` → opens `/etc/passwd` in the configured editor.

## File Existence Oracle (Confirmed)

The endpoint reveals whether a file exists via three distinct response codes, verified from `middleware-response.ts`:

| Response | Meaning | Evidence |
|----------|---------|----------|
| **204 No Content** | File exists, editor launched successfully | `res.status(204).end()` |
| **404 Not Found** | File does not exist | `res.status(404).json({ error: 'File not found' })` |
| **500 Internal Server Error** | File exists but cannot be read (permissions, binary, etc.) | `res.status(500).json({ error: inspect(error) })` — leaks error details via `inspect()` |

**Triage impact**: File existence oracle is independently useful for information gathering (enumerating config files, SSH keys, certificates) even without the editor trigger.

## RCE Vector (Windows) — FACT-CHECKED CORRECTIONS

Line 401:
```typescript
p = child_process.spawn('cmd.exe', ['/C', editor].concat(args), {
  stdio: 'inherit',
  detached: true,
})
```

`cmd.exe /C` evaluates shell metacharacters in the concatenated command line. **However, not all env vars enable RCE**:

### ⚠️ REACT_EDITOR → NO RCE

`REACT_EDITOR` value is parsed by `shellQuote.parse()` (line 228-236). This function splits the input into an array of tokens, converting shell metacharacters (`&`, `|`, `;`) into **operator objects** `{op:"&"}` rather than the raw character:

```javascript
// shellQuote.parse("code & calc.exe")
// → ["code", {op:"&"}, "calc.exe"]
```

When passed to `child_process.spawn()`, JavaScript calls `.toString()` on the objects, producing the string `"[object Object]"` — **not** the operator character. This means:

- `REACT_EDITOR=code & calc.exe` → spawn receives `["code", "[object Object]", "calc.exe"]` → harmless
- `REACT_EDITOR=code|whoami` → spawn receives `["code", "[object Object]", "whoami"]` → harmless
- `REACT_EDITOR=$(whoami)` → shellQuote.parse treats this as an operator too

**Verified via Node.js test**: `child_process.spawn('cmd.exe', ['/C', '[object Object]'])` prints `[object Object]` — no shell injection.

### ✅ EDITOR / VISUAL → RCE CONFIRMED

`EDITOR` and `VISUAL` are used **directly as raw strings** (no shellQuote.parse). They are concatenated into the spawn args array as-is:

```typescript
// launch-editor.ts:250 (getAvailableEditors)
// EDITOR and VISUAL are pushed directly to editors array
```

**Constraint — space-free payloads only**: When the editor name (from `EDITOR` or `VISUAL`) contains spaces, Node.js wraps the entire arg in double quotes. This makes the `&` character **literal** rather than a shell metacharacter:

| Payload | Spawn arg | cmd.exe sees | RCE? |
|---------|-----------|--------------|------|
| `code&whoami` | `["code&whoami"]` | `code` then `whoami` | ✅ YES |
| `code & whoami` | `['"code & whoami"']` | `"code & whoami"` as literal | ❌ NO |

**Working PoC payload**: `EDITOR=code&calc.exe` (no quotes, no spaces).

## Editor Resolution Order

`guessEditor()` at line 225:

1. `REACT_EDITOR` env var → parsed by `shellQuote.parse()` — splits into array. **No RCE via this path** (operator objects → `[object Object]`).
2. `ps x` running process detection (macOS/Linux) or `wmic` (Windows)
3. `VISUAL` env var → **used directly as raw string** → RCE possible
4. `EDITOR` env var → **used directly as raw string** → RCE possible

## CSRF Auto-Trigger (No Origin/Referer Check)

The endpoint is a simple `GET` request with **no CSRF protection** — no Origin header validation, no Referer check, no anti-CSRF token:

```typescript
// middleware-webpack.ts:634 — GET handler, no CSRF middleware
server.get('/__nextjs_launch-editor', async (req, res) => {
  const file = String(req.query.file || '');
  // ... no Origin/Referer validation ...
});
```

Auto-trigger via HTML:
```html
<img src="http://target-dev:3000/__nextjs_launch-editor?file=C:\Windows\System32\calc.exe" />
```

On Windows with `EDITOR=code&calc.exe` set, this triggers RCE without user interaction.

## Supply Chain Attack Vector

1. Attacker creates a malicious npm/git repo with a `.env.local` containing:
   ```
   EDITOR=code&calc.exe
   ```
2. Developer clones the repo, runs `npm install && npm run dev`
3. Any compilation error triggers the error overlay
4. A `<img src="...">` on a visited page auto-triggers the endpoint
5. On Windows, cmd.exe executes the injected command

Note: `REACT_EDITOR` in `.env.local` does NOT enable RCE. Must use `EDITOR` or `VISUAL`.

## CVE-2025-11953 Precedent (Metro4Shell, CVSS 9.8)

React Native Metro Bundler had the **exact same vulnerability** — an `/open-stack-frame` endpoint that opened files via the developer's editor without path validation. Key details:

- **Active exploitation**: 3,500+ exposed servers found by Censys / Shodan
- **Malware deployed**: Rust-based malware on developer machines (data exfiltration, persistence)
- **Fix**: Sanitized file paths to restrict within project directory
- **Our bug**: Same class — different product (Next.js vs Metro), same issue (no path restrict on launch-editor)

**Why this matters for triage**: The CVE proves this vulnerability class IS exploitable in the wild, not just theoretical. Our `--host 0.0.0.0` pattern matches the Metro exploitation vector exactly.

## Limitations

- Dev-only (`next dev`) — not reachable in production builds
- On Linux, `spawn(editor, args)` without shell — attacker controls which binary runs, not shell injection
- File existence check limits arbitrary read to known files
- Windows RCE via EDITOR/VISUAL requires space-free payloads
- REACT_EDITOR does NOT enable RCE (shellQuote.parse strips shell metacharacters)

## Related Code Paths

The `__nextjs_launch-editor` endpoint is also exposed via the dev overlay:
- `packages/next/src/next-devtools/dev-overlay/utils/use-open-in-editor.ts`
- `packages/next/src/next-devtools/dev-overlay/components/overview/segment-explorer.tsx`
- `packages/next/src/next-devtools/dev-overlay/components/overview/segment-boundary-trigger.tsx`

These components trigger the endpoint on user click — they don't auto-fire.
