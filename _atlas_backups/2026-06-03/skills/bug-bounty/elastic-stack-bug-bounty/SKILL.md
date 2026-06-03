---
title: Elastic Stack / Kibana Bug Bounty
name: elastic-stack-bug-bounty
description: Systematic approach to finding vulnerabilities in Elastic Stack (Kibana + Elasticsearch) for the HackerOne bug bounty program. Combines static analysis on the cloned Kibana TypeScript/Node.js codebase with live testing against cloud.elastic.co.
---

# Elastic Stack / Kibana Bug Bounty Methodology

## Program Overview

> **NOTE:** This skill overlaps significantly with `elastic-kibana-bug-bounty`. The other skill is more comprehensive and up-to-date (includes live cloud testing results, `.http` connector URL override, latest CVEs). Use `elastic-kibana-bug-bounty` as the primary reference; this skill is maintained for Kibana+ES combined context but may lag behind.

- **Program**: Elastic on HackerOne ($50-$7k bounties for Kibana)
- **Source**: `github.com/elastic/kibana` — TypeScript/Node.js (~110K files)
- **Clone**: `~/Dev/elastic/kibana/`
- **Key vulnerabilities wanted**: Kibana XSS with CSP bypass, authenticated SSRF, IDOR, privilege escalation, RCE
- **Triaged by H1**: Yes. Avg 10h first response.

## Kibana Codebase Architecture

### Modern Directory Layout (v9+)

Kibana no longer uses the old `x-pack/plugins/` layout. Key locations:

```
src/core/packages/           # Core server packages (HTTP, ES client, CSP, saved objects)
  http/server-internal/src/csp/  # CSP configuration (config.ts, csp_directives.ts, csp_config.ts)
  saved-objects/import-export-server-internal/src/import/  # Saved objects import pipeline

x-pack/platform/plugins/shared/  # Shared plugins (actions, stack_connectors, triggers_actions_ui)
  actions/server/               # Main actions/connectors framework
  actions/server/sub_action_framework/  # Sub-action framework (executor, validators)
  stack_connectors/server/      # Built-in connector types (webhook, email, etc.)

x-pack/solutions/            # Solution-specific code (security, observability, search)
  security/packages/connectors/  # Security-specific connector code
```

There is NO `x-pack/plugins/` directory — it's been reorganized.

### CSP Configuration (`src/core/packages/http/server-internal/src/csp/`)

**Critical finding**: Kibana's CSP is NOT nonce-based despite `strict: true`.

- `strict: true` (default) only **validates** that `unsafe-inline` isn't used with `script-src`
- It does NOT add `nonce-*` or `strict-dynamic` to the CSP header
- Default CSP: `script-src 'report-sample' 'self'`
- `style-src` includes `'unsafe-inline'` — CSS injection is possible
- `disableUnsafeEval: true` (default) — `unsafe-eval` omitted from script-src
- Admin can override via `csp.script_src: [...]` in kibana.yml
- `nonce-*` is explicitly **rejected** by validation (`allowNonce: false` in all directive validators)

**Implication for XSS hunting**: With `script-src 'self'`, executing arbitrary JS requires:
1. Uploading a .js file to the Kibana server (file upload feature? custom integration?)
2. JSONP endpoint abuse on the same origin
3. Angular template injection (if Angular is still used anywhere)
4. CSS injection via `style-src 'unsafe-inline'` to exfiltrate data or manipulate DOM

### Actions/Connectors Framework (`x-pack/platform/plugins/shared/actions/server/`)

**SSRF Protection Architecture:**

1. **`actions_config.ts`** — Core configuration utilities
   - `isHostnameAllowed(hostname)` — checks hostname against `allowedHosts` set
   - `isUriAllowed(uri)` — parses URL with Node `url.parse()`, extracts hostname, checks against set
   - `AllowedHosts.Any = '*'` — if `allowedHosts` contains `'*'`, **all hosts are allowed**
   - Default: `allowedHosts` is `['*']` on self-managed Kibana
   - Admin must explicitly configure `xpack.actions.allowedHosts` to restrict

2. **`request()` function (axios_utils.ts)** — Every outbound HTTP request goes through:
   - `configurationUtilities.ensureUriAllowed(url)` at line 60
   - Custom HTTP/HTTPS agents via `getCustomAgents()`
   - Redirect validation via `getBeforeRedirectFn(configurationUtilities)`
   - Proxy: `false` (explicitly disables axios proxy — uses custom agents)

3. **`getBeforeRedirectFn` (before_redirect.ts)** — Validates redirect targets:
   - Checks `options.hostname` from axios redirect options
   - Runs `configurationUtilities.ensureHostnameAllowed(hostname)`
   - Throws on null/non-string hostname
   - Protected against redirect-to-internal attacks

4. **Sub-action framework (`sub_action_framework/`)**:
   - `executor.ts` — calls `service[method]()` which delegates to connector instance
   - `sub_action_connector.ts` — base class, HTTP calls via `request()` from axios_utils:
     1. `assertURL(url)` — validates protocol is http/https, has hostname
     2. `ensureUriAllowed(url)` — checks hostname against allowlist
     3. `normalizeURL(url)` — removes trailing slashes, deduplicates slashes
   - `helpers/validators.ts` — `urlAllowListValidator` used during connector config validation

5. **OAuth token requests**: `requestOAuthToken()` also goes through the same `request()` function with `ensureUriAllowed` — no bypass.

**Key SSRF bypass angles to check:**
- DNS rebinding (TOCTOU between validation and request) — considered but well-known
- URL parser differential (Node `url.parse()` in validation vs axios URL in request) — not found
- `new URL()` vs `url.parse()` parsing differences — `isUriAllowed` uses `url.parse()`
- `proxyRequest` in Console plugin uses direct `http.request()` — different code path

### Console/Dev Tools Proxy (`src/platform/plugins/shared/console/server/routes/api/console/proxy/`)

**SSRF Protection:**
- Host must match configured Elasticsearch hosts exactly
- `stripCredentialsFromUrl(h) === stripCredentialsFromUrl(requestHost)` — credentials stripped before comparison
- `toURL(host, path)` constructs URL: `new URL(${trimEnd(base, '/')}/${trimStart(path, '/')})`
- Path traversal in `path` parameter does NOT affect the target host
- If `requestHost` is missing, defaults to `hosts[0]`
- If `requestHost` doesn't match any configured host, returns 400

**Potential bypass:**
- `stripCredentialsFromUrl` returns the raw string if `new URL()` fails — if configured ES hosts contain non-parseable URLs, the comparison could behave unexpectedly. (Unlikely in practice.)

### Saved Objects Import (`src/core/packages/saved-objects/import-export-server-internal/src/import/`)

**Pipeline:**
1. Route handler (`routes/import.ts`) — validates `.ndjson` file extension, calls `createSavedObjectsStreamFromNdJson`
2. `createSavedObjectsStreamFromNdJson` (routes/utils.ts) — splits on `\n`, `JSON.parse` each line, filters export details
3. `collectSavedObjects` (lib/collect_saved_objects.ts) — stream pipeline: limit → filter → map (adds migration version, manages references) → collect
4. Import processing: `checkReferenceOrigins` → `validateReferences` → `checkConflicts` → `checkOriginConflicts` → `regenerateIds` → `createSavedObjects`
5. `executeImportHooks` — runs per-plugin import hooks after objects are created

**Security assessment:** The NDJSON parsing is safe (standard `JSON.parse`, no eval). No path traversal in the import pipeline. Import hooks are plugin-specific and worth checking individually.

## Attack Vector Priority

| Priority | Vector | Best for cloud? | Notes |
|----------|--------|----------------|-------|
| 1 | **Actions/Connectors SSRF** | ❌ (restricted) | Default `'*'` on self-managed, restricted on cloud |
| 2 | **Saved Objects Import** | ❌ (safe) | Clean pipeline, no injection vectors |
| 3 | **Console Proxy** | ❌ (restricted) | Host whitelist against configured ES hosts |
| 4 | **XSS with CSP bypass** | Maybe | `script-src 'self'` needs same-origin JS injection |
| 5 | **Auth/IDOR in newer features** | ✅ | AI chat, workflows, connectors — less audited |
| 6 | **Custom Integration connectors** | ✅ | Newer connector types may have weaker validation |

## Live Testing Workflow

```bash
# 1. Check the actual CSP header on cloud.elastic.co
curl -s -I https://<instance>.kibana.cloud.elastic.co/ | grep -i content-security-policy

# 2. Test Connector endpoint restrictions
# Create a webhook connector pointing to webhook.site — does it work?

# 3. Check Console proxy host restrictions
# Can you send ES requests to arbitrary hosts?

# 4. Check for IDOR in saved objects API
# Try accessing other users' saved objects
```

## Source Code Files to Watch

When looking for vulnerabilities in the Kibana codebase:

1. **SSRF**: Files that make outbound HTTP requests:
   - `x-pack/platform/plugins/shared/actions/server/lib/axios_utils.ts` — core request function
   - `x-pack/platform/plugins/shared/actions/server/actions_config.ts` — allowed host validation
   - `x-pack/platform/plugins/shared/actions/server/sub_action_framework/helpers/validators.ts` — URL validation
   - `src/platform/plugins/shared/console/server/lib/proxy_request.ts` — Console proxy
   - `src/platform/plugins/shared/console/server/lib/utils/to_url.ts` — URL construction

2. **IDOR**: Files with unscoped `find()` or `get()`:
   - `src/core/packages/saved-objects/api-server-internal/src/lib/apis/get.ts` — saved objects API
   - Plugin-specific saved objects routes

3. **XSS**: Files that render user content:
   - Canvas workpad renderers: `x-pack/platform/plugins/private/canvas/canvas_plugin_src/renderers/`
   - Markdown rendering
   - Dashboard/visualization titles and descriptions

4. **CSP**: 
   - `src/core/packages/http/server-internal/src/csp/config.ts` — CSP config schema
   - `src/core/packages/http/server-internal/src/csp/csp_directives.ts` — CSP directives construction
   - `src/core/packages/http/server-internal/src/csp/csp_config.ts` — CSP config class

## Known Security Boundaries

- **External URL Policy**: Browser-side URL validation in `external_url` package, hashes hosts with SHA256. Only affects browser, not server-side connector requests.
- **AllowedHosts**: Server-side host allowlist for connector outbound requests. Defaults to `'*'`.
- **CSP**: `script-src 'self'` (no nonces), `style-src 'self' 'unsafe-inline'`.
- **Console proxy**: Host matched against configured ES hosts via `stripCredentialsFromUrl`.
