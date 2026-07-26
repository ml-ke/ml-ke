# Kibana Codebase Map (v9+)

Directory structure discovered during codebase audit (May 2026).

## Directory Layout Changes

**OLD** (pre-v9): `x-pack/plugins/`, `src/core/`, packages in `packages/`
**NEW** (v9+): `x-pack/platform/plugins/shared/`, `x-pack/solutions/`, `src/core/packages/`, `src/platform/plugins/shared/`

## Key File Paths by Functional Area

### CSP Configuration
```
src/core/packages/http/server-internal/src/csp/
├── config.ts              # Schema: strict=true default, allowNonce=false everywhere
├── csp_config.ts          # CspConfig class — reads config, builds header
└── csp_directives.ts      # CspDirectives — default rules, report-only rules, parsing
```

**Key finding**: `strict: true` does NOT add nonce or `strict-dynamic`. It only validates that `unsafe-inline` is not used with `script-src`.
Default CSP: `script-src 'report-sample' 'self'` with NO nonce mechanism.

### Actions/Connectors Framework
```
x-pack/platform/plugins/shared/actions/server/
├── actions_config.ts                         # isHostnameAllowed, isUriAllowed, getBeforeRedirectFn
├── config.ts                                 # AllowedHosts.Any = '*', default enabledActionTypes = '*'
├── sub_action_framework/
│   ├── executor.ts                           # Calls service[method]() with schema validation
│   ├── sub_action_connector.ts               # Base class: assertURL → ensureUriAllowed → normalizeURL → request()
│   └── helpers/validators.ts                 # assertURL (http/https only), urlAllowListValidator
├── lib/
│   ├── axios_utils.ts                        # request() — core HTTP function with ensureUriAllowed + getBeforeRedirect
│   ├── before_redirect.ts                    # getBeforeRedirectFn — validates redirect hostname
│   ├── get_custom_agents.ts                  # Proxy/SSL agent creation
│   └── request_oauth_token.ts                # OAuth token requests — also goes through request() with ensureUriAllowed
```

**Key finding**: `allowedHosts` defaults to `'*'`. On self-managed Kibana with default config, ALL hosts are allowed.
URL validation uses `url.parse()` (legacy Node API), not `new URL()`.

### Stack Connectors (Webhooks, etc.)
```
x-pack/platform/plugins/shared/stack_connectors/server/
└── connector_types/
    └── webhook/
        ├── index.ts                          # getConnectorType — config schema, executor
        ├── get_axios_config.ts               # getAxiosConfig — creates axios instance, handles OAuth2
        ├── validations.ts                    # validateConnectorTypeConfig
        └── types.ts                          # ConnectorTypeConfigType, ConnectorTypeSecretsType
```

**Key finding**: Webhook URL (`config.url`) is user-provided. Flows directly to `request()` which validates via `ensureUriAllowed`.
OAuth2 token URL (`config.accessTokenUrl`) is also user-provided — also goes through `request()`.

### Console / Dev Tools Proxy
```
src/platform/plugins/shared/console/server/
├── routes/api/console/proxy/
│   ├── create_handler.ts                     # Handler: validates host against configured ES hosts
│   ├── validation_config.ts                  # Query schema: method, path, withProductOrigin, host
│   └── index.ts                              # Route registration
└── lib/
    ├── proxy_request.ts                      # proxyRequest — direct http.request() / https.request()
    ├── elasticsearch_proxy_config.ts          # getElasticsearchProxyConfig — agent creation
    └── utils/
        ├── to_url.ts                         # toURL(base, path) — URL construction
        ├── sanitize_hostname.ts              # Strip IPv6 brackets
        └── strip_credentials_from_url.ts     # stripCredentialsFromUrl — for host comparison
```

**SSRF protection**: Host must match configured ES hosts. `stripCredentialsFromUrl(h) === stripCredentialsFromUrl(requestHost)`.
Path appended to base URL via `new URL()` — path traversal does NOT change the target host.

### Saved Objects Import Pipeline
```
src/core/packages/saved-objects/import-export-server-internal/src/import/
├── import_saved_objects.ts                  # importSavedObjectsFromStream — orchestrator
├── saved_objects_importer.ts                # Importer class wrapper
├── lib/
│   ├── collect_saved_objects.ts             # Stream pipeline: limit → filter → map → concat
│   ├── create_saved_objects.ts              # Bulk create in ES
│   ├── check_conflicts.ts                   # Conflict detection
│   ├── check_origin_conflicts.ts            # Multi-namespace conflict check
│   ├── check_reference_origins.ts           # External reference resolution
│   ├── validate_references.ts               # Reference validation
│   ├── regenerate_ids.ts                    # ID regeneration for createNewCopies
│   └── execute_import_hooks.ts              # Per-plugin import hooks

# Route handler
src/core/packages/saved-objects/server-internal/src/routes/
├── import.ts                                # POST /_import — file validation, NDJSON parsing
└── utils.ts                                 # createSavedObjectsStreamFromNdJson
```

**NDJSON parsing** (utils.ts lines 30-45):
```
split on '\n' → JSON.parse each line → filter export details → concat array → createListStream
```
Standard `JSON.parse`, no eval, no dynamic includes.

## CSP Bypass Notes

Default CSP (`script-src 'self'`):
- **No nonces** — can't inject `<script>alert(1)</script>` with a nonce anyway
- **No `unsafe-inline`** — inline scripts are blocked
- **`strict-dynamic` not present** — can't chain through a trusted script
- **`style-src 'unsafe-inline'`** — CSS injection (but limited — can't inject `javascript:` URLs)
- **`report-sample` present** — violation reports include sample of violating content

To achieve XSS with this CSP, you need:
1. **Same-origin JS file upload**: If you can upload a .js file to Kibana's server, `script-src 'self'` allows executing it
2. **JSONP/gadget abuse**: Find existing endpoints on the same origin that generate executable JS
3. **File upload in Canvas workpads or Custom Integrations**

## SSRF Protection Summary

| Component | Validation Method | Bypass Risk |
|-----------|------------------|-------------|
| Actions `request()` | `ensureUriAllowed` → `url.parse()` hostname check | Low (parser differential unlikely) |
| Actions redirect | `getBeforeRedirectFn` → `ensureHostnameAllowed` | Low (strict hostname check) |
| Actions sub-action | `assertURL` + `ensureUriAllowed` + `normalizeURL` | Low (chained validation) |
| Console proxy | Host match via `stripCredentialsFromUrl` | Very low (exact string match) |
| OAuth token | Goes through same `request()` as connectors | Low (same validation path) |
