# Workflows Engine Dual-Allowlist Risk (Kibana)

## The Gap

Kibana has **two separate `allowedHosts` configuration keys** that govern outbound HTTP requests:

| Config Key | Plugin | Location | Default |
|------------|--------|----------|---------|
| `xpack.actions.allowedHosts` | Actions plugin | `x-pack/platform/plugins/shared/actions/server/actions_config.ts` | `['*']` |
| `workflowsExecutionEngine.http.allowedHosts` | Workflows engine | `src/platform/plugins/shared/workflows_execution_engine/server/config.ts` | `['*']` |

An admin who configures one but not the other leaves the unconfigured one wide open.

## Bypass Risk

If an admin restricts `xpack.actions.allowedHosts` to `['example.com']` but leaves `workflowsExecutionEngine.http.allowedHosts` at `['*']`:

1. Workflow validates the URL with `UrlValidator` (workflows config) — passes
2. But the actual HTTP request goes through `request()` in `axios_utils.ts` (actions plugin)
3. Actions plugin restricts to `example.com` only → **request blocked**

This direction is safe. The **dangerous** direction:

1. Admin restricts `workflowsExecutionEngine.http.allowedHosts` to `['example.com']`
2. But leaves `xpack.actions.allowedHosts` at `['*']`
3. Workflow validates with `UrlValidator` — checks against workflows config → passes for `example.com`
4. Actual request goes through `request()` — checks against actions config → `['*']` = any URL allowed
5. **Template injection in `params.url`** can override the connector's target URL at execution time

## Source Code Evidence

**`http_connector.ts` line 306:**
```typescript
const baseUrl = params.url || config.url;
```
`params.url` takes precedence over `config.url` at execution time.

**Line 160-162:**
```typescript
const variables = extractVariables(step, context);
const renderedParams = renderTemplate(params, variables);
```
Mustache templates render workflow context variables into `params.url`.

**`url_validator.ts`:**
The UrlValidator only checks against the Workflows engine's own `allowedHosts` config — not the Actions plugin's.

## Impact

- An attacker who can create or modify workflows can bypass URL restrictions
- Template injection in workflow variables can override the target URL
- Even with proper configuration of ONE allowlist, the other remains a bypass vector
- Default `['*']` on both means no protection anyway on most deployments

## Verification

On cloud.elastic.co (both defaults at `['*']`), the `.http` connector with URL override successfully targets arbitrary internal IPs. On a restricted deployment, test:
1. Configure `xpack.actions.allowedHosts` to restrict a domain
2. Create a workflow with `.http` connector targeting that domain
3. Modify workflow step to override `params.url` to an internal IP
4. Observe whether the execution-time allowlist check blocks or allows it
