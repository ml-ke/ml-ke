# Atlassian MCP Auth Bypass — Full Post-Mortem (Jun 2026)

## Finding Summary

The Atlassian MCP Server at `mcp.atlassian.com/v1/mcp` accepted **any non-empty Bearer token**. 
The server checked only for presence of `Authorization: Bearer <non-empty>` — zero validation.
This was confirmed with tokens "x", "null", path-traversal strings — all returned 200.

**Status**: Rejected as Not Applicable by Bugcrowd triage.

## What Was Found (Proven)

| Test | Result | Evidence |
|------|--------|----------|
| No Authorization header | 401 (auth check exists) | Confirmed |
| Bearer "x" (single char) | 200 (bypass confirmed) | Confirmed |
| Bearer "null" (literal string) | 200 (bypass confirmed) | Confirmed |
| Basic auth (arbitrary) | 200 (bypass also works in Basic) | Confirmed |
| Initialize session | 200, session returned | Confirmed |
| tools/list | 200, 2 tools returned | Confirmed |
| getTeamworkGraphContext | 200, permission error (not 401) | Confirmed — reaches authz layer |
| getTeamworkGraphObject | 200, "Failed to fetch cloud ID" | Confirmed — needs valid tenant |
| resources/list | "Method not found" | Not implemented |
| resources/read | "Method not found" | Not implemented |
| notifications/initialized | "Method not found" | Not implemented |
| logging/setLevel | `{}` | Exists, no attack value |

## What Was Missing (Why Rejected)

| What triage needed | What we had | The gap |
|-------------------|-------------|---------|
| Protected data returned | Permission error only | Couldn't show actual Teamwork Graph data |
| Successful tool execution | `isError: true` response | Tool returned error, not data |
| Client credentials usable | "failed to retrieve client" from OAuth | Registration creds don't chain to auth.atlassian.com |

## Schema Discovery via Error Messages

Validation errors from `getTeamworkGraphContext` with empty args revealed **33 object types**:

- AtlassianGoal, AtlassianGoalUpdate, AtlassianHomeComment, AtlassianHomeTag, AtlassianProject, AtlassianProjectUpdate, AtlassianTeam, AtlassianUser
- CompassComponent
- ConfluenceBlogPost, ConfluenceComment, ConfluenceDatabase, ConfluencePage, ConfluenceSpace, ConfluenceWhiteboard
- ExternalCalendarEvent, ExternalConversation, ExternalDeployment, ExternalDesign, ExternalDocument, ExternalOrganisation, ExternalPosition, ExternalPullRequest, ExternalRepository, ExternalService
- FocusFocusArea
- JiraSpace, JiraSprint, JiraVersion, JiraWorkItem, JiraWorkItemComment
- LoomMeeting, LoomMeetingRecurrence, LoomVideo

This confirms the tool spans the **entire Atlassian ecosystem** — Jira, Confluence, Compass, Loom, Bitbucket, external integrations.

## OAuth Chaining — Dead End

Registered client credentials (`/v1/register`):
- `auth.atlassian.com/oauth/token` → `"failed to retrieve client"` — NOT linked
- `id.atlassian.com/oauth/token` → empty response — NOT linked
- Client_id as Bearer token on MCP → HTTP 400 (format error)
- Client_secret as Bearer token on MCP → HTTP 400 (format error)
- Basic auth (client_id:client_secret) on MCP → HTTP 400 (format error)

**Verdict**: `/v1/register` issues MCP-service-scoped credentials only. They do NOT provide access to Atlassian's broader OAuth infrastructure.

## Auth Method Testing

| Method | Result |
|--------|--------|
| Bearer "x" (any non-empty) | 200 OK |
| Bearer with trailing space only | 401 |
| Bearer without space after keyword | 401 |
| Basic auth (arbitrary base64) | 200 OK (same bypass) |
| X-API-Key header | 401 |
| No auth header | 401 |

Only `Authorization: Bearer <anything>` and `Authorization: Basic <anything>` pass.
The parser strips the scheme prefix and only checks if the remainder is non-empty.

## The Server-Side Architecture

The server has three distinct layers:
1. **Transport/HTTP** — Cloudflare + AtlassianEdge (Atl-Paas)
2. **Authentication** — Checks for non-empty Authorization header. BROKEN.
3. **Authorization** — Validates cloudId is a real domain, then checks API token permissions against that tenant's org. INTACT (still blocked us).

The cloudId validation accepts domain-like strings (e.g., `test.atlassian.net`) but rejects anything that "does not look like a valid domain or URL" (e.g., `bugbounty-test-h0d4r1` without TLD).

## Key Lessons

1. **Auth bypass without data access = Informative/P4.** The mechanism is real, but triage values impact over mechanism. Without demonstrating actual data returned through the bypass, the finding is "endpoint reachability" not "authentication bypass."

2. **The permission error PROVES the auth bypass works** (auth accepted → authz evaluated) but triage sees it as FAILURE not PROOF. Never rely on the triager making this inference — show them real data.

3. **Always test OAuth chaining before claiming it.** Unauthenticated client registration endpoints may issue service-scoped credentials that don't work outside the service. One curl command to `auth.target.com/oauth/token` kills the chain.

4. **The cloud instance is the unlock.** For any finding that needs a valid tenant to demonstrate impact (MCP tools, cross-instance leakage), the cloud instance setup is an essential prerequisite — not optional.

5. **Error messages are recon gold.** Input validation errors from tool calls reveal the full parameter schema including enum values, field types, and required/optional fields. Always collect these before moving on.

## Phase 2: Testing with Real Tenant (Jun 11, 2026)

After the rejection, we set up a Bugcrowd cloud instance (`h0d4r1testerbb.atlassian.net`) and obtained a JIRA API token + admin session JWT to test end-to-end.

### Cloud ID Discovery

The cloud ID is required for MCP tool calls. Found in the JIRA Dashboard HTML:

```bash
curl -s 'https://instance.atlassian.net/secure/Dashboard.jspa' \
  -H 'Authorization: Basic <base64(email:token)>' \
  | grep -oP 'cloudId=[a-f0-9-]+' | head -1
# cloudId=ac343abe-2ea8-487e-89ef-6ecbbecd32ba
```

Also appears in JS bundles as `cloudId\\":\\"<UUID>\\"`. The admin portal URL `https://admin.atlassian.com/o/<orgId>/overview` contains the org ID (a UUID) which is different from the cloud ID.

### The "Allow API Token Authentication" Setting

In the admin portal at `https://admin.atlassian.com/o/<orgId>/products`, there is a setting for the MCP/Rovo config that fundamentally changes auth behaviour:

| Setting | Fake token "x" | JIRA API token | Session JWT |
|---------|---------------|----------------|-------------|
| OFF (default) | Init success, tool call returns "no permission" | Same as "x" | Same as "x" |
| ON | "invalid access token" | Init success, TWG says "not a compact JWS" | "invalid access token" |

The setting does NOT fix the auth bypass — it switches from "accept any token" to "validate against real auth." When ON, JIRA API tokens (`ATATT3xFfGF0...`) pass MCP init but fail at the upstream TWG backend because they are opaque tokens, not JWTs.

### TWG Backend JWT Requirement

Error message when setting is ON and using JIRA API token:

```
TWG request failed: 401 Unauthorized. build: failed to parse token not a compact JWS
```

The MCP server forwards the Bearer token to Teamwork Graph. TWG tries to parse it as compact JWS (JWT format). Opaque API tokens fail. Only tokens that are actual JWTs can reach data.

### Admin Session JWT Domain Scoping

The `cloud.session.token` from `admin.atlassian.com` IS a JWT but contains `domains: [admin.atlassian.com]` in its payload. The MCP server at `mcp.atlassian.com` rejects it because it was issued for a different subdomain. Session JWTs are not portable across Atlassian subdomains.

### Token Type Compatibility

| Token Type | Format | JIRA API Works? | MCP Init Works? | TWG Works? |
|-----------|--------|----------------|-----------------|-----------|
| JIRA API token | Opaque (`ATATT3x...`) | Yes | Yes | No (not JWT) |
| Admin session JWT | JWT (header.payload.sig) | No (wrong domain) | No (setting ON) | Would need correct domain |
| Fake "x" | Plain text | N/A | Yes (setting OFF) | No (permission denied) |

### Summary

The "Allow API token" setting toggles between:
- OFF (default): No real auth validation — auth bypass exists for all non-empty tokens.
- ON: Real auth validation enabled — but only JWT-format tokens work through to TWG.

The default (OFF) state means every new org starts with a broken auth gate. The bypass is structural, not a runtime configuration leak.

## Target Info

- **URL**: `https://mcp.atlassian.com/v1/mcp` (Tier 1 — Rovo / AI Features)
- **Server**: Atlassian MCP Server v1.0.0 (protocol 2024-11-05)
- **Infra**: Cloudflare + AtlassianEdge on AWS ap-south-1
- **Known issues**: 0 (at time of testing)
- **Status**: Still live, auth bypass likely unpatched
