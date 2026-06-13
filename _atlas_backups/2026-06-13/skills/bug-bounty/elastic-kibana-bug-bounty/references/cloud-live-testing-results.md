# cloud.elastic.co Live Testing Results (May 2026)

## Deployment Info
- **Provider**: Elastic Cloud Hosted (GCP us-central1)
- **Kibana Version**: 9.4.2 (traditional, build 101974, build_hash ca34dedcfe61)
- **GitLab.com Version**: 19.1.0-pre (enterprise)
- **Elastic Cloud URL**: `https://8fd3b6ec08d6420a87f41592ac01f3bd.us-central1.gcp.cloud.es.io`
- **API Key**: Created from Kibana UI (Stack Management → API Keys)

## Confirmed Findings

### SSRF via Webhook Connector (HIGH)
- Created webhook to `http://127.0.0.1:9200/` → **accepted** (no host restriction)
- Created webhook to `http://10.0.0.1:9200/` → **accepted**
- Created webhook to `http://169.254.169.254/` → **accepted**
- Created `.http` (Workflows) connector with URL override at execution time → **accepted**

### SSRF via Vega Same-Origin Data Source (MEDIUM)
- `../api/status` → loaded successfully (shows version 9.4.2)
- `../api/features` → loaded successfully
- External URLs blocked (`enableExternalUrls: false`)

### CSP Confirmed Strict (INFO)
- `script-src 'report-sample' 'self'` — no `unsafe-eval`
- `style-src 'report-sample' 'self' 'unsafe-inline'`
- No nonces, no `strict-dynamic`

### Internal Network Topology (cloud.elastic.co)
- No services on localhost (ES/Kibana in separate containers)
- No K8s DNS for internal services
- GCP metadata endpoint `169.254.169.254` → EHOSTUNREACH (GKE blocks metadata)
- `10.x.x.x` subnet routable but port-filtered
- `172.x.x.x` subnet routable but port-filtered

## Connector Types Available on cloud.elastic.co
.webhook, .email, .index, .pagerduty, .swimlane, .server-log, .slack, .slack_api, .http, .cases-webhook, .xmatters, .servicenow, .servicenow-sir, .servicenow-itom, .jira, .teams, .torq, .opsgenie, .jira-service-management, .tines, .gen-ai, .bedrock, .gemini, .d3security, .resilient, .thehive, .xsoar, .mcp, .sentinelone, .crowdstrike, .inference, .microsoft_defender_endpoint, .abuseipdb, .alienvault-otx, .jira-cloud, .confluence-cloud, .aws_lambda, .brave-search, .figma, .github (+ many more)

## Report Filed
- **Elastic/001**: SSRF via Webhook and HTTP Connectors (CVSS 8.6 High)
- Submitted to HackerOne May 31, 2026
