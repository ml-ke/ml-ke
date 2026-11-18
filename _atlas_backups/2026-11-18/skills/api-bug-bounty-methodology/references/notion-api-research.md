# Notion API Research — Session Notes

## Documentation Source
Notion provides a full documentation index: https://developers.notion.com/llms.txt
This file lists every available documentation page.

## Discovery from llms.txt

### Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/v1/oauth/token` | Create OAuth access token |
| GET | `/v1/oauth/token/{id}/introspect` | Introspect token |
| GET | `/v1/users` | List all users |
| GET | `/v1/users/{id}` | Retrieve user |
| POST | `/v1/search` | Search across workspace |
| GET | `/v1/pages/{id}` | Retrieve page |
| PATCH | `/v1/pages/{id}` | Update page properties |
| PATCH | `/v1/pages/{id}/properties/{property_id}` | Update page property |
| GET | `/v1/pages/{id}/properties/{property_id}` | Retrieve page property item |
| POST | `/v1/databases` | Create database |
| GET | `/v1/databases/{id}` | Retrieve database |
| PATCH | `/v1/databases/{id}` | Update database |
| POST | `/v1/databases/{id}/query` | Query database |
| GET | `/v1/databases/{id}/query/filter` | Filter database entries |
| PATCH | `/v1/blocks/{id}/children` | Append block children |
| GET | `/v1/blocks/{id}/children` | Retrieve block children |
| DELETE | `/v1/blocks/{id}` | Delete block |
| PATCH | `/v1/blocks/{id}` | Update block |
| POST | `/v1/comments` | Create comment |
| GET | `/v1/comments` | List comments |
| DELETE | `/v1/comments/{id}` | Delete comment |
| GET | `/v1/comments/{id}` | Retrieve comment |
| POST | `/v1/files` | Create file upload |
| POST | `/v1/files/{id}/complete` | Complete multi-part upload |
| GET | `/v1/files` | List file uploads |
| GET | `/v1/files/{id}` | Retrieve file upload |
| POST | `/v1/views` | Create view |
| GET | `/v1/views` | List views |
| GET | `/v1/views/{id}` | Retrieve view |
| PATCH | `/v1/views/{id}` | Update view |
| DELETE | `/v1/views/{id}` | Delete view |
| GET | `/v1/views/{id}/query/results` | Get view query results |
| POST | `/v1/views/{id}/query` | Create view query |
| DELETE | `/v1/views/{id}/query` | Delete view query |
| POST | `/v1/data_sources` | Create data source |
| GET | `/v1/data_sources/{id}` | Retrieve data source |
| POST | `/v1/data_sources/{id}/query` | Query data source |
| GET | `/v1/data_sources/{id}/templates` | List data source templates |
| POST | `/v1/data_sources/{id}/templates/{id}/pages` | Create page from template |
| POST | `/v1/data_sources/{id}/entries/filter` | Filter data source entries |
| POST | `/v1/custom_emojis` | Upload custom emoji |
| GET | `/v1/custom_emojis` | List custom emojis |

### API Conventions
- **Base URL**: `https://api.notion.com`
- **Auth**: Bearer token in Authorization header
- **Version header**: `Notion-Version` (current: `2026-03-11`)
- **Rate limit**: ~3 requests/second average
- **Pagination**: Cursor-based via `start_cursor` / `next_cursor`
- **IDs**: UUIDv4, dashes are optional

### Notable Features for Testing
1. **File upload** — Supports multi-part uploads, external URL imports
2. **Markdown content API** — Create/read/update page content using markdown
3. **Views** — Database views with saved queries
4. **MCP** — Model Context Protocol integration (AI agent access)
5. **Webhooks** — Event-driven notifications
6. **Link previews** — URL unfurling (SSRF vector?)
7. **Compliance/SIEM** — Audit log events via webhooks

### Initial Attack Surface Priorities
1. **File Upload SSRF** — External URL import, multi-part upload
2. **IDOR** — Pages, databases, blocks, comments, views
3. **Mass Assignment** — PATCH endpoints with extra properties
4. **Search abuse** — POST /v1/search for data discovery
5. **User enumeration** — GET /v1/users, GET /v1/users/{id}
