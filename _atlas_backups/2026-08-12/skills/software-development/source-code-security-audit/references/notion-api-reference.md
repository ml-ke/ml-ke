# Notion API — Bug Bounty Testing Reference

Gathered from: `developers.notion.com`, API documentation, and live API testing.
Date: 2026-05-29

## API Basics

| Property | Value |
|----------|-------|
| **Base URL** | `https://api.notion.com` |
| **Auth** | Bearer token in `Authorization` header (`Bearer ntn_...`) |
| **Version header** | `Notion-Version: 2026-03-11` (latest) |
| **Content-Type** | `application/json` |
| **Rate limit** | Avg 3 req/s (burst), 2700 req / 15 min window |
| **Error format** | `{"object":"error","status":NNN,"code":"error_code","message":"..."}` |
| **PAT (Personal Access Token)** | Created in Settings → My connections → Personal access tokens. Only accesses pages shared with the integration. |
| **Page IDs** | UUIDv4 format. Dashes optional in API calls. Extract from URL: `https://www.notion.so/Title-32e9ba9fc9ba4d8f...` |

## Key Endpoints

### Data APIs
| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/pages/{id}` | Retrieve page properties |
| PATCH | `/v1/pages/{id}` | Update page properties |
| GET | `/v1/blocks/{id}/children` | Get block children (page content) |
| PATCH | `/v1/blocks/{id}/children` | Append block children |
| DELETE | `/v1/blocks/{id}` | Delete a block |
| POST | `/v1/databases` | Create a database |
| GET | `/v1/databases/{id}` | Retrieve database |
| POST | `/v1/databases/{id}/query` | Query database entries |
| POST | `/v1/comments` | Create a comment |
| GET | `/v1/comments?block_id={id}` | List comments |
| POST | `/v1/search` | Search across workspace |
| POST | `/v1/files` | Initiate file upload |

### User & Auth APIs
| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/users/me` | Get bot/user info |
| GET | `/v1/users` | List workspace users |
| GET | `/v1/users/{id}` | Get specific user |
| POST | `/v1/oauth/token` | Exchange OAuth code for token |
| POST | `/v1/oauth/token/introspect` | Token introspection (endpoint may differ) |

### File Uploads
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/files` | Initiate upload (get upload URL) |
| POST | `{upload_url}` | Upload file data |
| POST | `/v1/files/{id}/complete` | Complete multi-part upload |
| GET | `/v1/files` | List file uploads |

## Auth & Permissions Model

- **PAT (Personal Access Token)**: Internal integration. Only accesses pages **explicitly shared** with the integration via Share → Add connections.
- **OAuth (Public integration)**: User-authorized. Can access workspace content with user's permissions.
- **Share link sharing** ("Anyone with the link"): Does NOT grant API access. Only human users with the link can view/edit. The API token needs explicit integration sharing even if the page is publicly shared.
- **No workspace-level page creation**: PATs cannot create pages at workspace root (`parent.workspace: true`) — must have a parent `page_id` or `database_id`.

## Interesting Attack Vectors to Test

1. **User enumeration**: `GET /v1/users` returns all workspace users with names, emails, and avatar URLs.
2. **IDOR on pages**: Try accessing pages across workspaces by guessing/changing page UUIDs in `GET /v1/pages/{id}`.
3. **IDOR on users**: `GET /v1/users/{id}` returns user info — test if other workspace members' data is accessible.
4. **Integration scope bypass**: Test if PAT can access resources beyond what was shared.
5. **File upload SSRF**: The file upload flow involves the API returning an upload URL that gets fetched. Test for redirect following to internal IPs.
6. **Rate limit bypass**: 3 req/s average. Test if race conditions exist in page creation/updates.
7. **Mass assignment**: Extra fields in `PATCH /v1/pages/{id}` properties.
8. **Search scope**: `POST /v1/search` — test if it returns pages not shared with the integration (should return empty, but worth verifying).
