# GitLab BBP — Program Intelligence

Applied the Step 0 Program Intelligence workflow to `hackerone.com/gitlab`.

## Program Meta

- **Bounty launched**: 2018 (mature, well-funded)
- **Response efficiency**: >90% (Top Response Efficiency badge)
- **Gold Standard Safe Harbor**: ✅
- **Managed by HackerOne**: ✅
- **Collaboration enabled**: ✅
- **Includes Retesting**: ✅
- **First response**: 6 hours average
- **Time to bounty**: 1 month 1 week average
- **Time to resolution**: 1 month 4 weeks average
- **Platform deviation**: Vulnerabilities requiring an account are NOT scored with "Privilege Required: None" because most GitLab instances don't allow self-registration

## Bounty Ranges

| Severity | Range | Avg Bounty | % Submissions |
|----------|-------|-----------|---------------|
| Low | $100 – $750 | $625 | 27.8% |
| Medium | $1,000 – $2,500 | $2,081 | 46.26% |
| High | $5,000 – $15,000 | $8,245 | 20.9% |
| Critical | $20,000 – $35,000 | $11,250 | 5.04% |

**Bonus payments**: $1,000 at triage for Critical/High, $500 for Medium. Remainder paid after severity analysis. Security documentation updates = $100.

**Ultimate License**: 3+ valid reports → 1-year self-hosted Ultimate license (5 users). Extends with each additional valid submission.

**Reward calculator**: Publicly accessible — [CVSS-based bounty calculator](https://gitlab-com.gitlab.io/crn-team/developer-experience/bounty-calculator/)

## Scope — In Scope Assets

### Critical Severity (bounty-eligible)

| Asset | Type | Notes |
|-------|------|-------|
| `gitlab.com` | Domain | Main SaaS. Free account available. |
| `registry.gitlab.com` | Domain | Container registry |
| `customers.gitlab.com` | Domain | Customer portal |
| `gitlab-org/gitlab` | Source code | Main Rails app — **49k Ruby files** |
| `gitlab-org/gitlab-shell` | Source code | Go SSH/HTTP client |
| `gitlab-org/gitlab-runner` | Source code | Go CI runner |
| `gitlab-org/gitlab-pages` | Source code | Go static page hosting |
| `gitlab-org/gitaly` | Source code | Go Git RPC server |
| `gitlab-org/gitlab-vscode-extension` | Source code | VS Code extension |
| `Your Own GitLab Instance` | Other | Self-hosted — run locally! |

### Medium Severity (bounty-eligible)

| Asset | Type | Notes |
|-------|------|-------|
| `docs.gitlab.com` | Domain | Documentation |
| `about.gitlab.com` | Domain | Marketing site |
| `design.gitlab.com` | Domain | Design system |
| `advisories.gitlab.com` | Domain | Security advisories |
| `*.gitlap.com` | Wildcard | Typo-squat protection domain |
| `*.gitlab.org` | Wildcard | |
| `*.gitlab.net` | Wildcard | |
| GitLab for Jira Cloud | Other | Integration |
| Non-production infrastructure | Other | (medium max) |

### Out of Scope (no bounty)

`us-federal-gitlab.com`, `support.gitlab.com`, `status.gitlab.com`, `shop.gitlab.com`, `partners.gitlab.com`, `packages.gitlab.com`, `translate.gitlab.com`, `levelup.gitlab.com`, `ir.gitlab.com`, `forum.gitlab.com`, `*.gitlab.cn`, `*.gitlab-private.org`, `*.service-now.com`, `*.runway.gitlab.net`, `opstrace/*`, `gitlab-org/cli/`

## Codebase Overview

### Main Repo: `gitlab-org/gitlab`

- **Language**: Ruby on Rails (49k .rb files)
- **Default branch**: `master`
- **Total size**: ~1.2GB / 99k files
- **AGENTS.md**: Present at root — follow .ai/* instruction files per task type
- **Key directories**:

| Directory | Contents |
|-----------|----------|
| `app/` | Rails application (controllers, models, views, helpers) |
| `lib/` | Library code including API modules |
| `config/` | Routes, initializers, application config |
| `db/` | Database migrations and schema |
| `spec/` | RSpec tests |
| `ee/` | Enterprise Edition features |
| `workhorse/` | Go reverse proxy (separate AGENTS.md) |
| `qa/` | End-to-end test suite |
| `doc/` | Documentation |

### Workhorse (`workhorse/`) — Reverse Proxy

- **Language**: Go
- **Purpose**: Smart reverse proxy sitting in front of Puma. Intercepts HTTP for file uploads, downloads, Git operations, artifact processing.
- **AGENTS.md**: Present in `workhorse/`
- **Key packages**:

| Package | Function | Attack Surface |
|---------|----------|---------------|
| `internal/download/` | File download handling | Path traversal? |
| `internal/upload/` | File upload handling | SSRF via URL upload? |
| `internal/sendurl/` | URL-based response sending | **SSRF vector** — see below |
| `internal/sendfile/` | File sending | |
| `internal/senddata/` | Response data injection | |
| `internal/proxy/` | HTTP proxy logic | |
| `internal/gitaly/` | Gitaly integration | |
| `internal/artifacts/` | Build artifacts | |
| `internal/dependencyproxy/` | Dependency proxy | SSRF if URL fetching |
| `internal/ai_assist/duoworkflow/` | AI feature proxy | Proxies to Duo Workflow |

**Workhorse sendurl SSRF hypothesis**: The `sendurl/` package handles URL-based response sending — if it fetches a user-supplied URL without proper validation, it's an SSRF vector.

## Cloned Repositories

All repos cloned to `~/Dev/gitlab/`:

| Repo | Go Files | Size | Key SSRF Surface |
|------|----------|------|-----------------|
| `gitlab` | 315 .go files + 49k .rb | 1.2GB | Webhooks, imports, integrations, avatar URLs, CI/CD |
| `gitlab-shell` | Go | Small | HTTP client to internal API (`/api/v4/internal`) |
| `gitlab-runner` | Go | Medium | CI job execution, cache backends (S3/GCS/Azure) |
| `gitlab-pages` | Go | 3.7MB | Static hosting, redirect validation, ACME |
| `gitaly` | Go | Medium | Git SSH proxy, gRPC operations |

## SSRF Attack Vectors in GitLab (Known History)

GitLab has a **long history of SSRF vulnerabilities**, which means:
1. The attack surface is known to exist (webhooks, imports, integrations, pipeline triggers)
2. New bypass techniques are a valid hunting ground
3. The codebase has patches that can be studied for incomplete fixes

### Known SSRF Patches to Study

Search the git log for SSRF-related fixes to understand what was already found and what might have been missed:
```bash
cd ~/Dev/gitlab/gitlab
git log --all --oneline --grep="SSRF" --grep="ssrf" --grep="url.*valid" --grep="internal.*network" --all-match -20
git log --all --oneline --grep="CVE" --grep="ssrf" --all-match -10
```

### High-Value Import/Webhook URLs

```bash
# Search for URL fetching patterns in Ruby code
grep -rn "HTTParty\|Faraday\.\|RestClient\|Net::HTTP\|open-uri\|URI\.open\|HTTP\.\(get\|post\)" \
  --include="*.rb" app/ lib/ | grep -v spec/ | grep -v vendor | head -30

# Search for URL validation (potential bypasses)
grep -rn "valid_url\|url_valid\|validate.*url\|allowed.*uri\|allowed.*url\|blocked.*url" \
  --include="*.rb" app/ lib/ | grep -v spec/ | grep -v vendor | head -20

# Search for SSRF-specific words
grep -rn "\"ssrf\"\|\"SSRF\"\|ssrf_protection" --include="*.rb" app/ lib/ | head -20
```

## Key Things to Know Before Testing

### Account Setup
- **Free account**: ✅ Available at `gitlab.com` — no KYC, no payment
- **Self-hosted**: ✅ "Your Own GitLab Instance" is explicitly in scope — can run locally with Docker
- **Personal Access Token**: Required for API testing — create at `gitlab.com/-/user_settings/personal_access_tokens`
- **Header**: `X-HackerOne-Research: [username]` should be included on all requests (best practice even if not explicitly required)
- **Two accounts needed**: For IDOR testing — create two free accounts

### API Info
- **Base URL**: `https://gitlab.com/api/v4/`
- **Auth**: `PRIVATE-TOKEN: <token>` header or `Authorization: Bearer <token>`
- **GraphQL**: `https://gitlab.com/api/graphql`
- **Docs**: https://docs.gitlab.com/ee/api/

### Initial Probing Commands
```bash
# After getting a PAT
export GL_TOKEN='your-token-here'

# Test auth
curl -s -H "PRIVATE-TOKEN: $GL_TOKEN" "https://gitlab.com/api/v4/user"

# List projects
curl -s -H "PRIVATE-TOKEN: $GL_TOKEN" "https://gitlab.com/api/v4/projects?per_page=50"

# GraphQL introspection
curl -s -X POST -H "Authorization: Bearer $GL_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"{ __schema { types { name fields { name } } } }"}' \
  "https://gitlab.com/api/graphql" | head -100

# Probing SSRF vectors
# Check webhook creation
curl -s -H "PRIVATE-TOKEN: $GL_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"http://localtest.me:9999/test","push_events":true}' \
  "https://gitlab.com/api/v4/projects/:id/hooks"

# Check import functionality
curl -s -H "PRIVATE-TOKEN: $GL_TOKEN" \
  "https://gitlab.com/api/v4/import?url=http://localtest.me:9999/import"
```

## Workhorse SSRF — sendurl Deep Dive

The `workhorse/internal/sendurl/` package is a promising SSRF target:

```bash
cd ~/Dev/gitlab/gitlab/workhorse
grep -rn "url\." --include="*.go" internal/sendurl/ | head -20
grep -rn "http\.\|fetch\|proxy\|Host\|Dial" --include="*.go" internal/sendurl/ | head -20
```

Look for patterns where the URL is used without validation:
- Is there a hostname/IP allowlist?
- Is DNS resolved before making the request?
- Can we use DNS bypass domains (localtest.me, nip.io)?
- Does it follow redirects without re-validation?

## Reference: Previous SSRF found using this methodology

See `references/ssrf-dns-bypass-techniques.md` in this skill for the full SSRF technique catalog (bypass domains, IP formats, redirect chaining) that applies directly to GitLab.
