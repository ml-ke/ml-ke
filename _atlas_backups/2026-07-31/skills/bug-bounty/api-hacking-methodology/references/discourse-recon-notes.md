# Discourse Recon Notes

Source: ~/Dev/discourse/discourse/ (22,581 files)
Tech stack: Ruby on Rails + Ember.js + PostgreSQL
Program: HackerOne (hackerone.com/discourse) — test on try.discourse.org only
CVEs: 246 since 2018 (34 in 2024, 24 in 2025). Well-audited. Average CVSS 4.0.
Security: SECURITY.md at docs/SECURITY.md

## SSRF Protection (Stronger Than Most)

Discourse's FinalDestination + SSRFDetector is the gold standard:

**lib/final_destination.rb** — Main HTTP client class
- Resolves DNS to actual IPs before connecting (prevents DNS rebinding)
- Two code paths with different protection:
  - `resolve()` uses Excon with explicit DNS resolution
  - `get()` → `safe_get()` uses FinalDestination::HTTP
- Both validate IPs via SSRFDetector

**lib/final_destination/ssrf_detector.rb** — IP validation
- Blocks CGNAT (100.64.0.0/10) — comprehensive
- Covers ALL IANA private IPv4 and IPv6 ranges
- `lookup_and_filter_ips()` resolves DNS and filters
- `ip_allowed?()` checks individual IPs
- Custom block list via `SiteSetting.blocked_ip_blocks`
- Custom allow list via `SiteSetting.allowed_internal_hosts`

**lib/final_destination/http.rb** — Net::HTTP subclass
- Overrides `connect()` to validate IPs via SSRFDetector
- Iterates through multiple resolved IPs
- Thread-safe DNS resolution with timeout

**lib/final_destination/resolver.rb** — DNS resolver
- Uses Addrinfo.getaddrinfo
- 2-second timeout (configurable)
- Thread-safe with Mutex + Queue

### Key Difference from Vercel AI SDK
Vercel's `validateDownloadUrl()` never resolves DNS — only checks hostname string against an IP blocklist. Discourse resolves DNS to actual IPs, filters, then connects using the IP (not hostname). This prevents DNS rebinding.

## Plugin Architecture

52 plugins in `plugins/`. Most are Rails Engines mounted at Discourse route.

### Plugins That Skip CSRF (highest signal for webhook bugs):
- `chat` — incoming_webhooks_controller.rb (key-authenticated)
- `discourse-ai` — shared_ai_conversations_controller.rb, discord/bot_controller.rb
- `discourse-github` — webhooks_controller.rb
- `discourse-subscriptions` — hooks_controller.rb (Stripe, signature-verified)
- `discourse-workflows` — forms_controller.rb, webhooks_controller.rb

### Plugin Controller Audit Commands
```bash
# Find all plugin controllers
find plugins/*/app/controllers -name '*.rb' -type f

# Find plugins skipping CSRF
for p in plugins/*/; do
  has=$(grep -rl "skip_before_action.*verify_authenticity_token" $p/app/controllers/ 2>/dev/null)
  [ -n "$has" ] && echo "$(basename $p): $has"
done

# Find plugins skipping login
grep -rn "skip_before_action.*redirect_to_login_if_required" plugins/*/app/controllers/

# Find unscoped ID lookups in plugins
grep -rn "\.find(params\[:id\])" plugins/*/app/controllers/ --include='*.rb'
```

### AI Plugin Audit (discourse-ai)

The AI plugin is newer and less audited. Key controllers:

| Controller | Route | Auth | Notes |
|-----------|-------|------|-------|
| bot_controller.rb | /ai-bot/show-debug-info/:id | login + group | Unscoped AiApiAuditLog.find() |
| shared_ai_conversations_controller.rb | /ai-bot/shared-ai-conversations/:share_key | login (show) | Share key = UUID |
| artifact_key_values_controller.rb | /ai-bot/artifact-key-values/:id | login | Key-value storage |
| assistant_controller.rb | /ai-helper/suggest | login | AI suggestions |
| mcp_oauth_controller.rb | /mcp/oauth | varies | MCP protocol OAuth |

### AI Audit Log IDOR Finding

**Root cause**: `AiApiAuditLog.find(params[:id])` in `show_debug_info_by_id` — unscoped integer ID lookup. The guardian only checks `can_see?(topic/post)`, not ownership.

**Data exposed via serializer**: `raw_request_payload`, `raw_response_payload`, `user_id`, `topic_id`, `post_id`, `provider_id`, `feature_name`, `llm_id`, `language_model`, `prev_log_id`, `next_log_id`.

**Critical column constraint**: Both payload columns are `t.string` (varchar(255) in PostgreSQL). Full AI request/response bodies (typically 10K+ chars) are truncated to 255 chars. This makes the finding Medium (CVSS 5.3) rather than High.

**Guard conditions for exploitability**:
- `discourse_ai_enabled` = true
- `ai_bot_enabled` = true
- `ai_bot_debugging_allowed_groups_map` non-empty
- Attacker is in that group (typically staff)
- Attacker can see the topic/post

## Chat Plugin

Chat is mounted as a Rails Engine. Incoming webhooks at `/hooks/:key` and `/hooks/:key/slack`. Key-authenticated (looked up via `Chat::IncomingWebhook.find_by(key: key)`). No login required. Rate limited (10 msg/min). If key is leaked, anyone can post as system user to the channel.

## Workflows Plugin

Mounted at `/`. Routes:
- `/workflows/form/:uuid` — Public forms (by design, UUID-based)
- `/workflows/webhooks/*path` — Token-authenticated webhooks
- `/admin/plugins/discourse-workflows/*` — Admin-only (AdminConstraint)

Forms skip CSRF and login for `show`. Webhooks skip both. Rate limited by IP.

## Useful Source File Paths

| File | Purpose |
|------|---------|
| `lib/final_destination.rb` | SSRF protection main class |
| `lib/final_destination/ssrf_detector.rb` | IP blocking logic |
| `lib/final_destination/http.rb` | Custom Net::HTTP with SSRF |
| `lib/final_destination/resolver.rb` | DNS resolver |
| `lib/onebox.rb` | URL preview system (SSRF vector) |
| `lib/file_helper.rb` | File download (uses FinalDestination) |
| `lib/retrieve_title.rb` | URL title fetcher (uses FinalDestination) |
| `app/controllers/webhooks_controller.rb` | Email provider webhooks |
| `app/controllers/uploads_controller.rb` | File uploads |
| `config/routes.rb` | Main routes |
| `plugins/*/config/routes.rb` | Plugin routes |
