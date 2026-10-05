# Discourse Plugin Audit — Reference Notes

Extracted from session analyzing discourse-ai, chat, discourse-subscriptions, and discourse-workflows plugins.

## Plugin Inventory Pattern (Rails Engines)

```bash
# In a Discourse checkout:
ls plugins/                           # ~50+ plugins
find plugins/*/app/controllers -name '*.rb' | wc -l
grep -rn "skip_before_action" plugins/*/app/controllers/ --include='*.rb'
```

## SSRF Protection Assessment

Discourse's FinalDestination is best-in-class for Ruby SSRF protection:

| Feature | Discourse | Vercel AI SDK |
|---------|-----------|---------------|
| DNS resolution before connect | Yes (Addrinfo.getaddrinfo) | No |
| CGNAT (100.64.0.0/10) blocked | Yes | No (missed) |
| Private IPv6 blocked | Yes (::1, fc00::/7, fe80::/10) | No |
| Custom HTTP client with IP validation | Yes (FinalDestination::HTTP) | No |
| Redirect target validation | Yes (recursive get/validate) | Partial (final URL only) |
| DNS rebinding protection | Yes (resolves IP first, connects to IP) | No |

Key files:
- `lib/final_destination.rb` — Main class with resolve/get flows
- `lib/final_destination/ssrf_detector.rb` — IP range validation
- `lib/final_destination/resolver.rb` — Thread-safe DNS with timeout
- `lib/final_destination/http.rb` — Custom Net::HTTP subclass overrides connect()

## AI Plugin — Audit Log IDOR

**File**: `plugins/discourse-ai/app/controllers/discourse_ai/ai_bot/bot_controller.rb`
**Route**: `GET /discourse-ai/ai-bot/show-debug-info/:id`

```ruby
def show_debug_info_by_id
  log = AiApiAuditLog.find(params[:id])  # ← UNSCOPED
  raise Discourse::NotFound if !log.topic
  if log.post_id.present?
    raise Discourse::NotFound if !log.post
    guardian.ensure_can_debug_ai_bot_conversation!(log.post)  # checks can_see? not ownership
  else
    guardian.ensure_can_debug_ai_bot_conversation!(log.topic)
  end
  render json: AiApiAuditLogSerializer.new(log, root: false), status: :ok
end
```

Guardian gate: `ai_bot_debugging_allowed_groups_map` (site setting)
Guardian file: `plugins/discourse-ai/lib/guardian_extensions.rb`

**Fields exposed** via AiApiAuditLogSerializer:
- `raw_request_payload` — full prompt sent to AI provider
- `raw_response_payload` — full AI response
- `user_id`, `topic_id`, `post_id`, `provider_id`, `feature_name`
- `prev_log_id`, `next_log_id` — allows sequential enumeration

Model fields (from `db/migrate/20230424055354_create_ai_api_audit_logs.rb`):
- `raw_request_payload` (string) — full prompt
- `raw_response_payload` (string) — full AI response
- `request_tokens`, `response_tokens` (integer)
- `provider_id` (integer) — OpenAI=1, Anthropic=2, HuggingFace=3, Gemini=4, Vllm=5, etc.
- `feature_name` (string)
- `topic_id`, `post_id` (integer)
- `user_id` (integer)
- `llm_id` (integer)
- `duration_msecs` (integer)
- `response_status` (integer)
- `feature_context` (jsonb)

## Chat Plugin — Webhook Controller

**File**: `plugins/chat/app/controllers/chat/incoming_webhooks_controller.rb`
**Skips**: `verify_authenticity_token`, `redirect_to_login_if_required`, `redirect_to_profile_if_required`
**Auth**: Webhook key (`params[:key]`), looked up in `Chat::IncomingWebhook.find_by(key: key)`
**Rate limit**: 10 messages/minute per webhook
**Supports**: Slack-compatible payloads with `text`, `attachments`, or `payload` JSON string formats

## Workflows Plugin — Forms & Webhooks

**FormsController** (`plugins/discourse-workflows/app/controllers/discourse_workflows/forms_controller.rb`):
- Skips CSRF for ALL actions
- `show`, `waiting_show`, `waiting_status`: NO login required
- `create`, `test_create`: requires login + origin verification
- Auth: resume token, signature token

**WebhooksController** (`plugins/discourse-workflows/app/controllers/discourse_workflows/webhooks_controller.rb`):
- Skips CSRF, login, XHR check
- Rate limit: 20 req/10s per IP
- Auth: token/signature query param
- Passes `raw_authorization`, `raw_body`, `headers` to service

## Plugin Vulnerability Search Patterns (General)

```bash
# 1. Find controllers that skip auth
grep -rn "skip_before_action" plugins/*/app/controllers/ --include='*.rb'

# 2. Find unscoped ActiveRecord find calls
grep -rn "\.find(params" plugins/*/app/controllers/ --include='*.rb'

# 3. Find serializers exposing raw/sensitive fields
grep -rn "raw_\|_payload\|_token\|_secret\|_key" plugins/*/app/serializers/ --include='*.rb'

# 4. Check guardian/policy extensions in plugins
find plugins/*/lib -name '*guardian*' -o -name '*policy*'

# 5. Check plugin routes for debug/unauthenticated endpoints
grep -rn "get\|post\|put\|delete" plugins/*/config/routes.rb
```
