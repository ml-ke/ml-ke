---
title: Discourse Bug Bounty Methodology
name: discourse-bug-bounty
description: Systematic approach to finding vulnerabilities in Discourse open source forum software for the HackerOne bug bounty program. Covers source code analysis on the Ruby on Rails + Ember.js codebase and live testing against try.discourse.org.
---

# Discourse Bug Bounty Methodology

Target: Discourse (hackerone.com/discourse)
Bounties: Severity at discretion of Discourse team (no published table)
Test instance: https://try.discourse.org (resets daily)
Scope: 100% open source code at github.com/discourse/discourse + plugins
Tech stack: Ruby on Rails, Ember.js, PostgreSQL
Code: Cloned to ~/Dev/discourse/discourse/

## Program Info

Security policy at `docs/SECURITY.md`:
- Report via HackerOne: https://hackerone.com/discourse
- Email fallback: team@discourse.org (discouraged — low-quality reports filtered)
- Requires proof of concept with step-by-step reproduction
- Theoretical reports without PoC are not accepted

Known CVEs: 246 since 2018 (34 in 2024, 24 in 2025), avg CVSS 4.0
Latest notable: CVE-2025-46813 (GHSA-v3h7-c287-pfg9) — private data leak on login-required sites via routing change

## Source Code Navigation

### SSRF Protection (Comprehensive)
Discourse's SSRF protection is significantly better than most (e.g., Vercel's AI SDK):
- **FinalDestination**: `lib/final_destination.rb` — Main URL validation + fetch class
- **SSRFDetector**: `lib/final_destination/ssrf_detector.rb` — IP range validation
- **Resolver**: `lib/final_destination/resolver.rb` — Thread-safe DNS lookup with timeout
- **Custom HTTP**: `lib/final_destination/http.rb` — Overrides `Net::HTTP#connect` to resolve DNS and filter IPs before connecting

Key protections:
- Resolves DNS to actual IPs **before connecting** (prevents DNS rebinding)
- **Blocks CGNAT** (100.64.0.0/10) — better than Vercel which missed this
- Full IPv4 private ranges: 0.0.0.0/8, 10.0.0.0/8, 127.0.0.0/8, 169.254.0.0/16, 172.16.0.0/12, 192.168.0.0/16, 192.0.0.0/24, 198.18.0.0/15, 198.51.100.0/24, 203.0.113.0/24, 240.0.0.0/4, 255.255.255.255/32 and more
- Full IPv6 private ranges: ::1, ::, 64:ff9b:1::/48, fc00::/7, fe80::/10, etc.
- Excludes `::ffff:0:0/96` (IPv4-mapped IPv6) — allows this by design
- `allowed_internal_hosts` configurable via SiteSetting for whitelisting
- Uses resolved IP for actual HTTP connection, sets correct Host header for vhost routing
- Thread-safe DNS resolver with 2-second timeout (configurable via GlobalSetting)
- Custom `FinalDestination::HTTP` override of `Net::HTTP#connect()` — iterates through all resolved IPs

### Onebox (URL Preview)
- `lib/onebox.rb`, `lib/onebox/preview.rb` — URL preview system
- Uses FinalDestination for URL validation (not a separate SSRF vector)
- DEFAULTS: connect_timeout=5, timeout=10, max_download_kb=2048, allowed_ports=[80,443], allowed_schemes=[http,https], redirect_limit=5
- CSP ensures onebox content can't execute JS

### File Upload URL Fetch
- `app/controllers/uploads_controller.rb:307-324` — `create_upload` downloads from URL
- ONLY when `file.nil? && url.present? && is_api` — requires API authentication
- Uses `FileHelper.download()` which uses FinalDestination (SSRF protected)
- `follow_redirect: true` — redirects are followed but FinalDestination validates redirect targets

### Webhook Handlers
- `app/controllers/webhooks_controller.rb` — Mailgun, SendGrid, Mailjet, Mailpace, Mandrill, Postmark, SparkPost, AWS
- All have signature verification (API keys, HMAC, etc.)
- Accept raw POST data from email providers — potential deserialization/injection surface but all signature-checked

### File Uploaders
- Upload path uses hashed identifiers (no user-controlled paths)
- `lib/file_helper.rb` — `FileHelper.download()` fetches external URLs through FinalDestination
- Path traversal: uploader paths are auto-generated and not user-controllable

### Admin Authorization
- All admin controllers inherit from `AdminController` which has `before_action :ensure_admin`
- No admin-only actions found without proper auth checks

### Chat Plugin
- `plugins/chat/` — full chat system
- Check for authorization gaps in chat message access (separate from core forum permissions)

### AI Plugin (discourse-ai)
- `plugins/discourse-ai/` — 29 controllers, AI bot, embeddings, translations, spam detection
- **IDOR finding**: `AiApiAuditLog.find(params[:id])` in `bot_controller.rb:show_debug_info_by_id`
  - Unscoped integer ID lookup — no ownership check
  - Guardian only checks `can_see?(topic/post)`, not audit log ownership
  - **Column constraint (FACT-CHECKED)**: `raw_request_payload` and `raw_response_payload` are `t.string` → `varchar(255)` in PostgreSQL. AI request bodies are 10K+ chars. **Only first 255 chars stored.** This is truncated exposure, not full prompt leakage.
  - Data exposed (truncated): system prompt prefixes, model config, tool definition prefixes, first few words of user messages
  - Data NOT exposed: full conversation bodies, complete AI responses, large tool call results
  - Gated behind: requires login + `SiteSetting.ai_bot_debugging_allowed_groups_map` membership
  - CVSS: 5.3 (Medium) — not High due to truncation
- Source: `plugins/discourse-ai/app/controllers/discourse_ai/ai_bot/bot_controller.rb`
- Guardian: `plugins/discourse-ai/lib/guardian_extensions.rb:63`
- Serializer: `plugins/discourse-ai/app/serializers/ai_api_audit_log_serializer.rb`
- Migration (check column types!): `plugins/discourse-ai/db/migrate/20230424055354_create_ai_api_audit_logs.rb`

### Subscriptions Plugin (discourse-subscriptions)
- `plugins/discourse-subscriptions/app/controllers/discourse_subscriptions/hooks_controller.rb`
- Stripe webhook handler with signature verification (`Stripe::Webhook.construct_event`)
- Properly uses `User.find_signed` with purpose for user references
- Low vulnerability probability — well-structured Stripe integration

## Attack Surface Summary

| Area | Risk | Notes |
|------|------|-------|
| SSRF (FinalDestination) | **LOW** | Comprehensive IP filtering, DNS resolution, redirect validation |
| Command injection | **LOW** | No shell commands found in app/controllers or app/models |
| File upload path traversal | **LOW** | Hashed paths, no user control |
| Admin auth bypass | **LOW** | All admin controllers have `ensure_admin` |
| Webhook injection | **LOW** | All have signature verification |
| AI plugin audit log IDOR | **MEDIUM (5.3)** | Unscoped ID lookup, but varchar(255) truncates payload — truncated exposure, not full breach |
| Plugin auth gaps | **MEDIUM** | 52 plugins, newer ones may have weaker auth (AI, Chat, Subscriptions) |
| Email parsing / webhook | **LOW-MED** | Signature-verified but complex parsing logic |

## Known CVE Patterns

Recent CVEs in Discourse (2024-2025):
- **CVE-2025-46813** — Info disclosure via routing (login-required sites exposed content to unauthenticated users)
- **CVE-2024-53991** — Backup disclosure via Rails `send_file` quirk + predictable filenames + local file storage
- **GHSA-mq82-7v5x-rhv8** — Discourse Reactions plugin: reaction notification data exposure
- Common themes: routing misconfigurations, data exposure, plugin-specific bugs

## Testing Approach

1. Register on try.discourse.org (resets daily — work fast)
2. Check for auth gaps in newer plugins (AI, Chat, Subscriptions)
3. Test admin endpoints for CSRF or missing auth on new features
4. Check email webhook handlers for injection in parsed fields
5. Look for IDOR in AI audit logs if debugging is enabled
6. Review recent GitHub security advisories for plugin vulnerabilities
7. Check for bypasses in domain checker (`lib/onebox/domain_checker.rb`)

## Report Structure

When submitting to HackerOne:
1. Descriptive title
2. Summary with impact
3. Steps to reproduce with PoC
4. Source code reference (file + line numbers)
5. CVSS score (if applicable)
6. Fix recommendation

Reports go in `~/Dev/REPORTS/Discourse/<SubmissionNumber>/<finding-name>/` with:
- `REPORT.md` (standalone, not in zip)
- `poc/` subfolder with PoC code + submission.zip

## Pre-Submission Fact-Check Checklist

Before submitting any finding:
1. **Check migration files** — `t.string` = varchar(255). Don't claim full payload exposure for truncated columns.
2. **Trace the full auth chain** — Read the guardian/policy file, not just the controller. Does it verify OWNERSHIP or just VISIBILITY?
3. **Test on try.discourse.org** — Verify route exists, auth works, error codes match claims
4. **Downgrade honestly** — Staff-only access + truncated 255-char data = Medium, not Critical

## Reference Files

- `references/recon-summary.md` — Full initial recon results: source analysis, live instance fingerprinting, CVEs studied, plugin catalog, SSRF protection analysis
