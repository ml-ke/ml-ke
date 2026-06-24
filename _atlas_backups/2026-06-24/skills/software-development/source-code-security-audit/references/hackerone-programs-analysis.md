# HackerOne Program Analysis (May 2026)

Programs analyzed during this session, with scope, tech stack, and verdict notes.

## Verified Active (Bounties + Open Source)

| Program | Stack | Bounty Range | Reports | Verdict |
|---------|-------|-------------|---------|---------|
| **GitLab** | Ruby on Rails | $100-$35k | 1440/yr | Source cloned, PAT ready. SSRF saturated. AI features gated behind license. |
| **Vercel OSS** | TypeScript/Node.js | $50-$10k | 65 total | Already submitted to. SSRF was duplicate. launch-editor path traversal pending. |
| **MongoDB** | C++ | N/A | N/A | Wrong skillset. |
| **Ruby** | C | N/A | N/A | Core language — tiny attack surface. |
| **Cloudflare** | Mixed | $250-$10k | 407 | Ultra-competitive, top researchers. |
| **Django** | Python | N/A | N/A | Well-audited, Python-focused. |

## Paused / Suspended (Do Not Submit)

| Program | Status | Reason |
|---------|--------|--------|
| **Discourse** | Bounties suspended | Confirmed by user |
| **Node.js** | Bounties paused | Funding lost |
| **Internet Bug Bounty (IBB)** | Paused | AI flood of submissions overwhelmed triage |
| **n8n** | Temporarily closed | Recent CVEs being addressed internally |
| **Dify** | Temporarily closed | Recent CVEs being addressed internally |

## Strong Next Targets

### Elastic / Kibana (Current Target)
- **Stack**: TypeScript/Node.js (Kibana) — our proven skillset
- **Payout**: $150-$7,000 (Critical $3k-$7k, High $1.5k-$3k)
- **Source**: github.com/elastic/kibana (110K files, cloned)
- **Response**: 10 hours first response
- **Explicitly wants**: Kibana XSS with CSP bypass, authenticated SSRF, IDOR, privilege escalation
- **Program page**: hackerone.com/elastic
- **Scope**: Kibana, Elasticsearch, cloud.elastic.co, *.elastic.dev, Logstash, Beats, etc.
- **Out of scope**: Third-party systems, admin-level findings, legacy versions, XSS without CSP bypass (case-by-case)
- **Notes**: "Our code is open so use that to your advantage" — encourages source auditing

### Kubernetes
- **Stack**: Go
- **Payout**: $100-$10k
- **Source**: 72 repos in scope
- **Response**: Triaged by HackerOne
- **Notes**: Go is learnable but not our primary strength. Very competitive.

### Twilio
- **Stack**: Various (API-focused)
- **Payout**: $50-$8k
- **Source**: API-focused, limited source code in scope
- **Notes**: API testing skills apply but harder without source access.

## Key GitLab AI Surface

GitLab's `ee/` directory contains extensive AI features exposed via GraphQL:

- `aiAction` mutation — 625 mutation fields, main entry for all AI features
- `AiFeaturesCatalogue` — 30+ features defined, 11 external (accessible via aiAction)
- External methods: chat, explain_vulnerability, resolve_vulnerability, summarize_review, generate_description, generate_commit_message, description_composer, summarize_new_merge_request, agentic_chat
- Gated behind GitLab Ultimate subscription on gitlab.com
- Free accounts get "AI features are not enabled" error
- Thread safety: proper user scoping via `current_user.ai_conversation_threads`
- Rate limits: `ai_action` throttle per user
- Source: `ee/lib/gitlab/llm/utils/ai_features_catalogue.rb`
