# GitLab AI Trial Testing Results (May 31, 2026)

## Access Level
- `duoChatAvailable: true` — feature flag active
- `duoStatus: null` — no subscription/license data present
- `aiAction` mutation accepts all chat requests, stores messages, returns thread IDs
- AI response consistently: "I'm sorry, you don't have the GitLab Duo subscription required to use Duo Chat."
- Non-chat methods (`explain_vulnerability`, `generate_description`): return empty error arrays, `requestId: null`, silently skip execution

## GraphQL Surface Verified
All queries confirmed working on gitlab.com with PAT auth:

- `aiConversationThreads` — returns user's threads, scoped to `current_user`
- `aiMessages` — returns messages with fields: `id`, `content`, `role`, `requestId`, `timestamp`, `errors`, `extras (AiMessageExtras)`
- `aiAction` mutation — accepts `chat`, `explain_vulnerability`, `generate_description`, `summarize_review` methods
- `duoSettings` — `duoCliEnabled` field confirmed
- `currentUser.duoChatAvailable` — boolean field
- `currentUser.duoStatus` — type `UserDuoStatus` with unknown subfields (none of the probed names matched)

## Authorization Verification

### Thread Access (Properly Scoped) ✅
- `ThreadFinder` scopes to `current_user.ai_conversation_threads`
- `ThreadEnsurer` uses `user.ai_conversation_threads.in_organization(organization)`
- Enumeration test: Thread IDs 2606590-2606596 — only own thread (2606593) accessible
- Other users' threads: return empty nodes (no error — just not found)

### Message Access (Properly Scoped) ✅
- `aiMessages` returns only current user's messages (10 messages in test)
- No thread/owner filter available in query to access other users' data

### Resource Authorization (Properly Scoped) ✅
- Invalid project ID (`Project/999999999`): returns "The resource that you are attempting to access does not exist or you don't have permission to perform this action"
- Empty content: returns "content can't be blank"
- Rate limiting: `ai_action` throttle per user confirmed in source code

## Duo Workflow Surface
- `ee/app/graphql/mutations/ai/duo_workflows/create.rb`
- `ee/app/graphql/mutations/ai/duo_workflows/delete_workflow.rb`
- `ee/app/graphql/mutations/ai/duo_workflows/update_tool_call_approvals.rb`
- Resolvers: `workflows_resolver.rb`, `workflow_events_resolver.rb`, `session_artifacts_resolver.rb`
- NOT tested — requires licensed account

## AI Features Catalogue
Full list at `ee/lib/gitlab/llm/utils/ai_features_catalogue.rb`
- External methods (accessible via aiAction): chat, explain_vulnerability, resolve_vulnerability, summarize_review, measure_comment_temperature, generate_description, generate_commit_message, description_composer, summarize_new_merge_request, agentic_chat
- Internal methods: categorize_question, review_merge_request, classify_code_review_mention_intent, duo_workflow, code_suggestions, troubleshoot_job, glab_ask_git_command, etc.
- Execute methods defined for: chat, resolve_vulnerability, summarize_review, measure_comment_temperature, generate_description, generate_commit_message, description_composer, summarize_new_merge_request, categorize_question, review_merge_request, glab_ask_git_command

## Source Code Structure
```
ee/app/graphql/mutations/ai/           — AI GraphQL mutations
ee/app/graphql/resolvers/ai/           — AI GraphQL resolvers
ee/app/graphql/types/ai/               — AI GraphQL types
ee/app/services/llm/                   — AI service layer (ChatService, BaseService, etc.)
ee/lib/gitlab/llm/                     — AI core logic (AiMessage, ChatStorage, etc.)
ee/lib/gitlab/llm/utils/               — AI utilities (AiFeaturesCatalogue, Authorizer)
ee/lib/gitlab/llm/completions/         — AI completion handling
ee/lib/gitlab/llm/templates/           — AI prompt templates
```

## Key Takeaway
Duo Chat is gated behind a PAID subscription at the backend gateway level. The `duoChatAvailable` feature flag alone is insufficient — the AI gateway subscription check (`user.allowed_to_use?(:chat)`) must also pass. Without a paid GitLab Ultimate license, AI features cannot be tested end-to-end on gitlab.com. Non-chat methods also skip execution without proper licensing.
