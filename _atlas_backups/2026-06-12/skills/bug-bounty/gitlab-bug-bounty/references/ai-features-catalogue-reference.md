# GitLab AI Features Catalogue

Complete listing of AI features from `ee/lib/gitlab/llm/utils/ai_features_catalogue.rb` (May 2026).

## External Methods (accessible via `aiAction` mutation)

| Method | Execute Method | Feature Category | Maturity | Self-Managed | File Path |
|--------|---------------|------------------|----------|-------------|-----------|
| `explain_vulnerability` | nil | vulnerability_management | ga | true | — |
| `resolve_vulnerability` | Llm::ResolveVulnerabilityService | vulnerability_management | ga | true | ee/app/services/llm/resolve_vulnerability_service.rb |
| `summarize_review` | Llm::MergeRequests::SummarizeReviewService | duo_code_review | experimental | true | — |
| `measure_comment_temperature` | Llm::Notes::MeasureCommentTemperatureService | ai_abstraction_layer | experimental | false | — |
| `generate_description` | Llm::GenerateDescriptionService | team_planning | experimental | false | — |
| `generate_commit_message` | Llm::GenerateCommitMessageService | code_review_workflow | ga | true | — |
| `description_composer` | Llm::DescriptionComposerService | code_review_workflow | experimental | true | — |
| `chat` (Duo Chat) | Llm::ChatService | duo_chat | ga | true | ee/app/services/llm/chat_service.rb |
| `summarize_new_merge_request` | Llm::SummarizeNewMergeRequestService | duo_code_review | beta | true | — |
| `agentic_chat` | nil | duo_chat | ga | false | — |

## Internal Methods (not exposed via `aiAction`)

Includes: `categorize_question`, `review_merge_request`, `classify_code_review_mention_intent`, `summarize_duo_workflow`, `generate_duo_workflow_title`, `code_suggestions`, `troubleshoot_job`, `duo_workflow`, `duo_agent_platform`, `foundational_flows`, `glab_ask_git_command`, `anthropic_proxy`, `vertex_ai_proxy`.

## Key Source Files

| Component | Path |
|-----------|------|
| Catalogue | `ee/lib/gitlab/llm/utils/ai_features_catalogue.rb` |
| aiAction mutation | `ee/app/graphql/mutations/ai/action.rb` |
| ChatService | `ee/app/services/llm/chat_service.rb` |
| BaseService (auth) | `ee/app/services/llm/base_service.rb` |
| ExecuteMethodService | `ee/app/services/llm/execute_method_service.rb` |
| ThreadEnsurer | `ee/lib/gitlab/llm/thread_ensurer.rb` |
| ThreadFinder | `ee/app/finders/ai/conversations/thread_finder.rb` |
| ThreadsResolver | `ee/app/graphql/resolvers/ai/conversations/threads_resolver.rb` |
| Guardian extension | `ee/lib/gitlab/llm/guardian_extensions.rb` |
| ChatMessage model | `ee/lib/gitlab/llm/chat_message.rb` |
| AiMessage model | `ee/lib/gitlab/llm/ai_message.rb` |
| AiApiAuditLog model | `plugins/discourse-ai/app/models/ai_api_audit_log.rb` |
| DeleteConversationThread | `ee/app/graphql/mutations/ai/delete_conversation_thread.rb` |
| DuoWorkflows create | `ee/app/graphql/mutations/ai/duo_workflows/create.rb` |

## Auth Chain

For `aiAction` mutation:

1. Rate limit check: `Gitlab::ApplicationRateLimiter.throttled?(:ai_action, scope: [current_user])`
2. Feature flag check: `Gitlab::Llm::Utils::FlagChecker.flag_enabled_for_feature?(method)`
3. Resource auth: `Authorizer.resource(resource, user).allowed?` (READ access check)
4. License check: `user.allowed_to_use?(ai_action)`
5. Chat-specific: `Gitlab::Llm::Chain::Utils::ChatAuthorizer.user(user: user).allowed?`

## License Gating

Duo Chat requires a paid GitLab subscription. Free accounts receive:
```
"AI features are not enabled or resource is not permitted to be sent."
```

Non-chat methods (`generate_description`, etc.) return empty errors but don't queue workers.
