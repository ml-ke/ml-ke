# GitLab AI Features Catalogue

Source: `ee/lib/gitlab/llm/utils/ai_features_catalogue.rb`

## External Methods (accessible via `aiAction` mutation)

| Method | Execute Service | Feature Category | Maturity | Self-Managed | Description |
|--------|----------------|-----------------|----------|-------------|-------------|
| `chat` | `Llm::ChatService` | duo_chat | GA | Yes | Duo Chat conversations |
| `explain_vulnerability` | nil | vulnerability_management | GA | Yes | Explain a vulnerability |
| `resolve_vulnerability` | `Llm::ResolveVulnerabilityService` | vulnerability_management | GA | Yes | Auto-resolve vulnerability |
| `summarize_review` | `Llm::MergeRequests::SummarizeReviewService` | duo_code_review | Experimental | Yes | Summarize MR review |
| `measure_comment_temperature` | `Llm::Notes::MeasureCommentTemperatureService` | ai_abstraction_layer | Experimental | No (SaaS only) | Measure comment tone |
| `generate_description` | `Llm::GenerateDescriptionService` | team_planning | Experimental | No (SaaS only) | Auto-generate issue description |
| `generate_commit_message` | `Llm::GenerateCommitMessageService` | code_review_workflow | GA | Yes | Auto-generate commit message |
| `description_composer` | `Llm::DescriptionComposerService` | code_review_workflow | Experimental | Yes | Compose descriptions |
| `summarize_new_merge_request` | `Llm::SummarizeNewMergeRequestService` | duo_code_review | Beta | Yes | Summarize new MR |
| `agentic_chat` | nil | duo_chat | GA | Yes | Agent-based chat |

## Internal Methods (not exposed via `aiAction`)

| Method | Execute Service | Feature Category | Maturity | Description |
|--------|----------------|-----------------|----------|-------------|
| `categorize_question` | `Llm::Internal::CategorizeChatQuestionService` | duo_chat | GA | Categorize chat questions |
| `review_merge_request` | `Llm::ReviewMergeRequestService` | duo_code_review | GA | Deep MR review |
| `classify_code_review_mention_intent` | nil | code_suggestions | Experimental | Classify mentions |
| `summarize_duo_workflow` | nil | duo_agent_platform | Experimental | Summarize workflow |
| `generate_duo_workflow_title` | nil | duo_agent_platform | Experimental | Generate workflow title |
| `glab_ask_git_command` | `Llm::GitCommandService` | source_code_management | GA | Git command assistance |
| `code_suggestions` | nil | continuous_integration | GA | Code suggestions |
| `troubleshoot_job` | nil | code_suggestions | GA | Job troubleshooting |
| `duo_workflow` | nil | duo_agent_platform | GA | Duo Workflows |
| `duo_agent_platform` | nil | duo_agent_platform | GA | Agent platform |
| `foundational_flows` | nil | duo_agent_platform | GA | Foundational flows |
| `ai_catalog` | nil | workflow_catalog | GA | AI catalog |
| `ai_catalog_flows` | nil | workflow_catalog | Beta | Catalog flows |
| `ai_catalog_third_party_flows` | nil | workflow_catalog | GA | Third-party flows |
| `anthropic_proxy` | nil | ai_abstraction_layer | GA | Anthropic proxy |
| `vertex_ai_proxy` | nil | ai_abstraction_layer | GA | Vertex AI proxy |
| `summarize_comments` | nil | duo_chat | GA | Summarize comments |
| `ask_build`, `ask_issue`, `ask_epic`, `ask_merge_request`, `ask_commit` | nil | duo_chat | GA | Context-aware chat |

## Key Source Files

| File | Purpose |
|------|---------|
| `ee/app/graphql/mutations/ai/action.rb` | Main aiAction mutation |
| `ee/app/graphql/mutations/ai/delete_conversation_thread.rb` | Delete thread |
| `ee/app/graphql/mutations/ai/duo_workflows/create.rb` | Create workflow |
| `ee/app/graphql/mutations/ai/feature_settings/update.rb` | AI feature settings |
| `ee/app/graphql/mutations/ai/domain_settings/namespace_update.rb` | Namespace AI settings |
| `ee/app/graphql/resolvers/ai/conversations/threads_resolver.rb` | List threads |
| `ee/app/graphql/resolvers/ai/conversations/title_resolver.rb` | Thread title |
| `ee/app/graphql/resolvers/ai/chat_messages_resolver.rb` | Chat messages |
| `ee/app/finders/ai/conversations/thread_finder.rb` | Thread scope logic |
| `ee/app/services/llm/execute_method_service.rb` | Method execution dispatch |
| `ee/app/services/llm/base_service.rb` | Base service (auth logic) |
| `ee/app/services/llm/chat_service.rb` | Chat service |
| `ee/lib/gitlab/llm/utils/ai_features_catalogue.rb` | Feature definitions |
| `ee/lib/gitlab/llm/utils/authorizer.rb` | Resource authorization |
| `ee/lib/gitlab/llm/thread_ensurer.rb` | Thread creation/retrieval |
| `ee/lib/gitlab/llm/chat_message.rb` | Chat message model |
| `ee/lib/gitlab/llm/ai_message.rb` | Base AI message |

## Auth Flow (aiAction → execution)

```
aiAction mutation
  → verify_rate_limit! (per-user throttle)
  → extract_method_params! (parse method + args)  
  → check_feature_flag_enabled! (FlagChecker)
  → find_resource (authorized_find! via GlobalID)
  → handle_chat_arguments (ThreadEnsurer for chat)
  → Llm::ExecuteMethodService
    → AiFeaturesCatalogue lookup
    → find execute_method class
    → BaseService#execute
      → valid? check
        → Authorizer.resource allowed? (READ access)
        → ai_integration_enabled? (FlagChecker)
        → user_can_send_to_ai? (license check)
      → perform (actual work)
```

## Constraint Notes

- `chat` requires Duo Chat license (`user.allowed_to_use?(:chat)`)
- Non-chat methods silently fail on free accounts (empty errors, no work queued)
- Thread access scoped to `current_user.ai_conversation_threads` (ActiveRecord scope)
- Message access scoped via `Ai::Conversation::Message.for_user(user)`
- Rate limit: `ai_action` throttle (per user, configurable)
- AI features live in `ee/` (Enterprise Edition) — not present in CE
