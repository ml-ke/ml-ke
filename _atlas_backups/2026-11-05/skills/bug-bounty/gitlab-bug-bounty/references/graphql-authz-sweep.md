# GitLab GraphQL Authorization Sweep (May 2026)

## Summary
Comprehensive sweep of GitLab's GraphQL API and REST API for authorization gaps, information disclosure, and privilege escalation vectors.

## Methodology
1. Full GraphQL introspection (3,865 types, 625 mutations, 163 queries, 34 subscriptions)
2. Mutation authorization testing (member invite, token creation, snippet update)
3. REST API endpoint testing (CI variables, runner tokens, deploy keys, project export)
4. Source code analysis of permission patterns (Guardian, policies, resolvers)
5. Field suggestion probing for hidden/undocumented fields

## Results

### Confirmed (properly secured)
| Operation | Result | Notes |
|-----------|--------|-------|
| CI variables on gitlab-org/gitlab | 403 Forbidden | Properly scoped to maintainer+ |
| Runner tokens on gitlab-org/gitlab | 403 Forbidden | Not exposed via API |
| Deploy keys on gitlab-org/gitlab | 403 Forbidden | Not visible to non-members |
| Group variables on gitlab-org | 403 Forbidden | Properly scoped |
| Project export (unauthenticated) | 404 | Not accessible without auth |
| Sudo endpoint | 403 | Requires admin token |
| projectMemberCreate mutation | BLOCKED | Proper authorization |
| projectAccessTokenCreate mutation | BLOCKED | Requires maintainer+ |
| updateSnippet mutation | BLOCKED | Not authorized |
| todoCreate mutation | BLOCKED | Resource not found |
| REST /users (unauthenticated) | 403 | Properly blocked |

### Confirmed (by design / public)
| Operation | Result | Notes |
|-----------|--------|-------|
| User enumeration via GraphQL | 100 users/query | Authenticated only, public data |
| Public snippet content | Readable | By design — public snippets are public |
| Project member info (public projects) | Visible | By design — public project members are public |
| Internal project discovery | Visible | By design — "internal" visibility accessible to any auth user |
| GraphQL introspection | Full access | 3,865 types — intentionally enabled |
| Field suggestions | Enabled | Error messages suggest field names |
| Echo resolver (test endpoint) | Works | Returns `username says: text` — no injection possible |

### Potential Vectors (need deeper investigation)
| Vector | Risk | Code Path |
|--------|------|-----------|
| CI_JOB_TOKEN scope (disabled by default) | Medium | `ci_job_token_scope_enabled = False` by default. JWT-based tokens may have wider scope than intended on self-hosted |
| Project export NDJSON injection | Low | Exported files could contain malicious content re-imported later. Source: `app/services/projects/import_export/` |
| AI/Duo Chat features | Medium | Newest features, least audited. `aiAction` mutation in schema |
| Bulk import path traversal | Low | `app/services/bulk_imports/archive_extraction_service.rb` — tar extraction |
| Runner registration token | Low-Medium | Token is `None` in API response (intentionally hidden), but reset endpoint exists |
| Two-phase URL validation (creation vs execution) | Medium | Creation validation is less strict (`dns_rebind_protection: false`). Exploitable if admin changes settings |

## Key File References

### SSRF Protection
- `gems/gitlab-http/lib/gitlab/http_v2/url_blocker.rb` — Main URL validation
- `app/validators/public_url_validator.rb` — Public URL validator
- `app/validators/addressable_url_validator.rb` — Base URL validator (line 64: `dns_rebind_protection` default)

### CI_JOB_TOKEN
- `lib/ci/job_token/jwt.rb` — JWT encode/decode for job tokens
- `app/finders/ci/auth_job_finder.rb` — Job auth with JWT support
- `app/models/ci/build.rb:743` — CI_JOB_TOKEN generation
- `app/models/concerns/ci/job_token/expanded_job_token_policies.rb` — Scope policies

### GraphQL
- `app/graphql/resolvers/autocomplete_users_resolver.rb` — User autocomplete
- `app/graphql/resolvers/echo_resolver.rb` — Debug echo endpoint
- `app/graphql/resolvers/snippets_resolver.rb` — Snippet access
- `app/policies/personal_snippet_policy.rb` — Snippet authorization

### Runners
- `app/services/ci/runners/reset_registration_token_service.rb` — Token reset
- `app/services/ci/runners/register_runner_service.rb` — Runner registration
- `app/models/ci/runner.rb` — Runner model

### Import/Export
- `app/services/projects/import_export/relation_export_service.rb` — Relation export
- `app/services/bulk_imports/archive_extraction_service.rb` — Bulk import extraction
- `app/services/projects/import_service.rb` — Project import

## Test Accounts
- GitLab user: h0d4r1-bugbounty (ID 38684912)
- PAT: ~/.gitlab-token (glpat-..., full api scope)
- Project: ID 82711200, path h0d4r1-bugbounty-group/h0d4r1-bugbounty-project
- Group: h0d4r1-bugbounty-group
