# Pre-Submission Verification Workflow

Used to verify Supabase self-hosted auth bypass before submission.
Prevents wasted submission slots by catching errors BEFORE clicking submit.

## The 4-Step Verification Cycle

### Step 1: Claim Extraction
Before verifying, list every factual claim in the report:

```
Claim 1: apiWrapper.ts line 41 uses `if (IS_PLATFORM && withAuth)`
Claim 2: api-keys.ts returns SUPABASE_SERVICE_KEY directly
Claim 3: pg-meta query has { withAuth: true } but auth is skipped
...
```

### Step 2: Source Code Verification
For each claim, read the ACTUAL source file:

```
read_file(path="apps/studio/lib/api/apiWrapper.ts")
# → CONFIRMED: line 41 `if (IS_PLATFORM && withAuth)`
```

**Check three things per claim:**
- [ ] Does the file exist at the claimed path?
- [ ] Does the claimed line/behavior match what's in the file?
- [ ] Are there any imports, wrappers, or conditionals BETWEEN the auth check and the handler that could change behavior?

**Critical gotchas caught by this method:**
- Supabase api-keys endpoint calls `apiWrapper(req, res, handler)` with NO fourth arg — `withAuth` is `undefined`, not just bypassed
- pg-meta query has `{ withAuth: true }` — the issue is `IS_PLATFORM=false` shorts the check, not that withAuth is missing
- edge-functions/test is a bare handler that doesn't call apiWrapper at ALL

### Step 3: Test & Adjacent Code Review
Check the code that's NOT the main target:

- **Test files**: Do tests cover the vulnerable path? (Supabase tests only mock IS_PLATFORM=true)
- **Constants**: How is the flag defined? IS_PLATFORM = NEXT_PUBLIC_IS_PLATFORM === 'true'
- **Imports**: Is the import what you expect? (import { IS_PLATFORM } from '@/lib/constants')
- **Connected files**: The handler calls executeQuery() — what does THAT function do? Does it use superuser?

### Step 4: External Verification
Before investing time, check:

- **CVE database**: Search for the vulnerability class + project
- **Merged fix**: `git log --oneline --all --grep="auth\|IS_PLATFORM\|middleware"` in the target repo
- **Unmerged fix branches**: `git branch -a | grep -i "fix\|auth\|security\|middleware"`
- **H1 policy**: Read the program's full scope CSV. Look for specific endpoint exclusions
- **GitHub issues**: Search for related open/closed issues
- **Ask**: "Is this working as designed?" — if the program could argue it's intentional, reconsider framing

## Supabase Fact-Check Results (June 2026)

| Claim | Verification | Result |
|-------|-------------|--------|
| IS_PLATFORM && withAuth (apiWrapper.ts:41) | Read source | CONFIRMED |
| api-keys.ts bare handler (no apiWrapper) | Read source line 13 | CONFIRMED |
| pg-meta has withAuth:true but auth skipped | Read source line 9 | CONFIRMED |
| edge-functions/test no apiWrapper | Read source line 6 | CONFIRMED (bare handler) |
| Tests cover IS_PLATFORM=false? | Read test file line 10 | GAP — always IS_PLATFORM:true |
| CVE exists? | Search CVE database | No CVE for this bypass |
| Fix merged? | git log --grep="auth" | NOT MERGED |
| Fix branch exists? | git branch -a | middleware-studio (Feb 2026, no PR) |
| H1 policy excludes? | Read policy CSV | pg-meta SQLi excluded but auth bypass != SQLi |

## Signals That a Finding Is Wasted

From 6 real submissions:

| Signal | What It Means | Example |
|--------|---------------|---------|
| "Working as designed" in triage | Intentional behavior | Anthropic WebFetch — prompt IS the boundary |
| Asset in wrong org | You checked wrong scope | MCP Fetch under modelcontextprotocol, not anthropic |
| "Endpoint intentionally accepts X" | Intended functionality | Supabase pg-meta raw SQL, MCP Fetch any-URL |
| Multiple prior SSRF reports | Crowded class, high duplicate risk | Vercel OSS (3/6 duplicates = SSRF) |
| Fix exists but unmerged | Vendor knows — accelerate disclosure | Supabase middleware-studio branch |
| All live tests return 403/404 | Platform proxy blocks — check code instead | Supabase hosted Vercel Edge proxy |
