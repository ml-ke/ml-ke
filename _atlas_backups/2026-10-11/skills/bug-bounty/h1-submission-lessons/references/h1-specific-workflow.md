# H1-Specific Pre-Submission Workflow

This file covers HackerOne-specific workflow steps. For the generic pre-submission verification framework (class-specific gates, impact assessment, VRT classification), use the standalone skill `bug-bounty/pre-submission-verification`.

## Step 1: Verify Scope (5 min)

- [ ] Load the program's H1 scope CSV (`curl -sL "https://hackerone.com/teams/{program}/assets/download_csv.csv"` or `https://hackerone.com/teams/{program}-vdp/assets/download_csv.csv`)
- [ ] Check each asset's `eligible_for_submission` column
- [ ] Check `instruction` column for inclusions/exclusions
- [ ] Check for "Out of Scope" notes in the policy text

**Case study — Anthropic**: If we had loaded the scope CSV, we'd have seen MCP repos have `eligible_for_submission: false`. 3+ days saved.

## Step 2: Understand the Security Model (10 min)

- [ ] For AI agent tools: is the permission prompt the intended security boundary?
- [ ] For SSRF in agent tools: does the tool show the URL to the user first?
- [ ] Check Hacktivity for how they classify similar reports (Duplicate vs Informative vs Triaged)

**Red flags for working-as-designed:**
- Behavior documented in README/docs
- Explicit comments saying "this is intentional"
- Similar findings previously marked Informative

## Step 3: Verify Novelty (15 min)

- [ ] Search program Hacktivity for keywords matching the vulnerability class
- [ ] Search GitHub Security Advisories
- [ ] Search CVE database
- [ ] For bypass findings: verify technique is novel, not a variant of known issue

## Step 4: Assess Impact (5 min)

- [ ] Read past reports — what severity for similar findings?
- [ ] Unauthenticated (PR:N) or requires session (PR:L)?
- [ ] User approval required? (UI:R vs UI:N)
- [ ] Core (bounty-eligible) or Non-Core (discretionary)?

## Step 5: Decision Table

| If... | Then... |
|-------|---------|
| Asset is OOS | Stop. Move on. |
| Similar finding was Informative | Don't submit without fundamentally different angle |
| 10+ SSRF reports exist | Skip SSRF on this program |
| Permission prompt = boundary | Focus on prompt bypass, not post-approval effects |
| Self-hosted + source scope | Green field — low competition |
