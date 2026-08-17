---
name: antigravity-principles
description: "Use when planning, building, reviewing, or evaluating work with Google Antigravity agents (CLI agy, 2.0 desktop, Python SDK). System design theory applied to the harness: permission boundaries, lifecycle hooks, subagent topology, artifacts as message passing. Theory + orientation — not a command manual."
version: 1.2.0
author: ATLAS
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Antigravity, System-Design, Theory, Orchestration]
    related_skills: [opencode-principles, opencode, atlas-principles, system-design-theory]
---

# Antigravity Principles

Google Antigravity is an agent harness with multiple surfaces: **Antigravity CLI** (`agy`, keyboard-first TUI, low overhead, SSH-friendly), **Antigravity 2.0** (desktop command center: projects, worktrees, parallel local subagents, scheduled tasks), **Antigravity SDK** (Python — `pip install google-antigravity`, the `Agent` class wraps the full agentic loop), and an **IDE** extension. All surfaces share one core agent engine. Install CLI: `curl -fsSL https://antigravity.google/cli/install.sh | bash`; auth via keyring/Google Sign-In (SSH prints a URL). Docs: antigravity.google/docs.

This skill is the *theory*: the "Scale from Zero to Millions" theorems applied to how Antigravity's harness works and how to think about building on it. For interactive mechanics, consult the official docs. The principles are what survive version bumps.

## The Harness as an Architecture

Antigravity gives you the primitives of a distributed system out of the box: an **agent engine** (compute), **permissions** (security boundary), **hooks** (middleware), **subagents** (workers), **artifacts** (message passing), **MCP/tools/skills** (services), and **sidecars/cron** (async + automation). If you were designing an agent platform from the "millions of users" playbook, these are the tiers you'd build.

## 1. Permissions are the security boundary — least privilege by design

- The SDK `Agent` runs **read-only by default**; writes require explicitly enabling capabilities. That is the principle of least privilege applied to an autonomous worker — a stateless service with a minimal permission surface has a smaller blast radius.
- Declarative safety policies (allow/deny/ask per capability) are the firewall rules of the harness. Define them at the boundary, not inside the logic.
- **Redundancy at this tier** — a permission layer is defense in depth: even if the model's reasoning is compromised (prompt injection), the permission layer is a second, independent gate that doesn't reason, it just denies.

## 2. Lifecycle hooks (inspect/decide/transform) are middleware

Hooks are the interceptor chain of the agent loop — inspect before, decide during, transform after. This is the middleware pattern from web frameworks applied to agent execution:

- **Decoupling** — hooks observe and modify without the agent logic knowing they exist. Logging, validation, and policy enforcement plug in like middleware, not like code edits.
- **Observability** — the inspect hook is where metrics and logging belong (the monitoring tier). An agent you can't inspect is a system you can't operate.
- **Separation of concerns** — policy lives in hooks, behavior lives in the agent, tools live behind interfaces. Each layer scales and changes on its own axis.

## 3. Subagents are horizontal scaling

- Parallel local subagents = the load-balanced worker pool. Partition work by module or concern; each subagent is a stateless worker with its own scope.
- **Sharding discipline applies** — give each subagent a disjoint slice (file, module, worktree) so workers never write the same data. Unpartitioned parallelism corrupts like two masters on one database.
- **Back-of-envelope first** — choose the surface by task size: CLI for quick/SSH interactions, 2.0 for heavy orchestration, SDK when you need to embed the loop in your own system. Vertical scaling (one conversation doing everything) hits the same limits as a single beefy server: cost, context, and no redundancy.

## 4. Artifacts are the message queue

- Agents produce **artifacts** (plans, reviews, diffs, files) — durable outputs that outlive the session. That is the message-passing layer: producers and consumers never need to be alive at the same time.
- **Statelessness via durability** — session history is persisted; a session can be exported from CLI to the 2.0 GUI and continue elsewhere. State lives in the platform, not in the agent's head — the same reason you move session data out of web servers.
- **Replay and audit** — because artifacts persist, a failure can be replayed and an outcome audited. This is the write-ahead log of agent operations.

## 5. Tools, MCP, skills = services behind interfaces

- Custom tools and MCP servers are microservices: the agent calls an interface, never the implementation. You can swap the implementation (upgrade a tool, point at a different MCP server) without touching the agent logic.
- **Decoupling** — the tool registry is the service registry; the skill store is the cached knowledge tier (progressive disclosure keeps the hot cache small).
- **Sidecars and scheduled tasks (cron sidecars) = async workers + automation** — background work decouples long-running jobs from the interactive path, exactly like a message queue consumer does for a web tier.

## 6. Security is a first-class theorem

The vendor's own warning names the failure modes: autonomous code execution, data exfiltration, prompt injection, supply chain. Theory response:

- **Redundancy**: monitor and verify every action the agent takes — its self-report is unverified until you check the diff, the files, the command history.
- **Isolation**: sandbox permissions and workspace boundaries are the containerization tier — a compromised worker must not reach the host.
- **Supply chain**: skills/plugins/MCP servers are third-party dependencies — scan them (SkillSpector pattern) and treat them like any dependency with an attack surface.

## Failure-handling playbook (theory)

| Failure | System analogue | Correct response |
|---------|----------------|------------------|
| Agent makes a wrong edit | Corrupt write | Review artifacts/diff before promotion; permission layer limited blast radius |
| Session interrupted over SSH | Connection loss | Persisted history + artifacts survive; resume, don't restart |
| Prompt injection in fetched content | Input validation bypass | Permission layer denies writes; hooks log the attempt (inspect) |
| Tool/MCP server down | Service failure | Interface indirection lets you swap the implementation; retry via artifact replay |
| Subagent conflict on shared file | Split-brain | Disjoint worktree/module assignment (sharding) prevents it |

## 7. Coordination & consistency (multi-agent)

- Multiple subagents form a distributed system: decide the consistency model before spawning. **One writer per artifact** (disjoint worktree/module); eventual consistency via artifacts otherwise.
- **Leader-election analog**: one agent owns the plan (the leader); workers execute shards. If the leader dies, the plan artifact is the surviving state — a new leader resumes from it. Never let two agents both "own" the same plan (split-brain).
- Conflict resolution is expensive — avoid shared writes entirely (sharding), don't rely on last-writer-wins.
- During a partition (agent unreachable, tool down), decide in advance: continue degraded (availability) or halt (consistency). PACELC applies to agent pools too.

## 8. Resilience at the harness level

- **Circuit breaker**: repeated permission denials and tool failures are signals — stop the agent, review, adjust the policy; never let it retry the same denied action in a loop.
- **Bulkhead**: sandbox permissions and per-project boundaries are the isolation compartments — a compromised or misbehaving agent in one project cannot sink the others.
- **Backpressure**: scheduled tasks/cron sidecars must respect limits — don't pile unbounded background work onto the harness.
- **Timeouts & verification**: agent self-reports are unverified claims (redundancy theorem) — check diffs, files, command logs before promoting any result.
- **Error budgets**: track agent task success rate and verification pass rate as SLIs with an SLO; exhaust the budget → halt new automation, fix the harness.

## 9. Observability & audit

- **Hooks = the tracing layer**: inspect hooks capture what the agent saw, decide hooks what it chose, transform hooks what it did. Wire them before you need them.
- **Artifacts = the audit log**: plans, reviews, diffs are replayable evidence — the write-ahead log of agent operations. Export sessions, tag artifacts with task lineage.
- **Metrics**: agent runs, tool failures, permission denials, cost — the golden signals of the harness (latency, traffic, errors, saturation).

## 10. Evaluating Antigravity-built systems & harness config

Load `system-design-theory` §9 for the full framework. Evaluation targets:

1. **Harness config review**: least privilege (can the agent do more than the task needs?), sandbox scope, hook coverage, third-party skill/MCP supply chain (scanned?).
2. **Delivered system review**: six-pillar lens on artifacts (reliability, security, performance, operability, cost, sustainability) — standard code review with extra weight on *security* because the agent acted autonomously.
3. **Process review**: did plans/artifacts show reasoning, or did the agent tunnel to a result? Is the session history exportable/auditable?
4. **Improvement planning**: next iteration from error-budget burn (SLOs on agent success rate), hook gaps, permission drift — not vibes.
5. **Pattern-match against reference architectures**: before accepting agent output as done, verify the produced architecture follows the solved patterns in `system-design-theory` §11 / `references/case-study-patterns.md` — fanout decisions, cache keys (stable, coarse), dedupe-by-ID, per-tier failure matrices, batch-built indexes.

## Anti-patterns

- Giving the agent blanket write permission because it's easier (no least privilege).
- Treating agent self-reports as fact without reviewing artifacts (no redundancy).
- One giant conversation for a task that needs parallel workers (vertical scaling).
- Subagents sharing mutable files (unsharded writes).
- No hooks/observability — discovering failures in production instead of at the inspect hook.

## Verification checklist

- [ ] Permissions scoped to the minimum the task needs (least privilege)
- [ ] Hooks in place to inspect/validate before and after critical actions (observability + middleware)
- [ ] Work partitioned across subagents by disjoint slices (sharding)
- [ ] Outputs materialized as artifacts for replay/audit (durability/message passing)
- [ ] Third-party skills/tools/MCP servers scanned before use (supply chain)
- [ ] Agent claims verified against diffs/files/logs before reporting done (redundancy)
