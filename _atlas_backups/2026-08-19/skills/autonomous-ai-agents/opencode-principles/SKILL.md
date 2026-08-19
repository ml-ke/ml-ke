---
name: opencode-principles
description: "System design theory for orchestrating OpenCode multi-agent coding work: statelessness, redundancy, caching, sharding, decoupling, load balancing. Theory companion to the opencode skill — load when architecting agent workflows, not for CLI mechanics."
version: 1.2.0
author: ATLAS
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, OpenCode, System-Design, Theory, Orchestration]
    related_skills: [opencode, antigravity-principles, atlas-principles, system-design-theory]
---

# OpenCode Orchestration Principles

The theory behind orchestrating OpenCode as a multi-agent worker pool. For concrete commands, model availability, and CLI mechanics, load the `opencode` skill. This skill is the *why* — the five theorems from "Scale from Zero to Millions of Users" mapped onto delegating coding work. When you architect a multi-agent workflow, check it against these theorems, not against yesterday's commands.

## The Five Theorems

1. **Keep components stateless** — a delegated task must be a complete, self-contained request.
2. **Build redundancy at every tier** — no single point of truth, no single point of failure.
3. **Cache aggressively, invalidate honestly** — reuse what you already know; treat every cache as TTL'd.
4. **Shard by key** — partition work by a stable, evenly-distributing key; avoid hot shards.
5. **Decouple everything** — stages talk through durable artifacts, never shared memory.

## 1. Statelessness — the task prompt is the request

A stateless web tier scales horizontally because no server remembers you. A stateless agent task is the same: the prompt must carry everything the worker needs (goal, context, constraints, file paths, acceptance criteria). Consequences:

- **No cross-session dependencies** — task B must never rely on task A's in-memory state. If B needs A's output, A must have *materialized* it (file, commit, artifact).
- **Kill-and-resume safety** — any session can be killed and restarted with zero data loss because all state lives in the repo/artifacts, not the agent.
- **Stateful things belong in a shared datastore** — the filesystem/git is the session store, exactly like moving session data out of web servers into a shared store to enable auto-scaling.

## 2. Redundancy — verify, fall back, never trust self-reports

- **Every agent self-report is an unverified claim.** The equivalent of a single master database: if it's the only source of truth and it lies, everything downstream is corrupt. External verification (read the files, run the tests, exercise the happy path) is the slave replica that keeps read-path honest.
- **Model availability is a single point of failure.** Providers disable models without notice. A fallback chain (auto-picking, auto-failing-over) is the redundant tier that absorbs the failure.
- **N+1 for critical work** — run a second independent confirmation (different model, different worker, or your own review) for anything that will be reported as fact.
- **Redundancy in evidence** — a result confirmed two independent ways is the only kind worth reporting as done.

## 3. Caching — know where your caches are and their TTL

- **Model availability cache (6h)** — a write-through cache with a TTL. Its only correctness rule: *stale cache must be invalidated by a live probe before critical runs*. Docs are not proof (the docs-lie rule) — the live test is the source of truth, the cache is just a speedup.
- **Skill bridge** — curated methodology pre-positioned in the worker's skill store is a content cache: it avoids recomputing methodology per task.
- **Lesson bank** — durable cross-session cache of outcomes ("model X died", "this pattern works"). Read it before recomputing; write to it after learning.
- **Cache invalidation discipline** — a cached fact that contradicts a live observation must be evicted immediately, not defended. This is the agent version of the "docs lie" rule.

## 4. Sharding — partition work by a good key

- **The sharding key decides everything.** Split a codebase by module, domain, or dependency boundary — a key that distributes work evenly. Bad keys create hot shards (see: celebrity problem).
- **One session per shard, disjoint ownership** — parallel sessions each own a separate workdir/worktree. Sharing one mutable directory across parallel workers is unpartitioned data: it corrupts like two masters writing the same rows.
- **The celebrity problem is real in code too** — one module (auth, billing) attracts disproportionate traffic/changes. When a shard overheats, split it further (allocate a shard per celebrity), don't just add more requests to the same worker.
- **Resharding is deliberate** — when scope grows, rebalance tasks with a stable mapping (consistent-hashing mindset): most tasks stay put, only a few move, so in-flight work survives.

## 5. Decoupling — artifacts are your message queue

- Producers (task definitions) and consumers (coding sessions) never share mutable state. The interface between stages is **durable artifacts**: files, commits, worktrees, PRs. That is the message queue — it buffers, it survives crashes, it allows replay.
- Decoupled stages each scale on their own axis: the planner tier (decomposition, verification, reporting) scales differently from the worker tier (heavy bounded coding). Same as web tier vs data tier.
- **Retry without cascade** — a failed consumer doesn't corrupt the producer; the message (artifact) is still there, a fresh consumer can pick it up.

## 6. Load balancing & capacity (back-of-envelope first)

- Distribute tasks across sessions by size, model, and dependency graph — the load balancer sends traffic to healthy servers.
- **Estimate before delegating** — the 100-line rule, context-cost math, expected duration. Back-of-envelope estimation prevents vertical-scaling mistakes (one mega-prompt doing everything = adding RAM to one server — it hits limits and has no redundancy).
- Health checks: poll logs, don't blind-wait. A worker that stops responding is a dead server — reroute.

## Failure-handling playbook (theory)

| Failure | System analogue | Correct response |
|---------|----------------|------------------|
| Worker dies mid-task | Slave DB offline | Redirect reads to another worker; restart with durable artifact |
| Planner/you interrupted | Master DB down | The artifact queue survives; resume from last materialized state |
| Model list stale | Cache cold | Live probe to warm it; fallback chain absorbs the miss |
| Module too big | Shard exhausted | Reshard with stable mapping (consistent hashing) |
| Worker produces garbage | Corrupt replica | Never promote it; verify everything before promotion |

## 7. Advanced resilience — retries, circuit breakers, bulkheads

- **Idempotency is the unlock**: a delegated task must be retry-safe. If a worker dies mid-task, re-running it must produce the same result (or artifacts make the difference detectable). Prompt for idempotency: state what exists, what to create, what to skip.
- **Retry with backoff + jitter**: provider failures are transient — retry, but never in lockstep. All parallel workers retrying simultaneously is a thundering herd against the provider (retry storm).
- **Circuit breaker**: a model that fails N consecutive times is *open* — stop trying it, evict it from the model cache, use the fallback chain; half-open probe after cooldown. This is the difference between "model died" and "we kept hammering a corpse".
- **Bulkhead**: isolate workstreams by workdir/worktree AND by model. One poisoned stream (a degrading model) must not consume the shared retry budget of the others.
- **Timeout every delegation** — a hung agent is worse than a failed one. Poll with patience, kill with a deadline.

## 8. Consistency model for parallel agents

- Decide and *state* the consistency model of your worker pool: **eventual** (artifacts converge; stale reads acceptable — fine for most code tasks), **strong** (one writer per artifact; ownership-based locking), **quorum** (N independent workers must agree — the verification pattern).
- Same rule as data sharding: **one writer per file/shard**, disjoint workdirs, no shared mutable state. Two agents writing the same file is split-brain; resolving it via last-writer-wins (git) is the same dangerous default as LWW in databases — avoid the need, don't repair the damage.
- When a worker is unreachable (partition), decide in advance: does the pipeline continue (availability) or stop (consistency)? PACELC: most pipelines pick availability during partitions + eventual consistency in the normal case.

## 9. Observability & capacity

- **Metrics**: sessions, tokens, cost per task (opencode stats). **Logs**: process logs are the audit trail. **Traces**: task lineage — which prompt produced which files (name sessions, tag artifacts).
- **Backpressure**: don't spawn unbounded parallelism. Queue tasks with a limit; when workers are saturated, hold new work — the bounded-queue + 503 pattern.
- **Back-of-envelope**: cost and context are capacity. Estimate tokens/context per task before delegating (the 100-line rule); a mega-prompt is a vertical-scaling mistake.
- **Error budgets for the pipeline**: define SLIs (task success rate, verification pass rate) and an SLO; when the budget is exhausted, stop shipping new task types and fix the orchestration.

## 10. Evaluating systems built with OpenCode (success criteria)

Load `system-design-theory` §9 for the full framework. Evaluate the *artifacts*, not the process:

1. **Six-pillar lens on delivered code**: reliability (tests, error handling, retries, timeouts), security (least privilege, secrets, input validation), performance (caching, query shapes), operability (logs, config, deployment), cost (unnecessary dependencies/amplification), sustainability.
2. **The 10x test**: what breaks at 10× users/data/load? Did the worker build for today or for the ceiling?
3. **Idempotency & retries**: can delivered components be retried safely? Timeouts on every external call?
4. **Trade-off ledger**: did the code document its decisions? If not, reconstruct them and judge whether they were *made* or *accidental*.
5. **Improvement planning**: next iteration = error-budget burn, SPOFs found in review, 10x ceiling — not vibes.
6. **Pattern-match against reference architectures**: compare the delivered system to the case-study patterns in `system-design-theory` §11 and `references/case-study-patterns.md` (cache-first reads, fanout push/pull, chunked/resumable transfers, dedupe-by-ID, batch-built serving indexes). A design that reinvents a solved pattern — or skips one (no dedupe, unstable cache keys, unbounded retries) — is a review finding.

## Anti-patterns (what the theorems forbid)

- Session coupling: task B depends on A's in-memory state (stateful tier).
- Trusting self-reports without verification (no redundancy).
- Monolith delegation: one giant prompt for everything (vertical scaling).
- Shared mutable workdir across parallel sessions (unsharded data).
- Recomputing what the lesson bank already knows (cache miss every time).
- Defending a stale cached fact against a live observation (no invalidation).

## Verification checklist

- [ ] Every task is self-contained — kill any session and nothing is lost (stateless)
- [ ] Every claim that will be reported is independently verified (redundancy)
- [ ] Known facts came from cache/lesson bank before recompute, and caches were live-probed when critical (caching)
- [ ] Work is partitioned by stable, even keys; no shared mutable dirs (sharding)
- [ ] Stages communicate via durable artifacts, not memory (decoupling)
- [ ] Capacity estimated before delegating; heavy work horizontally scaled (load balancing / back-of-envelope)
