---
name: atlas-principles
description: "Use when designing, reviewing, or improving the ATLAS learning pipeline itself. System design theory applied to recon → learn → synthesize → verify → report as a multi-tier architecture: redundancy of sources, lesson-bank caching, domain sharding, decoupled stages, error budgets, architecture-aware targeting. Theory companion to atlas-continuous-learning and atlas-lesson-bank."
version: 1.2.0
author: ATLAS
license: MIT
metadata:
  hermes:
    tags: [bug-bounty, learning, system-design, theory, methodology]
    related_skills: [atlas-continuous-learning, atlas-lesson-bank, opencode-principles, system-design-theory]
---

# ATLAS Pipeline Principles

The theory of ATLAS as a distributed learning system. ATLAS is not one activity — it is a multi-tier pipeline: **recon → learn → synthesize → verify → report**. Each tier is a component with its own scaling axis, and the "Scale from Zero to Millions" theorems apply to the whole pipeline. For the operational how-to (source lists, commands, workflows), load `atlas-continuous-learning`; for durable memory mechanics, load `atlas-lesson-bank`. This skill is the *why* — check pipeline design against these theorems.

## The Pipeline as Tiers

| Tier | ATLAS component | System analogue |
|------|-----------------|-----------------|
| Recon | target sandboxes (`~/Dev/<target>/`), parallel probing | Data tier (ingestion) |
| Learn | continuous-learning resource hub, worldwide sweep | Cache/ingest tier |
| Synthesize | meta-analysis, chaining, impact assessment | Compute tier |
| Verify | pre-submission-verification gates, live PoC | Redundancy gate |
| Report | submission templates, humanizer | Delivery tier |
| Remember | LESSONS.md, memory, skills | Durable cache / lesson bank |

## 1. Redundancy — never trust a single source

- **The meta-analysis rule is the theorem in action**: a conclusion is only as good as the number of independent source types behind it (CVEs, disclosed reports, top-hunter writeups, program rules, live tests). One source = a single replica = a lie propagates. Ten iterations across different source types = the read quorum.
- **Verification is a gate, not a step** — pre-submission gates exist because the pipeline's output (a report) must not ship on self-report. Live PoC = the independent replica confirming the write.
- **Data redundancy too** — the same finding confirmed via two different endpoints/techniques is the only kind worth submitting.

## 2. Caching — the lesson bank is a durable cache with TTL

- **LESSONS.md is the write-through cache** of outcomes: rejections, acceptances, model deaths, target quirks. Read it before recomputing knowledge; write to it after every outcome.
- **Memory (2200-char cap) is the hot cache** — small, fast, evicts under pressure. When it fills, compress to a pointer (`See LESSONS.md`) — eviction policy, not data loss.
- **Skills are materialized views** — frequently-needed knowledge precomputed into loadable procedures, so the hot path doesn't recompute.
- **Cache invalidation is the shallow-conclusion trap**: a cached conclusion (bumper-sticker sized) must be invalidated the moment a counterexample arrives — that's exactly the meta-analysis workflow's job. Serving a stale single-source conclusion is serving a cache you refused to invalidate.

## 3. Sharding — partition knowledge and work by key

- **Domain folders are shards**: `~/Dev/<target>/` sandboxes, `~/Dev/REPORTS/<Target>/` reports, skills per vulnerability class. The shard key is the domain — stable, evenly distributable, and independently scalable (a hot target doesn't block a cold one).
- **Parallel recon across targets (Skoda + Auth0 + Torfs pattern) is horizontal scaling** — several shards queried in parallel, each with its own workers; cross-pollination between shards is the join that sharding makes hard (hence: denormalize lessons into a shared lesson bank).
- **The celebrity problem applies to attention** — popular targets/vuln classes get hot. Shard attention deliberately: cap deep-dives, skip no-bounty shards (effort estimation before deep-dive = back-of-envelope).
- **Stable keys, minimal resharding** — stable skill names and stable LESSONS.md section headings mean existing knowledge doesn't move when new knowledge arrives (consistent-hashing mindset).

## 4. Statelessness — self-contained units of work

- **Every skill is a stateless worker**: it carries its own trigger conditions, steps, and pitfalls. Loading a skill gives a fresh process everything it needs — no dependence on the session that created it.
- **Cron prompts are stateless requests**: they must be self-contained because they run in a fresh session with zero conversation context. A cron job that depends on "last time's" state is a stateful tier that will fail on restart.
- **Sessions are resumable from durable state** — session_search + lesson bank let you rebuild context after a crash, the same way a stateless web tier rebuilds session state from the shared store.

## 5. Decoupling — stages communicate through files

- Recon writes artifacts (findings, notes), learning writes lessons, synthesis writes reports, verification gates them. The artifacts are the message queue: each stage consumes the previous stage's durable output, and no stage holds another's state.
- **A failure in one stage doesn't cascade** — a rejected report (delivery failure) doesn't corrupt the recon data; the artifacts survive and a new submission can be produced.
- **Tier independence** — learning can run while recon is blocked (cron sweeps, weekly studies) because the stages don't hold locks on each other.

## 6. Monitoring, automation & back-of-envelope

- **Automation is the ops tier**: weekly cron (skills-learning, blog-poster) runs the pipeline on schedule; verification checklists gate every output.
- **Observability**: session_search is the queryable log; the weekly report to chat is the dashboard. If you can't explain what ATLAS did last week, the pipeline is unobservable.
- **Capacity planning before work**: check bounty tier before deep-diving an asset (skip No-bounty), estimate iterations before a meta-analysis (10+ source types), estimate scope before recon (big scope = heavy recon). Back-of-envelope prevents vertical-scaling mistakes (one giant deep-dive with no redundancy).

## Failure-handling playbook (theory)

| Failure | System analogue | Correct response |
|---------|----------------|------------------|
| Report rejected as Informative | Write failed validation | Treat as feedback data — cache it in LESSONS.md, adjust the gate, don't retry blind |
| Duplicate finding | Cache hit (someone else cached first) | Speed matters; the cache told you the hot path is saturated |
| Target locked down / dead | Shard offline | Pivot to another shard; reads redirect to healthy shards |
| Memory full | Cache pressure | Evict to lesson bank (compression to pointer) |
| Conclusion "lacking" | Stale cache served | Run meta-analysis: invalidate with 10+ iterations across source types |
| Session interrupted | Connection loss | Resume from session_search + lesson bank (stateless rebuild) |

## 7. Pipeline resilience — idempotency, circuit breakers, backpressure

- **Idempotency**: recon probes and lesson entries must be re-runnable without side effects (anew/dedupe, dated entries, stable skill names). Re-running a session after a crash must not duplicate lessons or corrupt reports.
- **Circuit breaker**: a target that fails N probes is *open* — stop hammering it and pivot (the pivot table is a circuit breaker with a fallback list). Never retry the same dead gateway.
- **Backpressure**: verification gates are admission control — unverified claims must not flow toward submission. A bounded pipeline (recon → gates → report) beats an unbounded one.
- **Retry with backoff**: transient failures (network, rate limits) → retry; persistent rejection → stop, cache the outcome in the lesson bank, change approach.

## 8. Data consistency across sessions

- **Lesson bank = the source of truth** (single-leader model): memory is the hot replica (stale-able, evictable), skills are materialized views, LESSONS.md is the write-ahead log. Write to the leader, read through caches — never let a session become a second leader.
- **Eventual consistency is acceptable** for session context (slightly stale memory is fine; the lesson bank converges). **Strong consistency is required** for submissions (idempotency keys = report IDs, dedupe) and for conclusions presented as fact.
- **Conflict resolution**: two sessions learning different lessons about the same class → the meta-analysis quorum (10+ source types) decides. Never last-writer-wins by recency.

## 9. Pipeline observability & capacity

- **Metrics**: acceptance rate, dupe rate, findings per target, iterations per target — the golden signals of the pipeline (is it healthy, is it improving?).
- **Logs**: session_search is the queryable log; LESSONS.md is the audit log. **Traces**: target → iterations → finding → report → outcome, linkable end to end.
- **Capacity planning**: effort estimation before deep-dive (bounty tier), iteration budgets (the 10+ meta-analysis rule), memory budget (2200-char eviction). The pipeline has physical limits; budget them.
- **Error budgets**: define SLIs (acceptance rate, findings/week) and an SLO; exhaust the budget → stop hunting, fix methodology.

## 10. Evaluating the ATLAS pipeline itself

Load `system-design-theory` §9. ATLAS is a system — review it the same way:

1. **Six-pillar lens on ATLAS**: reliability (do gates catch bad reports? is LESSONS.md backed up?), security (sandboxing, credential handling), performance (time-to-finding), operability (weekly cron, checklists), cost (tokens per finding), sustainability.
2. **SLOs for learning**: SLIs (acceptance rate, findings/week, dupe rate) with an SLO (e.g., ≥30% acceptance); the error budget is the gap. Budget exhausted → stop hunting, fix methodology.
3. **Improvement planning**: next week's work = error-budget burn (which rejections dominate?), SPOFs (what single failure kills the pipeline?), 10x test (10× targets, 10× lessons, 10× reports).
4. **The trade-off ledger for methodology**: every rejected report is a recorded decision with alternatives — that is the meta-analysis workflow.
5. **Architecture-aware targeting**: recognize the target's reference architecture from the case-study patterns (`system-design-theory` §11 / `references/case-study-patterns.md`) — chat → KV + WebSocket + presence; file storage → block servers + metadata DB + delta sync; video → CDN + transcoding DAG + pre-signed URLs; proximity → geohash + cache; news feed → fanout workers + ID caches; notification → queue + dedupe-by-ID — then hunt where the pattern puts the interesting surface: metadata/API tier, service discovery, pre-signed URLs, fanout workers, cache keys, sync/upload endpoints, batch pipelines. The architecture tells you where authz and data live.

## Anti-patterns

- Single-source conclusions presented as fact (no redundancy).
- Cached lesson defended against a live counterexample (no invalidation).
- Deep-diving a hot target while cold shards starve (unbalanced load).
- Cron/skill that depends on previous-session state (stateful worker).
- Stages sharing mutable state instead of durable artifacts (coupling).
- Relearning what LESSONS.md already recorded (cache miss every time).

## Verification checklist

- [ ] Every conclusion backed by multiple independent source types (redundancy)
- [ ] Lesson bank consulted before recompute; outcomes appended after (caching)
- [ ] Work partitioned by stable domain keys; parallel targets sharded (sharding)
- [ ] Every unit of work (skill, cron prompt) self-contained (statelessness)
- [ ] Stages pass durable artifacts, not memory (decoupling)
- [ ] Effort estimated before deep-dive; monitoring via session_search + weekly reports (back-of-envelope / ops)
