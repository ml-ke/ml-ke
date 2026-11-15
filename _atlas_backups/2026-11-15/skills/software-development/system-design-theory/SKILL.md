---
name: system-design-theory
description: "Use when designing, implementing, scaling, reviewing, or planning improvements to any system or agent workflow — even small ones. Master knowledge base: estimation, CAP/consistency, resilience patterns (circuit breaker, bulkhead, saga, outbox), caching, observability/SLOs, 12-factor, reference architectures (case studies), and the evaluation framework (six-pillar review, design checklist, 10x test). Load before writing architecture, judging whether a project succeeded, or deciding what to improve next."
version: 1.1.0
author: ATLAS
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [system-design, distributed-systems, architecture, evaluation, theory]
    related_skills: [opencode-principles, antigravity-principles, atlas-principles]
---

# System Design Theory — Master Knowledge Base

The complete theoretical core for designing, evaluating, and improving systems. This is the *curriculum*; the tool-specific skills (`opencode-principles`, `antigravity-principles`, `atlas-principles`) map these theorems onto agent workflows. Load this skill whenever you are **implementing a system, evaluating whether a project succeeded, or planning future improvements** — then apply Section 10 (Evaluation Framework) to the concrete situation.

Core mental model: **every design is a set of trade-offs, not a list of best practices.** There is no free lunch — CAP exists, consistency costs latency, redundancy costs money, decoupling costs complexity. The job is to make the trade-off *explicit and reversible*.

## 1. Foundations — estimate before you build

### Power-of-two & latency tables (memorize)

| Unit | Value |
|------|-------|
| 2^10 | 1 K |
| 2^20 | 1 M |
| 2^30 | 1 G |
| 2^40 | 1 T |
| 2^50 | 1 P |

| Operation (2020s order of magnitude) | Time |
|------|------|
| L1 cache ref | ~0.5 ns |
| L2 cache ref | ~7 ns |
| RAM ref | ~100 ns |
| SSD random read | ~150 µs |
| Network round trip (same DC) | ~0.5 ms |
| Disk seek | ~10 ms |
| Network round trip (cross-continent) | ~150 ms |
| Read 1 MB sequentially from SSD | ~1 ms |
| Read 1 MB from network | ~10 ms |
| Read 1 MB from disk | ~30 ms |

### Back-of-envelope recipe
1. State assumptions: DAU, requests/user/day, data per request, read:write ratio.
2. Convert to per-second: `req/s = (DAU × req/day) / 86400`; multiply by 2–5 for peak.
3. Convert bytes to bandwidth: `req/s × bytes/req`.
4. Size storage: `users × data/user × retention × replication factor`.
5. Sanity-check against latency table — is the target physically possible?

### Availability (nines)
| Uptime | Downtime/year |
|--------|---------------|
| 99% | 3.65 days |
| 99.9% | 8.77 h |
| 99.99% | 52.6 min |
| 99.999% | 5.26 min |

### Performance vs scalability vs latency vs throughput
- **Performance** = low latency per request. **Scalability** = handle more load without degrading latency. A fast car ≠ a truck that carries more.
- **Latency** = time for one operation. **Throughput** = operations per second. Latency-bound (single user) and throughput-bound (system) are different problems.

## 2. Consistency & data

### CAP / PACELC
- CAP: Partitioned systems must choose Availability or Consistency. **During a partition you cannot have both.**
- PACELC (the honest version): **if Partition, choose A or C; Else, choose Latency or Consistency.** Most systems pick latency in the normal case (eventual consistency) and consistency during partitions — the opposite of the naive reading.
- Agent implication: when a worker/tool is unreachable, decide *now* whether the pipeline continues (availability) or stops to stay correct (consistency).

### Consistency models
- **Strong (linearizable)** — reads see the latest write. Costs latency/availability; needs quorum or single leader.
- **Eventual** — replicas converge given time. Cheap, highly available; may serve stale reads.
- **Weak/causal** — between the two; causal ordering without global order.
- **ACID vs BASE** — ACID (atomicity, consistency, isolation, durability) for correctness-critical data (money, auth); BASE (basically available, soft state, eventual consistency) for scale (feeds, counters, presence).

### Replication topologies
- **Single-leader**: all writes to one node, reads from replicas. Simple, strong-ish consistency; leader is a SPOF, failover is hard (lost writes risk).
- **Multi-leader**: writes to several. Low latency per region; conflict resolution is the price (last-writer-wins is dangerous).
- **Leaderless (quorum)**: write to N nodes, read from R, success if R+W > N. Tolerates node loss; needs conflict resolution (version vectors, LWW).
- **Synchronous vs async replication** — sync guarantees durability but blocks on failure; async risks losing the last writes on failover. Log-based and trigger-based replication differ in consistency guarantees.

### Partitioning (sharding) — data tier
- **Hash partitioning** (by key hash) — even distribution, but kills range queries. **Range partitioning** — good range scans, bad hot spots.
- **Consistent hashing**: nodes on a ring, key assigned to next node clockwise; adding/removing a node moves only ~1/N of keys — minimal resharding. **Virtual nodes** smooth uneven distribution.
- **Hot spots (celebrity problem)**: popular keys overload one shard. Mitigate: split hot shards, add local caches, shard per-celebrity, replicate hot keys.
- **Sharding keys must distribute evenly** — the key choice IS the design. A bad key means you reshard under load (the worst time to do it).
- Joins and transactions across shards are hard → **denormalize** (duplicate data into query-shaped form) and accept write amplification.

## 3. Reliability & resilience patterns

### Retries, timeouts, backoff
- **Timeout every call** — a hung dependency is worse than a failed one.
- **Exponential backoff + jitter** — retry at 1s, 2s, 4s, 8s with random jitter; without jitter, retries synchronize into thundering herds (retry storm).
- **Retry only idempotent operations** — retrying a non-idempotent POST duplicates the side effect. (See idempotency keys, §7.)
- **Retry budget** — cap total retries; over-retrying during an outage is an amplification attack on your own system.

### Circuit breaker
- Track failure rate; when it crosses a threshold, **open the circuit** — fail fast for a cooldown window instead of hammering the dead dependency. Half-open state lets a few probes through to test recovery.
- Prevents cascading failure: the dependency is down, and you're not making it worse.

### Bulkhead (isolation)
- Partition resources by client/workload (like ship compartments). One failing consumer cannot exhaust the pool for everyone else.
- Connection pools, thread pools, per-tenant limits are bulkheads.

### Fallback & graceful degradation
- Fallback: serve a cached/stale/default response when the live path fails (degrade to last-known-good).
- Graceful degradation: when overloaded, drop non-critical features (recommendations, personalization) before critical ones (checkout, auth).
- **Prioritize**: decide in advance what degrades first. Do not discover the priority order during the incident.

### Rate limiting (the algorithms)
- **Token bucket** — tokens refill at fixed rate, requests consume; allows bursts, memory-efficient. Most popular.
- **Leaking bucket** — FIFO queue processed at fixed rate; smooth outflow, no bursts.
- **Fixed window counter** — simple, but burst at window edges can double the limit.
- **Sliding window log** — exact but memory-heavy. **Sliding window counter** — hybrid approximation.
- Distributed rate limiting needs shared state (Redis) or per-node bucketing with sync; where to enforce: client (unreliable), server (control), API gateway (microservices).

### Backpressure
- When a consumer can't keep up, propagate the pressure upstream instead of buffering unboundedly: bounded queues → drop with 503 → client retries with backoff. Unbounded queues turn a slow consumer into an OOM.

### Message queues & delivery semantics
- **At-most-once**: may lose messages (fast, no retry). **At-least-once**: may duplicate (retry; needs idempotent consumer). **Exactly-once**: impossible to guarantee end-to-end in general — implement as *at-least-once + idempotent consumer* (dedupe keys, outbox).
- **Dead-letter queue**: after N failed deliveries, park the message for manual inspection — never silently drop.

## 4. Distributed transactions & events

### 2PC vs Saga
- **2PC**: all-or-nothing across participants; correct but blocking (coordinator failure freezes everyone) — use only for small, high-correctness transactions.
- **Saga**: chain of local transactions with compensating actions (order created → payment taken → if shipping fails, refund). Eventual consistency, high availability, no global lock. The production answer for most distributed writes.

### Outbox pattern
- Write the DB change and the event in the **same local transaction** (outbox table), then a relay publishes to the queue. Guarantees the event is never lost even if the queue write fails. The missing piece of most "send event after saving" bugs.

### Event sourcing & CQRS
- **Event sourcing**: store the event stream as the source of truth; current state is a projection. Auditability, replay, time travel — at the cost of read-model complexity.
- **CQRS**: separate write model from read model — optimize each independently (eventual consistency between them). Use when reads and writes have very different shapes/loads.

### Idempotency (the most underrated correctness tool)
- An operation is idempotent if repeating it produces the same result. Implementation: client sends `Idempotency-Key`; server stores the key → response and replays it on duplicates.
- Makes retries safe, which makes every resilience pattern safe. **Design for idempotency first**; it unlocks retries, queues, and at-least-once semantics.

## 5. Caching

### Strategies
- **Cache-aside** (lazy): app checks cache, on miss reads DB and populates cache. Simple; cache misses on cold start (stampede risk).
- **Read-through**: cache itself fetches from DB on miss — cache owns the policy.
- **Write-through**: write to cache AND DB synchronously — always consistent-ish, higher write latency.
- **Write-back** (write-behind): write to cache, flush to DB async — fast writes, risk of data loss on cache crash.
- **CDN** = cache at the edge: static assets, geo-distributed; watch cost, invalidation, fallback to origin.

### Invalidation & pitfalls
- TTL is the simple invalidation; too short = useless, too long = stale. Event-driven invalidation (publish DB change → evict) is the correct one for hot data.
- **Cache stampede/thundering herd**: many simultaneous misses on an expired hot key → all hit the DB. Mitigate: request coalescing (single flight), locks, early recompute before expiry, jittered TTLs.
- **Cache as a SPOF**: a cache tier is not durable — never design correctness around it. Multiple cache servers across zones.

## 6. APIs & interfaces

- **Idempotency keys** on mutating endpoints (see §4).
- **Pagination**: cursor-based for large, changing datasets (offset pagination breaks under inserts).
- **Versioning**: URL or header versioning; never break clients silently. Deprecation policy documented.
- **Webhooks**: sign payloads (HMAC), retry with backoff, dedupe by event ID, respect ordering requirements, expose replay.
- **Rate limits surfaced**: return `Retry-After`, `X-RateLimit-*` headers so clients can behave.
- **Timeouts & deadlines propagate** downstream (grpc deadlines, HTTP timeout headers).

## 7. Observability & operations

### The three pillars
- **Metrics** (what): counters, gauges, histograms — the four golden signals: **latency, traffic, errors, saturation** (USE: utilization, saturation, errors; RED: rate, errors, duration).
- **Logs** (what happened): structured, request-ID correlated.
- **Traces** (where time went): distributed tracing across services — one trace per request.

### SLOs, SLIs, error budgets
- **SLI** = measured indicator (p99 latency, error rate). **SLO** = target (p99 < 200ms, 99.9% success). **Error budget** = 100% − SLO; the amount of unreliability you may spend.
- Error budgets govern *shipping*: when the budget is exhausted, stop shipping features and fix reliability. This is the mechanical link between reliability and improvement planning.
- **Alert on SLO burn, not on symptoms** — alert when you're burning budget faster than expected, not on every spike.

### Deployment & release
- **Rolling**, **blue/green** (two full environments, instant switch), **canary** (ship to 1% → 10% → 100%), **feature flags** (ship code dark, release behavior by flag).
- **12-factor** (implementation-grade principles): one codebase, declare deps, config in env, backing services as attached resources, stateless processes (state in backing store), disposability (fast start/graceful stop), dev/prod parity, logs as streams, admin tasks as one-off processes, port binding, concurrency via process model, build/release/run separation.
- **Infrastructure as code** — environments reproducible from source; no snowflake servers.

## 8. Security & cost

- **Security** (as a design pillar, not a feature): least privilege, defense in depth, secrets management (never in code/config files), zero trust (verify every call, not just the perimeter), input validation at every trust boundary, encryption in transit and at rest, audit logging.
- **Cost**: right-size before scale-out; know your unit economics (cost per request/user); caching and tiering reduce cost; **sustainability** (6th WAF pillar): efficiency is also carbon efficiency.

## 9. The evaluation framework (the point of this skill)

Use this whenever asked to evaluate a system, judge whether a project succeeded, or plan improvements. Run it as a structured review, not an opinion.

### A. The six-pillar lens (AWS Well-Architected + sustainability)
1. **Operational excellence** — Can it be operated? Runbooks, observability, automated rollback, deployment automation? Toil is a design defect.
2. **Security** — Least privilege? Secrets safe? Input boundaries validated? Data encrypted? Audit trail?
3. **Reliability** — Single points of failure? Retries/backoff/circuit breakers? Backup/restore tested? Recovery time known? SLOs defined and measured?
4. **Performance efficiency** — Right-sized resources? Caching where hot? Latency within target? Scale-out path tested?
5. **Cost optimization** — Unit economics understood? Waste identified? Pay-for-what-you-use?
6. **Sustainability** — Efficiency; unnecessary amplification?

### B. The design review checklist (10 questions)
1. What breaks at **10× load, 10× data, 10× users**? (the 10x test)
2. What is the **single point of failure** at every tier?
3. What happens when each dependency **times out / returns garbage / is slow**?
4. Are **writes idempotent**? Can retries happen safely?
5. What is the **consistency model**, and is it documented as a decision, not an accident?
6. Is **state** where it should be (backing store) or smuggled into process memory?
7. How is **capacity estimated** (numbers, not vibes)? What's the growth plan?
8. What **degrades first** under load (explicit priority)?
9. Is the system **observable** — can you answer "is it healthy, and why" in 5 minutes?
10. What **trade-offs were made**, and are they recorded (trade-off ledger)?

### C. Measuring success (SLO-driven)
- Success is not "it deployed". Success = **SLIs meet SLOs over a window** while error budget allows shipping.
- Define per project: the 1–3 SLIs that matter (latency p99, error rate, data freshness, uptime), the SLO target, and how the budget is spent.
- Retrospective question: **did we meet the SLO, and what did the error budget tell us?**

### D. Planning improvements (postmortem-driven)
- Every incident/regression → **blameless postmortem**: timeline, root cause (not proximate cause), action items with owners, verify the fix.
- Prioritize improvements by: error-budget burn > toil reduction > scalability ceiling > cost.
- Re-run the 10x test after each change — the ceiling moves.

### E. The trade-off ledger
- For every significant decision, record: **decision, alternatives considered, why chosen, what it costs, when to revisit**. A system without a ledger is a system whose decisions look like accidents.

## 10. How agents use this skill

- **Implementing**: before writing code, state the constraints and the consistency model; apply 12-factor; make idempotency a default; time out every call.
- **Evaluating**: run the six-pillar lens + the 10-question review on the actual artifacts (code, config, running system). Report verdicts with evidence — never "looks good" without numbers.
- **Planning**: derive the next work items from error-budget burn, SPOFs found in review, and the 10x ceiling — not from vibes or feature lists.

## 11. Reference architectures — patterns from real designs

Distilled from the classic design case studies (KV store, unique-ID, URL shortener, web crawler, notification, news feed, chat, autocomplete, YouTube, Google Drive, proximity service). Full per-design digests: `references/case-study-patterns.md`. The recurring meta-patterns:

1. **The read:write ratio picks the storage** — URL shortener 10:1 read → cache-first + replicas; chat history → KV store; proximity read-heavy → read replicas. Compute reads vs writes before choosing the data layer.
2. **Fanout: push vs pull vs hybrid** — news feed pushes at write for normal users, pulls for celebrities (celebrity special-case); chat copies to inboxes; Drive notifies via long-poll + offline queue. Decide by: real-time requirement × number of consumers × write amplification.
3. **Dedupe by event ID = at-least-once + idempotent consumer** — notification system checks event IDs before sending. Every queue consumer must dedupe.
4. **Chunked, resumable transfers** — Drive 4MB blocks + delta sync; YouTube chunked uploads; transcoding splits GOPs with temp storage for retries. Big transfers are chunked and resumable, always.
5. **Batch pipelines for derived data** — autocomplete rebuilds the trie weekly from append-only logs; proximity batch-processes business updates daily. Logs → aggregators → workers → serving store; the serving index is a rebuildable cache, never written per-query.
6. **Cache keys must be coarse and stable** — news feed caches post IDs (not full content); proximity caches by geohash (not raw GPS coords). Cache the lookup, hydrate the details.
7. **Scoped ordering beats global ordering** — chat uses per-channel local sequence numbers; global Snowflake IDs only where cross-system ordering matters. Strong global ordering is expensive; scope it.
8. **Cost-tiered everything** — YouTube: CDN for popular, encode-on-demand for rare, regionalized; Drive: cold storage + block dedupe + limited revisions. Put data where it's cheapest while still meeting its SLA.
9. **Conflict resolution is a designed decision, never an accident** — KV: vector clocks + app-level resolution; Drive: first-writer-wins + save both for merge; LWW is the default only because it's cheap. State which one you are.
10. **Per-tier failure matrices** — Drive's failure handling: LB failover, block-server reassignment, master promotion, cross-region fetch. Every tier needs a documented "when this dies" story.
11. **Politeness and rate limits shape load** — crawler: one request per host at a time (host→queue mapping); notifications: per-user caps. Control outbound load and inbound abuse.
12. **Service discovery for stateful tiers** — chat's ZooKeeper picks the best server by geography/capacity; stateless tiers just scale behind a load balancer.
13. **Spatial indexing has edge cases** — geohash boundary problems → search neighbors; quadtree rebuild storms → incremental rollouts. Index choice = query-type × update-frequency trade-off.
14. **Back-of-envelope opens every design** — proximity: 100M×5/86400 ≈ 5k QPS; YouTube: CDN cost ≈ $150k/day. The numbers choose the architecture before any code.

## Sources
- liquidslr/system-design-notes (Alex Xu–style chapters), donnemartin/system-design-primer, AWS Well-Architected Framework, Google SRE book (SLI/SLO/error budget), 12factor.net, PortSwigger/DDIA ecosystem for distributed systems theory.
