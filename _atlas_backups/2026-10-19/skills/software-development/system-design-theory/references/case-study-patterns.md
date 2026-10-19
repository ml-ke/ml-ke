# Case Study Patterns — Real Designs Distilled

Per-design digests from liquidslr/system-design-notes ch. 6–16 (Alex Xu–style). For each: the problem, key decisions, the **generalizable pattern**, and the trade-offs. The cross-cutting meta-patterns live in SKILL.md §11 — this file is the per-design detail. Use both when pattern-matching a system under review or design.

## 1. Distributed Key-Value Store (Cassandra/Dynamo style)
- **Partitioning**: consistent hashing + virtual nodes (vnode count ∝ server capacity) for even, heterogeneous distribution.
- **Replication**: N replicas walking clockwise from the key's position; replicas placed across data centers.
- **Consistency**: quorum — write to W replicas, read from R; **W + R > N ⇒ strong consistency** (N=3, W=R=2 typical). Tuning: R=1, W=N → fast reads; W=1, R=N → fast writes. W/R/N is a latency-vs-consistency knob.
- **Conflict detection**: vector clocks ([server, version] pairs) → sibling versions; resolution is application/client logic (never silent LWW for correctness-critical data).
- **Failure detection**: gossip protocol — nodes exchange heartbeats with random peers; **at least two independent sources required to mark a node down**.
- **Temporary failures**: sloppy quorum (first W healthy nodes take over) + hinted handoff (offline node catches up on recovery).
- **Permanent failures**: Merkle trees — compare root hashes, recurse to find divergent buckets, sync ONLY the inconsistent data.
- **Write path**: commit log → memory cache → flush to SSTable. **Read path**: memory → Bloom filter → SSTables.
- **Pattern**: decentralized ring, no coordinator, no SPOF, automatic scale. Trade-off: eventual consistency unless quorum raised; conflict resolution pushed to the client.
- **Why it matters for review**: any "highly available KV" claim should answer W/R/N, conflict strategy, and failure detection — if those are undefined, the design is a wish.

## 2. Unique-ID Generator (Twitter Snowflake)
- Options rejected: multi-master auto-increment (doesn't scale, not time-ordered), UUID (128-bit, unordered), ticket server (SPOF).
- **Snowflake**: 1 sign + 41-bit ms timestamp + 5-bit DC + 5-bit machine + 12-bit sequence = 64-bit, time-sortable, decentralized, 4096 IDs/ms/machine, no coordination.
- NTP for clock sync; tune section sizes per use case; ID generation is mission-critical → redundancy/failover.
- **Pattern**: derive globally-unique, orderable IDs from *uncoordinated local state* (time + node id + counter). Trade-off: clock-skew sensitivity.
- **Why it matters**: when a system needs ordered IDs at scale, snowflake-style beats central sequences; sequence bits are capacity planning.

## 3. URL Shortener (TinyURL)
- **Redirect**: 301 (permanent — browser caches, loses click analytics) vs 302 (temporary — passes through for analytics). Choosing the status code IS a product decision.
- **ID → Base62** (7 chars ≈ 3.5T URLs, collision-free, needs an ID generator, sequential IDs can be enumerated → security concern) vs **hash + collision resolution** (fixed length, collisions possible, use Bloom filters for dedupe).
- Read path is cache-first (10:1 read:write); rate limiting per IP for abuse; stateless web tier + replicated/sharded DB.
- **Pattern**: encode a unique ID rather than hash the content when you control the namespace; always decide cache-first reads when reads dominate.

## 4. Web Crawler (Googlebot)
- **URL frontier** = FIFO queues; **politeness**: one request per host at a time — host→queue mapping, worker thread per queue, delay between downloads; **priority**: front queues (priority) separate from back queues (politeness); **freshness**: recrawl by update history/importance.
- Robustness: consistent hashing for load distribution, error handling, data validation; **spider traps** (URL length limits); dedupe via content hashing; robots.txt compliance; DNS cache + geo-distributed downloaders + short timeouts.
- **Pattern**: separate *what to do* (priority) from *how politely* (per-host queues); the frontier is a sharded queue with per-partition rate limits.
- **Why it matters**: this is the canonical "bounded, polite, resumable batch worker pool" — same shape as any large scraping/recon pipeline.

## 5. Notification System
- Single server = SPOF → **horizontal scaling + message queues + workers** (producers → queue → workers → third-party APNS/FCM/Twilio/SendGrid).
- **Reliability**: persist events + retry + **dedupe by event ID** (at-least-once delivery made safe).
- Rate limit per user (don't spam); templates; per-channel opt-in/out settings; monitor queue depth to scale workers; security via AppKey/AppSecret.
- **Pattern**: queues decouple trigger services from delivery; dedupe-by-ID is the universal idempotency pattern for fan-out delivery.

## 6. News Feed
- **Fanout on write** (push to friends' feeds at publish: real-time, expensive for high-connection users) vs **fanout on read** (pull at request: cheap for inactive, slow) vs **hybrid** (push for normal users, pull for celebrities — the celebrity special-case AGAIN).
- Fanout pipeline: fetch friend IDs (graph DB) → filter (mute/share settings from cache) → queue → workers append post IDs to per-user feed caches (ID-only, configurable recency limit).
- **5 cache layers**: news feed (post IDs), content, social graph, action (likes), counter. Retrieval hydrates IDs → details.
- **Pattern**: cache the *index* (IDs), hydrate the *content* on demand; write amplification of fanout is bounded by the hybrid push/pull split.

## 7. Chat System
- **Protocol**: WebSocket (bi-directional persistent) vs polling/long-polling (wasteful). Send via HTTP or WS; receive via WS.
- Tier split: stateless services (login/profile) + **stateful chat servers** (hold WS connections) + service discovery (ZooKeeper) picking best server by geography/capacity + presence servers (heartbeats, threshold = offline).
- **History in a KV store** (horizontal scale, low latency, better than relational for long tail); **local sequence numbers per channel** (not global) — ordering scoped to where it matters; multi-device sync via per-device `cur_max_message_id`.
- Group chat: copy message to each recipient's inbox (simple, expensive for large groups).
- Presence: heartbeat + pub/sub fanout over per-pair channels.
- **Patterns**: keep connection-holding state isolated behind service discovery; scope ordering locally; KV for append-heavy history.

## 8. Search Autocomplete
- **Trie** with frequency at nodes; **cache top-k at each node** (no subtree traversal on query); cap prefix length (50).
- **Batch, not real-time**: append-only analytics logs → aggregators → workers rebuild the trie weekly; serving from trie cache (memory) + trie DB (document-store snapshot or KV prefix→node).
- Shard trie by prefix ranges (a–m, n–z), sub-shard hot prefixes; filter layer removes harmful suggestions async.
- **Pattern**: derived/serving structures are rebuilt in batch from logs — the serving index is a disposable cache, never written per-query.

## 9. YouTube (video platform)
- **CDN serves streaming** (edge, MPEG-DASH/HLS); API servers handle everything else; uploads go to blob storage in parallel with metadata updates.
- **Transcoding as a DAG**: preprocessor splits video into GOP chunks → DAG scheduler stages tasks (encode, thumbnail, watermark) → resource manager (task/worker/running queues + scheduler) → temp storage for retry. Parallelism from the DAG, not sequential pipelines.
- Uploads: chunked/resumable, pre-signed URLs (authz), CDN as upload hub.
- **Cost engineering**: CDN only for popular videos; encode-on-demand for rare; regionalize; custom CDNs. $150k/day CDN cost estimate drove the design.
- Errors: recoverable → retry; non-recoverable → abort with code.
- **Patterns**: heavy async processing = DAG of tasks behind queues with persisted intermediate state; cost tiers by popularity; pre-signed URLs are the standard authz for direct-to-storage uploads.

## 10. Google Drive (file sync)
- **Blocks**: files split into 4MB blocks with hashes → dedupe at block level, compress, encrypt; blocks in S3, metadata in relational DB (user/file/block/version tables) + cache.
- **Delta sync** (transfer only changed blocks); resumable upload; long-poll notification + offline backup queue for sync; cold storage for inactive files; limited revisions.
- **Conflict resolution**: first-writer-wins + save both copies for manual merge.
- **Failure matrix**: LB failover, block server task reassignment, master promotion, cross-region fetch on storage failure.
- **Pattern**: content-addressed blocks make dedupe/delta trivial; the metadata DB is the real system of record; every tier has an explicit failure story.

## 11. Proximity Service (Yelp-style)
- Back-of-envelope first: 100M DAU × 5 searches / 86400 ≈ **5,000 QPS**.
- Spatial indexing options: 2D scan (bad — one-dimensional indexes can't do two dimensions), even grid (uneven density), **geohash** (hierarchical string; boundary problems → search 8 neighbors; precision table by radius), **quadtree** (in-memory, k-nearest, rebuild = minutes → incremental rollouts, cache invalidation on updates), **Google S2** (Hilbert curve cells, geofencing, min/max level + max cells).
- Scale: shard business table by business ID; geohash index with read replicas (fits one server — don't shard what fits).
- **Cache: key = geohash, value = business IDs** — NOT raw GPS coords (inaccurate, moving). Parallel Redis calls; hydrate details from a second cache; sort by distance.
- Updates batch-processed daily (no real-time writes); multi-region deploy.
- **Patterns**: index choice is a query-type × update-frequency trade-off; cache keys must be stable and coarse; parallelize fan-out reads.

## How to use this file
- **Designing**: pick the closest reference, then steal its solved patterns (frontier, fanout, blocks, DAG, geohash cache).
- **Reviewing**: run the system against its reference architecture — a missing solved pattern (no dedupe-by-ID, no per-host politeness, cache keyed on unstable data) is a finding.
- **Hunting (bug bounty)**: the architecture tells you where authz and data live — metadata/API tier, service discovery, pre-signed URLs, fanout workers, cache keys, block/delta sync endpoints, presence servers.
