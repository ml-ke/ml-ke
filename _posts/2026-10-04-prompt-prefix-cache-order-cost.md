---
title: "Prompt Order Is a Cost Control: Measuring Prefix-Cache Reuse in Standard-Library Python"
date: 2026-10-04 00:00:00 +0300
categories: [AI Engineering, ML Ops]
tags: [prompt caching, prefix caching, kv cache, llm inference, cost optimization, vllm, inference serving]
math: false
image:
  path: /assets/img/cover-prompt-prefix-cache-order-cost.webp
  alt: A prompt drawn as a ruler of 16-token blocks, the static-first lane green and reused with a short red tail while the volatile-early lane shows a red block early and every block after it recomputed
---

## Introduction

Prefix caching is the cheapest optimisation in LLM serving and the easiest to lose by accident. Nothing about the model changes; the bytes of your prompt change order, and the input bill moves by nearly an order of magnitude.

> **The framing**
> This is the reuse side of the cache, not the representation side. [KV Cache Quantization for Long Context](/posts/kv-cache-quantization-long-context/) covered what the cache *costs in memory* when you round it; [vLLM and LLM Serving](/posts/vllm-llm-serving/) covered the paged allocator and listed `--enable-prefix-caching` as a flag; [DeepSeek V4 Peak/Off-Peak Pricing](/posts/deepseek-v4-peak-offpeak-pricing/) showed that a cache-hit input token is priced 30x-50x below a miss and told you to keep stable system prefixes. This post measures the thing all three assume: how much of your prompt gets reused, what kills the hit rate, and how to check it in a test.
{: .prompt-info }

Four layouts of the *same 2,180-token prompt* were replayed against a block-level cache built the way vLLM builds one. Two of them are ordinary prompt engineering mistakes, not exotic ones. The measured spread, in steady state:

- static content first, per-request data last: **2,000 of 2,180 tokens reused (91.7%)**
- per-request data placed before the knowledge base: **176 tokens reused (8.1%)**
- a request-specific value inside the system prompt itself: **0 tokens reused (0.0%)**

On DeepSeek's published off-peak rate for `deepseek-flash`, that is **$0.036 versus $0.327 per 1,000 requests** for the same content, model and answers.

## The cache key is a prefix, not a lookup

Every provider implements the same idea with different bookkeeping. The invariant is that a cache entry is keyed by *everything from the start of the prompt up to a breakpoint*, so one changed byte early invalidates all of it.

Anthropic's documentation states the mechanism plainly: a cache write "includes the block designated with `cache_control`" and the hash is **cumulative**, "so changing any block at or before the breakpoint produces a different hash on the next request." Cache reads then "walk backward" from the breakpoint looking for the longest prefix a previous request wrote, inside a 20-block lookback window with at most four breakpoints. Prefixes are assembled in a fixed order (**tools, then system, then messages**), "a hierarchy where each level builds upon the previous ones."

OpenAI stores key-value tensors, not text, and requires "the entire rendered prefix to match", including the hidden system message, tool definitions and schemas, and conversation history. Caching is on by default for supported models, cached input is "discounted up to 95%", and the response usage block reports the split the billing uses: `cached_tokens` and `cache_write_tokens`.

DeepSeek's on-disk context cache is enabled for every request with no code change, and a hit requires that the request "fully match a cache prefix unit" — units created at the end of the user input and the end of the model output, detected common prefixes, and fixed token intervals on long inputs. That is why `A+B` followed by `A+C` misses, but leaves `A` persisted for a later `A+D`.

Self-hosted, vLLM chooses a hash-based design: the key is `hash(tuple[components])` over the **parent block hash**, the block's own token tuple, and extra hashes for LoRA IDs or image inputs. Its documentation is explicit that "we only cache full blocks", and that SHA256 is recommended over the default Python hash for multi-tenant deployments, at a cost of "about 100-200ns per token (~6ms for 50k tokens of context)".

Every one of those designs shares the same failure mode: the chain. A block's key contains its parent's key, so the first block that differs takes every block behind it down with it.

## A block-level prefix cache in the standard library

The harness below is the smallest thing that reproduces all of it: 16-token blocks (vLLM's example block size), SHA256 keys chained through the parent hash, only full blocks stored, LRU eviction under a capacity. The workload models an assistant with a 2,000-token static prefix (180 tokens of system prompt, 220 of tool schemas, 1,600 of policy text) and 180 tokens of per-request content: a live-data block (date, customer name, balance, ticket id), the ticket itself, and the instruction.

{% raw %}
```python
import hashlib, itertools

BLOCK = 16  # vLLM's default KV block size; cache keys are built per full block


class PrefixCache:
    """Block-level prefix cache with chained hashing (vLLM design: parent hash + block tokens)."""

    def __init__(self, capacity_blocks):
        self.capacity = capacity_blocks
        self.blocks = {}     # key -> block tokens (only full blocks are stored)
        self.used = {}       # key -> logical clock for LRU eviction
        self.clock = itertools.count()
        self.writes = 0

    @staticmethod
    def _key(parent, block):
        h = hashlib.sha256()
        h.update((parent or "root").encode())
        h.update(repr(tuple(block)).encode())
        return h.hexdigest()

    def match(self, tokens):
        """Longest cached prefix, counted in whole blocks."""
        parent, hit = None, 0
        for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
            key = self._key(parent, tokens[i:i + BLOCK])
            if key not in self.blocks:
                break
            self.used[key] = next(self.clock)
            parent, hit = key, hit + BLOCK
        return hit

    def insert(self, tokens, already_cached):
        parent = None
        for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
            block = tokens[i:i + BLOCK]
            key = self._key(parent, block)
            if i < already_cached:
                parent = key
                continue
            if key not in self.blocks:
                self.writes += 1
            self.blocks[key] = tuple(block)
            self.used[key] = next(self.clock)
            parent = key
        while len(self.blocks) > self.capacity:
            victim = min(self.used, key=self.used.get)
            del self.blocks[victim], self.used[victim]


def tokens(n, salt):
    """Stand-in for a tokenizer: n distinct integer token ids for a section.

    Ids come from blake2b, not hash(), so the run is identical on every machine
    (Python randomises string hashing per process).
    """
    return [int.from_bytes(hashlib.blake2b(f"{salt}:{i}".encode(), digest_size=4).digest(), "big")
            % 10 ** 6 for i in range(n)]


SYSTEM, TOOLS, POLICY = 180, 220, 1600      # static: 2,000 tokens = 125 full blocks
VOLATILE, TICKET, INSTR = 48, 112, 20       # per-request: 180 tokens


def prompt(layout, req):
    volatile = tokens(VOLATILE, "volatile")
    volatile[0] = 900000 + req              # date, name, balance, ticket id all change
    ticket = tokens(TICKET, ("ticket", req))
    static_a, static_b = tokens(SYSTEM, "sys"), tokens(TOOLS + POLICY, "body")
    if layout == "A: volatile early":        # system -> live data -> tools -> policy -> ticket
        return static_a + volatile + static_b + ticket + tokens(INSTR, "instr")
    if layout == "B: static first":          # system -> tools -> policy -> live data -> ticket
        return static_a + static_b + volatile + ticket + tokens(INSTR, "instr")
    if layout == "C: timestamp in system":   # live date inside the system prompt itself
        system = tokens(SYSTEM, "sys")
        system[3] = 800000 + req
        return system + static_b + volatile + ticket + tokens(INSTR, "instr")
    raise ValueError(layout)


REQUESTS = 40
print(f"{'layout':<24} {'prompt':>7} {'cold req':>9} {'req 2+ reused':>14} {'hit':>6} {'blocks stored':>14}")
for layout in ["A: volatile early", "B: static first", "C: timestamp in system"]:
    cache = PrefixCache(capacity_blocks=4096)
    first = steady = 0
    for req in range(REQUESTS):
        toks = prompt(layout, req)
        cached = cache.match(toks)
        if req == 0:
            first = cached
        else:
            steady += cached
        cache.insert(toks, cached)
    steady //= REQUESTS - 1
    print(f"{layout:<24} {len(toks):>7,} {first:>9,} {steady:>14,} "
          f"{100 * steady / len(toks):>5.1f}% {cache.writes:>14,}")
```
{% endraw %}

Run it and you get three lines that should be uncomfortable to read if layout A or C resembles your production prompt:

```
layout                    prompt  cold req  req 2+ reused    hit  blocks stored
A: volatile early          2,180         0            176   8.1%          5,011
B: static first            2,180         0          2,000  91.7%            565
C: timestamp in system     2,180         0              0   0.0%          5,440
```

Layout A loses the prefix at block 11, where the live-data block begins: 11 blocks of system prompt survive, and the 1,820 tokens of tools and policy behind it are re-prefilled on every single request. Layout C is the same mistake moved one level up — a timestamp or user name interpolated into the system prompt changes token index 3, block 0 differs, and the chained hash makes the entire prompt unique. Its `blocks stored` count is the tell: 5,440 blocks held for 40 requests, because nothing is ever reused. Layout B pays for the static prefix once and re-reads it 39 times.

## Where prefix hits die

Three failures are worth measuring separately, because each has a different fix. Each block below is self-contained: same cache class, repeated so you can paste it alone.

**Alignment.** Only whole blocks are cacheable, so a static prefix that is not a multiple of the block or increment size leaks tokens on every request. With 16-token blocks, a 2,012-token prefix re-processes 12 tokens per request — 12,000 tokens across 1,000 requests, for a prefix that never changes.

**A single changed token.** The chained hash means the damage is positional, not proportional. Changing token 3 of a 2,000-token prefix drops steady-state reuse to zero; changing token 1,995 costs 16 tokens.

**Capacity and tenancy.** When several tenants share one cache and the capacity cannot hold the whole working set, LRU eviction turns a 90% hit rate into a 0% one — a cliff, not a slope.

{% raw %}
```python
import hashlib, itertools

BLOCK = 16


def tokens(n, salt):
    return [int.from_bytes(hashlib.blake2b(f"{salt}:{i}".encode(), digest_size=4).digest(), "big")
            % 10 ** 6 for i in range(n)]


class PrefixCache:
    def __init__(self, capacity_blocks):
        self.capacity, self.blocks, self.used, self.clock, self.writes = (
            capacity_blocks, {}, {}, itertools.count(), 0)

    @staticmethod
    def _key(parent, block):
        h = hashlib.sha256()
        h.update((parent or "root").encode())
        h.update(repr(tuple(block)).encode())
        return h.hexdigest()

    def match(self, toks):
        parent, hit = None, 0
        for i in range(0, len(toks) - len(toks) % BLOCK, BLOCK):
            key = self._key(parent, toks[i:i + BLOCK])
            if key not in self.blocks:
                break
            self.used[key] = next(self.clock)
            parent, hit = key, hit + BLOCK
        return hit

    def insert(self, toks, cached):
        parent = None
        for i in range(0, len(toks) - len(toks) % BLOCK, BLOCK):
            block = toks[i:i + BLOCK]
            key = self._key(parent, block)
            if i < cached:
                parent = key
                continue
            if key not in self.blocks:
                self.writes += 1
            self.blocks[key], self.used[key] = tuple(block), next(self.clock)
            parent = key
        while len(self.blocks) > self.capacity:
            victim = min(self.used, key=self.used.get)
            del self.blocks[victim], self.used[victim]


def static_prefix(policy_len):
    """system (180) + tools (220) + policy (policy_len) — the part that never changes."""
    return tokens(180, "sys") + tokens(220, "tools") + tokens(policy_len, "policy")


# --- (1) block alignment: only whole 16-token blocks are cacheable -----------------
print("static prefix tokens | cacheable | re-processed every request | over 1,000 requests")
for policy_len in (1600, 1606, 1612, 1590):
    static = static_prefix(policy_len)
    cacheable = 16 * (len(static) // BLOCK)
    print(f"{len(static):>19,} | {cacheable:>9,} | {len(static) - cacheable:>28,} | "
          f"{(len(static) - cacheable) * 1000:>22,}")

# --- (2) one changed token inside the static region --------------------------------
print("\nchanged token position | steady-state reuse per request (2,000-token static prefix)")
for change_at in (None, 3, 900, 1995):
    cache, last = PrefixCache(1 << 20), 0
    for req in range(6):
        static = tokens(2000, "static")
        if change_at is not None and req > 0:
            static[change_at] = 700000 + req
        toks = static + tokens(180, ("turn", req))
        cached = cache.match(toks)
        cache.insert(toks, cached)
        last = cached
    label = "none (stable prefix)" if change_at is None else f"token #{change_at}"
    print(f"{label:>22} | {last:>40,}")

# --- (3) cache capacity vs a multi-tenant working set ------------------------------
TENANTS, REQUESTS = 4, 60
print(f"\ncache capacity (blocks) | prefix hit across {TENANTS} tenants sharing ONE cache")
for capacity in (128, 256, 512, 640, 768, 1024, 4096):
    cache = PrefixCache(capacity)
    total = reused = 0
    for req in range(REQUESTS):
        for t in range(TENANTS):
            toks = tokens(180, f"sys{t}") + tokens(1820, f"body{t}") + tokens(180, (t, req))
            cached = cache.match(toks)
            total, reused = total + len(toks), reused + cached
            cache.insert(toks, cached)
    print(f"{capacity:>23,} | {100 * reused / total:>33.1f}%")
```
{% endraw %}

```
static prefix tokens | cacheable | re-processed every request | over 1,000 requests
              2,000 |     2,000 |                            0 |                      0
              2,006 |     2,000 |                            6 |                  6,000
              2,012 |     2,000 |                           12 |                 12,000
              1,990 |     1,984 |                            6 |                  6,000

changed token position | steady-state reuse per request (2,000-token static prefix)
  none (stable prefix) |                                    2,000
              token #3 |                                        0
            token #900 |                                      896
           token #1995 |                                    1,984

cache capacity (blocks) | prefix hit across 4 tenants sharing ONE cache
                    128 |                               0.0%
                    256 |                               0.0%
                    512 |                               0.0%
                    640 |                              90.2%
                    768 |                              90.2%
                  1,024 |                              90.2%
                  4,096 |                              90.2%
```

The capacity table is the one to act on. Four tenants, each needing 136 blocks for a request (125 static + 11 dynamic), need 544 blocks of cache for all four prefixes to survive. At 512 blocks they thrash — every prefix is evicted before its tenant asks again, and the hit rate is **0.0%**. At 640 it jumps to 90.2%. A cache sized "roughly right" is not half as good as one sized correctly; it is worthless. Under `--enable-prefix-caching`, that capacity is whatever your `--gpu-memory-utilization` budget leaves for KV blocks, so the flag alone is not the fix.

## What the reuse is worth on a published price list

The final block converts reused tokens into money using rates you can check. DeepSeek publishes `deepseek-flash` off-peak cache-miss input at **$0.15 per 1M tokens** and cache-hit input at **$0.003 per 1M**, a 50x spread. For Anthropic and OpenAI the published structure is multipliers (Anthropic: 0.1x for cache reads, 1.25x for 5-minute cache writes; OpenAI: cached input "discounted up to 95%"), so they are modelled on a $1/1M base to keep the shape visible.

{% raw %}
```python
STATIC, DYNAMIC, REQUESTS = 2000, 180, 1000
COLD = 0.01          # fraction of requests that find no cache entry (first request + TTL expiry)

# Published, body-verified rates: DeepSeek deepseek-flash off-peak (api-docs.deepseek.com,
# 4 Oct 2026) — cache-miss input $0.15/1M, cache-hit input $0.003/1M. Anthropic and OpenAI
# are modelled as multipliers on a $1/1M base so the shape is visible without any price table.
PROVIDERS = {
    "deepseek-flash off-peak ($0.15 miss / $0.003 hit)": (0.15, 0.003, 0.0),
    "anthropic-style (1.25x write / 0.1x read)": (1.00, 0.10, 1.25),
    "openai-style (up to 95% off cached input)": (1.00, 0.05, 0.0),
}
LAYOUTS = {"no cache": 0, "volatile early": 176, "static first": 2000}


def cost(provider, reused_per_request):
    miss_rate, hit_rate, write_rate = PROVIDERS[provider]
    cold_n = max(1, round(REQUESTS * COLD))
    warm_n = REQUESTS - cold_n
    units = 0.0
    for _ in range(warm_n):                      # warm requests reuse the prefix
        units += reused_per_request * hit_rate + (STATIC + DYNAMIC - reused_per_request) * miss_rate
    for _ in range(cold_n):                      # cold requests pay full miss price
        units += (STATIC + DYNAMIC) * miss_rate
        units += reused_per_request * write_rate if write_rate else 0.0
    return units / 1e6                            # per 1,000 requests, in USD or base units


for provider in PROVIDERS:
    print(f"\n{provider} — cost per 1,000 requests (input tokens only)")
    baseline = cost(provider, LAYOUTS["no cache"])
    for layout, reused in LAYOUTS.items():
        c = cost(provider, reused)
        saved = 100 * (baseline - c) / baseline
        print(f"  {layout:>15s} | {c:>7.3f} | saved {saved:>5.1f}% | "
              f"tokens reused/req (steady state) {reused:,}")

print("\nTTL sensitivity — deepseek-flash off-peak, layout 'static first', share of requests that miss:")
for cold in (0.00, 0.01, 0.05, 0.20, 0.50, 1.00):
    COLD = cold
    print(f"  {100 * cold:>5.0f}% cold | ${cost('deepseek-flash off-peak ($0.15 miss / $0.003 hit)', 2000):>7.3f} per 1,000 requests")
```
{% endraw %}

```
deepseek-flash off-peak ($0.15 miss / $0.003 hit) — cost per 1,000 requests (input tokens only)
         no cache |   0.327 | saved   0.0% | tokens reused/req (steady state) 0
   volatile early |   0.301 | saved   7.8% | tokens reused/req (steady state) 176
     static first |   0.036 | saved  89.0% | tokens reused/req (steady state) 2,000

anthropic-style (1.25x write / 0.1x read) — cost per 1,000 requests (input tokens only)
         no cache |   2.180 | saved   0.0% | tokens reused/req (steady state) 0
   volatile early |   2.025 | saved   7.1% | tokens reused/req (steady state) 176
     static first |   0.423 | saved  80.6% | tokens reused/req (steady state) 2,000

openai-style (up to 95% off cached input) — cost per 1,000 requests (input tokens only)
         no cache |   2.180 | saved   0.0% | tokens reused/req (steady state) 0
   volatile early |   2.014 | saved   7.6% | tokens reused/req (steady state) 176
     static first |   0.299 | saved  86.3% | tokens reused/req (steady state) 2,000

TTL sensitivity — deepseek-flash off-peak, layout 'static first', share of requests that miss:
      0% cold | $  0.033 per 1,000 requests
      1% cold | $  0.036 per 1,000 requests
      5% cold | $  0.048 per 1,000 requests
     20% cold | $  0.092 per 1,000 requests
     50% cold | $  0.180 per 1,000 requests
    100% cold | $  0.327 per 1,000 requests
```

Two things to read carefully. First, layout A's 7.8% saving is not a rounding error away from doing nothing: ordering the volatile block early captures the system prompt and abandons the 1,820 tokens of tools and policy, which is where almost all the static mass lives. Second, the TTL row: cache entries expire, and an expired prefix means a full-price prefill. At a 5% cold rate the DeepSeek bill doubles from $0.033 to $0.048 per 1,000 requests; the same workload with a 50% cold rate costs more than half of not caching at all. Prompt caching is a lease you keep renewing with traffic, not a permanent discount. Output tokens are excluded here because nothing in this harness generates them, and no latency is claimed: cache effects on time-to-first-token are a real and separate measurement.

## When you cannot reorder the prompt

Some workloads genuinely need the volatile part early, or have no stable part at all. Three cases and what is still recoverable:

- **Multi-turn conversations.** Appending turns at the end extends the matched prefix, which is why Anthropic's automatic caching can simply "move forward as conversations grow". Editing or compacting mid-history instead rewrites the prefix, so a summarisation step is a full-price re-prefill of the whole context for every session that shares that path. Do it deliberately, not on a timer.
- **Retrieved documents that rotate.** If the chunk set changes per request, cache the layer above it: Anthropic allows up to four explicit breakpoints, so the tools and system layer can be cached separately from the document layer; DeepSeek's common-prefix detection persists `A` after seeing `A+B` and `A+C`, so a document-heavy workload still converges on the shared head once the patterns repeat.
- **Tool-schema churn.** Tools are assembled before system and messages, so adding, deleting or reordering one tool invalidates every cached prefix on every request. A debug tool left in a staging deploy can cost more than the tokens it returns.

Worth auditing too: any middleware that prepends a request id, trace id or timestamp into the system message. That is layout C above, shipped by an observability agent, and nothing in the response will tell you.

## How to apply this to your own stack

1. **Order the prompt by changerate, not by narrative.** Tool schemas and system instructions first, then stable documents, then per-request context, then the instruction. Mark the breakpoint at the end of the static region.
2. **Hunt for interpolated values in the static region.** A date, a user name, a "current balance", a session id, a build hash. Any of these inside the system prompt collapses the cache to zero, and the failure is invisible in the response.
3. **Assert the hit in a test.** Both usage formats exist: OpenAI returns `cached_tokens` and `cache_write_tokens` (and offers a Prompt Caching Dashboard for hit-rate monitoring), Anthropic returns cache read/write token counts. In your staging suite, send the same request twice and fail if the second response reports zero cached tokens. That single assertion catches layout regressions that no reviewer notices.
4. **Respect the granularity.** Anthropic requires a minimum cacheable prefix length that varies by model and rewrites in 20-block units of lookback; OpenAI adds explicit breakpoints and a 30-minute TTL on GPT-5.6-and-later (`prompt_cache_options.ttl`), while earlier models default to `in_memory` retention of roughly 5-10 minutes of inactivity; DeepSeek matches whole persisted prefix units. A 300-token system prompt may simply be below the minimum, and no amount of reordering will cache it there.
5. **Route similar requests together.** OpenAI's `prompt_cache_key` exists to keep requests that share a prefix landing on the same cache; without it, load balancing can scatter them.
6. **If you self-host, size the cache for the working set.** Enable prefix caching, use SHA256 hashing in multi-tenant setups, and check the capacity curve before trusting a hit rate — the 512-block case above had the right code and a 0% hit rate.
7. **Re-measure after every prompt change.** The harness is 70 lines and computes the number the provider charges you for.

## Key takeaways

| Finding | Measured | Action |
| --- | --- | --- |
| Static content first | 2,000 of 2,180 tokens reused (91.7%) | Order tool schemas and instructions before live data |
| Live data before the knowledge base | 176 of 2,180 reused (8.1%) | Move the volatile block to the end |
| Request value inside the system prompt | 0 reused, 5,440 blocks stored for 40 requests | Never interpolate dates, names or session ids there |
| Cost, DeepSeek off-peak `deepseek-flash` | $0.036 vs $0.327 per 1,000 requests (89% saved) | Check your actual hit rate against the price list |
| One changed token in a 2,000-token prefix | reuse 0 at token 3, 1,984 at token 1,995 | Chained hashes make damage positional; check the head of the prompt first |
| Prefix not block-aligned (2,012 tokens) | 12 tokens re-processed per request, 12,000 per 1,000 requests | Pad or trim the static region to block/increment boundaries |
| Cache below the working set (512 vs 544 blocks) | 0.0% vs 90.2% hit rate | Size for the whole working set, not "roughly" |
| Cache expiry | $0.033 → $0.180 per 1,000 requests as cold share goes 0% → 50% | Keep traffic inside the TTL; batch jobs need it more than chat |

## References

- Anthropic, *Prompt caching* — cumulative prefix hashes, tools → system → messages ordering, 20-block lookback, 1.25x write / 0.1x read multipliers, 5-minute and 1-hour TTLs: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- OpenAI, *Prompt caching* — cached KV tensors, entire rendered prefix must match, cached input discounted up to 95%, `prompt_cache_key`, `prompt_cache_options.ttl` of 30 minutes, `cached_tokens` / `cache_write_tokens`: https://platform.openai.com/docs/guides/prompt-caching
- DeepSeek, *Context Caching* — on-disk cache enabled by default, full-match requirement, cache prefix units persisted at request boundaries, common-prefix detection and fixed token intervals: https://api-docs.deepseek.com/guides/kv_cache
- DeepSeek, *Models & Pricing* — `deepseek-flash` off-peak cache hit $0.003 vs cache miss $0.15 per 1M input tokens, peak hours 01:00-04:00 and 06:00-10:00 UTC: https://api-docs.deepseek.com/quick_start/pricing
- vLLM, *Automatic Prefix Caching* — block hash over parent hash, block tokens and extra hashes, "we only cache full blocks", SHA256 for multi-tenant deployments at 100-200ns per token: https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv:2312.07104 — RadixAttention for KV cache reuse, up to 6.4x higher throughput on agent, RAG and multi-turn workloads: https://arxiv.org/abs/2312.07104
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, arXiv:2309.06180 — the paged KV cache that prefix sharing builds on: https://arxiv.org/abs/2309.06180

## Related posts

- [KV Cache Quantization for Long Context](/posts/kv-cache-quantization-long-context/) — what the cache costs in memory once you round it
- [vLLM and LLM Serving](/posts/vllm-llm-serving/) — the paged allocator and the `--enable-prefix-caching` flag
- [DeepSeek V4 Peak/Off-Peak Pricing](/posts/deepseek-v4-peak-offpeak-pricing/) — cache-hit input tokens priced 30x-50x below misses
- [Self-Hosting Open-Weight LLMs](/posts/self-hosting-open-weight-llms/) — where the cache capacity in this post comes from
- [Model Serving 101](/posts/model-serving-101/) — from notebook to production endpoint

**Next in this series:** the same 16-token-block view applied to multi-turn agent loops, where the working set grows every turn and the eviction policy starts choosing between your system prompt and your last tool result.
