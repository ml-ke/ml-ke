# Prefix-cache (prompt cache) harness — reusable bank

Added 4 Oct 2026 with the post `prompt-prefix-cache-order-cost`. Reuse this instead of re-deriving
the mechanics: it is stdlib-only (`hashlib`, `itertools`), runs in ~4 s, and is deterministic
(blake2b-derived pseudo token ids, never `hash()`, because Python randomises string hashing per
process and would change every published figure between runs).

## The harness

A block-level prefix cache built the way vLLM builds one: 16-token blocks, SHA256 key over
(parent hash, block tokens), full blocks only, LRU eviction under a capacity, plus a `writes`
counter that doubles as a memory-waste metric.

```python
import hashlib, itertools
BLOCK = 16

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

    def match(self, toks):                    # longest cached prefix, whole blocks only
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


def tokens(n, salt):
    return [int.from_bytes(hashlib.blake2b(f"{salt}:{i}".encode(), digest_size=4).digest(), "big")
            % 10 ** 6 for i in range(n)]
```

## Measured anchors (4 Oct 2026, quoted in the post)

2,180-token prompt = 180 system + 220 tool schemas + 1,600 policy (2,000 static) + 48 live-data +
112 ticket + 20 instruction; 40 requests; capacity 4,096 blocks.

| Layout | Reuse (request 2+) | Hit | Blocks stored |
| --- | --- | --- | --- |
| static first | 2,000 tok (91.7%) | 91.7% | 565 |
| volatile early (live data at ~token 180) | 176 tok | 8.1% | 5,011 |
| timestamp interpolated into the system prompt | 0 | 0.0% | 5,440 |

Other verified anchors: alignment leak 12 tokens/request (12,000 per 1,000 requests) at a 2,012-token
static prefix; one changed token gives reuse 0 (token #3), 896 (#900), 1,984 (#1995) on a 2,000-token
prefix; 4 tenants x 136 blocks need 544 blocks of cache — 512 blocks thrashes to **0.0%** hit while
640 gives 90.2%; cost per 1,000 requests on deepseek-flash off-peak $0.327 (no cache) / $0.301
(volatile early) / $0.036 (static first), and $0.033 > $0.180 as the TTL cold share goes 0% > 50%.

## Source facts worth not re-researching

- Anthropic: hash is cumulative from the start of the prompt to the breakpoint; order is tools, then
  system, then messages; 20-block lookback; max 4 breakpoints; 5-min writes 1.25x, 1-h writes 2x,
  reads 0.1x (0.05x Opus 5.5, 0.025x Fable 5.1 / Mythos 5.1); 5-min TTL refreshed free per reuse.
- OpenAI: caches KV tensors, requires the entire rendered prefix (hidden system message, tools,
  developer message, history) to match; cached input "discounted up to 95%"; `prompt_cache_key`;
  GPT-5.6+ uses `prompt_cache_options.ttl` (only value 30m); earlier models `prompt_cache_retention`
  in_memory (~5-10 min idle, up to 1 h) or 24h; usage fields `cached_tokens`, `cache_write_tokens`.
- DeepSeek: on-disk context cache on by default; hit needs a full match of a persisted cache prefix
  unit (created at request boundaries, by common-prefix detection, and at fixed token intervals).
- vLLM: block key = parent hash + block tokens + extra hashes; "we only cache full blocks"; SHA256
  recommended multi-tenant, ~100-200 ns/token (~6 ms per 50k tokens).
- SGLang RadixAttention (arXiv:2312.07104): up to 6.4x throughput on agent/RAG/multi-turn workloads.

## Extension points for a follow-up post

Swap the synthetic workload for a real trace (log your prompts, keep the rendered text, replay it),
model `--gpu-memory-utilization` as the capacity knob, or grow the working set per turn to show the
point where LRU eviction starts choosing between the system prompt and the last tool result.
