---
title: "The Retry That Charged Twice: Idempotency Keys and Retry Budgets for LLM Calls"
date: 2026-10-23 00:00:00 +0300
categories: [LLM, AI Engineering]
tags: [idempotency, retries, exponential backoff, agent tools, error handling, llmops]
image:
  path: /assets/img/cover-retry-that-charged-twice.webp
  alt: One request entering a ledger twice versus entering once through a keyed dedupe store
---

> **The short version**
> A timeout is not a failure. It is the absence of an answer, and the work may already have happened. This post separates the failures you can retry from those you cannot, measures what a naive loop costs in duplicate side effects, and derives the backoff schedule and retry budget that keep a bad minute from becoming a bad hour. Every number comes from the three seeded simulations below.
{: .prompt-info }

The retry loop is the least reviewed code in an LLM application: three lines inside an SDK call, or a `for` loop added after a 429, and the only part that runs precisely when the system is already unhealthy.

This post is not about how fast calls come back, how often a cache skips one, or how to order a prompt so its prefix is reused: [the load-testing post](/posts/load-test-latency-coordinated-omission/) hides latency, [the semantic cache post](/posts/semantic-cache-threshold-precision/) skips a call, and [the prompt prefix post](/posts/prompt-prefix-cache-order-cost/) makes one cheaper. This one starts after a call has failed and asks the question that matters then: if I send it again, what happens twice?

For a completion with no tools, the answer is a second invoice. For an agent that just charged a card or sent an email, it is the customer's problem. The difference is whether the operation has a name the server can recognise.

## Which failures are safe to retry

| What the client saw | Did the work happen? | Retry? |
|---|---|---|
| Connection refused, DNS failure | No, the request never left | Yes, immediately |
| Connection reset after the body was sent | Unknown | Only with a key |
| Client timeout | Unknown | Only with a key |
| 429 with `Retry-After` | No | Yes, after the delay |
| 500, 502, 503, 504 | Unknown, the commit may precede the crash | Only with a key |
| 529 overloaded | No | Yes, with backoff |
| 400, 401, 403, 404, 422 | No | No, fix the request |
| 409 conflict | No | Yes, after reading the state |

Three rows are the whole problem. A 5xx usually means the request failed, except when it means the work succeeded and the response did not. A timeout says nothing about the server. And the SDKs retry both by default.

The OpenAI Python client retries connection errors, `408`, `409`, `429` and any status `>= 500` twice by default, and its documentation notes that timed-out requests are among them ([openai-python README](https://github.com/openai/openai-python#retries)). Anthropic's SDK retries transient failures twice by default and honours `Retry-After` ([Claude API errors](https://docs.anthropic.com/en/api/errors)). The ambiguous case is already retried twice before your code sees an error.

`400`, `401`, `403`, `404` and `422` describe the request itself, and Anthropic's reference adds that a `429` can come from a spend cap rather than a rate limit: no `Retry-After`, and it keeps failing until access resumes. Treating every 429 as backpressure hammers a hard stop for as long as the budget allows.

## A timeout is an answer you did not get

The distinction to install is between a call that failed and a call that is *unknown*. AWS's builders put it plainly: a timeout or failure "doesn't necessarily mean that side effects haven't happened", and if repeating the side effects is undesirable, the API should be designed to be idempotent ([AWS Builders' Library](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)).

The HTTP spec agrees. A method is idempotent "if the intended effect on the server of multiple identical requests with that method is the same as the effect for a single such request", and it is singled out so that "the request can be repeated automatically if a communication failure occurs before the client is able to read the server's response" ([RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)). `PUT` and `DELETE` are idempotent. `POST` is not, which is why payment APIs invent a key for it.

Shorter timeouts make this worse. A tight timeout catches a slow server sooner and converts more successes into unknowns, so the chance that an attempt committed but the client never learned about it rises with your impatience. If you tightened a timeout to protect a p99 and watched duplicate records appear, that is the mechanism.

## What an idempotency key has to be keyed on

Stripe's implementation is the reference, and its details are the ones people get wrong. The client sends `Idempotency-Key`, and Stripe returns the saved status and body for any retry carrying the same key, "including 500 errors". Keys are pruned once at least 24 hours old, and the layer compares incoming parameters against the original, erroring when they differ so a reused key cannot mean a different operation ([Stripe idempotent requests](https://docs.stripe.com/api/idempotent_requests)).

Home-grown versions usually miss the key's source. It must identify the logical operation: tenant, conversation, plan step, tool, arguments. Derive it from payload and caller, never from an attempt number or a timestamp. Then store three states, not two, because the interesting case is a key that is *in flight*.

Stripe describes that case directly: when a request conflicts with another executing under the same key, no result is saved and the request can be retried. That is a reservation, not a record. A store that writes only after success cannot answer "is someone else doing this", so a second attempt walks past it.

## Measuring the duplicate, not the retry

The simulation models 1,000 logical write calls against a server that commits the side effect for both `ok` and `ambiguous` outcomes. Each call gets three attempts, and the mix is unkind: 6% ambiguous, 3% clean 5xx, 4% rate limited.

{% raw %}
```python
import random

SEED = 7
N = 1000
MAX_ATTEMPTS = 3

# what one attempt can return, as the client sees it.
# "ambiguous" = the server committed the work, the response never arrived.
MIX = [("ok", 0.87), ("ambiguous", 0.06), ("server_5xx", 0.03), ("rate_429", 0.04)]

rng = random.Random(SEED)
GRID = []
for _ in range(N):
    row = []
    for _ in range(MAX_ATTEMPTS):
        u, acc = rng.random(), 0.0
        for name, weight in MIX:
            acc += weight
            if u < acc:
                row.append(name)
                break
    GRID.append(row)


def run(policy):
    """policy: 'naive', 'attempt_key', 'op_key'.

    The server executes a side effect for 'ok' and 'ambiguous' attempts. It
    remembers every key it has executed, so a replay of the same key returns
    the stored result instead of doing the work again.
    """
    effects = committed = answered = zero_effect = replays = 0
    for op, row in enumerate(GRID):
        executed = set()
        here = 0
        got = False
        for attempt, outcome in enumerate(row, 1):
            if policy == "naive":
                key = None
            elif policy == "attempt_key":
                key = (op, attempt)
            else:
                key = (op,)

            if outcome not in ("ok", "ambiguous"):
                continue  # 5xx and 429 did no work server-side

            if key is not None and key in executed:
                got = True
                replays += 1
                break

            if key is not None:
                executed.add(key)
            effects += 1
            here += 1
            if outcome == "ok":
                got = True
                break

        if here:
            committed += 1
        if got:
            answered += 1
        else:
            zero_effect += 1

    return effects, committed, answered, zero_effect, replays


rows = []
for policy, label in (("naive", "no key"),
                      ("attempt_key", "key per attempt"),
                      ("op_key", "key per operation")):
    effects, committed, answered, zero, replays = run(policy)
    rows.append((label, effects, effects - committed, answered, zero, replays))

print(f"{N} logical write calls, up to {MAX_ATTEMPTS} attempts each, seed {SEED}")
print(f"attempt mix: {MIX[0][1] * 100:.0f}% ok, {MIX[1][1] * 100:.0f}% ambiguous, "
      f"{MIX[2][1] * 100:.0f}% 5xx, {MIX[3][1] * 100:.0f}% rate limited")
print(f"{'client policy':18s}{'effects':>8s}{'dupes':>7s}{'answered':>10s}{'no effect':>11s}{'replays':>9s}")
for label, effects, dup, answered, zero, replays in rows:
    print(f"{label:18s}{effects:8d}{dup:7d}{answered:10d}{zero:11d}{replays:9d}")

naive_dup, naive_effects, naive_answered = rows[0][2], rows[0][1], rows[0][3]
print(f"\nkey per attempt removes {naive_dup - rows[1][2]} of {naive_dup} duplicates: "
      f"a key minted per attempt is a new operation every time")
print(f"key per operation removes {naive_dup - rows[2][2]} of {naive_dup} duplicates, "
      f"cutting billed executions from {naive_effects} to {rows[2][1]}")
print(f"answers served from the dedupe store instead of recomputed: {rows[2][5]}")
print(f"logical calls with no effect at all: {rows[2][4]} (idempotency is not delivery)")
print(f"duplicate rate of the unkeyed client: {naive_dup / N * 100:.1f}% of calls")
print("duplicates on a read-only call: 0 under all three policies")
```
{% endraw %}

```text
1000 logical write calls, up to 3 attempts each, seed 7
attempt mix: 87% ok, 6% ambiguous, 3% 5xx, 4% rate limited
client policy      effects  dupes  answered  no effect  replays
no key                1070     71       999          1        0
key per attempt       1070     71       999          1        0
key per operation      999      0       999          1       70

key per attempt removes 0 of 71 duplicates: a key minted per attempt is a new operation every time
key per operation removes 71 of 71 duplicates, cutting billed executions from 1070 to 999
answers served from the dedupe store instead of recomputed: 70
logical calls with no effect at all: 1 (idempotency is not delivery)
duplicate rate of the unkeyed client: 7.1% of calls
duplicates on a read-only call: 0 under all three policies
```

The middle row is why this post exists.

**The default costs 71 extra side effects per 1,000 writes.** A 7.1% duplicate rate is not a rounding error on a payment or an outbound email.

**A key minted per attempt changes nothing.** That row is identical to `no key`, because the server sees a new operation each time. It is the most common failure I have seen in agent wrappers: the header is populated, the code reads as idempotent, and the behaviour is unchanged. Compute the key before the first attempt and reuse it.

**Only the operation key also recovers answers.** The keyed client gets 70 responses back it never recomputed, because an ambiguous attempt left a result the replay could read. Idempotency is not only a guard; it turns a timeout into a cache hit.

The one call with no effect under every policy is the counterweight. Idempotency makes retries safe, not delivery certain, and something still has to notice the operation that never completed.

## Backoff is a schedule, not a sleep

Full jitter is the best-known answer to the thundering herd, and the formula is short:

```
sleep = random_between(0, min(cap, base * 2 ** attempt))
```

Marc Brooker introduced it by simulating optimistic concurrency control with 100 contending clients. Capped exponential backoff without jitter helped slightly and made completion times much worse; full jitter "uses less work" than equal jitter and far less than no jitter, and cut the call count by more than half at that contention ([AWS Architecture Blog](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)). Google's Cloud Storage guidance describes the same truncated exponential backoff ([Cloud Storage retry strategy](https://cloud.google.com/storage/docs/retry-strategy)).

Written this way, the schedule is arithmetic, which makes the attempt cap a derived number rather than a preference. The block below computes sleep ceilings, worst-case elapsed time (full jitter can sleep the whole ceiling), and mean elapsed time for two starting bases against the same 20-second deadline.


{% raw %}
```python
BASE = 250      # first sleep ceiling, ms
CAP = 8000      # ceiling cap, ms
DEADLINE = 20000  # total budget for the whole logical call, ms
SERVICE = 400   # assumed server time per attempt, ms


def schedule(base, cap, attempts, deadline, service):
    rows, worst, expect = [], 0.0, 0.0
    for n in range(1, attempts + 1):
        ceiling = min(cap, base * 2 ** (n - 1))
        worst += ceiling + service          # full jitter can sleep the whole ceiling
        expect += ceiling / 2 + service     # mean of uniform(0, ceiling)
        rows.append((n, ceiling, worst, expect, worst <= deadline))
    return rows


def show(label, rows):
    print(label)
    print(f"{'attempt':>7s}{'ceiling ms':>12s}{'worst ms':>10s}{'mean ms':>10s}{'in deadline':>13s}")
    for n, ceiling, worst, expect, fits in rows:
        print(f"{n:7d}{ceiling:12d}{worst:10.0f}{expect:10.0f}{('yes' if fits else 'NO'):>13s}")
    last_ok = max(n for n, _, _, _, fits in rows if fits)
    print(f"attempts that fit the deadline: {last_ok}\n")


show("base 250 ms, cap 8000 ms, deadline 20000 ms", schedule(BASE, CAP, 7, DEADLINE, SERVICE))
show("base 1000 ms, cap 8000 ms, deadline 20000 ms", schedule(1000, CAP, 7, DEADLINE, SERVICE))

print(f"deadline {DEADLINE / 1000:.0f} seconds, service time {SERVICE} ms per attempt")
print("failure rates tested: 1%, 5%, 8%, 12%, 25%")
print("retry budget: 10% of 1000 requests per second = 100 retries per second")
print(f"{'failure rate':>13s}{'attempts/req':>14s}{'retries/s':>11s}{'budget':>9s}")
for p in (0.01, 0.05, 0.08, 0.12, 0.25):
    attempts = sum(p ** i for i in range(4))
    retries = (attempts - 1) * 1000
    print(f"{p:13.2f}{attempts:14.3f}{retries:11.1f}{('ok' if retries <= 100 else 'TRIP'):>9s}")

trip = next(p / 1000 for p in range(1, 1000)
            if (sum((p / 1000) ** i for i in range(4)) - 1) * 1000 > 100)
print(f"\nthe 10% budget trips at a {trip * 100:.1f}% failure rate, "
      f"well below the 25% the client would happily retry")
```
{% endraw %}

```text
base 250 ms, cap 8000 ms, deadline 20000 ms
attempt  ceiling ms  worst ms   mean ms  in deadline
      1         250       650       525          yes
      2         500      1550      1175          yes
      3        1000      2950      2075          yes
      4        2000      5350      3475          yes
      5        4000      9750      5875          yes
      6        8000     18150     10275          yes
      7        8000     26550     14675           NO
attempts that fit the deadline: 6

base 1000 ms, cap 8000 ms, deadline 20000 ms
attempt  ceiling ms  worst ms   mean ms  in deadline
      1        1000      1400       900          yes
      2        2000      3800      2300          yes
      3        4000      8200      4700          yes
      4        8000     16600      9100          yes
      5        8000     25000     13500           NO
      6        8000     33400     17900           NO
      7        8000     41800     22300           NO
attempts that fit the deadline: 4

deadline 20 seconds, service time 400 ms per attempt
failure rates tested: 1%, 5%, 8%, 12%, 25%
retry budget: 10% of 1000 requests per second = 100 retries per second
 failure rate  attempts/req  retries/s   budget
         0.01         1.010       10.1       ok
         0.05         1.053       52.6       ok
         0.08         1.087       86.9       ok
         0.12         1.136      136.1     TRIP
         0.25         1.328      328.1     TRIP

the 10% budget trips at a 9.1% failure rate, well below the 25% the client would happily retry
```

At a 250 ms base, six attempts fit inside 20 seconds. Raise the base to one second and only four fit: the fifth attempt's worst case is 25,000 ms. Nothing changed except the wait before the first retry, and the attempt budget fell by a third.

So "retry 5 times" is not a policy. The deadline is the policy, and attempts are what it can afford. The mean column matters too: at attempt six it is 10,275 ms against a worst case of 18,150 ms, so a p99 target comes from the right-hand column and a capacity plan from the left.

## A retry budget caps the blast radius

A per-call cap says nothing about aggregate retry traffic, and that aggregate is what melts backends. The SRE book walks through it: a frontend sending 10,100 QPS at a backend that rejects 100 of them retries those failures, the retries push the total to 10,200, the failure count rises to 200, and the retry volume compounds from 100 to 200 to 300 per second until "retries can destabilize a system" ([SRE Book](https://sre.google/sre-book/addressing-cascading-failures/)).

The counterweight is a retry budget, and it exists in production proxies rather than diagrams. Envoy's circuit breaker exposes `budget_percent`: "the limit on concurrent retries as a percentage of the sum of active requests and active pending requests. For example, if there are 100 active requests and the budget_percent is set to 25, there may be 25 active retries", defaulting to 20%, with `min_retry_concurrency` defaulting to 3 ([Envoy](https://www.envoyproxy.io/docs/envoy/latest/api-v3/config/cluster/v3/circuit_breaker.proto)).

The budget table above is that arithmetic. At 1,000 requests per second with a 10% budget, 100 retries per second are affordable: a 1% failure rate demands 10.1, 5% demands 52.6, 8% demands 86.9. At 12% the demand is 136.1 and the budget trips, with the crossing at 9.1%. A failure rate becomes a hard ceiling, which is what a client that cannot see the rest of the fleet needs. Once spent, the retry returns the original error, worse for one caller and better for everyone sharing the backend.

## Streaming breaks the status-code rule

A streamed completion moves the error to where your retry logic is not looking. Once the response has started, the status line was already `200`. Anthropic's docs are explicit that a stream error "can occur after the API returns a 200 response", and that in that case "error handling doesn't follow these standard mechanisms" ([Claude API errors](https://docs.anthropic.com/en/api/errors)).

There is no 5xx to classify, no `Retry-After`, no status to branch on. What you have is a partial message and a closed socket, so three things move.

Retry decisions move into the parser: a mid-stream `error` event is a failure signal, and so is the absence of the terminal event. Treat a closed connection without it as an unknown outcome, not a short answer.

Tool arguments arrive as deltas, so an executor that fires on the first plausible JSON object can execute a partial call and then execute a second one on retry. Buffer the arguments and dispatch only after the message completes.


## Tool calls are where the money moves

An agent loop multiplies the problem, because the operation that fails is a completion plus whatever the model decided to do. The write is what cannot be repeated, and the model has no concept of a retry: it reasons about the task, not the transport.

| Tool call | The side effect | How to make the retry safe |
|---|---|---|
| Charge or payout | Money leaves an account | Provider idempotency key, stored before the first attempt |
| Send email or SMS | A message reaches a person | Dedupe key on recipient, template and operation id |
| Create ticket or record | A new row exists | Upsert on an external id you generate |
| Update a CRM field | A value is overwritten | `PUT` with an ETag or version precondition |

The rule worth defending in review: never retry a write without a key, and never let the agent choose the key. The key belongs to the caller's bookkeeping, generated when the step is planned, so a re-plan, a resume from a checkpoint, and a retry all produce the same key for the same intention. Recovery has two forms: make the effect idempotent, or reconcile by querying the provider with the key.



{% raw %}
```python
import random

SEED = 3
N = 1000
P_CONCURRENT = 0.10  # a second attempt arrives while the first is still running
P_CRASH = 0.02       # the worker dies mid-call
LEASE_S = 30

rng = random.Random(SEED)
OPS = []
for _ in range(N):
    concurrent = rng.random() < P_CONCURRENT
    crashed = rng.random() < P_CRASH
    after_commit = rng.random() < 0.5   # crash lands after the side effect
    OPS.append((concurrent, crashed, after_commit))


def store(policy):
    """policy: 'write_after', 'reserve', 'reserve_no_lease', 'reserve_upsert'."""
    doubles = deferred = stuck = 0
    for concurrent, crashed, after_commit in OPS:
        if policy == "write_after":
            if concurrent:
                doubles += 1          # nothing recorded yet, so the retry executes
            if crashed and after_commit:
                doubles += 1          # effect landed, no record, later retry re-executes
            continue

        # key is reserved before the effect runs
        if policy == "reserve_no_lease":
            if crashed:
                stuck += 1            # in-flight forever, every retry gets 409
                continue
            if concurrent:
                deferred += 1
            continue

        if concurrent:
            deferred += 1             # 409 while the first attempt holds the lease
        if crashed and after_commit:
            if policy == "reserve":
                doubles += 1          # lease expires, sweeper re-runs a plain POST
            # reserve_upsert: the sweep re-runs an upsert on the key, so no double

    return doubles, deferred, stuck


print(f"{N} write operations, {int(P_CONCURRENT * 100)}% with a concurrent retry, "
      f"{int(P_CRASH * 100)}% with a dead worker, lease {LEASE_S}s, seed {SEED}")
print(f"{'dedupe store':22s}{'double effects':>16s}{'deferred':>10s}{'stuck':>7s}")
for policy, label in (("write_after", "write after completion"),
                      ("reserve_no_lease", "reserve, no lease"),
                      ("reserve", "reserve + lease"),
                      ("reserve_upsert", "reserve + keyed upsert")):
    doubles, deferred, stuck = store(policy)
    print(f"{label:22s}{doubles:16d}{deferred:10d}{stuck:7d}")

plain = store("reserve")[0]
upsert = store("reserve_upsert")[0]
print(f"\nreserving the key before the effect removes {store('write_after')[0] - plain} "
      f"of {store('write_after')[0]} double effects")
print(f"the lease alone still leaves {plain}: a crash after commit is invisible to the store")
print(f"only an effect that is itself keyed gets to {upsert}")
print(f"without a lease, {store('reserve_no_lease')[2]} keys never clear")
```
{% endraw %}

```text
1000 write operations, 10% with a concurrent retry, 2% with a dead worker, lease 30s, seed 3
dedupe store            double effects  deferred  stuck
write after completion             102         0      0
reserve, no lease                    0        95     13
reserve + lease                      5        97      0
reserve + keyed upsert               0        97      0

reserving the key before the effect removes 97 of 102 double effects
the lease alone still leaves 5: a crash after commit is invisible to the store
only an effect that is itself keyed gets to 0
without a lease, 13 keys never clear
```

The first row is the store most teams write: save the result when the call returns. It produces 102 double effects per 1,000 operations, because a concurrent attempt arrives before the record exists and a crash after the commit leaves the effect invisible.

Reserving the key before the effect removes 97 of those 102 and adds a cost: 95 operations are deferred with a conflict while the first attempt holds the lease. That is correct, but the caller now needs a path for "in progress, ask again", with its own timeout.

The remaining 5 are the interesting ones: crashes that landed after the effect committed and before the store was told. No reservation fixes those, because the store is not what performed the write. Only an effect that is itself keyed, an upsert on the operation id, reaches zero. A lease buys mutual exclusion for the duration of a call, not exactly-once.

The second row is the trap in the other direction: a reservation without a lease never clears when the worker dies. Thirteen keys stay stuck, no retry can proceed, and the operations are lost. Add the expiry and the sweeper on the same day you add the reservation.

## What to ship

| Layer | Setting | Why |
|---|---|---|
| Operation identity | Key on tenant, conversation, plan step, tool, arguments | The same intention must produce the same key across retries and resumes |
| Retryable set | Connection errors, 408, 429, 5xx, 529 | The rest is a request bug or a hard stop |
| Backoff | Full jitter, base 200 to 500 ms, cap 8 to 10 s | Spreads the herd without long tails |
| Attempts | Derived from the deadline | A 20 s deadline affords six at a 250 ms base |
| Budget | 10 to 20% of request volume, failing closed | Caps the amplification the SRE book describes |
| Dedupe store | Reserve before the effect, lease plus sweeper | In-flight keys are the race a result-only store cannot see |
| Streaming | No tool dispatch until the message completes | A 200 status can still carry a mid-stream error |
| Recovery | A reconciliation sweep for operations with no terminal state | Idempotency is not delivery |

## Key takeaways

| Finding | Measured here | What to do |
|---|---|---|
| A naive loop duplicates writes | 71 extra effects per 1,000 calls at a 13% non-success rate | Key every write before the first attempt |
| A key per attempt is no key | 71 duplicates, identical to the unkeyed client | Derive the key from the operation and reuse it |
| A keyed retry can recover a lost answer | 70 responses served from the store | Treat the store as a cache of outcomes |
| The deadline sets the attempt count | 6 attempts at a 250 ms base, 4 at 1,000 ms, same 20 s | Derive the cap, do not pick it |
| A retry budget fails closed | 10% budget trips at a 9.1% failure rate | Cap retries as a share of traffic |
| A reservation without an effect key still duplicates | 102 to 5 double effects, then 0 with an upsert | Make the write idempotent, not just the call |

The retry that charged twice is not a story about a bad model or a flaky network. It is a story about a client that could not tell "it failed" from "I did not hear back", and an API that could not be told the difference. Both halves are fixable, and the second one is a header.

## References

- Stripe API reference, "Idempotent requests" (24-hour pruning, parameter comparison, concurrent conflicts) — https://docs.stripe.com/api/idempotent_requests
- Stripe Dot Dev Blog, "Because nobody likes being charged twice" (duplicate charges, idempotency keys) — https://stripe.dev/blog/because-nobody-likes-being-charged-twice
- RFC 9110, "HTTP Semantics", §9.2.2 Idempotent Methods — https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2
- AWS Builders' Library, "Timeouts, retries, and backoff with jitter" (timeouts and side effects) — https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
- Marc Brooker, "Exponential Backoff And Jitter", AWS Architecture Blog — https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- Google Cloud, "Retry strategy" for Cloud Storage (idempotency table, truncated backoff) — https://cloud.google.com/storage/docs/retry-strategy
- Google SRE Book, "Addressing Cascading Failures" (retry amplification) — https://sre.google/sre-book/addressing-cascading-failures/
- Envoy API reference, `RetryBudget` (`budget_percent`, `min_retry_concurrency`) — https://www.envoyproxy.io/docs/envoy/latest/api-v3/config/cluster/v3/circuit_breaker.proto
- OpenAI Python SDK README, "Retries" and "Timeouts" (default retries, timed-out requests) — https://github.com/openai/openai-python#retries
- OpenAI API reference, "Error codes" (429 with `Retry-After`, 500, 503) — https://platform.openai.com/docs/guides/error-codes
- Anthropic API reference, "Claude API errors" (retryable statuses, SDK retries, mid-stream SSE errors) — https://docs.anthropic.com/en/api/errors
- OpenAI Cookbook, "How to handle rate limits" (backoff with jitter) — https://cookbook.openai.com/examples/how_to_handle_rate_limits

## Related posts

- [Coordinated Omission: why your load test reports a latency that users never see](/posts/load-test-latency-coordinated-omission/) — the measurement side of the same failure
- [The Cache Was Confident and Wrong: choosing a semantic cache threshold](/posts/semantic-cache-threshold-precision/) — when skipping a call is the risky path
- [Prompt Prefix Order and Cost](/posts/prompt-prefix-cache-order-cost/) — why a retry re-bills the prefix
- [Tool Use and Function Calling for agents](/posts/agent-tool-calling/) — where a retried call becomes a side effect
