# Load-test latency harness: coordinated omission, p99 and utilization ceilings

Verified 2026-10-09 for the post `load-test-latency-coordinated-omission`. Every figure below came out of
an executed block in that post (stdout byte-identical to what shipped). Reuse this instead of re-deriving
or re-running the twelve-second 2M-arrival table. Stdlib only: no numpy, no model downloads.

## Fixture and measured results (the headline numbers)

Server: healthy service 10 ms, one 500 ms freeze starting at t = 1000 ms, declared test intent 80 req/s
(interval 12.5 ms), 450 requests. Freeze model: a request whose service interval intersects the freeze
window does the work before the window, then finishes at `stall[1] + remaining`.

| generator | n | p50 | p90 | p99 | max | >100 ms |
|---|---|---|---|---|---|---|
| closed loop, raw | 450 | 10.0 | 10.0 | **10.0** | 507.5 | 1 |
| closed loop + Tene fill-in correction | 490 | 10.0 | 10.0 | 445.0 | 507.5 | 33 |
| closed loop, naive intended-start subtraction ("lagged") | 450 | 510.0 | 510.0 | 510.0 | 510.0 | 370 |
| open loop, arrivals every 12.5 ms | 450 | 10.0 | 397.5 | **500.0** | 510.0 | 164 (36%) |

Backlog arithmetic the block also prints: 500 ms of arrivals = 40 requests of backlog; server retires
100 req/s against 80 req/s arriving; queue drains in `40/(100-80) = 2.0 s` at 80% utilization.

**The lie in one line:** raw closed-loop p99 = 10.0 ms for a service that froze for half a second.
One slow sample in 450 sits above the p99 cut (1/450 = 0.22% > 1%), so the percentile never moves.

## The correction (Gil Tene / HdrHistogram `recordValueWithExpectedInterval`)

```python
def fill_in_corrected(samples, interval):
    out = []
    for v in samples:
        while v > interval:
            out.append(v)
            v -= interval
        out.append(v)
    return out
```

Expand each recorded sample into one sample per missed interval. Honest bound: here corrected p99 = 445.0 ms
vs measured open-loop 500.0 ms (12% apart). The naive variant (subtract `i * interval` from each completion)
is a trap: a single-threaded generator never recovers the slots it lost, so p50 becomes the whole stall (510 ms).

## M/M/1 table (2,000,000 arrivals per row, service-time units)

| rho | mean sim | mean 1/(1-rho) | p99 sim | p99 ln(100)/(1-rho) |
|---|---|---|---|---|
| 0.50 | 2.001 | 2.000 | 9.28 | 9.21 |
| 0.70 | 3.340 | 3.333 | 15.47 | 15.35 |
| 0.80 | 5.016 | 5.000 | 23.26 | 23.03 |
| 0.90 | 10.067 | 10.000 | 47.08 | 46.05 |
| 0.95 | 20.404 | 20.000 | 92.22 | 92.10 |
| 0.98 | 53.695 | 50.000 | 309.83 | 230.26 |

rho = 0.90 detail: p50 6.94, p90 23.10, p95 30.22, p99 47.08, p99.9 77.62, max 139.7.

- Basis: sojourn time is exponential with parameter mu(1-rho) (TU/e lecture notes h4.pdf), so
  `p99 = ln(100)/(1-rho)` mean service times; mean sojourn `1/(mu-lambda)` (Wikipedia M/M/1).
- The p99/mean ratio is a **constant 4.605** at every utilization level.
- rho = 0.98 does not converge at 2M arrivals (mean 53.695 vs 50.000): the answer is dominated by a few long
  busy periods. Use it as the sample-size lesson, not as a capacity number.

## Replica study (why a benchmark p99 is noisy)

30 independent runs of 20,000 requests per level. Theory p99: 23.03 / 46.05 / 92.10.

| rho | median | min | max | max/min |
|---|---|---|---|---|
| 0.80 | 23.48 | 18.60 | 29.26 | 1.57 |
| 0.90 | 44.46 | 30.18 | 71.49 | 2.37 |
| 0.95 | 78.28 | 44.39 | 125.56 | 2.83 |

Long sojourns arrive in clusters inside long busy periods, so n requests hold far fewer than n independent
tail observations, and fewer as rho rises. A one-run 20k-request benchmark cannot separate 50 ms from 71 ms.

## SLO -> utilization ceiling, and fan-out

`rho_max = 1 - ln(100)/S` for a p99 budget of S mean service times.

| p99 SLO | rho ceiling | mean wait | mean queue |
|---|---|---|---|
| 5x | 0.079 | 0.09x | 0.01x |
| 10x | 0.539 | 1.17x | 0.63x |
| 20x | 0.770 | 3.34x | 2.57x |
| 50x | 0.908 | 9.86x | 8.95x |
| 100x | 0.954 | 20.71x | 19.76x |

Fan-out (each replica slow on 1% of calls): 1 replica 0.0100, 10 replicas 0.0956, 100 replicas 0.6340
(Dean & Barroso, Tail at Scale). Per-replica slow rate needed for a 1% overall rate: 0.01 / 0.001 / 0.0001,
i.e. a p99.99 budget per shard when 100 shards fan out.

## Traps that cost time

- **Blocks run in isolation** (`verify-post-code.py`): every block must define its own `mm1`/`pct`. Repeating
  small helpers is correct, not sloppy.
- **A short simulation misleads on the tail.** 200k arrivals at rho=0.90 gave p99 62.98 vs theory 46.05; 2M gave
  47.08. Budget a few seconds of runtime, or the post will quote a wrong number.
- `sed`/naive pct indexes: use `xs[int(round(p/100*(len(xs)-1)))]` after sorting; the standard index formula
  matters when comparing against a closed form.
- Model assumption to state: M/M/1 assumes exponential service; a batched GPU is less variable, so the formula
  is a conservative planning bound and the `1/(1-rho)` shape is the part that transfers.

## Sources (all body-verified)

- k6 docs: open-vs-closed models, constant-arrival-rate executor - https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/
- Gil Tene, How NOT to Measure Latency (ACM Queue 2013, 403s to curl; cross-verify via k6 + Today Software Magazine + BTW paper)
- HdrHistogram README: `recordValueWithExpectedInterval()` - https://github.com/HdrHistogram/HdrHistogram/blob/master/README.md
- Friedrich, Wingerath, Ritter, Coordinated Omission in NoSQL Database Benchmarking (BTW 2017) - https://www.btw2017.informatik.uni-stuttgart.de/slidesandpapers/E4-11-107/paper_web.pdf
- Wikipedia M/M/1 queue; TU/e lecture notes h4.pdf - https://iadan.win.tue.nl/que/h4.pdf
- Dean & Barroso, The Tail at Scale (CACM 2013) - 63% fan-out figure
- Google SRE ch.21 Handling Overload / ch.22 Addressing Cascading Failures
- NVIDIA Perf Analyzer load modes (concurrency vs request-rate) - https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/inference_load_modes.md
- NVIDIA NIM metrics (TTFT includes queueing) - https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
- Anyscale serving metrics; vLLM metrics (queue intervals, TTFT, TPOT)
