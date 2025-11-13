# Load Test Bottlenecks Resolved (30‑User Simulation)

Last updated: 2025-11-13 15:43

This document focuses exclusively on the bottlenecks uncovered during the 30‑user load simulation and the changes that resolved them. It highlights the exponential backoff retry mechanism, rate limiting, and the load‑test simulation you can run to validate behavior end‑to‑end.

## Summary of bottlenecks observed

- Rate limit bursts (429 / RESOURCE_EXHAUSTED)
  - Concurrent calls could exceed provider quotas, causing transient failures and batch retries.
- Concurrency spikes beyond configured maximums
  - Non‑atomic slot checks allowed brief surges above `maxConcurrent`.
- Long‑tail latencies under load
  - Lack of backoff/jitter coordination made collisions more likely; retry storms increased tail latency.
- Limited visibility into real request volume and timing
  - No per‑request logs or easy dashboards to understand throughput and errors.

## What we changed

### 1) Exponential backoff + retry (highlight)

- Function: `with429Retry<T>(fn, opts?)` in `library/src/rate_limiter_with_retries.ts`
- Behavior:
  - Detects rate‑limit conditions (HTTP 429, gRPC `RESOURCE_EXHAUSTED`/code 8, or matching message).
  - Honors `Retry-After` header when present (seconds or HTTP‑date).
  - Otherwise applies exponential backoff with jitter:
    - `delay = min(maxDelayMs, baseDelayMs * 2^attempt) + random(0..200ms)`
    - Defaults: `maxRetries=5`, `baseDelayMs=500`, `maxDelayMs=10_000`.
- Result: Reduces collision probability and prevents immediate retry storms; stabilizes success rate under quota pressure.

### 2) Concurrency + RPM limiting (race fix)

- Class: `RpmLimiter` in `library/src/rate_limiter_with_retries.ts`
- Key improvement: Queue‑based, atomic slot acquisition
  - Only the head waiter attempts acquisition; on success it increments `running` and records a start timestamp before waking the next.
  - Prevents temporary spikes above `maxConcurrent`.
  - Enforces both sliding‑window RPM and parallelism caps.
- Helper: `runRpmLimited(callbacks, { rpm, maxConcurrency, retryOpts })`
  - Schedules a batch using `RpmLimiter` and wraps each task with `with429Retry`.

### 3) Application‑level batching entrypoint

- Function: `executeConcurrently(callbacks, opts?)` in `library/src/sensemaker_utils.ts`
- Modes:
  - Legacy parallel (no options): simple `Promise.all` behavior.
  - Retry mode: `enableRetry: true` applies `with429Retry` to each task in parallel.
  - RPM + concurrency: `{ rpm, maxConcurrency, retryOpts }` uses `runRpmLimited`.

### 4) Real request metrics + dashboard

- Wrapper model: `MetricsVertexModel` logs each Vertex request start/end.
- Logger: `VertexMetrics` writes JSONL records with `tsStart`, `tsEnd`, `status`, `latencyMs`, `userId`.
- Report: `library/runner-cli/report_vertex_metrics.ts` builds an HTML dashboard showing
  - Requests over time (total/ok/error per 1s bucket)
  - Latency p50/p90/p99 over time
  - Overall summary cards

## How to simulate 30 users (for validation)

1) Authenticate to Vertex AI:
```
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service_account.json"
```

2) Run the load test (adjust project and paths):
```
npx ts-node ./library/runner-cli/vertex_loadtest.ts \
  --users 30 \
  --iterations 1 \
  --vertexProject "YOUR_GCP_PROJECT_ID" \
  --inputFile ./sample_input.csv \
  --outDir ./.out \
  --rpm 120 \
  --maxConcurrency 8 \
  --additionalContext "Short description of the conversation"
```

Notes:
- `--rpm` and `--maxConcurrency` enable provider‑friendly throttling; omit them to rely on per‑call retry/backoff only.
- Traffic is real and billable; tune parameters to fit quota.

3) Inspect outputs:
- Raw logs: `./.out/vertex_requests.jsonl`
- Dashboard: `./.out/dashboard.html`

## Expected outcomes after fixes

- Fewer 429s and automatic recovery via backoff when they occur.
- Peak concurrency stays at or below configured `maxConcurrency`.
- Smoother request rate within the RPM budget; reduced long‑tail latency.
- Clear visibility into request volumes, errors, and latency distributions.

## Configuration quick reference

- `with429Retry` options:
  - `maxRetries` (default 5)
  - `baseDelayMs` (default 500)
  - `maxDelayMs` (default 10_000)

- `executeConcurrently` options:
  - `enableRetry: true` to apply per‑call backoff with no global throttling
  - `rpm: number` and `maxConcurrency: number` to enforce budgets
  - `retryOpts` to pass through to `with429Retry`

## Files touched (relevant to these fixes)

- `library/src/rate_limiter_with_retries.ts` — backoff + RpmLimiter with atomic acquisition
- `library/src/sensemaker_utils.ts` — `executeConcurrently` options for retry/throttle
- `library/src/models/metrics_vertex_model.ts` — per‑request logging
- `library/src/metrics/vertex_metrics.ts` — JSONL write with `tsStart`/`tsEnd`/latency
- `library/runner-cli/vertex_loadtest.ts` — 30‑user simulation harness
- `library/runner-cli/report_vertex_metrics.ts` — dashboard generator

## Troubleshooting tips

- Seeing many 429s? Lower `--rpm` and/or `--maxConcurrency`; check `Retry-After` handling in logs.
- Long‑tail latencies? Ensure backoff/jitter defaults are in place; consider reducing concurrency.
- No logs? Confirm `VertexMetrics.init(outDir)` runs (the load test does this for you).

---

