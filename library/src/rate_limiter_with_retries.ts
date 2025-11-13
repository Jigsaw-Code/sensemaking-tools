// typescript

export type RetryOpts = {
  maxRetries?: number;
  baseDelayMs?: number; // for backoff if Retry-After not provided
  maxDelayMs?: number;
};

export class RpmLimiter {
  private readonly rpm: number;
  private readonly maxConcurrent: number;
  private readonly windowMs = 60_000;
  private readonly startTimes: number[] = []; // timestamps of started requests (sliding window)
  private running = 0;

  // Serialize acquisition attempts to avoid race conditions where many callers
  // simultaneously pass the checks and exceed maxConcurrent.
  private waiters: Array<() => void> = [];

  constructor(rpm: number, maxConcurrent: number) {
    if (rpm <= 0) throw new Error("rpm must be > 0");
    if (maxConcurrent <= 0) throw new Error("maxConcurrent must be > 0");
    this.rpm = rpm;
    this.maxConcurrent = maxConcurrent;
  }

  async schedule<T>(task: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await task();
    } finally {
      this.release();
    }
  }

  private async acquire(): Promise<void> {
    // Only the head of the queue is allowed to attempt to acquire a slot.
    await this.enqueue();
    while (true) {
      const now = Date.now();
      this.trimWindow(now);

      const withinRpm = this.startTimes.length < this.rpm;
      const withinConcurrency = this.running < this.maxConcurrent;

      if (withinRpm && withinConcurrency) {
        // Atomically claim the slot before allowing the next waiter to proceed.
        this.running++;
        this.startTimes.push(now);
        this.dequeueAndWakeNext();
        return;
      }

      // Compute delay: if RPM is the blocker, wait until the oldest timestamp exits the window.
      const delayForRpm = withinRpm ? 0 : Math.max(1, this.windowMs - (now - this.startTimes[0]!));
      // If concurrency is the blocker, short sleep and recheck.
      const delayForConcurrency = withinConcurrency ? 0 : 25;
      const delay = Math.max(delayForRpm, delayForConcurrency);
      await sleep(delay);
    }
  }

  private release(): void {
    this.running = Math.max(0, this.running - 1);
    // Wake the head waiter to re-attempt immediately (especially if concurrency was the blocker).
    this.wakeHead();
  }

  private enqueue(): Promise<void> {
    return new Promise((resolve) => {
      this.waiters.push(resolve);
      if (this.waiters.length === 1) {
        // No one is ahead of us; proceed immediately.
        resolve();
      }
    });
  }

  private dequeueAndWakeNext(): void {
    // Remove current head (us) and wake next waiter, if any, to begin its acquisition loop.
    this.waiters.shift();
    this.wakeHead();
  }

  private wakeHead(): void {
    const next = this.waiters[0];
    if (next) next();
  }

  private trimWindow(now: number): void {
    while (this.startTimes.length && this.startTimes[0] <= now - this.windowMs) {
      this.startTimes.shift();
    }
  }
}

export async function with429Retry<T>(fn: () => Promise<T>, opts: RetryOpts = {}): Promise<T> {
  const { maxRetries = 5, baseDelayMs = 500, maxDelayMs = 10_000 } = opts;

  let attempt = 0;
   
  while (true) {
    try {
      return await fn();
    } catch (err: unknown) {
      const { shouldRetry, retryAfterMs } = getRetryInfo(err);
      if (!shouldRetry || attempt >= maxRetries) throw err;
      const backoff =
        (retryAfterMs ?? Math.min(maxDelayMs, baseDelayMs * 2 ** attempt)) + Math.random() * 200;
      attempt++;
      await sleep(backoff);
    }
  }
}

function getRetryInfo(err: unknown): { shouldRetry: boolean; retryAfterMs?: number } {
  // Generic detection for REST/gRPC style rate limit errors
  const status =
    typeof err === "object" && err !== null
      ? ((err as { status?: unknown; code?: unknown }).status ?? (err as { code?: unknown }).code)
      : undefined;
  const message: string =
    typeof err === "object" &&
    err !== null &&
    "message" in err &&
    typeof (err as { message?: unknown }).message === "string"
      ? (err as { message: string }).message
      : "";

  // 429 (Too Many Requests) or gRPC RESOURCE_EXHAUSTED (8)
  const isRateLimited =
    status === 429 || status === "429" || status === 8 || /RESOURCE_EXHAUSTED/i.test(message);

  let retryAfterMs: number | undefined;

  // Try to parse Retry-After header (seconds or HTTP-date)
  const headers =
    typeof err === "object" && err !== null
      ? (err as {
          response?: { headers?: Record<string, unknown> };
          headers?: Record<string, unknown>;
        })
      : undefined;
  const retryAfterValue =
    headers?.response?.headers?.["retry-after"] ??
    headers?.response?.headers?.["Retry-After"] ??
    headers?.headers?.["retry-after"];

  if (typeof retryAfterValue === "string" || typeof retryAfterValue === "number") {
    const asNumber = Number(retryAfterValue);
    if (!Number.isNaN(asNumber)) {
      retryAfterMs = asNumber * 1000;
    } else {
      const dateMs = Date.parse(String(retryAfterValue));
      if (!Number.isNaN(dateMs)) {
        retryAfterMs = Math.max(0, dateMs - Date.now());
      }
    }
  }

  return { shouldRetry: !!isRateLimited, retryAfterMs };
}

function sleep(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms));
}

// Convenience: run a batch respecting both RPM and concurrency, with 429-aware retries.
export async function runRpmLimited<T>(
  callbacks: Array<() => Promise<T>>,
  { rpm, maxConcurrency, retryOpts }: { rpm: number; maxConcurrency: number; retryOpts?: RetryOpts }
): Promise<T[]> {
  const limiter = new RpmLimiter(rpm, maxConcurrency);
  return Promise.all(callbacks.map((cb) => limiter.schedule(() => with429Retry(cb, retryOpts))));
}
