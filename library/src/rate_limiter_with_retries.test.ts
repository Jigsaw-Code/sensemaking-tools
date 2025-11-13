import { RpmLimiter, with429Retry } from "./rate_limiter_with_retries";

jest.useFakeTimers();

describe("rate_limiter_with_retries", () => {
  beforeEach(() => {
    jest.clearAllTimers();
  });

  it("with429Retry should retry on 429 with backoff and eventually succeed", async () => {
    const fn = jest
      .fn()
      .mockRejectedValueOnce({ status: 429, message: "Too Many Requests" })
      .mockResolvedValueOnce("ok") as jest.Mock<Promise<string>, []>;

    // Make jitter deterministic
    const randSpy = jest.spyOn(Math, "random").mockReturnValue(0);

    const promise = with429Retry(fn, { baseDelayMs: 500, maxRetries: 3 });

    // Allow the backoff timer (500ms) to elapse
    await jest.advanceTimersByTimeAsync(500);

    const res = await promise;
    expect(res).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(2);

    randSpy.mockRestore();
  });

  it("RpmLimiter should enforce max concurrency", async () => {
    let inFlight = 0;
    let peak = 0;

    const makeTask = (id: number) => async () => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      // Simulate 100ms work
      await new Promise((res) => setTimeout(res, 100));
      inFlight--;
      return id;
    };

    const limiter = new RpmLimiter(1000, 2);

    const promises = [0, 1, 2, 3, 4].map((i) => limiter.schedule(makeTask(i)));

    // Flush all timers deterministically
    await jest.advanceTimersByTimeAsync(1000);

    const results = await Promise.all(promises);
    expect(results.sort()).toEqual([0, 1, 2, 3, 4]);
    expect(peak).toBeLessThanOrEqual(2);
  });
});
