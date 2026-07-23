# benchmarks/arrival.py
"""Arrival process: assign each request a timestamp, relative to t0, at which it is submitted.

This is what makes the M4 benchmark *open-loop*: requests are submitted on their schedule
regardless of whether earlier ones have finished. The M1-M3 benchmarks are closed-loop (send
the next request only after the previous one completes), which never produces a queue, so it
cannot show queueing delay or the saturation knee.
"""
import math
import random


def generate_arrival_times(
    num_requests: int,
    qps: float,
    seed: int = 42,
    distribution: str = "poisson",
) -> list[float]:
    """Return num_requests non-decreasing arrival times in seconds, starting at 0.0.

    Args:
        num_requests: how many timestamps to produce.
        qps: target arrival rate (lambda). ``float("inf")`` means every request arrives at
            t=0 -- the saturated / offline-throughput mode, equivalent to vLLM's
            ``--request-rate inf``.
        seed: seed for the Poisson draw, so a run is reproducible.
        distribution: "poisson" (gaps ~ Exponential(lambda), the usual model of real
            traffic) or "uniform" (a fixed 1/qps gap).

    Raises:
        ValueError: if qps is non-positive, or the distribution name is unknown.
    """
    if num_requests <= 0:
        return []
    if math.isinf(qps):
        return [0.0] * num_requests  # everything arrives at once = saturated
    if qps <= 0:
        raise ValueError(f"qps must be > 0 (or inf), got {qps}")

    if distribution == "uniform":
        return [i / qps for i in range(num_requests)]
    if distribution != "poisson":
        raise ValueError(f"unknown distribution: {distribution!r}")

    rng = random.Random(seed)
    times: list[float] = []
    t = 0.0
    for _ in range(num_requests):
        times.append(t)
        t += rng.expovariate(qps)  # inter-arrival gaps ~ Exp(lambda), mean 1/lambda
    return times
