import math

import numpy as np
from dataclasses import dataclass
import time

@dataclass
class RequestMetrics: 
    """单个请求的指标"""
    prompt_len: int      # tokens
    output_len: int      # tokens 
    total_time: float    # End-to-end time (s)  non-default fields must come before fields that have a default.
    ttft: float = 0.0    # Time To First Token (s)
    tpot: float = 0.0    # Time Per Output Token (s)
    request_id: str = ""

@dataclass
class ScenarioMetrics:
    """场景级别的指标"""
    scenario_name: str
    num_requests: int 

    # Throughput
    output_throughput: float # output tokens / s
    total_throughput: float # all tokens / s 
    request_throughput: float # requests / s

    # Latency (end-to-end)
    avg_latency: float 
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float 

    # TTFT
    avg_ttft: float
    p50_ttft: float
    p95_ttft: float
    p99_ttft: float

    # TPOT 
    avg_tpot: float 
    p50_tpot: float

    # Meta
    avg_prompt_len: float
    avg_output_len: float 
    total_duration: float 

    def to_dict(self):
        return dict(self.__dict__)
    
    def print_report(self):
        print(f"\n{'='*60}")
        print(f"Scenario: {self.scenario_name}")
        print(f"{'='*60}")
        print(f"Requests: {self.num_requests}")
        print(f"Total duration: {self.total_duration:.2f}s")
        print(f"")
        print(f"Throughput:")
        print(f"  Output tokens/s: {self.output_throughput:.1f}")
        print(f"  Total tokens/s:  {self.total_throughput:.1f}")
        print(f"  Requests/s:      {self.request_throughput:.2f}")
        print(f"")
        print(f"End-to-end latency (s):")
        print(f"  Avg:  {self.avg_latency:.3f}")
        print(f"  P50:  {self.p50_latency:.3f}")
        print(f"  P95:  {self.p95_latency:.3f}")
        print(f"  P99:  {self.p99_latency:.3f}")
        print(f"  Max:  {self.max_latency:.3f}")
        print(f"")
        print(f"TTFT (s):")
        print(f"  Avg:  {self.avg_ttft:.3f}")
        print(f"  P50:  {self.p50_ttft:.3f}")
        print(f"  P99:  {self.p99_ttft:.3f}")
        print(f"")
        print(f"TPOT (ms/token):")
        print(f"  Avg:  {self.avg_tpot*1000:.2f}")
        print(f"  P50:  {self.p50_tpot*1000:.2f}")

def compute_metrics(
        scenario_name: str, 
        request_metrics: list[RequestMetrics],
        total_duration: float
) -> ScenarioMetrics: 
    """从单请求指标算场景指标"""

    latencies = [r.total_time for r in request_metrics]
    ttfts = [r.ttft for r in request_metrics]
    tpots = [r.tpot for r in request_metrics]
    output_lens = [r.output_len for r in request_metrics]
    prompt_lens = [r.prompt_len for r in request_metrics]

    total_output_tokens = sum(output_lens)
    total_input_tokens = sum(prompt_lens)

    return ScenarioMetrics(
        scenario_name=scenario_name,
        num_requests=len(request_metrics),

        # Throughput (基于 total duration, 不是 latencies 之和)
        output_throughput=total_output_tokens/total_duration,
        total_throughput=(total_input_tokens + total_output_tokens) / total_duration,
        request_throughput=len(request_metrics) / total_duration,

        # Latency 
        avg_latency=np.mean(latencies),
        p50_latency=np.percentile(latencies, 50), 
        p95_latency=np.percentile(latencies, 95),
        p99_latency=np.percentile(latencies, 99), 
        max_latency=max(latencies),

        # TTFT
        avg_ttft=np.mean(ttfts),
        p50_ttft=np.percentile(ttfts, 50), 
        p95_ttft=np.percentile(ttfts, 95),
        p99_ttft=np.percentile(ttfts, 99), 

        # TPOT 
        avg_tpot=np.mean(tpots),
        p50_tpot=np.percentile(tpots,50),
        
        # Meta 
        avg_prompt_len=np.mean(prompt_lens),
        avg_output_len=np.mean(output_lens),
        total_duration=total_duration,
    )


# ======================================================================================
# Open-loop serving metrics (M4)
#
# Everything above measures a closed-loop, one-request-at-a-time run: a request never waits
# behind another, so its TTFT is pure prefill compute. The types below are for concurrent,
# open-loop load, where the numbers that actually matter in production live: queueing delay
# folded into TTFT, tail percentiles, and goodput.
# ======================================================================================

@dataclass
class ServingRecord:
    """Timestamps for one request under open-loop load. All times are seconds since t0."""
    request_id: str
    arrival_t: float      # when it was submitted to the engine
    first_token_t: float  # when its first output token appeared
    finish_t: float       # when it finished
    output_len: int
    prompt_len: int = 0

    @property
    def ttft(self) -> float:
        """Time to first token, measured from arrival -- so it *includes queueing*.

        This is the difference that matters versus the closed-loop TTFT above, which starts
        the clock when the engine begins work and therefore never sees a queue.
        """
        return self.first_token_t - self.arrival_t

    @property
    def e2e(self) -> float:
        """End-to-end latency as the user experiences it: arrival to completion."""
        return self.finish_t - self.arrival_t

    @property
    def tpot(self) -> float:
        """Mean time per output token after the first one.

        Undefined for a single-token output (there is no inter-token gap to measure), which
        is reported as 0.0 rather than dividing by zero.
        """
        if self.output_len <= 1:
            return 0.0
        return (self.finish_t - self.first_token_t) / (self.output_len - 1)


@dataclass
class ServingMetrics:
    """Aggregate metrics for one open-loop run at a fixed arrival rate."""
    scenario_name: str
    qps: float
    num_requests: int
    total_duration: float

    # Throughput
    output_throughput: float   # output tokens / s
    request_throughput: float  # completed requests / s
    goodput: float             # SLO-compliant requests / s  <- the production headline

    # Latency distributions
    avg_ttft: float
    p50_ttft: float
    p95_ttft: float
    p99_ttft: float

    avg_tpot: float
    p50_tpot: float
    p95_tpot: float
    p99_tpot: float

    avg_e2e: float
    p50_e2e: float
    p95_e2e: float
    p99_e2e: float

    # SLO definition this run was scored against
    slo_ttft: float
    slo_tpot: float
    num_slo_met: int

    def to_dict(self):
        return dict(self.__dict__)

    def print_report(self):
        qps = "inf" if math.isinf(self.qps) else f"{self.qps:g}"
        print(f"\n{'='*64}")
        print(f"Serving: {self.scenario_name}   (QPS={qps})")
        print(f"{'='*64}")
        print(f"Requests: {self.num_requests}    Duration: {self.total_duration:.2f}s")
        print(f"Output tokens/s: {self.output_throughput:.1f}    "
              f"Requests/s: {self.request_throughput:.2f}")
        print(f"Goodput: {self.goodput:.2f} req/s  "
              f"({self.num_slo_met}/{self.num_requests} met "
              f"TTFT<={self.slo_ttft*1000:.0f}ms and TPOT<={self.slo_tpot*1000:.0f}ms)")
        print("")
        for label, p50, p95, p99 in (
            ("TTFT", self.p50_ttft, self.p95_ttft, self.p99_ttft),
            ("TPOT", self.p50_tpot, self.p95_tpot, self.p99_tpot),
            ("E2E ", self.p50_e2e, self.p95_e2e, self.p99_e2e),
        ):
            print(f"{label} (ms)   P50 {p50*1000:9.1f}   P95 {p95*1000:9.1f}   "
                  f"P99 {p99*1000:9.1f}")


def compute_serving_metrics(
    scenario_name: str,
    records: list[ServingRecord],
    total_duration: float,
    qps: float,
    slo_ttft: float = 0.2,   # 200 ms to first token
    slo_tpot: float = 0.05,  # 50 ms per output token
) -> ServingMetrics:
    """Aggregate per-request records into serving metrics.

    Goodput counts only requests that satisfy *every* SLO, divided by wall-clock time. That
    is the number worth optimising: throughput that ignores latency can always be raised by
    batching harder until every user experience is bad.

    Throughput denominators are wall-clock duration, never the sum of per-request latencies
    -- under concurrency those latencies overlap and summing them inflates the result.
    """
    if not records:
        raise ValueError("no records to aggregate")

    ttfts = [r.ttft for r in records]
    tpots = [r.tpot for r in records]
    e2es = [r.e2e for r in records]
    total_output_tokens = sum(r.output_len for r in records)

    num_slo_met = sum(1 for r in records if r.ttft <= slo_ttft and r.tpot <= slo_tpot)

    def pct(values, p):
        return float(np.percentile(values, p))

    return ServingMetrics(
        scenario_name=scenario_name,
        qps=qps,
        num_requests=len(records),
        total_duration=total_duration,

        output_throughput=total_output_tokens / total_duration,
        request_throughput=len(records) / total_duration,
        goodput=num_slo_met / total_duration,

        avg_ttft=float(np.mean(ttfts)),
        p50_ttft=pct(ttfts, 50), p95_ttft=pct(ttfts, 95), p99_ttft=pct(ttfts, 99),

        avg_tpot=float(np.mean(tpots)),
        p50_tpot=pct(tpots, 50), p95_tpot=pct(tpots, 95), p99_tpot=pct(tpots, 99),

        avg_e2e=float(np.mean(e2es)),
        p50_e2e=pct(e2es, 50), p95_e2e=pct(e2es, 95), p99_e2e=pct(e2es, 99),

        slo_ttft=slo_ttft,
        slo_tpot=slo_tpot,
        num_slo_met=num_slo_met,
    )




