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




