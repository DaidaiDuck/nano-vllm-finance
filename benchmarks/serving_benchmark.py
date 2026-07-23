# benchmarks/serving_benchmark.py
"""Open-loop concurrent serving benchmark for M4.

How this differs from benchmarks/runner.py: that one is *closed-loop* (send the next request
only after the previous finishes) and measures single-request latency. This one is
*open-loop* -- requests are submitted on a fixed arrival schedule regardless of what the
engine is doing -- which is the only way to observe queueing, tail latency and goodput.
Only an engine with continuous batching can be measured this way at all.

The driver is single-threaded: between engine steps it submits whichever requests have come
due. No threads means no lock contention with the GPU, and the polling granularity is one
step (tens of milliseconds during decode), which is fine for QPS-scale load.

Timestamps are honest without extra synchronisation: the sampler calls .item(), which forces
a CUDA sync, so step() has already waited for the GPU by the time it returns.

Usage:
    python -m benchmarks.serving_benchmark --model Qwen/Qwen2.5-3B-Instruct \
        --num-requests 64 --qps 1 2 4 8 inf
"""
import argparse
import time

from benchmarks.arrival import generate_arrival_times
from benchmarks.metrics import ServingMetrics, ServingRecord, compute_serving_metrics
from nano_vllm.core.types import Request, SamplingParams


def run_serving_benchmark(
    engine,
    specs: list[tuple[str, int]],
    qps: float,
    scenario_name: str = "sharegpt",
    slo_ttft: float = 0.2,
    slo_tpot: float = 0.05,
    seed: int = 42,
) -> ServingMetrics:
    """Drive `engine` with `specs` arriving at `qps` and return the aggregated metrics.

    Args:
        engine: a PagedEngine (needs add_request / step / has_unfinished_requests).
        specs: (prompt_text, output_len) pairs. Each request runs to *its own* output_len,
            which is what makes throughput and goodput meaningful -- real traffic has a
            distribution of response lengths, not one hardcoded value.
        qps: arrival rate; float("inf") submits everything at once (saturated mode).
    """
    arrivals = generate_arrival_times(len(specs), qps, seed=seed)

    arrival_t: dict[str, float] = {}
    first_token_t: dict[str, float] = {}
    finish_t: dict[str, float] = {}
    output_len: dict[str, int] = {}
    prompt_len: dict[str, int] = {}

    next_idx = 0
    t0 = time.perf_counter()

    def now() -> float:
        return time.perf_counter() - t0

    while next_idx < len(specs) or engine.has_unfinished_requests():
        # 1. Submit everything that has come due. Note this does not depend on completions
        #    in any way -- that independence is what "open-loop" means.
        while next_idx < len(specs) and arrivals[next_idx] <= now():
            prompt, out_len = specs[next_idx]
            request_id = str(next_idx)
            token_ids = engine.tokenizer.encode(engine._format_prompt(prompt))
            engine.add_request(
                Request(
                    request_id,
                    token_ids,
                    SamplingParams(temperature=0.0, max_tokens=out_len),
                )
            )
            arrival_t[request_id] = now()
            prompt_len[request_id] = len(token_ids)
            next_idx += 1

        # 2. Nothing in flight: sleep until the next arrival. At low QPS the engine is
        #    genuinely idle between requests, and that idle time belongs in the wall clock.
        if not engine.has_unfinished_requests():
            time.sleep(max(0.0, arrivals[next_idx] - now()))
            continue

        # 3. Advance one step and timestamp whatever came out.
        outputs = engine.step()
        t = now()
        for out in outputs:
            request_id = out.request_id
            # A request appears in `outputs` exactly when it produced a token this step:
            # requests still working through a chunked prefill are skipped by
            # update_from_output, so they never show up early. That makes "first appearance"
            # the correct trigger for the first-token timestamp.
            first_token_t.setdefault(request_id, t)
            output_len[request_id] = len(out.token_ids)
            if out.finished:
                finish_t[request_id] = t

    duration = now()

    records = [
        ServingRecord(
            request_id=rid,
            arrival_t=arrival_t[rid],
            first_token_t=first_token_t[rid],
            finish_t=finish_t[rid],
            output_len=output_len[rid],
            prompt_len=prompt_len[rid],
        )
        for rid in sorted(finish_t, key=int)
    ]
    return compute_serving_metrics(
        scenario_name, records, duration, qps, slo_ttft=slo_ttft, slo_tpot=slo_tpot
    )


def main():
    parser = argparse.ArgumentParser(description="M4 open-loop serving benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--qps", nargs="+", default=["1", "2", "4", "8", "inf"],
                        help="arrival rates to sweep; 'inf' means saturated")
    parser.add_argument("--dataset", choices=["sharegpt", "synthetic"], default="sharegpt")
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--slo-ttft", type=float, default=0.2, help="TTFT SLO in seconds")
    parser.add_argument("--slo-tpot", type=float, default=0.05, help="TPOT SLO in seconds")
    parser.add_argument(
        "--specs-file", default="benchmarks/baselines/specs.json",
        help="shared (prompt, output_len) specs; reused if it exists, generated otherwise. "
             "Feed this same file to vllm_client.py so both backends run identical prompts.",
    )
    args = parser.parse_args()

    from benchmarks.datasets import (
        dump_specs, generate_synthetic_prompts, load_sharegpt, load_specs,
    )
    from benchmarks.reporter import Reporter
    from nano_vllm.paged.engine import PagedEngine

    engine = PagedEngine(args.model)

    # Reuse an existing specs file so nano and vLLM run the exact same prompts; only generate
    # (and persist) a fresh set when the file is absent.
    import os
    if os.path.exists(args.specs_file):
        specs = load_specs(args.specs_file)
        print(f"Loaded {len(specs)} specs from {args.specs_file}")
    else:
        if args.dataset == "sharegpt":
            specs = load_sharegpt(
                args.num_requests, engine.tokenizer, max_output_len=args.max_output_len
            )
        else:
            prompts = generate_synthetic_prompts(args.num_requests, (50, 150), engine.tokenizer)
            specs = [(p, 100) for p in prompts]
        dump_specs(specs, args.specs_file)
        print(f"Wrote {len(specs)} specs to {args.specs_file}")

    results = {}
    for raw_qps in args.qps:
        qps = float("inf") if raw_qps == "inf" else float(raw_qps)
        metrics = run_serving_benchmark(
            engine,
            specs,
            qps,
            scenario_name=f"{args.dataset}_qps{raw_qps}",
            slo_ttft=args.slo_ttft,
            slo_tpot=args.slo_tpot,
        )
        metrics.print_report()
        results[f"qps_{raw_qps}"] = metrics

    Reporter().save_results(results, engine_name="nano-vllm", version="m4")


if __name__ == "__main__":
    main()
