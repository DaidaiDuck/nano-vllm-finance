# benchmarks/vllm_client.py
"""Open-loop load driver for a vLLM OpenAI-compatible server -- the vLLM side of the M4
nano-vs-vLLM comparison.

Fairness is the whole point. Rather than use vLLM's own benchmark_serving.py (which
re-samples and re-filters ShareGPT internally, so it would never pick the same prompts as
load_sharegpt), this driver:

  1. replays the *same specs file* nano ran (dump_specs / load_specs),
  2. uses the *same* generate_arrival_times schedule and seed,
  3. tokenizes each prompt with the *same* chat template and tokenizer, sending token ids so
     there is no re-encoding discrepancy,
  4. scores the result with the *same* compute_serving_metrics.

That leaves the engine as the only variable between the two runs.

Unlike serving_benchmark.py (single-threaded, one in-flight step loop over an in-process
engine), this talks to an HTTP server, so it needs real concurrency: each request is an
asyncio task that sleeps until its arrival time, then streams a completion.

Prereqs on the pod:
    pip install vllm aiohttp
    vllm serve Qwen/Qwen2.5-3B-Instruct --disable-log-requests

Usage (feed it the SAME specs file nano wrote):
    python -m benchmarks.vllm_client \
        --model Qwen/Qwen2.5-3B-Instruct \
        --specs-file benchmarks/baselines/specs.json \
        --qps 1 2 4 8 inf
"""
import argparse
import asyncio
import json
import time

from benchmarks.arrival import generate_arrival_times
from benchmarks.datasets import load_specs
from benchmarks.metrics import ServingMetrics, ServingRecord, compute_serving_metrics


def _format_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Apply the same chat template nano uses and return token ids.

    Sending ids (not a string) to /v1/completions removes any chance of the server
    tokenizing differently than nano did, so both engines see byte-identical input tokens.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text)


async def _run_one(session, url, model, prompt_ids, out_len, arrival_delay, t0, sem):
    """Fire one request at its scheduled arrival time and stream the completion.

    Returns a ServingRecord, or None if the request errored (so the run can continue).
    """
    # Open-loop: wait until this request's arrival instant, regardless of what else is
    # in flight. `sem` caps simultaneous sockets so a huge qps=inf burst does not exhaust
    # file descriptors -- it bounds concurrency, not arrival timing.
    await asyncio.sleep(max(0.0, arrival_delay - (time.perf_counter() - t0)))

    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": out_len,
        "temperature": 0.0,  # greedy, matching nano
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    async with sem:
        arrival_t = time.perf_counter() - t0
        first_token_t = None
        n_tokens = 0
        try:
            async with session.post(url, json=payload) as resp:
                async for raw in resp.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if choices and choices[0].get("text"):
                        if first_token_t is None:
                            first_token_t = time.perf_counter() - t0
                        n_tokens += 1
        except Exception as e:  # noqa: BLE001 - one bad request must not kill the sweep
            print(f"  request errored: {e}")
            return None

    finish_t = time.perf_counter() - t0
    if first_token_t is None:  # server produced nothing (e.g. immediate stop)
        return None
    return ServingRecord(
        request_id="",  # positional id filled in by the caller
        arrival_t=arrival_t,
        first_token_t=first_token_t,
        finish_t=finish_t,
        output_len=n_tokens,
    )


async def _run_sweep(base_url, model, specs, qps, seed, slo_ttft, slo_tpot):
    import aiohttp
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_ids = [_format_prompt_ids(tokenizer, prompt) for prompt, _ in specs]
    arrivals = generate_arrival_times(len(specs), qps, seed=seed)

    url = base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(256)  # cap concurrent sockets, not arrival timing

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [
            _run_one(session, url, model, prompt_ids[i], specs[i][1], arrivals[i], t0, sem)
            for i in range(len(specs))
        ]
        results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - t0

    records = []
    for i, rec in enumerate(results):
        if rec is not None:
            rec.request_id = str(i)
            records.append(rec)

    return compute_serving_metrics(
        f"vllm_qps{qps if qps != float('inf') else 'inf'}",
        records, duration, qps, slo_ttft=slo_ttft, slo_tpot=slo_tpot,
    )


def main():
    parser = argparse.ArgumentParser(description="Open-loop load driver against a vLLM server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--specs-file", default="benchmarks/baselines/specs.json",
        help="the SAME specs file nano ran, so both backends use identical prompts",
    )
    parser.add_argument("--qps", nargs="+", default=["1", "2", "4", "8", "inf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slo-ttft", type=float, default=0.2)
    parser.add_argument("--slo-tpot", type=float, default=0.05)
    args = parser.parse_args()

    specs = load_specs(args.specs_file)
    print(f"Loaded {len(specs)} specs from {args.specs_file}")

    from benchmarks.reporter import Reporter

    results = {}
    for raw_qps in args.qps:
        qps = float("inf") if raw_qps == "inf" else float(raw_qps)
        metrics: ServingMetrics = asyncio.run(
            _run_sweep(args.base_url, args.model, specs, qps,
                       args.seed, args.slo_ttft, args.slo_tpot)
        )
        metrics.print_report()
        results[f"qps_{raw_qps}"] = metrics

    Reporter().save_results(results, engine_name="vllm", version="m4")


if __name__ == "__main__":
    main()
