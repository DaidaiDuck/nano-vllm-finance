# benchmarks/m2_benchmark.py
# Run as a module from the repo root:
#   python -m benchmarks.m2_benchmark --cache custom --scenarios short_chat medium_chat long_context
#
#   python -m benchmarks.m2_benchmark --cache custom --version m2  --scenarios short_chat medium_chat long_context  --baseline benchmarks/results/nano_vllm_m1_rerun_<时间戳>.json

import argparse
import json

import torch

from nano_vllm import LLM
from transformers import AutoTokenizer
from benchmarks.scenarios import SCENARIOS
from benchmarks.runner import BenchmarkRunner
from benchmarks.reporter import Reporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="nano_vllm")
    parser.add_argument("--version", default="m2")
    parser.add_argument("--scenarios", nargs="+", default=["short_chat", "medium_chat", "long_context"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--cache",
        choices=["custom", "hf"],
        default="custom",
        help="KV cache backend: 'custom' = MyKVCache (M2), 'hf' = DynamicCache (M1). "
             "Run once with each on the same scenarios for a fair M1-vs-M2 parity check.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to an M1 result JSON; if given, print an M1 -> M2 comparison.",
    )
    args = parser.parse_args()

    # 初始化
    print(f"Loading model: {args.model}  (cache backend: {args.cache})")
    llm = LLM(args.model, use_custom_cache=(args.cache == "custom"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    runner = BenchmarkRunner(llm, tokenizer)
    reporter = Reporter()

    # 跑指定场景
    results = {}
    peak_mem_mb = {}
    for name in args.scenarios:
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            continue

        scenario = SCENARIOS[name]

        # One honest memory number: peak allocated over the scenario. Reset the
        # high-water mark first so it's per-scenario.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        metrics = runner.run_scenario(scenario)
        metrics.print_report()
        results[name] = metrics

        if torch.cuda.is_available():
            peak_mem_mb[name] = torch.cuda.max_memory_allocated() / 1e6
            print(f"  peak GPU memory (allocated): {peak_mem_mb[name]:.1f} MB")

    # 保存延迟/吞吐结果 (JSON) — 用于 parity 对比
    filepath = reporter.save_results(results, args.engine, args.version)

    # 峰值显存 (memory tradeoff): 单请求下 MyKVCache 因预留 max_seq_len 反而更高
    if peak_mem_mb:
        print("\nPeak GPU memory (MB) — expect M2 >= M1 (pre-allocation tradeoff):")
        for scen, mb in peak_mem_mb.items():
            print(f"  {scen:<16} {mb:.1f}")
        mem_path = filepath.with_name(filepath.stem + "_peakmem.json")
        with open(mem_path, "w") as f:
            json.dump(peak_mem_mb, f, indent=2)
        print(f"Peak memory saved to: {mem_path}")

    # 可选: 和 M1 baseline 对比 (吞吐/延迟/TTFT/TPOT). 期望 ≈ 相等 (parity, 无退化).
    if args.baseline:
        reporter.compare(args.baseline, filepath)


if __name__ == "__main__":
    main()
