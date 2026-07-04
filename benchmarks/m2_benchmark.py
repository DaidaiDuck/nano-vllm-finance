# benchmarks/m2_benchmark.py
# Run as a module from the repo root so the packages resolve:
#   python -m benchmarks.m2_benchmark --scenarios short_chat medium_chat long_context
#
# M2 = custom MyKVCache replacing HF's DynamicCache. This mirrors m1_benchmark.py
# but adds the two things that make an M2 run meaningful:
#   (a) tags results as "m2",
#   (b) records PEAK GPU MEMORY per scenario (the M2 headline metric, see
#       docs/design/m2_design.md §7 Table B), and
#   (c) optionally prints an M1 -> M2 comparison via --baseline <m1_result.json>.
#
# NOTE: the benchmark drives LLM.generate_stream (for TTFT/TPOT). Make sure the
# engine's generate_stream actually uses MyKVCache, otherwise this measures HF's
# cache and the M1-vs-M2 comparison is meaningless.

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
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to an M1 result JSON; if given, print an M1 -> M2 comparison.",
    )
    args = parser.parse_args()

    # 初始化
    print(f"Loading model: {args.model}")
    llm = LLM(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    runner = BenchmarkRunner(llm, tokenizer)
    reporter = Reporter()

    # 跑所有指定场景
    results = {}
    peak_mem_mb = {}
    for name in args.scenarios:
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            continue

        scenario = SCENARIOS[name]

        # Peak GPU memory: reset the high-water mark right before the scenario, then
        # read it after. reset_peak_memory_stats() sets the peak to the currently
        # allocated bytes (model weights + pre-allocated KV cache), so the value we
        # read back = weights + cache + peak activations. Model weights are identical
        # across M1/M2, so the delta reflects the KV-cache behaviour we care about.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        metrics = runner.run_scenario(scenario)
        metrics.print_report()
        results[name] = metrics

        if torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated() / 1e6
            peak_mem_mb[name] = mb
            print(f"  Peak GPU memory: {mb:.1f} MB")

    # 保存标准延迟/吞吐结果 (JSON)
    filepath = reporter.save_results(results, args.engine, args.version)

    # 峰值显存汇总 (ScenarioMetrics 目前没有 memory 字段, 所以单独打印 + 存一个
    # sidecar JSON, 放在主结果文件旁边)
    if peak_mem_mb:
        print("\nPeak GPU memory (MB):")
        for scen, mb in peak_mem_mb.items():
            print(f"  {scen:<16} {mb:.1f}")
        mem_path = filepath.with_name(filepath.stem + "_peakmem.json")
        with open(mem_path, "w") as f:
            json.dump(peak_mem_mb, f, indent=2)
        print(f"Peak memory saved to: {mem_path}")

    # 可选: 和 M1 baseline 对比 (吞吐/延迟/TTFT/TPOT)
    if args.baseline:
        reporter.compare(args.baseline, filepath)


if __name__ == "__main__":
    main()
