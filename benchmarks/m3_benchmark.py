# benchmarks/m3_benchmark.py
# Run as a module from the repo root:
#   python -m benchmarks.m3_benchmark --scenarios short_chat medium_chat long_context
#
# Compare against M1 and M2 baselines (JSONs saved by their benchmarks):
#   python -m benchmarks.m3_benchmark \
#       --baseline-m1 benchmarks/results/nano_vllm_m1_<ts>.json \
#       --baseline-m2 benchmarks/results/nano_vllm_m2_<ts>.json
#
# NOTE: M3 (PagedAttention) needs flash-attn + a CUDA GPU -> run on the pod, not local Mac.
import argparse

from nano_vllm.paged.engine import LLM          # M3's own LLM (paged); NOT the M1/M2 LLM
from transformers import AutoTokenizer
from benchmarks.scenarios import SCENARIOS
from benchmarks.runner import BenchmarkRunner
from benchmarks.reporter import Reporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="nano_vllm")
    parser.add_argument("--version", default="m3")
    parser.add_argument("--scenarios", nargs="+", default=["short_chat", "medium_chat", "long_context"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--baseline-m1", default=None, help="M1 result JSON -> print M1->M3 comparison.")
    parser.add_argument("--baseline-m2", default=None, help="M2 result JSON -> print M2->M3 comparison.")
    args = parser.parse_args()

    print(f"Loading model: {args.model}  (engine: M3 PagedAttention)")
    llm = LLM(args.model)                     # M3 LLM signature: (model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    runner = BenchmarkRunner(llm, tokenizer)
    reporter = Reporter()

    results = {}
    for name in args.scenarios:
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            continue
        scenario = SCENARIOS[name]

        metrics = runner.run_scenario(scenario)
        metrics.print_report()
        results[name] = metrics

    filepath = reporter.save_results(results, args.engine, args.version)

    # Latency/throughput vs baselines. Expect M3 ~= M2 ~= M1 at single request;
    # the paged win is architectural (shows up under M4 batching, not here).
    if args.baseline_m1:
        print("\n=== M1 -> M3 ===")
        reporter.compare(args.baseline_m1, filepath)
    if args.baseline_m2:
        print("\n=== M2 -> M3 ===")
        reporter.compare(args.baseline_m2, filepath)


if __name__ == "__main__":
    main()
