# benchmarks/m1_benchmark.py
# Run as a module from the repo root so the `benchmarks` and `nano_vllm`
# packages resolve:
#   python -m benchmarks.m1_benchmark --scenarios short_chat medium_chat long_context

import argparse
from nano_vllm import LLM
from transformers import AutoTokenizer
from benchmarks.scenarios import SCENARIOS
from benchmarks.runner import BenchmarkRunner
from benchmarks.reporter import Reporter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="nano_vllm")
    parser.add_argument("--version", default="m1")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    # 初始化
    print(f"Loading model: {args.model}")
    llm = LLM(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    runner = BenchmarkRunner(llm, tokenizer)
    reporter = Reporter() 

    # 跑所有指定场景
    results = {}
    for name in args.scenarios: 
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            continue
            
        scenario = SCENARIOS[name] 
        metrics = runner.run_scenario(scenario)
        metrics.print_report()
        results[name] = metrics 
    
    # 保存
    reporter.save_results(results, args.engine, args.version)

if __name__ == "__main__":
    main()


