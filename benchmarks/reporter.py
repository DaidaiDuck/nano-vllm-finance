import json
from pathlib import Path
from datetime import datetime

from benchmarks.metrics import ScenarioMetrics

class Reporter:
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True) 

    def save_results(
            self, 
            results: dict[str, ScenarioMetrics],
            engine_name: str, 
            version: str
    ): 
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{engine_name}_{version}_{timestamp}.json"
        filepath = self.output_dir / filename

        data = {
            "engine": engine_name, 
            "version": version, 
            "timestamp": timestamp, 
            "scenarios": {
                name: metrics.to_dict()
                for name, metrics in results.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2) 
        
        print(f"\nResults saved to: {filepath}")
        return filepath 
    
    def compare(self, baseline_file: str, current_file: str):
        """对比两个结果"""
        with open(baseline_file) as f: 
            baseline = json.load(f) 
        
        with open(current_file) as f:
            current = json.load(f) 
        
        print(f"\n{'='*70}")
        print(f"Comparison: {baseline['engine']} vs {current['engine']}")
        print(f"{'='*70}")

        for scenario in baseline['scenarios']:
            if scenario not in current['scenarios']:
                continue

            b = baseline['scenarios'][scenario]
            c = current['scenarios'][scenario]

            print(f"\nScenario: {scenario}")
            print(f"{'Metric':<25} {'Baseline':<15} {'Current':<15} {'Change':<10}")
            print("-" * 65)

            metrics_to_compare = [
                'output_throughput', 'p50_latency', 'p95_latency',
                'p99_latency', 'avg_ttft', 'avg_tpot','request_throughput'
            ]

            for metric in metrics_to_compare:
                b_val = b[metric]
                c_val = c[metric] 

                if metric in ["output_throughput", "request_throughput"]:
                    # higher is better
                    change_pct = (c_val - b_val) / b_val * 100 
                    direction = "↑" if change_pct > 0 else "↓"
                else:
                    # lower is better
                    change_pct = (b_val - c_val) / b_val * 100 
                    direction = "↑" if change_pct > 0 else "↓"

                print(f"{metric:<25} {b_val:<15.3f} {c_val:<15.3f} "
                    f"{direction}{abs(change_pct):.1f}%")
            