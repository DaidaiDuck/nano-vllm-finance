# 测试运行器
import time
import torch 
from nano_vllm import LLM, SamplingParams
from benchmarks.metrics import RequestMetrics, ScenarioMetrics, compute_metrics
from benchmarks.scenarios import Scenario
from benchmarks.datasets import generate_synthetic_prompts

class BenchmarkRunner:
    def __init__(self, engine: LLM, tokenizer): 
        self.engine = engine 
        self.tokenizer = tokenizer
    
    def warmup(self, scenario: Scenario, num_warmup: int = 3):
        """充分 warmup"""
        warmup_prompts = generate_synthetic_prompts(
            num_warmup,
            scenario.prompt_len_range,
            self.tokenizer
        )
        params = scenario.get_sampling_params()

        for prompt in warmup_prompts:
            _ = self.engine.generate(prompt, params) 

        torch.cuda.synchronize()

    def run_scenario(self, scenario: Scenario) -> ScenarioMetrics:
        """运行一个场景"""
        print(f"\nRunning scenario: {scenario.name}")
        print(f"Description: {scenario.description}")

        # 1. 准备数据
        prompts = generate_synthetic_prompts(
            scenario.num_requests,
            scenario.prompt_len_range,
            self.tokenizer,
        )
        params = scenario.get_sampling_params() 

        # 2. Warmup 
        print("Warming up...")
        self.warmup(scenario)

        # 3. 清理 GPU 状态
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 4. 测试主体
        print(f"Benchmarking {scenario.num_requests} requests...")
        request_metrics = []

        global_start = time.perf_counter()

        for i, prompt in enumerate(prompts): 
            metric = self.run_single_request(prompt, params, request_id=str(i))
            request_metrics.append(metric) 

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{scenario.num_requests}")
            
        total_duration = time.perf_counter() - global_start 

        # 5. 计算指标
        return compute_metrics(
            scenario.name,
            request_metrics,
            total_duration
        )

    def run_single_request(
            self, 
            prompt: str, 
            params: SamplingParams, 
            request_id: str = "0", 
    ) -> RequestMetrics: 
        """运行单个请求并测量"""
        # 真实 prefill 长度 (含 chat template 标记), 与引擎实际处理的输入一致
        prompt_len = self.engine.count_prompt_tokens(prompt)

        # 测 TTFT
        start = time.perf_counter()
        first_token_time = None 
        token_ids = []

        for token in self.engine.generate_stream(prompt, params):
            if first_token_time is None:
                first_token_time = time.perf_counter() 
            token_ids.append(token) 
        
        # Total Duration
        end = time.perf_counter()
        total_time = end - start
        output_len = len(token_ids) 

        # Calculate benchmark 
        if output_len > 0: 
            ttft = first_token_time - start 

            if output_len > 1:
                decode_time = end - first_token_time
                tpot = decode_time/ (output_len - 1)
            else:
                tpot = 0.0
        else:
            ttft = total_time
            tpot = 0.0
    
        return RequestMetrics(
            prompt_len=prompt_len,
            output_len=output_len,
            total_time=total_time,
            request_id=request_id, 
            tpot=tpot,
            ttft=ttft,
        )        
    

