# examples/simple_generate.py
from nano_vllm import LLM, SamplingParams 

llm = LLM("Qwen/Qwen2.5-3B-Instruct")

prompts = [
    "Hello, who are you?",
    "What is 2 + 2?",
    "Tell me a joke about programming.",
]

params = SamplingParams(temperature=0.0, max_tokens=80)
outputs = llm.generate(prompts, params) 

for output in outputs:
    print(f"\n=== Request {output.request_id} ===")
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.text}")
    print(f"Tokens generated: {len(output.token_ids)}")
    print(f"Finish reason: {output.finish_reason}")



