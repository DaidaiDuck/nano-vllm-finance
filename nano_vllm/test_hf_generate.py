# nano_vllm/test_hf_generate.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_hf_generate():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    
    prompt = "Hello, who are you?"
    # 用 chat template
    messages = [{"role":"user", "content":prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda") 

    # 用 HF generate (基准)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50, 
            do_sample=False, 
            temperature=1.0 #greedy
        )

    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_hf_generate()