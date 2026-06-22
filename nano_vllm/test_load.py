# nano_vllm/test_load.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 


def test_load():
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # 节省显存 TODO: 还有哪些数据类型，区别是啥？ 
        device_map="cuda"
        )
    model.eval()

    print(f"Model loaded. Type: {type(model).__name__}")
    print(f"Num params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num attention heads: {model.config.num_attention_heads}")

    # 测试 tokenizer
    text = "你好"
    tokens = tokenizer.encode(text) 
    print(f"Encode '{text}' → {tokens}")
    print(f"Decode {tokens} → {tokenizer.decode(tokens)}")


if __name__ == "__main__":
    test_load()
