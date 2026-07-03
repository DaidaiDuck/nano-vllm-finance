# tests/test_m1_vs_hf.py
"""M1 正确性回归测试: nano-vllm greedy 输出必须与 HuggingFace generate() 逐 token 一致。

这是 M1 最重要的测试 —— 证明自实现的 prefill/decode loop 与 HF 参考实现等价。
任何后续优化 (M2 自管 KV cache、M3 PagedAttention ...) 都必须保持这个测试为绿。

默认跳过, 需真实 HF 模型 + CUDA:
    NANO_VLLM_INTEGRATION=1 python -m pytest tests/test_m1_vs_hf.py -v
可用 NANO_VLLM_TEST_MODEL 覆盖默认模型。
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.getenv("NANO_VLLM_INTEGRATION") or not torch.cuda.is_available(),
    reason="set NANO_VLLM_INTEGRATION=1 and run on a CUDA GPU",
)

MODEL_NAME = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS = 32
PROMPTS = [
    "Hello",
    "What is 2 + 2?",
    "Explain photosynthesis in one sentence.",
]


@pytest.fixture(scope="module")
def llm():
    from nano_vllm import LLM
    return LLM(MODEL_NAME)


def _hf_greedy_ids(model, tokenizer, prompt):
    """用 HF generate() 复现 nano 的 greedy 解码, 返回生成部分的 token id 列表。

    必须 1:1 复现 nano 的输入构造与解码设置, 否则对照不公平:
    - 同样套 chat template + add_generation_prompt
    - do_sample=False 强制 greedy, 覆盖 Qwen generation_config 里的采样默认
      (Qwen 默认 do_sample=True / temperature=0.7, 不覆盖会导致结果随机)
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    # HF 的输出包含 prompt, 切掉前缀只留新生成的 token
    return out[0][input_ids.shape[1]:].tolist()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf(llm, prompt):
    """nano-vllm 的 greedy token 序列必须与 HF generate() 完全一致。"""
    from nano_vllm import SamplingParams

    model = llm.engine.model
    tokenizer = llm.engine.tokenizer

    nano = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))[0]
    hf_ids = _hf_greedy_ids(model, tokenizer, prompt)

    # 逐 token 一致是最强断言: 证明 prefill/decode/采样路径与 HF 等价
    assert nano.token_ids == hf_ids, (
        f"token mismatch for {prompt!r}:\n  nano={nano.token_ids}\n  hf  ={hf_ids}"
    )
    # 解码文本也应一致 (两边都用默认 decode 口径)
    assert nano.text == tokenizer.decode(hf_ids)
