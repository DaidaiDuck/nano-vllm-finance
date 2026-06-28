# tests/test_generation.py
"""generation.generate 的集成测试。

默认跳过，需真实 HF 模型 + CUDA。开启方式:
    NANO_VLLM_INTEGRATION=1 python -m pytest tests/test_generation.py -v
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


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


def test_generate_returns_nonempty_str(model_and_tokenizer):
    from nano_vllm.generation import generate
    from nano_vllm.types import SamplingParams

    model, tokenizer = model_and_tokenizer
    out = generate(model, tokenizer, "Hello", SamplingParams(temperature=0.0, max_tokens=8))
    assert isinstance(out, str)
    assert len(out) > 0


def test_generate_greedy_deterministic(model_and_tokenizer):
    from nano_vllm.generation import generate
    from nano_vllm.types import SamplingParams

    model, tokenizer = model_and_tokenizer
    params = SamplingParams(temperature=0.0, max_tokens=8)
    a = generate(model, tokenizer, "What is 2 + 2?", params)
    b = generate(model, tokenizer, "What is 2 + 2?", params)
    assert a == b
