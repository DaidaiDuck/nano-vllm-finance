# tests/test_engine_m1.py
"""LLM 入口的 M1 集成测试 (基础生成行为)。

默认跳过，需真实 HF 模型 + CUDA。开启方式:
    NANO_VLLM_INTEGRATION=1 python -m pytest tests/test_engine_m1.py -v
可用 NANO_VLLM_TEST_MODEL 覆盖默认模型 (默认 0.5B, 与其它集成测试一致)。
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.getenv("NANO_VLLM_INTEGRATION") or not torch.cuda.is_available(),
    reason="set NANO_VLLM_INTEGRATION=1 and run on a CUDA GPU",
)

MODEL_NAME = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def llm():
    from nano_vllm import LLM
    return LLM(MODEL_NAME)


def test_basic_generation(llm):
    """基本生成"""
    from nano_vllm import SamplingParams
    outputs = llm.generate("Hello", SamplingParams(temperature=0.0, max_tokens=20))
    assert len(outputs) == 1
    assert outputs[0].text != ""
    assert outputs[0].finished


def test_batch_prompts(llm):
    """多个 prompt (M1 是串行的)"""
    from nano_vllm import SamplingParams
    prompts = ["Hello", "World", "Bye"]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=10))
    assert len(outputs) == 3


def test_max_tokens(llm):
    """max_tokens 限制"""
    from nano_vllm import SamplingParams
    outputs = llm.generate("Tell me a long story", SamplingParams(max_tokens=20))
    assert len(outputs[0].token_ids) <= 20


def test_greedy_deterministic(llm):
    """Greedy 模式确定性"""
    from nano_vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=30)
    out1 = llm.generate("Hello", params)
    out2 = llm.generate("Hello", params)
    assert out1[0].text == out2[0].text
