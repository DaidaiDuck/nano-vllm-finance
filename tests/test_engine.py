# tests/test_engine.py
"""engine.LLM / SimpleEngine 的集成测试 (回归测试)。

默认跳过，需真实 HF 模型 + CUDA。开启方式:
    NANO_VLLM_INTEGRATION=1 python -m pytest tests/test_engine.py -v
可用 NANO_VLLM_TEST_MODEL 覆盖默认模型。

注意: engine.py 在修好已知 bug（encode 未转 tensor、outputs 缺 .logits、
`false` 语法、decode loop list 未转 tensor、finish_reason 命名）之前，本测试会失败。
它正是用来验证这些修复是否到位的回归测试。
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.getenv("NANO_VLLM_INTEGRATION") or not torch.cuda.is_available(),
    reason="set NANO_VLLM_INTEGRATION=1 and run on a CUDA GPU",
)

MODEL_NAME = os.getenv("NANO_VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def test_llm_generate_returns_request_outputs():
    from nano_vllm import LLM, SamplingParams, RequestOutput

    llm = LLM(MODEL_NAME)
    outputs = llm.generate("Hello", SamplingParams(temperature=0.0, max_tokens=8))

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    out = outputs[0]
    assert isinstance(out, RequestOutput)
    assert isinstance(out.finished, bool)
    assert isinstance(out.text, str)
    assert len(out.text) > 0


def test_generate_stream_matches_generate():
    """流式输出的 token 序列应与非流式 generate 完全一致。

    用 greedy (temperature=0.0) 保证确定性, 两条路径走的是同一套
    prefill/decode 逻辑, 产出的 token_ids 必须逐个相等。
    """
    from nano_vllm import LLM, SamplingParams

    llm = LLM(MODEL_NAME)
    params = SamplingParams(temperature=0.0, max_tokens=16)
    prompt = "Hello"

    full = llm.generate(prompt, params)[0]
    streamed = list(llm.generate_stream(prompt, params))

    assert streamed == full.token_ids
    # 解码结果也应一致
    assert llm.engine.tokenizer.decode(streamed) == full.text
