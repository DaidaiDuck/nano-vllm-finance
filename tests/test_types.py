# tests/test_types.py
"""SamplingParams / RequestOutput 数据类的单元测试。"""
# 直接从子模块导入，避免经由 nano_vllm/__init__.py 拖入 engine.py（需要 GPU/模型）
from nano_vllm.types import SamplingParams, RequestOutput


def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.temperature == 0.0
    assert p.top_k == -1
    assert p.top_p == 1.0
    assert p.max_tokens == 100


def test_sampling_params_override():
    p = SamplingParams(temperature=0.7, top_k=50, top_p=0.9, max_tokens=10)
    assert p.temperature == 0.7
    assert p.top_k == 50
    assert p.top_p == 0.9
    assert p.max_tokens == 10


def test_request_output_fields():
    out = RequestOutput(
        request_id="1",
        prompt="hi",
        text="hello",
        token_ids=[1, 2, 3],
        finished=True,
    )
    assert out.request_id == "1"
    assert out.prompt == "hi"
    assert out.text == "hello"
    assert out.token_ids == [1, 2, 3]
    assert out.finished is True
    # 字段名是 finish_reason（默认 None）。注意 engine.py 用的是 finished_reason=，二者不一致。
    assert out.finish_reason is None


def test_request_output_finish_reason():
    out = RequestOutput(
        request_id="1",
        prompt="hi",
        text="hello",
        token_ids=[1],
        finished=True,
        finish_reason="stop",
    )
    assert out.finish_reason == "stop"
