# tests/test_sampler.py
"""Sampler 的纯 CPU 单元测试，无需模型/GPU。

注意: Sampler.sample 会就地修改传入的 logits（top-k/top-p masking），
所以每个用例都用 _logits() 现造一份新的 tensor。
"""
import torch

# 直接从子模块导入，避免经由 nano_vllm/__init__.py 拖入 engine.py（需要 GPU/模型）
from nano_vllm.core.sampler import Sampler
from nano_vllm.core.types import SamplingParams


def _logits(values):
    return torch.tensor(values, dtype=torch.float32)


def test_greedy_returns_argmax():
    sampler = Sampler()
    params = SamplingParams(temperature=0.0)
    # index 2 明显最大
    for _ in range(5):
        out = sampler.sample(_logits([0.1, 0.2, 5.0, 0.3, -1.0]), params)
        assert out == 2


def test_returns_python_int():
    sampler = Sampler()
    out = sampler.sample(_logits([1.0, 2.0, 3.0]), SamplingParams(temperature=0.0))
    assert type(out) is int  # 来自 .item()，不是 0-d tensor


def test_temperature_near_onehot():
    # 某个 logit 极大，即使 temperature=1.0 也应恒选该 token
    sampler = Sampler()
    params = SamplingParams(temperature=1.0)
    torch.manual_seed(0)
    for _ in range(200):
        out = sampler.sample(_logits([0.0, 0.0, 50.0, 0.0]), params)
        assert out == 2


def test_top_k_one_equals_greedy():
    # top_k=1 → 候选集只剩 argmax，温度采样也只能选它
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_k=1)
    torch.manual_seed(0)
    for _ in range(200):
        out = sampler.sample(_logits([1.0, 4.0, 2.0, 3.0]), params)
        assert out == 1


def test_top_k_restricts_candidates():
    # top_k=2 → 抽样结果必须落在 top-2 indices (这里是 3 和 1)
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_k=2)
    allowed = {1, 3}
    torch.manual_seed(0)
    for _ in range(200):
        out = sampler.sample(_logits([0.5, 3.0, 1.0, 4.0, 0.1]), params)
        assert out in allowed


def test_top_k_larger_than_vocab():
    # top_k 远大于词表大小 → min() 兜底，不应崩溃
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_k=1000)
    out = sampler.sample(_logits([1.0, 2.0, 3.0, 4.0, 5.0]), params)
    assert 0 <= out <= 4


def test_top_p_restricts_to_nucleus():
    # 构造一个 top token 概率 > 0.5 的分布，top_p=0.5 → 只剩它
    # logits=[10,0,0,0] 时 softmax 后 top token 概率 ~0.9999 > 0.5
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_p=0.5)
    torch.manual_seed(0)
    for _ in range(200):
        out = sampler.sample(_logits([10.0, 0.0, 0.0, 0.0]), params)
        assert out == 0


def test_top_p_one_no_truncation():
    # top_p=1.0 → 不做截断，应返回合法 index 且不崩溃
    sampler = Sampler()
    params = SamplingParams(temperature=1.0, top_p=1.0)
    torch.manual_seed(0)
    out = sampler.sample(_logits([1.0, 2.0, 3.0]), params)
    assert 0 <= out <= 2
