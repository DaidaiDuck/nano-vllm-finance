# tests/test_kv_cache.py
"""MyKVCache 的单元测试 (纯张量逻辑, 无需模型)。

MyKVCache 不依赖 HF 模型, 只做预分配 + 定位写入, 所以这些用例默认在 CPU 上跑
(CI 处处可用); 检测到 CUDA 时额外参数化跑一遍 GPU。直接从子模块导入, 避免经由
nano_vllm/__init__.py 拖入 engine.py (需要 GPU/模型)。
"""
import pytest
import torch

from nano_vllm.simple.kv_cache import MyKVCache

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.bfloat16]

# 小配置, 单测跑得快; 真实模型配置不影响逻辑正确性
NUM_LAYERS = 4
MAX_SEQ_LEN = 16
NUM_KV_HEADS = 2
HEAD_DIM = 8


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def _make_cache(device, dtype=torch.float32, max_seq_len=MAX_SEQ_LEN):
    return MyKVCache(
        num_layers=NUM_LAYERS,
        max_seq_len=max_seq_len,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=dtype,
        device=device,
    )


def _rand(seq_len, device, dtype, seed):
    """造一份确定性的 [1, num_kv_heads, seq_len, head_dim] 张量。

    用 randn (值在 head/seq/dim 三个维度上都不同) 才能抓出 transpose/layout bug;
    先在 float32 生成再 cast, 保证 cast 后的值能被 dtype 精确表示, roundtrip 仍逐位相等。
    """
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, generator=g, device=device)
    return t.to(dtype)


# --------------------------------------------------------------------------- #
# 基础状态
# --------------------------------------------------------------------------- #
def test_initial_state(device):
    cache = _make_cache(device)
    assert cache.get_seq_length() == 0
    assert cache.get_max_length() == MAX_SEQ_LEN
    assert cache.current_len == 0


# --------------------------------------------------------------------------- #
# Prefill: 形状 + 数值 + dtype/device + 逐层隔离
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", DTYPES)
def test_prefill_roundtrip_per_layer(device, dtype):
    """每层写入不同的数据, 校验读回的值逐位相等 —— 能抓出"写错层 / 层间串数据"。"""
    cache = _make_cache(device, dtype)
    seq_len = 5
    # 每层一份不同的 K/V (seed 随 layer 变)
    layer_data = {
        i: (_rand(seq_len, device, dtype, seed=100 + i),
            _rand(seq_len, device, dtype, seed=200 + i))
        for i in range(NUM_LAYERS)
    }

    for i in range(NUM_LAYERS):
        k, v = layer_data[i]
        k_full, v_full = cache.update(k, v, layer_idx=i)

        assert k_full.shape == (1, NUM_KV_HEADS, seq_len, HEAD_DIM)
        assert v_full.shape == (1, NUM_KV_HEADS, seq_len, HEAD_DIM)
        # dtype / device / 连续性必须保持
        assert k_full.dtype == dtype and v_full.dtype == dtype
        assert k_full.device.type == torch.device(device).type
        assert k_full.is_contiguous()
        # 值逐位相等 (update 只做 squeeze/transpose/contiguous, 无算术)
        assert torch.equal(k_full, k)
        assert torch.equal(v_full, v)

    assert cache.current_len == seq_len


def test_current_len_advances_only_on_last_layer(device):
    """current_len 只能在最后一层推进; 中间层调用不得改变它。"""
    cache = _make_cache(device)
    seq_len = 3
    k = _rand(seq_len, device, torch.float32, seed=1)

    for i in range(NUM_LAYERS - 1):
        cache.update(k, k, layer_idx=i)
        assert cache.current_len == 0, f"current_len 在第 {i} 层就被推进了"

    cache.update(k, k, layer_idx=NUM_LAYERS - 1)
    assert cache.current_len == seq_len


# --------------------------------------------------------------------------- #
# Decode: 多步累积
# --------------------------------------------------------------------------- #
def test_multi_step_decode(device):
    """prefill 后逐个 token decode 多步, 校验长度递增且历史不被破坏。"""
    cache = _make_cache(device)
    dtype = torch.float32
    prefill_len = 4
    k_pre = _rand(prefill_len, device, dtype, seed=10)
    v_pre = _rand(prefill_len, device, dtype, seed=11)

    for i in range(NUM_LAYERS):
        cache.update(k_pre, v_pre, layer_idx=i)
    assert cache.current_len == prefill_len

    num_decode = 8  # 累积到 4 + 8 = 12 < max_seq_len(16)
    for step in range(num_decode):
        k_dec = _rand(1, device, dtype, seed=1000 + step)
        v_dec = _rand(1, device, dtype, seed=2000 + step)
        for i in range(NUM_LAYERS):
            k_full, v_full = cache.update(k_dec, v_dec, layer_idx=i)
            expected_len = prefill_len + step + 1
            assert k_full.shape[2] == expected_len
            if i == 0:
                # 历史 prefill 部分保持不变, 末尾是本步 decode
                assert torch.equal(k_full[:, :, :prefill_len, :], k_pre)
                assert torch.equal(k_full[:, :, -1:, :], k_dec)
        assert cache.current_len == prefill_len + step + 1


# --------------------------------------------------------------------------- #
# 溢出: 单次大写入 / 逐步累积 / 精确边界
# --------------------------------------------------------------------------- #
def test_overflow_single_write(device):
    cache = _make_cache(device)
    k = _rand(MAX_SEQ_LEN + 1, device, torch.float32, seed=5)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.update(k, k, layer_idx=0)


def test_exact_boundary_fill(device):
    """写入正好 max_seq_len 应成功; 再写 1 个才溢出 (off-by-one)。"""
    cache = _make_cache(device)
    k_full_len = _rand(MAX_SEQ_LEN, device, torch.float32, seed=6)
    for i in range(NUM_LAYERS):
        out, _ = cache.update(k_full_len, k_full_len, layer_idx=i)
        assert out.shape[2] == MAX_SEQ_LEN
    assert cache.current_len == MAX_SEQ_LEN

    k_one = _rand(1, device, torch.float32, seed=7)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.update(k_one, k_one, layer_idx=0)


def test_incremental_overflow(device):
    """逐步累积超过 max_seq_len 时应在越界那一步报错。"""
    cache = _make_cache(device, max_seq_len=4)
    k = _rand(2, device, torch.float32, seed=8)
    for i in range(NUM_LAYERS):  # 写 2 个 token; current_len 只在最后一层推进 -> 2
        cache.update(k, k, layer_idx=i)
    assert cache.current_len == 2
    # current_len=2, 再写 3 个 -> end=2+3=5 > 4, 越界
    k2 = _rand(3, device, torch.float32, seed=9)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.update(k2, k2, layer_idx=0)


# --------------------------------------------------------------------------- #
# reset
# --------------------------------------------------------------------------- #
def test_reset_zeroes_length(device):
    cache = _make_cache(device)
    k = _rand(5, device, torch.float32, seed=20)
    for i in range(NUM_LAYERS):
        cache.update(k, k, layer_idx=i)
    assert cache.current_len == 5

    cache.reset()
    assert cache.current_len == 0
    assert cache.get_seq_length() == 0


def test_reset_then_reuse_no_stale_data(device):
    """reset 不清零内存; 复用后读回的必须是新数据, 不能带出上一轮残留。"""
    cache = _make_cache(device)
    old = _rand(6, device, torch.float32, seed=30)
    for i in range(NUM_LAYERS):
        cache.update(old, old, layer_idx=i)

    cache.reset()

    new = _rand(2, device, torch.float32, seed=31)  # 比上一轮短
    for i in range(NUM_LAYERS):
        k_full, _ = cache.update(new, new, layer_idx=i)
        assert k_full.shape[2] == 2
        assert torch.equal(k_full, new)  # 是新数据, 不是 old 的前 2 个
    assert cache.current_len == 2


# --------------------------------------------------------------------------- #
# 与 HF DynamicCache 对齐
# --------------------------------------------------------------------------- #
def test_compatibility_with_hf_dynamic_cache(device):
    """逐层 + decode 后都应与 HF DynamicCache 的形状/数值一致。"""
    DynamicCache = pytest.importorskip("transformers").DynamicCache
    cache = _make_cache(device)
    hf = DynamicCache()
    dtype = torch.float32

    # Prefill 所有层
    prefill = {i: (_rand(5, device, dtype, seed=40 + i),
                   _rand(5, device, dtype, seed=50 + i))
               for i in range(NUM_LAYERS)}
    for i in range(NUM_LAYERS):
        k, v = prefill[i]
        my_k, my_v = cache.update(k, v, layer_idx=i)
        hf_k, hf_v = hf.update(k.clone(), v.clone(), layer_idx=i)
        assert my_k.shape == hf_k.shape
        assert torch.equal(my_k, hf_k)
        assert torch.equal(my_v, hf_v)

    # Decode 一个 token, 再对齐
    for i in range(NUM_LAYERS):
        k = _rand(1, device, dtype, seed=60 + i)
        v = _rand(1, device, dtype, seed=70 + i)
        my_k, my_v = cache.update(k, v, layer_idx=i)
        hf_k, hf_v = hf.update(k.clone(), v.clone(), layer_idx=i)
        assert my_k.shape == hf_k.shape
        assert torch.equal(my_k, hf_k)
        assert torch.equal(my_v, hf_v)
