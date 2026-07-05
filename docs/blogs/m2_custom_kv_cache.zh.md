# 从0构建nano-vllm (M2): 自定义KV Cache —— 一次"预测更快、实测更慢"的优化

[English](m2_custom_kv_cache.md) | **中文**

> 系列:一次一个里程碑,重新实现vLLM的核心思想.
> **M2 - 自定义KV Cache** 用预分配的固定大小连续Tensor(MyKVCache)替换HuggingFace的DynamicCache.
> 初衷是干掉 M1 里使用Dynamic Cache `torch.cat` 造成的 O(n²) 拷贝 —— 但实测讲了一个完全相反的故事.
>
> M2设计文档:[m2_design.md](../design/m2_design.md) · 代码 tag:
> [m2](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m2)

---
M1 的结论是:引擎正确、能跑,但依赖 HuggingFace 的 `DynamicCache`,而它每步用 `torch.cat` 增长 KV,理论上是 O(n²) 拷贝。M2 的目标就是自己写一个**预分配、连续存储**的 `MyKVCache` 替换它,并证明"输出和 M1 逐 token 一致"。正确性做到了。但当我把 M1(HF cache)和 M2(MyKVCache)放在同一 benchmark 里对比,结果**出乎意料:M2 更慢、更费显存,而那个 O(n²) 瓶颈并不显著**。这篇就讲清楚:为什么,以及 M2 为什么依然值得做。

# 设计思考

## 1. 为什么要自己写KV Cache?

两个原因。**表面原因**:M1 用的 `DynamicCache` 每个 decode 步做 `torch.cat([历史, 新token])`,理论上每步拷贝全部历史 → 生成 n 个 token 总拷贝量 O(n²),还伴随反复分配/释放导致的显存碎片。**根本原因**:后续的 M3(PagedAttention)和 M4(Continuous Batching)**必须自己掌控 KV cache 的内存布局**——HF 的 DynamicCache 做不了分页和高效并发。所以必须要自己实现cache。

## 2. MyKVCache 的设计
```
LLM (用户入口)
    └─ SimpleEngine
        └─ Tokenizer (HF)
        └─ Model (HF, past_key_values = self.my_kv_cache, use_cache=True) <- KV cache = self.my_kv_cache (M2自研)
        └─ Sampler (M1自研)
```

核心是**一次性预分配、之后只做定位写入**:

```python
# 开局一次性分配, 之后再不 alloc/free
self.k_cache = torch.zeros(num_layers, max_seq_len, num_kv_heads, head_dim, ...)
self.v_cache = torch.zeros_like(self.k_cache)
self.current_len = 0    # 已写入的 token 数, 所有层共享
```

布局选了 **seq-major**:`[num_layers, max_seq_len, num_kv_heads, head_dim]`——序列维在头维之前。这么选是为了**和 M3 的 paged(token-major)布局对齐**,少改一次。代价是:HF 传进来的 K/V 是 heads-first 的 `[1, num_kv_heads, seq_len, head_dim]`,所以每次 `update` 都要**转置**:写入前 `squeeze→transpose→contiguous`,读取返回时再 `transpose→unsqueeze→contiguous`。先记住这个"转置税",后面它会成为主角。

`update` 每步只把新 token 写进 `[current_len : current_len+seq_len]` 这个切片,不碰历史——理论上把每步的 O(n) 拷贝降成 O(1) 写入。


## 3. 接入引擎

引擎里加了个 `use_custom_cache` 开关:`True` 用 MyKVCache(M2),`False` 用 HF DynamicCache(M1),这样**同一套 benchmark 能公平对比两种后端**。

接入时踩了个坑:HF 模型 forward 时会调 `past_key_values.get_mask_sizes(query_length, layer_idx)` 来构造 attention mask,而我最初写的MyKVCache 没这个方法，后面补充了一下。

```python
def get_mask_sizes(self, query_length, layer_idx):
    kv_length = self.get_seq_length(layer_idx) + query_length  # 已存 + 本次新token
    kv_offset = 0                                             # 完整非滑窗cache为0
    return kv_length, kv_offset
```

注意 `query_length` 在 trasnformer 5.13.0 里是 **int** ——签名随 transformers 版本会变,最稳的做法是 inspect 你环境里 `DynamicLayer.get_mask_sizes` 的源码照抄。这是唯一需要补的方法。

## 4. 正确性验证

在 `tests/test_m1_vs_hf.py` 里断言:greedy 模式下 MyKVCache 的输出必须和 HF 逐 token 一致。

```
# ① 正确性
NANO_VLLM_INTEGRATION=1 python -m pytest tests/test_m1_vs_hf.py tests/test_kv_cache.py -v
```

```python
assert nano.token_ids == hf_greedy_ids
```

在 **Qwen2.5-3B-Instruct** 上 **3/3 全过** —— MyKVCache 是 HF DynamicCache 的正确 drop-in 替换。

**正确性,达标。** 接下来是性能——故事从这里开始变得有意思。

# M2 的 Benchmark —— 意外的结果

同一台 A100 80GB SXM、Qwen2.5-3B bf16,用 `--cache hf` 和 `--cache custom` 各跑一遍。

```
# ② Parity(M1 重跑 vs M2,期望延迟≈相等)
python -m benchmarks.m2_benchmark --cache hf --version m1_rerun \
    --scenarios short_chat medium_chat long_context
python -m benchmarks.m2_benchmark --cache custom --version m2 \
    --scenarios short_chat medium_chat long_context \
    --baseline benchmarks/results/nano_vllm_m1_rerun_<时间戳>.json

# ③ cache 结构层面的 O(n²)→O(n)(模型无关,几秒)
python -m benchmarks.cache_microbench --max-steps 4096 --checkpoints 256
```

**延迟 / 吞吐(M1 → M2):**

| 场景 | TPOT | 吞吐(output tok/s) | 变化 |
|---|---|---|---|
| short_chat | 34.6 → 36.8 ms | 28.9 → 27.2 | **↓5.8%** |
| medium_chat | 34.7 → 36.9 ms | 28.8 → 27.1 | **↓6.0%** |
| long_context | 34.6 → 36.9 ms | 28.4 → 26.7 | **↓6.1%** |

**峰值显存:**

| 场景 | M1 | M2 | 差 |
|---|---|---|---|
| short | 6242 MB | 6538 MB | +296 |
| medium | 6397 MB | 6677 MB | +280 |
| long | 6953 MB | 7174 MB | +221 |

**微基准(去掉模型,只测 cache 操作,每步 36 层;已加 warmup 消除冷启动):**

| seq_len | DynamicCache | MyKVCache |
|---|---|---|
| 289 | 803 µs | 3236 µs |
| 1057 | 841 µs | 3246 µs |
| 2081 | 873 µs | 3189 µs |
| 4128 | 880 µs | 3406 µs |
| **总计(4096步)** | **4.6 s** | **13.3 s** |

**结论一句话:M2 稳定慢 ~6%、多用 ~300MB 显存,cache 操作本身慢 ~3.8 倍。和"更快"的直觉完全相反。**

## 分析:那么预言中的 O(n²) 去哪了?

先建立一个心智模型 —— **每步耗时 = 固定开销 + 拷贝开销**:

- **固定开销**:每步要发起的 GPU kernel(launch + 分配),数量和序列长度**无关**。
- **拷贝**:搬运 KV 字节的时间,**每步随序列长度增长**(O(n))。

两个发现,都能用这个拆解算清楚。

---

### 发现 1:每步`torch.cat` 的 O(n) 拷贝,算出来到底多小

先算**拷贝**这部分。序列长度 n 时,每步 `torch.cat` 要把历史 KV 复制一遍:

```
每层拷贝字节 = n × 2头 × 128 × 2字节(bf16) × 2(K和V) ≈ 1024·n 字节
36 层合计     ≈ 36 × 1024·n ≈ 37 KB × n
```

代入两端(A100 显存带宽 ≈ 2 TB/s):

| 序列长度 n | 拷贝字节 | 拷贝耗时 |
|---|---|---|
| 33 | ~1.2 MB | **~1 µs** |
| 4128 | ~150 MB | **~75 µs** |

再看**固定开销**:36 层 × 每层 `cat(K)`+`cat(V)` = 72 次 kernel launch,每次 ~11 µs(**这个 11µs 是从实测的 800µs 反推的**:`800 ÷ 72 ≈ 11`;它恰好落在已知的 kernel-launch 开销区间 ~5–20µs 内,所以自洽)→ **~800 µs**,和 n 无关。

合起来:

```
每步 = 800 µs(固定,不变)  +  1 µs → 75 µs(拷贝,随 n 涨)
```

**所以拷贝确实每步都在涨,但它埋在 ~800 µs 的固定开销下,几乎看不见。** 实测正好印证:803 µs(n=289)→ 880 µs(n=4128),涨了 ~77 µs。

> **结论**:在短Prompt下，DynamicCache 是 **overhead-bound(固定开销受限)**,不是 copy-bound。O(n²) 是真的,但要到几万 token、拷贝超过 800 µs 时才会主导 —— 现实推理很难遇到。**`torch.cat 很贵` 是被高估的都市传说。**

---

### 发现 2:MyKVCache 慢 3.8 倍 —— 数一数 kernel 就懂了

MyKVCache 慢**不是因为拷更多字节,而是因为每次 `update` 发起了更多 GPU kernel**。数一下每层每步:

| | 做的操作 | kernel 数 |
|---|---|---|
| **DynamicCache** | `cat(K)` + `cat(V)` | **2** |
| **MyKVCache** | 写:transpose+`contiguous`;切片赋值;读:transpose+`contiguous` —— K 和 V 各 3 个 | **6** |

MyKVCache 的 seq-major 布局(第 2 节的"转置税")逼它每次多做几个带拷贝的 `contiguous`。算固定开销:

```
DynamicCache:  36 层 × 2 kernel × ~11 µs ≈  800 µs   (实测 ~860)
MyKVCache:     36 层 × 6 kernel × ~15 µs ≈ 3240 µs   (实测 ~3260)
```

**3 倍的 kernel 数量 + 每个 transpose kernel 还略贵(跨步访问)→ ~3.8 倍时间。**(这里 MyKVCache 的 ~15µs 同样是反推的:`3260 ÷ (36×6) ≈ 15`,也在 5–20µs 区间内。)

---

### 对回端到端的 6%

cache 操作只是 TPOT 的一小部分,但 M2 让这部分变贵了:

```
MyKVCache 的 cache 操作     ≈ 3260 µs/步
DynamicCache 的 cache 操作  ≈  860 µs/步
多出来                      ≈ 2400 µs/步

2400 µs ÷ M1 的 TPOT(34600 µs)≈ 6.9%
```

**≈ 端到端实测慢的那 ~6%。完全对得上** —— 慢就慢在这 2400 µs 的额外 kernel 开销上。(这里的 `34600 µs` 就是 M1 实测的 TPOT,即一个 decode 步的耗时;这个除法的意义是把"孤立微基准测到的 cache 开销"对回"端到端的变慢",证明慢就慢在 cache 上。)

> **声明**:分析里面的实测数据有—DynamicCache ~860 µs、MyKVCache ~3260 µs(3.8 倍)、端到端慢 ~6%。而"**N 个 kernel × 每个 ~11/~15 µs**"这部分数据是**从实测结果反推的假设**:两个 per-kernel 数字都是"总时间 ÷ kernel 数"除出来的,但是这些数据也落在了 kernel-launch 合理的开销区间(~5–20µs)内,所以自洽——但**没有直接 profile 证实**。想把它从"猜"变成"测",用 `torch.profiler` 跑一遍数真实的 kernel 数和各自耗时即可。

# M2 教会我的

1. **"自己写就更快"是错觉,必须实测。** 我带着"干掉 O(n²)"的预期开工,结果做出来一个更慢的 cache。如果我只写理论、不测,就不会知道实际场景是怎么样的。**测量比直觉重要。**

2. **理论复杂度 ≠ 实际瓶颈。** O(n²) 在渐进意义上是对的,但在真实小规模请求下,固定开销(kernel launch)才是主导。

3. **M2 的价值不在单请求提速,而在"所有权"。** 现在这点转置开销是"控制权的成本"——一旦拥有了自己的 cache,M3 就能在它上面做 PagedAttention,M4 能做 batching。而那个转置税,**到 M3 用 paged attention kernel 直接消费 seq-major 布局时就消失了**(不再需要转回 HF 的 heads-first)。所以 M2 是"先交税、M3 退税"。

# M2 的局限性 / 下一步

- **单请求下比 M1 慢 ~6%、多 ~300MB 显存** —— seq-major 布局的转置开销,以及预分配的固定占用。
- **O(n²) / 碎片的收益没兑现** —— 因为单请求 + 中等长度下两种 cache 都是开销受限。这些收益要到 **M4(并发变长请求)** 才会真正显现。
- **下一步 M3(PagedAttention)**:把连续 buffer 换成块(block)存储 + block_table,并写一个直接吃 seq-major 布局的 paged kernel —— 那时转置税消失,同时解锁按需分配、跨请求共享。M2 铺的路,M3 才开始收租。

# 附录 / 备注

- 设计文档: [m2_design.md](../design/m2_design.md)
- Benchmark 配置: [benchmark_environment.md](../design/benchmark_environment.md)
- 上一篇: [M1 —— 一个简单正确的单请求引擎](m1_building_nano_vllm.zh.md)
