# 从0构建nano-vllm (M3): 实现PagedAttention

[English](m3_paged_attention.md) | **中文**

> 从 M2 按照**模型最大序列长**预分配的连续 KV Cache 换成M3的**Block Pool管理**: 所有blocks由Block Pool管理，一个block能装下自定义数量token的KV, 按照需求的上下文长度分配自己的blocks，M3实现的**按需分配思想**相比M2在短上下文请求中实现了巨大的内存节约，这个在下面的benchmark结果展示中可以体现。此外，每个请求所对应的KV Blocks在物理上不需要是连续内存，相比M2可以充分利用碎片内存空间，实现内存使用最大化。 

> M3 设计文档:[m3_design.md](../design/m3_design.md) · 代码 tag:
> [m3](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m3)

---

M2 证明了自研连续 cache 的可行性和正确性、但因为是按照模型最大序列长分配内存所以短请求下浪费很多内存空间——它的真正价值是**为 M3/M4 掌控内存布局打基础**。
M3 把M2中连续 buffer 换成**按KV块管理**,让每个请求只占 `ceil(seq_len/block_size)` 块、
请求结束后KV块释放回Block Pool池复用。M3第一次用上 FlashAttention 的 paged 内核。我下面会把从零实现 PagedAttention 里的**所有核心知识点**逐个讲清。


# 核心知识点

## 0. Milestone 定位
- **M1**:实现最简单的单请求引擎,用 HF `DynamicCache`。
- **M2**:自写连续 KV cache(`MyKVCache`)，实现按照模型最长上下文预分配内存。
- **M3**:PagedAttention —— 将KV cache分块管理，使用Block Pool管理哪些块是自由的，或是正在使用的，通过KV Cache Manager来管理请求和分配给该请求的KV块之间的映射。 

## 2. M3架构
```
LLM (用户入口)
    └─ PagedEngine
        └─ Tokenizer (HF)
        └─ Model (HF) -> GPU Forward: paged_attn_forward (custom) 
        └─ BlockPool (custom)
        └─ KVCacheManager (custom) 
        └─ Sampler (custom)
```


## 3. 核心代码文件 
| 名字 | 是什么 | 存哪 |
|---|---|---|
| Physical KV | 存 K/V 的大张量,切成 block | engine 的 `self.k_cache/v_cache` |
| BlockPool | 管理哪些 block 是 free/used | `BlockPool` |
| Block Table | 管理请求和分配给该请求的KV块的映射关系 | `KVCacheManager.req_to_blocks` |

## 4. 实现自研paged_attn_forward
因为M3使用了 HF 模型骨架(embedding/MLP/norm/RoPE),我需要把每层 `Qwen2Attention.forward` 换成我自研的
`paged_attn_forward`才能使用我自研的KV Cache. 

### 4.1 为什么 M1/M2 能直接 self.model(past_key_values=my_kv_cache),M3 不能? 
|   | Cache的Shape | HF的Attention是否能读懂?|
|---|---|---| 
| M1 HF's DynamicCache | 连续的[1, num_kv_heads, seq_len, head_dim] | 能
| M2 MyKVCache | 自研[1, seq_len， num_kv_heads, head_dim]结构，但是传给Attention之前转化回了HF能接受的[1, num_kv_heads, seq_len, head_dim]结构. | 能       
| M3 Paged Cache | 分块 + BlockTable间接寻址，物理内存空间上是分散的 | 不能

因为KV块是分散的，无法聚合拼成一个大张量的结构，所以不能用self.model(past_key_values=my_kv_cache)，需要自研。


## 4.2 为什么使用flash_attn_with_kvcache?
flash_attn_with_kvcache支持分页布局，把k_cache/v_cache, block_table, cache_seqlens交给它，它自己支持:

1. 把新的K/V写进block table指向的KV块
2. 计算Attention时支持按照block_table从分散的块中获取K/V信息来算注意力 


### 4.3 `types.MethodType` 原理
`attn.forward = types.MethodType(paged_attn_forward, attn)`
- `MethodType(func, obj)`:造**绑定方法**,把 `obj` 固定成 `func` 的第一个参数 `self`。
  于是 `attn.forward(x)` == `paged_attn_forward(attn, x)`。
- 赋给**实例属性** `forward`,遮住类方法。
- `nn.Module.__call__` 内部执行 `self.forward(...)`;Python 属性查找**先命中实例属性** → 走我的`paged_attn_forward`。
- duck typing,**不需要继承** `Qwen2Attention`。

### 4.4 调用链(一次 `self.model()`)
`self.model()` → `Qwen2ForCausalLM.forward` → `Qwen2Model.forward` → `for layer` →
`Qwen2DecoderLayer.forward` → `self.self_attn(...)` → `nn.Module.__call__` → `self.forward` →
**`paged_attn_forward`**(读 ENGINE_CTX → flash_attn_with_kvcache)。

## 4.5 ENGINE_CTX(= M3 版 vLLM的attn_metadata)
- 模块级**单例**,存本步的 `block_table` / `cache_seqlens`。
- **为什么需要**:
Paged Kernel必须放在Attention真正计算的地方，也就是Qwen2Attention.forward. 要让HF的模型使用我们的Paged kernel,要么重写整个模型(vLLM 的做法,工程量大)，要么只替换 forward 这一个方法(monkey-patch)——我选的方式。因为工程量相对较少。
- 如何实现:HF本身没给我任何API注入口，我只能money-patch到forward内部去修改方法。给GPU Forward的每一层都使用上我自研的paged_attn_forward，里面支持KV Cache按分块的布局，并通过ENGINE_CTX将请求最新的block table和cache sequence length传进去。 

## 5. flash_attn_with_kvcache
**一次调用 = 写 paged cache(append 新 K/V)+ 算 attention**。
### 形状
| 参数 | 形状 |
|---|---|
| `q` | `[batch, seq, n_heads, head_dim]` |
| `k_cache`/`v_cache` | `[num_blocks, block_size, n_kv, head_dim]` |
| `k`/`v`(新 token) | `[batch, seq, n_kv, head_dim]` |
| `block_table` | `[batch, max_blocks]` **int32 cuda** |
| `cache_seqlens` | `[batch]` **int32 cuda** |
| 返回 `out` | `[batch, seq, n_heads, head_dim]` |
### `cache_seqlens` 语义
= 本次 forward **之前** cache 里已有的 token 数。**prefill=0;decode=num_computed_tokens**。
flash_attn 把新 K/V 写到 `[cache_seqlens : cache_seqlens+seq]`。
### 为什么必须 int32
kernel 的 ABI 就按 int32 读索引张量;torch 默认造 int64 → 报错/读错位。

## 6. BlockPool / KVCacheManager
- **BlockPool**:`FreeKVCacheBlockQueue` 实现双向链表 + Dummy head/tail; 分配KV块`get_new_blocks`(ref_cnt+1);  释放KV块 `free_blocks`(ref_cnt-1,归零回池)。
- **KVCacheManager**:`req_to_blocks: dict[str, list[KVCacheBlock]]`，通过request_id映射请求对应的KV块; 
为请求分配KV块 `allocate_slots` 按 `ceil(请求所有的tokens/block_size) - 请求已有块数` 补差额;

---

## M3 Benchmark:两条独立的线

A100 80GB(Qwen2.5-3B bf16,单请求)上跑完 M1/M2/M3。

### **回归测试**:测试M3 greedy 输出与 HF greedy输出是否一致
结果: 逐 token 一致(`test_m3_vs_hf.py` 3/3)，测试通过✅。

命令: `NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct python -m pytest tests/test_m3_vs_hf.py -v`
```
============================================================= test session starts =============================================================
platform linux -- Python 3.11.10, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python
cachedir: .pytest_cache
rootdir: /workspace/nano-vllm-finance
configfile: pyproject.toml
plugins: asyncio-1.4.0, anyio-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 3 items                                                                                                                             

tests/test_m3_vs_hf.py::test_greedy_matches_hf[Hello] PASSED                                                                            [ 33%]
tests/test_m3_vs_hf.py::test_greedy_matches_hf[What is 2 + 2?] PASSED                                                                   [ 66%]
tests/test_m3_vs_hf.py::test_greedy_matches_hf[Explain photosynthesis in one sentence.] PASSED                                          [100%]

============================================================= 3 passed in 21.75s ==============================================================
```

### Latency:延迟降低靠 flash-attn kernel,不是 paging

| 场景 | TPOT M1 | TPOT M2 | **TPOT M3** | M3 vs M1 | **M3 吞吐** | **M3 TTFT** |
|---|---|---|---|---|---|---|
| short_chat | 30.2 | 31.0 | **26.2** ms | −13% | **38.2** tok/s | 26.9 ms |
| medium_chat | 30.3 | 31.2 | **26.4** ms | −13% | **37.9** tok/s | 30.6 ms |
| long_context | 30.0 | 31.1 | **26.3** ms | −12% | **37.0** tok/s | 91.7 ms |

> TTFT 取自 [nano_vllm_m3_20260711_014557.json](../../benchmarks/results/m3/nano_vllm_m3_20260711_014557.json)
> 的 `avg_ttft`(短→长:0.0269 / 0.0306 / 0.0917 s)。长 prompt 的 TTFT 明显更高,是因为 prefill 要
> 一次算完整个长 prompt.

M3 decode 速度快 ~13%(vs M1)、~15%(vs M2)。**关键:延迟降低来自融合的 `flash_attn_with_kvcache`
kernel + 停止 M2 的 transpose开销,不是分页本身**——分页是内存管理,不改单请求的计算速度。
M2 那句"pay the tax now, refund at M3"应验了:M2 比 M1 慢 ~6%,M3 把这开销退了还叠加 kernel 收益。


### Memory内存占用:靠 PagedAttention实现巨大的内存节约, M3目标达成✅

每请求 KV 占用(M2 按 `max_seq_len` 预留 302MB;M3 只占 `ceil(实际seq_len/256)` 块)。

**302MB 怎么来的**:先算每个 token 的 KV 字节(所有层、K 和 V 各一份):
```
每 token KV = num_layers × num_kv_heads × head_dim × 2(bf16 字节) × 2(K,V)
            = 36 × 2 × 128 × 2 × 2 = 36864 字节 ≈ 36 KB / token
```
M2 按模型最大序列长 `max_seq_len = 8192` **无条件预留**(和实际用多少无关):
```
M2 每请求 = 36 KB × 8192 ≈ 301,989,888 字节 ≈ 302 MB
```
M3 则按实际长度向上取整到块:`M3 = 36KB × ceil(seq_len/256) × 256`。短请求实际只用几十~几百 token,
差距就是下表的节约。


| 场景 | 平均长度 | M2(固定) | M3(分页) | 省 |
|---|---|---|---|---|
| short_chat | 225 | 302 MB | 9.4 MB | **−96.9%** |
| medium_chat | 726 | 302 MB | 28.3 MB | **−90.6%** |
| long_context | 2099 | 302 MB | 84.9 MB | **−71.9%** |

请求越短Memory省得越多，请求越长省得越多，符合我们的预期。

# M3 的局限性 / 下一步

- M3相比M2本质上还是个单请求引擎，不支持多请求并发处理。M4会实现Continuous Batching,通过Scheduler调度来在每一步同时处理多个请求的推理，预计会降低延迟并且提高Throughput. 

---
# 附录 / 备注

- 设计文档: [m3_design.md](../design/m3_design.md)
- Benchmark 配置: [benchmark_environment.md](../design/benchmark_environment.md)
- 上一篇: [从0构建nano-vllm (M2):自定义KV Cache —— 一次"预测更快、实测更慢"的优化](m2_custom_kv_cache.zh.md) 
