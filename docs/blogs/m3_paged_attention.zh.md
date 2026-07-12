# 从0构建nano-vllm (M3): 实现PagedAttention

[English](m3_paged_attention.md) | **中文**

> 从 M2 按照**模型最大序列长**预分配的连续 KV cache 换成**Block Pool管理**:所有的blocks由Block Pool管理，一个block能装下256个token的KV, 按照需求的上下文长度分配自己的blocks，M3实现的**按需分配思想**相比M2在短上下文请求中实现了巨大的内存节约，这个在下面的benchmark结果展示中可以体现。此外，每个请求所对应的KV Block在物理上不需要是连续内存，相比M2可以充分利用碎片内存空间，实现内存使用最大化的优势。 

> M3 设计文档:[m3_design.md](../design/m3_design.md) · 代码 tag:
> [m3](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m3)

---

M2 证明了自研连续 cache 的可行性和正确性、但在单请求下反而更慢——它的真正价值是**为 M3/M4 掌控内存布局打基础**。
M3 把M2中连续 buffer 换成**分块**,让每个请求只占 `ceil(len/block_size)` 块、
请求结束后自己的KV块释放后回池复用,并第一次用上 FlashAttention 的 paged 内核。我下面会把从零实现 PagedAttention
(Route A:monkey-patch HF `Qwen2Attention`)过程里的**所有核心知识点**逐个讲清。

# 核心知识点

## 0. Milestone 定位
- **M1**:单请求引擎,用 HF `DynamicCache`。
- **M2**:自写连续 KV cache(`MyKVCache`)。结论:单请求反而略慢(~6%),但为 M3/M4 打基础。
- **M3**:PagedAttention —— 分块 KV cache + FlashAttention paged kernel。

## 1. 心智模型:仓库 / 账本 / 索引(最重要)
| 名字 | 是什么 | 存哪 |
|---|---|---|
| **仓库** physical KV | 真正存 K/V 的大 tensor,切成 block | engine 的 `self.k_cache/v_cache` |
| **账本** BlockPool | 哪些 block id 是 free/used | `BlockPool` |
| **索引** block_table | 一个 request 的「逻辑位置 → 物理 block id」 | `KVCacheManager.req_to_blocks` |

Attention **读仓库、靠索引定位**;账本只发/收 block id,**不碰张量**。这三者一旦混在一起,就是 M3 最大的困惑源。

## 2. Route A:monkey-patch
保留 HF 模型骨架(embedding/MLP/norm/RoPE),只把每层 `Qwen2Attention.forward` 换成自写的
`paged_attn_forward`。

### 2.1 `types.MethodType` 原理
`attn.forward = types.MethodType(paged_attn_forward, attn)`
- `MethodType(func, obj)`:造**绑定方法**,把 `obj` 固定成 `func` 的第一个参数 `self`。
  于是 `attn.forward(x)` == `paged_attn_forward(attn, x)`。
- 赋给**实例属性** `forward`,遮住类方法。
- `nn.Module.__call__` 内部执行 `self.forward(...)`;Python 属性查找**先命中实例属性** → 走你的函数。
- duck typing,**不需要继承** `Qwen2Attention`。

### 2.2 调用链(一次 `self.model()`)
`self.model()` → `Qwen2ForCausalLM.forward` → `Qwen2Model.forward` → `for layer` →
`Qwen2DecoderLayer.forward` → `self.self_attn(...)` → `nn.Module.__call__` → `self.forward` →
**`paged_attn_forward`**(读 ENGINE_CTX → flash_attn)。

## 3. ENGINE_CTX(= M3 版 attn_metadata)
- 模块级**单例**,存本步的 `block_table` / `cache_seqlens`。
- **为什么需要**:HF 的 decoder layer 调 attention 时**不传 paging 参数**;Route A 改不了 HF 调用链、
  加不了函数参数。用一个**共享全局对象**隔空传参:engine 每步 forward 前设好,attention 里读。
- vLLM 把 `attn_metadata` 当真正的函数参数一路传下来(因为模型 forward 是它自己写的);Route A 做不到,
  所以退化成共享状态。单请求同步执行 → 无并发 → 全局共享安全(M4 batching 才需更讲究)。

## 4. flash_attn_with_kvcache
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

## 5. paged_attn_forward 关键点
### 5.1 布局约定(为什么要 transpose)
- HF/PyTorch SDPA:`[batch, n_heads, seq, head_dim]`(**heads 在前**)。
- flash_attn:`[batch, seq, n_heads, head_dim]`(**seq 在前**)。

两者都并行 heads,**只是轴顺序约定不同**。流程:proj 后 `.transpose(1,2)` → heads-first;
在 heads-first 上做 RoPE;之后**再 `.transpose(1,2)` 回 seq-first** 喂 flash_attn。
### 5.2 apply_rotary_pos_emb 是模块级函数
不是 `self` 的方法:`from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb`。
`position_embeddings` 是 `(cos, sin)` 元组。
### 5.3 concat heads + o_proj
- `attn_output` `[b,s,h,d]` 是**这一层**的注意力输出,**不是 logits**(logits 在所有层 + final
  norm + lm_head 之后才有)。
- `reshape [b,s,h,d]→[b,s,hidden]` = **concat 所有 head**。
- `o_proj` = `W_O`:混合各 head + 投回 hidden。**HF 的 o_proj 是 nn.Linear,返回单张量**
  (返回 tuple 的是 vLLM 的 `RowParallelLinear`)。
### 5.4 forward 返回 tuple
HF `forward` 返回 `(attn_output, attn_weights)`,decoder layer 会解包 → 你 `return attn_output, None`。
### 5.5 .contiguous()
`transpose`/`reshape` **只改 stride、不搬数据** → 可能非连续。某些 kernel/matmul 要求连续内存。
经验:**transpose 后、接对布局敏感的算子前**加 `.contiguous()`。

## 6. Engine wiring
### 6.1 物理 KV cache
`torch.zeros(num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype=..., device=...)`
- `torch.zeros(*size, dtype=, device=)`:`*size` 是**可变个数的形状整数**;`dtype/device` 必须
  **关键字**传(位置传会被当成额外维度)。
- 每层绑定:`attn.k_cache = self.k_cache[i]`。
### 6.2 从 config 读维度(别让调用方传)
```python
cfg = self.model.config
num_layers   = cfg.num_hidden_layers
num_kv_heads = cfg.num_key_value_heads
head_dim     = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
dtype        = self.model.dtype   # bf16
```
### 6.3 num_blocks
`bytes_per_block = num_layers * block_size * num_kv_heads * head_dim * 2(bf16) * 2(K,V)`;
`free,_ = torch.cuda.mem_get_info(); num_blocks = int(free*0.9)//bytes_per_block`。M3 写死够用即可。

> **踩坑:官方 flash-attn 要求 page block size 是 256 的倍数**(`must be divisible by 256`)。
> vLLM 能用 16 是因为它带自己 patch 过的 kernel;走官方 wheel 就得用 **`block_size=256`**
> (相应把 `num_blocks` 降到 ~625,保持池子 ~5.9GB)。
### 6.4 必须显式传 position_ids
M3 绕过 HF cache,HF 无法从 cache 长度推 position → RoPE 会错。
prefill:`torch.arange(prompt_len, device="cuda").unsqueeze(0)` `[1,seq]`;
decode:`torch.tensor([[num_computed_tokens]], device="cuda")` `[1,1]`。
### 6.5 num_computed_tokens 谁更新
**engine 每步 forward 后手动 `+=`**。不更新 → `allocate_slots` 永远按 prefill 数算 → decode 写越界。
### 6.6 use_cache 无关
你的 `k_cache/v_cache` 由 **flash_attn 内部写更新**,和 HF 的 `use_cache` 无关。`use_cache=False`
只是别让 HF 白建一个你不用的 DynamicCache。

## 7. device 规则
只有「**凭空造新张量**」的函数(`torch.tensor/arange/zeros/randn/empty`)默认造在 CPU,才要加
`device="cuda"`。对**已有张量做运算**(`a+b`、`x.reshape`、`q.transpose`)**继承输入 device**,不用加。

## 8. HF vs vLLM API 对照(别抄错模型)
你用的是 **HF 模型**;vLLM 的 `qwen2.py` 是**另一套模型**,API 不通用。
| vLLM 写法 | HF (Route A) |
|---|---|
| `qkv_proj` + `split` | `q_proj/k_proj/v_proj` 分开 |
| `rotary_emb(positions,q,k)` | `apply_rotary_pos_emb(q,k,cos,sin)` |
| `self.attn(q,k,v)`(内部=写cache+算attn) | `flash_attn_with_kvcache(...)` |
| `o_proj` 返回 `(out,bias)` | `o_proj` 返回单张量 |
| `qk_norm`(Qwen3) | Qwen2.5 无 |

**核心领悟**:vLLM 里 `self.attn(q,k,v)` 那**一行**就是 paged attention,你替换的就是它;周围 proj/RoPE/o_proj 一律用 HF 等价物。

## 9. BlockPool / KVCacheManager
- **BlockPool**:`FreeKVCacheBlockQueue` 双向链表 + 哨兵 head/tail;`get_new_blocks`(ref_cnt+1)/
  `free_blocks`(ref_cnt-1,归零回池)。
- **KVCacheManager**:`req_to_blocks` dict;`allocate_slots` 按 `ceil(total/block_size) - 已有块数`
  补差额;`get_block_table` 转 `[1,N]` int32 cuda。
- `ref_cnt` / prefix caching / LRU eviction:M3 用不到(单请求 ref_cnt 恒 1→0),但保留结构以贴近真 vLLM、方便 M4。

---

## Benchmark:两条独立的线

A100 80GB(Qwen2.5-3B bf16,单请求)上跑完 M1/M2/M3。**正确性先过**:M3 greedy 输出与 HF
逐 token 一致(`test_m3_vs_hf.py` 3/3)。然后 M3 讲了一个和 M2 相反的故事——**第一次真的变快了**。

### Latency:靠 flash-attn kernel,不是 paging

| 场景 | TPOT M1 | TPOT M2 | **TPOT M3** | M3 vs M1 | **M3 吞吐** |
|---|---|---|---|---|---|
| short_chat | 30.2 | 31.0 | **26.2** ms | −13% | **38.2** tok/s |
| medium_chat | 30.3 | 31.2 | **26.4** ms | −13% | **37.9** tok/s |
| long_context | 30.0 | 31.1 | **26.3** ms | −12% | **37.0** tok/s |

decode 快 ~13%(vs M1)、~15%(vs M2)。**关键:加速来自融合的 `flash_attn_with_kvcache`
kernel + 甩掉 M2 的 transpose 税,不是分页本身**——分页是内存管理,不改单请求的计算量。
M2 那句"pay the tax now, refund at M3"应验了:M2 比 M1 慢 ~6%,M3 把这税退了还叠加 kernel 收益。

一个诚实的"输":**长 prompt 的 prefill(TTFT)慢 ~8%**(long_context 85→92ms,flash-attn 的
paged prefill vs HF dense SDPA),但 decode 主导,端到端 M3 仍快 ~12%。

### Memory:靠 paging

每请求 KV 占用(M2 按 `max_seq_len` 预留 302MB;M3 只占 `ceil(len/256)` 块):

| 场景 | 平均长度 | M2(固定) | M3(分页) | 省 |
|---|---|---|---|---|
| short_chat | 225 | 302 MB | 9.4 MB | **−96.9%** |
| medium_chat | 726 | 302 MB | 28.3 MB | **−90.6%** |
| long_context | 2099 | 302 MB | 84.9 MB | **−71.9%** |

请求越短省得越多(M2 的固定预留纯浪费)。这份每请求的节省,正是 **M4** 能把多个并发序列塞进
一个池子的基础。

### 一句话
**latency 靠 flash-attn kernel,memory 靠 paging——两条独立的线。**

---

> 后续会补上"从踩坑到跑通"的完整叙事(monkey-patch 的坑、transpose 方向、返回 tuple、
> block_table 传 id 而非 count、256 约束 等)。英文版见 [m3_paged_attention.md](m3_paged_attention.md)。
