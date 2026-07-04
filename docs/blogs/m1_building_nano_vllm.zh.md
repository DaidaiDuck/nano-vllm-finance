# 从0构建nano-vllm (M1): 一个简单正确的单请求引擎

[English](m1_building_nano_vllm.md) | **中文** 

> 系列:一次一个里程碑,重新实现vLLM的核心思想. 
> **M1 - 基线** 基于HuggingFace把单请求LLM推理(Prefill + Decode)流程正确跑通. 
> 参考vLLM风格写一个干净的API，之后每个版本的优化都有参照物去超越。 
>
> M1设计文档:[m1_design.md](../design/m1_design.md) · 代码 tag:
> [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1)

---
nano-vllm-finance从零构建vLLM核心思想，以揭示现代LLM Serving的运作方式。第一篇(M1)是基线：一个简单的单请求引擎，使用HuggingFace的模型和KV Cache跑通单请求Prefill+Decode,配上自研的Sampler和干净的`LLM` API. M1里没有自定义KV Cache, 没有Paged Attention, 没有Continuous Batching. 
M1唯一的要求是正确性: Greedy模式下M1代码的输出内容必须和HuggingFace的输出内容逐token一致，从而为后续每一步优化提供可信的参考。这一条件在后续每个里程碑都会约束。 

# 设计思考
## 1. 为什么我要重建vLLM?
最主要的目的是为了帮助我理解现代LLM推理的运作方式以及结构，以及回答如下问题：
1. 现代LLM推理的结构是什么，为什么是这样的结构?
2. 为什么存在Prefill和Decode两个阶段? 
3. 什么是KV Cache? KV Cache提供了什么?
4. 如何理解Paged Attention?
5. 什么是Continuous Batching? 

## 2. M1为什么只实现正确的单请求推理，不实现vLLM的核心组件呢？
M1最大的特点是简单，那些让vLLM更快的组件 --- 自定义KV Cache, PagedAttention, Continuous Batching, Scheduler被刻意排除。因为我把这些组件分配到了不同的里程碑中去，如果在M1就引入，会让 M1 的目标变复杂:既要保证 generation loop 正确,又要保证新组件本身正确,等于同时 debug。其次，需要对比用与不用该组件的差异;没有这个对比,项目的意义就丢失了。

## 3.M1的架构是什么? 

```
LLM (用户入口)
    └─ SimpleEngine
        └─ Tokenizer (HF)
        └─ Model (HF, use_cache=True) <- KV cache = HF DynamicCache 
        └─ Sampler (M1自研)
```

`LLM`是面向用户的入口, 里面包裹一个`SimpleEngine`.后者实现Prefill + Decode循环, 协调Tokenizer，模型和Sampler。KV Cache使用HuggingFace的DynamicCache. 

## 4. Prefill + Decode循环是怎么实现的? 
Prefill是对整个Prompt的一次GPU前向(GPU Forward)，产出下一个token的logits，并用prompt的Key/Value填满KV Cache. Decode随后逐token进行: 
```python
# === Prefill ===
with torch.no_grad():
    outputs = self.model(
        input_ids = input_ids, 
        use_cache = True,
    )
past_key_values = outputs.past_key_values 
logits = outputs.logits[0, -1, :]
# Sample first token
next_token = self.sampler.sample(logits, params) 

torch.cuda.synchronize()
yield next_token # First token out. 

# === Decode loop ===
for _ in range(params.max_tokens - 1): 
    if next_token == self.tokenizer.eos_token_id:
        break

    inp = torch.tensor([[next_token]], device="cuda") 
    with torch.no_grad():
        outputs = self.model(
            input_ids = inp,
            past_key_values = past_key_values,
            use_cache = True,
        )
    past_key_values = outputs.past_key_values # Update KV Cache 
    logits = outputs.logits[0, -1, :] 
    next_token = self.sampler.sample(logits, params) 

    torch.cuda.synchronize()
    yield next_token
```
每个Decode步骤只喂入最新的token `next_token`而非整段序列。`use_cache=True`保留此前全部key/value. logits取自`outputs.logits[0, -1, :]` 0表示单请求，-1表示序列里最后一个位置，: 表示完整词表 --- 因为模型在序列每个位置都会输出一个logits分布，但我们Decode阶段要的是当前文本最后一个token的logits。

## 5. 为什么要自研Sampler?
主要是参考vLLM的Sampler实现和自己学习目的。我的Sampler是一条短流水线: temperature=0时走greedy -> 否则根据temperature进行scaling -> 然后进行top-k判断 -> top-p判断 -> softmax -> multinomial. 

## 6. 如何判断自己单请求推理的正确性? 
在tests/test_m1_vs_hf.py实现单元测试:
Greedy模式下，我的引擎的输出token ids必须和HuggingFace的输出结果完全一致. 
```python
# tests/test_m1_vs_hf.py — greedy 必须和 HF 完全一致
assert nano.token_ids == hf_greedy_ids
```
一处需注意:Qwen 的 generation_config 默认采样(do_sample=True,temperature=0.7),因此必须传入 do_sample=False 强制 HF 进入 greedy——否则参照本身是随机的,测试失去意义。完整 M1 测试套件在 A100 80GB SXM + Qwen2.5-3B-Instruct 上 23/23 全过(12 个 CPU 单测 + 11 个 GPU 集成测试),test_m1_vs_hf 在每个 prompt 上均匹配。这种逐 token 一致性确立了 generation loop 的正确性,并作为之后每个里程碑都必须保持为绿的 non-regression 契约。

# M1的Benchmark

对三个单请求场景进行了 benchmark(`Throughput` 为 output tokens/s):

| 场景 | Prompt | Output | Throughput | P99 延迟 | TTFT | TPOT |
|----------|--------|--------|------------|-------------|------|------|
| short_chat | ~125 | 100 | 29.4 tok/s | 3.62 s | 36 ms | 34.0 ms |
| medium_chat | ~526 | 200 | 29.2 tok/s | 7.38 s | 38 ms | 34.2 ms |
| long_context | ~1999 | 100 | 29.2 tok/s | 3.47 s | 89 ms | 33.7 ms |

## 分析: 
Output Throughput 恒定在29 tokens/s，它约等于decode速度(1/TPOT).

TPOT维持在~34ms, 从125到2000个prompt token几乎不变，这是因为Decode是Memory-bound, 每一步的开销几乎全部来自**把模型权重从显存搬进计算单元（显存带宽受限）**，而非对KV Cache做attention的计算开销。 

TTFT随prompt长度增长而增长(36 -> 89ms)，因为Prefill是compute-bound。但这里我们可以看到125的prompt和526的prompt它们的TTFT几乎一样，说明加载模型权重的开销大约35ms左右，短Prompt场景下仍然是Memory Bound. 但是随着prompt增长越来越大，则变为compute-bound, 当prompt为2000tokens时TTFT变成89ms. 
 
核心结论: 
```
latency ≈ TTFT + output_len * TPOT
```
该公式把两个物理阶段分离——compute-bound 的 prefill(计入 TTFT)与 memory-bound 的 decode，并解释了为何 long_context 与 short_chat 延迟几乎相同,尽管 prompt 相差 16 倍:两者都产出 100 个 token,而主导墙钟时间的是 output 长度。这些解释了medium_chat的用时为何是最长的，因为它的输出长度是200个tokens. 

最后说明: 这~29tokens/s的Output Throughput是基线而非亮点数字。这个数据大概由三个因素压低 -- 每个token都执行cuda.synchronize()以保证计时准确, Batching未实现，KV Cache没有自定义等因素。这些是后续里程碑的优化空间。 


# M1的局限性
1. 仅支持单请求 -- batch=1写死，无法支持并发. 
2. 依赖HuggingFace的KV Cache，而非自定义实现。-- HuggingFace的KV Cache因为使用`torch.cat`导致性能较差，M2会讲到。 
3. 不支持Prefix Caching -- 重复的Prompt会从头算

# 附录 / 备注
设计文档: [m1_design.md](../design/m1_design.md)
Benchmark 配置:[benchmark_environment.md](../design/benchmark_environment.md)