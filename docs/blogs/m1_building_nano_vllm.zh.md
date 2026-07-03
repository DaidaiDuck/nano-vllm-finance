# 从零构建 nano-vllm(一):一个正确的单请求引擎

[English](m1_building_nano_vllm.md) | **中文**

> 系列:一次一个里程碑,重新实现 vLLM 的核心思想。
> **M1 —— 基线。** 在 HuggingFace 之上把单请求 LLM 推理(prefill + decode)正确跑通,
> 配一个干净的 vLLM 风格 API,让之后每一步优化都有一个可信的参照物去超越。
>
> 设计文档:[m1_design.md](../design/m1_design.md) · 代码 tag:
> [m1](https://github.com/DaidaiDuck/nano-vllm-finance/tree/m1)

---

## TL;DR

nano-vllm 从零重建 vLLM 的核心,目的是揭示现代 LLM serving 的运作方式。第一篇(M1)
是基线:一个单请求引擎,在 HuggingFace 模型之上完成 prefill + decode,配一个自研
sampler 和干净的 `LLM` API。它刻意省略了全部优化——没有自定义 KV cache、没有 paging、
没有 batching。它唯一的要求是正确性:greedy 模式下其输出与 HuggingFace
**逐 token 一致**,从而为之后每一步优化提供一个可信的参照。

## 1. 为什么重建 vLLM?

vLLM 很快,但速度本身并不等于理解。本系列的目标是解释 LLM 推理**为什么**是这样的
结构——为什么存在 prefill 与 decode 两个阶段、KV cache 究竟提供了什么、sampling 与
streaming 如何工作——通过从第一性原理出发、一个里程碑一个里程碑地重建整个系统。M1 是
地基;后续篇章依次处理 KV cache、paging、batching。有缺陷的地基会把 bug 传播到其上的
一切,因此 M1 的唯一目标是正确。

## 2. 先正确,后速度

简单是 M1 的核心美德。那些让 vLLM 快的组件——自定义 KV cache、PagedAttention、
continuous batching、scheduler——被刻意排除在外。它们各自都是独立的里程碑,现在引入
任何一个,都意味着同时 debug 两个难题:generation loop **与**该优化。除了正在研究的
部分之外,其余全部委托给 HuggingFace,从而能够孤立地验证该循环,并为之后每一步优化
提供干净的前后对比。

## 3. 架构

该引擎是一个门面之后的三个组件:

```
LLM (用户入口)
  └─ SimpleEngine
       ├─ Tokenizer (HF)
       ├─ Model (HF, use_cache=True)   ← KV cache = HF DynamicCache
       └─ Sampler (自研)
```

`LLM` 是面向用户的入口(vLLM 风格:`llm.generate(prompt, params)`)。它包裹一个
`SimpleEngine`,后者持有 prefill + decode 循环,并协调 tokenizer、模型与 sampler。
KV cache 采用 HuggingFace 内置实现,通过 `use_cache=True` 复用,而非重新实现。

## 4. 生成循环:prefill + decode

生成循环是 M1 的核心,分两个阶段运行。**Prefill** 是对整个 prompt 的一次前向:它产出
下一个 token 的 logits,并用 prompt 的 key/value 填充 KV cache。**Decode** 随后一次
一个 token 地进行:

```python
# prefill 一次, 之后每步只喂一个 token, 复用 KV cache
outputs = model(input_ids, use_cache=True)          # prefill
next_token = sampler.sample(outputs.logits[0, -1])  # 第一个 token
yield next_token
for _ in range(max_tokens - 1):
    outputs = model([[next_token]], past_key_values=past_kv, use_cache=True)
    next_token = sampler.sample(outputs.logits[0, -1])
    yield next_token
```

关键技术在于:每个 decode 步**只喂入新的 token**,而非整段序列。`use_cache=True` 保留
此前全部的 key/value,因此第 *n* 步只做 O(1) 的输入工作,而非重读全部 *n* 个 token。
所需 logits 取自 `logits[0, -1, :]`——batch 0、**最后**一个位置、完整词表——因为模型在
每个位置都会输出一个分布,但只需要接在当前文本之后的那一个。循环在遇到 EOS token 或
达到 `max_tokens` 时终止。

## 5. 手写 sampler

HuggingFace 的 `generate()` 被刻意规避:它是一个不便定制的黑盒,且 batched sampling
(后续里程碑)无论如何都需手写,因此自研实现优于之后会被弃用的实现。`Sampler` 是一条
短流水线:`temperature == 0` 时走 greedy 捷径(直接 `argmax`),否则 temperature 缩放
→ top-k → top-p → softmax → `multinomial`。微妙之处在于 masking 细节——过滤通过向被
拒绝的 logits 写入 `-inf` 实现(softmax 后其概率归零),而 top-p 需要一个右移一格,
以便**刚好越过**阈值的那个 token 被保留而非丢弃。

## 6. Chat template 要求

一个常见错误是把裸 prompt 直接喂给 instruction-tuned 模型。此类模型在一套带特殊
token 的对话格式上训练;缺少它,模型会**续写**文本而非回答。修正方式是 tokenizer 的
chat template:

```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
```

`add_generation_prompt=True` 会追加 assistant 的起始标记(如
`<|im_start|>assistant\n`),标示轮到模型作答。省略它会产出退化的输出——一种常见且
容易被忽视的失效模式。

## 7. Streaming 与 TTFT 测量

Streaming 并不等同于 HTTP 或 SSE——一个 Python generator **本身就是** streaming。将
循环中的 `return` 改为 `yield` 几乎零成本,却暴露出正确测量延迟所需的唯一量:第一个
token 产出的精确时刻。这即是 TTFT(Time To First Token),没有逐 token 的接口便无法
测量。有一处微妙之处——CUDA kernel 异步执行,因此在取时间戳之前调用
`torch.cuda.synchronize()`;否则测得的将是任务被**排队**的时刻,而非 GPU 完成计算的
时刻。

## 8. 正确性:nano-vllm == HuggingFace

最重要的测试断言 token 级别的一致:

```python
# tests/test_m1_vs_hf.py — greedy 必须和 HF 完全一致
assert nano.token_ids == hf_greedy_ids
```

greedy 模式下,引擎的输出必须与 HuggingFace **逐 token 一致**。有一处需注意:Qwen 的
`generation_config` 默认采样(`do_sample=True`,`temperature=0.7`),因此必须传入
`do_sample=False` 强制 HF 进入 greedy——否则参照本身是随机的,测试失去意义。完整的
M1 测试套件在 A100 80GB SXM + Qwen2.5-3B-Instruct 上 **23/23 全过**(12 个 CPU 单测 +
11 个 GPU 集成测试),`test_m1_vs_hf` 在每个 prompt 上均匹配。这种逐 token 的一致性
确立了 generation loop 的正确性,并作为之后每个里程碑都必须保持为绿的 non-regression
契约。

## 9. Benchmark

对三个单请求场景进行了 benchmark(`Throughput` 为 output tokens/s):

| 场景 | Prompt | Output | Throughput | P99 延迟 | TTFT | TPOT |
|----------|--------|--------|------------|-------------|------|------|
| short_chat | ~125 | 100 | 29.4 tok/s | 3.62 s | 36 ms | 34.0 ms |
| medium_chat | ~526 | 200 | 29.2 tok/s | 7.38 s | 38 ms | 34.2 ms |
| long_context | ~1999 | 100 | 29.2 tok/s | 3.47 s | 89 ms | 33.7 ms |

结果高度一致。**Throughput 恒定在 ~29 tok/s**,与 prompt 长度无关——对单请求而言,它
等于 decode 速度(≈ 1/TPOT)。**TPOT 维持在 ~34 ms**,从 125 到 2000 个 prompt token
几乎不变,因为 decode 是 memory-bandwidth-bound 的:每一步的主要开销是把模型权重流过
来,而非对 cache 做 attention。**TTFT 随 prompt 长度增长**(36 → 89 ms),因为 prefill
是 compute-bound 的——只是一个 ~35 ms 的固定开销将其掩盖,直到 prompt 变长才显现。

核心结论是:端到端延迟遵循一个简单的规律,并已验证到 1% 以内:

```
latency ≈ TTFT + output_len × TPOT
```

该公式将两个物理阶段分离——compute-bound 的 prefill(计入 TTFT)与 memory-bound 的
decode(计入 TPOT)——并解释了为何 `long_context` 与 `short_chat` 的延迟几乎相同,
尽管 prompt 相差 16 倍:两者都产出 100 个 token,而主导墙钟时间的是 **output** 长度。
一处需说明:~29 tok/s 是基线而非亮点数字——为保证计时准确,每个 token 都执行了
`cuda.synchronize()`,且尚无 batching。

## 10. 留待后续里程碑

M1 是刻意精简的。其明确的局限:

- **仅支持单请求** —— `batch=1` 写死,无并发。
- **依赖 HuggingFace 内置的 KV cache**,而非自定义实现。
- **无 HTTP serving** —— streaming 是进程内的 generator。
- **无 prefix caching** —— 重复的 prompt 会从头重算。
- **EOS 是唯一的停止条件** —— 不支持自定义 stop strings。

这些都不是 M1 的缺陷,而是构成了路线图。在拥有一个正确、可测量的基线,以及一个保证
它不会被悄然破坏的测试之后,后续篇章将一件件替换这些组件——从 KV cache 开始。

---

## 附录 / 备注

- 设计文档:[m1_design.md](../design/m1_design.md)
- Benchmark 配置:[benchmark_environment.md](../design/benchmark_environment.md)
- 下一篇:**Part 2 —— KV cache**([m2_design.md](../design/m2_design.md))
