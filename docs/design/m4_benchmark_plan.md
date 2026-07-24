# M4 生产级 Benchmark — 方案

## Context / 背景
M1–M3 的 benchmark 是**单请求、顺序执行(closed-loop)、synthetic prompt、输出长度写死**——
测长度驱动的 latency/显存够用,但**不代表生产**。M4 加了**连续批处理**(`nano_vllm/paged/scheduler.py`),
所以现在可以、也必须在**并发 + open-loop + 真实数据**下测,并和 **vLLM** 对比。本方案只覆盖那 4 项
benchmark 补充(scheduler/engine 是另外的活)。

目标:一个诚实的"能不能扛生产"的答案——**并发负载下的吞吐 / P99 / goodput,和 vLLM 同数据同负载对比**。

## ⚠️ 依赖顺序(先读这个)
并发 benchmark 只有在 **M4 engine 真能并发服务**之后才能跑。把工作拆开,让"不依赖 engine"的部分先落地:

| 项 | 依赖 M4 engine? | 什么时候做 |
|---|---|---|
| 1. 真实数据集(`load_sharegpt` → specs) | ❌ 纯 CPU | 现在 |
| 2. 到达过程(Poisson/QPS) | ❌ 纯 CPU | 现在 |
| 3. 并发压测 harness + goodput 指标 | ✅ 需要 engine API | scheduler 跑通后 |
| 4. vLLM 对比 | 独立工具(pod 上 `pip install vllm`) | 最后 |

**harness 需要 M4 engine 提供的 API**(写 scheduler 时顺手定好):一个 open-loop 的提交 + step 循环——
`engine.add_request(request)`(到达时入队)、一个反复调 `scheduler.schedule()` → 跑模型 →
`scheduler.update_from_output()` 的 driver、以及**逐 token 的 stream/回调**,这样 harness 才能给每个请求的
**首 token**和**完成**打时间戳。**没有逐 token 时间戳,就算不出负载下的 TTFT/TPOT。**

---

## 组件 1 — 真实数据集:重写 `load_sharegpt`,产出 (prompt, output_len)
文件:`benchmarks/datasets.py`(现在的 `load_sharegpt` 是个 stub:只取第一条文本、没有输出长度、失败返回 None)。

返回**请求规格**,不是裸字符串:`list[(prompt_text: str, output_len: int)]`。
- 用 `datasets` 加载 `Aeala/ShareGPT_Vicuna_unfiltered`(或 `anon8231489123/ShareGPT_Vicuna_unfiltered`)。
- 每段对话:第一条 **human** = `prompt`;紧跟的 **gpt** 那条的**分词后长度** = `output_len`
  (**真实回复长度** → 决定该请求的 `max_tokens`)。
- 分词;**过滤**:`prompt_len < 4` 丢掉,`prompt_len + output_len > max_model_len` 丢掉;`output_len` 可选封顶。
  用固定 `seed` 确定性采样 `num` 条。
- 签名:`load_sharegpt(num, tokenizer, max_model_len, seed=42) -> list[tuple[str,int]]`。

**为什么这是头号真实性修复**:现在每个请求都跑到**全局写死**的 `output_len`(scenario 的字段)。真实流量的
**输出长度是分布**;让每个请求跑到它自己的真实长度,吞吐/goodput 才有意义。
**照抄参照**:vLLM 的 `benchmarks/benchmark_serving.py::sample_sharegpt_requests` 就是干这个的过滤,照它的形状写。

## 组件 2 — 到达过程:Poisson / 固定 QPS(open-loop)
新建:`benchmarks/arrival.py`(或并进 serving benchmark)。
- 给定目标 `qps`(λ)和 N 个请求,产出**到达时间戳**(相对 t0 的秒):
  - Poisson:间隔 ~ Exponential(λ) → `t[i] = t[i-1] + random.expovariate(qps)`。
  - 固定:`t[i] = i / qps`。
  - `qps = inf` → 全部 t=0(满载 / 离线吞吐模式;= vLLM `--request-rate inf`)。
- load driver 在墙钟 `t0 + t[i]` 提交第 i 个请求——**和完成无关**(open-loop)。对比 M1–M3 的 closed-loop
  (上一个跑完才发下一个)。open-loop 才能暴露排队和饱和拐点。
- **扫 QPS** {1, 2, 4, 8, …, inf},画"吞吐 vs P99"曲线,找饱和点。

## 组件 3 — 并发 harness + goodput 指标(P50/P99 + goodput)
新建:`benchmarks/serving_benchmark.py`(open-loop、并发)——m1/m2/m3 那套顺序 latency benchmark 保留原样,测单请求。
- Driver:一个 producer(线程或 asyncio task)按时间戳提交到达;engine step 循环并发跑;每个请求记录
  `arrival_t, first_token_t, finish_t, output_len`。
- **要加的指标**(扩展 `benchmarks/metrics.py`——它有 latency/TTFT/TPOT 的 P50/P95/P99,但**没有 goodput**、
  也没有"含排队"的 TTFT):
  - **TTFT = first_token_t − arrival_t** —— 现在**包含排队等待**(open-loop 的精髓;现在的 closed-loop TTFT 从不排队)。
  - **E2E 延迟 = finish_t − arrival_t**;**TPOT = (finish_t − first_token_t)/(output_len−1)**。
  - 报 TTFT / TPOT / E2E 的 **P50/P95/P99**;窗口内的 **输出 tok/s** 和 **req/s**。
  - **Goodput** = `满足所有 SLO 的请求数 / 墙钟时间`。SLO 阈值自定,比如 `TTFT ≤ 200ms 且 TPOT ≤ 50ms`(做成 CLI 参数)。
    这是**生产核心指标**:**达标请求的吞吐**(对应 vLLM 的 `--goodput`)。
  - 新 `ServingMetrics` dataclass + `compute_serving_metrics(records, duration, slo)`。
- 扩展 `benchmarks/scenarios.py`:`Scenario` 已有 `concurrency`;加 `qps: float` 和 `dataset: str`
  ("sharegpt" | "synthetic")。保留 `high_concurrency`,但用 **QPS 扫描**驱动,不用写死的并发数。

## 组件 4 — vLLM 对比(生产参照系)
vLLM **本地没装**(要 CUDA)→ 在 pod 上 `pip install vllm`。
- **同数据、同负载**:把**同一批 ShareGPT specs + 同一个 QPS** 喂给 vLLM。
- 推荐(权威、可信):直接用 **vLLM 自带的 `benchmark_serving.py`** 打 vLLM server
  (`vllm serve Qwen/Qwen2.5-3B-Instruct`),参数 `--dataset-name sharegpt --request-rate <qps>`。
  把 nano-vllm 的数放它旁边一张表;vLLM 的结果存进已存在但为空的 `benchmarks/baselines/`。
- **诚实预期**(M2/M3 的一贯基调):nano-vllm 会**吞吐更低 / P99 更高**(vLLM 有 CUDA graph、chunked prefill、
  优化 kernel/scheduler)。价值在于**如实测出差距**——并展示连续批处理相比朴素顺序循环把大部分差距补上了。

---

## Build order / 构建顺序
1. 组件 1(`load_sharegpt` → specs)—— CPU,单测长度分布。
2. 组件 2(arrival 生成)—— CPU,单测平均间隔 ≈ 1/λ。
3. [engine] scheduler + serving 循环 + 逐 token stream —— 是第 4 步的前提。
4. 组件 3(`serving_benchmark.py` + `ServingMetrics`)—— 需要 engine API。
5. 组件 4(pod 上 vLLM baseline),再出对比表。

## 验证
- 单测(CPU):`load_sharegpt` 产出真实的 `(prompt_len, output_len)` 分布(打印直方图);
  arrival 生成器平均间隔 ≈ `1/qps`;`qps=inf` → 全 t=0。
- 集成(pod):`serving_benchmark.py` 扫 QPS {1,2,4,8,inf} → "吞吐 vs P99" 曲线 + 每个 QPS 的 goodput。
  Sanity:低 QPS 时 P99 ≈ 单请求延迟;接近饱和时 P99 暴涨。
- 交叉核对:nano-vllm vs vLLM 在**完全相同**的 (数据, QPS) 下 → 一张对比表
  (吞吐、P50/P99 的 TTFT & TPOT、goodput)。

## Files / 涉及文件
- 新建:`benchmarks/serving_benchmark.py`、`benchmarks/arrival.py`。
- 重写:`benchmarks/datasets.py`(`load_sharegpt` → specs)。
- 扩展:`benchmarks/metrics.py`(`ServingMetrics` + goodput + SLO)、`benchmarks/scenarios.py`(`qps` + `dataset`)。
- engine 钩子(M4):`add_request`、跑 `Scheduler.schedule()/update_from_output()` 的 step 循环、逐 token 时间戳 stream。
- 结果:vLLM baseline → `benchmarks/baselines/`。

---

## 相关文档
- Scheduler 的 TODO 解答 & 设计决策:[m4_scheduler_notes.md](m4_scheduler_notes.md)
- Scheduler `schedule()` 的单元测试:`tests/test_scheduler.py`(纯 CPU,真实 `Request` + `KVCacheManager`)
