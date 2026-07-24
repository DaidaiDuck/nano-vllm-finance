M4目标: 实现Continuous Batching，支持多请求并发处理

具体功能:
- 实现Scheduler
    - Prefill + Decode混合GPU Forward
    - 实现最简单的Recompute抢占策略 

实现步骤:
1. 开发Scheduler (scheduler.py), 写Scheduler的单元测试 (test_scheduler.py) 
2. 修改paged_engine.py, 引入scheduler. 
2. 开发GPUModelRunner (gpu_model_runner.py)
3. 修改ENGINE_CTX + paged_attn_forward，paged_attn_forward内部换成flash_attn_varlen_func，用随机数据对paged_attn_forward进行单元测试。
4. 正确性验证，确保batch = 1,2,4,8,..并发场景下每个Request的输出和它单独跑的时候一致。 
5. 减少num_blocks逼出preemption，测试抢占恢复能力.
6. 使用真实share_gpt数据进行测试，和M1,M2,M3进行性能比较，主要比较Throughput和延迟. 

测试验收:
- Scheduler CPU测试，验证功能是否符合预期
- 正确性测试: 确保并发的请求输出和单个请求的输出完全一致，因为Continous Batching最容易出现的问题是requestA的token串到requestB了.

---

# M4 测试 & Benchmark 清单

分三类：① Mac 本地 CPU 单测（纯算术，不碰 GPU/模型）；② pod 集成测试（需 CUDA + flash-attn）；③ benchmark 脚本（pod）。

## ① CPU 单元测试（Mac 本地，`.venv` 直接跑）

| 文件 | 测什么 | 命令 |
|---|---|---|
| `tests/test_gpu_model_runner.py` | `_prepare_inputs` 打包逻辑（cu_seqlens / slot_mapping / block_table / logits_indices），含乱序物理块、chunked prefill、纯 decode 等边界 | `python -m pytest tests/test_gpu_model_runner.py -q` |
| `tests/test_scheduler.py` | `schedule()` + `update_from_output()`：调度 / token 预算 / 尾部抢占 / 停止判定 | `python -m pytest tests/test_scheduler.py -q` |
| `tests/test_sampler.py` | `sample_one`（greedy/top-k/top-p）+ batched `sample`（每请求各自的 SamplingParams） | `python -m pytest tests/test_sampler.py -q` |
| `tests/test_types.py` | `Request` / `RequestOutput` / `FinishReason` / `SamplingParams` | `python -m pytest tests/test_types.py -q` |
| `tests/test_block_pool.py` | `BlockPool` + `KVCacheManager` 分配 / 释放 / OOM | `python -m pytest tests/test_block_pool.py -q` |
| `tests/test_kv_cache.py` | KV cache 结构 | `python -m pytest tests/test_kv_cache.py -q` |
| `tests/test_serving_benchmark.py` | `arrival` 到达过程 + `ServingMetrics` / goodput / 百分位 算术 | `python -m pytest tests/test_serving_benchmark.py -q` |

一条命令全跑 CPU 部分（GPU 门控的会自动 skip）：
```bash
.venv/bin/python -m pytest tests/ -q      # 预期 ~71 passed, ~20 skipped
```

## ② GPU / pod 集成测试（需 CUDA + flash-attn，`NANO_VLLM_INTEGRATION=1`）

| 文件 | 测什么 | 命令 |
|---|---|---|
| `tests/test_m4_vs_hf.py` | **M4 正确性验收**：batch ∈ {1,2,4,6,8} 每个请求 == HF greedy；batch 不变性（单发 vs 混批一致）；per-request 不同 max_tokens 各自生效 | 见下 |
| `tests/test_m3_vs_hf.py` | M3 回归：确认 M4 改动没破坏 M3 单请求路径 | 见下 |
| `tests/test_flash_attn_shapes.py` | flash-attn 张量形状约定 | `python -m pytest tests/test_flash_attn_shapes.py -q` |

```bash
# M4 正确性（核心验收门）
NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct \
    python -m pytest tests/test_m4_vs_hf.py -v

# M3 回归
NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct \
    python -m pytest tests/test_m3_vs_hf.py -v
```

> ⚠️ 默认用 **3B** 不用 0.5B：小模型 bf16 数值会翻转接近平手的 argmax，导致晚期 token 假性不一致（数值问题，非 bug）。

## ③ Benchmark 脚本（pod）

| 脚本 | 干什么 | 命令 |
|---|---|---|
| `benchmarks/serving_benchmark.py` | **M4 主 benchmark**：open-loop 扫 QPS，出吞吐 / TTFT-TPOT-E2E 的 P50/P95/P99 / goodput；首跑落盘共享 specs | 见下 |
| `benchmarks/vllm_client.py` | **vLLM 对照**：同一 specs + 同到达表 + 同指标，打 vLLM OpenAI server | 见 vLLM 对照 |
| `benchmarks/m3_benchmark.py` | M1/M2/M3 单请求 latency（closed-loop 对照基线） | `python -m benchmarks.m3_benchmark` |
| `benchmarks/kv_footprint.py` | 分页节省的 KV 显存 | `python -m benchmarks.kv_footprint --block-size 256` |

```bash
# M4 serving benchmark：扫 QPS + ShareGPT 真实数据
python -m benchmarks.serving_benchmark \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-requests 64 \
    --qps 1 2 4 8 inf \
    --dataset sharegpt
```

### vLLM 对照（组件 4）—— 必须逐字节同 prompt 才公平

**不要用 vLLM 自带的 `benchmark_serving.py`**：它会自己重新采样+过滤+分词 ShareGPT，选不到和 `load_sharegpt(seed=42)` 相同的那批 prompt。正确做法是**共享一个 specs 文件 + 同一套测量口径去打两个后端**，让"唯一变量只剩引擎"。

流程（pod 上）：
```bash
# 0. 装依赖 + 起 vLLM server（一个终端）
pip install vllm aiohttp
vllm serve Qwen/Qwen2.5-3B-Instruct --disable-log-requests

# 1. 跑 nano：首次运行会把选中的 specs 落盘到 --specs-file
python -m benchmarks.serving_benchmark \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-requests 64 --qps 1 2 4 8 inf --dataset sharegpt \
    --specs-file benchmarks/baselines/specs.json

# 2. 跑 vLLM：读【同一个】 specs 文件，同 QPS、同到达时间表（seed=42）、同 ServingMetrics
python -m benchmarks.vllm_client \
    --model Qwen/Qwen2.5-3B-Instruct \
    --base-url http://localhost:8000 \
    --specs-file benchmarks/baselines/specs.json \
    --qps 1 2 4 8 inf
```

公平性由三点保证（都在 `benchmarks/vllm_client.py` 里）：
1. 两个后端读**同一个** `benchmarks/baselines/specs.json`（`dump_specs`/`load_specs`）——prompt 与 output_len 完全一致。
2. vLLM 那边用**同一套 open-loop driver**（同 `generate_arrival_times` + 同 seed + 同 `compute_serving_metrics`），不走 vLLM 自己的 harness。
3. prompt 用**同一个 chat template + tokenizer** 编码后以 **token ids** 发给 `/v1/completions`，消除重新分词的偏差。

结果两边都存进 `benchmarks/baselines/`（nano 存 `nano-vllm_m4_*`，vLLM 存 `vllm_m4_*`），并排出表。诚实预期：nano 吞吐更低 / P99 更高（vLLM 有 CUDA graph、优化 kernel/调度器）；价值在于如实测出差距，并展示连续批处理相比 M1–M3 顺序循环补上了多少。详见 `docs/design/m4_benchmark_plan.md`。

## 跑 ②③ 之前必须先修的引擎 blocker

1. `PagedEngine.step()`：空调度步（`total_num_scheduled_tokens == 0`）要 `return []`，否则 `execute_model` 返回 `None` → `update_from_output` deref 崩，且 `LLM.generate` 空转死循环。
2. `scheduler.update_from_output`：`RequestOutput.token_ids = list(request._output_token_ids)`（拷贝），否则 benchmark 每步存下的 output 会随后续 step 一起变长，TPOT 全错。
3. pod 上先验 `inspect.signature(flash_attn.flash_attn_varlen_func)`，确认 `seqused_k` / `block_table` / `cu_seqlens_k` 参数与 `batch_attn_forward` 的调用一致。

## 推荐执行顺序

```
本地（现在就能做）：
  1. .venv/bin/python -m pytest tests/ -q                 # CPU 全绿

修完上面 3 个 blocker 后，上 pod：
  2. 验 flash_attn_varlen_func 签名
  3. pytest tests/test_m4_vs_hf.py + test_m3_vs_hf.py     # 正确性门
  4. python -m benchmarks.serving_benchmark ...           # 性能数据（扫 QPS）
  5. vLLM 对照表
```


