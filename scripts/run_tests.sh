#!/usr/bin/env bash
# scripts/run_tests.sh
#
# Reproducible M1 test run, ordered fast -> slow (CPU sanity first, GPU last).
# Pinned environment: docs/design/benchmark_environment.md
#   (A100 80GB SXM, Qwen2.5-3B bf16, PyTorch 2.4 / CUDA 12.4, RunPod).
#
# Usage:
#   bash scripts/run_tests.sh                                   # default: 0.5B model, fast
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_tests.sh    # test with the benchmark model
#
# Notes:
# - Stage 1 needs neither GPU nor a model (pure logic) — always runs.
# - Stage 2 needs CUDA + downloads the HF model; auto-skips if no CUDA.
# - The single most important test is tests/test_m1_vs_hf.py: it proves nano-vllm
#   greedy output matches HuggingFace token-for-token. It runs first in Stage 2.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "=================================================================="
echo " Stage 1/2  CPU unit tests (no GPU, no model) — sampler / types"
echo "=================================================================="
python -m pytest tests/test_sampler.py tests/test_types.py -v

echo
echo "=================================================================="
echo " Stage 2/2  GPU integration tests (needs CUDA + model)"
echo "            model = ${MODEL}"
echo "=================================================================="
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    NANO_VLLM_INTEGRATION=1 NANO_VLLM_TEST_MODEL="${MODEL}" \
        python -m pytest \
            tests/test_m1_vs_hf.py \
            tests/test_engine.py \
            tests/test_engine_m1.py \
            tests/test_generation.py \
            -v
else
    echo "CUDA not available — skipping Stage 2. (On the A100 pod these run.)"
fi

# --- M2 (not part of M1; MyKVCache is pure tensor logic, runs on CPU) ----------
# Uncomment once you start M2:
#   python -m pytest tests/test_kv_cache.py -v

echo
echo "Done. All requested M1 tests finished."
