# nanoSGLang

A minimal, high-performance LLM serving engine built from scratch in ~4000 lines of Python + PyTorch.
Implements the core techniques from [SGLang](https://github.com/sgl-project/sglang) in a clean, educational codebase.

**Matches SGLang throughput** on Qwen2.5-0.5B (3493 tok/s vs 3201 tok/s) with:
- FlashInfer paged attention (zero-copy KV cache)
- CUDA Graph for decode
- Fused RMSNorm + residual kernels
- Fused GPU sampling

## Architecture

```
nano_sglang/
  engine/
    engine.py         # Continuous batching loop, FlashInfer integration, CUDA Graph
    scheduler.py      # Batch scheduling with chunked prefill
    paged_kv_cache.py # Block-based KV cache (FlashInfer NHD layout)
    radix_cache.py    # Prefix-sharing radix tree cache
    request.py        # Request lifecycle (WAITING -> RUNNING -> FINISHED)
    sampling.py       # Token sampling (FlashInfer fused / PyTorch fallback)
    kv_cache.py       # Naive contiguous KV cache (Phase 1)
    overlap.py        # CPU/GPU overlap scheduling (experimental)
  model/
    causal_lm.py      # Llama/Qwen2 model from scratch, fused RMSNorm
    tokenizer.py      # HuggingFace tokenizer wrapper
  server/
    api.py            # OpenAI-compatible REST API (FastAPI)
    metrics.py        # TTFT, TBT, throughput tracking
  decode/
    quantization.py   # FP8 / AWQ quantization
    speculative.py    # Speculative decoding (draft + verify)
    structured.py     # JSON-schema constrained generation
  distributed/
    tensor_parallel.py # Multi-GPU tensor parallelism
main.py               # Server entry point
benchmark.py           # Load testing tool
benchmarks/            # Correctness & throughput benchmarks
```

## Quick Start

### Install

```bash
pip install -r requirements.txt

# Required for best performance:
pip install flashinfer-python

# Optional (SM >= 80):
pip install flash-attn --no-build-isolation
```

### Run Server

```bash
# Continuous batching mode (recommended):
python main.py --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --num-blocks 8000 --max-batch-tokens 8192

# Naive mode (single request, no batching):
python main.py --model-path Qwen/Qwen2.5-0.5B-Instruct --naive
```

### API Usage

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 64,
  "stream": true
}'

# Text completion
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "qwen",
  "prompt": "The capital of France is",
  "max_tokens": 32
}'
```

### Offline Batch Inference

```python
from benchmarks.nano_llm import NanoLLM, make_sampling_params

llm = NanoLLM(model_path="Qwen/Qwen2.5-0.5B-Instruct", num_blocks=8000)
results = llm.generate(
    prompt_token_ids=[[1, 2, 3], [4, 5, 6]],
    sampling_params=[make_sampling_params(max_tokens=64)] * 2,
)
for r in results:
    print(r.text)
```

## Performance

Benchmark: 32 requests, prompt 100-512 tokens, output 64-128 tokens, Qwen2.5-0.5B, RTX A5000.

| System | Throughput (tok/s) |
|--------|-------------------|
| nanoSGLang (all optimizations) | **3493** |
| SGLang 0.5.6 (default) | 3201 |
| nanoSGLang (no CUDA Graph) | 1400 |
| nanoSGLang (legacy, no FlashInfer) | 437 |

### Key Optimizations

| Technique | Impact |
|-----------|--------|
| FlashInfer paged attention | Eliminates O(layers x reqs x tokens) KV copy |
| CUDA Graph for decode | Removes kernel launch overhead (~2x decode speedup) |
| Fused RMSNorm + residual | Single kernel for norm + residual add |
| Fused sampling | GPU-native softmax + top-k/top-p sampling |
| Logit-index optimization | Only compute lm_head for last token per request |

## Features

- **Continuous batching** with mixed prefill + decode
- **Chunked prefill** for long prompts
- **Paged KV cache** with block-level memory management
- **FlashInfer paged attention** (zero-copy, NHD layout)
- **CUDA Graph** capture/replay for decode batches
- **Fused operators** (RMSNorm, sampling) via FlashInfer
- **OpenAI-compatible API** with streaming SSE
- **Speculative decoding** (draft model + verify)
- **FP8 / AWQ quantization**
- **Tensor parallelism** (multi-GPU)
- **Radix prefix cache** (KV sharing across requests)

## Supported Models

- Llama 2 / 3 / 3.1
- Qwen2 / Qwen2.5
- Any model with the Llama architecture

## Benchmarks

```bash
# Correctness (compare with SGLang, greedy decoding):
python benchmarks/bench_correctness.py --model-path Qwen/Qwen2.5-0.5B --system both

# Throughput:
python benchmarks/bench_throughput.py --model-path Qwen/Qwen2.5-0.5B --system nano

# Load testing:
python benchmark.py --url http://localhost:8000 --num-requests 100 --concurrency 32
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA GPU (SM >= 80 for FlashInfer / FlashAttention)
- `flashinfer-python` (for paged attention + fused ops)

---

# nanoSGLang (中文)

一个从零实现的高性能 LLM 推理引擎，约 4000 行 Python + PyTorch 代码。
以简洁的教学代码实现了 [SGLang](https://github.com/sgl-project/sglang) 的核心技术。

**吞吐量与 SGLang 持平**：在 Qwen2.5-0.5B 上达到 3493 tok/s（SGLang 为 3201 tok/s），核心优化包括：
- FlashInfer paged attention（零拷贝 KV cache）
- CUDA Graph（decode 阶段图捕获回放）
- 融合 RMSNorm + 残差连接算子
- 融合 GPU 采样

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
pip install flashinfer-python

# 启动服务
python main.py --model-path Qwen/Qwen2.5-0.5B-Instruct --num-blocks 8000

# 测试
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

## 性能数据

基准测试：32 请求，prompt 100-512 tokens，输出 64-128 tokens，Qwen2.5-0.5B，RTX A5000。

| 系统 | 吞吐量 (tok/s) |
|------|---------------|
| nanoSGLang（全部优化） | **3493** |
| SGLang 0.5.6（默认配置） | 3201 |
| nanoSGLang（无 CUDA Graph） | 1400 |
| nanoSGLang（无 FlashInfer） | 437 |

## 项目结构

```
nano_sglang/
  engine/          # 推理引擎：连续批处理、调度、KV 缓存管理
  model/           # 模型实现：Llama/Qwen2 from scratch
  server/          # OpenAI 兼容 REST API
  decode/          # 量化、投机解码、结构化输出
  distributed/     # 多卡张量并行
```

## 核心优化

| 优化技术 | 作用 |
|---------|------|
| FlashInfer paged attention | 消除 O(层数 x 请求数 x 序列长度) 的 KV 拷贝 |
| CUDA Graph | 消除 decode 阶段 kernel launch 开销（~2x 加速） |
| 融合 RMSNorm + 残差 | 将 norm 和残差连接合并为单个 kernel |
| 融合采样 | GPU 原生 softmax + top-k/top-p 采样 |
| Logit 索引优化 | 只对每个请求最后一个 token 计算 lm_head |

## 支持的模型

- Llama 2 / 3 / 3.1
- Qwen2 / Qwen2.5
- 所有 Llama 架构兼容模型
