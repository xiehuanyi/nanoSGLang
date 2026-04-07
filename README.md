# nanoSGLang

[中文版](README_zh.md)

A minimal, high-performance LLM serving engine built from scratch in ~4000 lines of Python + PyTorch.
Implements the core techniques from [SGLang](https://github.com/sgl-project/sglang) in a clean, educational codebase.

Tested on **Qwen2.5-0.5B** — reaches **3493 tok/s** vs SGLang's 3201 tok/s on RTX A5000.

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
    causal_lm.py      # Llama/Qwen2 model implementation, fused RMSNorm
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

Benchmark: 32 requests, prompt 100-512 tokens, output 64-128 tokens, **Qwen2.5-0.5B**, RTX A5000, greedy decoding.

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

## Tested Models

- **Qwen2.5-0.5B** (verified correctness and throughput)

The model implementation follows the Llama/Qwen2 architecture. Other models sharing this architecture (e.g. Llama, Qwen2) may work but have not been tested.

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
