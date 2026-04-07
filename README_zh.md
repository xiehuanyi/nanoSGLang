# nanoSGLang

[English](README.md)

一个从零实现的高性能 LLM 推理引擎，约 4000 行 Python + PyTorch 代码。
以简洁的教学代码实现了 [SGLang](https://github.com/sgl-project/sglang) 的核心技术。

在 **Qwen2.5-0.5B** 上测试，吞吐量达到 **3469 tok/s**（SGLang 为 3201 tok/s），RTX A5000。

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

基准测试：32 请求，prompt 100-512 tokens，输出 64-128 tokens，**Qwen2.5-0.5B**，RTX A5000，greedy decoding。

| 系统 | 吞吐量 (tok/s) |
|------|---------------|
| nanoSGLang（FlashInfer + 融合算子） | **3469** |
| SGLang 0.5.6（默认配置，CUDA Graph 开启） | 3201 |
| nanoSGLang（Legacy，无 FlashInfer） | 177 |

### 逐项消融

在同一负载上逐个开关各优化项，实测增量：

| 配置 | tok/s | 增量 |
|------|-------|------|
| Legacy（flash_attn + 逐 token KV 拷贝） | 177 | 基线 |
| + FlashInfer paged attention（零拷贝 KV） | 3076 | **+1642%** |
| + CUDA Graph（decode） | 3077 | +0% |
| + 融合 RMSNorm + 融合采样 | 3469 | +13% |

**核心提升来自 FlashInfer paged attention**：通过 page table 间接寻址，消除了逐 token、逐 layer 的 KV 缓存拷贝循环（O(层数 × 请求数 × 序列长度) → O(1)）。这贡献了约 97% 的总加速。

CUDA Graph 在 0.5B 小模型上无可测量提升——模型太小，kernel launch 不是瓶颈。在更大的模型（7B+）上会有效果。

融合 RMSNorm（`flashinfer.norm.fused_add_rmsnorm`）和融合采样（`flashinfer.sampling`）贡献了真实的 +13%。

### 正确性

Greedy decoding（temperature=0）下与 SGLang 的输出对比，Qwen2.5-0.5B，6 条 prompt（关闭融合算子）：

| # | Prompt | 状态 | nanoSGLang | SGLang |
|---|--------|------|------------|--------|
| 0 | `Hello, my name is` | token 1 发散 | John **and** I am a 20 year old male... | John**.** I am a student of the University... |
| 1 | `The capital of France is` | **完全一致** | Paris. It is the largest city in Europe... | *（相同）* |
| 2 | `Write a Python function...` | **完全一致** | def sum_list(lst): return sum(lst)... | *（相同）* |
| 3 | `Question: What is 2 + 2?` | **完全一致** | 4\nIs the above claim true?... | *（相同）* |
| 4 | `Once upon a time, in a small village,` | token 12 发散 | ...Owl had **a big family** ... | ...Owl had **a special talent for predicting**... |
| 5 | `The three laws of robotics are: 1.` | token 19 发散 | ...The law of the minimum **time**... | ...The law of the minimum **effort**... |

3/6 完全一致。发散处是 top-2 logits 非常接近的位置，bf16 精度下不同 attention 后端的浮点舍入差异导致 argmax 翻转。所有输出语义连贯。SGLang 自身切换后端（FlashInfer vs Triton）也有同等程度的发散。

## 项目结构

```
nano_sglang/
  engine/          # 推理引擎：连续批处理、调度器、KV 缓存管理
  model/           # 模型实现（Llama/Qwen2 架构）
  server/          # OpenAI 兼容 REST API
  decode/          # 量化、投机解码、结构化输出
  distributed/     # 多卡张量并行
```

## 已测试模型

- **Qwen2.5-0.5B**（已验证正确性和吞吐量）

模型实现基于 Llama/Qwen2 架构。其他同架构模型（如 Llama、Qwen2）理论上可用，但尚未测试。

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- CUDA GPU（SM >= 80，用于 FlashInfer / FlashAttention）
- `flashinfer-python`（paged attention + 融合算子）
