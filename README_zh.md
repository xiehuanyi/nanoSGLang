# nanoSGLang

[English](README.md)

一个从零实现的高性能 LLM 推理引擎，约 4000 行 Python + PyTorch 代码。
以简洁的教学代码实现了 [SGLang](https://github.com/sgl-project/sglang) 的核心技术。

在 **Qwen2.5-0.5B** 上测试，吞吐量达到 **3493 tok/s**（SGLang 为 3201 tok/s），RTX A5000。

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
| nanoSGLang（全部优化） | **3493** |
| SGLang 0.5.6（默认配置） | 3201 |
| nanoSGLang（无 CUDA Graph） | 1400 |
| nanoSGLang（无 FlashInfer） | 437 |

## 项目结构

```
nano_sglang/
  engine/          # 推理引擎：连续批处理、调度器、KV 缓存管理
  model/           # 模型实现（Llama/Qwen2 架构）
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

## 功能

- **连续批处理**（continuous batching），混合 prefill + decode
- **分块预填充**（chunked prefill）
- **分页 KV 缓存**，块级内存管理
- **FlashInfer paged attention**（零拷贝，NHD 布局）
- **CUDA Graph** 捕获/回放（decode 阶段）
- **融合算子**（RMSNorm、采样），通过 FlashInfer 实现
- **OpenAI 兼容 API**，支持流式 SSE

## 已测试模型

- **Qwen2.5-0.5B**（已验证正确性和吞吐量）

模型实现基于 Llama/Qwen2 架构。其他同架构模型（如 Llama、Qwen2）理论上可用，但尚未测试。

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- CUDA GPU（SM >= 80，用于 FlashInfer / FlashAttention）
- `flashinfer-python`（paged attention + 融合算子）
