#!/bin/bash
#SBATCH --job-name=nano_sglang_test
#SBATCH --output=nano_sglang_test_%j.log
#SBATCH --error=nano_sglang_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

echo "========================================"
echo "  nanoSGLang Test"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Date: $(date)"
echo "========================================"

cd /ibex/project/c2334/huanyi/nanoSGLang

# Model — _resolve_model_path handles HF hub download automatically
SNAPSHOT_DIR="Qwen/Qwen2.5-0.5B-Instruct"
echo "Model: $SNAPSHOT_DIR"

# ========================================
# Test 1: Smoke test — model loading + single forward pass
# ========================================
echo ""
echo "=== Test 1: Model loading + forward pass ==="
python3 -c "
import torch
from nano_sglang.model.llama import load_model_from_pretrained

model, config = load_model_from_pretrained('${SNAPSHOT_DIR}', device='cuda', dtype=torch.float16)
print(f'Config: {config.model_type}, layers={config.num_hidden_layers}, hidden={config.hidden_size}')

# Single forward pass
ids = torch.tensor([[1, 2, 3, 4, 5]], device='cuda')
pos = torch.arange(5, device='cuda').unsqueeze(0)
mask = torch.zeros(1, 1, 5, 5, device='cuda', dtype=torch.float16)
mask[0, 0] = torch.triu(torch.full((5,5), float('-inf')), diagonal=1).half()

with torch.inference_mode():
    logits = model(ids, pos, attention_mask=mask)
print(f'Logits shape: {logits.shape}')
print(f'GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB')
print('Test 1 PASSED')
"

# ========================================
# Test 2: Naive mode — single request generation
# ========================================
echo ""
echo "=== Test 2: Naive single-request generation ==="
python3 -c "
import asyncio
import torch
from nano_sglang.engine.engine import InferenceEngine
from nano_sglang.engine.request import SamplingParams

async def test():
    engine = InferenceEngine(
        model_path='${SNAPSHOT_DIR}',
        device='cuda',
        dtype=torch.float16,
        max_seq_len=512,
        naive=True,
    )

    prompt_tokens = engine.tokenizer.apply_chat_template([
        {'role': 'user', 'content': 'What is 2+2? Answer briefly.'}
    ])
    print(f'Prompt tokens: {len(prompt_tokens)}')

    sampling = SamplingParams(max_tokens=50, temperature=0.0)
    output_text = ''
    token_count = 0

    async for out in engine.generate_stream(prompt_tokens, sampling):
        output_text += out.text
        token_count += 1
        if out.finished:
            break

    print(f'Generated {token_count} tokens')
    print(f'Output: {output_text}')
    print(f'GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB')
    print('Test 2 PASSED')

asyncio.run(test())
"

# ========================================
# Test 3: Continuous batching mode — concurrent requests
# ========================================
echo ""
echo "=== Test 3: Continuous batching engine ==="
python3 -c "
import asyncio
import time
import torch
from nano_sglang.engine.engine import InferenceEngine
from nano_sglang.engine.request import SamplingParams

async def test():
    engine = InferenceEngine(
        model_path='${SNAPSHOT_DIR}',
        device='cuda',
        dtype=torch.float16,
        max_seq_len=512,
        num_blocks=128,
        block_size=16,
        max_batch_tokens=512,
        max_running_requests=8,
        prefill_chunk_size=256,
        naive=False,
    )
    await engine.start()

    prompts = [
        'What is Python?',
        'Explain gravity briefly.',
        'Write a haiku about coding.',
    ]

    sampling = SamplingParams(max_tokens=30, temperature=0.0)

    async def run_one(prompt):
        tokens = engine.tokenizer.apply_chat_template([
            {'role': 'user', 'content': prompt}
        ])
        text = ''
        count = 0
        async for out in engine.generate_stream(tokens, sampling):
            text += out.text
            count += 1
            if out.finished:
                break
        return prompt, text, count

    start = time.time()
    results = await asyncio.gather(*[run_one(p) for p in prompts])
    elapsed = time.time() - start

    total_tokens = 0
    for prompt, text, count in results:
        print(f'  [{count} tokens] {prompt} -> {text[:80]}...')
        total_tokens += count

    print(f'Total: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.1f} tok/s)')
    print(f'GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB')

    await engine.stop()
    print('Test 3 PASSED')

asyncio.run(test())
"

# ========================================
# Test 4: HTTP server + benchmark
# ========================================
echo ""
echo "=== Test 4: HTTP server + benchmark ==="

# Start server in background
python3 main.py \
    --model-path "${SNAPSHOT_DIR}" \
    --device cuda \
    --dtype float16 \
    --max-seq-len 512 \
    --num-blocks 128 \
    --block-size 16 \
    --port 18234 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
for i in $(seq 1 60); do
    if curl -s http://localhost:18234/health | grep -q '"ok"'; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Health check
echo ""
echo "Health check:"
curl -s http://localhost:18234/health | python3 -m json.tool

# Single non-streaming request
echo ""
echo "Single chat request (non-streaming):"
curl -s http://localhost:18234/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 30,
        "temperature": 0
    }' | python3 -m json.tool

# Single streaming request
echo ""
echo "Streaming chat request:"
curl -s -N http://localhost:18234/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 30,
        "temperature": 0,
        "stream": true
    }' 2>&1 | head -20

# Run benchmark
echo ""
echo "Running benchmark (10 requests, concurrency=2):"
python3 benchmark.py \
    --url http://localhost:18234 \
    --num-requests 10 \
    --concurrency 2 \
    --max-tokens 32 \
    --temperature 0

# Check metrics
echo ""
echo "Server metrics:"
curl -s http://localhost:18234/metrics | python3 -m json.tool

# Cleanup (|| true to avoid set -e exit)
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "========================================"
echo "  ALL TESTS COMPLETED"
echo "========================================"
