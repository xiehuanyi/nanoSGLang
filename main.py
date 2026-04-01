"""
nanoSGLang — Entry point.

Usage:
    # Naive mode (Phase 1, single request):
    python main.py --model-path Qwen/Qwen2.5-0.5B-Instruct --naive

    # Continuous batching mode (Phase 2+):
    python main.py --model-path Qwen/Qwen2.5-0.5B-Instruct --port 8000

    # With FP8 quantization:
    python main.py --model-path Qwen/Qwen2.5-0.5B-Instruct --quantize fp8

    # Multi-GPU tensor parallelism:
    torchrun --nproc_per_node=2 main.py --model-path meta-llama/Llama-3.1-8B-Instruct --tp 2
"""

import argparse
import torch
import uvicorn

from nano_sglang.engine.engine import InferenceEngine
from nano_sglang.server import api


def main():
    parser = argparse.ArgumentParser(description="nanoSGLang server")
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model path or local directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-seq-len", type=int, default=4096)

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    # Engine mode
    parser.add_argument("--naive", action="store_true",
                        help="Use Phase 1 naive single-request mode")

    # Paged attention
    parser.add_argument("--num-blocks", type=int, default=256,
                        help="Number of KV cache blocks")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Tokens per KV cache block")

    # Scheduler
    parser.add_argument("--max-batch-tokens", type=int, default=4096,
                        help="Max tokens per forward step")
    parser.add_argument("--max-running-requests", type=int, default=64,
                        help="Max concurrent requests")
    parser.add_argument("--prefill-chunk-size", type=int, default=512,
                        help="Max prefill tokens per chunk (0=no chunking)")

    # Quantization
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["fp8", "awq"],
                        help="Quantize model weights")

    # Tensor parallelism
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree")

    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype) if args.dtype else None

    print("=" * 60)
    print("  nanoSGLang — Minimal LLM Serving Engine")
    print("=" * 60)
    print(f"  Model:       {args.model_path}")
    print(f"  Device:      {args.device}")
    print(f"  Mode:        {'naive' if args.naive else 'continuous batching'}")
    if args.quantize:
        print(f"  Quantize:    {args.quantize}")
    if args.tp > 1:
        print(f"  TP degree:   {args.tp}")
    print()

    # Tensor parallelism init
    if args.tp > 1:
        from nano_sglang.distributed.tensor_parallel import init_distributed
        rank, world_size = init_distributed()
        args.device = f"cuda:{rank}"
        print(f"  [Rank {rank}/{world_size}] initialized")

    # Initialize engine
    engine = InferenceEngine(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch_tokens=args.max_batch_tokens,
        max_running_requests=args.max_running_requests,
        prefill_chunk_size=args.prefill_chunk_size,
        naive=args.naive,
    )

    # Apply quantization
    if args.quantize:
        from nano_sglang.decode.quantization import quantize_model
        print(f"\nApplying {args.quantize} quantization...")
        quantize_model(engine.model, method=args.quantize)

    # Apply tensor parallelism
    if args.tp > 1:
        from nano_sglang.distributed.tensor_parallel import tensor_parallel_model
        tensor_parallel_model(engine.model, args.tp, rank)

    # Inject into API
    api.engine = engine

    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/completions")
    print(f"  GET  /v1/models")
    print(f"  GET  /health")
    print(f"  GET  /metrics")
    print()

    uvicorn.run(api.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
