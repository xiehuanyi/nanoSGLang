"""
Experiment 2 — Throughput benchmark (nanovllm-style).

Generates a fixed workload of random prompt_token_ids + random max_tokens
and runs it through either nanoSGLang or sglang. Reports:
  - Total generated tokens
  - Wall-clock time
  - Throughput (tok/s, req/s)
  - TTFT / end-to-end latency percentiles (p50/p90/p99)
  - Peak GPU memory

Workload follows the reference script in the task:
    num_seqs = 256, prompt_len ~ U[100, 1024], output_len ~ U[100, 1024]
    temperature=0.6, ignore_eos=True

Usage:
    python benchmarks/bench_throughput.py --system nano   --model-path ...
    python benchmarks/bench_throughput.py --system sglang --model-path ...
"""

import argparse
import json
import os
import statistics
import sys
import time
from random import randint, seed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_workload(num_seqs: int, min_in: int, max_in: int,
                   min_out: int, max_out: int, vocab_top: int = 10000):
    seed(0)
    prompts = [
        [randint(0, vocab_top) for _ in range(randint(min_in, max_in))]
        for _ in range(num_seqs)
    ]
    out_lens = [randint(min_out, max_out) for _ in range(num_seqs)]
    return prompts, out_lens


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = max(0, min(len(xs_sorted) - 1, int(round(p * (len(xs_sorted) - 1)))))
    return xs_sorted[k]


def run_nano(model_path: str, prompts, out_lens, args):
    from benchmarks.nano_llm import NanoLLM, make_sampling_params

    print("[nano-sglang] initializing...")
    llm = NanoLLM(
        model_path=model_path,
        max_seq_len=args.max_model_len,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch_tokens=args.max_batch_tokens,
        max_running_requests=args.max_running,
        prefill_chunk_size=args.prefill_chunk_size,
    )

    sps = [
        make_sampling_params(
            temperature=args.temperature, max_tokens=n, ignore_eos=True
        )
        for n in out_lens
    ]

    # Warmup
    print("[nano-sglang] warmup...")
    warm_prompt = [1, 2, 3, 4, 5]
    warm_sp = make_sampling_params(temperature=0.0, max_tokens=8)
    llm.generate([warm_prompt], [warm_sp])

    torch.cuda.reset_peak_memory_stats()
    print(f"[nano-sglang] generating {len(prompts)} requests...")
    t0 = time.perf_counter()
    results = llm.generate(prompts, sps)
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    ttfts = [r.ttft for r in results if r.ttft is not None]
    e2es = [r.e2e_latency for r in results if r.e2e_latency is not None]
    total_out = sum(r.num_tokens for r in results)
    total_in = sum(len(p) for p in prompts)
    planned = sum(out_lens)

    llm.shutdown()

    return {
        "system": "nano-sglang",
        "elapsed_s": elapsed,
        "num_requests": len(prompts),
        "planned_output_tokens": planned,
        "generated_output_tokens": total_out,
        "prompt_tokens": total_in,
        "throughput_out_tok_per_s": total_out / elapsed,
        "throughput_total_tok_per_s": (total_out + total_in) / elapsed,
        "req_per_s": len(prompts) / elapsed,
        "ttft_ms": {
            "avg": statistics.mean(ttfts) * 1000 if ttfts else None,
            "p50": percentile(ttfts, 0.5) * 1000 if ttfts else None,
            "p90": percentile(ttfts, 0.9) * 1000 if ttfts else None,
            "p99": percentile(ttfts, 0.99) * 1000 if ttfts else None,
        },
        "e2e_ms": {
            "avg": statistics.mean(e2es) * 1000 if e2es else None,
            "p50": percentile(e2es, 0.5) * 1000 if e2es else None,
            "p90": percentile(e2es, 0.9) * 1000 if e2es else None,
            "p99": percentile(e2es, 0.99) * 1000 if e2es else None,
        },
        "peak_gpu_gib": peak_mem,
    }


def run_sglang(model_path: str, prompts, out_lens, args):
    import sglang as sgl

    print("[sglang] initializing...")
    engine = sgl.Engine(
        model_path=model_path,
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_cuda_graph=args.disable_cuda_graph,
        mem_fraction_static=args.sgl_mem_fraction,
        max_running_requests=args.max_running,
        chunked_prefill_size=args.prefill_chunk_size,
        random_seed=0,
        log_level="warning",
        skip_server_warmup=True,
    )

    sps = [
        {
            "temperature": args.temperature,
            "max_new_tokens": n,
            "ignore_eos": True,
        }
        for n in out_lens
    ]

    # Warmup
    print("[sglang] warmup...")
    engine.generate(input_ids=[1, 2, 3, 4, 5],
                    sampling_params={"temperature": 0.0, "max_new_tokens": 8})

    torch.cuda.reset_peak_memory_stats()
    print(f"[sglang] generating {len(prompts)} requests...")
    t0 = time.perf_counter()
    results = engine.generate(input_ids=prompts, sampling_params=sps)
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    e2es = [r["meta_info"].get("e2e_latency") for r in results
            if r.get("meta_info", {}).get("e2e_latency") is not None]
    total_out = sum(r["meta_info"].get("completion_tokens", 0) for r in results)
    total_in = sum(r["meta_info"].get("prompt_tokens", 0) for r in results)
    planned = sum(out_lens)

    engine.shutdown()

    return {
        "system": "sglang",
        "elapsed_s": elapsed,
        "num_requests": len(prompts),
        "planned_output_tokens": planned,
        "generated_output_tokens": total_out,
        "prompt_tokens": total_in,
        "throughput_out_tok_per_s": total_out / elapsed,
        "throughput_total_tok_per_s": (total_out + total_in) / elapsed,
        "req_per_s": len(prompts) / elapsed,
        "ttft_ms": {"avg": None, "p50": None, "p90": None, "p99": None},
        "e2e_ms": {
            "avg": statistics.mean(e2es) * 1000 if e2es else None,
            "p50": percentile(e2es, 0.5) * 1000 if e2es else None,
            "p90": percentile(e2es, 0.9) * 1000 if e2es else None,
            "p99": percentile(e2es, 0.99) * 1000 if e2es else None,
        },
        "peak_gpu_gib": peak_mem,
    }


def print_report(r: dict):
    print("\n" + "=" * 66)
    print(f"  {r['system']}")
    print("=" * 66)
    print(f"  requests:              {r['num_requests']}")
    print(f"  prompt tokens (total): {r['prompt_tokens']}")
    print(f"  output tokens (plan):  {r['planned_output_tokens']}")
    print(f"  output tokens (got):   {r['generated_output_tokens']}")
    print(f"  wall-clock:            {r['elapsed_s']:.2f} s")
    print(f"  throughput (out):      {r['throughput_out_tok_per_s']:.1f} tok/s")
    print(f"  throughput (in+out):   {r['throughput_total_tok_per_s']:.1f} tok/s")
    print(f"  requests/s:            {r['req_per_s']:.2f}")
    if r["ttft_ms"]["avg"] is not None:
        t = r["ttft_ms"]
        print(f"  TTFT ms: avg={t['avg']:.0f} p50={t['p50']:.0f} "
              f"p90={t['p90']:.0f} p99={t['p99']:.0f}")
    if r["e2e_ms"]["avg"] is not None:
        e = r["e2e_ms"]
        print(f"  E2E  ms: avg={e['avg']:.0f} p50={e['p50']:.0f} "
              f"p90={e['p90']:.0f} p99={e['p99']:.0f}")
    print(f"  peak GPU mem:          {r['peak_gpu_gib']:.2f} GiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--system", type=str, required=True,
                        choices=["nano", "sglang"])
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--min-input", type=int, default=100)
    parser.add_argument("--max-input", type=int, default=1024)
    parser.add_argument("--min-output", type=int, default=100)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=4096)

    # nano-sglang engine knobs
    parser.add_argument("--num-blocks", type=int, default=12000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-running", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)

    # sglang knobs
    parser.add_argument("--sgl-mem-fraction", type=float, default=0.85)
    parser.add_argument("--disable-cuda-graph", action="store_true", default=True)

    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    prompts, out_lens = build_workload(
        args.num_seqs, args.min_input, args.max_input,
        args.min_output, args.max_output,
    )
    print(f"Workload: {len(prompts)} seqs, "
          f"prompt_len=[{args.min_input},{args.max_input}], "
          f"out_len=[{args.min_output},{args.max_output}], "
          f"temperature={args.temperature}")
    print(f"Total prompt tokens: {sum(len(p) for p in prompts)}, "
          f"total output tokens (planned): {sum(out_lens)}")

    if args.system == "nano":
        r = run_nano(args.model_path, prompts, out_lens, args)
    else:
        r = run_sglang(args.model_path, prompts, out_lens, args)

    print_report(r)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
