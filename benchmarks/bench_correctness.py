"""
Experiment 1 — Correctness / output-parity benchmark.

Runs a small set of fixed prompts through both nanoSGLang and sglang in
greedy mode (temperature=0, deterministic) and compares:
  - Output token ids (exact match preferred)
  - Output text (after tokenizer decode)
  - First-token id, longest-common-prefix length
  - Per-prompt generation latency

Purpose: verify nanoSGLang's model forward + sampling path produces the
same tokens as sglang for the same model / same input. Differences point
at bugs in attention, RoPE, weight loading, or sampling.

Usage:
    python benchmarks/bench_correctness.py \
        --model-path /path/to/Qwen2.5-0.5B-Instruct \
        --max-tokens 32
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Write a Python function that returns the sum of a list:\n",
    "Question: What is 2 + 2?\nAnswer:",
    "Once upon a time, in a small village,",
    "The three laws of robotics are:\n1.",
]


def run_nano_sglang(model_path: str, prompt_token_ids: list[list[int]], max_tokens: int):
    from benchmarks.nano_llm import NanoLLM, make_sampling_params

    print("\n[nano-sglang] initializing...")
    llm = NanoLLM(
        model_path=model_path,
        max_seq_len=2048,
        num_blocks=2000,
        block_size=16,
        max_batch_tokens=4096,
        max_running_requests=16,
        prefill_chunk_size=1024,
    )

    sps = [make_sampling_params(temperature=0.0, max_tokens=max_tokens) for _ in prompt_token_ids]

    # Warmup
    llm.generate([prompt_token_ids[0]], [sps[0]])

    t0 = time.perf_counter()
    results = llm.generate(prompt_token_ids, sps)
    elapsed = time.perf_counter() - t0

    outputs = []
    for r in results:
        outputs.append({
            "output_ids": r.output_token_ids,
            "text": r.text,
            "ttft": r.ttft,
            "e2e": r.e2e_latency,
        })
    llm.shutdown()
    return outputs, elapsed


def run_sglang(model_path: str, prompt_token_ids: list[list[int]], max_tokens: int):
    import sglang as sgl

    print("\n[sglang] initializing...")
    engine = sgl.Engine(
        model_path=model_path,
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        mem_fraction_static=0.5,
        random_seed=0,
        log_level="warning",
        skip_server_warmup=True,
    )

    sp = {"temperature": 0.0, "max_new_tokens": max_tokens}

    # Warmup
    engine.generate(input_ids=prompt_token_ids[0], sampling_params=sp)

    t0 = time.perf_counter()
    raw_results = engine.generate(
        input_ids=prompt_token_ids,
        sampling_params=[sp] * len(prompt_token_ids),
    )
    elapsed = time.perf_counter() - t0

    outputs = []
    for r in raw_results:
        meta = r["meta_info"]
        outputs.append({
            "output_ids": r["output_ids"],
            "text": r["text"],
            "ttft": None,
            "e2e": meta.get("e2e_latency"),
        })
    engine.shutdown()
    return outputs, elapsed


def common_prefix_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--system", type=str, default="both",
                        choices=["nano", "sglang", "both"])
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_ids = [tok.encode(p, add_special_tokens=True) for p in PROMPTS]

    print("=" * 70)
    print("  Correctness Benchmark (greedy, temperature=0)")
    print("=" * 70)
    print(f"  Model:      {args.model_path}")
    print(f"  Prompts:    {len(PROMPTS)}")
    print(f"  Max tokens: {args.max_tokens}")

    nano_outs = sgl_outs = None
    if args.system in ("nano", "both"):
        nano_outs, nano_time = run_nano_sglang(args.model_path, prompt_ids, args.max_tokens)
        print(f"\n  [nano-sglang] total: {nano_time:.2f}s")
    if args.system in ("sglang", "both"):
        sgl_outs, sgl_time = run_sglang(args.model_path, prompt_ids, args.max_tokens)
        print(f"  [sglang]      total: {sgl_time:.2f}s")

    if nano_outs and sgl_outs:
        print("\n" + "=" * 70)
        print("  Per-prompt comparison")
        print("=" * 70)
        exact_matches = 0
        for i, (prompt, n, s) in enumerate(zip(PROMPTS, nano_outs, sgl_outs)):
            cp = common_prefix_len(n["output_ids"], s["output_ids"])
            match = n["output_ids"] == s["output_ids"]
            if match:
                exact_matches += 1
            status = "EXACT" if match else f"diverge@{cp}/{min(len(n['output_ids']), len(s['output_ids']))}"
            print(f"\n  [{i}] prompt: {prompt!r}")
            print(f"      status: {status}")
            print(f"      nano: {n['text']!r}")
            print(f"      sgl : {s['text']!r}")
        print(f"\n  Exact matches: {exact_matches}/{len(PROMPTS)}")

    if nano_outs and not sgl_outs:
        print("\n  [nano-sglang] outputs:")
        for i, (p, o) in enumerate(zip(PROMPTS, nano_outs)):
            print(f"  [{i}] {p!r} -> {o['text']!r}")
    if sgl_outs and not nano_outs:
        print("\n  [sglang] outputs:")
        for i, (p, o) in enumerate(zip(PROMPTS, sgl_outs)):
            print(f"  [{i}] {p!r} -> {o['text']!r}")


if __name__ == "__main__":
    main()
