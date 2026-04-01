"""
Benchmark tool — Load test nanoSGLang (or any OpenAI-compatible server).

Sends concurrent requests and measures:
  - TTFT (Time To First Token)
  - TBT (Time Between Tokens)
  - End-to-end latency
  - Throughput (tokens/second)
  - Request success rate

Usage:
    # Basic benchmark:
    python benchmark.py --url http://localhost:8000 --num-requests 10

    # High concurrency:
    python benchmark.py --url http://localhost:8000 --num-requests 100 --concurrency 16

    # Compare with SGLang/vLLM:
    python benchmark.py --url http://localhost:30000 --num-requests 50 --name sglang
    python benchmark.py --url http://localhost:8000 --num-requests 50 --name nano-sglang
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp


@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    completion_tokens: int = 0
    ttft: Optional[float] = None   # seconds
    total_time: Optional[float] = None
    token_times: list[float] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    @property
    def tbt_avg(self) -> Optional[float]:
        if len(self.token_times) < 2:
            return None
        deltas = [self.token_times[i] - self.token_times[i - 1]
                  for i in range(1, len(self.token_times))]
        return sum(deltas) / len(deltas)

    @property
    def throughput(self) -> Optional[float]:
        if self.total_time and self.total_time > 0:
            return self.completion_tokens / self.total_time
        return None


# Sample prompts of varying lengths
SAMPLE_PROMPTS = [
    "Hello, how are you today?",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Rust?",
    "Summarize the history of artificial intelligence in three paragraphs.",
    "Write a function in Python that finds the longest common subsequence of two strings.",
    "Explain how transformers work in machine learning, including attention mechanisms.",
    "What is the meaning of life? Give a philosophical perspective.",
    "Describe the process of photosynthesis step by step.",
    "Write a short story about a robot learning to paint.",
]

SAMPLE_MESSAGES = [
    [{"role": "user", "content": p}] for p in SAMPLE_PROMPTS
]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request_id: int,
    max_tokens: int,
    use_chat: bool,
    temperature: float,
) -> RequestResult:
    """Send a single streaming request and measure timing."""
    messages = SAMPLE_MESSAGES[request_id % len(SAMPLE_MESSAGES)]
    prompt = SAMPLE_PROMPTS[request_id % len(SAMPLE_PROMPTS)]

    result = RequestResult(
        request_id=request_id,
        prompt_tokens=len(prompt.split()) * 2,  # rough estimate
    )

    if use_chat:
        endpoint = f"{url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
    else:
        endpoint = f"{url}/v1/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

    start_time = time.time()
    first_token_received = False

    try:
        async with session.post(endpoint, json=payload) as resp:
            if resp.status != 200:
                result.success = False
                result.error = f"HTTP {resp.status}"
                return result

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                now = time.time()

                # Extract content
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                has_content = False
                if use_chat:
                    delta = choice.get("delta", {})
                    has_content = bool(delta.get("content"))
                else:
                    has_content = bool(choice.get("text"))

                if has_content:
                    result.token_times.append(now)
                    result.completion_tokens += 1

                    if not first_token_received:
                        result.ttft = now - start_time
                        first_token_received = True

                if choice.get("finish_reason") is not None:
                    break

        result.total_time = time.time() - start_time

    except Exception as e:
        result.success = False
        result.error = str(e)
        result.total_time = time.time() - start_time

    return result


async def run_benchmark(
    url: str,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    use_chat: bool,
    temperature: float,
) -> list[RequestResult]:
    """Run the benchmark with controlled concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(session, req_id):
        async with semaphore:
            return await send_request(session, url, req_id, max_tokens, use_chat, temperature)

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [bounded_request(session, i) for i in range(num_requests)]

        print(f"\nSending {num_requests} requests (concurrency={concurrency})...")
        start = time.time()

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start

    return list(results), total_time


def print_results(results: list[RequestResult], total_time: float, name: str):
    """Pretty-print benchmark results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n{'=' * 60}")
    print(f"  Benchmark Results: {name}")
    print(f"{'=' * 60}")
    print(f"  Total requests:    {len(results)}")
    print(f"  Successful:        {len(successful)}")
    print(f"  Failed:            {len(failed)}")
    print(f"  Total time:        {total_time:.2f}s")

    if not successful:
        print("  No successful requests!")
        return

    # TTFT
    ttft_values = [r.ttft for r in successful if r.ttft is not None]
    if ttft_values:
        ttft_sorted = sorted(ttft_values)
        print(f"\n  TTFT (Time To First Token):")
        print(f"    avg:  {sum(ttft_values)/len(ttft_values)*1000:.1f} ms")
        print(f"    p50:  {ttft_sorted[len(ttft_sorted)//2]*1000:.1f} ms")
        print(f"    p99:  {ttft_sorted[min(int(len(ttft_sorted)*0.99), len(ttft_sorted)-1)]*1000:.1f} ms")

    # TBT
    tbt_values = [r.tbt_avg for r in successful if r.tbt_avg is not None]
    if tbt_values:
        tbt_sorted = sorted(tbt_values)
        print(f"\n  TBT (Time Between Tokens):")
        print(f"    avg:  {sum(tbt_values)/len(tbt_values)*1000:.1f} ms")
        print(f"    p50:  {tbt_sorted[len(tbt_sorted)//2]*1000:.1f} ms")
        print(f"    p99:  {tbt_sorted[min(int(len(tbt_sorted)*0.99), len(tbt_sorted)-1)]*1000:.1f} ms")

    # Throughput
    total_gen_tokens = sum(r.completion_tokens for r in successful)
    total_prompt_tokens = sum(r.prompt_tokens for r in successful)
    e2e_times = [r.total_time for r in successful if r.total_time]

    print(f"\n  Throughput:")
    print(f"    Total generated tokens:  {total_gen_tokens}")
    print(f"    Overall (tok/s):         {total_gen_tokens / total_time:.1f}")
    print(f"    Requests/second:         {len(successful) / total_time:.2f}")

    if e2e_times:
        per_req_throughputs = [r.throughput for r in successful if r.throughput]
        if per_req_throughputs:
            print(f"    Per-request avg (tok/s): {sum(per_req_throughputs)/len(per_req_throughputs):.1f}")

    # Latency
    if e2e_times:
        e2e_sorted = sorted(e2e_times)
        print(f"\n  End-to-end Latency:")
        print(f"    avg:  {sum(e2e_times)/len(e2e_times)*1000:.1f} ms")
        print(f"    p50:  {e2e_sorted[len(e2e_sorted)//2]*1000:.1f} ms")
        print(f"    p99:  {e2e_sorted[min(int(len(e2e_sorted)*0.99), len(e2e_sorted)-1)]*1000:.1f} ms")

    print(f"\n{'=' * 60}\n")

    # Errors
    if failed:
        print(f"  Errors:")
        for r in failed[:5]:
            print(f"    Request {r.request_id}: {r.error}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")


def main():
    parser = argparse.ArgumentParser(description="nanoSGLang benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="Server URL")
    parser.add_argument("--num-requests", "-n", type=int, default=10,
                        help="Number of requests to send")
    parser.add_argument("--concurrency", "-c", type=int, default=4,
                        help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--endpoint", type=str, default="chat",
                        choices=["chat", "completion"],
                        help="Which endpoint to test")
    parser.add_argument("--name", type=str, default="nanoSGLang",
                        help="Name for this benchmark run")
    args = parser.parse_args()

    use_chat = args.endpoint == "chat"

    results, total_time = asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        use_chat=use_chat,
        temperature=args.temperature,
    ))

    print_results(results, total_time, args.name)


if __name__ == "__main__":
    main()
