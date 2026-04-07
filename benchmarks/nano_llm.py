"""
Offline batch wrapper around nanoSGLang's async InferenceEngine.

Exposes a synchronous `LLM.generate(prompt_token_ids, sampling_params)` API
modeled after nanovllm / vllm / sglang.Engine offline mode, so the same
benchmark harness can drive nanoSGLang and sglang.

Engine runs on a persistent background-thread event loop; all asyncio
primitives (Locks, Queues) are instantiated inside that loop's context to
avoid cross-loop binding issues.
"""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Optional

from nano_sglang.engine.engine import InferenceEngine
from nano_sglang.engine.request import SamplingParams as NanoSamplingParams


@dataclass
class GenerateResult:
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    text: str = ""
    ttft: Optional[float] = None      # seconds
    e2e_latency: Optional[float] = None  # seconds
    num_tokens: int = 0


class NanoLLM:
    """Synchronous offline batch API over nanoSGLang's async engine."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype=None,
        max_seq_len: int = 4096,
        num_blocks: int = 8000,
        block_size: int = 16,
        max_batch_tokens: int = 8192,
        max_running_requests: int = 64,
        prefill_chunk_size: int = 2048,
        naive: bool = False,
    ):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="nano-llm-loop", daemon=True
        )
        self._thread.start()

        async def _build():
            return InferenceEngine(
                model_path=model_path,
                device=device,
                dtype=dtype,
                max_seq_len=max_seq_len,
                num_blocks=num_blocks,
                block_size=block_size,
                max_batch_tokens=max_batch_tokens,
                max_running_requests=max_running_requests,
                prefill_chunk_size=prefill_chunk_size,
                naive=naive,
            )

        self.engine: InferenceEngine = self._submit(_build()).result()
        self._submit(self.engine.start()).result()

    # ------------------------------------------------------------------
    # Loop plumbing
    # ------------------------------------------------------------------
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: list[NanoSamplingParams],
    ) -> list[GenerateResult]:
        """Run all prompts concurrently and return per-request results."""
        if len(prompt_token_ids) != len(sampling_params):
            raise ValueError("prompt_token_ids and sampling_params length mismatch")

        async def _run_all():
            async def _one(pt: list[int], sp: NanoSamplingParams) -> GenerateResult:
                import time
                t0 = time.perf_counter()
                req = await self.engine.add_request(pt, sp)
                tokens: list[int] = []
                ttft = None
                async for out in self._iter_request(req):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens.append(out.token_id)
                    if out.finished:
                        break
                e2e = time.perf_counter() - t0
                return GenerateResult(
                    prompt_token_ids=pt,
                    output_token_ids=tokens,
                    ttft=ttft,
                    e2e_latency=e2e,
                    num_tokens=len(tokens),
                )

            return await asyncio.gather(
                *[_one(pt, sp) for pt, sp in zip(prompt_token_ids, sampling_params)]
            )

        results: list[GenerateResult] = self._submit(_run_all()).result()
        # Decode texts on caller side (tokenizer is thread-safe for decode)
        for r in results:
            r.text = self.tokenizer.decode(r.output_token_ids)
        return results

    async def _iter_request(self, req):
        """Yield TokenOutput items from an already-submitted Request."""
        while True:
            out = await req.output_queue.get()
            yield out
            if out.finished:
                break

    def shutdown(self):
        try:
            self._submit(self.engine.stop()).result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def make_sampling_params(
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_tokens: int = 64,
    ignore_eos: bool = False,
) -> NanoSamplingParams:
    return NanoSamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        ignore_eos=ignore_eos,
    )
