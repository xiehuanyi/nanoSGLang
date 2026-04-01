"""
OpenAI-compatible HTTP API with metrics and graceful shutdown.

Endpoints:
  POST /v1/chat/completions  — chat interface (streaming + non-streaming)
  POST /v1/completions       — raw completion interface
  GET  /v1/models            — list loaded model
  GET  /health               — health check
  GET  /metrics              — performance metrics (TTFT, TBT, throughput)
"""

import json
import signal
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from nano_sglang.engine.engine import InferenceEngine, TokenOutput
from nano_sglang.engine.request import SamplingParams
from nano_sglang.server.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    stop: Optional[list[str]] = None


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    stop: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

engine: Optional[InferenceEngine] = None
metrics_collector: Optional[MetricsCollector] = None
_shutting_down = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global metrics_collector
    metrics_collector = MetricsCollector()

    # Start engine loop if using continuous batching
    if engine is not None and not engine.naive:
        await engine.start()
        print("Engine loop started (continuous batching mode)")

    yield

    # Graceful shutdown
    global _shutting_down
    _shutting_down = True
    print("\nShutting down...")

    if engine is not None and not engine.naive:
        await engine.stop()
    print("Engine stopped. Goodbye!")


app = FastAPI(title="nanoSGLang", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    if _shutting_down:
        raise HTTPException(503, "Server is shutting down")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_tokens = engine.tokenizer.apply_chat_template(messages)

    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    request_id = _make_id()
    model_name = req.model
    created = int(time.time())

    # Track metrics
    req_metrics = None
    if metrics_collector:
        req_metrics = metrics_collector.on_request_arrival(request_id, len(prompt_tokens))

    if req.stream:
        return EventSourceResponse(
            _stream_chat(prompt_tokens, sampling, request_id, model_name, created, req_metrics),
            media_type="text/event-stream",
        )
    else:
        output_text = ""
        completion_tokens = 0

        async for out in engine.generate_stream(prompt_tokens, sampling):
            if req_metrics and metrics_collector:
                metrics_collector.on_token_generated(req_metrics)
            if not out.finished or out.text:
                output_text += out.text
                completion_tokens += 1

        if req_metrics and metrics_collector:
            metrics_collector.on_request_finished(req_metrics)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_tokens) + completion_tokens,
            },
        }


async def _stream_chat(prompt_tokens, sampling, request_id, model_name, created, req_metrics):
    if metrics_collector:
        metrics_collector.on_request_started()

    async for out in engine.generate_stream(prompt_tokens, sampling):
        if req_metrics and metrics_collector:
            metrics_collector.on_token_generated(req_metrics)

        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": out.text} if not out.finished else {},
                "finish_reason": "stop" if out.finished else None,
            }],
        }
        yield json.dumps(chunk)

    if req_metrics and metrics_collector:
        metrics_collector.on_request_finished(req_metrics)

    yield "[DONE]"


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    if _shutting_down:
        raise HTTPException(503, "Server is shutting down")

    prompt_tokens = engine.tokenizer.encode(req.prompt)

    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    request_id = _make_id()
    model_name = req.model
    created = int(time.time())

    req_metrics = None
    if metrics_collector:
        req_metrics = metrics_collector.on_request_arrival(request_id, len(prompt_tokens))

    if req.stream:
        return EventSourceResponse(
            _stream_completion(prompt_tokens, sampling, request_id, model_name, created, req_metrics),
            media_type="text/event-stream",
        )
    else:
        output_text = ""
        completion_tokens = 0

        async for out in engine.generate_stream(prompt_tokens, sampling):
            if req_metrics and metrics_collector:
                metrics_collector.on_token_generated(req_metrics)
            if not out.finished or out.text:
                output_text += out.text
                completion_tokens += 1

        if req_metrics and metrics_collector:
            metrics_collector.on_request_finished(req_metrics)

        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": output_text,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_tokens) + completion_tokens,
            },
        }


async def _stream_completion(prompt_tokens, sampling, request_id, model_name, created, req_metrics):
    if metrics_collector:
        metrics_collector.on_request_started()

    async for out in engine.generate_stream(prompt_tokens, sampling):
        if req_metrics and metrics_collector:
            metrics_collector.on_token_generated(req_metrics)

        chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": out.text,
                "finish_reason": "stop" if out.finished else None,
            }],
        }
        yield json.dumps(chunk)

    if req_metrics and metrics_collector:
        metrics_collector.on_request_finished(req_metrics)

    yield "[DONE]"


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models():
    model_id = "nano-sglang"
    if engine is not None:
        model_id = engine.config.model_type
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "nanoSGLang"}],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if not _shutting_down else "shutting_down",
        "engine_ready": engine is not None,
    }


@app.get("/metrics")
async def get_metrics():
    if metrics_collector is None:
        raise HTTPException(503, "Metrics not initialized")
    stats = metrics_collector.get_stats()

    # Add engine-specific stats
    if engine is not None and not engine.naive:
        stats["kv_cache_free_blocks"] = engine.block_manager.num_free_blocks
        stats["kv_cache_used_blocks"] = engine.block_manager.num_used_blocks
        stats["kv_cache_total_blocks"] = engine.block_manager.num_blocks

    return stats
