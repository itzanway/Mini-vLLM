"""
Phase 3: FastAPI API Layer
Endpoints:
  POST /generate          → SSE stream of tokens
  GET  /stats             → live engine metrics (polled by dashboard)
  GET  /health            → liveness probe
  WS   /ws/stats          → WebSocket stream of stats (alternative to polling)
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.inference_engine import InferenceEngine

# ─────────────────────────────────────────────
# App Lifecycle
# ─────────────────────────────────────────────

ENGINE: InferenceEngine = None
ENGINE_TASK: asyncio.Task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE, ENGINE_TASK
    model_name = "gpt2"          # swap to "TinyLlama/TinyLlama-1.1B-Chat-v1.0" on GPU
    ENGINE = InferenceEngine(model_name=model_name)
    ENGINE.load_model()
    ENGINE_TASK = asyncio.create_task(ENGINE.run())
    print("[API] Engine scheduler running.")
    yield
    await ENGINE.shutdown()
    ENGINE_TASK.cancel()
    print("[API] Engine shut down.")


app = FastAPI(
    title="Mini-vLLM Inference API",
    description="Lightweight LLM inference server with continuous batching",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(200, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.01, le=1.0)


# ─────────────────────────────────────────────
# SSE Token Stream
# ─────────────────────────────────────────────

async def token_stream(req_id: str, token_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """
    Converts an asyncio.Queue of tokens into Server-Sent Events.
    Sentinel value None signals end-of-stream.
    """
    request_id_event = f"data: {json.dumps({'type': 'start', 'request_id': req_id})}\n\n"
    yield request_id_event

    while True:
        token = await token_queue.get()
        if token is None:
            # Stream finished
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            break
        payload = json.dumps({"type": "token", "text": token})
        yield f"data: {payload}\n\n"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": ENGINE.model_name if ENGINE else None}


@app.post("/generate")
async def generate(body: GenerateRequest):
    """
    Accept a prompt and stream generated tokens back via SSE.
    The client should consume the event stream until it receives type=done.
    """
    req = await ENGINE.submit(
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )
    return StreamingResponse(
        token_stream(req.id, req.token_queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",          # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/stats")
async def stats():
    """Return live engine metrics as JSON (polled every second by dashboard)."""
    s = await ENGINE.get_stats()
    return {
        "active_requests": s.active_requests,
        "queue_depth": s.queue_depth,
        "tokens_per_second": s.tokens_per_second,
        "batch_size": s.batch_size,
        "gpu_memory_used_mb": round(s.gpu_memory_used_mb, 1),
        "gpu_memory_total_mb": round(s.gpu_memory_total_mb, 1),
        "gpu_memory_pct": round(
            100 * s.gpu_memory_used_mb / max(s.gpu_memory_total_mb, 1), 1
        ),
        "total_tokens_generated": s.total_tokens_generated,
        "uptime_seconds": round(s.uptime_seconds, 1),
    }


@app.websocket("/ws/stats")
async def ws_stats(websocket: WebSocket):
    """
    WebSocket alternative to polling /stats.
    Pushes a stats JSON blob every second while connected.
    """
    await websocket.accept()
    try:
        while True:
            s = await ENGINE.get_stats()
            await websocket.send_json({
                "active_requests": s.active_requests,
                "queue_depth": s.queue_depth,
                "tokens_per_second": s.tokens_per_second,
                "batch_size": s.batch_size,
                "gpu_memory_used_mb": round(s.gpu_memory_used_mb, 1),
                "gpu_memory_total_mb": round(s.gpu_memory_total_mb, 1),
                "gpu_memory_pct": round(
                    100 * s.gpu_memory_used_mb / max(s.gpu_memory_total_mb, 1), 1
                ),
                "total_tokens_generated": s.total_tokens_generated,
                "uptime_seconds": round(s.uptime_seconds, 1),
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass