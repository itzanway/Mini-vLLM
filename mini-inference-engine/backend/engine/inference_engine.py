"""
Phase 1 + 2: Core Inference Engine with Continuous Batching
Mini-vLLM — Custom generation loop, no model.generate() allowed.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Request:
    """Represents a single inference request in the system."""
    id: str
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 0.9
    input_ids: Optional[torch.Tensor] = None       # shape: (seq_len,)
    generated_ids: list = field(default_factory=list)
    finish_reason: Optional[str] = None            # "eos" | "length"
    created_at: float = field(default_factory=time.time)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    # SSE queue — consumers await tokens here
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class EngineStats:
    """Live statistics emitted by the engine."""
    active_requests: int = 0
    queue_depth: int = 0
    tokens_per_second: float = 0.0
    batch_size: int = 0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    total_tokens_generated: int = 0
    uptime_seconds: float = 0.0


# ─────────────────────────────────────────────
# Sampler Helpers
# ─────────────────────────────────────────────

def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling filter — masks logits below probability mass threshold."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens beyond the nucleus
    sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_indices_to_remove[0] = False            # always keep the top token
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float("-inf")
    return logits


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """
    Custom single-token sampler.
    1. Apply temperature scaling.
    2. Apply top-p nucleus filter.
    3. Sample from the resulting distribution.
    """
    logits = logits / max(temperature, 1e-8)
    logits = top_p_filter(logits.clone(), top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ─────────────────────────────────────────────
# Continuous-Batching Inference Engine
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Phase 1 + 2 combined:
    • Custom generation loop (no model.generate()).
    • Continuous batching — new requests join mid-flight; finished sequences
      are immediately swapped out without stalling the running batch.
    """

    MAX_BATCH_SIZE = 8          # concurrent sequences in one forward pass
    SCHEDULER_HZ   = 50         # how often the scheduler loop ticks (ms)

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None
        self.eos_token_id: int = 0

        # Queues & state
        self._waiting_queue: asyncio.Queue[Request] = asyncio.Queue()
        self._active: list[Request] = []           # currently in a batch
        self._lock = asyncio.Lock()

        # Stats
        self._stats = EngineStats()
        self._token_counter = 0
        self._token_window_start = time.time()
        self._started_at = time.time()
        self._running = False

    # ── Model Loading ──────────────────────────────────────────────────────

    def load_model(self):
        """Load model + tokenizer onto the target device."""
        print(f"[Engine] Loading '{self.model_name}' on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        # Handle missing pad token (common with GPT-2 family)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        print(f"[Engine] Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    # ── Public API ─────────────────────────────────────────────────────────

    async def submit(self, prompt: str, max_new_tokens: int = 200,
                     temperature: float = 0.8, top_p: float = 0.9) -> Request:
        """Enqueue a new request and return it (caller can stream from req.token_queue)."""
        req = Request(
            id=str(uuid.uuid4()),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Tokenise once here so the scheduler doesn't block
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        req.input_ids = input_ids
        await self._waiting_queue.put(req)
        return req

    async def get_stats(self) -> EngineStats:
        self._stats.queue_depth = self._waiting_queue.qsize()
        self._stats.active_requests = len(self._active)
        self._stats.uptime_seconds = time.time() - self._started_at
        self._refresh_gpu_stats()
        return self._stats

    # ── Scheduler / Engine Loop ────────────────────────────────────────────

    async def run(self):
        """
        Main continuous-batching scheduler loop.
        Runs forever as an asyncio task; call engine.run() once at startup.
        """
        self._running = True
        print("[Engine] Scheduler started.")
        loop_sleep = self.SCHEDULER_HZ / 1000

        while self._running:
            # 1. Fill available batch slots from the waiting queue
            async with self._lock:
                while (len(self._active) < self.MAX_BATCH_SIZE
                       and not self._waiting_queue.empty()):
                    req = await self._waiting_queue.get()
                    self._active.append(req)
                    print(f"[Engine] Admitted request {req.id[:8]} (batch={len(self._active)})")

            if not self._active:
                await asyncio.sleep(loop_sleep)
                continue

            # 2. Run ONE forward step across the entire active batch
            await asyncio.get_event_loop().run_in_executor(
                None, self._step_batch
            )

            # 3. Retire finished sequences (continuous batching swap-out)
            async with self._lock:
                still_active = []
                for req in self._active:
                    if req.finish_reason is not None:
                        await req.token_queue.put(None)   # sentinel → stream done
                        req.finished_at = time.time()
                        print(f"[Engine] Request {req.id[:8]} finished ({req.finish_reason}), "
                              f"{len(req.generated_ids)} tokens in "
                              f"{req.finished_at - req.created_at:.2f}s")
                    else:
                        still_active.append(req)
                self._active = still_active

            await asyncio.sleep(0)   # yield to event loop (allows SSE to flush)

    def _step_batch(self):
        """
        Phase 1 core: one manual forward pass for every active request.
        We deliberately avoid model.generate() — this is the custom loop.

        For simplicity, each request gets its own forward pass (no KV-cache
        padding across sequences of different lengths). A production system
        would pad + mask, but this keeps the code readable.
        """
        t0 = time.time()
        tokens_this_step = 0

        for req in list(self._active):
            if req.finish_reason is not None:
                continue

            # Build full sequence: prompt tokens + already-generated tokens
            all_ids = torch.cat([
                req.input_ids,
                torch.tensor(req.generated_ids, dtype=torch.long)
            ]).unsqueeze(0).to(self.device)        # shape: (1, seq_len)

            with torch.no_grad():
                outputs = self.model(input_ids=all_ids)

            # Logits for the last position only → (vocab_size,)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = sample_next_token(next_token_logits, req.temperature, req.top_p)

            req.generated_ids.append(next_token_id)
            tokens_this_step += 1
            self._token_counter += 1

            if req.first_token_at is None:
                req.first_token_at = time.time()

            # Decode just the new token and push it into the SSE queue
            token_text = self.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )
            req.token_queue.put_nowait(token_text)

            # Check stopping criteria
            if next_token_id == self.eos_token_id:
                req.finish_reason = "eos"
            elif len(req.generated_ids) >= req.max_new_tokens:
                req.finish_reason = "length"

        # Update tokens-per-second (rolling 2-second window)
        elapsed = time.time() - self._token_window_start
        if elapsed >= 2.0:
            self._stats.tokens_per_second = round(self._token_counter / elapsed, 1)
            self._stats.total_tokens_generated += self._token_counter
            self._token_counter = 0
            self._token_window_start = time.time()

        self._stats.batch_size = len(self._active)

    def _refresh_gpu_stats(self):
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self._stats.gpu_memory_total_mb = props.total_memory / 1024 / 1024
            self._stats.gpu_memory_used_mb = (
                torch.cuda.memory_allocated() / 1024 / 1024
            )
        else:
            # CPU fallback — report process RSS
            import psutil, os
            proc = psutil.Process(os.getpid())
            self._stats.gpu_memory_used_mb = proc.memory_info().rss / 1024 / 1024
            self._stats.gpu_memory_total_mb = psutil.virtual_memory().total / 1024 / 1024

    async def shutdown(self):
        self._running = False