# ⚡ mini-vLLM — Custom LLM Inference Engine

> A lightweight LLM inference server built from scratch in Python — custom token generation loop, continuous batching scheduler, FastAPI SSE streaming, and a React live metrics dashboard. No `model.generate()` used anywhere.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

---

## What is this?

The biggest unsolved problem in production AI is not model quality — it is serving models efficiently without burning through GPU memory.

This project builds a miniature version of [vLLM](https://github.com/vllm-project/vllm) from scratch to understand exactly how production LLM inference engines work at the systems level. Every layer is hand-written: the token loop, the batching scheduler, the streaming API, and the live dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    docker-compose stack                  │
│                                                          │
│  ┌──────────────────┐         ┌───────────────────────┐ │
│  │  React Dashboard  │──REST──▶│     FastAPI :8000     │ │
│  │   localhost:3000  │◀──SSE──│                       │ │
│  │                  │         │  POST /generate        │ │
│  │  • Chat window   │         │  GET  /stats           │ │
│  │  • GPU gauge     │         │  WS   /ws/stats        │ │
│  │  • TPS sparkline │         │  GET  /health          │ │
│  │  • Batch slots   │         └──────────┬────────────┘ │
│  └──────────────────┘                    │ asyncio queue │
│                                          ▼               │
│                               ┌───────────────────────┐ │
│                               │   Inference Engine    │ │
│                               │                       │ │
│                               │  • Scheduler loop     │ │
│                               │  • Continuous batch   │ │
│                               │  • Custom gen loop    │ │
│                               │  • Token sampler      │ │
│                               └──────────┬────────────┘ │
│                                          │               │
│                               ┌──────────▼────────────┐ │
│                               │   GPT-2 / TinyLlama   │ │
│                               │   HuggingFace weights  │ │
│                               └───────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

### Phase 1 — Custom generation loop
Wrote every step of token generation manually in PyTorch. No `model.generate()` anywhere.

```
Input prompt → Tokenise → Forward pass → Logits[last position]
→ Temperature scaling → Top-p nucleus filter → Multinomial sample
→ Decode token → Append → Repeat until EOS or max_tokens
```

### Phase 2 — Continuous batching
The technique that makes production LLM serving 3–5x more efficient than naive batching.

```
Batch slots:  [req-A ▶] [req-B ▶] [req-C ⟨EOS⟩] [idle]
                                        ↓ immediately
                                [req-D from queue fills slot]
Batch slots:  [req-A ▶] [req-B ▶] [req-D ▶]      [idle]
```

When any sequence hits `<EOS>`, its slot is immediately freed and the next queued request fills it — without pausing the other active sequences.

### Phase 3 — FastAPI + Server-Sent Events
Tokens stream to the client one by one as they are generated, exactly like ChatGPT.

```
POST /generate  →  text/event-stream
data: {"type":"start","request_id":"a3f9..."}
data: {"type":"token","text":" a"}
data: {"type":"token","text":" growing"}
...
data: {"type":"done"}
```

### Phase 4 — React developer dashboard
Live metrics pulled from the engine every second:
- GPU memory gauge (circular, animated)
- Tokens per second sparkline
- Queue depth counter
- Active batch slot visualizer (8 slots, green when active)
- Total tokens generated
- Engine uptime

### Phase 5 — Docker Compose deployment
One command brings up the full stack — CUDA backend + Nginx frontend.

---

## File Structure

```
mini-inference-engine/
├── quickstart.py                    ← Phase 1 smoke test — run this first
├── docker-compose.yml               ← Phase 5 full stack
├── README.md
│
├── backend/
│   ├── main.py                      ← Phase 3 FastAPI server
│   ├── requirements.txt
│   ├── Dockerfile                   ← CUDA Ubuntu base image
│   └── engine/
│       ├── __init__.py
│       └── inference_engine.py      ← Phase 1 + 2 core engine
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── Dockerfile
    └── src/
        ├── main.jsx
        ├── App.jsx                  ← Phase 4 live dashboard
        └── index.css
```

---

## Getting Started

### Prerequisites

```bash
python3 --version    # 3.10+
node --version       # v18+
docker --version     # 20+
nvidia-smi           # optional — works on CPU too
```

### Step 1 — Smoke test (no server needed)

```bash
pip install torch transformers psutil
python quickstart.py
```

Expected output:
```
Device: cuda
Prompt: The future of AI inference is
Output:  a growing field of research that...
Generated 60 tokens.
```

### Step 2 — Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
[Engine] Loading 'gpt2' on cuda...
[Engine] Loaded. Parameters: 124,439,808
[Engine] Scheduler started.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Test the API:

```bash
# Health check
curl http://localhost:8000/health

# Live stats
curl http://localhost:8000/stats

# Stream a generation
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Continuous batching works by", "max_new_tokens": 80}'
```

Windows PowerShell:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
Invoke-RestMethod -Uri "http://localhost:8000/stats"
```

### Step 3 — Run the dashboard

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

### Step 4 — Full Docker stack

```bash
# GPU (requires nvidia-container-toolkit)
docker-compose up --build

# CPU only — remove the deploy.resources block in docker-compose.yml first
docker-compose up --build
```

- Frontend → `http://localhost:3000`
- API docs → `http://localhost:8000/docs`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/stats` | Live engine metrics as JSON |
| `POST` | `/generate` | Submit prompt — returns SSE token stream |
| `WS` | `/ws/stats` | WebSocket push of stats every 1 second |

### POST /generate

Request:
```json
{
  "prompt": "The future of AI inference is",
  "max_new_tokens": 200,
  "temperature": 0.8,
  "top_p": 0.9
}
```

Response (Server-Sent Events):
```
data: {"type": "start", "request_id": "a3f9-..."}
data: {"type": "token", "text": " a"}
data: {"type": "token", "text": " growing"}
data: {"type": "done"}
```

### GET /stats

```json
{
  "active_requests": 4,
  "queue_depth": 2,
  "tokens_per_second": 38.6,
  "batch_size": 4,
  "gpu_memory_used_mb": 487.2,
  "gpu_memory_total_mb": 8192.0,
  "gpu_memory_pct": 5.9,
  "total_tokens_generated": 1240,
  "uptime_seconds": 134.5
}
```

---

## Switching to TinyLlama

For better output quality on a GPU with 4 GB+ VRAM, change one line in `backend/main.py`:

```python
# Before
ENGINE = InferenceEngine(model_name="gpt2")

# After
ENGINE = InferenceEngine(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

The first boot downloads ~2 GB of weights into the `hf_cache` Docker volume. Subsequent restarts are instant.

---

## How Continuous Batching Works

Standard batching waits for the entire batch to finish before admitting new requests. GPU sits idle whenever any sequence finishes early.

Continuous batching solves this:

```
Standard:   [A████████] [B██] idle [C███████]   ← GPU wastes time waiting
Continuous: [A████████]
            [B██][D fills immediately ████████]  ← zero idle time
            [C███████]
```

The scheduler in `inference_engine.py → run()` checks for finished sequences after every step and immediately replaces them from the waiting queue. Active sequences never pause.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: engine` | Run `uvicorn` from inside `backend/` not the root |
| `CUDA out of memory` | Remove `torch_dtype=torch.float16` to fall back to CPU |
| Connection error in browser | Backend is not running — check your uvicorn terminal |
| `npm install` fails | Install Node 18+ via `nvm install 18` |
| Docker GPU error | Install `nvidia-container-toolkit` then restart Docker |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | GPT-2 / TinyLlama via HuggingFace Transformers |
| Inference | PyTorch 2.3 — hand-written generation loop |
| API | FastAPI 0.111 + uvicorn |
| Streaming | Server-Sent Events + WebSocket |
| Frontend | React 18 + Vite |
| Deployment | Docker + Docker Compose + Nginx |
| GPU | CUDA 12.1 (CPU fallback supported) |

---

## Key Insight

> Training a model is a one-time cost. Serving it runs 24/7 forever. Inference efficiency is the real engineering challenge.

GPU memory is the bottleneck — not compute. Every design decision in vLLM, TGI, and TensorRT-LLM exists to maximize memory utilization. Continuous batching is the core technique that makes production LLM serving economically viable.

---

## License

MIT — use it, learn from it, build on it.
