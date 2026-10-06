# RouteMind 🧠
> **Adaptive LLM Gateway with Intelligent Cost-, Quality-, Latency-, and Reliability-Aware Model Routing**

RouteMind is an enterprise-grade LLM routing gateway designed with a **deterministic zero-cost barrier** (`FREE_ONLY=true`). It dynamically routes inference requests across free-tier model providers (OpenRouter free models, Google Gemini free tier, and local LLMs) using adaptive Bayesian contextual bandits and resiliency mechanisms (circuit breakers, exponential backoff, fallback cascades).

---

## ⚡ Quickstart

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

### 2. Run Test Suite
```bash
cd backend
PYTHONPATH=app pytest -v
```

### 3. Start Gateway Locally
```bash
cd backend
PYTHONPATH=app uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🛡️ Core Guarantees
- **Strict Free-Tier Invariant**: Non-free models are rejected with `HTTP 403 FreeTierViolationError`.
- **OpenAI-Compatible Ingress**: Direct drop-in replacement for OpenAI SDK clients (`POST /v1/chat/completions`).
- **Distributed Observability**: Structured JSON logging with correlation IDs (`X-Request-ID`), OpenTelemetry tracing, and Prometheus metrics.
# RouteMind
