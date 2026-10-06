# RouteMind — Architectural Decision Records (ADRs)

This document formalizes the engineering rationale and trade-offs behind every technology selection in RouteMind.

---

## ADR-001: Selection of FastAPI for API Gateway Ingress

### Context
The gateway sits directly on the request path for all client LLM completions, requiring sub-millisecond route dispatching, non-blocking asynchronous I/O for concurrent provider streams, and native OpenAPI generation.

### Decision
We selected **FastAPI** running on the **Uvicorn ASGI** server.

### Consequences & Trade-offs
- **Pros**: Native `async/await` support, built-in dependency injection for RBAC, automatic OpenAPI schema generation, low CPU and memory footprint.
- **Cons**: Requires careful handling of blocking synchronous code (mitigated by using `asyncio` and worker pools).
- **Alternatives Rejected**:
  - *Flask / Django*: Synchronous WSGI architecture introduces connection starvation under high concurrency streaming.
  - *Express / Node.js*: Lacks native scientific computing libraries needed for the Bayesian bandit routing engine.

---

## ADR-002: Selection of PostgreSQL for Long-Term Audit & Tenant Storage

### Context
RouteMind requires structured storage for multi-tenant policies, scoped API keys, and comprehensive audit logs of all routing decisions.

### Decision
We selected **PostgreSQL** with **SQLAlchemy 2.0 (Async)**.

### Consequences & Trade-offs
- **Pros**: ACID transactions guarantee tenant quota and API key consistency; native `JSONB` indexing allows fast querying of dynamic feature vectors.
- **Cons**: Requires managed relational infrastructure (mitigated by Docker Compose).
- **Alternatives Rejected**:
  - *MongoDB*: Lacks strict relational constraints between tenants, policies, and API keys.
  - *SQLite (production)*: Write lock contention makes it unsuitable for concurrent multi-tenant audit ingestion.

---

## ADR-003: Selection of Redis for Sliding-Window Rate Limiting & Circuit State

### Context
Rate limits and circuit breaker trip states must be synchronized across distributed gateway instances in real-time with sub-millisecond latency.

### Decision
We selected **Redis** utilizing atomic pipelines and sliding-window logs (`ZSET`).

### Consequences & Trade-offs
- **Pros**: Sub-millisecond latency ($<0.8\text{ms}$), atomic sliding-window arithmetic, built-in key expiration (`TTL`).
- **Cons**: In-memory storage requires memory management (mitigated by setting 48-hour TTLs on quota keys).
- **Fallback**: Implemented thread-safe local in-memory fallback cache if Redis becomes temporarily unreachable.

---

## ADR-004: Selection of Disjoint LinUCB for Dynamic Model Routing

### Context
Static routing rules cannot adapt to live upstream degradation (e.g. rate limits or latency spikes). Conversely, using an LLM to route every request adds 500ms+ latency and consumes unnecessary tokens.

### Decision
We implemented a **Disjoint Linear Upper Confidence Bound (LinUCB)** Bayesian Contextual Bandit algorithm operating on a deterministic 10D feature vector.

### Consequences & Trade-offs
- **Pros**: Rigorous mathematical convergence, provable sub-linear regret, sub-millisecond ($<0.2\text{ms}$) decision overhead, balanced exploration vs. exploitation.
- **Cons**: Assumes linear relationship between context and reward (mitigated by rich feature engineering and multi-objective scalar rewards).
- **Alternatives Rejected**:
  - *LLM-as-a-Router*: Adds excessive latency (500–1000ms) and token expense per request.
  - *Static Rules*: Unable to dynamically adapt to live provider latency degradation.

---

## ADR-005: Selection of Next.js & Tailwind CSS for Operations Dashboard

### Context
Operators and researchers need an interactive dashboard to test live prompt routing, inspect latency percentiles, monitor circuit breakers, and visualize empirical bandit regret curves.

### Decision
We selected **Next.js 14** (App Router) with **Tailwind CSS** and **Lucide Icons**.

### Consequences & Trade-offs
- **Pros**: Server-Side Rendering (SSR), native Server-Sent Events (SSE) streaming support, dark-mode glassmorphic aesthetics.
- **Cons**: Requires Node.js runtime for build step.
