import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.logging import logger
from app.models.request_log import RequestLog


class TelemetryWorker:
    """Asynchronous batch telemetry worker writing audit logs to PostgreSQL."""

    def __init__(self, batch_size: int = 50, flush_interval_seconds: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._recent_cache: Dict[str, Dict[str, Any]] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._running: bool = False

    def start(self):
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._process_loop())
            logger.info("TelemetryWorker background task started.")

    async def stop(self):
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        logger.info("TelemetryWorker stopped.")

    def record_event_sync(self, event_data: Dict[str, Any]) -> None:
        """Non-blocking synchronous enqueue for incoming request telemetry."""
        req_id = event_data.get("request_id")
        if req_id:
            # Maintain hot memory cache for immediate introspection
            self._recent_cache[req_id] = event_data
            if len(self._recent_cache) > 1000:
                # Evict oldest entry
                oldest_key = next(iter(self._recent_cache))
                del self._recent_cache[oldest_key]

        try:
            self._queue.put_nowait(event_data)
        except asyncio.QueueFull:
            logger.warning("Telemetry queue is full. Dropping trace event to protect memory.")

    async def record_event(self, event_data: Dict[str, Any]) -> None:
        """Async non-blocking enqueue for incoming request telemetry."""
        self.record_event_sync(event_data)

    async def get_recent_event(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves request details from hot cache or database."""
        if request_id in self._recent_cache:
            return self._recent_cache[request_id]

        # Query Database
        try:
            async with async_session_factory() as session:
                stmt = select(RequestLog).where(RequestLog.request_id == request_id)
                result = await session.execute(stmt)
                log = result.scalars().first()
                if log:
                    return {
                        "request_id": log.request_id,
                        "tenant_id": log.tenant_id,
                        "selected_provider_id": log.selected_provider_id,
                        "selected_model_id": log.selected_model_id,
                        "routing_strategy": log.routing_strategy,
                        "routing_features": log.routing_features,
                        "routing_score": log.routing_score,
                        "prompt_tokens": log.prompt_tokens,
                        "completion_tokens": log.completion_tokens,
                        "total_tokens": log.total_tokens,
                        "latency_ms": log.latency_ms,
                        "status_code": log.status_code,
                        "is_fallback": log.is_fallback,
                        "fallback_attempts": log.fallback_attempts,
                        "created_at": log.created_at.isoformat() if log.created_at else None,
                    }
        except Exception as e:
            logger.warning(f"Failed to query database for request_id '{request_id}': {str(e)}")

        return None

    async def _process_loop(self):
        while self._running:
            batch: List[Dict[str, Any]] = []
            start_time = asyncio.get_event_loop().time()

            while len(batch) < self.batch_size:
                timeout = max(0.05, self.flush_interval - (asyncio.get_event_loop().time() - start_time))
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch.append(event)
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    break

            if batch:
                await self._write_batch_to_db(batch)

    async def _write_batch_to_db(self, batch: List[Dict[str, Any]]):
        try:
            async with async_session_factory() as session:
                async with session.begin():
                    for item in batch:
                        log_entry = RequestLog(
                            request_id=item.get("request_id"),
                            tenant_id=item.get("tenant_id", "default"),
                            api_key_id=item.get("api_key_id"),
                            selected_provider_id=item.get("selected_provider_id", "unknown"),
                            selected_model_id=item.get("selected_model_id", "unknown"),
                            routing_strategy=item.get("routing_strategy", "unknown"),
                            routing_features=item.get("routing_features", {}),
                            routing_score=item.get("routing_score"),
                            prompt_tokens=item.get("prompt_tokens", 0),
                            completion_tokens=item.get("completion_tokens", 0),
                            total_tokens=item.get("total_tokens", 0),
                            latency_ms=item.get("latency_ms", 0.0),
                            status_code=item.get("status_code", 200),
                            is_fallback=item.get("is_fallback", False),
                            fallback_attempts=item.get("fallback_attempts", 0),
                            error_message=item.get("error_message"),
                            quality_score=item.get("quality_score"),
                            calculated_reward=item.get("calculated_reward"),
                        )
                        session.add(log_entry)
                await session.commit()
        except Exception as e:
            logger.warning(f"Telemetry batch database write skipped/failed: {str(e)}")

    async def flush(self):
        """Drains remaining queue items and writes them to database."""
        batch: List[Dict[str, Any]] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._write_batch_to_db(batch)


# Global Telemetry Worker Singleton
telemetry_worker = TelemetryWorker()
