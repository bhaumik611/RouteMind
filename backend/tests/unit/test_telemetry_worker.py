import pytest
from app.services.telemetry_worker import TelemetryWorker


@pytest.mark.asyncio
async def test_telemetry_worker_recording_and_hot_cache():
    worker = TelemetryWorker(batch_size=10, flush_interval_seconds=0.1)

    event_data = {
        "request_id": "test-telemetry-req-123",
        "tenant_id": "tenant-corp",
        "selected_provider_id": "local_mock",
        "selected_model_id": "mock-free-model",
        "routing_strategy": "rule_based",
        "routing_features": {"category": "general"},
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40,
        "latency_ms": 55.4,
        "status_code": 200,
    }

    # Record sync
    worker.record_event_sync(event_data)

    # Hot cache lookup should immediately find the record
    cached = await worker.get_recent_event("test-telemetry-req-123")
    assert cached is not None
    assert cached["request_id"] == "test-telemetry-req-123"
    assert cached["selected_provider_id"] == "local_mock"
    assert cached["total_tokens"] == 40
