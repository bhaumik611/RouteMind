import time
import pytest
from app.core.constants import CircuitState
from app.resiliency.circuit_breaker import CircuitBreaker


def test_circuit_breaker_initial_state():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=0.5)
    assert cb.get_state("test-provider") == CircuitState.CLOSED
    assert cb.can_execute("test-provider") is True


def test_circuit_breaker_tripping_to_open():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=0.5)

    cb.record_failure("test-provider")
    assert cb.get_state("test-provider") == CircuitState.CLOSED

    cb.record_failure("test-provider")
    assert cb.get_state("test-provider") == CircuitState.CLOSED

    # 3rd failure trips the breaker
    cb.record_failure("test-provider")
    assert cb.get_state("test-provider") == CircuitState.OPEN
    assert cb.can_execute("test-provider") is False


def test_circuit_breaker_recovery_to_half_open_and_closed():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)

    cb.record_failure("test-provider")
    cb.record_failure("test-provider")
    assert cb.get_state("test-provider") == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(0.15)
    assert cb.get_state("test-provider") == CircuitState.HALF_OPEN
    assert cb.can_execute("test-provider") is True

    # Success in half-open state closes the circuit
    cb.record_success("test-provider")
    assert cb.get_state("test-provider") == CircuitState.CLOSED
    assert cb.can_execute("test-provider") is True
