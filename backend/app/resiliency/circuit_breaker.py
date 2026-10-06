import threading
import time
from typing import Dict, Optional

from app.core.config import settings
from app.core.constants import CircuitState
from app.core.logging import logger


class CircuitBreaker:
    """Distributed-ready 3-state Circuit Breaker protecting against cascading downstream provider failures."""

    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        recovery_timeout_seconds: Optional[float] = None,
    ):
        self.failure_threshold = failure_threshold or settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout_seconds or settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
        self._lock = threading.Lock()

        # In-memory circuit state storage (maps provider_id -> state dict)
        self._state: Dict[str, CircuitState] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_failure_times: Dict[str, float] = {}

    def get_state(self, provider_id: str) -> CircuitState:
        with self._lock:
            state = self._state.get(provider_id, CircuitState.CLOSED)
            if state == CircuitState.OPEN:
                last_fail = self._last_failure_times.get(provider_id, 0.0)
                # Check if recovery timeout has elapsed
                if time.time() - last_fail >= self.recovery_timeout:
                    logger.info(
                        f"Circuit for [{provider_id}] transitioning from OPEN to HALF_OPEN (probe mode).",
                        extra={"extra_data": {"provider": provider_id, "state": "HALF_OPEN"}},
                    )
                    self._state[provider_id] = CircuitState.HALF_OPEN
                    return CircuitState.HALF_OPEN
            return state

    def can_execute(self, provider_id: str) -> bool:
        """Returns True if the circuit is CLOSED or in HALF_OPEN trial mode."""
        state = self.get_state(provider_id)
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self, provider_id: str) -> None:
        """Records a successful response, resetting failures and closing half-open circuits."""
        with self._lock:
            current_state = self._state.get(provider_id, CircuitState.CLOSED)
            if current_state == CircuitState.HALF_OPEN:
                logger.info(
                    f"Circuit for [{provider_id}] recovered: HALF_OPEN -> CLOSED",
                    extra={"extra_data": {"provider": provider_id, "state": "CLOSED"}},
                )
            self._state[provider_id] = CircuitState.CLOSED
            self._failure_counts[provider_id] = 0

    def record_failure(self, provider_id: str) -> None:
        """Increments consecutive failures and transitions circuit to OPEN if threshold exceeded."""
        with self._lock:
            fails = self._failure_counts.get(provider_id, 0) + 1
            self._failure_counts[provider_id] = fails
            self._last_failure_times[provider_id] = time.time()

            if fails >= self.failure_threshold:
                if self._state.get(provider_id) != CircuitState.OPEN:
                    logger.warning(
                        f"Circuit for [{provider_id}] tripped to OPEN after {fails} consecutive failures.",
                        extra={"extra_data": {"provider": provider_id, "failures": fails, "state": "OPEN"}},
                    )
                self._state[provider_id] = CircuitState.OPEN

    def reset(self, provider_id: str) -> None:
        """Manually resets circuit breaker state."""
        with self._lock:
            self._state[provider_id] = CircuitState.CLOSED
            self._failure_counts[provider_id] = 0
            self._last_failure_times[provider_id] = 0.0


# Global Circuit Breaker Singleton
circuit_breaker = CircuitBreaker()
