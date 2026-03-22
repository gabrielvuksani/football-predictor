"""Circuit breaker pattern for data provider resilience.

Implements the Circuit Breaker pattern to protect against cascading failures
when calling unreliable external APIs. Prevents repeated calls to failing
services and allows them time to recover.

State Machine:
    CLOSED  → Normal operation, requests pass through
    ↓ (failure threshold exceeded)
    OPEN   → Requests rejected immediately, no API calls
    ↓ (recovery timeout elapsed)
    HALF_OPEN → Limited test requests allowed, monitor recovery
    ↓ (success threshold reached)
    CLOSED → Service recovered, normal operation

References:
    Release It! (Nygard, 2007) - Circuit breaker pattern
    Michael Nygard - "CircuitBreaker" pattern description
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Callable, Generic, TypeVar

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout_seconds: float = 60.0  # Seconds in OPEN state before attempting HALF_OPEN
    success_threshold: int = 3  # Successes in HALF_OPEN before closing
    timeout_seconds: float = 30.0  # Timeout for individual requests


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker health monitoring."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    last_state_change: float = field(default_factory=time.time)

    def reset(self) -> None:
        """Reset metrics to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.last_state_change = time.time()

    def record_success(self) -> None:
        """Record a successful request."""
        self.success_count += 1
        self.total_calls += 1
        self.last_success_time = time.time()

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.total_calls += 1
        self.last_failure_time = time.time()


class CircuitBreaker(Generic[T]):
    """Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker(
            name="api_football",
            config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=60),
        )

        @breaker.call
        def fetch_matches():
            return api.get_matches()

        try:
            matches = fetch_matches()
        except CircuitBreakerOpen:
            # Circuit is open, use cached data
            matches = cache.get_matches()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Provider name for logging (e.g., "api_football")
            config: CircuitBreakerConfig with thresholds and timeouts
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        self._lock = Lock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        self.logger.info(
            f"Initialized {name} with config: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout_seconds}s"
        )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit is open and not in HALF_OPEN state
            Timeout: If function execution exceeds timeout
            Exception: Any exception raised by func (when circuit allows it)
        """
        with self._lock:
            state = self.metrics.state

        if state == CircuitState.OPEN:
            if self._should_attempt_reset():
                with self._lock:
                    self.metrics.state = CircuitState.HALF_OPEN
                    self.metrics.success_count = 0
                    self.logger.info(f"{self.name} circuit HALF_OPEN, testing recovery")
                # Fall through to execute
            else:
                raise CircuitBreakerOpen(
                    f"{self.name} circuit is OPEN. Retry in "
                    f"{self._time_until_retry():.1f}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.metrics.record_success()

            if self.metrics.state == CircuitState.HALF_OPEN:
                if self.metrics.success_count >= self.config.success_threshold:
                    self.metrics.state = CircuitState.CLOSED
                    self.metrics.failure_count = 0
                    self.logger.info(f"{self.name} circuit CLOSED, service recovered")
                else:
                    self.logger.debug(
                        f"{self.name} in HALF_OPEN, {self.metrics.success_count}/"
                        f"{self.config.success_threshold} successes"
                    )

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self.metrics.record_failure()

            if self.metrics.state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN resets to OPEN
                self.metrics.state = CircuitState.OPEN
                self.metrics.last_state_change = time.time()
                self.logger.warning(
                    f"{self.name} circuit reopened after failure in HALF_OPEN"
                )
            elif self.metrics.state == CircuitState.CLOSED:
                if self.metrics.failure_count >= self.config.failure_threshold:
                    self.metrics.state = CircuitState.OPEN
                    self.metrics.last_state_change = time.time()
                    self.logger.warning(
                        f"{self.name} circuit OPEN after {self.metrics.failure_count} failures"
                    )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.metrics.last_state_change is None:
            return True
        time_since_open = time.time() - self.metrics.last_state_change
        return time_since_open >= self.config.recovery_timeout_seconds

    def _time_until_retry(self) -> float:
        """Time in seconds until retry is allowed."""
        if self.metrics.last_state_change is None:
            return 0.0
        time_since_open = time.time() - self.metrics.last_state_change
        return max(0.0, self.config.recovery_timeout_seconds - time_since_open)

    def reset(self) -> None:
        """Manually reset circuit to CLOSED state."""
        with self._lock:
            self.metrics.reset()
        self.logger.info(f"{self.name} circuit manually reset to CLOSED")

    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.metrics.state.value,
                "failure_count": self.metrics.failure_count,
                "success_count": self.metrics.success_count,
                "total_calls": self.metrics.total_calls,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
            }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerRegistry:
    """Central registry for managing multiple circuit breakers.

    Allows monitoring and managing circuit breakers for multiple providers.
    """

    def __init__(self):
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = Lock()
        self.logger = logging.getLogger("circuit_breaker.registry")

    def register(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Register a new circuit breaker.

        Args:
            name: Provider name (e.g., "api_football", "fbref")
            config: Configuration for this breaker

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name in self._breakers:
                return self._breakers[name]
            breaker = CircuitBreaker(name, config)
            self._breakers[name] = breaker
            return breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)

    def get_all_states(self) -> dict[str, dict]:
        """Get state of all registered circuit breakers."""
        with self._lock:
            return {name: breaker.get_state() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
        self.logger.info("All circuit breakers reset to CLOSED")


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry.

    Args:
        name: Provider name
        config: Optional configuration

    Returns:
        CircuitBreaker instance
    """
    return _global_registry.register(name, config)
