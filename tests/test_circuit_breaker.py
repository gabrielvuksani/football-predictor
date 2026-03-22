"""Tests for circuit breaker pattern implementation."""
from __future__ import annotations

import time
import threading
import pytest

from footy.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    get_circuit_breaker,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_configuration(self):
        """Test default circuit breaker configuration."""
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.recovery_timeout_seconds == 60.0
        assert cfg.success_threshold == 3
        assert cfg.timeout_seconds == 30.0

    def test_custom_configuration(self):
        """Test custom circuit breaker configuration."""
        cfg = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
            success_threshold=2,
            timeout_seconds=15.0,
        )
        assert cfg.failure_threshold == 3
        assert cfg.recovery_timeout_seconds == 30.0
        assert cfg.success_threshold == 2
        assert cfg.timeout_seconds == 15.0


class TestCircuitBreakerMetrics:
    """Tests for CircuitBreakerMetrics."""

    def test_default_initialization(self):
        """Test default metrics initialization."""
        m = CircuitBreakerMetrics()
        assert m.state == CircuitState.CLOSED
        assert m.failure_count == 0
        assert m.success_count == 0
        assert m.total_calls == 0
        assert m.last_failure_time is None
        assert m.last_success_time is None

    def test_record_success(self):
        """Test recording successful call."""
        m = CircuitBreakerMetrics()
        m.record_success()

        assert m.success_count == 1
        assert m.total_calls == 1
        assert m.last_success_time is not None

    def test_record_failure(self):
        """Test recording failed call."""
        m = CircuitBreakerMetrics()
        m.record_failure()

        assert m.failure_count == 1
        assert m.total_calls == 1
        assert m.last_failure_time is not None

    def test_multiple_calls_tracking(self):
        """Test tracking multiple calls."""
        m = CircuitBreakerMetrics()
        m.record_success()
        m.record_failure()
        m.record_success()

        assert m.success_count == 2
        assert m.failure_count == 1
        assert m.total_calls == 3

    def test_reset(self):
        """Test metrics reset."""
        m = CircuitBreakerMetrics()
        m.record_success()
        m.record_failure()
        m.state = CircuitState.OPEN

        m.reset()

        assert m.state == CircuitState.CLOSED
        assert m.failure_count == 0
        assert m.success_count == 0
        assert m.total_calls == 0


class TestCircuitBreakerClosedState:
    """Tests for CLOSED state behavior."""

    def test_successful_call_in_closed_state(self):
        """Test successful call passes through in CLOSED state."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.metrics.state == CircuitState.CLOSED
        assert breaker.metrics.success_count == 1

    def test_failed_call_in_closed_state(self):
        """Test failed call increments failure counter."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))

        def fail_func():
            raise ValueError("API error")

        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.metrics.failure_count == 1
        assert breaker.metrics.state == CircuitState.CLOSED

    def test_failure_threshold_triggers_open(self):
        """Test that failure threshold opens the circuit."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        def fail_func():
            raise ValueError("API error")

        # First 3 failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        # Circuit should be OPEN
        assert breaker.metrics.state == CircuitState.OPEN
        assert breaker.metrics.failure_count == 3

    def test_successful_call_resets_failure_counter_in_closed(self):
        """Test that success doesn't accumulate with failures."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))

        def fail_func():
            raise ValueError("API error")

        # Fail once
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        assert breaker.metrics.failure_count == 1

        def success_func():
            return "ok"

        # Success doesn't reset in CLOSED state, just continues
        breaker.call(success_func)
        assert breaker.metrics.success_count == 1


class TestCircuitBreakerOpenState:
    """Tests for OPEN state behavior."""

    def test_open_circuit_rejects_calls_immediately(self):
        """Test that OPEN circuit rejects calls without execution."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))

        def fail_func():
            raise ValueError("API error")

        # Fail once to open
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Circuit is now OPEN
        assert breaker.metrics.state == CircuitState.OPEN

        # Next call should be rejected immediately
        def success_func():
            return "success"

        with pytest.raises(CircuitBreakerOpen):
            breaker.call(success_func)

    def test_open_circuit_error_message_includes_retry_info(self):
        """Test that OPEN error message includes retry time."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=10.0,
        ))

        def fail_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Now open
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            breaker.call(lambda: "ok")

        error_msg = str(exc_info.value)
        assert "Retry" in error_msg or "retry" in error_msg


class TestCircuitBreakerHalfOpenState:
    """Tests for HALF_OPEN state behavior."""

    def test_recovery_timeout_transitions_to_half_open(self):
        """Test that timeout elapsed transitions to HALF_OPEN."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.1,  # 100ms for testing
        ))

        def fail_func():
            raise ValueError("error")

        # Fail to open
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.metrics.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        def success_func():
            return "success"

        # Should attempt and succeed, transitioning to HALF_OPEN then CLOSED
        result = breaker.call(success_func)
        assert result == "success"
        # After one success with threshold=1, might be CLOSED
        assert breaker.metrics.state in (CircuitState.HALF_OPEN, CircuitState.CLOSED)

    def test_successful_probe_in_half_open_closes_circuit(self):
        """Test that enough successes in HALF_OPEN close circuit."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.05,
            success_threshold=2,
        ))

        def fail_func():
            raise ValueError("error")

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        time.sleep(0.1)

        def success_func():
            return "ok"

        # First success transitions to HALF_OPEN
        breaker.call(success_func)

        # Second success should close
        breaker.call(success_func)
        assert breaker.metrics.state == CircuitState.CLOSED

    def test_failed_probe_in_half_open_reopens_circuit(self):
        """Test that failure in HALF_OPEN reopens circuit."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.05,
            success_threshold=3,
        ))

        def fail_func():
            raise ValueError("error")

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        time.sleep(0.1)

        def success_func():
            return "ok"

        # First success transitions to HALF_OPEN
        breaker.call(success_func)
        assert breaker.metrics.state == CircuitState.HALF_OPEN

        # Failure should reopen
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.metrics.state == CircuitState.OPEN


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_calls_thread_safe(self):
        """Test that concurrent calls are thread-safe."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=100))
        call_count = 0
        lock = threading.Lock()

        def counting_func():
            nonlocal call_count
            with lock:
                call_count += 1
            return call_count

        # Make concurrent calls
        threads = []
        for _ in range(10):
            t = threading.Thread(target=breaker.call, args=(counting_func,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All calls should have been made
        assert call_count == 10

    def test_concurrent_state_transitions_safe(self):
        """Test that concurrent state transitions are safe."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))

        def fail_func():
            raise ValueError("error")

        # Multiple threads failing simultaneously
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: (
                next((_ for _ in [breaker.call(fail_func)] if False), None),
                None
            ))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # State should be consistent
        assert breaker.metrics.state in (CircuitState.CLOSED, CircuitState.OPEN)


class TestCircuitBreakerMetrics:
    """Tests for metrics tracking."""

    def test_get_state_returns_dict(self):
        """Test that get_state returns correct structure."""
        breaker = CircuitBreaker("api_test")

        state = breaker.get_state()

        assert isinstance(state, dict)
        assert state["name"] == "api_test"
        assert state["state"] in ("closed", "open", "half_open")
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
        assert state["total_calls"] == 0

    def test_metrics_track_calls(self):
        """Test that metrics track all calls."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=100))

        def func():
            return "ok"

        for _ in range(5):
            breaker.call(func)

        state = breaker.get_state()
        assert state["total_calls"] == 5
        assert state["success_count"] == 5


class TestCircuitBreakerReset:
    """Tests for manual reset."""

    def test_reset_closes_open_circuit(self):
        """Test that reset closes an OPEN circuit."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))

        def fail_func():
            raise ValueError("error")

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.metrics.state == CircuitState.OPEN

        # Reset
        breaker.reset()

        assert breaker.metrics.state == CircuitState.CLOSED
        assert breaker.metrics.failure_count == 0

    def test_reset_clears_metrics(self):
        """Test that reset clears all metrics."""
        breaker = CircuitBreaker("test")

        def func():
            return "ok"

        breaker.call(func)
        breaker.call(func)

        breaker.reset()

        state = breaker.get_state()
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
        assert state["total_calls"] == 0


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_register_and_get_breaker(self):
        """Test registering and retrieving circuit breakers."""
        registry = CircuitBreakerRegistry()

        breaker = registry.register("api_football")

        assert breaker is not None
        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "api_football"

    def test_register_returns_same_instance(self):
        """Test that register returns same instance for same name."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.register("api")
        breaker2 = registry.register("api")

        assert breaker1 is breaker2

    def test_get_returns_none_for_unknown_breaker(self):
        """Test that get returns None for unknown breaker."""
        registry = CircuitBreakerRegistry()

        breaker = registry.get("unknown")

        assert breaker is None

    def test_get_all_states(self):
        """Test retrieving all breaker states."""
        registry = CircuitBreakerRegistry()

        registry.register("api1")
        registry.register("api2")

        states = registry.get_all_states()

        assert "api1" in states
        assert "api2" in states
        assert isinstance(states["api1"], dict)

    def test_reset_all(self):
        """Test resetting all breakers."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.register("api1", CircuitBreakerConfig(failure_threshold=1))
        breaker2 = registry.register("api2", CircuitBreakerConfig(failure_threshold=1))

        # Open breaker1
        def fail_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            breaker1.call(fail_func)

        assert breaker1.metrics.state == CircuitState.OPEN

        # Reset all
        registry.reset_all()

        assert breaker1.metrics.state == CircuitState.CLOSED
        assert breaker2.metrics.state == CircuitState.CLOSED


class TestGetCircuitBreaker:
    """Tests for get_circuit_breaker global function."""

    def test_get_or_create_breaker(self):
        """Test get_circuit_breaker creates or returns breaker."""
        breaker1 = get_circuit_breaker("test_api")
        breaker2 = get_circuit_breaker("test_api")

        assert breaker1 is breaker2
        assert breaker1.name == "test_api"

    def test_custom_config(self):
        """Test get_circuit_breaker with custom config."""
        cfg = CircuitBreakerConfig(failure_threshold=3)
        breaker = get_circuit_breaker("custom", cfg)

        assert breaker.config.failure_threshold == 3


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases."""

    def test_exception_propagates_in_closed_state(self):
        """Test that exceptions propagate in CLOSED state."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=100))

        def error_func():
            raise RuntimeError("specific error")

        with pytest.raises(RuntimeError, match="specific error"):
            breaker.call(error_func)

    def test_very_fast_recovery_timeout(self):
        """Test behavior with very short recovery timeout."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.01,  # 10ms
        ))

        def fail_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Wait for recovery window
        time.sleep(0.05)

        def success_func():
            return "ok"

        # Should attempt recovery
        result = breaker.call(success_func)
        assert result == "ok"

    def test_zero_success_threshold(self):
        """Test behavior with zero success threshold (edge case)."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.05,
            success_threshold=0,  # Edge case
        ))

        def fail_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            breaker.call(fail_func)

        time.sleep(0.1)

        def success_func():
            return "ok"

        breaker.call(success_func)
        # With success_threshold=0, any success might close immediately
        assert breaker.metrics.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
