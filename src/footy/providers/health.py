"""Health monitoring and fallback management for data providers.

Implements periodic health checks for data providers, tracks health metrics
(latency, error rate, availability), and manages automatic fallback ordering.

Features:
- Periodic health checks with configurable intervals
- Per-provider metrics (latency, error rate, last success/failure)
- Automatic fallback ordering based on health
- Health status categorization (healthy, degraded, unhealthy)
- Metrics export for monitoring dashboards
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class HealthStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"  # Operating normally
    DEGRADED = "degraded"  # Slow or occasional errors
    UNHEALTHY = "unhealthy"  # Frequent failures


@dataclass
class HealthMetrics:
    """Health metrics for a single provider."""
    name: str
    status: HealthStatus = HealthStatus.HEALTHY
    latency_ms: float = 0.0  # Last request latency
    error_rate: float = 0.0  # Recent error rate (0-1)
    availability: float = 100.0  # Uptime percentage (0-100)
    request_count: int = 0  # Total requests
    error_count: int = 0  # Total errors
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    last_check_time: Optional[float] = None
    check_duration_ms: float = 0.0  # Health check latency

    def update_success(self, latency_ms: float) -> None:
        """Record successful request."""
        self.latency_ms = latency_ms
        self.request_count += 1
        self.last_success_time = time.time()
        self._update_error_rate()
        self._update_availability()

    def update_failure(self, latency_ms: float = 0.0) -> None:
        """Record failed request."""
        self.latency_ms = latency_ms
        self.request_count += 1
        self.error_count += 1
        self.last_failure_time = time.time()
        self._update_error_rate()
        self._update_availability()

    def _update_error_rate(self) -> None:
        """Update error rate based on recent requests."""
        if self.request_count == 0:
            self.error_rate = 0.0
        else:
            # Rolling error rate: recent errors weighted more heavily
            # Use exponential moving average
            self.error_rate = self.error_count / self.request_count
            # Exponential smoothing to recent 20 requests
            if self.request_count > 20:
                recent_weight = 0.1
                self.error_rate = (
                    recent_weight * self.error_rate +
                    (1 - recent_weight) * min(1.0, self.error_count / 20.0)
                )

    def _update_availability(self) -> None:
        """Update availability percentage."""
        if self.request_count == 0:
            self.availability = 100.0
        else:
            self.availability = 100.0 * (1.0 - self.error_rate)

    def update_check_latency(self, latency_ms: float) -> None:
        """Record health check latency."""
        self.check_duration_ms = latency_ms
        self.last_check_time = time.time()

    def infer_status(self) -> None:
        """Infer health status from metrics."""
        if self.availability >= 95.0 and self.latency_ms < 5000:
            self.status = HealthStatus.HEALTHY
        elif self.availability >= 80.0 or self.latency_ms < 10000:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.UNHEALTHY


@dataclass
class HealthCheckConfig:
    """Configuration for health monitoring."""
    check_interval_seconds: float = 300.0  # Check every 5 minutes
    timeout_seconds: float = 10.0  # Health check timeout
    min_requests_for_status: int = 10  # Minimum requests before inferring status
    consecutive_failures_threshold: int = 3  # Open circuit after N failures
    healthy_latency_threshold_ms: float = 3000.0  # Healthy latency threshold
    degraded_latency_threshold_ms: float = 7000.0  # Degraded latency threshold


class ProviderHealthMonitor:
    """Monitor health of a single data provider.

    Tracks metrics and periodically runs health checks to verify provider
    availability and performance.
    """

    def __init__(
        self,
        name: str,
        health_check_fn: Callable[[], bool],
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize health monitor.

        Args:
            name: Provider name (e.g., "api_football", "fbref")
            health_check_fn: Callable that returns True if healthy, False otherwise
            config: Health check configuration
        """
        self.name = name
        self.health_check_fn = health_check_fn
        self.config = config or HealthCheckConfig()
        self.metrics = HealthMetrics(name=name)
        self._check_thread: Optional[threading.Thread] = None
        self._should_stop = False
        self.logger = logging.getLogger(f"health.{name}")

    def start_monitoring(self) -> None:
        """Start background health check thread."""
        if self._check_thread is not None and self._check_thread.is_alive():
            self.logger.warning(f"{self.name} monitoring already started")
            return

        self._should_stop = False
        self._check_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"health-monitor-{self.name}",
            daemon=True,
        )
        self._check_thread.start()
        self.logger.info(f"Started health monitoring for {self.name}")

    def stop_monitoring(self) -> None:
        """Stop background health check thread."""
        self._should_stop = True
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        self.logger.info(f"Stopped health monitoring for {self.name}")

    def _monitor_loop(self) -> None:
        """Background health check loop."""
        while not self._should_stop:
            try:
                self._perform_health_check()
                time.sleep(self.config.check_interval_seconds)
            except Exception as e:
                self.logger.exception(f"Health check error for {self.name}: {e}")
                time.sleep(min(self.config.check_interval_seconds, 30.0))

    def _perform_health_check(self) -> None:
        """Execute single health check."""
        start_time = time.time()

        try:
            # Run health check with timeout
            is_healthy = self._run_with_timeout(
                self.health_check_fn,
                timeout=self.config.timeout_seconds,
            )

            latency_ms = (time.time() - start_time) * 1000

            if is_healthy:
                self.metrics.update_success(latency_ms)
                self.logger.debug(
                    f"{self.name} health check OK ({latency_ms:.0f}ms)"
                )
            else:
                self.metrics.update_failure(latency_ms)
                self.logger.warning(f"{self.name} health check failed")

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.update_failure(latency_ms)
            self.logger.warning(f"{self.name} health check exception: {e}")

        self.metrics.update_check_latency((time.time() - start_time) * 1000)
        self.metrics.infer_status()

    def _run_with_timeout(self, fn: Callable, timeout: float) -> bool:
        """Run function with timeout.

        Note: Python doesn't support true thread killing, so this uses
        a simple threading approach. For production, consider using
        multiprocessing or async with timeout decorators.
        """
        result = [None]
        exception = [None]

        def wrapper() -> None:
            try:
                result[0] = fn()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if exception[0]:
            raise exception[0]
        if result[0] is None:
            raise TimeoutError(f"Health check exceeded {timeout}s timeout")
        return result[0]

    def record_success(self, latency_ms: float = 0.0) -> None:
        """Record successful API call."""
        self.metrics.update_success(latency_ms)
        self.metrics.infer_status()

    def record_failure(self, latency_ms: float = 0.0) -> None:
        """Record failed API call."""
        self.metrics.update_failure(latency_ms)
        self.metrics.infer_status()

    def get_metrics(self) -> HealthMetrics:
        """Get current health metrics."""
        return self.metrics


class HealthMonitorRegistry:
    """Central registry for managing multiple provider health monitors.

    Coordinates health monitoring across all providers and provides
    fallback ordering based on current health status.
    """

    def __init__(self):
        """Initialize registry."""
        self._monitors: dict[str, ProviderHealthMonitor] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger("health.registry")

    def register(
        self,
        name: str,
        health_check_fn: Callable[[], bool],
        config: Optional[HealthCheckConfig] = None,
    ) -> ProviderHealthMonitor:
        """Register a provider health monitor.

        Args:
            name: Provider name
            health_check_fn: Function that checks provider health
            config: Health check configuration

        Returns:
            ProviderHealthMonitor instance
        """
        with self._lock:
            if name in self._monitors:
                return self._monitors[name]

            monitor = ProviderHealthMonitor(name, health_check_fn, config)
            self._monitors[name] = monitor
            monitor.start_monitoring()
            return monitor

    def get_healthy_providers(self) -> list[str]:
        """Get list of healthy providers, ordered by preference.

        Returns:
            List of provider names, best-first (HEALTHY before DEGRADED before UNHEALTHY)
        """
        with self._lock:
            monitors = list(self._monitors.items())

        # Sort by status (healthy first), then by error rate (lower first)
        sorted_providers = sorted(
            monitors,
            key=lambda x: (
                x[1].metrics.status.value,  # CLOSED < DEGRADED < UNHEALTHY
                x[1].metrics.error_rate,  # Lower error rate first
                x[1].metrics.latency_ms,  # Lower latency first
            ),
        )

        return [name for name, _ in sorted_providers]

    def get_all_metrics(self) -> dict[str, HealthMetrics]:
        """Get metrics for all providers."""
        with self._lock:
            return {
                name: monitor.metrics
                for name, monitor in self._monitors.items()
            }

    def record_success(self, name: str, latency_ms: float = 0.0) -> None:
        """Record successful API call from provider."""
        with self._lock:
            if name in self._monitors:
                self._monitors[name].record_success(latency_ms)

    def record_failure(self, name: str, latency_ms: float = 0.0) -> None:
        """Record failed API call from provider."""
        with self._lock:
            if name in self._monitors:
                self._monitors[name].record_failure(latency_ms)

    def stop_all(self) -> None:
        """Stop all health monitoring threads."""
        with self._lock:
            for monitor in self._monitors.values():
                monitor.stop_monitoring()
        self.logger.info("Stopped all health monitoring")


# Global health monitor registry
_global_health_registry = HealthMonitorRegistry()


def get_health_monitor(
    name: str,
    health_check_fn: Callable[[], bool],
    config: Optional[HealthCheckConfig] = None,
) -> ProviderHealthMonitor:
    """Get or create a health monitor from the global registry.

    Args:
        name: Provider name
        health_check_fn: Health check function
        config: Optional configuration

    Returns:
        ProviderHealthMonitor instance
    """
    return _global_health_registry.register(name, health_check_fn, config)
