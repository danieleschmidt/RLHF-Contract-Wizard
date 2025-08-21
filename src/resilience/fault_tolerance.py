"""
Fault tolerance and recovery mechanisms.

Provides comprehensive fault tolerance, circuit breakers, retries,
and graceful degradation for RLHF contract systems.
"""

import time
import asyncio
import random
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
from collections import deque, defaultdict

from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"               # Fixed delay
    EXPONENTIAL = "exponential"   # Exponential backoff
    LINEAR = "linear"             # Linear backoff
    JITTERED = "jittered"         # Jittered exponential backoff


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance mechanisms."""
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    # Timeout settings
    operation_timeout: float = 30.0
    
    # Fallback settings
    enable_fallbacks: bool = True
    fallback_timeout: float = 5.0


@dataclass
class OperationMetrics:
    """Metrics for operation monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    circuit_breaker_trips: int = 0
    fallback_invocations: int = 0
    average_latency: float = 0.0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by temporarily disabling
    operations that are likely to fail.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # State
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        
        # Metrics
        self.metrics = OperationMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.metrics.total_calls += 1
            
            # Check if we should allow the call
            if not self._should_allow_call():
                self.logger.warning(
                    f"Circuit breaker {self.name} is OPEN - rejecting call"
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.name} is open"
                )
            
            # If in half-open state, track the call
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1
        
        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success
            with self._lock:
                self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure
            with self._lock:
                self._record_failure(execution_time)
            
            raise
    
    def _should_allow_call(self) -> bool:
        """Check if a call should be allowed based on current state."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.logger.info(
                    f"Circuit breaker {self.name} transitioning to HALF_OPEN"
                )
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def _record_success(self, execution_time: float):
        """Record a successful operation."""
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = time.time()
        
        # Update average latency
        self._update_average_latency(execution_time)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Check if we should close the circuit
            if self.half_open_calls >= self.half_open_max_calls:
                self.logger.info(
                    f"Circuit breaker {self.name} transitioning to CLOSED"
                )
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        
        # Reset failure count on success in closed state
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _record_failure(self, execution_time: float):
        """Record a failed operation."""
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = time.time()
        self.last_failure_time = time.time()
        
        # Update average latency
        self._update_average_latency(execution_time)
        
        self.failure_count += 1
        
        # Check if we should open the circuit
        if (
            self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]
            and self.failure_count >= self.failure_threshold
        ):
            self.logger.warning(
                f"Circuit breaker {self.name} transitioning to OPEN after {self.failure_count} failures"
            )
            self.state = CircuitBreakerState.OPEN
            self.metrics.circuit_breaker_trips += 1
    
    def _update_average_latency(self, execution_time: float):
        """Update running average latency."""
        if self.metrics.total_calls == 1:
            self.metrics.average_latency = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_latency = (
                alpha * execution_time + 
                (1 - alpha) * self.metrics.average_latency
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "half_open_calls": self.half_open_calls,
                "last_failure_time": self.last_failure_time,
                "metrics": {
                    "total_calls": self.metrics.total_calls,
                    "successful_calls": self.metrics.successful_calls,
                    "failed_calls": self.metrics.failed_calls,
                    "success_rate": self.metrics.success_rate,
                    "average_latency": self.metrics.average_latency,
                    "circuit_breaker_trips": self.metrics.circuit_breaker_trips
                }
            }
    
    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            self.logger.info(f"Circuit breaker {self.name} manually reset")


class RetryManager:
    """
    Retry manager with various backoff strategies.
    """
    
    def __init__(self, config: Optional[FaultToleranceConfig] = None):
        self.config = config or FaultToleranceConfig()
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry(func, *args, **kwargs)
        return wrapper
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(
                        f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Function {func.__name__} failed after {self.config.max_retries + 1} attempts"
                    )
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if self.config.retry_strategy == RetryStrategy.FIXED:
            return self.config.base_delay
        
        elif self.config.retry_strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2 ** attempt)
        
        elif self.config.retry_strategy == RetryStrategy.JITTERED:
            delay = self.config.base_delay * (2 ** attempt)
            # Add jitter (±25%)
            jitter = delay * 0.25 * (random.random() * 2 - 1)
            delay += jitter
        
        else:
            delay = self.config.base_delay
        
        return min(delay, self.config.max_delay)


class TimeoutManager:
    """
    Timeout manager for operation execution.
    """
    
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add timeout to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_timeout(func, *args, **kwargs)
        return wrapper
    
    def execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout."""
        import signal
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {self.timeout} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
            
        except TimeoutError:
            self.logger.error(
                f"Function {func.__name__} timed out after {self.timeout} seconds"
            )
            raise
            
        finally:
            signal.signal(signal.SIGALRM, old_handler)


class FallbackManager:
    """
    Fallback manager for graceful degradation.
    """
    
    def __init__(self):
        self.fallbacks: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallbacks[operation_name] = fallback_func
        self.logger.info(f"Registered fallback for operation: {operation_name}")
    
    def execute_with_fallback(
        self,
        operation_name: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)
            
        except Exception as e:
            self.logger.warning(
                f"Primary operation {operation_name} failed: {str(e)}, trying fallback"
            )
            
            if operation_name in self.fallbacks:
                try:
                    result = self.fallbacks[operation_name](*args, **kwargs)
                    self.logger.info(f"Fallback succeeded for operation: {operation_name}")
                    return result
                    
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback also failed for operation {operation_name}: {str(fallback_error)}"
                    )
                    raise fallback_error
            else:
                self.logger.error(f"No fallback registered for operation: {operation_name}")
                raise e


class FaultTolerantExecutor:
    """
    Comprehensive fault-tolerant executor combining all mechanisms.
    """
    
    def __init__(self, config: Optional[FaultToleranceConfig] = None):
        self.config = config or FaultToleranceConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager(self.config)
        self.fallback_manager = FallbackManager()
        self.logger = logging.getLogger(__name__)
        
        # Global metrics
        self.global_metrics = OperationMetrics()
    
    def get_or_create_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
                half_open_max_calls=self.config.half_open_max_calls
            )
        return self.circuit_breakers[name]
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function."""
        self.fallback_manager.register_fallback(operation_name, fallback_func)
    
    def execute(
        self,
        operation_name: str,
        func: Callable,
        *args,
        use_circuit_breaker: bool = True,
        use_retries: bool = True,
        use_timeout: bool = True,
        use_fallback: bool = None,
        **kwargs
    ) -> Any:
        """Execute function with full fault tolerance."""
        use_fallback = use_fallback if use_fallback is not None else self.config.enable_fallbacks
        
        self.global_metrics.total_calls += 1
        start_time = time.time()
        
        try:
            # Build execution chain
            execution_func = func
            
            # Add timeout
            if use_timeout:
                timeout_manager = TimeoutManager(self.config.operation_timeout)
                execution_func = timeout_manager(execution_func)
            
            # Add circuit breaker
            if use_circuit_breaker:
                circuit_breaker = self.get_or_create_circuit_breaker(operation_name)
                execution_func = circuit_breaker(execution_func)
            
            # Add retries
            if use_retries:
                execution_func = self.retry_manager(execution_func)
            
            # Execute with potential fallback
            if use_fallback:
                result = self.fallback_manager.execute_with_fallback(
                    operation_name, execution_func, *args, **kwargs
                )
            else:
                result = execution_func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self.global_metrics.successful_calls += 1
            self.global_metrics.last_success_time = time.time()
            self._update_global_latency(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self.global_metrics.failed_calls += 1
            self.global_metrics.last_failure_time = time.time()
            self._update_global_latency(execution_time)
            
            handle_error(
                error=e,
                operation=f"fault_tolerant_execution:{operation_name}",
                category=ErrorCategory.EXECUTION,
                severity=ErrorSeverity.MEDIUM,
                additional_info={
                    "operation_name": operation_name,
                    "execution_time": execution_time,
                    "use_circuit_breaker": use_circuit_breaker,
                    "use_retries": use_retries,
                    "use_fallback": use_fallback
                }
            )
            
            raise
    
    def _update_global_latency(self, execution_time: float):
        """Update global average latency."""
        if self.global_metrics.total_calls == 1:
            self.global_metrics.average_latency = execution_time
        else:
            alpha = 0.1
            self.global_metrics.average_latency = (
                alpha * execution_time + 
                (1 - alpha) * self.global_metrics.average_latency
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status."""
        return {
            "global_metrics": {
                "total_calls": self.global_metrics.total_calls,
                "successful_calls": self.global_metrics.successful_calls,
                "failed_calls": self.global_metrics.failed_calls,
                "success_rate": self.global_metrics.success_rate,
                "average_latency": self.global_metrics.average_latency,
                "last_success_time": self.global_metrics.last_success_time,
                "last_failure_time": self.global_metrics.last_failure_time
            },
            "circuit_breakers": {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            },
            "registered_fallbacks": list(self.fallback_manager.fallbacks.keys()),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "max_retries": self.config.max_retries,
                "operation_timeout": self.config.operation_timeout,
                "retry_strategy": self.config.retry_strategy.value
            }
        }
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers."""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.reset()
        self.logger.info("All circuit breakers reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Convenience decorators
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator to add circuit breaker to a function."""
    cb = CircuitBreaker(name, failure_threshold, recovery_timeout)
    return cb


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
):
    """Decorator to add retry logic to a function."""
    config = FaultToleranceConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retry_strategy=strategy
    )
    retry_manager = RetryManager(config)
    return retry_manager


def timeout(seconds: float):
    """Decorator to add timeout to a function."""
    timeout_manager = TimeoutManager(seconds)
    return timeout_manager


def fault_tolerant(
    operation_name: str,
    config: Optional[FaultToleranceConfig] = None
):
    """Decorator to add comprehensive fault tolerance to a function."""
    executor = FaultTolerantExecutor(config)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return executor.execute(operation_name, func, *args, **kwargs)
        return wrapper
    
    return decorator


# Global fault tolerance executor instance
_global_executor = FaultTolerantExecutor()


def get_global_executor() -> FaultTolerantExecutor:
    """Get the global fault tolerance executor."""
    return _global_executor


def configure_global_fault_tolerance(config: FaultToleranceConfig):
    """Configure global fault tolerance settings."""
    global _global_executor
    _global_executor = FaultTolerantExecutor(config)
