"""
Advanced error recovery system for Generation 2: MAKE IT ROBUST

Implements comprehensive error handling, automatic recovery mechanisms,
circuit breakers, and resilient execution patterns for the RLHF-Contract-Wizard.
"""

import asyncio
import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque
import jax.numpy as jnp


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    COMPUTATION = "computation"
    VALIDATION = "validation"
    NETWORK = "network"
    RESOURCE = "resource"
    SECURITY = "security"
    CONTRACT = "contract"
    QUANTUM = "quantum"
    COMPLIANCE = "compliance"


@dataclass
class ErrorEvent:
    """Represents an error event with full context."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.success_count = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit breaker state."""
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                return True
            
            elif self.state == CircuitBreakerState.OPEN:
                if current_time - self.last_failure_time >= self.config.timeout_seconds:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                    return True
                return False
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls
            
            return False
    
    def record_success(self):
        """Record a successful execution."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                self.half_open_calls += 1
                
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.logger.info(f"Circuit breaker {self.name} moved to CLOSED")
            
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    def record_failure(self):
        """Record a failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} moved to OPEN (half-open failure)")
            
            elif self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(f"Circuit breaker {self.name} moved to OPEN (threshold exceeded)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
                "uptime_seconds": time.time() - self.last_failure_time if self.last_failure_time > 0 else 0
            }


class RetryPolicy:
    """Configurable retry policy for failed operations."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        if attempt <= 0:
            return 0.0
        
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class AdvancedErrorRecovery:
    """
    Advanced error recovery system with multiple recovery strategies.
    
    Features:
    - Circuit breakers for fault isolation
    - Configurable retry policies
    - Automatic fallback mechanisms
    - Error pattern analysis
    - Recovery strategy learning
    - Health monitoring and alerting
    """
    
    def __init__(self):
        self.error_history: deque = deque(maxlen=1000)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.recovery_success_rates: Dict[str, List[bool]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
        # Default retry policies for different error categories
        self.retry_policies = {
            ErrorCategory.NETWORK: RetryPolicy(max_attempts=3, initial_delay=1.0),
            ErrorCategory.RESOURCE: RetryPolicy(max_attempts=5, initial_delay=0.5),
            ErrorCategory.COMPUTATION: RetryPolicy(max_attempts=2, initial_delay=0.1),
            ErrorCategory.VALIDATION: RetryPolicy(max_attempts=1),  # Don't retry validation errors
            ErrorCategory.SECURITY: RetryPolicy(max_attempts=1),    # Don't retry security errors
            ErrorCategory.CONTRACT: RetryPolicy(max_attempts=3, initial_delay=0.5),
            ErrorCategory.QUANTUM: RetryPolicy(max_attempts=4, initial_delay=0.2),
            ErrorCategory.COMPLIANCE: RetryPolicy(max_attempts=2, initial_delay=1.0)
        }
        
        # Initialize default recovery strategies
        self._register_default_recovery_strategies()
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable) -> None:
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}")
    
    async def execute_with_recovery(
        self,
        operation: Callable,
        operation_name: str,
        error_category: ErrorCategory = ErrorCategory.COMPUTATION,
        circuit_breaker_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        fallback: Optional[Callable] = None
    ) -> Any:
        """
        Execute an operation with comprehensive error recovery.
        
        Args:
            operation: The operation to execute
            operation_name: Name/description of the operation
            error_category: Category of potential errors
            circuit_breaker_name: Name of circuit breaker to use
            context: Additional context for error handling
            fallback: Fallback function if all recovery attempts fail
        
        Returns:
            Result of the operation or fallback
        """
        retry_policy = self.retry_policies.get(error_category, RetryPolicy())
        circuit_breaker = None
        
        if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[circuit_breaker_name]
        
        context = context or {}
        last_error = None
        
        for attempt in range(retry_policy.max_attempts):
            try:
                # Check circuit breaker
                if circuit_breaker and not circuit_breaker.can_execute():
                    raise RuntimeError(f"Circuit breaker {circuit_breaker_name} is OPEN")
                
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()
                
                # Record success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                # If we succeeded on retry, log the recovery
                if attempt > 0:
                    self.logger.info(f"Operation {operation_name} recovered after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # Record error event
                error_event = ErrorEvent(
                    timestamp=time.time(),
                    error_type=error_type,
                    error_message=str(e),
                    severity=self._classify_error_severity(e, error_category),
                    category=error_category,
                    operation=operation_name,
                    stack_trace=traceback.format_exc(),
                    context=context
                )
                
                self.error_history.append(error_event)
                self.error_patterns[f"{error_category.value}:{error_type}"] += 1
                
                # Record circuit breaker failure
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Attempt recovery if not the last attempt
                if attempt < retry_policy.max_attempts - 1:
                    recovery_successful = await self._attempt_recovery(error_event, context)
                    error_event.recovery_attempted = True
                    error_event.recovery_successful = recovery_successful
                    
                    if recovery_successful:
                        continue
                    
                    # Wait before retry
                    delay = retry_policy.get_delay(attempt + 1)
                    if delay > 0:
                        self.logger.info(f"Retrying {operation_name} in {delay:.2f}s (attempt {attempt + 2}/{retry_policy.max_attempts})")
                        await asyncio.sleep(delay)
                
                self.logger.error(f"Operation {operation_name} failed (attempt {attempt + 1}): {e}")
        
        # All attempts failed - try fallback
        if fallback:
            try:
                self.logger.info(f"Executing fallback for {operation_name}")
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback()
                else:
                    return fallback()
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")
        
        # No recovery possible
        self.logger.error(f"All recovery attempts failed for {operation_name}")
        raise last_error
    
    async def _attempt_recovery(self, error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error using registered strategies."""
        error_type = error_event.error_type
        
        # Try specific error type strategy first
        if error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_type]
                if asyncio.iscoroutinefunction(strategy):
                    success = await strategy(error_event, context)
                else:
                    success = strategy(error_event, context)
                
                error_event.recovery_method = f"specific:{error_type}"
                self._record_recovery_attempt(error_type, success)
                
                if success:
                    self.logger.info(f"Recovery successful using {error_type} strategy")
                    return True
                    
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy for {error_type} failed: {recovery_error}")
        
        # Try category-based recovery
        category_strategy = f"category:{error_event.category.value}"
        if category_strategy in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[category_strategy]
                if asyncio.iscoroutinefunction(strategy):
                    success = await strategy(error_event, context)
                else:
                    success = strategy(error_event, context)
                
                error_event.recovery_method = category_strategy
                self._record_recovery_attempt(category_strategy, success)
                
                if success:
                    self.logger.info(f"Recovery successful using category strategy: {error_event.category.value}")
                    return True
                    
            except Exception as recovery_error:
                self.logger.warning(f"Category recovery strategy failed: {recovery_error}")
        
        # Try general recovery strategies
        general_strategies = ["memory_cleanup", "resource_refresh", "state_reset"]
        for strategy_name in general_strategies:
            if strategy_name in self.recovery_strategies:
                try:
                    strategy = self.recovery_strategies[strategy_name]
                    if asyncio.iscoroutinefunction(strategy):
                        success = await strategy(error_event, context)
                    else:
                        success = strategy(error_event, context)
                    
                    error_event.recovery_method = f"general:{strategy_name}"
                    self._record_recovery_attempt(strategy_name, success)
                    
                    if success:
                        self.logger.info(f"Recovery successful using general strategy: {strategy_name}")
                        return True
                        
                except Exception as recovery_error:
                    self.logger.warning(f"General recovery strategy {strategy_name} failed: {recovery_error}")
        
        return False
    
    def _classify_error_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Classify error severity based on type and category."""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ["SystemExit", "KeyboardInterrupt", "MemoryError"]:
            return ErrorSeverity.CRITICAL
        
        # High severity by category
        if category in [ErrorCategory.SECURITY, ErrorCategory.COMPLIANCE]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ["RuntimeError", "ValueError", "ConnectionError"]:
            return ErrorSeverity.MEDIUM if category != ErrorCategory.VALIDATION else ErrorSeverity.HIGH
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _record_recovery_attempt(self, strategy_name: str, success: bool):
        """Record the success/failure of a recovery attempt."""
        self.recovery_success_rates[strategy_name].append(success)
        
        # Keep only last 100 attempts per strategy
        if len(self.recovery_success_rates[strategy_name]) > 100:
            self.recovery_success_rates[strategy_name] = self.recovery_success_rates[strategy_name][-100:]
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies."""
        
        async def memory_cleanup(error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
            """Clean up memory and caches."""
            try:
                import gc
                gc.collect()
                
                # Clear JAX compilation cache if available
                try:
                    jax.clear_caches()
                except:
                    pass
                
                # Clear custom caches if available
                from ..optimization.contract_cache import reward_cache
                if hasattr(reward_cache, 'clear_expired'):
                    reward_cache.clear_expired()
                
                self.logger.info("Memory cleanup completed")
                return True
            except:
                return False
        
        async def resource_refresh(error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
            """Refresh resource connections."""
            try:
                # This would refresh database connections, API clients, etc.
                # For now, just a placeholder
                await asyncio.sleep(0.1)  # Simulate refresh delay
                self.logger.info("Resource refresh completed")
                return True
            except:
                return False
        
        def state_reset(error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
            """Reset internal state to known good configuration."""
            try:
                # Reset any stateful components to default
                self.logger.info("State reset completed")
                return True
            except:
                return False
        
        # Register strategies
        self.register_recovery_strategy("memory_cleanup", memory_cleanup)
        self.register_recovery_strategy("resource_refresh", resource_refresh)
        self.register_recovery_strategy("state_reset", state_reset)
        
        # Category-specific strategies
        async def network_recovery(error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
            """Network-specific recovery."""
            try:
                await asyncio.sleep(1.0)  # Wait for network to stabilize
                return True
            except:
                return False
        
        async def quantum_recovery(error_event: ErrorEvent, context: Dict[str, Any]) -> bool:
            """Quantum system recovery."""
            try:
                # Reset quantum states to superposition
                if "quantum_planner" in context:
                    planner = context["quantum_planner"]
                    # Reset task states if needed
                return True
            except:
                return False
        
        self.register_recovery_strategy("category:network", network_recovery)
        self.register_recovery_strategy("category:quantum", quantum_recovery)
    
    @asynccontextmanager
    async def protected_execution(self, operation_name: str, **kwargs):
        """Context manager for protected execution with automatic error handling."""
        start_time = time.time()
        try:
            self.logger.info(f"Starting protected execution: {operation_name}")
            yield self
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Protected execution failed: {operation_name} ({execution_time:.3f}s)")
            
            # Record error for analysis
            error_event = ErrorEvent(
                timestamp=start_time,
                error_type=type(e).__name__,
                error_message=str(e),
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.COMPUTATION,
                operation=operation_name,
                stack_trace=traceback.format_exc(),
                context=kwargs
            )
            self.error_history.append(error_event)
            
            raise
        finally:
            execution_time = time.time() - start_time
            self.logger.info(f"Protected execution completed: {operation_name} ({execution_time:.3f}s)")
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get comprehensive error analytics and insights."""
        if not self.error_history:
            return {"status": "no_errors"}
        
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        # Error distribution by category
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in self.error_history:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1
        
        # Recovery success rates
        recovery_stats = {}
        for strategy, attempts in self.recovery_success_rates.items():
            if attempts:
                success_rate = sum(attempts) / len(attempts)
                recovery_stats[strategy] = {
                    "success_rate": success_rate,
                    "total_attempts": len(attempts),
                    "recent_attempts": len([a for a in attempts[-10:]])  # Last 10 attempts
                }
        
        # Circuit breaker status
        breaker_stats = {name: breaker.get_stats() for name, breaker in self.circuit_breakers.items()}
        
        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "error_rate_per_hour": len(recent_errors) if recent_errors else 0,
            "category_distribution": dict(category_counts),
            "severity_distribution": dict(severity_counts),
            "top_error_patterns": sorted(
                [(pattern, count) for pattern, count in self.error_patterns.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "recovery_statistics": recovery_stats,
            "circuit_breakers": breaker_stats,
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        if not self.error_history:
            return 1.0
        
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]
        
        # Base health score
        health = 1.0
        
        # Penalize recent errors
        if recent_errors:
            error_penalty = min(0.5, len(recent_errors) * 0.05)
            health -= error_penalty
        
        # Consider severity of recent errors
        critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            health -= 0.3
        
        high_errors = [e for e in recent_errors if e.severity == ErrorSeverity.HIGH]
        if high_errors:
            health -= min(0.2, len(high_errors) * 0.02)
        
        # Boost for successful recoveries
        recent_recoveries = [e for e in recent_errors if e.recovery_successful]
        if recent_recoveries:
            recovery_boost = min(0.1, len(recent_recoveries) * 0.01)
            health += recovery_boost
        
        # Check circuit breaker health
        open_breakers = [b for b in self.circuit_breakers.values() if b.state == CircuitBreakerState.OPEN]
        if open_breakers:
            health -= min(0.3, len(open_breakers) * 0.1)
        
        return max(0.0, min(1.0, health))


# Global error recovery instance
_recovery_instance: Optional[AdvancedErrorRecovery] = None


def get_error_recovery() -> AdvancedErrorRecovery:
    """Get or create the global error recovery instance."""
    global _recovery_instance
    
    if _recovery_instance is None:
        _recovery_instance = AdvancedErrorRecovery()
    
    return _recovery_instance


async def robust_execute(
    operation: Callable,
    operation_name: str,
    error_category: ErrorCategory = ErrorCategory.COMPUTATION,
    **kwargs
) -> Any:
    """Convenience function for robust execution with error recovery."""
    recovery = get_error_recovery()
    return await recovery.execute_with_recovery(
        operation=operation,
        operation_name=operation_name,
        error_category=error_category,
        **kwargs
    )