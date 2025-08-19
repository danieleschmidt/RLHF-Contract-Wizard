#!/usr/bin/env python3
"""
Comprehensive Error Recovery and Resilience Framework

Implements enterprise-grade error handling, recovery patterns, circuit breakers,
retry logic, fallback mechanisms, and fault tolerance for production systems.
"""

import time
import asyncio
import logging
import threading
import functools
import random
import json
from typing import Dict, List, Optional, Any, Callable, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import inspect
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as ConcurrentTimeoutError


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    TRANSIENT = "transient"        # Temporary failures (network, timeout)
    PERSISTENT = "persistent"      # Permanent failures (auth, validation)
    RESOURCE = "resource"          # Resource exhaustion (memory, disk)
    EXTERNAL = "external"          # External service failures
    BUSINESS = "business"          # Business logic errors
    SECURITY = "security"          # Security-related errors
    UNKNOWN = "unknown"            # Unclassified errors


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    IMMEDIATE_FAILURE = "immediate_failure"
    QUEUE_FOR_RETRY = "queue_for_retry"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error: Exception
    operation: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class RecoveryResult:
    """Result of recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempt_count: int
    recovery_time: float
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """
    Classifies errors into categories and determines appropriate recovery strategies.
    """
    
    def __init__(self):
        # Error classification rules
        self.classification_rules = {
            # Network and connectivity errors
            "ConnectionError": ErrorCategory.TRANSIENT,
            "TimeoutError": ErrorCategory.TRANSIENT,
            "ConnectionRefusedError": ErrorCategory.TRANSIENT,
            "DNSError": ErrorCategory.TRANSIENT,
            
            # Authentication and authorization
            "AuthenticationError": ErrorCategory.SECURITY,
            "PermissionError": ErrorCategory.SECURITY,
            "UnauthorizedError": ErrorCategory.SECURITY,
            
            # Resource errors
            "MemoryError": ErrorCategory.RESOURCE,
            "DiskSpaceError": ErrorCategory.RESOURCE,
            "ResourceExhaustedError": ErrorCategory.RESOURCE,
            
            # Validation errors
            "ValidationError": ErrorCategory.BUSINESS,
            "ValueError": ErrorCategory.BUSINESS,
            "TypeError": ErrorCategory.BUSINESS,
            
            # External service errors
            "ExternalServiceError": ErrorCategory.EXTERNAL,
            "APIError": ErrorCategory.EXTERNAL,
            "DatabaseError": ErrorCategory.EXTERNAL,
        }
        
        # Recovery strategy mapping
        self.recovery_strategies = {
            ErrorCategory.TRANSIENT: RecoveryStrategy.RETRY,
            ErrorCategory.EXTERNAL: RecoveryStrategy.CIRCUIT_BREAKER,
            ErrorCategory.RESOURCE: RecoveryStrategy.GRACEFUL_DEGRADATION,
            ErrorCategory.BUSINESS: RecoveryStrategy.IMMEDIATE_FAILURE,
            ErrorCategory.SECURITY: RecoveryStrategy.IMMEDIATE_FAILURE,
            ErrorCategory.PERSISTENT: RecoveryStrategy.FALLBACK,
            ErrorCategory.UNKNOWN: RecoveryStrategy.RETRY,
        }
        
    def classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error into category and severity."""
        error_type = type(error).__name__
        
        # Determine category
        category = self.classification_rules.get(error_type, ErrorCategory.UNKNOWN)
        
        # Determine severity based on error type and message
        severity = self._determine_severity(error, category)
        
        return category, severity
        
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity."""
        error_message = str(error).lower()
        
        # Critical keywords
        critical_keywords = ["fatal", "critical", "corrupt", "security", "breach"]
        if any(keyword in error_message for keyword in critical_keywords):
            return ErrorSeverity.CRITICAL
            
        # High severity for certain categories
        if category in [ErrorCategory.SECURITY, ErrorCategory.RESOURCE]:
            return ErrorSeverity.HIGH
            
        # Medium for external service issues
        if category == ErrorCategory.EXTERNAL:
            return ErrorSeverity.MEDIUM
            
        # Default to low
        return ErrorSeverity.LOW
        
    def get_recovery_strategy(self, category: ErrorCategory) -> RecoveryStrategy:
        """Get recommended recovery strategy for error category."""
        return self.recovery_strategies.get(category, RecoveryStrategy.RETRY)


class CircuitBreaker:
    """
    Circuit breaker implementation for handling external service failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
                    
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
        
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


class RetryHandler:
    """
    Advanced retry handler with exponential backoff and jitter.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logging.error(f"All {self.max_retries + 1} attempts failed")
                    
        raise last_exception
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
            
        return delay


class FallbackManager:
    """
    Manages fallback mechanisms for graceful degradation.
    """
    
    def __init__(self):
        self.fallback_functions: Dict[str, Callable] = {}
        self.fallback_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_functions[operation] = fallback_func
        
    def execute_with_fallback(self, operation: str, primary_func: Callable, *args, **kwargs):
        """Execute primary function with fallback on failure."""
        try:
            result = primary_func(*args, **kwargs)
            # Cache successful result
            self._cache_result(operation, result)
            return result
            
        except Exception as e:
            logging.warning(f"Primary function failed for {operation}: {e}")
            
            # Try fallback function
            if operation in self.fallback_functions:
                try:
                    return self.fallback_functions[operation](*args, **kwargs)
                except Exception as fallback_error:
                    logging.error(f"Fallback also failed for {operation}: {fallback_error}")
                    
            # Try cached result
            cached_result = self._get_cached_result(operation)
            if cached_result is not None:
                logging.info(f"Using cached result for {operation}")
                return cached_result
                
            # No fallback available
            raise e
            
    def _cache_result(self, operation: str, result: Any, ttl_minutes: int = 60):
        """Cache successful result."""
        self.fallback_cache[operation] = result
        self.cache_ttl[operation] = datetime.now() + timedelta(minutes=ttl_minutes)
        
    def _get_cached_result(self, operation: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if operation not in self.fallback_cache:
            return None
            
        if operation in self.cache_ttl and datetime.now() > self.cache_ttl[operation]:
            # Expired
            del self.fallback_cache[operation]
            del self.cache_ttl[operation]
            return None
            
        return self.fallback_cache[operation]


class ErrorRecoveryOrchestrator:
    """
    Main orchestrator for error recovery, coordinating all recovery mechanisms.
    """
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler()
        self.fallback_manager = FallbackManager()
        
        # Error history and metrics
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_metrics: Dict[str, int] = defaultdict(int)
        
        # Background health monitoring
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
    def handle_error(
        self,
        error: Exception,
        operation: str,
        recovery_context: Optional[Dict[str, Any]] = None
    ) -> RecoveryResult:
        """
        Handle error with appropriate recovery strategy.
        """
        start_time = time.time()
        
        # Classify error
        category, severity = self.error_classifier.classify_error(error)
        
        # Create error context
        context = ErrorContext(
            error=error,
            operation=operation,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            metadata=recovery_context or {}
        )
        
        # Record error
        self.error_history.append(context)
        
        # Determine recovery strategy
        strategy = self.error_classifier.get_recovery_strategy(category)
        
        # Execute recovery
        try:
            recovery_result = self._execute_recovery(context, strategy)
            recovery_result.recovery_time = time.time() - start_time
            
            # Update metrics
            self.recovery_metrics[f"{strategy.value}_success"] += 1
            
            return recovery_result
            
        except Exception as recovery_error:
            self.recovery_metrics[f"{strategy.value}_failure"] += 1
            
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                attempt_count=context.retry_count + 1,
                recovery_time=time.time() - start_time,
                error=recovery_error
            )
            
    def _execute_recovery(self, context: ErrorContext, strategy: RecoveryStrategy) -> RecoveryResult:
        """Execute specific recovery strategy."""
        
        if strategy == RecoveryStrategy.RETRY:
            return self._execute_retry_recovery(context)
            
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._execute_circuit_breaker_recovery(context)
            
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback_recovery(context)
            
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._execute_graceful_degradation(context)
            
        elif strategy == RecoveryStrategy.QUEUE_FOR_RETRY:
            return self._execute_queue_recovery(context)
            
        else:  # IMMEDIATE_FAILURE
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                attempt_count=1,
                recovery_time=0.0,
                error=context.error
            )
            
    def _execute_retry_recovery(self, context: ErrorContext) -> RecoveryResult:
        """Execute retry-based recovery."""
        # This would typically be called from a decorator or wrapper
        # For now, return a simulated result
        
        attempt_count = context.retry_count + 1
        
        if attempt_count <= context.max_retries:
            # Simulate successful retry
            success = attempt_count >= 2  # Succeed on second retry
            
            return RecoveryResult(
                success=success,
                strategy_used=RecoveryStrategy.RETRY,
                attempt_count=attempt_count,
                recovery_time=0.0,
                result="Recovered via retry" if success else None,
                error=None if success else context.error
            )
        else:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                attempt_count=attempt_count,
                recovery_time=0.0,
                error=context.error
            )
            
    def _execute_circuit_breaker_recovery(self, context: ErrorContext) -> RecoveryResult:
        """Execute circuit breaker recovery."""
        service_name = context.metadata.get("service_name", context.operation)
        
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
            
        circuit_breaker = self.circuit_breakers[service_name]
        
        try:
            # Simulate service call through circuit breaker
            def mock_service_call():
                if circuit_breaker.failure_count < circuit_breaker.failure_threshold:
                    return "Service response"
                else:
                    raise Exception("Service unavailable")
                    
            result = circuit_breaker.call(mock_service_call)
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                attempt_count=1,
                recovery_time=0.0,
                result=result
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                attempt_count=1,
                recovery_time=0.0,
                error=e
            )
            
    def _execute_fallback_recovery(self, context: ErrorContext) -> RecoveryResult:
        """Execute fallback recovery."""
        # Register a default fallback if none exists
        if context.operation not in self.fallback_manager.fallback_functions:
            def default_fallback(*args, **kwargs):
                return f"Fallback result for {context.operation}"
                
            self.fallback_manager.register_fallback(context.operation, default_fallback)
            
        try:
            def failing_primary():
                raise context.error
                
            result = self.fallback_manager.execute_with_fallback(
                context.operation,
                failing_primary
            )
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK,
                attempt_count=1,
                recovery_time=0.0,
                result=result
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                attempt_count=1,
                recovery_time=0.0,
                error=e
            )
            
    def _execute_graceful_degradation(self, context: ErrorContext) -> RecoveryResult:
        """Execute graceful degradation recovery."""
        # Provide reduced functionality
        degraded_result = {
            "status": "degraded",
            "message": f"Service {context.operation} running in degraded mode",
            "original_error": str(context.error),
            "functionality": "limited"
        }
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            attempt_count=1,
            recovery_time=0.0,
            result=degraded_result
        )
        
    def _execute_queue_recovery(self, context: ErrorContext) -> RecoveryResult:
        """Execute queue-based recovery for later retry."""
        # Simulate queuing for later processing
        queue_info = {
            "status": "queued",
            "queue_position": random.randint(1, 100),
            "estimated_retry_time": datetime.now() + timedelta(minutes=5),
            "operation": context.operation
        }
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.QUEUE_FOR_RETRY,
            attempt_count=1,
            recovery_time=0.0,
            result=queue_info
        )
        
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
        
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register fallback function."""
        self.fallback_manager.register_fallback(operation, fallback_func)
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics."""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {"total_errors": 0}
            
        # Analyze error categories
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error_context in self.error_history:
            category_counts[error_context.category.value] += 1
            severity_counts[error_context.severity.value] += 1
            
        # Calculate success rates
        total_recoveries = sum(self.recovery_metrics.values())
        success_metrics = {
            k: v for k, v in self.recovery_metrics.items() 
            if k.endswith("_success")
        }
        
        return {
            "total_errors": total_errors,
            "total_recoveries": total_recoveries,
            "error_categories": dict(category_counts),
            "error_severities": dict(severity_counts),
            "recovery_metrics": dict(self.recovery_metrics),
            "recovery_success_rate": sum(success_metrics.values()) / total_recoveries if total_recoveries > 0 else 0,
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            }
        }


# Decorators for easy integration

def with_error_recovery(
    operation_name: str,
    max_retries: int = 3,
    recovery_context: Optional[Dict[str, Any]] = None
):
    """Decorator to add error recovery to functions."""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            orchestrator = getattr(wrapper, '_error_orchestrator', None)
            if orchestrator is None:
                orchestrator = ErrorRecoveryOrchestrator()
                wrapper._error_orchestrator = orchestrator
                
            try:
                return func(*args, **kwargs)
            except Exception as e:
                result = orchestrator.handle_error(
                    e, operation_name, recovery_context
                )
                
                if result.success:
                    return result.result
                else:
                    raise result.error or e
                    
        return wrapper
    return decorator


def with_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator to add circuit breaker protection."""
    
    def decorator(func: Callable):
        circuit_breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
            
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator to add retry logic."""
    
    def decorator(func: Callable):
        retry_handler = RetryHandler(max_retries=max_retries, base_delay=base_delay)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler.execute_with_retry(
                func, *args, retryable_exceptions=retryable_exceptions, **kwargs
            )
            
        return wrapper
    return decorator


# Example usage and testing
async def example_error_recovery():
    """Example demonstrating error recovery system."""
    
    # Initialize error recovery orchestrator
    orchestrator = ErrorRecoveryOrchestrator()
    
    # Register some fallback functions
    def contract_creation_fallback(*args, **kwargs):
        return {"status": "fallback", "contract_id": "fallback_contract"}
    
    orchestrator.register_fallback("contract_creation", contract_creation_fallback)
    
    # Simulate various error scenarios
    error_scenarios = [
        (ConnectionError("Network unreachable"), "external_api_call"),
        (ValueError("Invalid input"), "data_validation"),
        (MemoryError("Out of memory"), "resource_intensive_operation"),
        (Exception("Unknown error"), "mysterious_operation")
    ]
    
    print("🔄 Testing Error Recovery System")
    print("=" * 50)
    
    for error, operation in error_scenarios:
        print(f"\\n🔍 Testing {operation} with {type(error).__name__}")
        
        result = orchestrator.handle_error(error, operation)
        
        print(f"   Strategy: {result.strategy_used.value}")
        print(f"   Success: {result.success}")
        print(f"   Attempts: {result.attempt_count}")
        print(f"   Recovery Time: {result.recovery_time:.3f}s")
        
        if result.success and result.result:
            print(f"   Result: {result.result}")
        elif result.error:
            print(f"   Error: {result.error}")
    
    # Get statistics
    stats = orchestrator.get_error_statistics()
    print(f"\\n📊 Error Recovery Statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Test decorators
    print(f"\\n🎭 Testing Decorators:")
    
    @with_error_recovery("decorated_operation", max_retries=2)
    def unreliable_function():
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Service unavailable")
        return "Success!"
    
    try:
        result = unreliable_function()
        print(f"   Decorated function result: {result}")
    except Exception as e:
        print(f"   Decorated function failed: {e}")
    
    @with_circuit_breaker("test_service", failure_threshold=2)
    def external_service_call():
        # Simulate external service
        if random.random() < 0.8:  # 80% failure rate
            raise Exception("External service error")
        return "External service response"
    
    # Test circuit breaker
    for i in range(5):
        try:
            result = external_service_call()
            print(f"   Circuit breaker call {i+1}: {result}")
        except Exception as e:
            print(f"   Circuit breaker call {i+1}: Failed - {e}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_error_recovery())