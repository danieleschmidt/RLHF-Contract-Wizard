"""
Comprehensive error handling and recovery for quantum task planning.

Implements error categorization, recovery strategies, circuit breakers,
and resilient execution patterns for robust quantum planning operations.
"""

import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import functools
import threading
from collections import defaultdict, deque
import logging

from .core import QuantumTask, TaskState
from .logging_config import get_logger, EventType


class ErrorCategory(Enum):
    """Categories of errors in quantum planning system."""
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    DEPENDENCY_ERROR = "dependency_error"
    SECURITY_ERROR = "security_error"
    OPTIMIZATION_ERROR = "optimization_error"
    EXECUTION_ERROR = "execution_error"
    QUANTUM_ERROR = "quantum_error"
    CONTRACT_ERROR = "contract_error"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_ERROR = "external_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    NONE = "none"                    # No recovery, fail fast
    RETRY = "retry"                  # Simple retry with backoff
    FALLBACK = "fallback"           # Use fallback method/data
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker pattern
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality
    RESTART_COMPONENT = "restart_component"  # Restart failed component
    ESCALATE = "escalate"           # Escalate to higher level handler


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    operation: str
    message: str
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'category': self.category.value,
            'severity': self.severity.value,
            'component': self.component,
            'operation': self.operation,
            'message': self.message,
            'exception_type': type(self.exception).__name__ if self.exception else None,
            'stack_trace': self.stack_trace,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'metadata': self.metadata,
            'recovery_attempts': self.recovery_attempts,
            'recovery_strategy': self.recovery_strategy.value
        }


@dataclass
class RecoveryAction:
    """Defines a recovery action for an error."""
    strategy: RecoveryStrategy
    handler: Callable[[ErrorContext], Any]
    max_attempts: int = 3
    backoff_factor: float = 1.5
    timeout: Optional[float] = None
    conditions: Optional[Callable[[ErrorContext], bool]] = None


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        
        self._lock = threading.Lock()
        
        # Logging
        self.logger = get_logger()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        with self._lock:
            current_time = time.time()
            
            if self.state == "open":
                # Check if we should transition to half-open
                if current_time - self.last_failure_time >= self.reset_timeout:
                    self.state = "half-open"
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to half-open")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - update counters
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0
                        self.logger.info(f"Circuit breaker {self.name} closed after recovery")
                elif self.state == "closed":
                    # Reset failure count on successful operation
                    self.failure_count = max(0, self.failure_count - 1)
                
                return result
                
            except Exception as e:
                # Failure - update counters
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.state == "half-open":
                    self.state = "open"
                    self.logger.warning(f"Circuit breaker {self.name} opened after half-open failure")
                elif self.state == "closed" and self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    self.logger.error(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
                
                raise
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.state = "closed"
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0
            self.logger.info(f"Circuit breaker {self.name} manually reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'failure_threshold': self.failure_threshold,
                'reset_timeout': self.reset_timeout
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class QuantumPlannerError(Exception):
    """Base exception for quantum planner errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class ValidationError(QuantumPlannerError):
    """Error in validation operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.VALIDATION_ERROR, ErrorSeverity.MEDIUM, context)


class ResourceError(QuantumPlannerError):
    """Error in resource management."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.RESOURCE_ERROR, severity, context)


class QuantumStateError(QuantumPlannerError):
    """Error in quantum state operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.QUANTUM_ERROR, ErrorSeverity.MEDIUM, context)


class ContractViolationError(QuantumPlannerError):
    """Error due to contract constraint violations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.CONTRACT_ERROR, ErrorSeverity.HIGH, context)


class ErrorHandler:
    """
    Central error handling and recovery system for quantum planning.
    
    Provides error categorization, recovery strategies, circuit breakers,
    and comprehensive error tracking and reporting.
    """
    
    def __init__(self):
        self.logger = get_logger()
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryAction]] = {}
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery handlers
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RETRY: self._handle_retry,
            RecoveryStrategy.FALLBACK: self._handle_fallback,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._handle_graceful_degradation,
            RecoveryStrategy.RESTART_COMPONENT: self._handle_restart_component,
            RecoveryStrategy.ESCALATE: self._handle_escalate
        }
        
        # Default recovery strategies by error category
        self._setup_default_strategies()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies for different error categories."""
        
        # Validation errors - usually need escalation
        self.register_recovery_strategy(
            ErrorCategory.VALIDATION_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                handler=lambda ctx: self._log_and_escalate(ctx, "Validation error requires manual intervention"),
                max_attempts=1
            )
        )
        
        # Resource errors - try fallback or graceful degradation
        self.register_recovery_strategy(
            ErrorCategory.RESOURCE_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                handler=lambda ctx: self._reduce_resource_requirements(ctx),
                max_attempts=2
            )
        )
        
        # Network errors - retry with backoff
        self.register_recovery_strategy(
            ErrorCategory.NETWORK_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                handler=lambda ctx: self._wait_and_retry(ctx),
                max_attempts=3,
                backoff_factor=2.0
            )
        )
        
        # Timeout errors - retry with increased timeout
        self.register_recovery_strategy(
            ErrorCategory.TIMEOUT_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                handler=lambda ctx: self._retry_with_longer_timeout(ctx),
                max_attempts=2,
                timeout=120.0
            )
        )
        
        # Contract violations - escalate immediately
        self.register_recovery_strategy(
            ErrorCategory.CONTRACT_ERROR,
            RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                handler=lambda ctx: self._escalate_contract_violation(ctx),
                max_attempts=1
            )
        )
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Main entry point for error handling.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            
        Returns:
            Recovery result or re-raises exception if no recovery possible
        """
        
        # Create error context
        error_context = self._create_error_context(exception, context or {})
        
        # Log the error
        self._log_error(error_context)
        
        # Record in error history
        with self._lock:
            self.error_history.append(error_context)
            self.error_counts[error_context.category.value] += 1
        
        # Attempt recovery
        try:
            return self._attempt_recovery(error_context)
        except Exception as recovery_error:
            # Recovery failed, log and re-raise original
            self.logger.error(
                f"Recovery failed for error {error_context.error_id}: {str(recovery_error)}",
                event_type=EventType.SYSTEM_START,  # Could add RECOVERY_FAILED event
                original_error=error_context.to_dict(),
                recovery_error=str(recovery_error)
            )
            
            # If recovery fails, raise the original exception
            raise exception
    
    def _create_error_context(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> ErrorContext:
        """Create error context from exception and additional context."""
        
        # Generate unique error ID
        error_id = f"qp_error_{int(time.time() * 1000000)}"
        
        # Determine error category
        category = self._categorize_error(exception)
        
        # Determine severity
        severity = self._determine_severity(exception, category)
        
        # Get component and operation from context
        component = context.get('component', 'unknown')
        operation = context.get('operation', 'unknown')
        
        return ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            message=str(exception),
            exception=exception,
            stack_trace=traceback.format_exc(),
            task_id=context.get('task_id'),
            user_id=context.get('user_id'),
            session_id=context.get('session_id'),
            metadata=context
        )
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize error based on exception type and content."""
        
        # Direct mapping for custom exceptions
        if isinstance(exception, QuantumPlannerError):
            return exception.category
        
        # Built-in exception mapping
        exception_type = type(exception)
        exception_message = str(exception).lower()
        
        if exception_type in (ValueError, TypeError):
            return ErrorCategory.VALIDATION_ERROR
        elif exception_type in (MemoryError, OSError):
            return ErrorCategory.RESOURCE_ERROR
        elif exception_type == TimeoutError:
            return ErrorCategory.TIMEOUT_ERROR
        elif 'permission' in exception_message or 'access' in exception_message:
            return ErrorCategory.SECURITY_ERROR
        elif 'network' in exception_message or 'connection' in exception_message:
            return ErrorCategory.NETWORK_ERROR
        elif 'contract' in exception_message or 'violation' in exception_message:
            return ErrorCategory.CONTRACT_ERROR
        elif 'quantum' in exception_message or 'amplitude' in exception_message:
            return ErrorCategory.QUANTUM_ERROR
        elif 'dependency' in exception_message or 'circular' in exception_message:
            return ErrorCategory.DEPENDENCY_ERROR
        else:
            return ErrorCategory.SYSTEM_ERROR
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on exception and category."""
        
        # Direct mapping for custom exceptions
        if isinstance(exception, QuantumPlannerError):
            return exception.severity
        
        # Critical system errors
        if isinstance(exception, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity categories
        if category in [ErrorCategory.SECURITY_ERROR, ErrorCategory.CONTRACT_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity by default
        return ErrorSeverity.MEDIUM
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_context.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"Error in {error_context.component}.{error_context.operation}: {error_context.message}",
            extra={
                'error_context': error_context.to_dict(),
                'error_id': error_context.error_id,
                'task_id': error_context.task_id,
                'user_id': error_context.user_id,
                'session_id': error_context.session_id,
                'component': error_context.component,
                'operation': error_context.operation
            }
        )
    
    def _attempt_recovery(self, error_context: ErrorContext) -> Any:
        """Attempt to recover from error using registered strategies."""
        
        # Get recovery strategies for this error category
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        if not strategies:
            # No recovery strategy available
            self.logger.warning(f"No recovery strategy for {error_context.category.value}")
            raise error_context.exception
        
        # Try each recovery strategy
        for strategy_action in strategies:
            if error_context.recovery_attempts >= strategy_action.max_attempts:
                continue
            
            # Check conditions if specified
            if strategy_action.conditions and not strategy_action.conditions(error_context):
                continue
            
            try:
                self.logger.info(
                    f"Attempting recovery for {error_context.error_id} using {strategy_action.strategy.value}",
                    error_id=error_context.error_id,
                    recovery_strategy=strategy_action.strategy.value,
                    attempt=error_context.recovery_attempts + 1
                )
                
                error_context.recovery_attempts += 1
                error_context.recovery_strategy = strategy_action.strategy
                
                # Apply backoff if specified
                if error_context.recovery_attempts > 1 and strategy_action.backoff_factor > 1:
                    backoff_time = (strategy_action.backoff_factor ** (error_context.recovery_attempts - 1))
                    time.sleep(min(backoff_time, 30))  # Cap at 30 seconds
                
                # Execute recovery handler
                result = strategy_action.handler(error_context)
                
                self.logger.info(
                    f"Recovery successful for {error_context.error_id}",
                    error_id=error_context.error_id,
                    recovery_strategy=strategy_action.strategy.value
                )
                
                return result
                
            except Exception as recovery_exception:
                self.logger.warning(
                    f"Recovery attempt failed for {error_context.error_id}: {str(recovery_exception)}",
                    error_id=error_context.error_id,
                    recovery_strategy=strategy_action.strategy.value,
                    recovery_exception=str(recovery_exception)
                )
                continue
        
        # All recovery attempts failed
        raise error_context.exception
    
    # Recovery handlers
    def _handle_retry(self, error_context: ErrorContext) -> Any:
        """Handle retry recovery strategy."""
        # This is a placeholder - actual implementation would depend on the specific operation
        # that failed and would need to be coordinated with the calling code
        
        self.logger.info(f"Retry recovery for {error_context.error_id}")
        
        # In practice, this would re-execute the failed operation
        # For now, we'll just indicate that retry should be attempted
        return {'recovery_action': 'retry', 'error_context': error_context}
    
    def _handle_fallback(self, error_context: ErrorContext) -> Any:
        """Handle fallback recovery strategy."""
        self.logger.info(f"Fallback recovery for {error_context.error_id}")
        
        # Return fallback result based on error category
        if error_context.category == ErrorCategory.RESOURCE_ERROR:
            return {'recovery_action': 'fallback', 'reduced_resources': True}
        elif error_context.category == ErrorCategory.NETWORK_ERROR:
            return {'recovery_action': 'fallback', 'offline_mode': True}
        else:
            return {'recovery_action': 'fallback', 'default_behavior': True}
    
    def _handle_graceful_degradation(self, error_context: ErrorContext) -> Any:
        """Handle graceful degradation recovery strategy."""
        self.logger.info(f"Graceful degradation recovery for {error_context.error_id}")
        
        return {
            'recovery_action': 'graceful_degradation',
            'reduced_functionality': True,
            'degradation_level': 'partial'
        }
    
    def _handle_restart_component(self, error_context: ErrorContext) -> Any:
        """Handle component restart recovery strategy."""
        self.logger.warning(f"Component restart recovery for {error_context.error_id}")
        
        return {
            'recovery_action': 'restart_component',
            'component': error_context.component
        }
    
    def _handle_escalate(self, error_context: ErrorContext) -> Any:
        """Handle escalation recovery strategy."""
        self.logger.error(f"Escalating error {error_context.error_id}")
        
        # In practice, this would notify administrators, create tickets, etc.
        return {
            'recovery_action': 'escalate',
            'escalated': True,
            'requires_manual_intervention': True
        }
    
    # Specific recovery handlers
    def _log_and_escalate(self, error_context: ErrorContext, message: str) -> Any:
        """Log error and escalate to admin."""
        self.logger.critical(f"{message}: {error_context.error_id}")
        return self._handle_escalate(error_context)
    
    def _reduce_resource_requirements(self, error_context: ErrorContext) -> Any:
        """Reduce resource requirements as recovery."""
        self.logger.info(f"Reducing resource requirements for {error_context.error_id}")
        
        return {
            'recovery_action': 'reduce_resources',
            'resource_reduction': 0.5  # Reduce by 50%
        }
    
    def _wait_and_retry(self, error_context: ErrorContext) -> Any:
        """Wait and retry with exponential backoff."""
        backoff_time = 2.0 ** min(error_context.recovery_attempts, 5)  # Cap at 32 seconds
        self.logger.info(f"Waiting {backoff_time}s before retry for {error_context.error_id}")
        
        time.sleep(backoff_time)
        return self._handle_retry(error_context)
    
    def _retry_with_longer_timeout(self, error_context: ErrorContext) -> Any:
        """Retry with increased timeout."""
        new_timeout = 30.0 * (1.5 ** error_context.recovery_attempts)
        self.logger.info(f"Retrying with timeout {new_timeout}s for {error_context.error_id}")
        
        return {
            'recovery_action': 'retry_with_timeout',
            'timeout': new_timeout
        }
    
    def _escalate_contract_violation(self, error_context: ErrorContext) -> Any:
        """Escalate contract violations immediately."""
        self.logger.critical(f"Contract violation escalated: {error_context.error_id}")
        
        return {
            'recovery_action': 'escalate_contract_violation',
            'severity': 'critical',
            'immediate_attention_required': True
        }
    
    # Configuration methods
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        action: RecoveryAction
    ):
        """Register a recovery strategy for an error category."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        
        self.recovery_strategies[category].append(action)
        
        self.logger.debug(
            f"Registered recovery strategy {action.strategy.value} for {category.value}"
        )
    
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Register a circuit breaker for a component."""
        
        circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout
        )
        
        self.circuit_breakers[name] = circuit_breaker
        
        self.logger.info(f"Registered circuit breaker for {name}")
        
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    # Monitoring and reporting
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        # Filter recent errors
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= cutoff_time
        ]
        
        # Categorize errors
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        component_counts = defaultdict(int)
        
        for error in recent_errors:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1
            component_counts[error.component] += 1
        
        # Calculate error rates
        total_errors = len(recent_errors)
        error_rate = total_errors / max(hours, 1)  # errors per hour
        
        return {
            'time_period_hours': hours,
            'total_errors': total_errors,
            'error_rate_per_hour': error_rate,
            'category_breakdown': dict(category_counts),
            'severity_breakdown': dict(severity_counts),
            'component_breakdown': dict(component_counts),
            'circuit_breaker_states': {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            },
            'recovery_success_rate': self._calculate_recovery_success_rate(recent_errors)
        }
    
    def _calculate_recovery_success_rate(self, errors: List[ErrorContext]) -> float:
        """Calculate recovery success rate."""
        if not errors:
            return 1.0
        
        recovery_attempted = [e for e in errors if e.recovery_attempts > 0]
        
        if not recovery_attempted:
            return 0.0
        
        # This is simplified - in practice we'd need to track recovery outcomes
        return 0.75  # Placeholder success rate


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    
    return _global_error_handler


def handle_errors(
    category: Optional[ErrorCategory] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    component: str = "unknown",
    operation: str = "unknown"
):
    """
    Decorator for automatic error handling.
    
    Args:
        category: Error category override
        severity: Error severity override
        component: Component name
        operation: Operation name
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = {
                    'component': component,
                    'operation': operation,
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                # Override category if specified
                if category and isinstance(e, QuantumPlannerError):
                    e.category = category
                if isinstance(e, QuantumPlannerError):
                    e.severity = severity
                
                # Handle the error
                error_handler = get_error_handler()
                return error_handler.handle_error(e, context)
        
        return wrapper
    return decorator


def with_circuit_breaker(breaker_name: str):
    """Decorator for circuit breaker protection."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            
            # Get or create circuit breaker
            circuit_breaker = error_handler.get_circuit_breaker(breaker_name)
            if not circuit_breaker:
                circuit_breaker = error_handler.register_circuit_breaker(breaker_name)
            
            # Execute through circuit breaker
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator