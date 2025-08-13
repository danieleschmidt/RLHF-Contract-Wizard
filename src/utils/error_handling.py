"""
Comprehensive error handling and monitoring for RLHF-Contract-Wizard.

Provides structured error handling, monitoring, alerting, and recovery mechanisms.
"""

import time
import traceback
import functools
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import logging

from .helpers import setup_logging, create_timestamp


# Error severity levels
class ErrorSeverity(Enum):
    """Error severity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Error categories for classification
class ErrorCategory(Enum):
    """Error category classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    NETWORK = "network"
    DATABASE = "database"
    BLOCKCHAIN = "blockchain"
    VERIFICATION = "verification"
    CONTRACT = "contract"
    SYSTEM = "system"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Structured error context information."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    error_type: str
    message: str
    traceback_str: str
    additional_info: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    contract_id: Optional[str] = None
    resolution_attempted: bool = False
    resolved: bool = False


@dataclass
class SystemHealth:
    """System health monitoring data."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    error_rate: float
    contract_processing_rate: float
    verification_success_rate: float
    blockchain_sync_status: bool


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceed threshold, stop calling
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    Prevents cascade failures by stopping requests to failing services
    and allowing gradual recovery.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger open state
            recovery_timeout: Time to wait before trying half-open state
            success_threshold: Successful calls needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                
                raise e


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ErrorHandler:
    """
    Comprehensive error handling and monitoring system.
    
    Provides structured error logging, recovery mechanisms,
    health monitoring, and alerting capabilities.
    """
    
    def __init__(
        self,
        max_error_history: int = 1000,
        monitoring_interval: float = 60.0
    ):
        """
        Initialize error handler.
        
        Args:
            max_error_history: Maximum number of errors to keep in memory
            monitoring_interval: Health monitoring interval in seconds
        """
        self.logger = setup_logging()
        self.error_history: List[ErrorContext] = []
        self.max_error_history = max_error_history
        self.monitoring_interval = monitoring_interval
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_metrics: List[SystemHealth] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        # Error recovery strategies
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.NETWORK: [self._retry_with_backoff],
            ErrorCategory.DATABASE: [self._retry_with_backoff, self._reset_connection],
            ErrorCategory.BLOCKCHAIN: [self._retry_with_backoff, self._switch_provider],
            ErrorCategory.VERIFICATION: [self._retry_with_different_backend],
            ErrorCategory.COMPUTATION: [self._reduce_batch_size, self._clear_cache]
        }
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        additional_info: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        contract_id: Optional[str] = None,
        attempt_recovery: bool = True
    ) -> ErrorContext:
        """
        Handle an error with comprehensive logging and recovery.
        
        Args:
            error: Exception that occurred
            operation: Operation that failed
            severity: Error severity
            category: Error category
            additional_info: Additional context information
            user_id: User ID if applicable
            contract_id: Contract ID if applicable
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Error context object
        """
        error_context = ErrorContext(
            error_id=self._generate_error_id(),
            timestamp=create_timestamp(),
            severity=severity,
            category=category,
            operation=operation,
            error_type=type(error).__name__,
            message=str(error),
            traceback_str=traceback.format_exc(),
            additional_info=additional_info or {},
            user_id=user_id,
            contract_id=contract_id
        )
        
        # Log error
        self._log_error(error_context)
        
        # Store in history
        self._store_error(error_context)
        
        # Attempt recovery if enabled
        if attempt_recovery and severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            recovery_successful = self._attempt_recovery(error_context)
            error_context.resolution_attempted = True
            error_context.resolved = recovery_successful
        
        # Send alerts for high/critical errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_alert(error_context)
        
        return error_context
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return f"ERR_{uuid.uuid4().hex[:8].upper()}"
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_data = {
            'error_id': error_context.error_id,
            'operation': error_context.operation,
            'category': error_context.category.value,
            'severity': error_context.severity.value,
            'error_type': error_context.error_type,
            'error_msg': error_context.message,
            'user_id': error_context.user_id,
            'contract_id': error_context.contract_id,
            'additional_info': error_context.additional_info
        }
        
        if error_context.severity == ErrorSeverity.LOW:
            self.logger.info("Error occurred", extra=log_data)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Error occurred", extra=log_data)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", extra=log_data)
        else:  # CRITICAL
            self.logger.critical("Critical error occurred", extra=log_data)
            # Also log full traceback for critical errors
            self.logger.critical(f"Traceback: {error_context.traceback_str}")
    
    def _store_error(self, error_context: ErrorContext):
        """Store error in memory history."""
        self.error_history.append(error_context)
        
        # Maintain maximum history size
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from error using registered strategies."""
        recovery_strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in recovery_strategies:
            try:
                if strategy(error_context):
                    self.logger.info(f"Recovery successful for error {error_context.error_id}")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery strategy failed for error {error_context.error_id}: {e}")
        
        return False
    
    def _send_alert(self, error_context: ErrorContext):
        """Send alert for high/critical errors."""
        # In production, this would integrate with alerting systems
        # like PagerDuty, Slack, email, etc.
        alert_message = f"""
        🚨 RLHF Contract Alert - {error_context.severity.value.upper()}
        
        Error ID: {error_context.error_id}
        Operation: {error_context.operation}
        Category: {error_context.category.value}
        Message: {error_context.message}
        Time: {datetime.fromtimestamp(error_context.timestamp, timezone.utc)}
        
        Additional Info: {error_context.additional_info}
        """
        
        self.logger.critical(f"ALERT: {alert_message}")
        
        # Here you would typically send to external alerting systems
        # self._send_to_slack(alert_message)
        # self._send_email_alert(alert_message)
        # self._send_to_pagerduty(error_context)
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def start_monitoring(self):
        """Start system health monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system health monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                health_data = self._collect_health_metrics()
                self.health_metrics.append(health_data)
                
                # Maintain reasonable history size
                if len(self.health_metrics) > 1440:  # 24 hours at 1-minute intervals
                    self.health_metrics.pop(0)
                
                # Check for health issues
                self._check_health_thresholds(health_data)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _collect_health_metrics(self) -> SystemHealth:
        """Collect current system health metrics."""
        try:
            import psutil
            
            # Get system resource usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate error rate from recent history
            recent_errors = [
                e for e in self.error_history
                if e.timestamp > time.time() - 300  # Last 5 minutes
            ]
            error_rate = len(recent_errors) / 5.0  # Errors per minute
            
            return SystemHealth(
                timestamp=create_timestamp(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_connections=len(psutil.net_connections()),
                error_rate=error_rate,
                contract_processing_rate=self._get_contract_processing_rate(),
                verification_success_rate=self._get_verification_success_rate(),
                blockchain_sync_status=self._check_blockchain_sync()
            )
            
        except ImportError:
            # Fallback metrics if psutil not available
            return SystemHealth(
                timestamp=create_timestamp(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                active_connections=0,
                error_rate=0.0,
                contract_processing_rate=0.0,
                verification_success_rate=1.0,
                blockchain_sync_status=True
            )
    
    def _check_health_thresholds(self, health_data: SystemHealth):
        """Check health metrics against thresholds and alert if needed."""
        alerts = []
        
        if health_data.cpu_usage > 90:
            alerts.append(f"High CPU usage: {health_data.cpu_usage}%")
        
        if health_data.memory_usage > 85:
            alerts.append(f"High memory usage: {health_data.memory_usage}%")
        
        if health_data.disk_usage > 90:
            alerts.append(f"High disk usage: {health_data.disk_usage}%")
        
        if health_data.error_rate > 10:
            alerts.append(f"High error rate: {health_data.error_rate} errors/min")
        
        if health_data.verification_success_rate < 0.8:
            alerts.append(f"Low verification success rate: {health_data.verification_success_rate:.2%}")
        
        if not health_data.blockchain_sync_status:
            alerts.append("Blockchain sync issues detected")
        
        if alerts:
            alert_message = f"Health Check Alerts: {', '.join(alerts)}"
            self.logger.warning(alert_message)
    
    def _get_contract_processing_rate(self) -> float:
        """Get contract processing rate from recent history."""
        # Mock implementation - would track actual contract operations
        return 5.0  # contracts per minute
    
    def _get_verification_success_rate(self) -> float:
        """Get verification success rate from recent history."""
        verification_errors = [
            e for e in self.error_history
            if e.category == ErrorCategory.VERIFICATION and e.timestamp > time.time() - 3600
        ]
        # Mock calculation - would track actual verification attempts
        if len(verification_errors) > 10:
            return 0.7  # 70% success rate if many errors
        return 0.95  # 95% success rate normally
    
    def _check_blockchain_sync(self) -> bool:
        """Check blockchain synchronization status."""
        # Mock implementation - would check actual blockchain connectivity
        return True
    
    # Recovery strategy implementations
    def _retry_with_backoff(self, error_context: ErrorContext) -> bool:
        """Retry operation with exponential backoff."""
        self.logger.info(f"Attempting retry with backoff for error {error_context.error_id}")
        # Mock retry logic - would implement actual retry
        return True
    
    def _reset_connection(self, error_context: ErrorContext) -> bool:
        """Reset database/network connection."""
        self.logger.info(f"Attempting connection reset for error {error_context.error_id}")
        # Mock connection reset - would implement actual reset
        return True
    
    def _switch_provider(self, error_context: ErrorContext) -> bool:
        """Switch to backup provider/endpoint."""
        self.logger.info(f"Attempting provider switch for error {error_context.error_id}")
        # Mock provider switch - would implement actual switching
        return True
    
    def _retry_with_different_backend(self, error_context: ErrorContext) -> bool:
        """Retry verification with different backend."""
        self.logger.info(f"Attempting backend switch for error {error_context.error_id}")
        # Mock backend switch - would implement actual switching
        return True
    
    def _reduce_batch_size(self, error_context: ErrorContext) -> bool:
        """Reduce batch size to handle memory issues."""
        self.logger.info(f"Attempting batch size reduction for error {error_context.error_id}")
        # Mock batch size reduction - would implement actual reduction
        return True
    
    def _clear_cache(self, error_context: ErrorContext) -> bool:
        """Clear caches to free memory."""
        self.logger.info(f"Attempting cache clear for error {error_context.error_id}")
        # Mock cache clear - would implement actual cache clearing
        return True
    
    # Public API methods
    def get_error_summary(self, time_window: float = 3600) -> Dict[str, Any]:
        """
        Get error summary for time window.
        
        Args:
            time_window: Time window in seconds (default 1 hour)
            
        Returns:
            Error summary statistics
        """
        cutoff_time = time.time() - time_window
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'by_category': {},
                'by_severity': {},
                'recovery_rate': 0.0
            }
        
        by_category = {}
        by_severity = {}
        recovered_count = 0
        
        for error in recent_errors:
            # Count by category
            cat = error.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            
            # Count by severity
            sev = error.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            # Count recovered
            if error.resolved:
                recovered_count += 1
        
        return {
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / (time_window / 60),  # errors per minute
            'by_category': by_category,
            'by_severity': by_severity,
            'recovery_rate': recovered_count / len(recent_errors) if recent_errors else 0.0
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.health_metrics:
            return {'status': 'unknown', 'message': 'No health data available'}
        
        latest_health = self.health_metrics[-1]
        
        # Determine overall status
        if (latest_health.cpu_usage > 90 or 
            latest_health.memory_usage > 90 or 
            latest_health.error_rate > 20 or
            latest_health.verification_success_rate < 0.5):
            status = 'critical'
        elif (latest_health.cpu_usage > 70 or 
              latest_health.memory_usage > 70 or 
              latest_health.error_rate > 5 or
              latest_health.verification_success_rate < 0.8):
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'timestamp': latest_health.timestamp,
            'metrics': {
                'cpu_usage': latest_health.cpu_usage,
                'memory_usage': latest_health.memory_usage,
                'disk_usage': latest_health.disk_usage,
                'error_rate': latest_health.error_rate,
                'verification_success_rate': latest_health.verification_success_rate,
                'blockchain_sync': latest_health.blockchain_sync_status
            }
        }


def error_handler_decorator(
    error_handler: ErrorHandler,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    reraise: bool = True
):
    """
    Decorator for automatic error handling.
    
    Args:
        error_handler: ErrorHandler instance
        operation: Operation description
        severity: Error severity
        category: Error category
        reraise: Whether to re-raise the exception
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = error_handler.handle_error(
                    error=e,
                    operation=f"{operation}:{func.__name__}",
                    severity=severity,
                    category=category,
                    additional_info={
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                if reraise:
                    raise
                
                return None
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(
    error: Exception,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    **kwargs
) -> ErrorContext:
    """Convenience function for handling errors with global handler."""
    return global_error_handler.handle_error(
        error=error,
        operation=operation,
        severity=severity,
        category=category,
        **kwargs
    )