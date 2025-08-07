"""
Advanced logging configuration for quantum task planning.

Implements structured logging, performance monitoring, audit trails,
and integration with monitoring systems for comprehensive observability.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from contextlib import contextmanager
import traceback

from .core import QuantumTask, TaskState


class LogLevel(Enum):
    """Extended log levels for quantum planning."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 60  # Custom level for audit events


class EventType(Enum):
    """Types of events to log."""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_EXECUTED = "task_executed"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    OPTIMIZATION_START = "optimization_start"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    VALIDATION_ERROR = "validation_error"
    SECURITY_EVENT = "security_event"
    CONTRACT_VIOLATION = "contract_violation"
    PERFORMANCE_ALERT = "performance_alert"
    RESOURCE_ALLOCATION = "resource_allocation"
    ENTANGLEMENT_CREATED = "entanglement_created"
    QUANTUM_STATE_CHANGE = "quantum_state_change"


@dataclass
class LogContext:
    """Context information for structured logging."""
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    contract_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    thread_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    error_count: int = 0
    warning_count: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True):
        """Mark operation as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        
        # Could add CPU/memory monitoring here
        # self.cpu_usage = psutil.cpu_percent()
        # self.memory_usage = psutil.virtual_memory().percent


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log entry
        log_entry = {
            'timestamp': time.time(),
            'iso_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime()),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': threading.current_thread().ident,
            'process_id': os.getpid()
        }
        
        # Add context if available
        if hasattr(record, 'context') and record.context:
            log_entry['context'] = record.context.to_dict() if hasattr(record.context, 'to_dict') else record.context
        
        # Add event type if available
        if hasattr(record, 'event_type'):
            log_entry['event_type'] = record.event_type
        
        # Add performance metrics if available
        if hasattr(record, 'metrics'):
            log_entry['metrics'] = asdict(record.metrics) if hasattr(record.metrics, '__dataclass_fields__') else record.metrics
        
        # Add extra fields from record
        extra_fields = ['request_id', 'session_id', 'user_id', 'task_id', 'contract_id', 
                       'operation', 'component', 'error_code', 'stack_trace']
        
        for field in extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stack_trace': ''.join(traceback.format_exception(*record.exc_info))
            }
        
        return json.dumps(log_entry, default=str)


class QuantumPlannerLogger:
    """
    Specialized logger for quantum task planning with structured logging,
    performance monitoring, and audit trail capabilities.
    """
    
    def __init__(
        self, 
        name: str = "quantum_planner",
        log_level: Union[str, int] = logging.INFO,
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_structured: bool = True,
        enable_performance: bool = True,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Configuration
        self.enable_structured = enable_structured
        self.enable_performance = enable_performance
        
        # Context storage (thread-local)
        self._context = threading.local()
        
        # Performance tracking
        self._active_operations: Dict[str, PerformanceMetrics] = {}
        self._operation_lock = threading.Lock()
        
        # Setup handlers
        self._setup_handlers(log_file, enable_console, max_file_size, backup_count)
        
        # Add custom log level
        logging.addLevelName(LogLevel.AUDIT.value, "AUDIT")
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
    
    def _setup_handlers(
        self, 
        log_file: Optional[str], 
        enable_console: bool,
        max_file_size: int,
        backup_count: int
    ):
        """Setup logging handlers."""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if self.enable_structured:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            
            if self.enable_structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                ))
            
            self.logger.addHandler(file_handler)
    
    def set_context(self, context: LogContext):
        """Set logging context for current thread."""
        self._context.current_context = context
    
    def get_context(self) -> Optional[LogContext]:
        """Get current logging context."""
        return getattr(self._context, 'current_context', None)
    
    def clear_context(self):
        """Clear current logging context."""
        if hasattr(self._context, 'current_context'):
            delattr(self._context, 'current_context')
    
    @contextmanager
    def context_manager(self, context: LogContext):
        """Context manager for temporary logging context."""
        old_context = self.get_context()
        try:
            self.set_context(context)
            yield
        finally:
            if old_context:
                self.set_context(old_context)
            else:
                self.clear_context()
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        event_type: Optional[EventType] = None,
        metrics: Optional[PerformanceMetrics] = None,
        **kwargs
    ):
        """Log message with current context."""
        
        extra = kwargs.copy()
        
        # Add context
        context = self.get_context()
        if context:
            extra['context'] = context
        
        # Add event type
        if event_type:
            extra['event_type'] = event_type.value
        
        # Add performance metrics
        if metrics:
            extra['metrics'] = metrics
        
        self.logger.log(level, message, extra=extra)
    
    # Standard logging methods with context
    def trace(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log trace message."""
        self._log_with_context(LogLevel.TRACE.value, message, event_type, **kwargs)
    
    def debug(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, event_type, **kwargs)
    
    def info(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, event_type, **kwargs)
    
    def warning(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, event_type, **kwargs)
    
    def error(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, event_type, **kwargs)
    
    def critical(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, event_type, **kwargs)
    
    def audit(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log audit message."""
        self._log_with_context(LogLevel.AUDIT.value, message, event_type, **kwargs)
    
    def exception(self, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Log exception with stack trace."""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, event_type, **kwargs)
    
    # Specialized logging methods for quantum planning
    def log_task_event(
        self,
        task: QuantumTask,
        event_type: EventType,
        message: Optional[str] = None,
        level: int = logging.INFO,
        **kwargs
    ):
        """Log task-related event."""
        
        if not message:
            message = f"Task {task.id} event: {event_type.value}"
        
        # Create task context
        task_context = LogContext(
            task_id=task.id,
            operation=event_type.value,
            component="task_manager"
        )
        
        # Merge with existing context
        existing_context = self.get_context()
        if existing_context:
            for key, value in task_context.to_dict().items():
                if not getattr(existing_context, key, None):
                    setattr(existing_context, key, value)
            context = existing_context
        else:
            context = task_context
        
        # Add task details
        task_details = {
            'task_name': task.name,
            'task_state': task.state.value,
            'task_priority': task.priority,
            'task_duration': task.estimated_duration,
            'task_probability': task.probability(),
            'dependencies_count': len(task.dependencies),
            'entanglements_count': len(task.entangled_tasks)
        }
        
        kwargs.update(task_details)
        
        with self.context_manager(context):
            self._log_with_context(level, message, event_type, **kwargs)
    
    def log_optimization_metrics(
        self,
        optimization_result: Dict[str, Any],
        level: int = logging.INFO
    ):
        """Log optimization performance metrics."""
        
        message = f"Optimization completed in {optimization_result.get('optimization_time', 0):.3f}s"
        
        metrics = {
            'fitness_score': optimization_result.get('fitness_score', 0),
            'iterations': optimization_result.get('iterations', 0),
            'converged': optimization_result.get('converged', False),
            'task_count': len(optimization_result.get('task_order', [])),
            'quantum_metrics': optimization_result.get('quantum_metrics', {})
        }
        
        self._log_with_context(
            level, 
            message, 
            EventType.OPTIMIZATION_COMPLETE,
            optimization_metrics=metrics
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security-related event."""
        
        message = f"Security event: {event_type} (severity: {severity})"
        
        security_context = LogContext(
            user_id=user_id,
            operation="security_check",
            component="security_validator"
        )
        
        kwargs = {
            'security_event_type': event_type,
            'security_severity': severity,
            'security_details': details or {}
        }
        
        with self.context_manager(security_context):
            self._log_with_context(
                logging.WARNING if severity in ['medium', 'high'] else logging.ERROR,
                message,
                EventType.SECURITY_EVENT,
                **kwargs
            )
    
    def log_contract_compliance(
        self,
        contract_name: str,
        compliance_score: float,
        violations: List[Dict[str, Any]],
        level: int = logging.INFO
    ):
        """Log contract compliance information."""
        
        message = f"Contract {contract_name} compliance: {compliance_score:.3f}"
        
        if violations:
            message += f" ({len(violations)} violations)"
            level = logging.WARNING if compliance_score > 0.5 else logging.ERROR
        
        contract_context = LogContext(
            contract_id=contract_name,
            operation="compliance_check",
            component="contract_validator"
        )
        
        kwargs = {
            'compliance_score': compliance_score,
            'violations_count': len(violations),
            'violations': violations
        }
        
        with self.context_manager(contract_context):
            self._log_with_context(level, message, EventType.CONTRACT_VIOLATION, **kwargs)
    
    # Performance monitoring
    @contextmanager
    def performance_monitor(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for performance monitoring."""
        
        if not self.enable_performance:
            yield
            return
        
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        # Start monitoring
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        with self._operation_lock:
            self._active_operations[operation_id] = metrics
        
        self.debug(f"Started operation: {operation_name}", operation=operation_name)
        
        try:
            yield metrics
            metrics.finish(success=True)
            
        except Exception as e:
            metrics.finish(success=False)
            metrics.error_count += 1
            self.error(
                f"Operation {operation_name} failed: {str(e)}",
                operation=operation_name,
                exception_type=type(e).__name__
            )
            raise
            
        finally:
            # Log completion
            self.info(
                f"Completed operation: {operation_name} in {metrics.duration:.3f}s",
                event_type=EventType.PERFORMANCE_ALERT if metrics.duration > 10 else None,
                metrics=metrics,
                operation=operation_name
            )
            
            # Remove from active operations
            with self._operation_lock:
                self._active_operations.pop(operation_id, None)
    
    def get_active_operations(self) -> Dict[str, PerformanceMetrics]:
        """Get currently active operations."""
        with self._operation_lock:
            return self._active_operations.copy()
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for recent operations."""
        # This would require storing completed operations
        # For now, return active operations
        active_ops = self.get_active_operations()
        
        return {
            'active_operations': len(active_ops),
            'long_running_operations': len([
                op for op in active_ops.values()
                if time.time() - op.start_time > 60  # Over 1 minute
            ]),
            'operations_by_type': {},  # Would aggregate by operation_name
            'average_duration': 0.0,   # Would calculate from historical data
            'error_rate': 0.0          # Would calculate from historical data
        }
    
    # Utility methods
    def flush(self):
        """Flush all handlers."""
        for handler in self.logger.handlers:
            handler.flush()
    
    def close(self):
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


# Global logger instance
_global_logger: Optional[QuantumPlannerLogger] = None


def get_logger(
    name: Optional[str] = None,
    **kwargs
) -> QuantumPlannerLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        logger_name = name or "quantum_planner"
        _global_logger = QuantumPlannerLogger(logger_name, **kwargs)
    
    return _global_logger


def configure_logging(
    log_file: Optional[str] = None,
    log_level: Union[str, int] = logging.INFO,
    enable_console: bool = True,
    enable_structured: bool = True,
    enable_performance: bool = True,
    **kwargs
) -> QuantumPlannerLogger:
    """Configure global logging."""
    global _global_logger
    
    _global_logger = QuantumPlannerLogger(
        log_file=log_file,
        log_level=log_level,
        enable_console=enable_console,
        enable_structured=enable_structured,
        enable_performance=enable_performance,
        **kwargs
    )
    
    return _global_logger


# Convenience functions for common logging patterns
def log_system_start(version: str = "unknown"):
    """Log system startup."""
    logger = get_logger()
    logger.info(
        f"Quantum Task Planner starting (version: {version})",
        event_type=EventType.SYSTEM_START,
        system_version=version
    )


def log_system_stop():
    """Log system shutdown."""
    logger = get_logger()
    logger.info(
        "Quantum Task Planner shutting down",
        event_type=EventType.SYSTEM_STOP
    )


def log_task_creation(task: QuantumTask):
    """Log task creation."""
    logger = get_logger()
    logger.log_task_event(task, EventType.TASK_CREATED)


def log_task_completion(task: QuantumTask, actual_duration: Optional[float] = None):
    """Log task completion."""
    logger = get_logger()
    logger.log_task_event(
        task, 
        EventType.TASK_COMPLETED,
        level=logging.INFO,
        actual_duration=actual_duration
    )


def log_task_failure(task: QuantumTask, error_message: str):
    """Log task failure."""
    logger = get_logger()
    logger.log_task_event(
        task,
        EventType.TASK_FAILED,
        message=f"Task {task.id} failed: {error_message}",
        level=logging.ERROR,
        error_message=error_message
    )