"""
Quantum-Inspired Task Planner

A quantum-inspired task planning system that leverages quantum computing principles
like superposition, entanglement, and interference to optimize task scheduling,
resource allocation, and execution planning within RLHF contract constraints.

Features comprehensive error handling, validation, security, logging, and monitoring
for production-ready quantum-inspired task planning operations.
"""

from .core import QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig

try:
    from .algorithms import QuantumOptimizer, SuperpositionSearch, EntanglementScheduler
    _has_algorithms = True
except ImportError:
    _has_algorithms = False

try:
    from .contracts import ContractualTaskPlanner, TaskConstraintValidator
    _has_contracts = True
except ImportError:
    _has_contracts = False

try:
    from .visualization import QuantumPlannerVisualizer
    _has_visualization = True
except ImportError:
    _has_visualization = False

try:
    from .security import SecurityValidator, SecurityContext, SecurityLevel, ThreatLevel
    _has_security = True
except ImportError:
    _has_security = False

try:
    from .validation import TaskValidator, ConfigValidator, ValidationResult, ValidationError
    _has_validation = True
except ImportError:
    _has_validation = False

try:
    from .error_handling import (
        ErrorHandler, QuantumPlannerError, handle_errors, with_circuit_breaker,
        ErrorCategory, ErrorSeverity, RecoveryStrategy
    )
    _has_error_handling = True
except ImportError:
    _has_error_handling = False

try:
    from .logging_config import (
        get_logger, configure_logging, QuantumPlannerLogger,
        log_task_creation, log_task_completion, log_task_failure
    )
    _has_logging = True
except ImportError:
    _has_logging = False

try:
    from .performance import (
        OptimizedQuantumPlanner, create_optimized_planner, PerformanceLevel,
        AdaptiveCache, CacheStrategy, PerformanceProfiler, get_profiler
    )
    _has_performance = True
except ImportError:
    _has_performance = False

try:
    from .monitoring import (
        MonitoringSystem, get_monitoring_system, monitor_operation,
        HealthStatus, MetricType, AlertSeverity
    )
    _has_monitoring = True
except ImportError:
    _has_monitoring = False

__version__ = "0.1.0"

# Build __all__ dynamically based on what imported successfully
__all__ = [
    # Core functionality (always available)
    "QuantumTaskPlanner",
    "QuantumTask", 
    "TaskState",
    "PlannerConfig",
]

# Add optional components
if _has_algorithms:
    __all__.extend([
        "QuantumOptimizer",
        "SuperpositionSearch", 
        "EntanglementScheduler",
    ])

if _has_contracts:
    __all__.extend([
        "ContractualTaskPlanner",
        "TaskConstraintValidator",
    ])

if _has_visualization:
    __all__.append("QuantumPlannerVisualizer")

if _has_security:
    __all__.extend([
        "SecurityValidator",
        "SecurityContext", 
        "SecurityLevel",
        "ThreatLevel",
    ])

if _has_validation:
    __all__.extend([
        "TaskValidator",
        "ConfigValidator",
        "ValidationResult",
        "ValidationError",
    ])

if _has_error_handling:
    __all__.extend([
        "ErrorHandler",
        "QuantumPlannerError",
        "handle_errors",
        "with_circuit_breaker",
        "ErrorCategory",
        "ErrorSeverity", 
        "RecoveryStrategy",
    ])

if _has_logging:
    __all__.extend([
        "get_logger",
        "configure_logging",
        "QuantumPlannerLogger",
        "log_task_creation",
        "log_task_completion",
        "log_task_failure",
    ])

if _has_performance:
    __all__.extend([
        "OptimizedQuantumPlanner",
        "create_optimized_planner",
        "PerformanceLevel",
        "AdaptiveCache",
        "CacheStrategy",
        "PerformanceProfiler",
        "get_profiler",
    ])

if _has_monitoring:
    __all__.extend([
        "MonitoringSystem",
        "get_monitoring_system",
        "monitor_operation",
        "HealthStatus",
        "MetricType",
        "AlertSeverity"
    ])