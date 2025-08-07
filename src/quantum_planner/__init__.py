"""
Quantum-Inspired Task Planner

A quantum-inspired task planning system that leverages quantum computing principles
like superposition, entanglement, and interference to optimize task scheduling,
resource allocation, and execution planning within RLHF contract constraints.

Features comprehensive error handling, validation, security, logging, and monitoring
for production-ready quantum-inspired task planning operations.
"""

from .core import QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig
from .algorithms import QuantumOptimizer, SuperpositionSearch, EntanglementScheduler
from .contracts import ContractualTaskPlanner, TaskConstraintValidator
from .visualization import QuantumPlannerVisualizer
from .security import SecurityValidator, SecurityContext, SecurityLevel, ThreatLevel
from .validation import TaskValidator, ConfigValidator, ValidationResult, ValidationError
from .error_handling import (
    ErrorHandler, QuantumPlannerError, handle_errors, with_circuit_breaker,
    ErrorCategory, ErrorSeverity, RecoveryStrategy
)
from .logging_config import (
    get_logger, configure_logging, QuantumPlannerLogger,
    log_task_creation, log_task_completion, log_task_failure
)
from .performance import (
    OptimizedQuantumPlanner, create_optimized_planner, PerformanceLevel,
    AdaptiveCache, CacheStrategy, PerformanceProfiler, get_profiler
)
from .monitoring import (
    MonitoringSystem, get_monitoring_system, monitor_operation,
    HealthStatus, MetricType, AlertSeverity
)

__version__ = "0.1.0"

__all__ = [
    # Core functionality
    "QuantumTaskPlanner",
    "QuantumTask", 
    "TaskState",
    "PlannerConfig",
    
    # Advanced algorithms
    "QuantumOptimizer",
    "SuperpositionSearch",
    "EntanglementScheduler",
    
    # Contract integration
    "ContractualTaskPlanner",
    "TaskConstraintValidator",
    
    # Visualization
    "QuantumPlannerVisualizer",
    
    # Security
    "SecurityValidator",
    "SecurityContext", 
    "SecurityLevel",
    "ThreatLevel",
    
    # Validation
    "TaskValidator",
    "ConfigValidator",
    "ValidationResult",
    "ValidationError",
    
    # Error handling
    "ErrorHandler",
    "QuantumPlannerError",
    "handle_errors",
    "with_circuit_breaker",
    "ErrorCategory",
    "ErrorSeverity", 
    "RecoveryStrategy",
    
    # Logging
    "get_logger",
    "configure_logging",
    "QuantumPlannerLogger",
    "log_task_creation",
    "log_task_completion",
    "log_task_failure",
    
    # Performance optimization
    "OptimizedQuantumPlanner",
    "create_optimized_planner",
    "PerformanceLevel",
    "AdaptiveCache",
    "CacheStrategy",
    "PerformanceProfiler",
    "get_profiler",
    
    # Monitoring and observability
    "MonitoringSystem",
    "get_monitoring_system",
    "monitor_operation",
    "HealthStatus",
    "MetricType",
    "AlertSeverity"
]