"""
Reliability Module - Generation 2: Make It Robust

Provides comprehensive reliability, fault tolerance, and monitoring
capabilities for the Terragon RLHF Contract Wizard system.

Key Components:
1. Circuit breaker patterns for fault isolation
2. Exponential backoff retry mechanisms
3. Health monitoring and alerting
4. Graceful degradation strategies
5. Security monitoring and incident response

Author: Terry (Terragon Labs)
"""

from .robust_execution import (
    RobustExecutionManager,
    CircuitBreaker,
    RetryManager,
    HealthMonitor,
    HealthStatus,
    CircuitState,
    AlertSeverity,
    HealthMetrics,
    Alert,
    robust_manager,
    initialize_robust_execution,
    shutdown_robust_execution,
    robust_operation,
    execute_robustly,
    get_system_health
)

__version__ = "2.0.0"
__author__ = "Terry (Terragon Labs)"

# Reliability exports
__all__ = [
    # Core classes
    "RobustExecutionManager",
    "CircuitBreaker", 
    "RetryManager",
    "HealthMonitor",
    
    # Enums
    "HealthStatus",
    "CircuitState",
    "AlertSeverity",
    
    # Data classes
    "HealthMetrics",
    "Alert",
    
    # Global instance and functions
    "robust_manager",
    "initialize_robust_execution",
    "shutdown_robust_execution",
    "robust_operation",
    "execute_robustly",
    "get_system_health"
]