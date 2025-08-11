"""
Optimization package for RLHF-Contract-Wizard.

Provides performance optimization, caching, and scalability enhancements.
"""

try:
    from .performance import (
        ContractCache,
        PerformanceMonitor,
        OptimizedRewardComputation,
        BatchProcessor
    )
    _PERFORMANCE_AVAILABLE = True
except ImportError:
    ContractCache = None
    PerformanceMonitor = None
    OptimizedRewardComputation = None
    BatchProcessor = None
    _PERFORMANCE_AVAILABLE = False

# Build __all__ based on availability
__all__ = []

if _PERFORMANCE_AVAILABLE:
    __all__.extend([
        'ContractCache',
        'PerformanceMonitor', 
        'OptimizedRewardComputation',
        'BatchProcessor'
    ])