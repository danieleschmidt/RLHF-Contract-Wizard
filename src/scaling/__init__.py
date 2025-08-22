"""
Scaling Module - Generation 3: Make It Scale

Provides comprehensive performance optimization, caching strategies,
auto-scaling capabilities, and resource management for the Terragon
RLHF Contract Wizard system.

Key Components:
1. Advanced multi-level caching with adaptive eviction
2. Connection pooling and resource management
3. Asynchronous batch processing
4. Auto-scaling based on metrics
5. Performance monitoring and optimization
6. Load balancing and distributed processing

Author: Terry (Terragon Labs)
"""

from .performance_optimization import (
    PerformanceOptimizer,
    AdvancedCache,
    ConnectionPool,
    BatchProcessor,
    AutoScaler,
    CacheStrategy,
    ScalingStrategy,
    CacheEntry,
    PerformanceMetrics,
    performance_optimizer,
    initialize_performance_optimization,
    shutdown_performance_optimization,
    get_performance_stats
)

__version__ = "3.0.0"
__author__ = "Terry (Terragon Labs)"

# Scaling exports
__all__ = [
    # Core classes
    "PerformanceOptimizer",
    "AdvancedCache",
    "ConnectionPool", 
    "BatchProcessor",
    "AutoScaler",
    
    # Enums
    "CacheStrategy",
    "ScalingStrategy",
    
    # Data classes
    "CacheEntry",
    "PerformanceMetrics",
    
    # Global instance and functions
    "performance_optimizer",
    "initialize_performance_optimization",
    "shutdown_performance_optimization",
    "get_performance_stats"
]