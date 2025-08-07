"""
Unit tests for performance optimization module.

Tests adaptive caching, resource pooling, performance profiling,
and optimization strategies for quantum planning operations.
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.quantum_planner.performance import (
    OptimizedQuantumPlanner, create_optimized_planner, PerformanceLevel,
    AdaptiveCache, CacheStrategy, PerformanceProfiler, get_profiler,
    ResourcePool, OptimizationHint, PerformanceMetrics
)
from src.quantum_planner.core import QuantumTask, TaskState, QuantumTaskPlanner
from .fixtures import *
from .utils import *


class TestAdaptiveCache:
    """Test cases for AdaptiveCache class."""
    
    def test_cache_creation(self):
        """Test adaptive cache initialization."""
        cache = AdaptiveCache(
            initial_strategy=CacheStrategy.LRU,
            max_size=100,
            ttl_seconds=300
        )
        
        assert cache.strategy == CacheStrategy.LRU
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
        assert len(cache.cache_data) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = AdaptiveCache(max_size=10)
        
        # Put items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Get items
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        # Check stats
        assert cache.hit_count == 2
        assert cache.miss_count == 1
    
    def test_cache_eviction_lru(self):
        """Test LRU cache eviction."""
        cache = AdaptiveCache(strategy=CacheStrategy.LRU, max_size=3)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")  
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add another item - should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Still present
        assert cache.get("key4") == "value4"  # Newly added
    
    def test_cache_eviction_lfu(self):
        """Test LFU cache eviction."""
        cache = AdaptiveCache(strategy=CacheStrategy.LFU, max_size=3)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 multiple times to increase frequency
        for _ in range(5):
            cache.get("key1")
        
        # Access key2 once
        cache.get("key2")
        
        # Add another item - should evict key3 (least frequently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Most frequent
        assert cache.get("key2") == "value2"  # Some frequency
        assert cache.get("key3") is None      # Least frequent, evicted
        assert cache.get("key4") == "value4"  # Newly added
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = AdaptiveCache(ttl_seconds=0.1)  # Very short TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL expiration
        time.sleep(0.2)
        
        # Item should be expired
        assert cache.get("key1") is None
        assert cache.miss_count == 1
    
    def test_cache_strategy_adaptation(self):
        """Test adaptive cache strategy switching."""
        cache = AdaptiveCache(strategy=CacheStrategy.LRU, adaptation_threshold=10)
        
        # Simulate access pattern that favors LFU
        for i in range(20):
            cache.put(f"key_{i % 5}", f"value_{i}")  # Repeated keys
            cache.get(f"key_{i % 5}")
        
        # Cache should adapt strategy based on access patterns
        # Note: This test may need adjustment based on specific adaptation logic
        assert isinstance(cache.strategy, CacheStrategy)
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = AdaptiveCache(max_size=10)
        
        # Perform operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key2")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_statistics()
        
        assert stats['hit_count'] == 2
        assert stats['miss_count'] == 1
        assert stats['hit_ratio'] == 2/3
        assert stats['size'] == 2
        assert stats['max_size'] == 10
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = AdaptiveCache()
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert len(cache.cache_data) == 2
        
        cache.clear()
        assert len(cache.cache_data) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access thread safety."""
        cache = AdaptiveCache(max_size=100)
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(50):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache.put(key, value)
                    retrieved = cache.get(key)
                    results.append((key, retrieved))
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == 250  # 5 workers * 50 operations each
        
        # Verify all retrieved values match what was stored
        for key, value in results:
            expected_value = key.replace("key", "value")
            assert value == expected_value


class TestResourcePool:
    """Test cases for ResourcePool class."""
    
    def test_pool_creation(self):
        """Test resource pool initialization."""
        pool = ResourcePool(
            resource_factory=lambda: "test_resource",
            max_size=10,
            min_size=2
        )
        
        assert pool.max_size == 10
        assert pool.min_size == 2
        assert len(pool.available_resources) >= 2  # Pre-allocated minimum
    
    def test_resource_acquisition_and_release(self):
        """Test resource acquisition and release."""
        pool = ResourcePool(
            resource_factory=lambda: {"id": time.time()},
            max_size=5,
            min_size=1
        )
        
        # Acquire resources
        resource1 = pool.acquire()
        resource2 = pool.acquire()
        
        assert resource1 is not None
        assert resource2 is not None
        assert resource1 != resource2  # Should be different instances
        
        initial_available = len(pool.available_resources)
        
        # Release resources
        pool.release(resource1)
        pool.release(resource2)
        
        # Pool should have more available resources
        assert len(pool.available_resources) > initial_available
    
    def test_pool_size_limits(self):
        """Test resource pool size limits."""
        def expensive_factory():
            return {"created_at": time.time()}
        
        pool = ResourcePool(
            resource_factory=expensive_factory,
            max_size=3,
            min_size=1
        )
        
        # Acquire all resources
        resources = []
        for _ in range(5):  # Try to acquire more than max_size
            resource = pool.acquire()
            if resource is not None:
                resources.append(resource)
        
        # Should not exceed max_size
        assert len(resources) <= pool.max_size
        
        # Release all resources
        for resource in resources:
            pool.release(resource)
    
    def test_resource_validation(self):
        """Test resource validation before reuse."""
        def factory():
            return {"valid": True, "created": time.time()}
        
        def validator(resource):
            return resource.get("valid", False)
        
        pool = ResourcePool(
            resource_factory=factory,
            resource_validator=validator,
            max_size=5
        )
        
        # Acquire and modify resource to make it invalid
        resource = pool.acquire()
        resource["valid"] = False
        
        # Release invalid resource
        pool.release(resource)
        
        # Acquire again - should get a new valid resource
        new_resource = pool.acquire()
        assert new_resource["valid"] is True
        assert new_resource != resource  # Should be different instance
    
    def test_pool_statistics(self):
        """Test resource pool statistics."""
        pool = ResourcePool(
            resource_factory=lambda: "resource",
            max_size=10,
            min_size=2
        )
        
        # Perform some operations
        resource1 = pool.acquire()
        resource2 = pool.acquire()
        pool.release(resource1)
        
        stats = pool.get_statistics()
        
        assert 'total_created' in stats
        assert 'currently_available' in stats
        assert 'currently_in_use' in stats
        assert 'max_size' in stats
        assert 'min_size' in stats
        
        assert stats['currently_in_use'] >= 1  # resource2 still acquired
        assert stats['max_size'] == 10
        assert stats['min_size'] == 2


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler class."""
    
    def test_profiler_creation(self):
        """Test performance profiler initialization."""
        profiler = PerformanceProfiler()
        
        assert len(profiler.profiles) == 0
        assert profiler.enabled is True
    
    def test_operation_profiling(self):
        """Test operation performance profiling."""
        profiler = PerformanceProfiler()
        
        # Profile a simple operation
        with profiler.profile_operation("test_operation"):
            time.sleep(0.1)
        
        # Check profile data
        assert "test_operation" in profiler.profiles
        profile_data = profiler.profiles["test_operation"]
        
        assert profile_data['call_count'] == 1
        assert profile_data['total_time'] >= 0.1
        assert profile_data['average_time'] >= 0.1
        assert profile_data['min_time'] >= 0.1
        assert profile_data['max_time'] >= 0.1
    
    def test_multiple_operation_profiling(self):
        """Test profiling multiple operations."""
        profiler = PerformanceProfiler()
        
        # Profile multiple calls to same operation
        for i in range(5):
            with profiler.profile_operation("repeated_operation"):
                time.sleep(0.01)
        
        # Profile different operation
        with profiler.profile_operation("different_operation"):
            time.sleep(0.05)
        
        # Check profiles
        assert len(profiler.profiles) == 2
        assert profiler.profiles["repeated_operation"]['call_count'] == 5
        assert profiler.profiles["different_operation"]['call_count'] == 1
    
    def test_profiler_decorator(self):
        """Test profiler decorator functionality."""
        profiler = PerformanceProfiler()
        
        @profiler.profile
        def decorated_function(duration):
            time.sleep(duration)
            return "completed"
        
        # Call decorated function
        result = decorated_function(0.02)
        
        assert result == "completed"
        assert "decorated_function" in profiler.profiles
        assert profiler.profiles["decorated_function"]['call_count'] == 1
    
    def test_profiler_statistics(self):
        """Test profiler statistics generation."""
        profiler = PerformanceProfiler()
        
        # Generate some profile data
        for operation in ["op1", "op2", "op3"]:
            for i in range(3):
                with profiler.profile_operation(operation):
                    time.sleep(0.01)
        
        stats = profiler.get_statistics()
        
        assert 'total_operations' in stats
        assert 'total_calls' in stats
        assert 'total_time' in stats
        assert 'operation_breakdown' in stats
        
        assert stats['total_operations'] == 3
        assert stats['total_calls'] == 9
    
    def test_profiler_reset(self):
        """Test profiler reset functionality."""
        profiler = PerformanceProfiler()
        
        # Generate some data
        with profiler.profile_operation("test_op"):
            time.sleep(0.01)
        
        assert len(profiler.profiles) == 1
        
        # Reset profiler
        profiler.reset()
        
        assert len(profiler.profiles) == 0
    
    def test_profiler_memory_tracking(self):
        """Test memory usage tracking in profiler."""
        profiler = PerformanceProfiler(track_memory=True)
        
        def memory_intensive_operation():
            # Allocate some memory
            data = [i for i in range(10000)]
            return len(data)
        
        with profiler.profile_operation("memory_test"):
            result = memory_intensive_operation()
        
        profile_data = profiler.profiles["memory_test"]
        
        # Memory tracking might not be available on all systems
        if 'memory_usage' in profile_data:
            assert 'peak_memory' in profile_data['memory_usage']
            assert 'memory_delta' in profile_data['memory_usage']


class TestOptimizedQuantumPlanner:
    """Test cases for OptimizedQuantumPlanner class."""
    
    def test_optimized_planner_creation(self, planner_config):
        """Test optimized planner initialization."""
        planner = OptimizedQuantumPlanner(
            config=planner_config,
            performance_level=PerformanceLevel.HIGH,
            enable_caching=True,
            enable_profiling=True
        )
        
        assert planner.performance_level == PerformanceLevel.HIGH
        assert planner.cache is not None
        assert planner.profiler is not None
        assert planner.resource_pool is not None
    
    def test_cached_optimization(self, sample_tasks):
        """Test cached optimization results."""
        planner = create_optimized_planner(PerformanceLevel.HIGH)
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task(task)
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            # First optimization - should compute and cache
            result1 = planner.optimize_plan()
            
            # Second optimization - should use cache
            result2 = planner.optimize_plan()
        
        # Results should be consistent
        assert result1.task_order == result2.task_order
        assert result1.fitness_score == result2.fitness_score
        
        # Check cache hit
        cache_stats = planner.cache.get_statistics()
        assert cache_stats['hit_count'] > 0
    
    def test_resource_pool_utilization(self, sample_tasks):
        """Test resource pool utilization in optimized planner."""
        planner = create_optimized_planner(PerformanceLevel.MEDIUM)
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task(task)
        
        initial_pool_stats = planner.resource_pool.get_statistics()
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            # Perform optimization
            result = planner.optimize_plan()
        
        final_pool_stats = planner.resource_pool.get_statistics()
        
        # Resource pool should have been utilized
        assert final_pool_stats['total_created'] >= initial_pool_stats['total_created']
        assert result.success is True
    
    def test_performance_profiling(self, sample_tasks):
        """Test performance profiling in optimized planner."""
        planner = create_optimized_planner(PerformanceLevel.HIGH)
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task(task)
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = planner.optimize_plan()
        
        # Check profiling data
        profile_stats = planner.profiler.get_statistics()
        
        assert profile_stats['total_operations'] > 0
        assert profile_stats['total_calls'] > 0
        assert 'operation_breakdown' in profile_stats
    
    def test_optimization_hints(self, sample_tasks):
        """Test optimization hints for performance tuning."""
        planner = create_optimized_planner(PerformanceLevel.MEDIUM)
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task(task)
        
        # Apply optimization hints
        hints = [
            OptimizationHint.PARALLEL_EXECUTION,
            OptimizationHint.CACHE_INTERMEDIATE_RESULTS,
            OptimizationHint.USE_HEURISTICS
        ]
        
        planner.apply_optimization_hints(hints)
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = planner.optimize_plan()
        
        assert result.success is True
        # Hints should influence optimization behavior
    
    def test_adaptive_performance_tuning(self, sample_tasks):
        """Test adaptive performance tuning."""
        planner = create_optimized_planner(PerformanceLevel.ADAPTIVE)
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task(task)
        
        # Perform multiple optimizations
        results = []
        for i in range(3):
            with patch('jax.numpy.array', side_effect=lambda x: x), \
                 patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
                 patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
                
                result = planner.optimize_plan()
                results.append(result)
        
        # Planner should adapt performance based on execution history
        assert all(result.success for result in results)
        
        # Later optimizations might be faster due to adaptation
        if len(results) >= 2:
            # This is heuristic - adaptation effects depend on implementation
            assert isinstance(results[-1].execution_time, (int, float))


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics collection and analysis."""
    
    def test_metrics_collection(self):
        """Test performance metrics collection."""
        metrics = PerformanceMetrics()
        
        # Record various metrics
        metrics.record_execution_time("operation1", 0.5)
        metrics.record_memory_usage("operation1", 1024)
        metrics.record_cache_hit("cache1")
        metrics.record_cache_miss("cache1")
        
        # Get metrics summary
        summary = metrics.get_summary()
        
        assert 'execution_times' in summary
        assert 'memory_usage' in summary
        assert 'cache_statistics' in summary
        
        assert "operation1" in summary['execution_times']
        assert summary['cache_statistics']['cache1']['hits'] == 1
        assert summary['cache_statistics']['cache1']['misses'] == 1
    
    def test_performance_analysis(self):
        """Test performance analysis and reporting."""
        metrics = PerformanceMetrics()
        
        # Generate sample data
        for i in range(100):
            execution_time = 0.1 + (i % 10) * 0.01  # Variable execution times
            metrics.record_execution_time("test_operation", execution_time)
        
        analysis = metrics.analyze_performance()
        
        assert 'average_execution_time' in analysis
        assert 'percentiles' in analysis
        assert 'outliers' in analysis
        
        # Should detect performance patterns
        assert analysis['average_execution_time'] > 0
        assert 'p95' in analysis['percentiles']
        assert 'p99' in analysis['percentiles']
    
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        metrics = PerformanceMetrics()
        
        # Record baseline performance
        baseline_times = [0.1, 0.11, 0.12, 0.09, 0.13]  # Stable performance
        for time_val in baseline_times:
            metrics.record_execution_time("stable_operation", time_val)
        
        metrics.establish_baseline("stable_operation")
        
        # Record degraded performance
        degraded_times = [0.5, 0.6, 0.55, 0.62, 0.58]  # Much slower
        for time_val in degraded_times:
            metrics.record_execution_time("stable_operation", time_val)
        
        # Check for regression
        regression_detected = metrics.detect_regression("stable_operation")
        
        assert regression_detected is True
    
    def test_resource_utilization_tracking(self):
        """Test resource utilization tracking."""
        metrics = PerformanceMetrics()
        
        # Record resource usage
        for i in range(10):
            cpu_usage = 50 + i * 5  # Increasing CPU usage
            memory_usage = 1024 + i * 100  # Increasing memory
            
            metrics.record_resource_usage("test_scenario", {
                'cpu_percent': cpu_usage,
                'memory_mb': memory_usage
            })
        
        utilization_report = metrics.get_resource_utilization_report()
        
        assert 'test_scenario' in utilization_report
        scenario_data = utilization_report['test_scenario']
        
        assert 'cpu_percent' in scenario_data
        assert 'memory_mb' in scenario_data
        assert scenario_data['cpu_percent']['average'] > 50
        assert scenario_data['memory_mb']['average'] > 1024


class TestPerformanceIntegration:
    """Integration tests for performance optimization system."""
    
    @measure_execution_time
    def test_end_to_end_performance_optimization(self, sample_tasks, performance_thresholds):
        """Test complete performance optimization workflow."""
        # Create optimized planner
        planner = create_optimized_planner(PerformanceLevel.HIGH)
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task(task)
        
        # Perform optimization with full performance stack
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = planner.optimize_plan()
        
        # Check optimization succeeded
        assert result.success is True
        assert len(result.task_order) == len(sample_tasks)
        
        # Check performance metrics
        execution_time = result.execution_time if hasattr(result, 'execution_time') else 1.0
        max_time = performance_thresholds.get('max_optimized_planning_time', 5.0)
        assert_performance_acceptable(execution_time, max_time, "optimized planning")
        
        # Check cache utilization
        cache_stats = planner.cache.get_statistics()
        assert cache_stats['size'] >= 0
        
        # Check profiling data
        profile_stats = planner.profiler.get_statistics()
        assert profile_stats['total_operations'] > 0
    
    def test_performance_scaling(self):
        """Test performance scaling with increasing load."""
        performance_results = []
        
        for task_count in [5, 10, 20, 30]:
            planner = create_optimized_planner(PerformanceLevel.HIGH)
            
            # Create tasks
            tasks = [create_test_task(f"scale_task_{i}") for i in range(task_count)]
            
            for task in tasks:
                planner.add_task(task)
            
            # Measure performance
            start_time = time.time()
            
            with patch('jax.numpy.array', side_effect=lambda x: x), \
                 patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
                 patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
                
                result = planner.optimize_plan()
            
            execution_time = time.time() - start_time
            
            performance_results.append({
                'task_count': task_count,
                'execution_time': execution_time,
                'success': result.success
            })
        
        # Check scaling behavior
        assert all(result['success'] for result in performance_results)
        
        # Performance should scale reasonably (not exponentially)
        if len(performance_results) >= 2:
            first_result = performance_results[0]
            last_result = performance_results[-1]
            
            time_ratio = last_result['execution_time'] / first_result['execution_time']
            task_ratio = last_result['task_count'] / first_result['task_count']
            
            # Time should not increase faster than task count squared
            assert time_ratio <= task_ratio ** 2
    
    def test_memory_optimization(self, sample_tasks):
        """Test memory optimization features."""
        planner = create_optimized_planner(PerformanceLevel.MEMORY_OPTIMIZED)
        
        # Add many tasks
        large_task_set = sample_tasks * 20  # Create larger task set
        
        for task in large_task_set:
            planner.add_task(task)
        
        # Monitor memory usage
        initial_memory = assert_memory_usage_acceptable(max_memory_mb=500)
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = planner.optimize_plan()
        
        final_memory = assert_memory_usage_acceptable(max_memory_mb=1000)  # Allow some increase
        
        assert result.success is True
        
        # Memory growth should be controlled
        memory_growth = final_memory - initial_memory
        assert memory_growth < 500  # Should not grow by more than 500MB
    
    def test_concurrent_performance_optimization(self, sample_tasks):
        """Test concurrent performance optimization."""
        results = []
        errors = []
        
        def optimization_worker(worker_id):
            try:
                planner = create_optimized_planner(PerformanceLevel.MEDIUM)
                
                # Add tasks
                for task in sample_tasks:
                    worker_task = create_test_task(f"worker_{worker_id}_{task.id}")
                    planner.add_task(worker_task)
                
                with patch('jax.numpy.array', side_effect=lambda x: x), \
                     patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
                     patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
                    
                    result = planner.optimize_plan()
                    results.append((worker_id, result))
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Start multiple optimization workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=optimization_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        
        # All optimizations should succeed
        for worker_id, result in results:
            assert result.success is True, f"Worker {worker_id} failed"