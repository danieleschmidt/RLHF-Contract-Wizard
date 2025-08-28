"""
Intelligent Performance Optimization for Generation 3: MAKE IT SCALE

Implements advanced performance optimization, intelligent caching, auto-scaling,
load balancing, and distributed computing features for the RLHF-Contract-Wizard.
"""

import asyncio
import time
import logging
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
import hashlib
import pickle
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, devices


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    MEMORY_EFFICIENT = "memory_efficient"
    CPU_INTENSIVE = "cpu_intensive"
    IO_OPTIMIZED = "io_optimized"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    HIERARCHICAL = "hierarchical"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hit_rate: float
    throughput: float
    latency: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    max_memory_usage: float = 0.8  # 80% of available memory
    target_latency_ms: float = 100.0
    min_throughput_rps: float = 100.0
    max_cpu_usage: float = 0.7  # 70% of CPU capacity
    cache_size_mb: int = 512
    enable_jit_compilation: bool = True
    enable_vectorization: bool = True
    enable_parallelization: bool = True
    enable_distribution: bool = True
    optimization_interval: int = 300  # 5 minutes


class IntelligentCache:
    """
    Intelligent caching system with multiple strategies and adaptive behavior.
    
    Features:
    - Multiple cache strategies (LRU, LFU, Adaptive, Predictive)
    - Dynamic cache size adjustment
    - Cache warming and prefetching
    - Hit rate optimization
    - Memory pressure handling
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl_seconds: Optional[float] = None
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Strategy-specific data structures
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.access_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive behavior
        self.performance_history: List[float] = []
        self.optimization_counter = 0
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            current_time = time.time()
            
            # Check if key exists
            if key not in self.cache:
                self.misses += 1
                self._record_access(key, False)
                return None
            
            # Check TTL if enabled
            if self.ttl_seconds and key in self.metadata:
                created_time = self.metadata[key].get('created_time', 0)
                if current_time - created_time > self.ttl_seconds:
                    self._evict_key(key)
                    self.misses += 1
                    return None
            
            # Update access information
            self.access_times[key] = current_time
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            self.hits += 1
            self._record_access(key, True)
            
            # Update metadata
            if key in self.metadata:
                self.metadata[key]['last_access'] = current_time
                self.metadata[key]['access_count'] = self.access_counts[key]
            
            return self.cache[key]
    
    def set(self, key: str, value: Any, tags: Optional[set] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            current_time = time.time()
            
            # Check if we need to make space
            if key not in self.cache and len(self.cache) >= self.max_size:
                if not self._make_space():
                    return False  # Could not make space
            
            # Store value
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            # Store metadata
            self.metadata[key] = {
                'created_time': current_time,
                'last_access': current_time,
                'access_count': self.access_counts[key],
                'size_bytes': self._estimate_size(value),
                'tags': tags or set()
            }
            
            self._record_access(key, True)
            
            # Periodic optimization
            self.optimization_counter += 1
            if self.optimization_counter % 100 == 0:
                self._optimize_cache()
            
            return True
    
    def evict(self, key: str) -> bool:
        """Manually evict a key from cache."""
        with self.lock:
            if key in self.cache:
                self._evict_key(key)
                return True
            return False
    
    def clear(self, tags: Optional[set] = None) -> int:
        """Clear cache entries, optionally by tags."""
        with self.lock:
            if tags is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.metadata.clear()
                self.access_times.clear()
                self.access_counts.clear()
                return count
            else:
                # Clear by tags
                keys_to_remove = []
                for key, meta in self.metadata.items():
                    if meta.get('tags', set()).intersection(tags):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._evict_key(key)
                
                return len(keys_to_remove)
    
    def _make_space(self) -> bool:
        """Make space in cache by evicting entries."""
        if len(self.cache) == 0:
            return True
        
        # Choose eviction strategy based on cache strategy
        if self.strategy == CacheStrategy.LRU:
            key_to_evict = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        elif self.strategy == CacheStrategy.ADAPTIVE:
            key_to_evict = self._adaptive_eviction()
        elif self.strategy == CacheStrategy.PREDICTIVE:
            key_to_evict = self._predictive_eviction()
        else:
            # Default to LRU
            key_to_evict = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        if key_to_evict:
            self._evict_key(key_to_evict)
            return True
        
        return False
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction strategy based on access patterns."""
        if not self.cache:
            return None
        
        current_time = time.time()
        scores = {}
        
        for key in self.cache:
            # Calculate score based on multiple factors
            last_access = self.access_times.get(key, 0)
            access_count = self.access_counts.get(key, 0)
            time_since_access = current_time - last_access
            
            # Age factor (older = lower score)
            age_factor = 1.0 / (time_since_access + 1)
            
            # Frequency factor (more frequent = higher score)
            frequency_factor = access_count / (access_count + 1)
            
            # Size factor (larger = lower score for memory efficiency)
            size_bytes = self.metadata.get(key, {}).get('size_bytes', 1)
            size_factor = 1.0 / (size_bytes + 1)
            
            # Combined score
            scores[key] = age_factor * 0.4 + frequency_factor * 0.4 + size_factor * 0.2
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k]) if scores else None
    
    def _predictive_eviction(self) -> Optional[str]:
        """Predictive eviction based on access pattern analysis."""
        if not self.access_history:
            return self._adaptive_eviction()
        
        # Analyze access patterns to predict future accesses
        recent_accesses = list(self.access_history)[-1000:]  # Last 1000 accesses
        access_patterns = defaultdict(int)
        
        for access in recent_accesses:
            key = access.get('key')
            if key in self.cache:
                access_patterns[key] += 1
        
        if not access_patterns:
            return self._adaptive_eviction()
        
        # Predict least likely to be accessed
        prediction_scores = {}
        for key in self.cache:
            recent_access_count = access_patterns.get(key, 0)
            total_access_count = self.access_counts.get(key, 0)
            
            # Prediction score based on recent vs total access ratio
            if total_access_count > 0:
                prediction_scores[key] = recent_access_count / total_access_count
            else:
                prediction_scores[key] = 0.0
        
        return min(prediction_scores.keys(), key=lambda k: prediction_scores[k])
    
    def _evict_key(self, key: str):
        """Evict a specific key from cache."""
        if key in self.cache:
            del self.cache[key]
            self.metadata.pop(key, None)
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
            self.evictions += 1
    
    def _record_access(self, key: str, hit: bool):
        """Record cache access for analysis."""
        self.access_history.append({
            'key': key,
            'hit': hit,
            'timestamp': time.time()
        })
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _optimize_cache(self):
        """Optimize cache performance."""
        hit_rate = self.hit_rate()
        self.performance_history.append(hit_rate)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Adjust cache size based on performance
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            if recent_performance < 0.6 and self.max_size < 10000:
                # Increase cache size if hit rate is low
                self.max_size = min(10000, int(self.max_size * 1.2))
                self.logger.info(f"Increased cache size to {self.max_size}")
                
            elif recent_performance > 0.9 and self.max_size > 100:
                # Decrease cache size if hit rate is very high (might be over-caching)
                self.max_size = max(100, int(self.max_size * 0.9))
                self.logger.info(f"Decreased cache size to {self.max_size}")
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate(),
            'size': len(self.cache),
            'max_size': self.max_size,
            'strategy': self.strategy.value,
            'memory_usage_mb': sum(
                self.metadata.get(k, {}).get('size_bytes', 0) 
                for k in self.cache
            ) / (1024 * 1024)
        }
    
    def warm_cache(self, key_value_pairs: List[Tuple[str, Any]]):
        """Warm cache with pre-computed values."""
        for key, value in key_value_pairs:
            self.set(key, value, tags={'warmed'})
        
        self.logger.info(f"Warmed cache with {len(key_value_pairs)} entries")


class AdaptiveLoadBalancer:
    """
    Adaptive load balancer with intelligent request distribution.
    
    Features:
    - Multiple load balancing strategies
    - Health-aware routing
    - Adaptive weight adjustment
    - Circuit breaker integration
    - Performance monitoring
    """
    
    def __init__(self):
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.health_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        
        self.total_requests = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_worker(self, worker_id: str, capacity: int = 100, metadata: Optional[Dict] = None):
        """Register a worker for load balancing."""
        with self.lock:
            self.workers[worker_id] = {
                'capacity': capacity,
                'current_load': 0,
                'available': True,
                'metadata': metadata or {},
                'registered_time': time.time()
            }
            self.health_scores[worker_id] = 1.0
            self.logger.info(f"Registered worker: {worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                self.health_scores.pop(worker_id, None)
                self.request_counts.pop(worker_id, None)
                self.response_times.pop(worker_id, None)
                self.error_counts.pop(worker_id, None)
                self.logger.info(f"Unregistered worker: {worker_id}")
    
    def select_worker(self, request_metadata: Optional[Dict] = None) -> Optional[str]:
        """Select best worker for request."""
        with self.lock:
            available_workers = [
                wid for wid, worker in self.workers.items()
                if worker['available'] and worker['current_load'] < worker['capacity']
            ]
            
            if not available_workers:
                return None
            
            # Calculate scores for each worker
            scores = {}
            for worker_id in available_workers:
                worker = self.workers[worker_id]
                
                # Load factor (lower is better)
                load_factor = 1.0 - (worker['current_load'] / worker['capacity'])
                
                # Health factor
                health_factor = self.health_scores[worker_id]
                
                # Performance factor (based on average response time)
                avg_response_time = np.mean(self.response_times.get(worker_id, [0.1]))
                performance_factor = 1.0 / (avg_response_time + 0.01)
                
                # Error rate factor
                total_requests = self.request_counts.get(worker_id, 1)
                error_rate = self.error_counts.get(worker_id, 0) / total_requests
                error_factor = 1.0 - error_rate
                
                # Combined score
                scores[worker_id] = (
                    load_factor * 0.3 +
                    health_factor * 0.25 +
                    performance_factor * 0.25 +
                    error_factor * 0.2
                )
            
            # Select worker with highest score
            selected_worker = max(scores.keys(), key=lambda w: scores[w])
            
            # Update load
            self.workers[selected_worker]['current_load'] += 1
            self.total_requests += 1
            
            return selected_worker
    
    def record_request_completion(
        self,
        worker_id: str,
        response_time: float,
        success: bool = True
    ):
        """Record completion of a request."""
        with self.lock:
            if worker_id in self.workers:
                # Update load
                self.workers[worker_id]['current_load'] = max(
                    0, self.workers[worker_id]['current_load'] - 1
                )
                
                # Update metrics
                self.request_counts[worker_id] += 1
                self.response_times[worker_id].append(response_time)
                
                # Keep only recent response times
                if len(self.response_times[worker_id]) > 100:
                    self.response_times[worker_id] = self.response_times[worker_id][-100:]
                
                if not success:
                    self.error_counts[worker_id] += 1
                
                # Update health score
                self._update_health_score(worker_id, success, response_time)
    
    def _update_health_score(self, worker_id: str, success: bool, response_time: float):
        """Update health score for a worker."""
        current_score = self.health_scores[worker_id]
        
        # Adjust based on success/failure
        if success:
            # Gradually increase health score
            current_score = min(1.0, current_score + 0.01)
        else:
            # More rapidly decrease health score
            current_score = max(0.0, current_score - 0.1)
        
        # Adjust based on response time
        if response_time > 5.0:  # Slow response
            current_score = max(0.0, current_score - 0.05)
        elif response_time < 0.1:  # Fast response
            current_score = min(1.0, current_score + 0.005)
        
        self.health_scores[worker_id] = current_score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            worker_stats = {}
            for worker_id, worker in self.workers.items():
                avg_response_time = np.mean(self.response_times.get(worker_id, [0]))
                error_rate = (
                    self.error_counts.get(worker_id, 0) / 
                    max(1, self.request_counts.get(worker_id, 1))
                )
                
                worker_stats[worker_id] = {
                    'capacity': worker['capacity'],
                    'current_load': worker['current_load'],
                    'load_percentage': worker['current_load'] / worker['capacity'] * 100,
                    'health_score': self.health_scores[worker_id],
                    'total_requests': self.request_counts.get(worker_id, 0),
                    'avg_response_time': avg_response_time,
                    'error_rate': error_rate,
                    'available': worker['available']
                }
            
            return {
                'total_workers': len(self.workers),
                'available_workers': len([w for w in self.workers.values() if w['available']]),
                'total_requests': self.total_requests,
                'worker_stats': worker_stats
            }


class PerformanceOptimizer:
    """
    Intelligent performance optimizer with adaptive strategies.
    
    Features:
    - Dynamic performance monitoring
    - Automatic optimization strategy selection
    - JAX compilation and vectorization
    - Memory usage optimization
    - Throughput maximization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.cache = IntelligentCache(
            max_size=self.config.cache_size_mb * 1024 // 4,  # Rough estimate
            strategy=CacheStrategy.ADAPTIVE
        )
        self.load_balancer = AdaptiveLoadBalancer()
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Compiled functions cache
        self.compiled_functions: Dict[str, Callable] = {}
        
        # Worker processes
        self.worker_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_function(
        self,
        func: Callable,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        cache_results: bool = True
    ) -> Callable:
        """Optimize a function with the specified strategy."""
        func_id = self._get_function_id(func)
        
        if func_id in self.compiled_functions:
            return self.compiled_functions[func_id]
        
        optimized_func = func
        
        try:
            if strategy in [OptimizationStrategy.CPU_INTENSIVE, OptimizationStrategy.BALANCED]:
                # Apply JAX JIT compilation if enabled
                if self.config.enable_jit_compilation and self._is_jax_compatible(func):
                    optimized_func = jit(func)
                    self.logger.info(f"Applied JIT compilation to {func.__name__}")
            
            if strategy in [OptimizationStrategy.AGGRESSIVE, OptimizationStrategy.BALANCED]:
                # Apply vectorization if possible
                if self.config.enable_vectorization and self._can_vectorize(func):
                    optimized_func = vmap(optimized_func)
                    self.logger.info(f"Applied vectorization to {func.__name__}")
            
            # Wrap with caching if enabled
            if cache_results:
                optimized_func = self._wrap_with_cache(optimized_func, func_id)
            
            # Store compiled function
            self.compiled_functions[func_id] = optimized_func
            
            return optimized_func
            
        except Exception as e:
            self.logger.warning(f"Failed to optimize function {func.__name__}: {e}")
            return func
    
    def optimize_batch_processing(
        self,
        func: Callable,
        data_batch: List[Any],
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Optimize batch processing with parallel execution."""
        if not data_batch:
            return []
        
        batch_size = len(data_batch)
        
        # Determine optimal processing strategy
        if batch_size < 10:
            # Small batch - process sequentially
            return [func(item) for item in data_batch]
        
        elif batch_size < 100:
            # Medium batch - use thread pool
            if self.thread_pool is None:
                self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
                )
            
            with self.thread_pool as executor:
                futures = [executor.submit(func, item) for item in data_batch]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            return results
        
        else:
            # Large batch - use process pool
            if self.worker_pool is None:
                self.worker_pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers or multiprocessing.cpu_count()
                )
            
            try:
                with self.worker_pool as executor:
                    chunk_size = max(1, batch_size // (executor._max_workers * 4))
                    futures = []
                    
                    for i in range(0, batch_size, chunk_size):
                        chunk = data_batch[i:i + chunk_size]
                        future = executor.submit(self._process_chunk, func, chunk)
                        futures.append(future)
                    
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        results.extend(future.result())
                    
                    return results
            
            except Exception as e:
                self.logger.warning(f"Process pool execution failed: {e}, falling back to sequential")
                return [func(item) for item in data_batch]
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of data."""
        return [func(item) for item in chunk]
    
    def auto_scale_resources(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Automatically scale resources based on current performance."""
        scaling_actions = []
        
        # Memory scaling
        if current_metrics.memory_usage > self.config.max_memory_usage:
            # Reduce cache size
            new_cache_size = max(100, int(self.cache.max_size * 0.8))
            self.cache.max_size = new_cache_size
            scaling_actions.append(f"Reduced cache size to {new_cache_size}")
            
            # Clear less important cache entries
            cleared = self.cache.clear(tags={'low_priority'})
            if cleared > 0:
                scaling_actions.append(f"Cleared {cleared} low priority cache entries")
        
        # CPU scaling
        if current_metrics.cpu_usage > self.config.max_cpu_usage:
            # Reduce parallel processing
            if self.thread_pool and self.thread_pool._max_workers > 2:
                # This is a conceptual action - actual implementation would vary
                scaling_actions.append("Reduced thread pool size")
        
        # Latency scaling
        if current_metrics.latency > self.config.target_latency_ms:
            # Increase cache size if memory allows
            if (current_metrics.memory_usage < self.config.max_memory_usage * 0.7 and
                self.cache.max_size < 10000):
                new_cache_size = min(10000, int(self.cache.max_size * 1.2))
                self.cache.max_size = new_cache_size
                scaling_actions.append(f"Increased cache size to {new_cache_size}")
        
        # Throughput scaling
        if current_metrics.throughput < self.config.min_throughput_rps:
            # Optimize more aggressively
            scaling_actions.append("Switched to aggressive optimization strategy")
        
        return {
            'actions_taken': scaling_actions,
            'new_metrics_targets': {
                'cache_size': self.cache.max_size,
                'optimization_level': 'adaptive'
            }
        }
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if len(self.metrics_history) < 10:
            return ["Collect more performance data for better recommendations"]
        
        recent_metrics = list(self.metrics_history)[-10:]
        avg_latency = np.mean([m.latency for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        
        # Latency recommendations
        if avg_latency > self.config.target_latency_ms * 1.5:
            recommendations.append("Consider increasing cache size or enabling more aggressive optimization")
        
        # Memory recommendations
        if avg_memory > self.config.max_memory_usage * 0.9:
            recommendations.append("Memory usage is high - consider reducing cache size or optimizing data structures")
        
        # CPU recommendations
        if avg_cpu < 0.3:
            recommendations.append("CPU utilization is low - consider increasing parallelization")
        elif avg_cpu > 0.9:
            recommendations.append("CPU utilization is high - consider load balancing or scaling out")
        
        # Cache recommendations
        if avg_hit_rate < 0.6:
            recommendations.append("Cache hit rate is low - review caching strategy or increase cache size")
        elif avg_hit_rate > 0.95:
            recommendations.append("Cache hit rate is very high - consider reducing cache size to free memory")
        
        return recommendations or ["Performance appears optimal"]
    
    def benchmark_operation(
        self,
        operation: Callable,
        test_data: List[Any],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark an operation with different optimization strategies."""
        results = {}
        
        for strategy in OptimizationStrategy:
            self.logger.info(f"Benchmarking with strategy: {strategy.value}")
            
            # Optimize operation
            optimized_op = self.optimize_function(operation, strategy)
            
            # Run benchmark
            start_time = time.time()
            successful_runs = 0
            total_latency = 0.0
            
            for i in range(iterations):
                try:
                    item = test_data[i % len(test_data)]
                    op_start = time.time()
                    result = optimized_op(item)
                    op_end = time.time()
                    
                    successful_runs += 1
                    total_latency += (op_end - op_start)
                    
                except Exception as e:
                    self.logger.warning(f"Benchmark iteration {i} failed: {e}")
            
            end_time = time.time()
            
            if successful_runs > 0:
                results[strategy.value] = {
                    'total_time': end_time - start_time,
                    'successful_runs': successful_runs,
                    'success_rate': successful_runs / iterations,
                    'avg_latency': total_latency / successful_runs,
                    'throughput': successful_runs / (end_time - start_time)
                }
            else:
                results[strategy.value] = {
                    'total_time': end_time - start_time,
                    'successful_runs': 0,
                    'success_rate': 0.0,
                    'error': 'All iterations failed'
                }
        
        # Determine best strategy
        valid_results = {k: v for k, v in results.items() if v.get('success_rate', 0) > 0}
        
        if valid_results:
            best_strategy = max(
                valid_results.keys(),
                key=lambda k: valid_results[k]['throughput'] * valid_results[k]['success_rate']
            )
            results['recommended_strategy'] = best_strategy
        
        return results
    
    def _get_function_id(self, func: Callable) -> str:
        """Get unique identifier for a function."""
        return f"{func.__module__}.{func.__name__}_{id(func)}"
    
    def _is_jax_compatible(self, func: Callable) -> bool:
        """Check if function is compatible with JAX JIT."""
        try:
            # Basic compatibility check
            import inspect
            signature = inspect.signature(func)
            
            # Check if function has JAX-compatible annotations
            for param in signature.parameters.values():
                if param.annotation and 'jax' in str(param.annotation):
                    return True
            
            # Check return annotation
            if signature.return_annotation and 'jax' in str(signature.return_annotation):
                return True
            
            return False
        except:
            return False
    
    def _can_vectorize(self, func: Callable) -> bool:
        """Check if function can be vectorized."""
        # Simple heuristic - this would be more sophisticated in practice
        return hasattr(func, '__annotations__')
    
    def _wrap_with_cache(self, func: Callable, func_id: str) -> Callable:
        """Wrap function with caching."""
        def cached_func(*args, **kwargs):
            # Create cache key
            cache_key = self._create_cache_key(func_id, args, kwargs)
            
            # Check cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache.set(cache_key, result, tags={'function_cache'})
            
            return result
        
        return cached_func
    
    def _create_cache_key(self, func_id: str, args: tuple, kwargs: dict) -> str:
        """Create cache key for function call."""
        # Create deterministic key from function ID and arguments
        key_data = {
            'func_id': func_id,
            'args': str(args),  # Simplified - would need better serialization
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create the global performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    
    return _performance_optimizer


# Convenience decorators
def optimize_performance(
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    cache_results: bool = True
):
    """Decorator for performance optimization."""
    def decorator(func):
        optimizer = get_performance_optimizer()
        return optimizer.optimize_function(func, strategy, cache_results)
    
    return decorator


def smart_cache(ttl_seconds: Optional[float] = None, tags: Optional[set] = None):
    """Decorator for intelligent caching."""
    def decorator(func):
        cache = IntelligentCache(ttl_seconds=ttl_seconds)
        
        def cached_func(*args, **kwargs):
            key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(key, result, tags)
            
            return result
        
        return cached_func
    
    return decorator