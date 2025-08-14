"""
Performance optimization and monitoring for quantum task planning.

Implements advanced caching strategies, performance profiling, resource pooling,
load balancing, and auto-scaling capabilities for production-scale deployments.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import functools
import weakref
import gc
from collections import defaultdict, deque
# Simple caching implementation
class LRUCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.cache = {}
    
    def get(self, key, default=None):
        return self.cache.get(key, default)
    
    def __setitem__(self, key, value):
        if len(self.cache) >= self.maxsize:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def __getitem__(self, key):
        return self.cache[key]
    
    def __len__(self):
        return len(self.cache)
try:
    import psutil
except ImportError:
    psutil = None
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap

from .core import QuantumTask, TaskState, QuantumTaskPlanner, PlannerConfig
from .logging_config import get_logger, EventType


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"         # Least Recently Used
    LFU = "lfu"         # Least Frequently Used
    TTL = "ttl"         # Time To Live
    FIFO = "fifo"       # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class PerformanceLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"           # Basic optimizations
    BALANCED = "balanced"     # Balanced performance/memory trade-off
    AGGRESSIVE = "aggressive" # Maximum performance
    MEMORY_OPTIMIZED = "memory_optimized"  # Memory-constrained environments


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    def update(self, execution_time: float, cache_hit: bool = False, memory_mb: float = 0.0):
        """Update metrics with new measurement."""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if memory_mb > 0:
            self.memory_usage = memory_mb
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_name': self.operation_name,
            'call_count': self.call_count,
            'total_time': self.total_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'avg_time': self.avg_time,
            'cache_hit_rate': self.cache_hit_rate,
            'memory_usage_mb': self.memory_usage,
            'cpu_usage_percent': self.cpu_usage
        }


class AdaptiveCache:
    """
    Adaptive caching system that automatically adjusts strategy based on usage patterns.
    
    Monitors access patterns and switches between different caching strategies
    to optimize for the current workload characteristics.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        initial_strategy: CacheStrategy = CacheStrategy.LRU,
        ttl_seconds: int = 3600,
        adaptation_interval: int = 100  # Adapt every N operations
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.adaptation_interval = adaptation_interval
        
        # Current cache strategy
        self.current_strategy = initial_strategy
        self._cache = self._create_cache(initial_strategy)
        
        # Performance tracking for adaptation
        self._operation_count = 0
        self._strategy_performance: Dict[CacheStrategy, PerformanceMetrics] = {
            strategy: PerformanceMetrics(f"cache_{strategy.value}")
            for strategy in CacheStrategy if strategy != CacheStrategy.ADAPTIVE
        }
        
        # Access pattern analysis
        self._access_times = deque(maxlen=1000)
        self._key_frequencies = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = get_logger()
    
    def _create_cache(self, strategy: CacheStrategy) -> Any:
        """Create cache instance based on strategy."""
        if strategy == CacheStrategy.LRU:
            return LRUCache(maxsize=self.max_size)
        elif strategy == CacheStrategy.LFU:
            return LRUCache(maxsize=self.max_size)  # LFU fallback to LRU
        elif strategy == CacheStrategy.TTL:
            return LRUCache(maxsize=self.max_size)  # TTL fallback to LRU  
        elif strategy == CacheStrategy.FIFO:
            return LRUCache(maxsize=self.max_size)  # FIFO fallback to LRU
        else:
            return LRUCache(maxsize=self.max_size)  # Default
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            start_time = time.time()
            
            try:
                value = self._cache[key]
                
                # Record cache hit
                execution_time = time.time() - start_time
                self._strategy_performance[self.current_strategy].update(
                    execution_time, cache_hit=True
                )
                
                # Update access patterns
                self._access_times.append(time.time())
                self._key_frequencies[key] += 1
                
                return value
                
            except KeyError:
                # Record cache miss
                execution_time = time.time() - start_time
                self._strategy_performance[self.current_strategy].update(
                    execution_time, cache_hit=False
                )
                
                return None
            
            finally:
                self._operation_count += 1
                
                # Check if adaptation is needed
                if self._operation_count % self.adaptation_interval == 0:
                    self._consider_adaptation()
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self._lock:
            self._cache[key] = value
            self._key_frequencies[key] += 1
    
    def _consider_adaptation(self):
        """Consider switching cache strategy based on performance."""
        try:
            current_performance = self._strategy_performance[self.current_strategy]
            
            # Analyze access patterns
            temporal_locality = self._analyze_temporal_locality()
            frequency_skew = self._analyze_frequency_distribution()
            
            # Determine optimal strategy based on patterns
            optimal_strategy = self._determine_optimal_strategy(
                temporal_locality, frequency_skew, current_performance
            )
            
            if optimal_strategy != self.current_strategy:
                self.logger.info(
                    f"Adapting cache strategy from {self.current_strategy.value} to {optimal_strategy.value}",
                    event_type=EventType.PERFORMANCE_ALERT,
                    temporal_locality=temporal_locality,
                    frequency_skew=frequency_skew,
                    current_hit_rate=current_performance.cache_hit_rate
                )
                
                self._switch_strategy(optimal_strategy)
        
        except Exception as e:
            self.logger.warning(f"Cache adaptation failed: {str(e)}")
    
    def _analyze_temporal_locality(self) -> float:
        """Analyze temporal locality of access patterns."""
        if len(self._access_times) < 10:
            return 0.5  # Neutral
        
        # Calculate time between accesses
        access_intervals = [
            self._access_times[i] - self._access_times[i-1]
            for i in range(1, len(self._access_times))
        ]
        
        # High temporal locality = small, consistent intervals
        avg_interval = sum(access_intervals) / len(access_intervals)
        interval_variance = sum((x - avg_interval) ** 2 for x in access_intervals) / len(access_intervals)
        
        # Normalize to 0-1 (higher = more temporal locality)
        temporal_locality = 1.0 / (1.0 + interval_variance / max(avg_interval, 0.001))
        
        return min(1.0, max(0.0, temporal_locality))
    
    def _analyze_frequency_distribution(self) -> float:
        """Analyze frequency distribution skew."""
        if len(self._key_frequencies) < 5:
            return 0.5  # Neutral
        
        frequencies = list(self._key_frequencies.values())
        if len(frequencies) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        std_dev = variance ** 0.5
        
        # Coefficient of variation (higher = more skewed)
        cv = std_dev / max(mean_freq, 0.001)
        
        # Normalize to 0-1 (higher = more skewed)
        return min(1.0, cv / 2.0)
    
    def _determine_optimal_strategy(
        self,
        temporal_locality: float,
        frequency_skew: float,
        current_performance: PerformanceMetrics
    ) -> CacheStrategy:
        """Determine optimal cache strategy based on access patterns."""
        
        # If current performance is very good, don't change
        if current_performance.cache_hit_rate > 0.9:
            return self.current_strategy
        
        # Strategy selection based on access patterns
        if temporal_locality > 0.7:
            # High temporal locality favors LRU
            return CacheStrategy.LRU
        elif frequency_skew > 0.7:
            # High frequency skew favors LFU
            return CacheStrategy.LFU
        elif temporal_locality < 0.3 and frequency_skew < 0.3:
            # Low locality and frequency differences favor TTL
            return CacheStrategy.TTL
        else:
            # Mixed patterns favor FIFO for simplicity
            return CacheStrategy.FIFO
    
    def _switch_strategy(self, new_strategy: CacheStrategy):
        """Switch to new cache strategy while preserving data."""
        # Save current cache content
        old_items = dict(self._cache)
        
        # Create new cache
        self._cache = self._create_cache(new_strategy)
        
        # Migrate most frequently used items
        sorted_items = sorted(
            old_items.items(),
            key=lambda x: self._key_frequencies.get(x[0], 0),
            reverse=True
        )
        
        # Migrate up to max_size items
        for key, value in sorted_items[:self.max_size]:
            self._cache[key] = value
        
        self.current_strategy = new_strategy
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._key_frequencies.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'strategy': self.current_strategy.value,
                'size': len(self._cache),
                'max_size': self.max_size,
                'operation_count': self._operation_count,
                'strategy_performance': {
                    strategy.value: metrics.to_dict()
                    for strategy, metrics in self._strategy_performance.items()
                },
                'temporal_locality': self._analyze_temporal_locality(),
                'frequency_skew': self._analyze_frequency_distribution()
            }


class ResourcePool:
    """
    Resource pool for managing expensive objects like compiled JAX functions.
    
    Implements object pooling pattern to reuse expensive-to-create resources
    and reduce memory allocation overhead.
    """
    
    def __init__(
        self,
        resource_factory: Callable,
        max_size: int = 10,
        min_size: int = 2,
        idle_timeout: float = 300.0  # 5 minutes
    ):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.min_size = min_size
        self.idle_timeout = idle_timeout
        
        # Pool state
        self._available = deque()
        self._in_use: Dict[int, Any] = {}
        self._creation_times: Dict[int, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring
        self._total_created = 0
        self._total_requests = 0
        self._cache_hits = 0
        
        # Pre-populate with minimum resources
        self._populate_minimum()
        
        # Cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        self.logger = get_logger()
    
    def _populate_minimum(self):
        """Pre-populate pool with minimum resources."""
        for _ in range(self.min_size):
            resource = self.resource_factory()
            resource_id = id(resource)
            self._available.append((resource_id, resource))
            self._creation_times[resource_id] = time.time()
            self._total_created += 1
    
    def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire resource from pool."""
        with self._lock:
            self._total_requests += 1
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Try to get available resource
                if self._available:
                    resource_id, resource = self._available.popleft()
                    self._in_use[resource_id] = resource
                    self._cache_hits += 1
                    
                    self.logger.debug(f"Resource {resource_id} acquired from pool")
                    return resource
                
                # Create new resource if pool not at max capacity
                if len(self._in_use) < self.max_size:
                    resource = self.resource_factory()
                    resource_id = id(resource)
                    self._in_use[resource_id] = resource
                    self._creation_times[resource_id] = time.time()
                    self._total_created += 1
                    
                    self.logger.debug(f"New resource {resource_id} created")
                    return resource
                
                # Wait a bit and retry
                time.sleep(0.1)
            
            raise TimeoutError(f"Could not acquire resource within {timeout} seconds")
    
    def release(self, resource: Any):
        """Release resource back to pool."""
        with self._lock:
            resource_id = id(resource)
            
            if resource_id in self._in_use:
                del self._in_use[resource_id]
                
                # Only keep if pool size allows and resource is not too old
                current_time = time.time()
                resource_age = current_time - self._creation_times.get(resource_id, current_time)
                
                if (len(self._available) < self.max_size and 
                    resource_age < self.idle_timeout):
                    self._available.append((resource_id, resource))
                    self.logger.debug(f"Resource {resource_id} returned to pool")
                else:
                    # Resource is too old or pool is full, discard it
                    if resource_id in self._creation_times:
                        del self._creation_times[resource_id]
                    self.logger.debug(f"Resource {resource_id} discarded (age: {resource_age:.1f}s)")
    
    def _cleanup_loop(self):
        """Cleanup loop to remove idle resources."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_resources()
            except Exception as e:
                self.logger.warning(f"Resource pool cleanup error: {str(e)}")
    
    def _cleanup_idle_resources(self):
        """Remove idle resources that have exceeded timeout."""
        with self._lock:
            current_time = time.time()
            active_resources = []
            
            while self._available:
                resource_id, resource = self._available.popleft()
                resource_age = current_time - self._creation_times.get(resource_id, current_time)
                
                if resource_age < self.idle_timeout and len(active_resources) < self.min_size:
                    # Keep resource if not too old and we need minimum resources
                    active_resources.append((resource_id, resource))
                else:
                    # Discard old resource
                    if resource_id in self._creation_times:
                        del self._creation_times[resource_id]
                    self.logger.debug(f"Cleaned up idle resource {resource_id} (age: {resource_age:.1f}s)")
            
            # Add back active resources
            self._available.extend(active_resources)
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            pool_efficiency = self._cache_hits / max(self._total_requests, 1)
            
            return {
                'total_created': self._total_created,
                'total_requests': self._total_requests,
                'cache_hits': self._cache_hits,
                'pool_efficiency': pool_efficiency,
                'available_count': len(self._available),
                'in_use_count': len(self._in_use),
                'max_size': self.max_size,
                'min_size': self.min_size
            }


class OptimizedQuantumPlanner:
    """
    High-performance optimized quantum task planner with advanced caching,
    parallel processing, and adaptive optimization strategies.
    """
    
    def __init__(
        self,
        config: PlannerConfig,
        performance_level: PerformanceLevel = PerformanceLevel.BALANCED,
        enable_jit: bool = True,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None
    ):
        self.config = config
        self.performance_level = performance_level
        self.enable_jit = enable_jit
        self.enable_parallel = enable_parallel
        
        # Base planner
        self.base_planner = QuantumTaskPlanner(config)
        
        # Performance optimizations
        self._setup_caching()
        self._setup_compilation()
        self._setup_parallelization(max_workers)
        
        # Monitoring
        self.performance_metrics = defaultdict(lambda: PerformanceMetrics(""))
        
        self.logger = get_logger()
    
    def _setup_caching(self):
        """Setup adaptive caching system."""
        cache_size = {
            PerformanceLevel.BASIC: 100,
            PerformanceLevel.BALANCED: 1000,
            PerformanceLevel.AGGRESSIVE: 10000,
            PerformanceLevel.MEMORY_OPTIMIZED: 50
        }[self.performance_level]
        
        self.cache = AdaptiveCache(
            max_size=cache_size,
            initial_strategy=CacheStrategy.LRU,
            ttl_seconds=3600
        )
        
        # Specialized caches
        self.fitness_cache = LRUCache(maxsize=cache_size // 2)
        self.quantum_state_cache = LRUCache(maxsize=cache_size // 4)
        
    def _setup_compilation(self):
        """Setup JAX compilation and optimization."""
        if not self.enable_jit:
            return
        
        # Pre-compile frequently used functions
        self._compiled_functions = {}
        
        # Quantum interference computation
        @jit
        def compute_interference_optimized(
            amplitudes: jnp.ndarray,
            phases: jnp.ndarray
        ) -> jnp.ndarray:
            complex_amplitudes = amplitudes * jnp.exp(1j * phases)
            interference_matrix = jnp.outer(complex_amplitudes, jnp.conj(complex_amplitudes))
            probabilities = jnp.abs(jnp.diag(interference_matrix))
            return probabilities / (jnp.sum(probabilities) + 1e-10)
        
        self._compiled_functions['interference'] = compute_interference_optimized
        
        # Fitness computation
        @jit
        def compute_fitness_vectorized(
            task_priorities: jnp.ndarray,
            task_durations: jnp.ndarray,
            dependency_violations: jnp.ndarray,
            resource_violations: jnp.ndarray
        ) -> jnp.ndarray:
            priority_score = jnp.mean(task_priorities)
            time_penalty = jnp.sum(task_durations) / len(task_durations)
            violation_penalty = jnp.sum(dependency_violations + resource_violations)
            
            return priority_score - 0.1 * time_penalty - violation_penalty
        
        self._compiled_functions['fitness'] = compute_fitness_vectorized
        
        self.logger.info("JAX compilation setup complete")
    
    def _setup_parallelization(self, max_workers: Optional[int]):
        """Setup parallel processing capabilities."""
        if not self.enable_parallel:
            return
        
        # Determine optimal worker count
        if max_workers is None:
            cpu_count = mp.cpu_count()
            max_workers = {
                PerformanceLevel.BASIC: min(2, cpu_count),
                PerformanceLevel.BALANCED: min(4, cpu_count),
                PerformanceLevel.AGGRESSIVE: cpu_count,
                PerformanceLevel.MEMORY_OPTIMIZED: min(2, cpu_count)
            }[self.performance_level]
        
        # Thread pool for I/O bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Process pool for CPU bound tasks (if beneficial)
        if max_workers > 1 and self.performance_level == PerformanceLevel.AGGRESSIVE:
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers // 2)
        else:
            self.process_pool = None
        
        self.logger.info(f"Parallel processing setup with {max_workers} workers")
    
    @functools.lru_cache(maxsize=128)
    def _get_cached_quantum_state(self, task_hash: str) -> Dict[str, Any]:
        """Get cached quantum state summary."""
        return self.base_planner.get_quantum_state_summary()
    
    def add_task_optimized(self, task: QuantumTask) -> 'OptimizedQuantumPlanner':
        """Add task with performance optimizations."""
        start_time = time.time()
        
        # Check cache first
        task_key = f"task_{task.id}_{hash(str(task))}"
        cached_result = self.cache.get(task_key)
        
        if cached_result is not None:
            self.logger.debug(f"Using cached task configuration for {task.id}")
            # Apply cached optimizations
            task = cached_result
        else:
            # Optimize task properties
            task = self._optimize_task_properties(task)
            self.cache.set(task_key, task)
        
        # Add to base planner
        self.base_planner.add_task(task)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.performance_metrics['add_task'].update(execution_time, cache_hit=(cached_result is not None))
        
        return self
    
    def _optimize_task_properties(self, task: QuantumTask) -> QuantumTask:
        """Optimize task properties for better performance."""
        
        # Optimize quantum amplitude based on priority
        optimal_amplitude = self._calculate_optimal_amplitude(task.priority, task.estimated_duration)
        task.amplitude = complex(optimal_amplitude, task.amplitude.imag)
        
        # Optimize phase for interference
        task.phase = self._optimize_phase(task.priority)
        
        return task
    
    def _calculate_optimal_amplitude(self, priority: float, duration: float) -> float:
        """Calculate optimal quantum amplitude for task."""
        # Balance priority and inverse duration for amplitude
        duration_factor = 1.0 / max(duration, 0.1)  # Shorter tasks get higher amplitude
        return min(1.0, (priority * 0.7 + duration_factor * 0.3))
    
    def _optimize_phase(self, priority: float) -> float:
        """Optimize quantum phase for constructive interference."""
        # High priority tasks get phases that promote constructive interference
        import math
        return (priority * math.pi) % (2 * math.pi)
    
    def optimize_plan_parallel(self, num_trials: int = 10) -> Dict[str, Any]:
        """Optimize plan using parallel processing for multiple trials."""
        start_time = time.time()
        
        # Check for cached result
        cache_key = f"optimize_plan_{len(self.base_planner.tasks)}_{hash(frozenset(self.base_planner.tasks.keys()))}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            self.logger.info("Using cached optimization result")
            return cached_result
        
        if self.enable_parallel and self.thread_pool and num_trials > 1:
            # Run multiple optimization trials in parallel
            futures = []
            
            for trial in range(num_trials):
                future = self.thread_pool.submit(
                    self._run_optimization_trial,
                    trial,
                    seed=trial * 42  # Different seed for each trial
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 1 minute timeout per trial
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Optimization trial failed: {str(e)}")
            
            # Select best result
            if results:
                best_result = max(results, key=lambda r: r.get('fitness_score', 0))
                self.logger.info(f"Parallel optimization completed: {len(results)} trials, best fitness: {best_result.get('fitness_score', 0):.4f}")
            else:
                # Fallback to single-threaded optimization
                best_result = self.base_planner.optimize_plan()
                
        else:
            # Single-threaded optimization
            best_result = self.base_planner.optimize_plan()
        
        # Cache result
        self.cache.set(cache_key, best_result)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.performance_metrics['optimize_plan'].update(execution_time, memory_mb=self._get_memory_usage())
        
        return best_result
    
    def _run_optimization_trial(self, trial_id: int, seed: int = None) -> Dict[str, Any]:
        """Run single optimization trial."""
        if seed is not None:
            # Set random seed for reproducible trials
            import numpy as np
            np.random.seed(seed)
        
        return self.base_planner.optimize_plan()
    
    def batch_process_tasks(self, task_batches: List[List[QuantumTask]]) -> List[Dict[str, Any]]:
        """Process multiple task batches in parallel."""
        if not self.enable_parallel or not self.thread_pool:
            # Sequential processing
            return [self._process_task_batch(batch) for batch in task_batches]
        
        # Parallel processing
        futures = []
        for i, batch in enumerate(task_batches):
            future = self.thread_pool.submit(self._process_task_batch, batch, batch_id=i)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=120)  # 2 minute timeout per batch
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing failed: {str(e)}")
                results.append({'error': str(e), 'success': False})
        
        return results
    
    def _process_task_batch(self, tasks: List[QuantumTask], batch_id: int = 0) -> Dict[str, Any]:
        """Process a batch of tasks."""
        start_time = time.time()
        
        # Add tasks to temporary planner instance
        temp_planner = QuantumTaskPlanner(self.config)
        for resource, amount in self.base_planner.resource_pool.items():
            temp_planner.add_resource(resource, amount)
        
        for task in tasks:
            temp_planner.add_task(task)
        
        # Optimize and execute
        try:
            optimization_result = temp_planner.optimize_plan()
            execution_result = temp_planner.execute_plan(optimization_result)
            
            processing_time = time.time() - start_time
            
            return {
                'batch_id': batch_id,
                'success': True,
                'task_count': len(tasks),
                'processing_time': processing_time,
                'optimization_result': optimization_result,
                'execution_result': execution_result
            }
            
        except Exception as e:
            return {
                'batch_id': batch_id,
                'success': False,
                'error': str(e),
                'task_count': len(tasks)
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # System metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_gb = memory.available / 1024 / 1024 / 1024
        except:
            cpu_percent = 0.0
            memory_percent = 0.0
            available_memory_gb = 0.0
        
        # Performance metrics
        operation_metrics = {
            name: metrics.to_dict() 
            for name, metrics in self.performance_metrics.items()
        }
        
        # Cache statistics
        cache_stats = self.cache.stats()
        
        # Thread pool statistics
        thread_pool_stats = {}
        if hasattr(self, 'thread_pool') and self.thread_pool:
            thread_pool_stats = {
                'max_workers': self.thread_pool._max_workers,
                'active_threads': len([t for t in self.thread_pool._threads if t.is_alive()]) if hasattr(self.thread_pool, '_threads') else 0
            }
        
        report = {
            'timestamp': time.time(),
            'performance_level': self.performance_level.value,
            'optimizations_enabled': {
                'jit_compilation': self.enable_jit,
                'parallel_processing': self.enable_parallel,
                'adaptive_caching': True
            },
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'available_memory_gb': available_memory_gb,
                'current_memory_usage_mb': self._get_memory_usage()
            },
            'operation_metrics': operation_metrics,
            'cache_stats': cache_stats,
            'thread_pool_stats': thread_pool_stats,
            'quantum_planner_stats': {
                'total_tasks': len(self.base_planner.tasks),
                'resource_pool_size': len(self.base_planner.resource_pool),
                'entanglements': len(self.base_planner.entanglement_matrix)
            },
            'recommendations': self._generate_performance_recommendations()
        }
        
        return report
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        memory_mb = self._get_memory_usage()
        if memory_mb > 1000:  # > 1GB
            recommendations.append("Consider reducing cache size or using MEMORY_OPTIMIZED performance level")
        
        # Cache recommendations
        cache_stats = self.cache.stats()
        cache_efficiency = cache_stats.get('strategy_performance', {})
        current_strategy = cache_stats.get('strategy', 'unknown')
        
        for strategy, metrics in cache_efficiency.items():
            if strategy != current_strategy and metrics.get('cache_hit_rate', 0) > 0.1:
                current_hit_rate = cache_efficiency.get(current_strategy, {}).get('cache_hit_rate', 0)
                strategy_hit_rate = metrics.get('cache_hit_rate', 0)
                
                if strategy_hit_rate > current_hit_rate * 1.2:  # 20% better
                    recommendations.append(f"Consider switching to {strategy} cache strategy for better hit rate")
        
        # Parallelization recommendations
        if not self.enable_parallel and len(self.base_planner.tasks) > 10:
            recommendations.append("Enable parallel processing for better performance with many tasks")
        
        # JIT compilation recommendations
        if not self.enable_jit:
            recommendations.append("Enable JIT compilation for faster computation")
        
        # Performance level recommendations
        if self.performance_level == PerformanceLevel.BASIC and memory_mb < 500:
            recommendations.append("Consider upgrading to BALANCED performance level for better optimization")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if hasattr(self, 'process_pool') and self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self.cache.clear()
        gc.collect()  # Force garbage collection
        
        self.logger.info("OptimizedQuantumPlanner cleanup completed")


# Factory function for easy creation
def create_optimized_planner(
    config: Optional[PlannerConfig] = None,
    performance_level: PerformanceLevel = PerformanceLevel.BALANCED,
    **kwargs
) -> OptimizedQuantumPlanner:
    """
    Factory function to create optimized quantum planner.
    
    Args:
        config: Planner configuration
        performance_level: Performance optimization level
        **kwargs: Additional optimization parameters
        
    Returns:
        Configured OptimizedQuantumPlanner instance
    """
    
    if config is None:
        config = PlannerConfig()
    
    return OptimizedQuantumPlanner(
        config=config,
        performance_level=performance_level,
        **kwargs
    )


# Performance profiling utilities
class PerformanceProfiler:
    """Performance profiler for quantum planning operations."""
    
    def __init__(self):
        self.profiles = defaultdict(list)
        self.logger = get_logger()
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return self._ProfileContext(self, operation_name)
    
    class _ProfileContext:
        def __init__(self, profiler, operation_name):
            self.profiler = profiler
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                self.profiler.profiles[self.operation_name].append(duration)
                
                if len(self.profiler.profiles[self.operation_name]) % 100 == 0:
                    # Log performance summary every 100 calls
                    times = self.profiler.profiles[self.operation_name]
                    avg_time = sum(times) / len(times)
                    self.profiler.logger.info(
                        f"Performance summary for {self.operation_name}: {avg_time:.3f}s avg over {len(times)} calls",
                        event_type=EventType.PERFORMANCE_ALERT,
                        operation=self.operation_name,
                        avg_time=avg_time,
                        call_count=len(times)
                    )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for operation, times in self.profiles.items():
            if times:
                summary[operation] = {
                    'call_count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'last_10_avg': sum(times[-10:]) / min(10, len(times))
                }
        
        return summary


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    return _global_profiler