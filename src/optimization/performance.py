"""
Performance optimization and scalability improvements for RLHF-Contract-Wizard.

Implements caching, batching, parallelization, and other performance optimizations
for high-throughput contract processing and verification.
"""

import time
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
from collections import defaultdict, OrderedDict
import hashlib
import pickle

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import flax

from ..models.reward_contract import RewardContract
from ..models.reward_model import ContractualRewardModel
from ..utils.helpers import setup_logging, create_timestamp
from ..utils.error_handling import CircuitBreaker, ErrorHandler, ErrorCategory


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    throughput: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    
    Provides fast caching with automatic eviction of least recently used items
    when capacity is reached.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to cache
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self._hits += 1
                return value
            else:
                self._misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self.cache:
                # Update existing item
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'capacity': self.capacity,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate
            }


class ContractCache:
    """
    Specialized cache for reward contract computations.
    
    Caches contract reward computations, constraint evaluations,
    and verification results for improved performance.
    """
    
    def __init__(self, capacity: int = 5000, ttl: float = 3600.0):
        """
        Initialize contract cache.
        
        Args:
            capacity: Maximum cache entries
            ttl: Time-to-live in seconds
        """
        self.reward_cache = LRUCache(capacity)
        self.constraint_cache = LRUCache(capacity // 2)
        self.verification_cache = LRUCache(capacity // 4)
        self.ttl = ttl
        self.timestamps = {}
        self._lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def _cache_key(self, contract_hash: str, state: jnp.ndarray, action: jnp.ndarray) -> str:
        """Generate cache key for state-action pair."""
        state_hash = hashlib.md5(state.tobytes()).hexdigest()[:8]
        action_hash = hashlib.md5(action.tobytes()).hexdigest()[:8]
        return f"{contract_hash}:{state_hash}:{action_hash}"
    
    def get_reward(
        self,
        contract_hash: str,
        state: jnp.ndarray,
        action: jnp.ndarray
    ) -> Optional[float]:
        """Get cached reward computation."""
        key = self._cache_key(contract_hash, state, action)
        
        with self._lock:
            if self._is_expired(key):
                return None
            return self.reward_cache.get(key)
    
    def put_reward(
        self,
        contract_hash: str,
        state: jnp.ndarray,
        action: jnp.ndarray,
        reward: float
    ) -> None:
        """Cache reward computation."""
        key = self._cache_key(contract_hash, state, action)
        
        with self._lock:
            self.reward_cache.put(key, reward)
            self.timestamps[key] = time.time()
    
    def get_constraint_result(self, constraint_key: str) -> Optional[bool]:
        """Get cached constraint evaluation."""
        with self._lock:
            if self._is_expired(constraint_key):
                return None
            return self.constraint_cache.get(constraint_key)
    
    def put_constraint_result(self, constraint_key: str, result: bool) -> None:
        """Cache constraint evaluation."""
        with self._lock:
            self.constraint_cache.put(constraint_key, result)
            self.timestamps[constraint_key] = time.time()
    
    def get_verification_result(self, contract_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached verification result."""
        with self._lock:
            if self._is_expired(contract_hash):
                return None
            return self.verification_cache.get(contract_hash)
    
    def put_verification_result(self, contract_hash: str, result: Dict[str, Any]) -> None:
        """Cache verification result."""
        with self._lock:
            self.verification_cache.put(contract_hash, result)
            self.timestamps[contract_hash] = time.time()
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'reward_cache': self.reward_cache.stats(),
            'constraint_cache': self.constraint_cache.stats(),
            'verification_cache': self.verification_cache.stats()
        }


class BatchProcessor:
    """
    Efficient batch processing for contract operations.
    
    Processes multiple contracts or computations in batches for improved
    throughput and resource utilization.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = None,
        use_gpu: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            max_workers: Maximum worker threads (None for auto)
            use_gpu: Whether to use GPU acceleration
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() + 4)
        self.use_gpu = use_gpu and len(jax.devices('gpu')) > 0
        self.logger = setup_logging()
        
        # JIT compile batch operations
        if self.use_gpu:
            self._batch_compute_rewards = jit(vmap(self._single_reward_computation))
            self._batch_check_constraints = jit(vmap(self._single_constraint_check))
        else:
            self._batch_compute_rewards = vmap(self._single_reward_computation)
            self._batch_check_constraints = vmap(self._single_constraint_check)
    
    def process_contracts_batch(
        self,
        contracts: List[RewardContract],
        states: jnp.ndarray,
        actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, List[Dict[str, bool]]]:
        """
        Process multiple contracts in batch.
        
        Args:
            contracts: List of contracts to process
            states: Batch of states [batch, state_dim]
            actions: Batch of actions [batch, action_dim]
            
        Returns:
            Tuple of (rewards, constraint_violations)
        """
        batch_size = len(contracts)
        assert states.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        
        start_time = time.time()
        
        # Batch reward computation
        rewards = jnp.zeros(batch_size)
        all_violations = []
        
        for i, contract in enumerate(contracts):
            reward = contract.compute_reward(states[i], actions[i])
            violations = contract.check_violations(states[i], actions[i])
            
            rewards = rewards.at[i].set(reward)
            all_violations.append(violations)
        
        processing_time = time.time() - start_time
        throughput = batch_size / processing_time
        
        self.logger.debug(f"Processed {batch_size} contracts in {processing_time:.4f}s "
                         f"(throughput: {throughput:.2f} contracts/s)")
        
        return rewards, all_violations
    
    def _single_reward_computation(self, contract_params: Dict, state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Single reward computation for vmapping."""
        # This would be implemented with actual contract computation logic
        # For now, return a mock computation
        return jnp.mean(state) * 0.5 + jnp.mean(action) * 0.5
    
    def _single_constraint_check(self, constraint_params: Dict, state: jnp.ndarray, action: jnp.ndarray) -> bool:
        """Single constraint check for vmapping."""
        # Mock constraint check
        return jnp.all(action >= -1.0) and jnp.all(action <= 1.0)
    
    def process_verification_batch(
        self,
        contracts: List[Dict[str, Any]],
        verification_service: Any
    ) -> List[Dict[str, Any]]:
        """Process contract verification in parallel batches."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all verification tasks
            future_to_contract = {
                executor.submit(verification_service.verify_contract, contract): contract
                for contract in contracts
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_contract):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Verification failed: {e}")
                    results.append({'valid': False, 'error': str(e)})
        
        return results


class ResourcePoolManager:
    """
    Manages pools of computational resources for scalable processing.
    
    Provides resource pooling for expensive operations like model inference,
    verification, and blockchain interactions.
    """
    
    def __init__(self):
        self.pools = {}
        self._lock = threading.RLock()
        self.logger = setup_logging()
    
    def create_pool(
        self,
        pool_name: str,
        factory_func: Callable,
        pool_size: int = 5,
        max_size: int = 20
    ):
        """Create a resource pool."""
        with self._lock:
            if pool_name in self.pools:
                raise ValueError(f"Pool {pool_name} already exists")
            
            pool = {
                'factory': factory_func,
                'available': [],
                'in_use': set(),
                'pool_size': pool_size,
                'max_size': max_size,
                'created_count': 0
            }
            
            # Pre-populate pool
            for _ in range(pool_size):
                resource = factory_func()
                pool['available'].append(resource)
                pool['created_count'] += 1
            
            self.pools[pool_name] = pool
            self.logger.info(f"Created resource pool '{pool_name}' with {pool_size} resources")
    
    def get_resource(self, pool_name: str, timeout: float = 5.0):
        """Get resource from pool with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                pool = self.pools.get(pool_name)
                if not pool:
                    raise ValueError(f"Pool {pool_name} not found")
                
                # Try to get available resource
                if pool['available']:
                    resource = pool['available'].pop()
                    pool['in_use'].add(id(resource))
                    return ResourceHandle(self, pool_name, resource)
                
                # Create new resource if under max size
                elif pool['created_count'] < pool['max_size']:
                    resource = pool['factory']()
                    pool['created_count'] += 1
                    pool['in_use'].add(id(resource))
                    return ResourceHandle(self, pool_name, resource)
            
            # Wait briefly and retry
            time.sleep(0.01)
        
        raise TimeoutError(f"Timeout waiting for resource from pool {pool_name}")
    
    def return_resource(self, pool_name: str, resource: Any):
        """Return resource to pool."""
        with self._lock:
            pool = self.pools.get(pool_name)
            if not pool:
                return
            
            resource_id = id(resource)
            if resource_id in pool['in_use']:
                pool['in_use'].remove(resource_id)
                pool['available'].append(resource)
    
    def get_pool_stats(self, pool_name: str) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            pool = self.pools.get(pool_name)
            if not pool:
                return {}
            
            return {
                'available': len(pool['available']),
                'in_use': len(pool['in_use']),
                'total_created': pool['created_count'],
                'pool_size': pool['pool_size'],
                'max_size': pool['max_size']
            }


class ResourceHandle:
    """Context manager for resource pool resources."""
    
    def __init__(self, manager: ResourcePoolManager, pool_name: str, resource: Any):
        self.manager = manager
        self.pool_name = pool_name
        self.resource = resource
    
    def __enter__(self):
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.return_resource(self.pool_name, self.resource)


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Coordinates caching, batching, resource pooling, and other optimizations
    for maximum system throughput and efficiency.
    """
    
    def __init__(
        self,
        cache_size: int = 10000,
        batch_size: int = 32,
        enable_gpu: bool = True,
        max_workers: int = None
    ):
        """
        Initialize performance optimizer.
        
        Args:
            cache_size: Size of various caches
            batch_size: Default batch size for operations
            enable_gpu: Whether to use GPU acceleration
            max_workers: Maximum worker threads
        """
        self.logger = setup_logging()
        self.error_handler = ErrorHandler()
        
        # Initialize components
        self.cache = ContractCache(capacity=cache_size)
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            max_workers=max_workers,
            use_gpu=enable_gpu
        )
        self.resource_manager = ResourcePoolManager()
        
        # Performance tracking
        self.metrics: List[PerformanceMetrics] = []
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = True
        self.scaling_thresholds = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'response_time_threshold': 1.0,
            'error_rate_threshold': 0.05
        }
        
        # Initialize resource pools
        self._initialize_resource_pools()
        
        self.logger.info("Performance optimizer initialized")
    
    def _initialize_resource_pools(self):
        """Initialize common resource pools."""
        # Reward model pool
        def create_reward_model():
            from ..models.reward_model import ContractualRewardModel, RewardModelConfig
            from ..models.reward_contract import RewardContract
            
            # Create a basic contract for the model
            contract = RewardContract(name="PooledModel", stakeholders={"default": 1.0})
            config = RewardModelConfig(hidden_dim=256)
            random_key = jax.random.PRNGKey(int(time.time()) % 2**32)
            
            return ContractualRewardModel(config, contract, random_key)
        
        self.resource_manager.create_pool(
            'reward_models',
            create_reward_model,
            pool_size=3,
            max_size=10
        )
        
        # Verification service pool
        def create_verification_service():
            from ..services.verification_service import VerificationService
            return VerificationService(backend='mock')
        
        self.resource_manager.create_pool(
            'verification_services',
            create_verification_service,
            pool_size=2,
            max_size=8
        )
    
    def optimize_contract_computation(
        self,
        contract: RewardContract,
        states: jnp.ndarray,
        actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        """
        Optimized contract computation with caching and batching.
        
        Args:
            contract: Contract to compute
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Tuple of (rewards, computation_time)
        """
        start_time = time.time()
        contract_hash = contract.compute_hash()
        
        # Check cache for each state-action pair
        rewards = []
        cache_hits = 0
        
        for i in range(len(states)):
            cached_reward = self.cache.get_reward(contract_hash, states[i], actions[i])
            if cached_reward is not None:
                rewards.append(cached_reward)
                cache_hits += 1
            else:
                # Compute and cache
                reward = contract.compute_reward(states[i], actions[i])
                rewards.append(reward)
                self.cache.put_reward(contract_hash, states[i], actions[i], reward)
        
        computation_time = time.time() - start_time
        cache_hit_rate = cache_hits / len(states) if states else 0.0
        
        # Record metrics
        metrics = PerformanceMetrics(
            operation="contract_computation",
            start_time=start_time,
            end_time=time.time(),
            duration=computation_time,
            throughput=len(states) / computation_time if computation_time > 0 else 0,
            cache_hit_rate=cache_hit_rate
        )
        self.metrics.append(metrics)
        
        self.logger.debug(f"Contract computation: {len(states)} samples in {computation_time:.4f}s "
                         f"(cache hit rate: {cache_hit_rate:.2%})")
        
        return jnp.array(rewards), computation_time
    
    def optimize_verification_batch(
        self,
        contracts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimized batch verification with caching and parallel processing.
        
        Args:
            contracts: List of contract specifications
            
        Returns:
            List of verification results
        """
        start_time = time.time()
        results = []
        cache_hits = 0
        
        # Check cache first
        contracts_to_verify = []
        cached_results = []
        
        for contract in contracts:
            contract_hash = self._compute_contract_hash(contract)
            cached_result = self.cache.get_verification_result(contract_hash)
            
            if cached_result is not None:
                cached_results.append((len(results), cached_result))
                results.append(None)  # Placeholder
                cache_hits += 1
            else:
                contracts_to_verify.append((len(results), contract))
                results.append(None)  # Placeholder
        
        # Verify uncached contracts in parallel
        if contracts_to_verify:
            with self.resource_manager.get_resource('verification_services') as verification_service:
                verification_results = self.batch_processor.process_verification_batch(
                    [contract for _, contract in contracts_to_verify],
                    verification_service
                )
                
                # Cache and store results
                for (result_idx, contract), result in zip(contracts_to_verify, verification_results):
                    contract_hash = self._compute_contract_hash(contract)
                    self.cache.put_verification_result(contract_hash, result)
                    results[result_idx] = result
        
        # Fill in cached results
        for result_idx, cached_result in cached_results:
            results[result_idx] = cached_result
        
        computation_time = time.time() - start_time
        cache_hit_rate = cache_hits / len(contracts) if contracts else 0.0
        
        # Record metrics
        metrics = PerformanceMetrics(
            operation="batch_verification",
            start_time=start_time,
            end_time=time.time(),
            duration=computation_time,
            throughput=len(contracts) / computation_time if computation_time > 0 else 0,
            cache_hit_rate=cache_hit_rate
        )
        self.metrics.append(metrics)
        
        self.logger.info(f"Batch verification: {len(contracts)} contracts in {computation_time:.4f}s "
                        f"(cache hit rate: {cache_hit_rate:.2%})")
        
        return results
    
    def _compute_contract_hash(self, contract_dict: Dict[str, Any]) -> str:
        """Compute hash for contract dictionary."""
        import json
        contract_str = json.dumps(contract_dict, sort_keys=True)
        return hashlib.sha256(contract_str.encode()).hexdigest()[:16]
    
    def get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get circuit breaker for operation."""
        if operation not in self._circuit_breakers:
            self._circuit_breakers[operation] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=3
            )
        return self._circuit_breakers[operation]
    
    def monitor_performance(self) -> Dict[str, Any]:
        """Get current performance metrics and system status."""
        try:
            import psutil
            
            # System metrics
            system_metrics = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
            }
            
        except ImportError:
            system_metrics = {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'network_io': {}
            }
        
        # Application metrics
        recent_metrics = [m for m in self.metrics if time.time() - m.end_time < 300]  # Last 5 minutes
        
        if recent_metrics:
            avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics if m.throughput) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate) / len(recent_metrics)
        else:
            avg_duration = 0
            avg_throughput = 0
            avg_cache_hit_rate = 0
        
        application_metrics = {
            'avg_response_time': avg_duration,
            'avg_throughput': avg_throughput,
            'avg_cache_hit_rate': avg_cache_hit_rate,
            'total_operations': len(recent_metrics),
            'cache_stats': self.cache.stats(),
            'resource_pools': {
                name: self.resource_manager.get_pool_stats(name)
                for name in ['reward_models', 'verification_services']
            }
        }
        
        # Circuit breaker stats
        circuit_breaker_stats = {}
        for name, cb in self._circuit_breakers.items():
            circuit_breaker_stats[name] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count
            }
        
        return {
            'timestamp': time.time(),
            'system': system_metrics,
            'application': application_metrics,
            'circuit_breakers': circuit_breaker_stats
        }
    
    def auto_scale_resources(self) -> Dict[str, Any]:
        """Automatically scale resources based on current load."""
        if not self.auto_scaling_enabled:
            return {'scaled': False, 'reason': 'auto-scaling disabled'}
        
        performance = self.monitor_performance()
        scaling_actions = []
        
        # Check CPU threshold
        if performance['system']['cpu_percent'] > self.scaling_thresholds['cpu_threshold']:
            # Scale up reward model pool
            current_stats = self.resource_manager.get_pool_stats('reward_models')
            if current_stats['total_created'] < current_stats['max_size']:
                scaling_actions.append('increase_reward_model_pool')
        
        # Check memory threshold
        if performance['system']['memory_percent'] > self.scaling_thresholds['memory_threshold']:
            # Clear caches to free memory
            self.cache.reward_cache.clear()
            scaling_actions.append('clear_caches')
        
        # Check response time threshold
        if performance['application']['avg_response_time'] > self.scaling_thresholds['response_time_threshold']:
            # Increase batch size for better throughput
            if self.batch_processor.batch_size < 64:
                self.batch_processor.batch_size = min(64, self.batch_processor.batch_size * 2)
                scaling_actions.append('increase_batch_size')
        
        self.logger.info(f"Auto-scaling actions: {scaling_actions}")
        
        return {
            'scaled': len(scaling_actions) > 0,
            'actions': scaling_actions,
            'performance_snapshot': performance
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by clearing caches and releasing resources."""
        start_memory = self._get_memory_usage()
        
        # Clear caches
        self.cache.reward_cache.clear()
        self.cache.constraint_cache.clear()
        self.cache.verification_cache.clear()
        
        # Trim old metrics
        cutoff_time = time.time() - 3600  # Keep last hour
        self.metrics = [m for m in self.metrics if m.end_time > cutoff_time]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        end_memory = self._get_memory_usage()
        memory_freed = start_memory - end_memory
        
        return {
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'memory_freed_mb': memory_freed,
            'optimization_time': time.time()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_performance = self.monitor_performance()
        
        # Historical analysis
        recent_metrics = [m for m in self.metrics if time.time() - m.end_time < 3600]  # Last hour
        
        operations_by_type = defaultdict(list)
        for metric in recent_metrics:
            operations_by_type[metric.operation].append(metric)
        
        operation_stats = {}
        for operation, metrics in operations_by_type.items():
            operation_stats[operation] = {
                'count': len(metrics),
                'avg_duration': sum(m.duration for m in metrics) / len(metrics),
                'min_duration': min(m.duration for m in metrics),
                'max_duration': max(m.duration for m in metrics),
                'avg_throughput': sum(m.throughput for m in metrics if m.throughput) / len(metrics) if any(m.throughput for m in metrics) else 0
            }
        
        return {
            'timestamp': time.time(),
            'current_performance': current_performance,
            'historical_stats': operation_stats,
            'cache_efficiency': self.cache.stats(),
            'recommendations': self._generate_performance_recommendations(current_performance, operation_stats)
        }
    
    def _generate_performance_recommendations(
        self,
        current_performance: Dict[str, Any],
        operation_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # High CPU usage
        if current_performance['system']['cpu_percent'] > 85:
            recommendations.append("Consider increasing batch size or using more parallel workers")
        
        # High memory usage
        if current_performance['system']['memory_percent'] > 90:
            recommendations.append("Consider reducing cache size or implementing more aggressive cache eviction")
        
        # Low cache hit rate
        cache_hit_rate = current_performance['application']['avg_cache_hit_rate']
        if cache_hit_rate < 0.5:
            recommendations.append("Consider increasing cache size or TTL to improve hit rate")
        
        # High response times
        if current_performance['application']['avg_response_time'] > 2.0:
            recommendations.append("Consider optimizing computation paths or increasing resource pool sizes")
        
        # Low throughput operations
        for operation, stats in operation_stats.items():
            if stats['avg_throughput'] < 10:  # Less than 10 ops/second
                recommendations.append(f"Consider optimizing {operation} operation for better throughput")
        
        return recommendations


# Global performance optimizer instance
global_optimizer = PerformanceOptimizer()


def optimize_performance(func: Callable) -> Callable:
    """Decorator for automatic performance optimization."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Record successful execution
            metrics = PerformanceMetrics(
                operation=func.__name__,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time
            )
            global_optimizer.metrics.append(metrics)
            
            return result
            
        except Exception as e:
            # Handle with circuit breaker
            circuit_breaker = global_optimizer.get_circuit_breaker(func.__name__)
            
            # Record failure
            global_optimizer.error_handler.handle_error(
                error=e,
                operation=f"optimized_function:{func.__name__}",
                category=ErrorCategory.COMPUTATION
            )
            
            raise
    
    return wrapper