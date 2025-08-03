"""
Cache management for RLHF-Contract-Wizard.

Provides caching strategies for contract evaluations, constraint checks,
and computational results to improve performance.
"""

import json
import pickle
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

from .connection import redis_connection


logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies for different data types."""
    LRU = "lru"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"


@dataclass
class CacheItem:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if item is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of item in seconds."""
        return time.time() - self.created_at


class ContractCache:
    """
    High-performance cache for contract operations.
    
    Provides caching for contract evaluations, constraint checks,
    and other expensive computations.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        """
        Initialize contract cache.
        
        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default TTL in seconds
            strategy: Caching strategy
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self._local_cache: Dict[str, CacheItem] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        # Check local cache first
        if key in self._local_cache:
            item = self._local_cache[key]
            if not item.is_expired:
                item.accessed_at = time.time()
                item.access_count += 1
                self._stats['hits'] += 1
                return item.value
            else:
                # Remove expired item
                del self._local_cache[key]
        
        # Check Redis cache
        try:
            redis_value = await redis_connection.get(key)
            if redis_value:
                try:
                    value = pickle.loads(redis_value.encode('latin1'))
                    # Add to local cache
                    self._local_cache[key] = CacheItem(
                        key=key,
                        value=value,
                        created_at=time.time(),
                        accessed_at=time.time(),
                        access_count=1
                    )
                    self._stats['hits'] += 1
                    return value
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached value: {e}")
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
        
        self._stats['misses'] += 1
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set item in cache."""
        ttl = ttl or self.default_ttl
        
        # Add to local cache
        item = CacheItem(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            ttl_seconds=ttl
        )
        
        # Evict if necessary
        if len(self._local_cache) >= self.max_size:
            self._evict_item()
        
        self._local_cache[key] = item
        
        # Add to Redis cache
        try:
            serialized_value = pickle.dumps(value).decode('latin1')
            await redis_connection.set(key, serialized_value, ex=ttl)
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")
        
        self._stats['sets'] += 1
        return True
    
    def _evict_item(self):
        """Evict item based on strategy."""
        if not self._local_cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently accessed
            oldest_key = min(
                self._local_cache.keys(),
                key=lambda k: self._local_cache[k].accessed_at
            )
        else:
            # Default to oldest
            oldest_key = min(
                self._local_cache.keys(),
                key=lambda k: self._local_cache[k].created_at
            )
        
        del self._local_cache[oldest_key]
        self._stats['evictions'] += 1
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        deleted_local = key in self._local_cache
        if deleted_local:
            del self._local_cache[key]
        
        try:
            deleted_redis = await redis_connection.delete(key) > 0
        except Exception as e:
            logger.warning(f"Redis cache delete failed: {e}")
            deleted_redis = False
        
        return deleted_local or deleted_redis
    
    async def clear(self):
        """Clear all cached items."""
        self._local_cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'size': len(self._local_cache),
            'max_size': self.max_size,
            'strategy': self.strategy.value
        }


class ContractCacheDecorator:
    """
    Decorator for caching contract method results.
    """
    
    def __init__(
        self,
        cache: ContractCache,
        prefix: str = "contract",
        ttl: Optional[int] = None,
        skip_cache: Optional[Callable] = None
    ):
        """
        Initialize cache decorator.
        
        Args:
            cache: Cache instance
            prefix: Cache key prefix
            ttl: TTL override
            skip_cache: Function to determine if caching should be skipped
        """
        self.cache = cache
        self.prefix = prefix
        self.ttl = ttl
        self.skip_cache = skip_cache or (lambda *args, **kwargs: False)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate function with caching."""
        async def wrapper(*args, **kwargs):
            # Check if we should skip caching
            if self.skip_cache(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Generate cache key
            key = self.cache._generate_key(f"{self.prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            cached_result = await self.cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await self.cache.set(key, result, self.ttl)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


class RewardComputationCache:
    """
    Specialized cache for reward computation results.
    """
    
    def __init__(self, base_cache: ContractCache):
        """Initialize reward computation cache."""
        self.base_cache = base_cache
        self.prefix = "reward"
    
    async def get_reward(
        self,
        contract_hash: str,
        state_hash: str,
        action_hash: str,
        context_hash: str = ""
    ) -> Optional[float]:
        """Get cached reward computation."""
        key = f"{self.prefix}:{contract_hash}:{state_hash}:{action_hash}:{context_hash}"
        return await self.base_cache.get(key)
    
    async def set_reward(
        self,
        contract_hash: str,
        state_hash: str,
        action_hash: str,
        reward: float,
        context_hash: str = "",
        ttl: Optional[int] = None
    ) -> bool:
        """Cache reward computation result."""
        key = f"{self.prefix}:{contract_hash}:{state_hash}:{action_hash}:{context_hash}"
        return await self.base_cache.set(key, reward, ttl)


class ConstraintCache:
    """
    Specialized cache for constraint evaluation results.
    """
    
    def __init__(self, base_cache: ContractCache):
        """Initialize constraint cache."""
        self.base_cache = base_cache
        self.prefix = "constraint"
    
    async def get_violations(
        self,
        constraint_hash: str,
        state_hash: str,
        action_hash: str,
        context_hash: str = ""
    ) -> Optional[Dict[str, bool]]:
        """Get cached constraint violations."""
        key = f"{self.prefix}:{constraint_hash}:{state_hash}:{action_hash}:{context_hash}"
        return await self.base_cache.get(key)
    
    async def set_violations(
        self,
        constraint_hash: str,
        state_hash: str,
        action_hash: str,
        violations: Dict[str, bool],
        context_hash: str = "",
        ttl: Optional[int] = None
    ) -> bool:
        """Cache constraint violation results."""
        key = f"{self.prefix}:{constraint_hash}:{state_hash}:{action_hash}:{context_hash}"
        return await self.base_cache.set(key, violations, ttl)


# Global cache instances
contract_cache = ContractCache(max_size=10000, default_ttl=3600)
reward_cache = RewardComputationCache(contract_cache)
constraint_cache = ConstraintCache(contract_cache)


def cached_contract_method(
    prefix: str = "contract",
    ttl: Optional[int] = None,
    skip_cache: Optional[Callable] = None
):
    """
    Decorator for caching contract method results.
    
    Args:
        prefix: Cache key prefix
        ttl: TTL override
        skip_cache: Function to determine if caching should be skipped
    """
    return ContractCacheDecorator(
        cache=contract_cache,
        prefix=prefix,
        ttl=ttl,
        skip_cache=skip_cache
    )


async def warm_cache_for_contract(contract):
    """
    Pre-warm cache for a contract with common operations.
    
    Args:
        contract: RewardContract instance
    """
    logger.info(f"Warming cache for contract {contract.metadata.name}")
    
    # Pre-compute common state/action combinations
    import jax.numpy as jnp
    
    common_states = [
        jnp.zeros(10),
        jnp.ones(10),
        jnp.random.normal(size=(10,))
    ]
    
    common_actions = [
        jnp.zeros(5),
        jnp.ones(5),
        jnp.random.normal(size=(5,))
    ]
    
    for state in common_states:
        for action in common_actions:
            try:
                # Pre-compute reward
                reward = contract.compute_reward(state, action)
                
                # Pre-compute violations
                violations = contract.check_violations(state, action)
                
                # Cache results
                contract_hash = contract.compute_hash()
                state_hash = hashlib.md5(str(state).encode()).hexdigest()
                action_hash = hashlib.md5(str(action).encode()).hexdigest()
                
                await reward_cache.set_reward(
                    contract_hash, state_hash, action_hash, reward
                )
                await constraint_cache.set_violations(
                    contract_hash, state_hash, action_hash, violations
                )
                
            except Exception as e:
                logger.warning(f"Failed to warm cache for state/action pair: {e}")
    
    logger.info("Cache warming completed")


async def get_cache_health() -> Dict[str, Any]:
    """Get health status of all caches."""
    return {
        'contract_cache': contract_cache.get_stats(),
        'redis_health': await redis_connection.health_check()
    }