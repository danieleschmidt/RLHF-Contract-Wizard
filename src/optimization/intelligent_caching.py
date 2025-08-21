"""
Intelligent caching system with ML-based optimization.

Provides advanced caching with predictive prefetching, adaptive
eviction policies, and distributed cache coordination.
"""

import time
import hashlib
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import pickle
import json
from queue import Queue, PriorityQueue
import heapq
import statistics

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None
    LinearRegression = None
    StandardScaler = None

from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"    # ML-based adaptive policy
    PREDICTIVE = "predictive"  # Predictive eviction


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"      # In-memory cache
    L2_DISK = "l2_disk"          # Disk-based cache
    L3_DISTRIBUTED = "l3_distributed"  # Distributed cache
    L4_REMOTE = "l4_remote"      # Remote cache storage


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    priority: float = 0.0
    tags: Set[str] = field(default_factory=set)
    prediction_score: float = 0.0
    
    def __post_init__(self):
        if self.size_bytes == 0 and self.value is not None:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of cached value."""
        try:
            return len(pickle.dumps(self.value))
        except:
            return len(str(self.value).encode('utf-8'))
    
    def access(self):
        """Record an access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at
    
    def time_since_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    insertions: int = 0
    updates: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value for key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of entries."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with advanced eviction policies."""
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 1000.0,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.eviction_policy = eviction_policy
        
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self.stats = CacheStats()
        
        self._lock = threading.RLock()
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        with self._lock:
            if key not in self.entries:
                self.stats.misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check if expired
            if entry.is_expired():
                del self.entries[key]
                if key in self.access_order:
                    del self.access_order[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Update access metadata
            entry.access()
            self.frequency_counter[key] += 1
            
            # Update LRU order
            if key in self.access_order:
                del self.access_order[key]
            self.access_order[key] = time.time()
            
            self.stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: Set[str] = None) -> bool:
        """Set value for key."""
        with self._lock:
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                tags=tags or set()
            )
            
            # Check if we need to evict
            self._maybe_evict(entry.size_bytes)
            
            # Insert/update entry
            if key in self.entries:
                old_entry = self.entries[key]
                self.stats.size_bytes -= old_entry.size_bytes
                self.stats.updates += 1
            else:
                self.stats.insertions += 1
                self.stats.entry_count += 1
            
            self.entries[key] = entry
            self.access_order[key] = time.time()
            self.frequency_counter[key] += 1
            self.stats.size_bytes += entry.size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        with self._lock:
            if key not in self.entries:
                return False
            
            entry = self.entries[key]
            del self.entries[key]
            
            if key in self.access_order:
                del self.access_order[key]
            
            if key in self.frequency_counter:
                del self.frequency_counter[key]
            
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self.entries and not self.entries[key].is_expired()
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self.entries.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.stats = CacheStats()
    
    def size(self) -> int:
        """Get number of entries."""
        return len(self.entries)
    
    def _maybe_evict(self, new_entry_size: int):
        """Evict entries if necessary to make room."""
        # Check size limits
        while (
            len(self.entries) >= self.max_size or
            self.stats.size_bytes + new_entry_size > self.max_memory_bytes
        ):
            evicted = self._evict_one()
            if not evicted:
                break  # Nothing to evict
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self.entries:
            return False
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used
            key_to_evict = next(iter(self.access_order))
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(
                self.entries.keys(),
                key=lambda k: self.frequency_counter[k]
            )
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Evict expired entries first, then oldest
            expired_keys = [
                k for k, e in self.entries.items() if e.is_expired()
            ]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = min(
                    self.entries.keys(),
                    key=lambda k: self.entries[k].created_at
                )
        
        else:
            # Default to LRU
            key_to_evict = next(iter(self.access_order))
        
        # Perform eviction
        self.delete(key_to_evict)
        self.stats.evictions += 1
        
        self.logger.debug(f"Evicted cache entry: {key_to_evict}")
        return True
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                insertions=self.stats.insertions,
                updates=self.stats.updates,
                size_bytes=self.stats.size_bytes,
                entry_count=self.stats.entry_count
            )


class RedisCacheBackend(CacheBackend):
    """Redis-based distributed cache backend."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "rlhf_cache:"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available - install redis-py")
        
        self.key_prefix = key_prefix
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Keep binary for pickle
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            redis_key = self._make_key(key)
            data = self.redis_client.get(redis_key)
            
            if data is None:
                return None
            
            return pickle.loads(data)
            
        except Exception as e:
            self.logger.error(f"Redis get failed for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value for key."""
        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(value)
            
            if ttl is not None:
                return self.redis_client.setex(redis_key, int(ttl), data)
            else:
                return self.redis_client.set(redis_key, data)
                
        except Exception as e:
            self.logger.error(f"Redis set failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            redis_key = self._make_key(key)
            return bool(self.redis_client.delete(redis_key))
            
        except Exception as e:
            self.logger.error(f"Redis delete failed for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            redis_key = self._make_key(key)
            return bool(self.redis_client.exists(redis_key))
            
        except Exception as e:
            self.logger.error(f"Redis exists failed for key {key}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all entries with our prefix."""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                
        except Exception as e:
            self.logger.error(f"Redis clear failed: {e}")
    
    def size(self) -> int:
        """Get number of entries with our prefix."""
        try:
            pattern = f"{self.key_prefix}*"
            return len(self.redis_client.keys(pattern))
            
        except Exception as e:
            self.logger.error(f"Redis size failed: {e}")
            return 0


class MLCachePredictor:
    """
    Machine learning-based cache access predictor.
    
    Predicts which cache entries are likely to be accessed
    in the near future for intelligent prefetching and eviction.
    """
    
    def __init__(self):
        if not ML_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Training data
        self.features = []
        self.labels = []
        self.feature_history = defaultdict(list)
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, entry: CacheEntry) -> List[float]:
        """Extract features from cache entry for ML model."""
        if not self.enabled:
            return []
        
        current_time = time.time()
        
        features = [
            entry.access_count,
            entry.age_seconds(),
            entry.time_since_access(),
            entry.size_bytes,
            entry.priority,
            1.0 if entry.ttl is not None else 0.0,
            len(entry.tags),
            # Time-based features
            current_time % 86400,  # Time of day (seconds)
            current_time % 604800,  # Day of week (seconds)
        ]
        
        return features
    
    def record_access(self, key: str, entry: CacheEntry, was_accessed: bool):
        """Record access pattern for training."""
        if not self.enabled:
            return
        
        features = self.extract_features(entry)
        
        # Store for training
        self.features.append(features)
        self.labels.append(1.0 if was_accessed else 0.0)
        
        # Keep only recent data
        max_samples = 10000
        if len(self.features) > max_samples:
            self.features = self.features[-max_samples:]
            self.labels = self.labels[-max_samples:]
    
    def train_model(self):
        """Train the prediction model."""
        if not self.enabled or len(self.features) < 100:
            return
        
        try:
            X = np.array(self.features)
            y = np.array(self.labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate training accuracy
            predictions = self.model.predict(X_scaled)
            accuracy = np.mean((predictions > 0.5) == (y > 0.5))
            
            self.logger.info(f"Cache predictor trained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to train cache predictor: {e}")
    
    def predict_access_probability(self, entry: CacheEntry) -> float:
        """Predict probability that entry will be accessed soon."""
        if not self.enabled or not self.is_trained:
            return 0.5  # Default probability
        
        try:
            features = self.extract_features(entry)
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            probability = self.model.predict(X_scaled)[0]
            return max(0.0, min(1.0, probability))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Failed to predict access probability: {e}")
            return 0.5


class IntelligentCache:
    """
    Intelligent multi-level cache with ML-based optimization.
    
    Features:
    - Multi-level cache hierarchy (memory -> disk -> distributed)
    - ML-based predictive prefetching
    - Adaptive eviction policies
    - Distributed cache coordination
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        l1_config: Optional[Dict[str, Any]] = None,
        l2_config: Optional[Dict[str, Any]] = None,
        l3_config: Optional[Dict[str, Any]] = None,
        enable_ml_prediction: bool = True
    ):
        # Initialize cache levels
        self.levels: Dict[CacheLevel, CacheBackend] = {}
        
        # L1: In-memory cache
        l1_config = l1_config or {}
        self.levels[CacheLevel.L1_MEMORY] = MemoryCacheBackend(
            max_size=l1_config.get('max_size', 10000),
            max_memory_mb=l1_config.get('max_memory_mb', 1000.0),
            eviction_policy=EvictionPolicy(l1_config.get('eviction_policy', 'lru'))
        )
        
        # L2: Disk cache (simplified - just use memory for now)
        if l2_config:
            self.levels[CacheLevel.L2_DISK] = MemoryCacheBackend(
                max_size=l2_config.get('max_size', 50000),
                max_memory_mb=l2_config.get('max_memory_mb', 5000.0)
            )
        
        # L3: Distributed cache (Redis)
        if l3_config and REDIS_AVAILABLE:
            try:
                self.levels[CacheLevel.L3_DISTRIBUTED] = RedisCacheBackend(
                    host=l3_config.get('host', 'localhost'),
                    port=l3_config.get('port', 6379),
                    password=l3_config.get('password')
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        # ML predictor
        self.predictor = MLCachePredictor() if enable_ml_prediction else None
        
        # Statistics
        self.global_stats = CacheStats()
        
        # Background tasks
        self._cleanup_task = None
        self._training_task = None
        self._is_running = False
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str, tags: Set[str] = None) -> Optional[Any]:
        """Get value from cache with intelligent promotion."""
        tags = tags or set()
        
        # Try each cache level in order
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_DISK, CacheLevel.L3_DISTRIBUTED]:
            if level not in self.levels:
                continue
            
            backend = self.levels[level]
            value = backend.get(key)
            
            if value is not None:
                # Cache hit - promote to higher levels
                self._promote_to_higher_levels(key, value, level, tags)
                self.global_stats.hits += 1
                
                # Record access for ML
                if self.predictor and level == CacheLevel.L1_MEMORY:
                    if hasattr(backend, 'entries') and key in backend.entries:
                        entry = backend.entries[key]
                        self.predictor.record_access(key, entry, True)
                
                return value
        
        # Cache miss
        self.global_stats.misses += 1
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Set[str] = None,
        priority: float = 0.5
    ) -> bool:
        """Set value in cache with intelligent placement."""
        tags = tags or set()
        
        # Always store in L1 (memory) first
        if CacheLevel.L1_MEMORY in self.levels:
            success = self.levels[CacheLevel.L1_MEMORY].set(key, value, ttl, tags)
            if success:
                self.global_stats.insertions += 1
                
                # Optionally store in L2/L3 based on priority
                if priority > 0.7 and CacheLevel.L2_DISK in self.levels:
                    self.levels[CacheLevel.L2_DISK].set(key, value, ttl)
                
                if priority > 0.9 and CacheLevel.L3_DISTRIBUTED in self.levels:
                    self.levels[CacheLevel.L3_DISTRIBUTED].set(key, value, ttl)
                
                return True
        
        return False
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache levels."""
        success = False
        
        for backend in self.levels.values():
            if backend.delete(key):
                success = True
        
        return success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache level."""
        for backend in self.levels.values():
            if backend.exists(key):
                return True
        return False
    
    def clear(self, level: Optional[CacheLevel] = None) -> None:
        """Clear cache entries."""
        if level is not None:
            if level in self.levels:
                self.levels[level].clear()
        else:
            for backend in self.levels.values():
                backend.clear()
            self.global_stats = CacheStats()
    
    def _promote_to_higher_levels(
        self,
        key: str,
        value: Any,
        found_level: CacheLevel,
        tags: Set[str]
    ):
        """Promote cache entry to higher (faster) levels."""
        levels_order = [
            CacheLevel.L1_MEMORY,
            CacheLevel.L2_DISK,
            CacheLevel.L3_DISTRIBUTED
        ]
        
        found_index = levels_order.index(found_level)
        
        # Promote to all higher levels
        for i in range(found_index):
            level = levels_order[i]
            if level in self.levels:
                self.levels[level].set(key, value, tags=tags)
    
    async def start_background_tasks(self):
        """Start background optimization tasks."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._background_cleanup())
        
        # Start ML training task
        if self.predictor:
            self._training_task = asyncio.create_task(self._background_training())
        
        self.logger.info("Started cache background tasks")
    
    async def stop_background_tasks(self):
        """Stop background optimization tasks."""
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        if self._training_task:
            self._training_task.cancel()
        
        self.logger.info("Stopped cache background tasks")
    
    async def _background_cleanup(self):
        """Background task for cache cleanup and optimization."""
        while self._is_running:
            try:
                # Cleanup expired entries
                for level, backend in self.levels.items():
                    if hasattr(backend, 'entries'):
                        expired_keys = [
                            key for key, entry in backend.entries.items()
                            if entry.is_expired()
                        ]
                        
                        for key in expired_keys:
                            backend.delete(key)
                
                # Wait before next cleanup
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Background cleanup failed: {e}")
                await asyncio.sleep(60)
    
    async def _background_training(self):
        """Background task for ML model training."""
        while self._is_running:
            try:
                if self.predictor:
                    self.predictor.train_model()
                
                # Wait before next training
                await asyncio.sleep(300)  # Train every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Background training failed: {e}")
                await asyncio.sleep(300)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "global": {
                "hits": self.global_stats.hits,
                "misses": self.global_stats.misses,
                "hit_rate": self.global_stats.hit_rate,
                "insertions": self.global_stats.insertions
            },
            "levels": {}
        }
        
        for level, backend in self.levels.items():
            if hasattr(backend, 'get_stats'):
                level_stats = backend.get_stats()
                stats["levels"][level.value] = {
                    "hits": level_stats.hits,
                    "misses": level_stats.misses,
                    "hit_rate": level_stats.hit_rate,
                    "size": level_stats.entry_count,
                    "memory_bytes": level_stats.size_bytes,
                    "evictions": level_stats.evictions
                }
            else:
                stats["levels"][level.value] = {
                    "size": backend.size()
                }
        
        # ML predictor stats
        if self.predictor:
            stats["ml_predictor"] = {
                "enabled": self.predictor.enabled,
                "trained": self.predictor.is_trained,
                "training_samples": len(self.predictor.features)
            }
        
        return stats
    
    def optimize_cache_configuration(self) -> Dict[str, Any]:
        """Analyze cache performance and suggest optimizations."""
        stats = self.get_cache_stats()
        recommendations = []
        
        # Analyze hit rates
        global_hit_rate = stats["global"]["hit_rate"]
        
        if global_hit_rate < 0.5:
            recommendations.append({
                "type": "low_hit_rate",
                "description": f"Low cache hit rate: {global_hit_rate:.2%}",
                "suggestion": "Consider increasing cache size or adjusting TTL values"
            })
        
        # Analyze level distribution
        for level, level_stats in stats["levels"].items():
            if "hit_rate" in level_stats:
                hit_rate = level_stats["hit_rate"]
                if hit_rate < 0.3:
                    recommendations.append({
                        "type": "level_underutilized",
                        "description": f"Cache level {level} has low hit rate: {hit_rate:.2%}",
                        "suggestion": "Consider adjusting promotion policies or cache sizes"
                    })
        
        # Memory usage analysis
        l1_stats = stats["levels"].get("l1_memory", {})
        if "memory_bytes" in l1_stats:
            memory_mb = l1_stats["memory_bytes"] / (1024 * 1024)
            if memory_mb > 800:  # Close to 1GB default
                recommendations.append({
                    "type": "high_memory_usage",
                    "description": f"L1 cache using {memory_mb:.1f}MB",
                    "suggestion": "Consider increasing memory limit or more aggressive eviction"
                })
        
        return {
            "current_stats": stats,
            "recommendations": recommendations,
            "timestamp": time.time()
        }


# Global cache instance
_global_cache = None


def get_global_cache() -> IntelligentCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache


def configure_global_cache(
    l1_config: Optional[Dict[str, Any]] = None,
    l2_config: Optional[Dict[str, Any]] = None,
    l3_config: Optional[Dict[str, Any]] = None,
    enable_ml: bool = True
):
    """Configure global cache with custom settings."""
    global _global_cache
    _global_cache = IntelligentCache(
        l1_config=l1_config,
        l2_config=l2_config,
        l3_config=l3_config,
        enable_ml_prediction=enable_ml
    )


# Convenience decorators
def cached(
    ttl: Optional[float] = None,
    tags: Set[str] = None,
    priority: float = 0.5,
    key_func: Optional[Callable] = None
):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            
            # Try to get from cache
            cache = get_global_cache()
            result = cache.get(cache_key, tags)
            
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl, tags, priority)
            
            return result
        
        return wrapper
    return decorator
