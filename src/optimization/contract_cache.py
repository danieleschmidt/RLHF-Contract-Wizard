"""
Advanced caching system for contract operations.

Provides intelligent caching with invalidation, compression, and performance optimization.
"""

import time
import threading
import pickle
import hashlib
from typing import Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None
    tags: set = field(default_factory=set)


class ContractCache:
    """
    High-performance cache for contract operations.
    
    Provides intelligent caching with multiple eviction policies,
    compression, and cache warming strategies.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 500,
        default_ttl: Optional[float] = 3600,  # 1 hour
        cache_policy: CachePolicy = CachePolicy.LRU,
        enable_compression: bool = True,
        compression_threshold: int = 1024  # Compress if > 1KB
    ):
        """
        Initialize contract cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            cache_policy: Cache eviction policy
            enable_compression: Whether to compress large values
            compression_threshold: Size threshold for compression
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache_policy = cache_policy
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list = []  # For LRU tracking
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_memory = 0
        
        # Cache warming
        self._warming_functions: Dict[str, Callable] = {}
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with automatic decompression.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return default
            
            # Check TTL expiration
            if self._is_expired(entry):
                self._remove_entry(key)
                self._misses += 1
                return default
            
            # Update access metadata
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Update LRU order
            if self.cache_policy == CachePolicy.LRU:
                self._access_order.remove(key)
                self._access_order.append(key)
            
            self._hits += 1
            
            # Decompress if needed
            value = self._decompress_value(entry.value)
            return value
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        tags: Optional[set] = None
    ) -> bool:
        """
        Set value in cache with optional compression.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live override
            tags: Tags for cache invalidation
            
        Returns:
            True if value was cached successfully
        """
        with self._lock:
            try:
                # Compress value if enabled and large enough
                compressed_value = self._compress_value(value)
                
                # Calculate size
                size_bytes = self._calculate_size(compressed_value)
                
                # Check if we need to make space
                if not self._make_space(size_bytes):
                    logger.warning(f"Could not make space for cache key: {key}")
                    return False
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    size_bytes=size_bytes,
                    ttl_seconds=ttl or self.default_ttl,
                    tags=tags or set()
                )
                
                # Remove old entry if exists
                if key in self._cache:
                    self._remove_entry(key)
                
                # Add new entry
                self._cache[key] = entry
                self._current_memory += size_bytes
                
                # Update access order
                if self.cache_policy == CachePolicy.LRU:
                    self._access_order.append(key)
                
                return True
                
            except Exception as e:
                logger.error(f"Error caching key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory = 0
            logger.info("Cache cleared")
    
    def invalidate_by_tags(self, tags: set) -> int:
        """
        Invalidate cache entries by tags.
        
        Args:
            tags: Tags to invalidate
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            logger.info(f"Invalidated {len(keys_to_remove)} entries by tags: {tags}")
            return len(keys_to_remove)
    
    def warm_cache(self, pattern: str) -> None:
        """
        Warm cache using registered warming functions.
        
        Args:
            pattern: Pattern to warm (e.g., "contracts", "verifications")
        """
        if pattern in self._warming_functions:
            try:
                warming_func = self._warming_functions[pattern]
                warming_data = warming_func()
                
                for key, value in warming_data.items():
                    self.set(key, value, tags={'warmed'})
                
                logger.info(f"Cache warmed with {len(warming_data)} entries for pattern: {pattern}")
                
            except Exception as e:
                logger.error(f"Error warming cache for pattern {pattern}: {e}")
    
    def register_warming_function(self, pattern: str, func: Callable[[], Dict[str, Any]]):
        """
        Register a function for cache warming.
        
        Args:
            pattern: Pattern name
            func: Function that returns dict of key-value pairs to cache
        """
        self._warming_functions[pattern] = func
        logger.info(f"Registered cache warming function for pattern: {pattern}")
    
    def _compress_value(self, value: Any) -> Union[bytes, Any]:
        """Compress value if enabled and above threshold."""
        if not self.enable_compression:
            return value
        
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            
            if len(serialized) < self.compression_threshold:
                return value  # Don't compress small values
            
            # Compress
            import gzip
            compressed = gzip.compress(serialized)
            
            # Only use compression if it actually reduces size
            if len(compressed) < len(serialized):
                return ('compressed', compressed)
            else:
                return value
                
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value
    
    def _decompress_value(self, value: Union[bytes, Tuple, Any]) -> Any:
        """Decompress value if compressed."""
        if isinstance(value, tuple) and len(value) == 2 and value[0] == 'compressed':
            try:
                import gzip
                decompressed = gzip.decompress(value[1])
                return pickle.loads(decompressed)
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                return None
        
        return value
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, tuple) and len(value) == 2 and value[0] == 'compressed':
                return len(value[1])  # Compressed size
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default size estimate
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl_seconds is None:
            return False
        
        age = time.time() - entry.created_at
        return age > entry.ttl_seconds
    
    def _make_space(self, needed_bytes: int) -> bool:
        """Make space in cache using configured policy."""
        # Check size limits
        if len(self._cache) >= self.max_size or self._current_memory + needed_bytes > self.max_memory_bytes:
            return self._evict_entries(needed_bytes)
        
        return True
    
    def _evict_entries(self, needed_bytes: int) -> bool:
        """Evict entries based on cache policy."""
        evicted_bytes = 0
        evicted_count = 0
        
        while (len(self._cache) >= self.max_size or 
               self._current_memory + needed_bytes > self.max_memory_bytes):
            
            if not self._cache:
                break
            
            # Choose eviction candidate based on policy
            if self.cache_policy == CachePolicy.LRU:
                key_to_evict = self._find_lru_key()
            elif self.cache_policy == CachePolicy.LFU:
                key_to_evict = self._find_lfu_key()
            elif self.cache_policy == CachePolicy.TTL:
                key_to_evict = self._find_expired_key()
            else:
                key_to_evict = self._find_largest_key()
            
            if key_to_evict:
                entry = self._cache[key_to_evict]
                evicted_bytes += entry.size_bytes
                evicted_count += 1
                self._remove_entry(key_to_evict)
            else:
                break  # No more entries to evict
        
        self._evictions += evicted_count
        
        if evicted_count > 0:
            logger.debug(f"Evicted {evicted_count} entries, freed {evicted_bytes} bytes")
        
        return self._current_memory + needed_bytes <= self.max_memory_bytes
    
    def _find_lru_key(self) -> Optional[str]:
        """Find least recently used key."""
        if self._access_order:
            return self._access_order[0]
        return None
    
    def _find_lfu_key(self) -> Optional[str]:
        """Find least frequently used key."""
        if not self._cache:
            return None
        
        min_count = float('inf')
        lfu_key = None
        
        for key, entry in self._cache.items():
            if entry.access_count < min_count:
                min_count = entry.access_count
                lfu_key = key
        
        return lfu_key
    
    def _find_expired_key(self) -> Optional[str]:
        """Find an expired key."""
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                return key
        
        # If no expired keys, fall back to LRU
        return self._find_lru_key()
    
    def _find_largest_key(self) -> Optional[str]:
        """Find the largest entry."""
        if not self._cache:
            return None
        
        max_size = 0
        largest_key = None
        
        for key, entry in self._cache.items():
            if entry.size_bytes > max_size:
                max_size = entry.size_bytes
                largest_key = key
        
        return largest_key
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update metadata."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            del self._cache[key]
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'current_entries': len(self._cache),
                'max_entries': self.max_size,
                'current_memory_mb': self._current_memory / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'memory_utilization': self._current_memory / self.max_memory_bytes,
                'cache_policy': self.cache_policy.value,
                'compression_enabled': self.enable_compression
            }
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize cache by removing expired entries and defragmenting."""
        with self._lock:
            initial_count = len(self._cache)
            initial_memory = self._current_memory
            
            # Remove expired entries
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            # Compress values that weren't compressed initially
            if self.enable_compression:
                for key, entry in self._cache.items():
                    if not isinstance(entry.value, tuple) or entry.value[0] != 'compressed':
                        old_size = entry.size_bytes
                        compressed_value = self._compress_value(entry.value)
                        new_size = self._calculate_size(compressed_value)
                        
                        if new_size < old_size:
                            self._current_memory -= old_size
                            entry.value = compressed_value
                            entry.size_bytes = new_size
                            self._current_memory += new_size
            
            final_count = len(self._cache)
            final_memory = self._current_memory
            
            optimization_results = {
                'expired_removed': len(expired_keys),
                'entries_before': initial_count,
                'entries_after': final_count,
                'memory_before_mb': initial_memory / 1024 / 1024,
                'memory_after_mb': final_memory / 1024 / 1024,
                'memory_freed_mb': (initial_memory - final_memory) / 1024 / 1024
            }
            
            logger.info(f"Cache optimization completed: {optimization_results}")
            return optimization_results


# Global cache instances
reward_cache = ContractCache(max_size=1000, max_memory_mb=100)
verification_cache = ContractCache(max_size=500, max_memory_mb=50, default_ttl=7200)  # 2 hours
deployment_cache = ContractCache(max_size=200, max_memory_mb=25, default_ttl=86400)  # 24 hours