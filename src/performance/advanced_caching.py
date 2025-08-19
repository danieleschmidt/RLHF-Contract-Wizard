#!/usr/bin/env python3
"""
Advanced Multi-Level Caching System for RLHF Contract Wizard

Implements sophisticated caching strategies including adaptive algorithms,
distributed caching, intelligent prefetching, and cache optimization for
high-performance production deployments.
"""

import time
import asyncio
import hashlib
import pickle
import json
import threading
import random
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque, OrderedDict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import statistics
import heapq
import weakref


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"      # In-process memory cache
    L2_SHARED = "l2_shared"      # Shared memory cache
    L3_REDIS = "l3_redis"        # Redis distributed cache
    L4_DISK = "l4_disk"          # Persistent disk cache
    L5_NETWORK = "l5_network"    # Network/CDN cache


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    ARC = "arc"                  # Adaptive Replacement Cache
    CLOCK = "clock"              # Clock algorithm
    RANDOM = "random"            # Random eviction
    TTL = "ttl"                  # Time-To-Live based
    SIZE_BASED = "size_based"    # Size-based eviction
    INTELLIGENT = "intelligent"   # ML-based eviction


class CacheStrategy(Enum):
    """Caching strategies."""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    REFRESH_AHEAD = "refresh_ahead"
    CACHE_ASIDE = "cache_aside"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    hit_ratio: float = 0.0
    miss_ratio: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update access information."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class AdaptiveCache:
    """
    Adaptive cache that adjusts its behavior based on access patterns.
    """
    
    def __init__(
        self,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        eviction_policy: EvictionPolicy = EvictionPolicy.INTELLIGENT,
        enable_analytics: bool = True
    ):
        self.max_size_bytes = max_size_bytes
        self.eviction_policy = eviction_policy
        self.enable_analytics = enable_analytics
        
        # Storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU
        
        # ARC algorithm state
        self.arc_t1 = OrderedDict()  # Recent cache entries
        self.arc_t2 = OrderedDict()  # Frequent cache entries
        self.arc_b1 = OrderedDict()  # Ghost entries for T1
        self.arc_b2 = OrderedDict()  # Ghost entries for T2
        self.arc_p = 0  # Target size for T1
        
        # Metrics
        self.metrics = CacheMetrics()
        self.detailed_metrics: deque = deque(maxlen=1000)
        
        # Access pattern analysis
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.predictive_prefetch_candidates: Set[str] = set()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background optimization
        self.optimization_thread: Optional[threading.Thread] = None
        self.optimization_active = False
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with pattern learning."""
        start_time = time.time()
        
        with self.lock:
            # Record access pattern
            if self.enable_analytics:
                self.access_patterns[key].append(datetime.now())
                # Keep only recent accesses
                cutoff = datetime.now() - timedelta(hours=1)
                self.access_patterns[key] = [
                    access_time for access_time in self.access_patterns[key]
                    if access_time > cutoff
                ]
            
            if key in self.entries:
                entry = self.entries[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self.metrics.misses += 1
                    return None
                
                # Update access information
                entry.touch()
                self._update_access_order(key)
                
                # Update metrics
                self.metrics.hits += 1
                access_time = (time.time() - start_time) * 1000
                self._update_avg_access_time(access_time)
                
                return entry.value
            else:
                self.metrics.misses += 1
                self._trigger_predictive_prefetch(key)
                return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Put value in cache with intelligent eviction."""
        
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default size estimate
            
            # Check if we need to make space
            if not self._ensure_space(size_bytes):
                return False  # Cannot make enough space
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl_seconds,
                tags=tags or set()
            )
            
            # Store entry
            self.entries[key] = entry
            self.metrics.size_bytes += size_bytes
            
            # Update access structures
            self._update_access_order(key)
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self.lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False
    
    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with specified tags."""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.entries.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure enough space is available."""
        while (self.metrics.size_bytes + required_bytes > self.max_size_bytes and 
               self.entries):
            
            if not self._evict_one_entry():
                return False  # Cannot evict any more entries
        
        return True
    
    def _evict_one_entry(self) -> bool:
        """Evict one entry based on current policy."""
        if not self.entries:
            return False
        
        if self.eviction_policy == EvictionPolicy.LRU:
            key = next(iter(self.access_order))
        elif self.eviction_policy == EvictionPolicy.LFU:
            key = min(self.entries.keys(), key=lambda k: self.frequency_counter[k])
        elif self.eviction_policy == EvictionPolicy.ARC:
            key = self._arc_evict()
        elif self.eviction_policy == EvictionPolicy.INTELLIGENT:
            key = self._intelligent_evict()
        elif self.eviction_policy == EvictionPolicy.RANDOM:
            key = random.choice(list(self.entries.keys()))
        else:  # TTL or SIZE_BASED
            key = self._evict_by_policy()
        
        if key:
            self._remove_entry(key)
            self.metrics.evictions += 1
            return True
        
        return False
    
    def _intelligent_evict(self) -> Optional[str]:
        """Intelligent eviction based on multiple factors."""
        if not self.entries:
            return None
        
        scores = {}
        now = datetime.now()
        
        for key, entry in self.entries.items():
            # Calculate composite score for eviction
            # Higher score = more likely to evict
            
            # Time since last access (normalized)
            time_score = (now - entry.last_accessed).total_seconds() / 3600  # hours
            
            # Frequency score (inverse of access count)
            freq_score = 1.0 / (entry.access_count + 1)
            
            # Size score (larger entries more likely to evict)
            size_score = entry.size_bytes / (1024 * 1024)  # MB
            
            # Age score
            age_score = (now - entry.created_at).total_seconds() / 86400  # days
            
            # Pattern-based score
            pattern_score = self._calculate_pattern_score(key)
            
            # Composite score with weights
            composite_score = (
                time_score * 0.3 +
                freq_score * 0.2 +
                size_score * 0.2 +
                age_score * 0.1 +
                pattern_score * 0.2
            )
            
            scores[key] = composite_score
        
        # Return key with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _calculate_pattern_score(self, key: str) -> float:
        """Calculate pattern-based score for eviction."""
        if key not in self.access_patterns:
            return 1.0  # High score for unknown patterns
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return 1.0
        
        # Calculate access frequency
        time_span = (accesses[-1] - accesses[0]).total_seconds()
        if time_span == 0:
            return 0.0
        
        frequency = len(accesses) / time_span  # accesses per second
        
        # Lower frequency = higher eviction score
        return 1.0 / (frequency * 3600 + 1)  # Normalize to hourly
    
    def _arc_evict(self) -> Optional[str]:
        """ARC (Adaptive Replacement Cache) eviction."""
        # Simplified ARC implementation
        if self.arc_t1:
            return next(iter(self.arc_t1))
        elif self.arc_t2:
            return next(iter(self.arc_t2))
        return None
    
    def _evict_by_policy(self) -> Optional[str]:
        """Evict based on TTL or size policy."""
        now = datetime.now()
        
        # First try to evict expired entries
        for key, entry in self.entries.items():
            if entry.is_expired():
                return key
        
        # Then evict by size (largest first)
        if self.entries:
            return max(self.entries.keys(), 
                      key=lambda k: self.entries[k].size_bytes)
        
        return None
    
    def _remove_entry(self, key: str):
        """Remove entry and update all tracking structures."""
        if key in self.entries:
            entry = self.entries[key]
            self.metrics.size_bytes -= entry.size_bytes
            del self.entries[key]
        
        if key in self.access_order:
            del self.access_order[key]
        
        if key in self.frequency_counter:
            del self.frequency_counter[key]
        
        # Remove from ARC structures
        for arc_dict in [self.arc_t1, self.arc_t2, self.arc_b1, self.arc_b2]:
            if key in arc_dict:
                del arc_dict[key]
    
    def _update_access_order(self, key: str):
        """Update access order for LRU."""
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = datetime.now()
        
        # Update frequency counter
        self.frequency_counter[key] += 1
    
    def _update_avg_access_time(self, access_time_ms: float):
        """Update average access time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self.metrics.avg_access_time_ms == 0:
            self.metrics.avg_access_time_ms = access_time_ms
        else:
            self.metrics.avg_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.metrics.avg_access_time_ms
            )
    
    def _trigger_predictive_prefetch(self, key: str):
        """Trigger predictive prefetching based on access patterns."""
        if not self.enable_analytics:
            return
        
        # Analyze access patterns to predict related keys
        # This is a simplified implementation
        base_key = key.split(':')[0] if ':' in key else key
        
        # Look for similar keys in access patterns
        related_keys = [
            k for k in self.access_patterns.keys()
            if k.startswith(base_key) and k != key
        ]
        
        # Add to prefetch candidates
        self.predictive_prefetch_candidates.update(related_keys[:5])
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics."""
        with self.lock:
            total_requests = self.metrics.hits + self.metrics.misses
            if total_requests > 0:
                self.metrics.hit_ratio = self.metrics.hits / total_requests
                self.metrics.miss_ratio = self.metrics.misses / total_requests
            
            return self.metrics
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        with self.lock:
            stats = {
                "basic_metrics": asdict(self.get_metrics()),
                "entry_count": len(self.entries),
                "size_utilization": self.metrics.size_bytes / self.max_size_bytes,
                "eviction_policy": self.eviction_policy.value,
                "access_patterns": {
                    key: len(accesses)
                    for key, accesses in self.access_patterns.items()
                },
                "prefetch_candidates": len(self.predictive_prefetch_candidates),
                "top_entries_by_access": [
                    (key, entry.access_count)
                    for key, entry in sorted(
                        self.entries.items(),
                        key=lambda x: x[1].access_count,
                        reverse=True
                    )[:10]
                ]
            }
            
            return stats
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.entries.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.arc_t1.clear()
            self.arc_t2.clear()
            self.arc_b1.clear()
            self.arc_b2.clear()
            self.metrics = CacheMetrics()


class DistributedCacheNode:
    """
    Node in a distributed cache system with consistent hashing.
    """
    
    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        cache: AdaptiveCache
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.cache = cache
        self.is_healthy = True
        self.last_heartbeat = datetime.now()
        
    def get_hash_range(self, total_nodes: int, node_index: int) -> Tuple[int, int]:
        """Get hash range for this node in consistent hashing."""
        hash_space = 2**32
        range_size = hash_space // total_nodes
        start = node_index * range_size
        end = start + range_size - 1
        return start, end
    
    def should_handle_key(self, key: str, total_nodes: int, node_index: int) -> bool:
        """Check if this node should handle the given key."""
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        start, end = self.get_hash_range(total_nodes, node_index)
        return start <= key_hash <= end


class DistributedCache:
    """
    Distributed cache system with consistent hashing and replication.
    """
    
    def __init__(self, replication_factor: int = 2):
        self.nodes: List[DistributedCacheNode] = []
        self.replication_factor = replication_factor
        self.local_cache = AdaptiveCache(max_size_bytes=50 * 1024 * 1024)  # 50MB local
        
        # Connection pool for inter-node communication
        self.connection_pool: Dict[str, Any] = {}
        
    def add_node(self, node: DistributedCacheNode):
        """Add node to distributed cache."""
        self.nodes.append(node)
        self._rebalance_cache()
        
    def remove_node(self, node_id: str):
        """Remove node from distributed cache."""
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
        self._rebalance_cache()
        
    def get_nodes_for_key(self, key: str) -> List[DistributedCacheNode]:
        """Get nodes that should store the given key."""
        if not self.nodes:
            return []
        
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Sort nodes by their distance from the key hash
        sorted_nodes = sorted(
            enumerate(self.nodes),
            key=lambda x: abs(x[1].get_hash_range(len(self.nodes), x[0])[0] - key_hash)
        )
        
        # Return the closest nodes up to replication factor
        return [node for _, node in sorted_nodes[:self.replication_factor]]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try local cache first
        local_value = self.local_cache.get(key)
        if local_value is not None:
            return local_value
        
        # Try distributed nodes
        nodes = self.get_nodes_for_key(key)
        
        for node in nodes:
            if node.is_healthy:
                try:
                    value = await self._get_from_node(node, key)
                    if value is not None:
                        # Cache locally for faster future access
                        self.local_cache.put(key, value, ttl_seconds=300)  # 5 min TTL
                        return value
                except Exception as e:
                    logging.warning(f"Failed to get from node {node.node_id}: {e}")
                    node.is_healthy = False
        
        return None
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in distributed cache."""
        # Store in local cache
        self.local_cache.put(key, value, ttl_seconds=ttl_seconds)
        
        # Store in distributed nodes
        nodes = self.get_nodes_for_key(key)
        successful_puts = 0
        
        for node in nodes:
            if node.is_healthy:
                try:
                    success = await self._put_to_node(node, key, value, ttl_seconds)
                    if success:
                        successful_puts += 1
                except Exception as e:
                    logging.warning(f"Failed to put to node {node.node_id}: {e}")
                    node.is_healthy = False
        
        # Require majority of replicas to succeed
        return successful_puts >= (self.replication_factor + 1) // 2
    
    async def _get_from_node(self, node: DistributedCacheNode, key: str) -> Optional[Any]:
        """Get value from specific node."""
        # In real implementation, this would use HTTP/TCP connection
        return node.cache.get(key)
    
    async def _put_to_node(
        self,
        node: DistributedCacheNode,
        key: str,
        value: Any,
        ttl_seconds: Optional[int]
    ) -> bool:
        """Put value to specific node."""
        # In real implementation, this would use HTTP/TCP connection
        return node.cache.put(key, value, ttl_seconds=ttl_seconds)
    
    def _rebalance_cache(self):
        """Rebalance cache after node changes."""
        # In real implementation, would migrate data between nodes
        logging.info(f"Rebalancing cache with {len(self.nodes)} nodes")


class IntelligentPrefetcher:
    """
    Intelligent prefetching system that learns access patterns.
    """
    
    def __init__(self, cache: Union[AdaptiveCache, DistributedCache]):
        self.cache = cache
        self.access_sequences: deque = deque(maxlen=10000)
        self.pattern_models: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.prefetch_queue: deque = deque(maxlen=1000)
        
        # Background prefetching
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetching_active = False
        
    def record_access(self, key: str):
        """Record access for pattern learning."""
        self.access_sequences.append((key, datetime.now()))
        self._update_patterns(key)
        
    def _update_patterns(self, current_key: str):
        """Update access pattern models."""
        # Look at recent accesses to find patterns
        recent_accesses = [
            key for key, timestamp in list(self.access_sequences)[-10:]
            if (datetime.now() - timestamp).total_seconds() < 60  # Last minute
        ]
        
        if len(recent_accesses) < 2:
            return
        
        # Update transition probabilities
        for i in range(len(recent_accesses) - 1):
            prev_key = recent_accesses[i]
            next_key = recent_accesses[i + 1]
            
            # Exponential decay for old patterns
            for key in self.pattern_models[prev_key]:
                self.pattern_models[prev_key][key] *= 0.99
            
            # Increase probability for observed transition
            self.pattern_models[prev_key][next_key] += 1.0
        
        # Trigger prefetch prediction
        self._predict_and_queue_prefetch(current_key)
    
    def _predict_and_queue_prefetch(self, current_key: str):
        """Predict next accesses and queue for prefetching."""
        if current_key not in self.pattern_models:
            return
        
        # Get top predictions
        predictions = self.pattern_models[current_key]
        if not predictions:
            return
        
        # Sort by probability
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Queue top predictions for prefetching
        for predicted_key, probability in sorted_predictions[:3]:
            if probability > 0.5:  # Minimum confidence threshold
                self.prefetch_queue.append({
                    'key': predicted_key,
                    'probability': probability,
                    'triggered_by': current_key,
                    'timestamp': datetime.now()
                })
    
    def start_prefetching(self, data_loader: Callable[[str], Any]):
        """Start background prefetching."""
        if self.prefetching_active:
            return
        
        self.prefetching_active = True
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            args=(data_loader,),
            daemon=True
        )
        self.prefetch_thread.start()
    
    def stop_prefetching(self):
        """Stop background prefetching."""
        self.prefetching_active = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5)
    
    def _prefetch_loop(self, data_loader: Callable[[str], Any]):
        """Background prefetching loop."""
        while self.prefetching_active:
            try:
                if self.prefetch_queue:
                    prefetch_item = self.prefetch_queue.popleft()
                    key = prefetch_item['key']
                    
                    # Check if already in cache
                    if isinstance(self.cache, AdaptiveCache):
                        if key not in self.cache.entries:
                            # Load and cache the data
                            try:
                                data = data_loader(key)
                                if data is not None:
                                    self.cache.put(key, data, ttl_seconds=1800)  # 30 min TTL
                                    logging.debug(f"Prefetched key: {key}")
                            except Exception as e:
                                logging.warning(f"Failed to prefetch {key}: {e}")
                    
                    time.sleep(0.1)  # Small delay between prefetches
                else:
                    time.sleep(1)  # Wait when queue is empty
                    
            except Exception as e:
                logging.error(f"Error in prefetch loop: {e}")
                time.sleep(1)


class CacheManager:
    """
    High-level cache manager coordinating all caching components.
    """
    
    def __init__(self):
        # Multi-level cache hierarchy
        self.l1_cache = AdaptiveCache(
            max_size_bytes=50 * 1024 * 1024,  # 50MB
            eviction_policy=EvictionPolicy.INTELLIGENT
        )
        
        self.l2_cache = AdaptiveCache(
            max_size_bytes=200 * 1024 * 1024,  # 200MB
            eviction_policy=EvictionPolicy.ARC
        )
        
        # Distributed cache (would be Redis in production)
        self.distributed_cache = DistributedCache(replication_factor=2)
        
        # Intelligent prefetcher
        self.prefetcher = IntelligentPrefetcher(self.l1_cache)
        
        # Cache strategies by operation type
        self.strategies: Dict[str, CacheStrategy] = {
            "contract_read": CacheStrategy.READ_THROUGH,
            "contract_write": CacheStrategy.WRITE_THROUGH,
            "verification_result": CacheStrategy.CACHE_ASIDE,
            "quantum_planning": CacheStrategy.REFRESH_AHEAD,
        }
        
        # Statistics
        self.operation_stats: Dict[str, CacheMetrics] = defaultdict(CacheMetrics)
        
    async def get(
        self,
        key: str,
        operation_type: str = "default",
        data_loader: Optional[Callable[[str], Any]] = None
    ) -> Optional[Any]:
        """Get value with multi-level cache lookup."""
        start_time = time.time()
        
        # Record access for prefetching
        self.prefetcher.record_access(key)
        
        # L1 cache lookup
        value = self.l1_cache.get(key)
        if value is not None:
            self._update_operation_stats(operation_type, hit=True, access_time=time.time() - start_time)
            return value
        
        # L2 cache lookup
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.put(key, value, ttl_seconds=300)
            self._update_operation_stats(operation_type, hit=True, access_time=time.time() - start_time)
            return value
        
        # Distributed cache lookup
        value = await self.distributed_cache.get(key)
        if value is not None:
            # Promote to L2 and L1
            self.l2_cache.put(key, value, ttl_seconds=1800)
            self.l1_cache.put(key, value, ttl_seconds=300)
            self._update_operation_stats(operation_type, hit=True, access_time=time.time() - start_time)
            return value
        
        # Cache miss - load data if loader provided
        if data_loader:
            strategy = self.strategies.get(operation_type, CacheStrategy.CACHE_ASIDE)
            
            if strategy in [CacheStrategy.READ_THROUGH, CacheStrategy.CACHE_ASIDE]:
                try:
                    value = data_loader(key)
                    if value is not None:
                        await self.put(key, value, operation_type=operation_type)
                        self._update_operation_stats(operation_type, hit=False, access_time=time.time() - start_time)
                        return value
                except Exception as e:
                    logging.error(f"Error loading data for key {key}: {e}")
        
        self._update_operation_stats(operation_type, hit=False, access_time=time.time() - start_time)
        return None
    
    async def put(
        self,
        key: str,
        value: Any,
        operation_type: str = "default",
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Put value with appropriate caching strategy."""
        strategy = self.strategies.get(operation_type, CacheStrategy.CACHE_ASIDE)
        
        success = True
        
        if strategy in [CacheStrategy.WRITE_THROUGH, CacheStrategy.CACHE_ASIDE]:
            # Store in all cache levels
            self.l1_cache.put(key, value, ttl_seconds=ttl_seconds)
            self.l2_cache.put(key, value, ttl_seconds=ttl_seconds)
            success &= await self.distributed_cache.put(key, value, ttl_seconds=ttl_seconds)
            
        elif strategy == CacheStrategy.WRITE_BACK:
            # Store in L1 only, will be written back later
            self.l1_cache.put(key, value, ttl_seconds=ttl_seconds)
            
        elif strategy == CacheStrategy.WRITE_AROUND:
            # Skip cache, store in distributed only
            success = await self.distributed_cache.put(key, value, ttl_seconds=ttl_seconds)
        
        return success
    
    def invalidate(self, key: str) -> bool:
        """Invalidate key from all cache levels."""
        success = True
        success &= self.l1_cache.invalidate(key)
        success &= self.l2_cache.invalidate(key)
        # Would also invalidate from distributed cache
        return success
    
    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with specified tags."""
        total_invalidated = 0
        total_invalidated += self.l1_cache.invalidate_by_tags(tags)
        total_invalidated += self.l2_cache.invalidate_by_tags(tags)
        return total_invalidated
    
    def start_background_processes(self, data_loader: Callable[[str], Any]):
        """Start background optimization processes."""
        self.prefetcher.start_prefetching(data_loader)
    
    def stop_background_processes(self):
        """Stop background optimization processes."""
        self.prefetcher.stop_prefetching()
    
    def _update_operation_stats(self, operation_type: str, hit: bool, access_time: float):
        """Update operation-specific statistics."""
        stats = self.operation_stats[operation_type]
        
        if hit:
            stats.hits += 1
        else:
            stats.misses += 1
        
        # Update average access time
        total_requests = stats.hits + stats.misses
        stats.avg_access_time_ms = (
            (stats.avg_access_time_ms * (total_requests - 1) + access_time * 1000) / 
            total_requests
        )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics."""
        return {
            "l1_cache": self.l1_cache.get_detailed_stats(),
            "l2_cache": self.l2_cache.get_detailed_stats(),
            "operation_stats": {
                op_type: asdict(stats)
                for op_type, stats in self.operation_stats.items()
            },
            "prefetcher": {
                "queue_size": len(self.prefetcher.prefetch_queue),
                "pattern_models": len(self.prefetcher.pattern_models),
                "active": self.prefetcher.prefetching_active
            },
            "distributed_cache": {
                "nodes": len(self.distributed_cache.nodes),
                "replication_factor": self.distributed_cache.replication_factor
            }
        }


# Example usage and demonstration
async def demonstrate_advanced_caching():
    """Demonstrate the advanced caching system."""
    
    print("🚀 Advanced Caching System Demonstration")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    # Mock data loader
    def mock_data_loader(key: str) -> Any:
        """Mock data loader for cache misses."""
        time.sleep(0.1)  # Simulate loading delay
        return f"data_for_{key}_{int(time.time())}"
    
    # Start background processes
    cache_manager.start_background_processes(mock_data_loader)
    
    # Test various cache operations
    print("\\n📝 Testing Cache Operations:")
    
    # Test cache misses and loads
    for i in range(10):
        key = f"contract_{i}"
        value = await cache_manager.get(key, "contract_read", mock_data_loader)
        print(f"  {key}: {value[:20]}..." if value else f"  {key}: None")
    
    # Test cache hits
    print("\\n🎯 Testing Cache Hits:")
    for i in range(5):
        key = f"contract_{i}"
        value = await cache_manager.get(key, "contract_read")
        print(f"  {key}: {'HIT' if value else 'MISS'}")
    
    # Test pattern-based prefetching
    print("\\n🔮 Testing Pattern Learning:")
    access_pattern = ["user_1", "profile_1", "settings_1"] * 3
    for key in access_pattern:
        await cache_manager.get(key, "user_data", mock_data_loader)
        await asyncio.sleep(0.1)
    
    # Test cache invalidation
    print("\\n🗑️ Testing Cache Invalidation:")
    await cache_manager.put("temp_data", "temporary", ttl_seconds=1)
    print(f"  Before TTL: {await cache_manager.get('temp_data')}")
    await asyncio.sleep(2)
    print(f"  After TTL: {await cache_manager.get('temp_data')}")
    
    # Get comprehensive statistics
    stats = cache_manager.get_comprehensive_stats()
    print(f"\\n📊 Cache Statistics:")
    print(f"  L1 Hit Ratio: {stats['l1_cache']['basic_metrics']['hit_ratio']:.2%}")
    print(f"  L1 Entries: {stats['l1_cache']['entry_count']}")
    print(f"  L1 Size Utilization: {stats['l1_cache']['size_utilization']:.2%}")
    print(f"  Prefetch Queue: {stats['prefetcher']['queue_size']} items")
    print(f"  Pattern Models: {stats['prefetcher']['pattern_models']} patterns")
    
    # Stop background processes
    cache_manager.stop_background_processes()
    
    print("\\n✅ Caching demonstration completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    asyncio.run(demonstrate_advanced_caching())