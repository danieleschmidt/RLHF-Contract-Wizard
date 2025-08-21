"""
Distributed computing and parallelization for RLHF contracts.

Provides distributed contract execution, parallel processing,
and scalable computation across multiple nodes and GPUs.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue, Empty
import pickle
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import jit, pmap, vmap, tree_map
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from ..models.reward_contract import RewardContract
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class ComputeBackend(Enum):
    """Available compute backends."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"


class DistributionStrategy(Enum):
    """Distribution strategies for parallel execution."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


@dataclass
class ComputeNode:
    """Represents a compute node in the cluster."""
    node_id: str
    host: str
    port: int
    device_type: ComputeBackend
    device_count: int = 1
    memory_gb: float = 0.0
    compute_capability: float = 1.0
    is_available: bool = True
    current_load: float = 0.0
    last_heartbeat: float = 0.0
    
    def __post_init__(self):
        self.last_heartbeat = time.time()
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy based on heartbeat."""
        return (time.time() - self.last_heartbeat) < 30.0  # 30 second timeout
    
    def update_load(self, load: float):
        """Update current load and heartbeat."""
        self.current_load = load
        self.last_heartbeat = time.time()


@dataclass
class DistributedTask:
    """Represents a task for distributed execution."""
    task_id: str
    operation: str
    data: Any
    priority: float = 0.5
    estimated_compute_time: float = 1.0
    memory_requirement: float = 1.0  # GB
    requires_gpu: bool = False
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get actual execution time if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class DistributedContractExecutor:
    """
    Distributed executor for RLHF contract computations.
    
    Provides scalable, fault-tolerant execution across multiple
    compute nodes with automatic load balancing and failover.
    """
    
    def __init__(
        self,
        strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL,
        backend: ComputeBackend = ComputeBackend.CPU
    ):
        self.strategy = strategy
        self.backend = backend
        self.nodes: Dict[str, ComputeNode] = {}
        self.task_queue = Queue()
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.is_running = False
        
        # Thread pools for different types of work
        self.cpu_executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        self.io_executor = ThreadPoolExecutor(max_workers=10)
        
        # JAX distributed setup
        self._setup_jax_distributed()
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_compute_time': 0.0,
            'average_task_time': 0.0,
            'throughput_tasks_per_second': 0.0
        }
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    def _setup_jax_distributed(self):
        """Setup JAX for distributed computing."""
        try:
            # Get available devices
            self.devices = jax.devices()
            self.device_count = len(self.devices)
            
            self.logger.info(
                f"Initialized JAX with {self.device_count} devices: {[d.device_kind for d in self.devices]}"
            )
            
            # Setup sharding for multi-device execution
            if self.device_count > 1:
                self.sharding = PositionalSharding(self.devices)
                self.logger.info("Multi-device sharding configured")
            else:
                self.sharding = None
                
        except Exception as e:
            self.logger.error(f"Failed to setup JAX distributed: {e}")
            self.devices = [jax.devices()[0]]  # Fallback to single device
            self.device_count = 1
            self.sharding = None
    
    def add_compute_node(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 8000,
        device_type: ComputeBackend = ComputeBackend.CPU,
        device_count: int = 1,
        memory_gb: float = 8.0
    ):
        """Add a compute node to the cluster."""
        node = ComputeNode(
            node_id=node_id,
            host=host,
            port=port,
            device_type=device_type,
            device_count=device_count,
            memory_gb=memory_gb
        )
        
        self.nodes[node_id] = node
        self.logger.info(f"Added compute node {node_id} at {host}:{port}")
    
    def remove_compute_node(self, node_id: str):
        """Remove a compute node from the cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Removed compute node {node_id}")
    
    def submit_task(
        self,
        task_id: str,
        operation: str,
        data: Any,
        priority: float = 0.5,
        **kwargs
    ) -> str:
        """Submit a task for distributed execution."""
        task = DistributedTask(
            task_id=task_id,
            operation=operation,
            data=data,
            priority=priority,
            **kwargs
        )
        
        self.task_queue.put(task)
        self.logger.debug(f"Submitted task {task_id} with operation {operation}")
        return task_id
    
    def _select_optimal_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Select the optimal node for a task based on load and requirements."""
        available_nodes = [
            node for node in self.nodes.values()
            if node.is_available and node.is_healthy
        ]
        
        if not available_nodes:
            return None
        
        # Filter by requirements
        if task.requires_gpu:
            available_nodes = [
                node for node in available_nodes
                if node.device_type in [ComputeBackend.GPU, ComputeBackend.MULTI_GPU]
            ]
        
        # Filter by memory requirements
        available_nodes = [
            node for node in available_nodes
            if node.memory_gb >= task.memory_requirement
        ]
        
        if not available_nodes:
            return None
        
        # Select node with lowest current load
        return min(available_nodes, key=lambda n: n.current_load)
    
    async def execute_contract_batch(
        self,
        contract: RewardContract,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        batch_size: int = 1000
    ) -> jnp.ndarray:
        """
        Execute contract reward computation on a batch of state-action pairs.
        
        Uses distributed computing for large batches with automatic
        load balancing and fault tolerance.
        """
        total_samples = len(states)
        
        if total_samples <= batch_size or self.device_count == 1:
            # Small batch or single device - execute locally
            return self._execute_local_batch(contract, states, actions)
        
        # Large batch - distribute across devices/nodes
        return await self._execute_distributed_batch(
            contract, states, actions, batch_size
        )
    
    def _execute_local_batch(
        self,
        contract: RewardContract,
        states: jnp.ndarray,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Execute batch locally with JAX optimizations."""
        @jit
        def compute_rewards_vectorized(states_batch, actions_batch):
            # Vectorized reward computation
            return vmap(
                lambda s, a: contract.compute_reward(s, a, use_cache=False)
            )(states_batch, actions_batch)
        
        try:
            rewards = compute_rewards_vectorized(states, actions)
            return rewards
            
        except Exception as e:
            # Fallback to sequential execution
            self.logger.warning(f"Vectorized execution failed, falling back: {e}")
            rewards = []
            for state, action in zip(states, actions):
                try:
                    reward = contract.compute_reward(state, action)
                    rewards.append(reward)
                except Exception as inner_e:
                    self.logger.error(f"Individual reward computation failed: {inner_e}")
                    rewards.append(0.0)  # Default reward
            
            return jnp.array(rewards)
    
    async def _execute_distributed_batch(
        self,
        contract: RewardContract,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        batch_size: int
    ) -> jnp.ndarray:
        """Execute batch in distributed manner across devices."""
        total_samples = len(states)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        # Split data into chunks
        state_chunks = jnp.array_split(states, num_batches)
        action_chunks = jnp.array_split(actions, num_batches)
        
        # Create distributed computation function
        if self.sharding and self.device_count > 1:
            # Multi-device execution with sharding
            rewards = await self._execute_with_sharding(
                contract, state_chunks, action_chunks
            )
        else:
            # Multi-threaded execution
            rewards = await self._execute_with_threading(
                contract, state_chunks, action_chunks
            )
        
        return jnp.concatenate(rewards)
    
    async def _execute_with_sharding(
        self,
        contract: RewardContract,
        state_chunks: List[jnp.ndarray],
        action_chunks: List[jnp.ndarray]
    ) -> List[jnp.ndarray]:
        """Execute using JAX sharding across devices."""
        @pmap
        def compute_chunk_rewards(states_chunk, actions_chunk):
            return vmap(
                lambda s, a: contract.compute_reward(s, a, use_cache=False)
            )(states_chunk, actions_chunk)
        
        # Prepare data for pmap (must have leading axis = device count)
        max_chunk_size = max(len(chunk) for chunk in state_chunks)
        
        # Pad chunks to same size
        padded_state_chunks = []
        padded_action_chunks = []
        
        for i in range(self.device_count):
            if i < len(state_chunks):
                states = state_chunks[i]
                actions = action_chunks[i]
                
                # Pad to max_chunk_size
                if len(states) < max_chunk_size:
                    pad_size = max_chunk_size - len(states)
                    states = jnp.pad(states, ((0, pad_size), (0, 0)))
                    actions = jnp.pad(actions, ((0, pad_size), (0, 0)))
                
                padded_state_chunks.append(states)
                padded_action_chunks.append(actions)
            else:
                # Empty chunk for extra devices
                padded_state_chunks.append(jnp.zeros((max_chunk_size, states.shape[1])))
                padded_action_chunks.append(jnp.zeros((max_chunk_size, actions.shape[1])))
        
        # Stack for pmap
        states_array = jnp.stack(padded_state_chunks)
        actions_array = jnp.stack(padded_action_chunks)
        
        # Execute across devices
        try:
            results = compute_chunk_rewards(states_array, actions_array)
            
            # Unpad and split back to original chunks
            rewards = []
            for i, orig_chunk in enumerate(state_chunks):
                if i < len(results):
                    chunk_rewards = results[i][:len(orig_chunk)]
                    rewards.append(chunk_rewards)
                else:
                    break
            
            return rewards
            
        except Exception as e:
            self.logger.error(f"Sharded execution failed: {e}")
            # Fallback to threading
            return await self._execute_with_threading(
                contract, state_chunks, action_chunks
            )
    
    async def _execute_with_threading(
        self,
        contract: RewardContract,
        state_chunks: List[jnp.ndarray],
        action_chunks: List[jnp.ndarray]
    ) -> List[jnp.ndarray]:
        """Execute using threading for parallelism."""
        def compute_chunk(states_chunk, actions_chunk):
            """Compute rewards for a chunk."""
            try:
                return vmap(
                    lambda s, a: contract.compute_reward(s, a, use_cache=False)
                )(states_chunk, actions_chunk)
            except Exception as e:
                self.logger.error(f"Chunk computation failed: {e}")
                # Fallback to sequential
                rewards = []
                for state, action in zip(states_chunk, actions_chunk):
                    try:
                        reward = contract.compute_reward(state, action)
                        rewards.append(reward)
                    except Exception as inner_e:
                        self.logger.error(f"Individual computation failed: {inner_e}")
                        rewards.append(0.0)
                return jnp.array(rewards)
        
        # Submit chunks to thread pool
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                self.cpu_executor,
                compute_chunk,
                state_chunk,
                action_chunk
            )
            for state_chunk, action_chunk in zip(state_chunks, action_chunks)
        ]
        
        # Wait for all chunks to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Handle exceptions
        rewards = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Chunk {i} failed: {result}")
                # Create zero rewards for failed chunk
                chunk_size = len(state_chunks[i])
                rewards.append(jnp.zeros(chunk_size))
            else:
                rewards.append(result)
        
        return rewards
    
    def optimize_contract_compilation(
        self,
        contract: RewardContract,
        sample_states: jnp.ndarray,
        sample_actions: jnp.ndarray
    ):
        """Optimize contract compilation for distributed execution."""
        # Pre-compile contract functions with JAX
        self.logger.info("Optimizing contract compilation for distributed execution")
        
        try:
            # Warm up JIT compilation
            for i in range(min(10, len(sample_states))):
                _ = contract.compute_reward(
                    sample_states[i], 
                    sample_actions[i],
                    use_cache=False
                )
            
            # Pre-compile vectorized version
            if len(sample_states) > 1:
                @jit
                def vectorized_reward(states, actions):
                    return vmap(
                        lambda s, a: contract.compute_reward(s, a, use_cache=False)
                    )(states, actions)
                
                # Warm up vectorized compilation
                batch_size = min(100, len(sample_states))
                _ = vectorized_reward(
                    sample_states[:batch_size],
                    sample_actions[:batch_size]
                )
            
            self.logger.info("Contract compilation optimization completed")
            
        except Exception as e:
            self.logger.warning(f"Contract compilation optimization failed: {e}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        healthy_nodes = [n for n in self.nodes.values() if n.is_healthy]
        available_nodes = [n for n in healthy_nodes if n.is_available]
        
        total_devices = sum(n.device_count for n in available_nodes)
        total_memory = sum(n.memory_gb for n in available_nodes)
        average_load = (
            sum(n.current_load for n in available_nodes) / len(available_nodes)
            if available_nodes else 0.0
        )
        
        return {
            "cluster_info": {
                "total_nodes": len(self.nodes),
                "healthy_nodes": len(healthy_nodes),
                "available_nodes": len(available_nodes),
                "total_devices": total_devices,
                "total_memory_gb": total_memory,
                "average_load": average_load
            },
            "jax_info": {
                "device_count": self.device_count,
                "devices": [d.device_kind for d in self.devices],
                "sharding_enabled": self.sharding is not None
            },
            "task_queue": {
                "pending_tasks": self.task_queue.qsize(),
                "completed_tasks": len(self.completed_tasks)
            },
            "performance_metrics": self.metrics,
            "nodes": {
                node_id: {
                    "host": node.host,
                    "port": node.port,
                    "device_type": node.device_type.value,
                    "device_count": node.device_count,
                    "memory_gb": node.memory_gb,
                    "is_available": node.is_available,
                    "is_healthy": node.is_healthy,
                    "current_load": node.current_load,
                    "last_heartbeat": node.last_heartbeat
                }
                for node_id, node in self.nodes.items()
            }
        }
    
    def benchmark_performance(
        self,
        contract: RewardContract,
        test_sizes: List[int] = [100, 1000, 10000, 100000]
    ) -> Dict[str, Any]:
        """Benchmark distributed execution performance."""
        self.logger.info("Starting distributed execution benchmark")
        
        benchmark_results = {}
        
        for size in test_sizes:
            self.logger.info(f"Benchmarking with {size} samples")
            
            # Generate test data
            key = jax.random.PRNGKey(42)
            states = jax.random.normal(key, (size, 10))
            actions = jax.random.normal(key, (size, 5))
            
            # Benchmark local execution
            start_time = time.time()
            try:
                local_rewards = self._execute_local_batch(contract, states, actions)
                local_time = time.time() - start_time
                local_success = True
            except Exception as e:
                self.logger.error(f"Local execution failed for size {size}: {e}")
                local_time = float('inf')
                local_success = False
            
            # Benchmark distributed execution
            start_time = time.time()
            try:
                distributed_rewards = asyncio.run(
                    self.execute_contract_batch(contract, states, actions)
                )
                distributed_time = time.time() - start_time
                distributed_success = True
            except Exception as e:
                self.logger.error(f"Distributed execution failed for size {size}: {e}")
                distributed_time = float('inf')
                distributed_success = False
            
            # Calculate metrics
            speedup = local_time / distributed_time if distributed_success and local_success else 0.0
            throughput = size / distributed_time if distributed_success else 0.0
            
            benchmark_results[size] = {
                "local_time": local_time,
                "distributed_time": distributed_time,
                "speedup": speedup,
                "throughput_samples_per_second": throughput,
                "local_success": local_success,
                "distributed_success": distributed_success
            }
        
        self.logger.info("Benchmark completed")
        return benchmark_results
    
    def shutdown(self):
        """Shutdown the distributed executor."""
        self.is_running = False
        self.cpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        self.logger.info("Distributed executor shutdown completed")


class ContractShardingStrategy:
    """
    Advanced sharding strategies for large-scale contract execution.
    """
    
    @staticmethod
    def create_data_parallel_sharding(
        contract: RewardContract,
        devices: List[Any]
    ) -> Dict[str, Any]:
        """Create data parallel sharding for contract execution."""
        device_count = len(devices)
        
        return {
            "strategy": "data_parallel",
            "device_count": device_count,
            "sharding_spec": {
                "data_axis": 0,  # Shard along batch dimension
                "model_replicated": True
            },
            "batch_size_per_device": lambda total_batch: total_batch // device_count
        }
    
    @staticmethod
    def create_model_parallel_sharding(
        contract: RewardContract,
        devices: List[Any]
    ) -> Dict[str, Any]:
        """Create model parallel sharding for large contracts."""
        # This would be used for very large contracts that don't fit on single device
        return {
            "strategy": "model_parallel",
            "device_count": len(devices),
            "sharding_spec": {
                "stakeholder_sharding": True,  # Shard stakeholders across devices
                "constraint_sharding": True   # Shard constraints across devices
            }
        }


def create_distributed_executor(
    strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL,
    backend: ComputeBackend = None
) -> DistributedContractExecutor:
    """Factory function for creating distributed executors."""
    # Auto-detect backend if not specified
    if backend is None:
        try:
            devices = jax.devices()
            if any(d.device_kind == 'gpu' for d in devices):
                backend = ComputeBackend.MULTI_GPU if len(devices) > 1 else ComputeBackend.GPU
            elif any(d.device_kind == 'tpu' for d in devices):
                backend = ComputeBackend.TPU
            else:
                backend = ComputeBackend.CPU
        except:
            backend = ComputeBackend.CPU
    
    return DistributedContractExecutor(strategy, backend)
