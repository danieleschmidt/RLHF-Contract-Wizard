"""
Intelligent batch processing for contract operations.

Provides efficient batch processing with dynamic batching and load balancing.
"""

import time
import asyncio
from typing import List, Dict, Any, Callable, Optional, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies."""
    SIZE_BASED = "size_based"      # Fixed batch sizes
    TIME_BASED = "time_based"      # Time-based batching
    ADAPTIVE = "adaptive"          # Adaptive batching based on load
    PRIORITY_BASED = "priority"    # Priority-aware batching


@dataclass
class BatchItem:
    """Individual item in a batch."""
    id: str
    data: Any
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    items: List[BatchItem]
    results: List[Any]
    errors: List[Optional[str]]
    processing_time: float
    success_count: int
    error_count: int


class BatchProcessor:
    """
    Intelligent batch processor for contract operations.
    
    Provides dynamic batching with intelligent load balancing and
    performance optimization.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        max_batch_size: int = 128,
        batch_timeout: float = 1.0,
        strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
        enable_dynamic_sizing: bool = True,
        performance_target_ms: float = 100.0
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size
            max_batch_size: Maximum batch size
            batch_timeout: Maximum wait time for batch completion
            strategy: Batching strategy
            enable_dynamic_sizing: Enable dynamic batch size adjustment
            performance_target_ms: Target processing time per batch
        """
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.strategy = strategy
        self.enable_dynamic_sizing = enable_dynamic_sizing
        self.performance_target_ms = performance_target_ms
        
        # Batch management
        self._pending_items: List[BatchItem] = []
        self._batch_counter = 0
        
        # Performance tracking
        self._processing_times: List[float] = []
        self._batch_sizes: List[int] = []
        self._throughput_samples: List[float] = []
        
        # Dynamic sizing
        self._current_batch_size = batch_size
        self._adjustment_factor = 1.1  # Growth/shrink factor
        
    async def process_batch_async(
        self,
        items: List[Any],
        processor_func: Callable[[List[Any]], List[Any]],
        item_id_func: Optional[Callable[[Any], str]] = None,
        priority_func: Optional[Callable[[Any], int]] = None
    ) -> BatchResult:
        """
        Process a batch of items asynchronously.
        
        Args:
            items: Items to process
            processor_func: Function to process batch
            item_id_func: Function to extract item ID
            priority_func: Function to extract item priority
            
        Returns:
            Batch processing result
        """
        batch_id = f"batch_{self._batch_counter}"
        self._batch_counter += 1
        
        # Convert items to BatchItems
        batch_items = []
        for i, item in enumerate(items):
            item_id = item_id_func(item) if item_id_func else f"{batch_id}_item_{i}"
            priority = priority_func(item) if priority_func else 0
            
            batch_items.append(BatchItem(
                id=item_id,
                data=item,
                priority=priority
            ))
        
        # Sort by priority if using priority-based strategy
        if self.strategy == BatchStrategy.PRIORITY_BASED:
            batch_items.sort(key=lambda x: x.priority, reverse=True)
        
        # Process batch
        start_time = time.time()
        
        try:
            # Extract data for processing
            data_items = [item.data for item in batch_items]
            
            # Process batch (wrap sync function for async)
            if asyncio.iscoroutinefunction(processor_func):
                results = await processor_func(data_items)
            else:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, processor_func, data_items
                )
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Handle results
            if len(results) != len(batch_items):
                # Pad with None if results don't match items
                results.extend([None] * (len(batch_items) - len(results)))
            
            errors = [None if result is not None else "Processing failed" for result in results]
            success_count = sum(1 for result in results if result is not None)
            error_count = len(results) - success_count
            
            # Update performance tracking
            self._update_performance_metrics(processing_time, len(batch_items))
            
            return BatchResult(
                batch_id=batch_id,
                items=batch_items,
                results=results,
                errors=errors,
                processing_time=processing_time,
                success_count=success_count,
                error_count=error_count
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            # All items failed
            errors = [str(e)] * len(batch_items)
            results = [None] * len(batch_items)
            
            return BatchResult(
                batch_id=batch_id,
                items=batch_items,
                results=results,
                errors=errors,
                processing_time=processing_time,
                success_count=0,
                error_count=len(batch_items)
            )
    
    async def stream_process(
        self,
        item_stream: AsyncGenerator[Any, None],
        processor_func: Callable[[List[Any]], List[Any]],
        item_id_func: Optional[Callable[[Any], str]] = None,
        priority_func: Optional[Callable[[Any], int]] = None
    ) -> AsyncGenerator[BatchResult, None]:
        """
        Process streaming items in batches.
        
        Args:
            item_stream: Async generator of items
            processor_func: Function to process batches
            item_id_func: Function to extract item ID
            priority_func: Function to extract item priority
            
        Yields:
            Batch processing results
        """
        batch_buffer = []
        last_batch_time = time.time()
        
        async for item in item_stream:
            batch_buffer.append(item)
            current_time = time.time()
            
            # Check if we should process batch
            should_process = (
                len(batch_buffer) >= self._current_batch_size or
                (current_time - last_batch_time) >= self.batch_timeout or
                len(batch_buffer) >= self.max_batch_size
            )
            
            if should_process and batch_buffer:
                # Process current batch
                result = await self.process_batch_async(
                    batch_buffer,
                    processor_func,
                    item_id_func,
                    priority_func
                )
                
                yield result
                
                # Reset for next batch
                batch_buffer = []
                last_batch_time = current_time
        
        # Process remaining items
        if batch_buffer:
            result = await self.process_batch_async(
                batch_buffer,
                processor_func,
                item_id_func,
                priority_func
            )
            yield result
    
    def process_batch_sync(
        self,
        items: List[Any],
        processor_func: Callable[[List[Any]], List[Any]],
        item_id_func: Optional[Callable[[Any], str]] = None,
        priority_func: Optional[Callable[[Any], int]] = None
    ) -> BatchResult:
        """Synchronous batch processing wrapper."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self.process_batch_async(
                    items, processor_func, item_id_func, priority_func
                )
            )
        finally:
            loop.close()
    
    def _update_performance_metrics(self, processing_time_ms: float, batch_size: int):
        """Update performance metrics for dynamic optimization."""
        self._processing_times.append(processing_time_ms)
        self._batch_sizes.append(batch_size)
        
        # Keep only recent samples
        max_samples = 100
        if len(self._processing_times) > max_samples:
            self._processing_times.pop(0)
            self._batch_sizes.pop(0)
        
        # Calculate throughput (items/second)
        if processing_time_ms > 0:
            throughput = (batch_size * 1000) / processing_time_ms
            self._throughput_samples.append(throughput)
            
            if len(self._throughput_samples) > max_samples:
                self._throughput_samples.pop(0)
        
        # Adjust batch size if dynamic sizing enabled
        if self.enable_dynamic_sizing and len(self._processing_times) >= 5:
            self._adjust_batch_size(processing_time_ms)
    
    def _adjust_batch_size(self, processing_time_ms: float):
        """Dynamically adjust batch size based on performance."""
        if processing_time_ms > self.performance_target_ms * 1.5:
            # Processing too slow, reduce batch size
            new_size = max(
                self.batch_size // 2,
                int(self._current_batch_size / self._adjustment_factor)
            )
            if new_size != self._current_batch_size:
                self._current_batch_size = new_size
                logger.debug(f"Reduced batch size to {new_size} (time: {processing_time_ms:.1f}ms)")
        
        elif processing_time_ms < self.performance_target_ms * 0.5:
            # Processing fast, increase batch size
            new_size = min(
                self.max_batch_size,
                int(self._current_batch_size * self._adjustment_factor)
            )
            if new_size != self._current_batch_size:
                self._current_batch_size = new_size
                logger.debug(f"Increased batch size to {new_size} (time: {processing_time_ms:.1f}ms)")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get batch processing performance statistics."""
        if not self._processing_times:
            return {'message': 'No performance data available'}
        
        avg_time = sum(self._processing_times) / len(self._processing_times)
        avg_batch_size = sum(self._batch_sizes) / len(self._batch_sizes)
        avg_throughput = sum(self._throughput_samples) / len(self._throughput_samples) if self._throughput_samples else 0
        
        return {
            'current_batch_size': self._current_batch_size,
            'target_batch_size': self.batch_size,
            'max_batch_size': self.max_batch_size,
            'avg_processing_time_ms': avg_time,
            'avg_batch_size': avg_batch_size,
            'avg_throughput_items_per_sec': avg_throughput,
            'performance_target_ms': self.performance_target_ms,
            'total_batches_processed': len(self._processing_times),
            'strategy': self.strategy.value,
            'dynamic_sizing_enabled': self.enable_dynamic_sizing
        }
    
    def optimize_batch_size(self, target_items: int, time_budget_ms: float) -> int:
        """
        Optimize batch size for given constraints.
        
        Args:
            target_items: Total number of items to process
            time_budget_ms: Available time budget in milliseconds
            
        Returns:
            Recommended batch size
        """
        if not self._processing_times or not self._batch_sizes:
            return self._current_batch_size
        
        # Estimate processing time per item
        total_items = sum(self._batch_sizes)
        total_time = sum(self._processing_times)
        avg_time_per_item = total_time / total_items if total_items > 0 else 1.0
        
        # Calculate optimal batch size
        estimated_batches = max(1, int(target_items * avg_time_per_item / time_budget_ms))
        optimal_batch_size = min(
            self.max_batch_size,
            max(1, target_items // estimated_batches)
        )
        
        return optimal_batch_size
    
    def reset_performance_metrics(self):
        """Reset performance tracking metrics."""
        self._processing_times.clear()
        self._batch_sizes.clear()
        self._throughput_samples.clear()
        self._current_batch_size = self.batch_size
        logger.info("Performance metrics reset")


# Helper functions for common batch operations

async def batch_reward_computation(
    processor: BatchProcessor,
    contracts: List[Any],
    states: List[Any],
    actions: List[Any]
) -> List[float]:
    """
    Batch compute rewards for multiple contracts.
    
    Args:
        processor: Batch processor instance
        contracts: List of reward contracts
        states: List of states
        actions: List of actions
        
    Returns:
        List of computed rewards
    """
    # Combine inputs for batch processing
    batch_inputs = list(zip(contracts, states, actions))
    
    def compute_batch_rewards(inputs):
        """Batch reward computation function."""
        results = []
        for contract, state, action in inputs:
            try:
                reward = contract.compute_reward(state, action)
                results.append(reward)
            except Exception as e:
                logger.error(f"Reward computation failed: {e}")
                results.append(None)
        return results
    
    # Process batch
    result = await processor.process_batch_async(
        batch_inputs,
        compute_batch_rewards,
        item_id_func=lambda x: f"reward_{id(x[0])}_{id(x[1])}_{id(x[2])}"
    )
    
    return result.results


async def batch_constraint_validation(
    processor: BatchProcessor,
    contracts: List[Any],
    states: List[Any], 
    actions: List[Any]
) -> List[Dict[str, bool]]:
    """
    Batch validate constraints for multiple contracts.
    
    Args:
        processor: Batch processor instance
        contracts: List of reward contracts
        states: List of states
        actions: List of actions
        
    Returns:
        List of constraint violation dictionaries
    """
    batch_inputs = list(zip(contracts, states, actions))
    
    def validate_batch_constraints(inputs):
        """Batch constraint validation function."""
        results = []
        for contract, state, action in inputs:
            try:
                violations = contract.check_violations(state, action)
                results.append(violations)
            except Exception as e:
                logger.error(f"Constraint validation failed: {e}")
                results.append({})
        return results
    
    result = await processor.process_batch_async(
        batch_inputs,
        validate_batch_constraints,
        item_id_func=lambda x: f"constraints_{id(x[0])}_{id(x[1])}_{id(x[2])}"
    )
    
    return result.results


# Global batch processor instances
reward_batch_processor = BatchProcessor(
    batch_size=16,
    max_batch_size=64,
    batch_timeout=0.5,
    strategy=BatchStrategy.ADAPTIVE,
    performance_target_ms=50.0
)

verification_batch_processor = BatchProcessor(
    batch_size=8,
    max_batch_size=32,
    batch_timeout=2.0,
    strategy=BatchStrategy.PRIORITY_BASED,
    performance_target_ms=200.0
)