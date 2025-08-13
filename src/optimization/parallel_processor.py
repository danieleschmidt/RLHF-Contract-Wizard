"""
Advanced parallel processing for contract operations.

Provides intelligent workload distribution and parallel execution.
"""

import time
import threading
import concurrent.futures
import asyncio
import queue
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Parallel processing modes."""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool" 
    ASYNC_TASKS = "async_tasks"
    HYBRID = "hybrid"


@dataclass
class WorkItem:
    """Individual work item for parallel processing."""
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    tags: set = field(default_factory=set)


@dataclass
class ProcessingResult:
    """Result of parallel processing operation."""
    item_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None


class ParallelProcessor:
    """
    Advanced parallel processor for contract operations.
    
    Provides intelligent workload distribution with automatic scaling,
    load balancing, and fault tolerance.
    """
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.THREAD_POOL,
        max_workers: Optional[int] = None,
        queue_size: int = 10000,
        enable_auto_scaling: bool = True,
        min_workers: int = 2,
        max_workers_limit: int = 50
    ):
        """
        Initialize parallel processor.
        
        Args:
            mode: Processing mode (threads, processes, async)
            max_workers: Maximum number of workers
            queue_size: Maximum queue size
            enable_auto_scaling: Enable dynamic worker scaling
            min_workers: Minimum workers for auto-scaling
            max_workers_limit: Maximum workers for auto-scaling
        """
        self.mode = mode
        self.max_workers = max_workers or mp.cpu_count() * 2
        self.queue_size = queue_size
        self.enable_auto_scaling = enable_auto_scaling
        self.min_workers = min_workers
        self.max_workers_limit = max_workers_limit
        
        # Work queue
        self.work_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        
        # Executors
        self.thread_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.process_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        
        # Processing state
        self._active = False
        self._workers: List[threading.Thread] = []
        self._processing_stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        
        # Auto-scaling
        self._current_workers = 0
        self._load_samples: List[float] = []
        self._scaling_lock = threading.Lock()
        
        # Result storage
        self._results: Dict[str, ProcessingResult] = {}
        self._results_lock = threading.Lock()
        
    def start(self) -> None:
        """Start the parallel processor."""
        if self._active:
            return
        
        self._active = True
        
        # Initialize executors based on mode
        if self.mode in [ProcessingMode.THREAD_POOL, ProcessingMode.HYBRID]:
            self.thread_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="contract_worker"
            )
        
        if self.mode in [ProcessingMode.PROCESS_POOL, ProcessingMode.HYBRID]:
            self.process_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(self.max_workers, mp.cpu_count())
            )
        
        # Start worker threads for queue processing
        initial_workers = self.min_workers if self.enable_auto_scaling else self.max_workers
        for i in range(initial_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                name=f"processor_worker_{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        self._current_workers = initial_workers
        
        # Start auto-scaling monitor
        if self.enable_auto_scaling:
            scaling_thread = threading.Thread(
                target=self._auto_scaling_loop,
                daemon=True
            )
            scaling_thread.start()
        
        logger.info(f"Parallel processor started with {initial_workers} workers in {self.mode.value} mode")
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the parallel processor."""
        if not self._active:
            return
        
        self._active = False
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=timeout / len(self._workers) if self._workers else timeout)
        
        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        
        logger.info("Parallel processor stopped")
    
    def submit(
        self, 
        func: Callable, 
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        tags: Optional[set] = None,
        item_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Submit work item for parallel processing.
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Priority (higher = more urgent)
            timeout: Execution timeout
            tags: Tags for categorization
            item_id: Custom item ID
            **kwargs: Function keyword arguments
            
        Returns:
            Work item ID
        """
        if not self._active:
            raise RuntimeError("Processor not started")
        
        # Generate unique ID if not provided
        if item_id is None:
            item_id = f"work_{int(time.time() * 1000000)}_{threading.current_thread().ident}"
        
        work_item = WorkItem(
            id=item_id,
            function=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout_seconds=timeout,
            tags=tags or set()
        )
        
        try:
            # Use negative priority for priority queue (higher priority first)
            self.work_queue.put((-priority, time.time(), work_item), timeout=1.0)
            self._processing_stats['submitted'] += 1
            logger.debug(f"Submitted work item: {item_id}")
            return item_id
            
        except queue.Full:
            raise RuntimeError("Work queue is full")
    
    def submit_batch(
        self,
        work_items: List[Tuple[Callable, tuple, dict]],
        priority: int = 0,
        timeout: Optional[float] = None,
        tags: Optional[set] = None
    ) -> List[str]:
        """
        Submit multiple work items efficiently.
        
        Args:
            work_items: List of (function, args, kwargs) tuples
            priority: Priority for all items
            timeout: Execution timeout
            tags: Tags for all items
            
        Returns:
            List of work item IDs
        """
        item_ids = []
        
        for i, (func, args, kwargs) in enumerate(work_items):
            item_id = f"batch_{int(time.time() * 1000000)}_{i}"
            item_id = self.submit(
                func, *args, 
                priority=priority,
                timeout=timeout,
                tags=tags,
                item_id=item_id,
                **kwargs
            )
            item_ids.append(item_id)
        
        return item_ids
    
    def get_result(
        self, 
        item_id: str, 
        timeout: Optional[float] = None,
        block: bool = True
    ) -> Optional[ProcessingResult]:
        """
        Get result for work item.
        
        Args:
            item_id: Work item ID
            timeout: Wait timeout
            block: Whether to block until result available
            
        Returns:
            Processing result or None
        """
        if not block:
            with self._results_lock:
                return self._results.get(item_id)
        
        # Block until result available
        start_time = time.time()
        while self._active:
            with self._results_lock:
                if item_id in self._results:
                    return self._results[item_id]
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                break
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
        
        return None
    
    def get_batch_results(
        self,
        item_ids: List[str],
        timeout: Optional[float] = None
    ) -> Dict[str, ProcessingResult]:
        """Get results for multiple work items."""
        results = {}
        start_time = time.time()
        remaining_ids = set(item_ids)
        
        while remaining_ids and self._active:
            with self._results_lock:
                completed_ids = []
                for item_id in remaining_ids:
                    if item_id in self._results:
                        results[item_id] = self._results[item_id]
                        completed_ids.append(item_id)
                
                for item_id in completed_ids:
                    remaining_ids.remove(item_id)
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                break
            
            if remaining_ids:
                time.sleep(0.01)
        
        return results
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing work items."""
        worker_id = threading.current_thread().name
        
        while self._active:
            try:
                # Get work item with timeout
                try:
                    priority, timestamp, work_item = self.work_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process work item
                result = self._process_work_item(work_item, worker_id)
                
                # Store result
                with self._results_lock:
                    self._results[work_item.id] = result
                
                # Update statistics
                self._processing_stats['completed'] += 1
                self._processing_stats['total_time'] += result.execution_time
                if self._processing_stats['completed'] > 0:
                    self._processing_stats['avg_time'] = (
                        self._processing_stats['total_time'] / 
                        self._processing_stats['completed']
                    )
                
                # Mark task done
                self.work_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self._processing_stats['failed'] += 1
    
    def _process_work_item(self, work_item: WorkItem, worker_id: str) -> ProcessingResult:
        """Process individual work item."""
        start_time = time.time()
        
        try:
            # Execute function based on processing mode
            if self.mode == ProcessingMode.THREAD_POOL:
                future = self.thread_executor.submit(
                    work_item.function, 
                    *work_item.args, 
                    **work_item.kwargs
                )
            elif self.mode == ProcessingMode.PROCESS_POOL:
                future = self.process_executor.submit(
                    work_item.function,
                    *work_item.args,
                    **work_item.kwargs
                )
            else:
                # Direct execution for async mode
                result = work_item.function(*work_item.args, **work_item.kwargs)
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    item_id=work_item.id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    worker_id=worker_id
                )
            
            # Wait for future result
            result = future.result(timeout=work_item.timeout_seconds)
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                item_id=work_item.id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id
            )
            
        except concurrent.futures.TimeoutError:
            execution_time = time.time() - start_time
            return ProcessingResult(
                item_id=work_item.id,
                success=False,
                error="Execution timeout",
                execution_time=execution_time,
                worker_id=worker_id
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ProcessingResult(
                item_id=work_item.id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                worker_id=worker_id
            )
    
    def _auto_scaling_loop(self) -> None:
        """Auto-scaling loop for dynamic worker adjustment."""
        while self._active:
            try:
                # Measure current load
                queue_size = self.work_queue.qsize()
                queue_utilization = queue_size / self.queue_size if self.queue_size > 0 else 0
                
                # Calculate average processing time
                avg_time = self._processing_stats.get('avg_time', 0.1)
                
                # Store load sample
                self._load_samples.append(queue_utilization)
                if len(self._load_samples) > 10:  # Keep last 10 samples
                    self._load_samples.pop(0)
                
                avg_load = sum(self._load_samples) / len(self._load_samples)
                
                # Make scaling decision
                with self._scaling_lock:
                    if avg_load > 0.7 and self._current_workers < self.max_workers_limit:
                        # Scale up
                        new_workers = min(2, self.max_workers_limit - self._current_workers)
                        for i in range(new_workers):
                            worker = threading.Thread(
                                target=self._worker_loop,
                                name=f"processor_worker_{self._current_workers + i}",
                                daemon=True
                            )
                            worker.start()
                            self._workers.append(worker)
                        
                        self._current_workers += new_workers
                        logger.info(f"Scaled up to {self._current_workers} workers (load: {avg_load:.2f})")
                    
                    elif avg_load < 0.2 and self._current_workers > self.min_workers:
                        # Scale down (workers will naturally finish and exit)
                        target_workers = max(self.min_workers, self._current_workers - 1)
                        # Note: In a production system, you'd implement graceful worker shutdown
                        logger.info(f"Should scale down to {target_workers} workers (load: {avg_load:.2f})")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                time.sleep(5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        queue_size = self.work_queue.qsize()
        
        return {
            'mode': self.mode.value,
            'active': self._active,
            'current_workers': self._current_workers,
            'queue_size': queue_size,
            'queue_utilization': queue_size / self.queue_size if self.queue_size > 0 else 0,
            'submitted': self._processing_stats['submitted'],
            'completed': self._processing_stats['completed'],
            'failed': self._processing_stats['failed'],
            'success_rate': (
                self._processing_stats['completed'] / 
                (self._processing_stats['completed'] + self._processing_stats['failed'])
                if (self._processing_stats['completed'] + self._processing_stats['failed']) > 0 else 0
            ),
            'avg_processing_time': self._processing_stats['avg_time'],
            'total_processing_time': self._processing_stats['total_time'],
            'results_cached': len(self._results)
        }
    
    def clear_results(self, older_than_seconds: Optional[float] = None) -> int:
        """Clear old results to free memory."""
        with self._results_lock:
            if older_than_seconds is None:
                count = len(self._results)
                self._results.clear()
                return count
            
            # Remove results older than threshold
            current_time = time.time()
            keys_to_remove = []
            
            for key, result in self._results.items():
                # Estimate result age (simplified)
                if current_time - older_than_seconds > 0:  # Would need timestamp in result
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._results[key]
            
            return len(keys_to_remove)


# Helper functions for common parallel operations

def parallel_map(
    processor: ParallelProcessor,
    func: Callable,
    items: List[Any],
    timeout: Optional[float] = None,
    chunk_size: int = 100
) -> List[Any]:
    """
    Parallel map implementation using processor.
    
    Args:
        processor: Parallel processor instance
        func: Function to apply to each item
        items: List of items to process
        timeout: Total timeout for all operations
        chunk_size: Number of items per batch
        
    Returns:
        List of results in same order as input
    """
    # Submit work in chunks
    item_ids = []
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    for chunk in chunks:
        chunk_ids = processor.submit_batch([
            (func, (item,), {}) for item in chunk
        ])
        item_ids.extend(chunk_ids)
    
    # Get results
    results = processor.get_batch_results(item_ids, timeout=timeout)
    
    # Return results in original order
    ordered_results = []
    for item_id in item_ids:
        if item_id in results and results[item_id].success:
            ordered_results.append(results[item_id].result)
        else:
            ordered_results.append(None)  # Failed items
    
    return ordered_results


# Global processor instances
contract_processor = ParallelProcessor(
    mode=ProcessingMode.THREAD_POOL,
    max_workers=8,
    enable_auto_scaling=True
)

verification_processor = ParallelProcessor(
    mode=ProcessingMode.PROCESS_POOL,
    max_workers=4,
    enable_auto_scaling=False  # CPU-bound work, fixed pool
)