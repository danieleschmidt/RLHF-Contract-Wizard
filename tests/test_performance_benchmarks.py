"""
Performance benchmarks and stress tests for RLHF-Contract-Wizard.

Provides comprehensive performance testing including:
- Throughput and latency benchmarks
- Memory usage and resource optimization
- Scalability testing under load
- Distributed execution performance
- Cache performance optimization
"""

import pytest
import time
import asyncio
import statistics
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil
import gc

import jax
import jax.numpy as jnp
import numpy as np

from src.models.reward_contract import RewardContract, AggregationStrategy
from src.models.legal_blocks import LegalBlocks, RLHFConstraints
from src.optimization.distributed_computing import create_distributed_executor
from src.optimization.intelligent_caching import get_global_cache, configure_global_cache
from src.optimization.contract_cache import reward_cache
from src.training.rlhf_trainer import ContractualPPO, TrainingConfig
from src.monitoring.real_time_monitoring import create_contract_monitor


class PerformanceTestContract:
    """Optimized test contract for performance benchmarks."""
    
    def __init__(self):
        self.contract = RewardContract(
            name="PerfTestContract",
            version="1.0.0",
            stakeholders={"user": 0.5, "safety": 0.3, "operator": 0.2}
        )
        
        # Add lightweight constraints for performance testing
        @self.contract.reward_function("user")
        def user_reward(state, action):
            return jnp.tanh(jnp.sum(state * action[:len(state)]))
        
        @self.contract.reward_function("safety")
        def safety_reward(state, action):
            return jnp.exp(-jnp.sum(jnp.square(action)) * 0.1)
        
        @self.contract.reward_function("operator")
        def operator_reward(state, action):
            return 1.0 / (1.0 + jnp.sum(jnp.abs(action)))


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        self.start_time = None
        self.end_time = None
        self.memory_usage = {}
        self.resource_usage = {}
    
    def start(self):
        """Start timing and resource monitoring."""
        gc.collect()  # Clean up before measurement
        self.start_time = time.time()
        
        if psutil:
            process = psutil.Process()
            self.memory_usage['start'] = process.memory_info().rss / 1024 / 1024  # MB
            self.resource_usage['cpu_start'] = psutil.cpu_percent()
    
    def end(self):
        """End timing and resource monitoring."""
        self.end_time = time.time()
        
        if psutil:
            process = psutil.Process()
            self.memory_usage['end'] = process.memory_info().rss / 1024 / 1024  # MB
            self.resource_usage['cpu_end'] = psutil.cpu_percent()
        
        # Calculate derived metrics
        self.metrics['total_time'] = self.end_time - self.start_time
        
        if psutil:
            self.metrics['memory_delta'] = (
                self.memory_usage['end'] - self.memory_usage['start']
            )
            self.metrics['peak_memory'] = self.memory_usage['end']
    
    def add_metric(self, name: str, value: float):
        """Add a custom metric."""
        self.metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        return {
            'name': self.name,
            'total_time': self.metrics.get('total_time', 0),
            'throughput': self.metrics.get('throughput', 0),
            'latency_mean': self.metrics.get('latency_mean', 0),
            'latency_p95': self.metrics.get('latency_p95', 0),
            'latency_p99': self.metrics.get('latency_p99', 0),
            'memory_delta': self.metrics.get('memory_delta', 0),
            'peak_memory': self.metrics.get('peak_memory', 0),
            'error_rate': self.metrics.get('error_rate', 0),
            'all_metrics': self.metrics
        }


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    @pytest.fixture
    def perf_contract(self):
        """Create performance test contract."""
        return PerformanceTestContract().contract
    
    @pytest.fixture(scope="module")
    def cache_config(self):
        """Configure cache for performance tests."""
        configure_global_cache(
            l1_config={'max_size': 10000, 'max_memory_mb': 500},
            enable_ml=False  # Disable ML for consistent benchmarks
        )
        yield
        # Cleanup
        get_global_cache().clear()
    
    def test_single_computation_latency(self, perf_contract):
        """Benchmark single reward computation latency."""
        benchmark = BenchmarkResult("single_computation_latency")
        
        # Warm up JIT compilation
        state = jnp.ones(10)
        action = jnp.ones(5)
        for _ in range(10):
            _ = perf_contract.compute_reward(state, action)
        
        # Measure latency distribution
        latencies = []
        num_samples = 1000
        
        benchmark.start()
        
        for i in range(num_samples):
            # Use different inputs to avoid caching effects
            state = jnp.array([float(i % 100)] * 10) / 100.0
            action = jnp.array([float((i * 7) % 100)] * 5) / 100.0
            
            start = time.time()
            reward = perf_contract.compute_reward(state, action, use_cache=False)
            end = time.time()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            assert jnp.isfinite(reward)
        
        benchmark.end()
        
        # Calculate statistics
        benchmark.add_metric('latency_mean', statistics.mean(latencies))
        benchmark.add_metric('latency_median', statistics.median(latencies))
        benchmark.add_metric('latency_p95', np.percentile(latencies, 95))
        benchmark.add_metric('latency_p99', np.percentile(latencies, 99))
        benchmark.add_metric('latency_min', min(latencies))
        benchmark.add_metric('latency_max', max(latencies))
        benchmark.add_metric('latency_std', statistics.stdev(latencies))
        benchmark.add_metric('throughput', num_samples / benchmark.metrics['total_time'])
        
        # Performance assertions
        assert benchmark.metrics['latency_mean'] < 10.0  # < 10ms average
        assert benchmark.metrics['latency_p95'] < 20.0   # < 20ms p95
        assert benchmark.metrics['latency_p99'] < 50.0   # < 50ms p99
        assert benchmark.metrics['throughput'] > 100     # > 100 ops/sec
        
        print(f"\nSingle Computation Latency Benchmark:")
        print(f"  Mean latency: {benchmark.metrics['latency_mean']:.2f}ms")
        print(f"  P95 latency: {benchmark.metrics['latency_p95']:.2f}ms")
        print(f"  P99 latency: {benchmark.metrics['latency_p99']:.2f}ms")
        print(f"  Throughput: {benchmark.metrics['throughput']:.1f} ops/sec")
    
    def test_batch_processing_throughput(self, perf_contract):
        """Benchmark batch processing throughput."""
        benchmark = BenchmarkResult("batch_processing_throughput")
        
        batch_sizes = [100, 500, 1000, 2000, 5000]
        results = {}
        
        for batch_size in batch_sizes:
            # Generate batch data
            key = jax.random.PRNGKey(42)
            states = jax.random.normal(key, (batch_size, 10))
            actions = jax.random.normal(key, (batch_size, 5))
            
            # Warm up
            for _ in range(3):
                _ = perf_contract.compute_reward(states[0], actions[0])
            
            # Measure batch processing
            start_time = time.time()
            
            rewards = []
            for i in range(batch_size):
                reward = perf_contract.compute_reward(states[i], actions[i], use_cache=False)
                rewards.append(reward)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            throughput = batch_size / total_time
            avg_latency = total_time / batch_size * 1000  # ms
            
            results[batch_size] = {
                'throughput': throughput,
                'avg_latency': avg_latency,
                'total_time': total_time
            }
            
            # Verify results
            assert len(rewards) == batch_size
            assert all(jnp.isfinite(r) for r in rewards)
            assert throughput > 50  # Minimum throughput requirement
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Throughput: {throughput:.1f} ops/sec")
            print(f"  Avg latency: {avg_latency:.2f}ms")
        
        # Test throughput scaling
        throughputs = [results[bs]['throughput'] for bs in batch_sizes]
        
        # Throughput should generally increase with batch size (up to a point)
        assert throughputs[-1] > throughputs[0] * 0.8  # Allow some variance
        
        benchmark.add_metric('max_throughput', max(throughputs))
        benchmark.add_metric('scaling_efficiency', throughputs[-1] / throughputs[0])
    
    def test_concurrent_execution_performance(self, perf_contract):
        """Benchmark concurrent execution performance."""
        benchmark = BenchmarkResult("concurrent_execution")
        
        num_threads = [1, 2, 4, 8]
        requests_per_thread = 100
        results = {}
        
        for thread_count in num_threads:
            print(f"\nTesting with {thread_count} threads...")
            
            def worker_function(worker_id: int) -> List[float]:
                """Worker function for concurrent execution."""
                latencies = []
                
                for i in range(requests_per_thread):
                    # Generate unique inputs per worker and iteration
                    state = jnp.array([float((worker_id * 1000 + i) % 100)] * 10) / 100.0
                    action = jnp.array([float((worker_id * 1000 + i * 7) % 100)] * 5) / 100.0
                    
                    start = time.time()
                    reward = perf_contract.compute_reward(state, action, use_cache=False)
                    end = time.time()
                    
                    latencies.append((end - start) * 1000)  # ms
                    
                    assert jnp.isfinite(reward)
                
                return latencies
            
            # Execute concurrent workload
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [
                    executor.submit(worker_function, worker_id)
                    for worker_id in range(thread_count)
                ]
                
                all_latencies = []
                for future in as_completed(futures):
                    worker_latencies = future.result()
                    all_latencies.extend(worker_latencies)
            
            end_time = time.time()
            
            # Calculate metrics
            total_requests = thread_count * requests_per_thread
            total_time = end_time - start_time
            throughput = total_requests / total_time
            mean_latency = statistics.mean(all_latencies)
            p95_latency = np.percentile(all_latencies, 95)
            
            results[thread_count] = {
                'throughput': throughput,
                'mean_latency': mean_latency,
                'p95_latency': p95_latency,
                'total_time': total_time
            }
            
            print(f"  Throughput: {throughput:.1f} ops/sec")
            print(f"  Mean latency: {mean_latency:.2f}ms")
            print(f"  P95 latency: {p95_latency:.2f}ms")
            
            # Verify performance
            assert len(all_latencies) == total_requests
            assert throughput > 20 * thread_count  # Scaling requirement
        
        # Analyze scaling efficiency
        single_thread_throughput = results[1]['throughput']
        max_thread_throughput = results[max(num_threads)]['throughput']
        scaling_efficiency = max_thread_throughput / (single_thread_throughput * max(num_threads))
        
        benchmark.add_metric('scaling_efficiency', scaling_efficiency)
        benchmark.add_metric('max_concurrent_throughput', max_thread_throughput)
        
        # Scaling should be reasonably efficient
        assert scaling_efficiency > 0.3  # At least 30% efficiency
    
    def test_cache_performance_impact(self, perf_contract, cache_config):
        """Benchmark cache performance impact."""
        benchmark = BenchmarkResult("cache_performance")
        
        cache = get_global_cache()
        cache.clear()  # Start with empty cache
        
        # Test data
        num_unique_inputs = 1000
        num_repeated_accesses = 5
        
        # Generate test inputs
        key = jax.random.PRNGKey(42)
        states = jax.random.normal(key, (num_unique_inputs, 10))
        actions = jax.random.normal(key, (num_unique_inputs, 5))
        
        # Measure cache miss performance (first access)
        print("\nMeasuring cache miss performance...")
        cache_miss_latencies = []
        
        start_time = time.time()
        for i in range(num_unique_inputs):
            start = time.time()
            reward = perf_contract.compute_reward(states[i], actions[i], use_cache=True)
            end = time.time()
            
            cache_miss_latencies.append((end - start) * 1000)
            assert jnp.isfinite(reward)
        
        cache_miss_time = time.time() - start_time
        cache_miss_throughput = num_unique_inputs / cache_miss_time
        
        # Measure cache hit performance (repeated access)
        print("Measuring cache hit performance...")
        cache_hit_latencies = []
        
        start_time = time.time()
        for _ in range(num_repeated_accesses):
            for i in range(num_unique_inputs):
                start = time.time()
                reward = perf_contract.compute_reward(states[i], actions[i], use_cache=True)
                end = time.time()
                
                cache_hit_latencies.append((end - start) * 1000)
                assert jnp.isfinite(reward)
        
        cache_hit_time = time.time() - start_time
        cache_hit_throughput = (num_unique_inputs * num_repeated_accesses) / cache_hit_time
        
        # Calculate metrics
        miss_mean_latency = statistics.mean(cache_miss_latencies)
        hit_mean_latency = statistics.mean(cache_hit_latencies)
        speedup_ratio = miss_mean_latency / hit_mean_latency
        throughput_improvement = cache_hit_throughput / cache_miss_throughput
        
        # Get cache statistics
        cache_stats = cache.get_cache_stats()
        hit_rate = cache_stats['global']['hit_rate']
        
        print(f"\nCache Performance Results:")
        print(f"  Cache miss latency: {miss_mean_latency:.2f}ms")
        print(f"  Cache hit latency: {hit_mean_latency:.2f}ms")
        print(f"  Speedup ratio: {speedup_ratio:.1f}x")
        print(f"  Throughput improvement: {throughput_improvement:.1f}x")
        print(f"  Cache hit rate: {hit_rate:.1%}")
        
        # Performance assertions
        assert speedup_ratio > 2.0  # Cache should provide significant speedup
        assert hit_rate > 0.8       # High hit rate for repeated accesses
        assert hit_mean_latency < miss_mean_latency * 0.5  # Substantial improvement
        
        benchmark.add_metric('cache_speedup', speedup_ratio)
        benchmark.add_metric('cache_hit_rate', hit_rate)
        benchmark.add_metric('throughput_improvement', throughput_improvement)
    
    @pytest.mark.asyncio
    async def test_distributed_execution_performance(self, perf_contract):
        """Benchmark distributed execution performance."""
        benchmark = BenchmarkResult("distributed_execution")
        
        executor = create_distributed_executor()
        
        batch_sizes = [1000, 5000, 10000]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting distributed execution with batch size {batch_size}...")
            
            # Generate test data
            key = jax.random.PRNGKey(42)
            states = jax.random.normal(key, (batch_size, 10))
            actions = jax.random.normal(key, (batch_size, 5))
            
            # Warm up
            small_batch = min(100, batch_size)
            await executor.execute_contract_batch(
                perf_contract,
                states[:small_batch],
                actions[:small_batch]
            )
            
            # Measure distributed execution
            start_time = time.time()
            
            rewards = await executor.execute_contract_batch(
                perf_contract,
                states,
                actions,
                batch_size=batch_size // 4  # Force distribution
            )
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            throughput = batch_size / total_time
            
            results[batch_size] = {
                'throughput': throughput,
                'total_time': total_time,
                'latency_per_item': total_time / batch_size * 1000  # ms
            }
            
            # Verify results
            assert rewards.shape == (batch_size,)
            assert jnp.all(jnp.isfinite(rewards))
            
            print(f"  Throughput: {throughput:.1f} ops/sec")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Latency per item: {results[batch_size]['latency_per_item']:.2f}ms")
        
        # Test scaling with batch size
        throughputs = [results[bs]['throughput'] for bs in batch_sizes]
        scaling_factor = throughputs[-1] / throughputs[0]
        
        benchmark.add_metric('max_distributed_throughput', max(throughputs))
        benchmark.add_metric('batch_scaling_factor', scaling_factor)
        
        # Distributed execution should scale with batch size
        assert scaling_factor > 1.5  # Should improve with larger batches
        assert max(throughputs) > 500  # Minimum distributed throughput
    
    def test_memory_efficiency_and_gc_behavior(self, perf_contract):
        """Test memory efficiency and garbage collection behavior."""
        benchmark = BenchmarkResult("memory_efficiency")
        
        if not psutil:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process()
        
        # Baseline memory measurement
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\nBaseline memory usage: {baseline_memory:.1f} MB")
        
        # Test memory usage during computation
        memory_samples = []
        num_computations = 10000
        
        benchmark.start()
        
        for i in range(num_computations):
            # Generate inputs
            state = jnp.array([float(i % 1000)] * 10) / 1000.0
            action = jnp.array([float((i * 7) % 1000)] * 5) / 1000.0
            
            # Compute reward
            reward = perf_contract.compute_reward(state, action, use_cache=False)
            
            # Sample memory usage periodically
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Force garbage collection to test cleanup
                if i % 5000 == 0:
                    gc.collect()
            
            assert jnp.isfinite(reward)
        
        benchmark.end()
        
        # Final memory measurement
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Calculate memory metrics
        max_memory = max(memory_samples) if memory_samples else final_memory
        memory_growth = final_memory - baseline_memory
        peak_memory_overhead = max_memory - baseline_memory
        
        print(f"  Peak memory usage: {max_memory:.1f} MB")
        print(f"  Final memory usage: {final_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(f"  Peak overhead: {peak_memory_overhead:.1f} MB")
        
        benchmark.add_metric('baseline_memory', baseline_memory)
        benchmark.add_metric('peak_memory', max_memory)
        benchmark.add_metric('final_memory', final_memory)
        benchmark.add_metric('memory_growth', memory_growth)
        benchmark.add_metric('peak_overhead', peak_memory_overhead)
        benchmark.add_metric('throughput', num_computations / benchmark.metrics['total_time'])
        
        # Memory efficiency assertions
        assert memory_growth < 100  # Should not grow by more than 100MB
        assert peak_memory_overhead < 200  # Peak overhead should be reasonable
        
        # Check for memory leaks (growth should stabilize)
        if len(memory_samples) > 5:
            recent_growth = memory_samples[-1] - memory_samples[-3]
            assert recent_growth < 50  # Recent growth should be limited
    
    def test_stress_test_high_load(self, perf_contract):
        """Stress test under high computational load."""
        benchmark = BenchmarkResult("stress_test")
        
        # High-intensity stress test parameters
        duration_seconds = 10  # 10-second stress test
        target_rps = 1000      # Target requests per second
        
        print(f"\nRunning stress test for {duration_seconds}s at {target_rps} RPS...")
        
        # Results tracking
        completed_requests = 0
        failed_requests = 0
        latencies = []
        errors = []
        
        benchmark.start()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        request_interval = 1.0 / target_rps
        next_request_time = start_time
        
        while time.time() < end_time:
            current_time = time.time()
            
            # Rate limiting
            if current_time < next_request_time:
                time.sleep(next_request_time - current_time)
            
            try:
                # Generate unique inputs
                seed = int((time.time() * 1000000) % 1000000)
                state = jnp.array([float((seed + i) % 1000) for i in range(10)]) / 1000.0
                action = jnp.array([float((seed * 7 + i) % 1000) for i in range(5)]) / 1000.0
                
                # Time the computation
                req_start = time.time()
                reward = perf_contract.compute_reward(state, action, use_cache=False)
                req_end = time.time()
                
                # Verify result
                if jnp.isfinite(reward):
                    completed_requests += 1
                    latencies.append((req_end - req_start) * 1000)  # ms
                else:
                    failed_requests += 1
                    errors.append("Non-finite reward")
                
            except Exception as e:
                failed_requests += 1
                errors.append(str(e))
            
            next_request_time += request_interval
        
        benchmark.end()
        
        # Calculate stress test metrics
        total_requests = completed_requests + failed_requests
        success_rate = completed_requests / total_requests if total_requests > 0 else 0
        actual_rps = completed_requests / benchmark.metrics['total_time']
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        if latencies:
            mean_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            mean_latency = p95_latency = p99_latency = 0
        
        print(f"\nStress Test Results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Completed requests: {completed_requests}")
        print(f"  Failed requests: {failed_requests}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Actual RPS: {actual_rps:.1f}")
        print(f"  Error rate: {error_rate:.1%}")
        print(f"  Mean latency: {mean_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  P99 latency: {p99_latency:.2f}ms")
        
        # Stress test assertions
        assert success_rate > 0.95      # 95% success rate under stress
        assert actual_rps > target_rps * 0.8  # Achieve 80% of target RPS
        assert error_rate < 0.05        # Less than 5% error rate
        assert mean_latency < 50        # Mean latency under 50ms
        assert p99_latency < 200        # P99 latency under 200ms
        
        benchmark.add_metric('success_rate', success_rate)
        benchmark.add_metric('actual_rps', actual_rps)
        benchmark.add_metric('error_rate', error_rate)
        benchmark.add_metric('mean_latency', mean_latency)
        benchmark.add_metric('p95_latency', p95_latency)
        benchmark.add_metric('p99_latency', p99_latency)
        
        # Log any errors for debugging
        if errors:
            unique_errors = list(set(errors))
            print(f"  Unique errors: {unique_errors[:5]}")  # Show first 5 unique errors
    
    def test_resource_utilization_efficiency(self, perf_contract):
        """Test resource utilization efficiency."""
        benchmark = BenchmarkResult("resource_utilization")
        
        if not psutil:
            pytest.skip("psutil not available for resource testing")
        
        # Monitor resource usage during computation
        cpu_samples = []
        memory_samples = []
        
        # Background monitoring
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def resource_monitor():
            while monitoring_active.is_set():
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                cpu_samples.append(cpu_percent)
                memory_samples.append(memory_info.percent)
                
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
        monitor_thread.start()
        
        # Perform computations while monitoring
        num_computations = 5000
        computation_start = time.time()
        
        for i in range(num_computations):
            state = jnp.array([float(i % 100)] * 10) / 100.0
            action = jnp.array([float((i * 3) % 100)] * 5) / 100.0
            
            reward = perf_contract.compute_reward(state, action, use_cache=False)
            assert jnp.isfinite(reward)
        
        computation_time = time.time() - computation_start
        
        # Stop monitoring
        monitoring_active.clear()
        monitor_thread.join(timeout=1.0)
        
        # Analyze resource utilization
        if cpu_samples and memory_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            avg_memory = statistics.mean(memory_samples)
            max_memory = max(memory_samples)
            
            # Calculate efficiency metrics
            throughput = num_computations / computation_time
            cpu_efficiency = throughput / max(avg_cpu, 1.0)  # ops per CPU%
            
            print(f"\nResource Utilization Results:")
            print(f"  Average CPU usage: {avg_cpu:.1f}%")
            print(f"  Peak CPU usage: {max_cpu:.1f}%")
            print(f"  Average memory usage: {avg_memory:.1f}%")
            print(f"  Peak memory usage: {max_memory:.1f}%")
            print(f"  Throughput: {throughput:.1f} ops/sec")
            print(f"  CPU efficiency: {cpu_efficiency:.1f} ops/sec per CPU%")
            
            benchmark.add_metric('avg_cpu_usage', avg_cpu)
            benchmark.add_metric('max_cpu_usage', max_cpu)
            benchmark.add_metric('avg_memory_usage', avg_memory)
            benchmark.add_metric('max_memory_usage', max_memory)
            benchmark.add_metric('throughput', throughput)
            benchmark.add_metric('cpu_efficiency', cpu_efficiency)
            
            # Resource efficiency assertions
            assert avg_cpu < 90        # Should not peg CPU constantly
            assert avg_memory < 80     # Should not use excessive memory
            assert cpu_efficiency > 1  # Should be reasonably efficient


if __name__ == "__main__":
    # Run performance benchmarks if executed directly
    pytest.main([__file__, "-v", "-s"])  # -s to show print output
