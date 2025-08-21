"""
Focused performance tests for RLHF-Contract-Wizard.

Tests key performance metrics without complex dependencies.
"""

import time
import statistics
import jax
import jax.numpy as jnp
from typing import List

from src.models.reward_contract import RewardContract


class PerformanceBenchmarks:
    """Focused performance benchmarks."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_single_computation_latency(self) -> dict:
        """Benchmark single reward computation latency."""
        print("Running single computation latency benchmark...")
        
        contract = RewardContract(
            name="LatencyTestContract",
            version="1.0.0",
            stakeholders={"user": 0.6, "safety": 0.4}
        )
        
        @contract.reward_function("user")
        def user_reward(state, action):
            return jnp.tanh(jnp.sum(state) + jnp.sum(action))
        
        @contract.reward_function("safety")
        def safety_reward(state, action):
            return jnp.exp(-jnp.sum(jnp.square(action)) * 0.1)
        
        # Warm up JIT compilation
        state = jnp.ones(10)
        action = jnp.ones(10)
        for _ in range(10):
            _ = contract.compute_reward(state, action)
        
        # Measure latency distribution
        latencies = []
        num_samples = 1000
        
        for i in range(num_samples):
            # Use different inputs to avoid caching effects
            state = jnp.array([float(i % 100)] * 10) / 100.0
            action = jnp.array([float((i * 7) % 100)] * 10) / 100.0
            
            start = time.time()
            reward = contract.compute_reward(state, action, use_cache=False)
            end = time.time()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            assert jnp.isfinite(reward)
        
        # Calculate statistics
        results = {
            'mean_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'p95_latency_ms': jnp.percentile(jnp.array(latencies), 95),
            'p99_latency_ms': jnp.percentile(jnp.array(latencies), 99),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'throughput_ops_per_sec': num_samples / (sum(latencies) / 1000)
        }
        
        return results
    
    def benchmark_batch_processing(self) -> dict:
        """Benchmark batch processing performance."""
        print("Running batch processing benchmark...")
        
        contract = RewardContract(
            name="BatchTestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def batch_reward(state, action):
            return jnp.sum(state * action[:len(state)])
        
        batch_sizes = [100, 500, 1000, 2000]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Generate batch data
            key = jax.random.PRNGKey(42)
            states = jax.random.normal(key, (batch_size, 10))
            actions = jax.random.normal(key, (batch_size, 10))
            
            # Warm up
            for _ in range(3):
                _ = contract.compute_reward(states[0], actions[0])
            
            # Measure batch processing
            start_time = time.time()
            
            rewards = []
            for i in range(batch_size):
                reward = contract.compute_reward(states[i], actions[i], use_cache=False)
                rewards.append(reward)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            throughput = batch_size / total_time
            avg_latency = total_time / batch_size * 1000  # ms
            
            results[batch_size] = {
                'throughput_ops_per_sec': throughput,
                'avg_latency_ms': avg_latency,
                'total_time_sec': total_time
            }
            
            # Verify results
            assert len(rewards) == batch_size
            assert all(jnp.isfinite(r) for r in rewards)
        
        return results
    
    def benchmark_memory_efficiency(self) -> dict:
        """Benchmark memory efficiency."""
        print("Running memory efficiency benchmark...")
        
        import psutil
        process = psutil.Process()
        
        contract = RewardContract(
            name="MemoryTestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def memory_reward(state, action):
            return jnp.sum(state) / len(state)
        
        # Baseline memory
        import gc
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory test
        memory_samples = []
        num_computations = 5000
        
        for i in range(num_computations):
            state = jnp.array([float(i % 1000)] * 10) / 1000.0
            action = jnp.array([float((i * 7) % 1000)] * 10) / 1000.0
            
            reward = contract.compute_reward(state, action, use_cache=False)
            
            # Sample memory usage periodically
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Force garbage collection
                if i % 2000 == 0:
                    gc.collect()
            
            assert jnp.isfinite(reward)
        
        # Final memory measurement
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        results = {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': max(memory_samples) if memory_samples else final_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': final_memory - baseline_memory,
            'num_computations': num_computations
        }
        
        return results
    
    def benchmark_cache_performance(self) -> dict:
        """Benchmark cache performance impact."""
        print("Running cache performance benchmark...")
        
        contract = RewardContract(
            name="CacheTestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def cached_reward(state, action):
            # Expensive computation simulation
            return jnp.sum(jnp.sin(state) * jnp.cos(action[:len(state)]))
        
        # Test data
        num_unique_inputs = 100
        num_repeated_accesses = 5
        
        # Generate test inputs
        key = jax.random.PRNGKey(42)
        states = jax.random.normal(key, (num_unique_inputs, 10))
        actions = jax.random.normal(key, (num_unique_inputs, 10))
        
        # Measure cache miss performance (first access)
        cache_miss_times = []
        for i in range(num_unique_inputs):
            start = time.time()
            reward = contract.compute_reward(states[i], actions[i], use_cache=True)
            end = time.time()
            cache_miss_times.append((end - start) * 1000)
            assert jnp.isfinite(reward)
        
        # Measure cache hit performance (repeated access)
        cache_hit_times = []
        for _ in range(num_repeated_accesses):
            for i in range(num_unique_inputs):
                start = time.time()
                reward = contract.compute_reward(states[i], actions[i], use_cache=True)
                end = time.time()
                cache_hit_times.append((end - start) * 1000)
                assert jnp.isfinite(reward)
        
        results = {
            'cache_miss_avg_ms': statistics.mean(cache_miss_times),
            'cache_hit_avg_ms': statistics.mean(cache_hit_times),
            'speedup_ratio': statistics.mean(cache_miss_times) / statistics.mean(cache_hit_times),
            'cache_miss_throughput': 1000 / statistics.mean(cache_miss_times),
            'cache_hit_throughput': 1000 / statistics.mean(cache_hit_times)
        }
        
        return results
    
    def run_all_benchmarks(self) -> dict:
        """Run all performance benchmarks."""
        print("=" * 60)
        print("RLHF-Contract-Wizard Performance Benchmarks")
        print("=" * 60)
        
        all_results = {}
        
        # Run benchmarks
        all_results['latency'] = self.benchmark_single_computation_latency()
        all_results['batch'] = self.benchmark_batch_processing()
        all_results['memory'] = self.benchmark_memory_efficiency()
        all_results['cache'] = self.benchmark_cache_performance()
        
        return all_results
    
    def print_results(self, results: dict):
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        # Latency results
        latency = results['latency']
        print(f"\n📊 SINGLE COMPUTATION LATENCY:")
        print(f"  Mean latency:     {latency['mean_latency_ms']:.2f}ms")
        print(f"  P95 latency:      {latency['p95_latency_ms']:.2f}ms")
        print(f"  P99 latency:      {latency['p99_latency_ms']:.2f}ms")
        print(f"  Throughput:       {latency['throughput_ops_per_sec']:.1f} ops/sec")
        
        # Batch results
        print(f"\n🚀 BATCH PROCESSING:")
        for batch_size, metrics in results['batch'].items():
            print(f"  Batch {batch_size:4d}:     {metrics['throughput_ops_per_sec']:.1f} ops/sec, "
                  f"{metrics['avg_latency_ms']:.2f}ms avg")
        
        # Memory results
        memory = results['memory']
        print(f"\n💾 MEMORY EFFICIENCY:")
        print(f"  Baseline memory:  {memory['baseline_memory_mb']:.1f} MB")
        print(f"  Peak memory:      {memory['peak_memory_mb']:.1f} MB")
        print(f"  Memory growth:    {memory['memory_growth_mb']:.1f} MB")
        print(f"  Computations:     {memory['num_computations']:,}")
        
        # Cache results
        cache = results['cache']
        print(f"\n⚡ CACHE PERFORMANCE:")
        print(f"  Cache miss:       {cache['cache_miss_avg_ms']:.2f}ms")
        print(f"  Cache hit:        {cache['cache_hit_avg_ms']:.2f}ms")
        print(f"  Speedup ratio:    {cache['speedup_ratio']:.1f}x")
        print(f"  Hit throughput:   {cache['cache_hit_throughput']:.1f} ops/sec")
        
        # Performance assertions
        print(f"\n✅ PERFORMANCE VALIDATION:")
        
        assertions = []
        assertions.append(("Latency < 10ms", latency['mean_latency_ms'] < 10.0))
        assertions.append(("P95 < 20ms", latency['p95_latency_ms'] < 20.0))
        assertions.append(("Throughput > 100 ops/sec", latency['throughput_ops_per_sec'] > 100))
        assertions.append(("Memory growth < 100MB", memory['memory_growth_mb'] < 100))
        assertions.append(("Cache speedup > 2x", cache['speedup_ratio'] > 2.0))
        
        for assertion, passed in assertions:
            status = "✓" if passed else "✗"
            print(f"  {status} {assertion}")
        
        passed_count = sum(1 for _, passed in assertions if passed)
        print(f"\n🎯 Performance Score: {passed_count}/{len(assertions)} assertions passed")
        
        return passed_count == len(assertions)


if __name__ == "__main__":
    benchmarks = PerformanceBenchmarks()
    results = benchmarks.run_all_benchmarks()
    success = benchmarks.print_results(results)
    
    if success:
        print(f"\n🎉 All performance benchmarks PASSED!")
        exit(0)
    else:
        print(f"\n⚠️  Some performance benchmarks FAILED!")
        exit(1)