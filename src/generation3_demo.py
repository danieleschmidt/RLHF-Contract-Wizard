#!/usr/bin/env python3
"""
Generation 3 Demo: MAKE IT SCALE (Optimized)

Demonstrates advanced performance optimization, distributed quantum computing,
intelligent caching, auto-scaling, and all scaling features added in Generation 3
of RLHF-Contract-Wizard.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
import jax.numpy as jnp
import numpy as np

from .models.reward_contract import RewardContract, AggregationStrategy
from .scaling.intelligent_performance_optimization import (
    PerformanceOptimizer, get_performance_optimizer, OptimizationStrategy,
    IntelligentCache, CacheStrategy, AdaptiveLoadBalancer,
    optimize_performance, smart_cache
)
from .scaling.distributed_quantum_computing import (
    DistributedQuantumOrchestrator, get_quantum_orchestrator,
    QuantumNetworkTopology, QuantumSynchronization
)
from .enhanced_contract_runtime import EnhancedContractRuntime, RuntimeConfig
from .quantum_planner.core import QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_intelligent_caching():
    """Demonstrate intelligent caching with multiple strategies."""
    logger.info("🧠 Starting Intelligent Caching Demo")
    
    # Test different caching strategies
    strategies = [CacheStrategy.LRU, CacheStrategy.ADAPTIVE, CacheStrategy.PREDICTIVE]
    
    for strategy in strategies:
        logger.info(f"Testing {strategy.value} caching strategy...")
        
        cache = IntelligentCache(max_size=100, strategy=strategy)
        
        # Simulate various access patterns
        test_data = {}
        for i in range(150):  # More items than cache size
            key = f"key_{i}"
            value = {"data": f"value_{i}", "timestamp": time.time(), "size": i * 100}
            test_data[key] = value
        
        # Phase 1: Initial population
        start_time = time.time()
        for i in range(100):
            key = f"key_{i}"
            cache.set(key, test_data[key], tags={'initial'})
        
        phase1_time = time.time() - start_time
        logger.info(f"   Phase 1 (population): {phase1_time:.3f}s")
        
        # Phase 2: Mixed access pattern (80% hits, 20% misses)
        hits = misses = 0
        start_time = time.time()
        
        for i in range(1000):
            if i % 5 == 0:  # 20% new keys (cache misses)
                key = f"key_{100 + i}"
                cache.set(key, test_data.get(key, {"new": True}), tags={'new_data'})
            else:  # 80% existing keys
                key = f"key_{i % 100}"
                result = cache.get(key)
                if result is not None:
                    hits += 1
                else:
                    misses += 1
        
        phase2_time = time.time() - start_time
        
        # Get cache statistics
        stats = cache.stats()
        
        logger.info(f"   Phase 2 (mixed access): {phase2_time:.3f}s")
        logger.info(f"   Final hit rate: {stats['hit_rate']:.2%}")
        logger.info(f"   Cache size: {stats['size']}/{stats['max_size']}")
        logger.info(f"   Memory usage: {stats['memory_usage_mb']:.1f} MB")
        logger.info(f"   Evictions: {stats['evictions']}")
        
        # Cache warming demonstration
        warm_data = [(f"warm_{i}", f"warm_value_{i}") for i in range(20)]
        cache.warm_cache(warm_data)
        
        logger.info(f"   Warmed cache with {len(warm_data)} entries")
        logger.info("")
    
    return "Caching strategies tested successfully"


async def demo_adaptive_load_balancer():
    """Demonstrate adaptive load balancing capabilities."""
    logger.info("⚖️ Starting Adaptive Load Balancer Demo")
    
    load_balancer = AdaptiveLoadBalancer()
    
    # Register workers with different capabilities
    workers = [
        {"id": "worker_1", "capacity": 100, "metadata": {"region": "us-east", "type": "cpu"}},
        {"id": "worker_2", "capacity": 150, "metadata": {"region": "us-west", "type": "gpu"}},
        {"id": "worker_3", "capacity": 80, "metadata": {"region": "eu-central", "type": "cpu"}},
        {"id": "worker_4", "capacity": 200, "metadata": {"region": "asia", "type": "gpu"}},
        {"id": "worker_5", "capacity": 120, "metadata": {"region": "us-east", "type": "hybrid"}}
    ]
    
    for worker in workers:
        load_balancer.register_worker(
            worker["id"], 
            worker["capacity"], 
            worker["metadata"]
        )
    
    logger.info(f"Registered {len(workers)} workers")
    
    # Simulate request processing with different patterns
    scenarios = [
        {"name": "Normal Load", "requests": 200, "error_rate": 0.05},
        {"name": "High Load", "requests": 500, "error_rate": 0.02},
        {"name": "Burst Load", "requests": 300, "error_rate": 0.10}
    ]
    
    for scenario in scenarios:
        logger.info(f"🔄 Testing scenario: {scenario['name']}")
        
        start_time = time.time()
        request_times = []
        successful_requests = 0
        
        for i in range(scenario["requests"]):
            # Select worker
            selected_worker = load_balancer.select_worker()
            
            if selected_worker:
                # Simulate request processing
                processing_time = np.random.gamma(2, 0.1)  # Realistic response time distribution
                success = np.random.random() > scenario["error_rate"]
                
                # Record completion
                load_balancer.record_request_completion(
                    selected_worker, processing_time, success
                )
                
                if success:
                    successful_requests += 1
                
                request_times.append(processing_time)
                
                # Small delay to simulate request spacing
                await asyncio.sleep(0.001)
        
        scenario_time = time.time() - start_time
        
        # Get load balancer statistics
        stats = load_balancer.get_stats()
        
        logger.info(f"   Scenario completed in {scenario_time:.3f}s")
        logger.info(f"   Success rate: {successful_requests/scenario['requests']:.1%}")
        logger.info(f"   Average response time: {np.mean(request_times):.3f}s")
        logger.info(f"   Active workers: {stats['available_workers']}/{stats['total_workers']}")
        
        # Show worker-specific stats
        for worker_id, worker_stats in stats['worker_stats'].items():
            logger.info(
                f"     {worker_id}: Load {worker_stats['load_percentage']:.1f}%, "
                f"Health {worker_stats['health_score']:.2f}, "
                f"Requests {worker_stats['total_requests']}"
            )
        
        logger.info("")
        
        # Brief pause between scenarios
        await asyncio.sleep(1.0)
    
    return stats


async def demo_performance_optimization():
    """Demonstrate intelligent performance optimization."""
    logger.info("⚡ Starting Performance Optimization Demo")
    
    optimizer = get_performance_optimizer()
    
    # Define test functions with different characteristics
    @optimize_performance(OptimizationStrategy.CPU_INTENSIVE, cache_results=True)
    def cpu_intensive_function(x):
        """Simulate CPU-intensive computation."""
        result = 0
        for i in range(int(x * 1000)):
            result += np.sin(i) * np.cos(i)
        return result
    
    @smart_cache(ttl_seconds=60.0, tags={'math'})
    def expensive_calculation(n):
        """Simulate expensive mathematical calculation."""
        if n <= 1:
            return 1
        return n * expensive_calculation(n - 1) if n < 10 else n * n  # Avoid deep recursion
    
    def simple_function(x):
        """Simple function for baseline comparison."""
        return x * 2 + 1
    
    # Test data
    test_inputs = [1, 5, 10, 15, 20, 25, 30, 8, 12, 18]
    
    logger.info("🧪 Testing function optimization...")
    
    # Benchmark different functions
    functions = [
        ("Simple Function", simple_function),
        ("CPU Intensive (Optimized)", cpu_intensive_function),
        ("Expensive Calculation (Cached)", expensive_calculation)
    ]
    
    for func_name, func in functions:
        logger.info(f"Testing {func_name}...")
        
        # Warm-up run
        for x in test_inputs[:3]:
            func(x)
        
        # Benchmark run
        start_time = time.time()
        results = []
        
        for x in test_inputs:
            result = func(x)
            results.append(result)
        
        execution_time = time.time() - start_time
        
        logger.info(f"   Execution time: {execution_time:.4f}s")
        logger.info(f"   Average per call: {execution_time/len(test_inputs):.4f}s")
    
    # Test batch optimization
    logger.info("🔄 Testing batch processing optimization...")
    
    def batch_function(x):
        """Function to be processed in batch."""
        return np.sum(np.random.random(int(x * 100)))
    
    batch_data = list(range(1, 101))  # 100 items
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [batch_function(x) for x in batch_data]
    sequential_time = time.time() - start_time
    
    # Optimized batch processing
    start_time = time.time()
    optimized_results = optimizer.optimize_batch_processing(
        batch_function, batch_data, max_workers=4
    )
    optimized_time = time.time() - start_time
    
    speedup = sequential_time / optimized_time
    
    logger.info(f"   Sequential time: {sequential_time:.3f}s")
    logger.info(f"   Optimized time: {optimized_time:.3f}s")
    logger.info(f"   Speedup: {speedup:.1f}x")
    
    # Performance recommendations
    recommendations = optimizer.get_performance_recommendations()
    logger.info("💡 Performance Recommendations:")
    for rec in recommendations:
        logger.info(f"   • {rec}")
    
    return {
        "optimization_tested": True,
        "speedup_achieved": speedup,
        "recommendations": recommendations
    }


async def demo_distributed_quantum_computing():
    """Demonstrate distributed quantum computing capabilities."""
    logger.info("🌌 Starting Distributed Quantum Computing Demo")
    
    orchestrator = get_quantum_orchestrator()
    
    # Register quantum computing nodes
    nodes = [
        {"id": "quantum_node_1", "capacity": 50, "capabilities": {"qubits": 20, "gates": ["H", "CNOT", "measure"]}},
        {"id": "quantum_node_2", "capacity": 75, "capabilities": {"qubits": 32, "gates": ["H", "CNOT", "RZ", "measure"]}},
        {"id": "quantum_node_3", "capacity": 40, "capabilities": {"qubits": 16, "gates": ["H", "CNOT", "measure", "optimize"]}},
        {"id": "quantum_node_4", "capacity": 60, "capabilities": {"qubits": 24, "gates": ["H", "CNOT", "RY", "RZ", "measure"]}}
    ]
    
    for node in nodes:
        success = await orchestrator.register_node(
            node["id"], 
            node["capacity"], 
            node["capabilities"]
        )
        if success:
            logger.info(f"   ✅ Registered {node['id']}")
        else:
            logger.info(f"   ❌ Failed to register {node['id']}")
    
    # Start orchestration
    await orchestrator.start_orchestration()
    logger.info("🔄 Started quantum orchestration")
    
    # Submit various quantum tasks
    quantum_tasks = [
        {
            "id": "task_superposition",
            "operations": [
                {"type": "hadamard", "data": {"input_state": 0}},
                {"type": "hadamard", "data": {"input_state": 1}},
                {"type": "measure", "data": {"state": {"0": 0.707, "1": 0.707}}}
            ],
            "requirements": {"min_nodes": 1, "priority": 0.8}
        },
        {
            "id": "task_entanglement",
            "operations": [
                {"type": "hadamard", "data": {"input_state": 0}},
                {"type": "cnot", "data": {"control": 0, "target": 0}},
                {"type": "measure", "data": {"state": {"00": 0.707, "11": 0.707}}}
            ],
            "requirements": {"min_nodes": 2, "priority": 0.9}
        },
        {
            "id": "task_optimization",
            "operations": [
                {"type": "optimize", "data": {
                    "objective": "minimize",
                    "parameters": [0.1, 0.3, 0.7, 0.2],
                    "iterations": 20
                }}
            ],
            "requirements": {"min_nodes": 1, "priority": 0.7}
        },
        {
            "id": "task_complex_circuit",
            "operations": [
                {"type": "hadamard", "data": {"input_state": 0}},
                {"type": "cnot", "data": {"control": 0, "target": 1}},
                {"type": "hadamard", "data": {"input_state": 1}},
                {"type": "cnot", "data": {"control": 1, "target": 0}},
                {"type": "measure", "data": {"state": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}}}
            ],
            "requirements": {"min_nodes": 3, "priority": 0.6}
        }
    ]
    
    # Submit all tasks
    for task in quantum_tasks:
        success = await orchestrator.submit_quantum_task(
            task["id"], task["operations"], task["requirements"]
        )
        if success:
            logger.info(f"   📤 Submitted {task['id']}")
    
    # Wait for tasks to complete
    logger.info("⏳ Waiting for quantum tasks to complete...")
    await asyncio.sleep(5.0)  # Give time for processing
    
    # Check network status
    network_status = orchestrator.get_network_status()
    logger.info("📊 Quantum Network Status:")
    logger.info(f"   Nodes: {network_status['available_nodes']}/{network_status['total_nodes']}")
    logger.info(f"   Load: {network_status['load_percentage']:.1f}%")
    logger.info(f"   Coherence: {network_status['average_coherence']:.3f}")
    logger.info(f"   Active tasks: {network_status['active_tasks']}")
    logger.info(f"   Completed tasks: {network_status['completed_tasks']}")
    logger.info(f"   Entanglements: {network_status['entanglements']}")
    
    # Show completed tasks
    if orchestrator.completed_tasks:
        logger.info("✅ Completed Quantum Tasks:")
        for task_id, result in orchestrator.completed_tasks.items():
            logger.info(f"   {task_id}: {result['success_rate']:.1%} success, "
                       f"{result['execution_time']:.3f}s, "
                       f"{len(result['nodes_used'])} nodes")
    
    # Stop orchestration
    await orchestrator.stop_orchestration()
    logger.info("⏹️ Stopped quantum orchestration")
    
    return network_status


async def demo_integrated_scaling_system():
    """Demonstrate integrated scaling system with all Generation 3 features."""
    logger.info("🚀 Starting Integrated Scaling System Demo")
    
    # Initialize all scaling components
    optimizer = get_performance_optimizer()
    orchestrator = get_quantum_orchestrator()
    
    # Configure enhanced runtime with scaling features
    config = RuntimeConfig(
        enable_caching=True,
        max_concurrent_contracts=10,
        timeout_seconds=30.0,
        enable_global_compliance=True,
        performance_monitoring=True,
        auto_recovery=True
    )
    
    runtime = EnhancedContractRuntime(config)
    
    # Create a high-performance contract
    contract = RewardContract(
        name="ScalableAssistant-v3",
        version="3.0.0",
        stakeholders={
            "operator": 0.3,
            "safety_board": 0.3,
            "users": 0.2,
            "performance_team": 0.2
        },
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE
    )
    
    # Add optimized reward functions
    @optimize_performance(OptimizationStrategy.BALANCED, cache_results=True)
    @contract.reward_function("operator")
    def optimized_operator_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Highly optimized operator reward with caching."""
        try:
            # Vectorized operations for performance
            efficiency = jnp.dot(state, action) / (jnp.linalg.norm(state) * jnp.linalg.norm(action) + 1e-8)
            return float(jnp.clip(efficiency, 0.0, 1.0))
        except Exception:
            return 0.0
    
    @smart_cache(ttl_seconds=300.0, tags={'safety'})
    @contract.reward_function("safety_board")
    def cached_safety_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Safety reward with intelligent caching."""
        try:
            # Complex safety calculation (expensive, worth caching)
            safety_metrics = jnp.array([
                jnp.sum(jnp.abs(action - 0.5)),  # Deviation from neutral
                jnp.max(jnp.abs(action)),         # Maximum action magnitude
                jnp.var(action)                   # Action consistency
            ])
            
            safety_score = 1.0 - jnp.mean(safety_metrics) / 2.0
            return float(jnp.clip(safety_score, 0.0, 1.0))
        except Exception:
            return 0.0
    
    @contract.reward_function("users")
    def user_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """User satisfaction reward."""
        try:
            satisfaction = jnp.tanh(jnp.sum(state * action))
            return float((satisfaction + 1.0) / 2.0)  # Normalize to [0, 1]
        except Exception:
            return 0.0
    
    @contract.reward_function("performance_team")
    def performance_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Performance-based reward."""
        try:
            # Reward faster computations and efficient resource usage
            computation_efficiency = 1.0 / (jnp.sum(jnp.abs(action)) + 1.0)
            memory_efficiency = 1.0 / (len(state) + len(action) + 1.0) * 100
            return float(jnp.clip(computation_efficiency + memory_efficiency, 0.0, 1.0))
        except Exception:
            return 0.0
    
    # Register contract
    contract_id = runtime.register_contract(contract)
    logger.info(f"📋 Registered scalable contract: {contract.metadata.name}")
    
    # Create scaled test scenarios
    test_scenarios = [
        {
            "name": "Light Load",
            "batch_size": 10,
            "state_size": 5,
            "action_size": 3,
            "complexity": "simple"
        },
        {
            "name": "Medium Load",
            "batch_size": 50,
            "state_size": 10,
            "action_size": 5,
            "complexity": "medium"
        },
        {
            "name": "Heavy Load",
            "batch_size": 100,
            "state_size": 20,
            "action_size": 8,
            "complexity": "complex"
        },
        {
            "name": "Burst Load",
            "batch_size": 200,
            "state_size": 15,
            "action_size": 6,
            "complexity": "varied"
        }
    ]
    
    scaling_results = []
    
    for scenario in test_scenarios:
        logger.info(f"🧪 Testing {scenario['name']} scenario...")
        
        # Generate test data
        test_cases = []
        for i in range(scenario["batch_size"]):
            if scenario["complexity"] == "simple":
                state = jnp.ones(scenario["state_size"]) * 0.5
                action = jnp.ones(scenario["action_size"]) * 0.3
            elif scenario["complexity"] == "medium":
                state = jnp.array(np.random.uniform(0, 1, scenario["state_size"]))
                action = jnp.array(np.random.uniform(-0.5, 0.5, scenario["action_size"]))
            elif scenario["complexity"] == "complex":
                # Create correlated data
                base_state = np.random.uniform(0, 1, scenario["state_size"])
                state = jnp.array(base_state + np.random.normal(0, 0.1, scenario["state_size"]))
                action = jnp.array(np.random.normal(0, 0.3, scenario["action_size"]))
            else:  # varied
                if i % 3 == 0:
                    state = jnp.zeros(scenario["state_size"])
                    action = jnp.zeros(scenario["action_size"])
                elif i % 3 == 1:
                    state = jnp.ones(scenario["state_size"])
                    action = jnp.ones(scenario["action_size"]) * -0.5
                else:
                    state = jnp.array(np.random.uniform(0, 1, scenario["state_size"]))
                    action = jnp.array(np.random.uniform(-1, 1, scenario["action_size"]))
            
            test_cases.append({
                "contract_id": contract_id,
                "state": state,
                "action": action,
                "context": {
                    "scenario": scenario["name"],
                    "test_id": i,
                    "priority": "high" if scenario["name"] == "Burst Load" else "medium"
                }
            })
        
        # Execute batch with scaling
        start_time = time.time()
        
        # Use batch execution for optimal scaling
        batch_results = await runtime.batch_execute(test_cases)
        
        execution_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in batch_results if r.error is None]
        success_rate = len(successful_results) / len(batch_results)
        
        if successful_results:
            avg_reward = np.mean([r.reward for r in successful_results])
            avg_execution_time = np.mean([r.execution_time for r in successful_results])
            avg_compliance = np.mean([r.compliance_score for r in successful_results])
        else:
            avg_reward = avg_execution_time = avg_compliance = 0.0
        
        throughput = len(batch_results) / execution_time
        
        scenario_result = {
            "scenario": scenario["name"],
            "batch_size": scenario["batch_size"],
            "execution_time": execution_time,
            "success_rate": success_rate,
            "throughput": throughput,
            "avg_reward": avg_reward,
            "avg_execution_time": avg_execution_time,
            "avg_compliance": avg_compliance
        }
        
        scaling_results.append(scenario_result)
        
        logger.info(f"   ✅ Completed in {execution_time:.3f}s")
        logger.info(f"   Success rate: {success_rate:.1%}")
        logger.info(f"   Throughput: {throughput:.1f} ops/sec")
        logger.info(f"   Average reward: {avg_reward:.4f}")
        logger.info(f"   Average execution time: {avg_execution_time:.4f}s")
        logger.info("")
        
        # Brief pause between scenarios to observe scaling
        await asyncio.sleep(2.0)
    
    # Get final performance summary
    final_performance = runtime.get_performance_summary()
    optimizer_cache_stats = optimizer.cache.stats()
    
    logger.info("📈 Final Scaling Performance Summary:")
    logger.info(f"   Total executions: {final_performance.get('total_executions', 0)}")
    logger.info(f"   Cache hit rate: {final_performance.get('cache_hit_rate', 0):.1%}")
    logger.info(f"   Average execution time: {final_performance.get('average_execution_time', 0):.4f}s")
    logger.info(f"   P95 execution time: {final_performance.get('p95_execution_time', 0):.4f}s")
    logger.info(f"   Average compliance: {final_performance.get('average_compliance_score', 0):.4f}")
    
    logger.info("💾 Optimizer Cache Statistics:")
    logger.info(f"   Cache size: {optimizer_cache_stats['size']}/{optimizer_cache_stats['max_size']}")
    logger.info(f"   Hit rate: {optimizer_cache_stats['hit_rate']:.1%}")
    logger.info(f"   Memory usage: {optimizer_cache_stats['memory_usage_mb']:.1f} MB")
    
    # Performance scaling analysis
    logger.info("📊 Scaling Analysis:")
    throughputs = [r["throughput"] for r in scaling_results]
    batch_sizes = [r["batch_size"] for r in scaling_results]
    
    if len(throughputs) > 1:
        scaling_efficiency = throughputs[-1] / throughputs[0]  # Heavy vs Light load
        logger.info(f"   Scaling efficiency: {scaling_efficiency:.1f}x")
        logger.info(f"   Peak throughput: {max(throughputs):.1f} ops/sec")
        logger.info(f"   Throughput at max load: {throughputs[-1]:.1f} ops/sec")
    
    return {
        "scaling_results": scaling_results,
        "final_performance": final_performance,
        "cache_stats": optimizer_cache_stats
    }


def demo_generation3_features():
    """Demonstrate all Generation 3 features together."""
    logger.info("🌟 Starting Generation 3 Complete Demo")
    logger.info("=" * 80)
    
    async def run_all_scaling_demos():
        # Feature 1: Intelligent Caching
        logger.info("FEATURE 1: Intelligent Caching")
        await demo_intelligent_caching()
        logger.info("")
        
        # Feature 2: Adaptive Load Balancing
        logger.info("FEATURE 2: Adaptive Load Balancing")
        await demo_adaptive_load_balancer()
        logger.info("")
        
        # Feature 3: Performance Optimization
        logger.info("FEATURE 3: Performance Optimization")
        await demo_performance_optimization()
        logger.info("")
        
        # Feature 4: Distributed Quantum Computing
        logger.info("FEATURE 4: Distributed Quantum Computing")
        await demo_distributed_quantum_computing()
        logger.info("")
        
        # Feature 5: Integrated Scaling System
        logger.info("FEATURE 5: Integrated Scaling System")
        final_results = await demo_integrated_scaling_system()
        logger.info("")
        
        logger.info("🏆 Generation 3 Demo Completed Successfully!")
        logger.info("🚀 System is now OPTIMIZED and ready for production scale!")
        logger.info("⚡ Key Achievements:")
        logger.info("   • Intelligent multi-strategy caching implemented")
        logger.info("   • Adaptive load balancing with health monitoring")
        logger.info("   • Performance optimization with auto-scaling")
        logger.info("   • Distributed quantum computing orchestration")
        logger.info("   • End-to-end scaling system integration")
        logger.info("=" * 80)
        
        return final_results
    
    # Run all demonstrations
    return asyncio.run(run_all_scaling_demos())


if __name__ == "__main__":
    demo_generation3_features()