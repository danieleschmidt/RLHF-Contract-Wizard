#!/usr/bin/env python3
"""
Generation 1 Demo: MAKE IT WORK (Simple)

Demonstrates the immediate functionality improvements for RLHF-Contract-Wizard.
This script showcases basic functionality with enhanced error handling,
performance optimization, and global compliance features.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
import jax.numpy as jnp
import numpy as np

from .models.reward_contract import RewardContract, AggregationStrategy
from .enhanced_contract_runtime import EnhancedContractRuntime, RuntimeConfig, execute_simple_contract
from .quantum_planner.core import QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_contract() -> RewardContract:
    """Create a sample reward contract for demonstration."""
    contract = RewardContract(
        name="SafeAssistant-Demo",
        version="1.0.0",
        stakeholders={
            "operator": 0.4,
            "safety_team": 0.3,
            "users": 0.3
        },
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE
    )
    
    # Add reward functions for each stakeholder
    @contract.reward_function("operator")
    def operator_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        # Efficiency-focused reward
        efficiency = jnp.mean(action) * 0.8
        return float(efficiency)
    
    @contract.reward_function("safety_team") 
    def safety_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        # Safety-focused reward with constraint checking
        safety_score = 1.0 - jnp.max(jnp.abs(action - 0.5)) * 2.0
        return float(jnp.clip(safety_score, 0.0, 1.0))
    
    @contract.reward_function("users")
    def user_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        # User satisfaction reward
        satisfaction = jnp.sum(state * action) / (jnp.sum(state) + 1e-8)
        return float(jnp.clip(satisfaction, 0.0, 1.0))
    
    # Add safety constraints
    contract.add_constraint(
        name="no_extreme_actions",
        constraint_fn=lambda state, action: bool(jnp.all(jnp.abs(action) <= 1.0)),
        description="Actions must be within [-1, 1] range",
        violation_penalty=-0.5
    )
    
    contract.add_constraint(
        name="state_action_compatibility",
        constraint_fn=lambda state, action: bool(len(state) >= len(action)),
        description="State dimension must be >= action dimension",
        violation_penalty=-0.2
    )
    
    return contract


def create_quantum_tasks() -> List[QuantumTask]:
    """Create sample quantum tasks for planning demo."""
    tasks = [
        QuantumTask(
            id="task_1",
            name="Initialize Contract",
            description="Set up reward contract with stakeholders",
            priority=0.9,
            estimated_duration=2.0,
            resource_requirements={"cpu": 0.2, "memory": 0.1}
        ),
        QuantumTask(
            id="task_2", 
            name="Validate Constraints",
            description="Check all contract constraints",
            priority=0.8,
            estimated_duration=1.5,
            resource_requirements={"cpu": 0.1, "memory": 0.05},
            dependencies={"task_1"}
        ),
        QuantumTask(
            id="task_3",
            name="Compute Rewards",
            description="Execute reward computation",
            priority=0.7,
            estimated_duration=3.0,
            resource_requirements={"cpu": 0.3, "memory": 0.15},
            dependencies={"task_1", "task_2"}
        ),
        QuantumTask(
            id="task_4",
            name="Global Compliance Check",
            description="Verify global regulatory compliance",
            priority=0.6,
            estimated_duration=2.5,
            resource_requirements={"cpu": 0.15, "memory": 0.1},
            dependencies={"task_2"}
        ),
        QuantumTask(
            id="task_5",
            name="Generate Report",
            description="Create execution summary report",
            priority=0.5,
            estimated_duration=1.0,
            resource_requirements={"cpu": 0.05, "memory": 0.05},
            dependencies={"task_3", "task_4"}
        )
    ]
    
    return tasks


async def demo_enhanced_runtime():
    """Demonstrate enhanced contract runtime capabilities."""
    logger.info("🚀 Starting Enhanced Runtime Demo")
    
    # Configure runtime
    config = RuntimeConfig(
        enable_caching=True,
        max_concurrent_contracts=5,
        timeout_seconds=10.0,
        enable_global_compliance=True,
        performance_monitoring=True
    )
    
    runtime = EnhancedContractRuntime(config)
    
    # Create and register contract
    contract = create_sample_contract()
    contract_id = runtime.register_contract(contract)
    
    logger.info(f"📋 Contract registered: {contract.metadata.name}")
    
    # Generate test data
    states = [
        jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        jnp.array([0.8, 0.7, 0.6, 0.5, 0.4]),
        jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ]
    
    actions = [
        jnp.array([0.2, 0.4, 0.6]),
        jnp.array([-0.1, 0.3, -0.2]),
        jnp.array([0.0, 0.0, 0.0])
    ]
    
    # Test single execution
    logger.info("🔍 Testing single contract execution...")
    result = await runtime.execute_contract(
        contract_id=contract_id,
        state=states[0],
        action=actions[0],
        context={"jurisdiction": "global", "priority": "high"}
    )
    
    logger.info(f"✅ Execution completed:")
    logger.info(f"   Reward: {result.reward:.4f}")
    logger.info(f"   Time: {result.execution_time:.4f}s")
    logger.info(f"   Compliance: {result.compliance_score:.4f}")
    logger.info(f"   Violations: {result.violations}")
    
    # Test batch execution
    logger.info("⚡ Testing batch execution...")
    batch_requests = [
        {"contract_id": contract_id, "state": state, "action": action}
        for state, action in zip(states, actions)
    ]
    
    batch_results = await runtime.batch_execute(batch_requests)
    
    logger.info(f"✅ Batch execution completed ({len(batch_results)} contracts)")
    for i, result in enumerate(batch_results):
        logger.info(f"   Contract {i+1}: Reward={result.reward:.4f}, Time={result.execution_time:.4f}s")
    
    # Performance summary
    performance = runtime.get_performance_summary()
    logger.info("📊 Performance Summary:")
    for key, value in performance.items():
        logger.info(f"   {key}: {value}")
    
    # Runtime optimization
    optimization = runtime.optimize_runtime()
    logger.info(f"🔧 Runtime optimizations applied: {optimization['optimizations_applied']}")
    
    # Health check
    health = await runtime.health_check()
    logger.info(f"❤️ Health status: {health['status']}")
    
    return runtime, batch_results


def demo_quantum_planning():
    """Demonstrate quantum-enhanced task planning."""
    logger.info("🌌 Starting Quantum Planning Demo")
    
    # Configure quantum planner
    config = PlannerConfig(
        max_iterations=100,
        quantum_interference_strength=0.2,
        enable_quantum_speedup=True,
        parallel_execution_limit=3
    )
    
    planner = QuantumTaskPlanner(config)
    
    # Add resources
    planner.add_resource("cpu", 1.0)
    planner.add_resource("memory", 0.5)
    
    # Add tasks
    tasks = create_quantum_tasks()
    for task in tasks:
        planner.add_task(task)
    
    logger.info(f"📝 Added {len(tasks)} quantum tasks")
    
    # Create entanglements between related tasks
    planner.create_entanglement("task_2", "task_4", strength=0.8)  # Both validation tasks
    planner.create_entanglement("task_1", "task_3", strength=0.9)  # Contract init -> compute
    
    # Optimize plan
    logger.info("🧮 Optimizing task execution plan...")
    start_time = time.time()
    
    plan = planner.optimize_plan()
    
    optimization_time = time.time() - start_time
    
    logger.info(f"✨ Plan optimization completed in {optimization_time:.3f}s")
    logger.info(f"   Optimal task order: {plan['task_order']}")
    logger.info(f"   Fitness score: {plan['fitness_score']:.4f}")
    logger.info(f"   Iterations: {plan['iterations']}")
    logger.info(f"   Converged: {plan['converged']}")
    
    # Quantum metrics
    quantum_metrics = plan['quantum_metrics']
    logger.info("🔬 Quantum Metrics:")
    logger.info(f"   Entanglements: {quantum_metrics['entanglements']}")
    logger.info(f"   Superposition tasks: {quantum_metrics['superposition_tasks']}")
    logger.info(f"   Average probability: {quantum_metrics['average_probability']:.4f}")
    
    # Execute plan
    logger.info("⚙️ Executing optimized plan...")
    execution_result = planner.execute_plan(plan)
    
    logger.info(f"✅ Plan execution completed:")
    logger.info(f"   Completed tasks: {len(execution_result['completed_tasks'])}")
    logger.info(f"   Failed tasks: {len(execution_result['failed_tasks'])}")
    logger.info(f"   Success rate: {execution_result['success_rate']:.2%}")
    logger.info(f"   Total time: {execution_result['total_execution_time']:.3f}s")
    
    # Quantum state summary
    quantum_state = planner.get_quantum_state_summary()
    logger.info("🌊 Final Quantum State:")
    for key, value in quantum_state.items():
        logger.info(f"   {key}: {value}")
    
    return planner, plan, execution_result


async def demo_simple_interface():
    """Demonstrate simple interface for quick contract execution."""
    logger.info("⚡ Starting Simple Interface Demo")
    
    # Create contract
    contract = create_sample_contract()
    
    # Test data
    state = jnp.array([0.3, 0.7, 0.2, 0.8, 0.1])
    action = jnp.array([0.4, -0.2, 0.6])
    
    # Execute using simple interface
    result = await execute_simple_contract(
        contract=contract,
        state=state,
        action=action,
        context={
            "jurisdiction": "EU",
            "gdpr_compliance": True,
            "priority": "medium"
        }
    )
    
    logger.info(f"🎯 Simple execution result:")
    logger.info(f"   Reward: {result.reward:.4f}")
    logger.info(f"   Execution time: {result.execution_time:.4f}s") 
    logger.info(f"   Compliance score: {result.compliance_score:.4f}")
    logger.info(f"   Metadata: {result.metadata}")
    
    return result


def demo_generation1_features():
    """Demonstrate all Generation 1 features together."""
    logger.info("🎉 Starting Generation 1 Complete Demo")
    logger.info("=" * 60)
    
    # Feature 1: Enhanced Contract Runtime
    logger.info("FEATURE 1: Enhanced Contract Runtime")
    asyncio.run(demo_enhanced_runtime())
    logger.info("")
    
    # Feature 2: Quantum Task Planning
    logger.info("FEATURE 2: Quantum Task Planning")
    demo_quantum_planning()
    logger.info("")
    
    # Feature 3: Simple Interface
    logger.info("FEATURE 3: Simple Interface")
    asyncio.run(demo_simple_interface())
    logger.info("")
    
    logger.info("🏆 Generation 1 Demo Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    demo_generation1_features()